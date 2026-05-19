#!/usr/bin/env python3
"""
Cross-Model Image Iteration Tool
==================================

Leverages Grok 4.3 (xAI) and GPT-5.4 (OpenAI Codex) for collaborative image generation:

1. GPT-5.4 generates image from prompt
2. Grok 4.3 analyzes and critiques
3. GPT-5.4 refines based on feedback
4. Repeat until满意

Usage:
    python cross_model_image_tool.py "A cyberpunk city at sunset" --iterations 3
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict


class CrossModelImageGenerator:
    """Orchestrate image generation across Grok 4.3 and GPT-5.4"""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or f"./cross_model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_history = []

    def generate_with_gpt(self, prompt: str, iteration: int) -> Dict:
        """Generate image using GPT-5.4"""
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: GPT-5.4 Generating Image")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}\n")

        cmd = [
            "hermes", "chat",
            "--provider", "openai-codex",
            "-m", "gpt-5.4",
            "-q", prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        # Extract image path from output
        output = result.stdout + result.stderr
        image_path = None

        # Look for image path in output
        import re
        for line in output.split('\n'):
            if 'Image saved at:' in line or 'openai_codex' in line:
                # Try multiple patterns
                matches = re.findall(r'/[/\w\-_\.\/]+\.png', line)
                if matches:
                    image_path = matches[0]
                    break

        # If not found in output, find most recent image
        if not image_path or not Path(image_path).exists():
            print(f"🔍 Searching for recently generated image...")
            # Check multiple possible locations
            search_paths = [
                Path("~/.hermes/cache/images").expanduser(),
                Path("~/.hermes").expanduser(),
                Path("~/Library/Caches/hermes").expanduser(),
            ]

            for search_path in search_paths:
                if search_path.exists():
                    images = sorted(search_path.glob("**/*gpt*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if images:
                        image_path = str(images[0])
                        print(f"✅ Found: {image_path}")
                        break
                    else:
                        images = sorted(search_path.glob("**/*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if images:
                            image_path = str(images[0])
                            print(f"✅ Found: {image_path}")
                            break

        if image_path and Path(image_path).exists():
            print(f"✅ Image generated: {image_path}")
            return {
                "success": True,
                "image_path": image_path,
                "prompt": prompt
            }

        # Debug: show output if image not found
        print(f"⚠️  Could not find generated image")
        print(f"Output preview: {output[:500]}")

        return {"success": False, "error": "Could not find generated image"}

    def analyze_with_grok(self, image_path: str, iteration: int) -> Dict:
        """Analyze image using Grok 4.3"""
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Grok 4.3 Analyzing")
        print(f"{'='*60}")

        analysis_prompt = """Analyze this image and provide:
1. Visual description (subject, style, mood)
2. Color palette analysis
3. Strengths (what works well)
4. Specific improvement suggestions (be detailed)
5. Refined prompt for next iteration

Be specific and actionable."""

        cmd = [
            "hermes", "chat",
            "--provider", "xai-oauth",
            "-m", "grok-4.3",
            "--image", image_path,
            "-q", analysis_prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        output = result.stdout + result.stderr

        # Extract analysis from Hermes output
        analysis = self._extract_hermes_response(output)

        print(f"\n✅ Grok Analysis Complete")
        print(f"{'─'*60}")
        print(analysis[:500])  # Show preview
        if len(analysis) > 500:
            print(f"\n... ({len(analysis) - 500} more characters)")

        return {
            "success": True,
            "analysis": analysis,
            "image_path": image_path
        }

    def refine_with_gpt(self, original_prompt: str, grok_analysis: str, iteration: int) -> Dict:
        """Refine image based on Grok's analysis"""
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: GPT-5.4 Refining")
        print(f"{'='*60}")

        refinement_prompt = f"""Original prompt: {original_prompt}

Grok 4.3's analysis and suggestions:
{grok_analysis}

Create an improved version incorporating Grok's suggestions. Focus on the specific improvements mentioned."""

        cmd = [
            "hermes", "chat",
            "--provider", "openai-codex",
            "-m", "gpt-5.4",
            "-q", refinement_prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        output = result.stdout + result.stderr

        # Extract refined image path
        import re
        image_path = None
        for line in output.split('\n'):
            if 'openai_codex_gpt-image-2' in line:
                match = re.search(r'/[/\w\-_\.]+\.png', line)
                if match:
                    image_path = match.group(0)
                    break

        if not image_path:
            # Find most recent
            images = sorted(Path("~/.hermes/cache/images").expanduser().glob("openai_codex_gpt-image-2*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if images:
                image_path = str(images[0])

        print(f"✅ Refined image: {image_path}")

        return {
            "success": True,
            "image_path": image_path,
            "refinement": True
        }

    def _extract_hermes_response(self, output: str) -> str:
        """Extract the actual response from Hermes output"""
        lines = output.split('\n')
        in_response = False
        response_lines = []

        for line in lines:
            if '╭─ ⚕ Hermes' in line:
                in_response = True
                continue
            if '╰─' in line and in_response:
                break
            if in_response and line.strip() and not line.startswith('│'):
                response_lines.append(line.strip())

        return '\n'.join(response_lines) if response_lines else output

    def run_iteration_pipeline(self, prompt: str, iterations: int = 3):
        """Run the full cross-model iteration pipeline"""
        print(f"\n{'🎨'*30}")
        print(f"Cross-Model Image Generation Pipeline")
        print(f"{'🎨'*30}")
        print(f"Prompt: {prompt}")
        print(f"Iterations: {iterations}")
        print(f"Output: {self.output_dir}")
        print(f"{'🎨'*30}")

        current_prompt = prompt
        current_image = None

        for i in range(1, iterations + 1):
            # Step 1: Generate or refine
            if i == 1:
                result = self.generate_with_gpt(current_prompt, i)
            else:
                result = self.refine_with_gpt(current_prompt, self.iteration_history[-1]['analysis'], i)

            if not result['success']:
                print(f"❌ Generation failed at iteration {i}")
                break

            current_image = result['image_path']

            # Step 2: Grok analyzes (skip if last iteration)
            if i < iterations:
                analysis_result = self.analyze_with_grok(current_image, i)

                if analysis_result['success']:
                    self.iteration_history.append({
                        "iteration": i,
                        "image_path": current_image,
                        "analysis": analysis_result['analysis']
                    })

                    # Copy image to output dir
                    import shutil
                    dest = self.output_dir / f"iteration_{i}.png"
                    shutil.copy2(current_image, dest)
                    print(f"💾 Saved to: {dest}")

        # Save iteration history
        history_file = self.output_dir / "pipeline_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.iteration_history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ Pipeline Complete!")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Images generated: {len(self.iteration_history)}")
        print(f"History saved: {history_file}")

        return self.iteration_history


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Model Image Generation using Grok 4.3 + GPT-5.4"
    )
    parser.add_argument("prompt", help="Image generation prompt")
    parser.add_argument("-i", "--iterations", type=int, default=3,
                        help="Number of iterations (default: 3)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: auto-generated)")

    args = parser.parse_args()

    generator = CrossModelImageGenerator(output_dir=args.output)
    generator.run_iteration_pipeline(args.prompt, args.iterations)


if __name__ == "__main__":
    main()
