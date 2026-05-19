#!/usr/bin/env python3
"""
SGFLIX QC Layer: Cross-Model Quality Control
============================================

Integrates Grok 4.3 + GPT-5.4 into existing SGFLIX pipeline:
- Analyzes generated images with Grok 4.3
- Provides quality scores (1-10)
- Suggests specific improvements
- Triggers refinement if score < threshold

Usage:
    # QC a single image
    python3 sgflix_qc_layer.py qc /path/to/image.png

    # Batch QC a directory
    python3 sgflix_qc_layer.py batch /path/to/frames/

    # Integrate into SGFLIX run
    python3 sgflix_qc_layer.py integrate <run_id>
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class SGFLIXQCLayer:
    def __init__(self, min_quality_score: int = 8):
        self.min_quality_score = min_quality_score
        self.results = []

    def log(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def analyze_with_grok(self, image_path: str) -> Dict:
        """Analyze image with Grok 4.3 for quality control"""
        self.log(f"Analyzing with Grok 4.3: {image_path}")

        qc_prompt = """Analyze this image for production quality:
1) Overall quality score (1-10, where 10 is perfect)
2) What works well? (strengths)
3) What needs improvement? (specific issues)
4) Color palette accuracy (if applicable)
5) Consistency check (any visual anomalies?)

Rate honestly - this is for production QC."""

        cmd = [
            "hermes", "chat",
            "--provider", "xai-oauth",
            "-m", "grok-4.3",
            "--image", image_path,
            "-q", qc_prompt
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr

        # Extract score from output
        import re
        score_match = re.search(r'(\d+)/10', output)
        score = int(score_match.group(1)) if score_match else None

        return {
            "image_path": image_path,
            "score": score,
            "analysis": output,
            "passes_qc": score is not None and score >= self.min_quality_score
        }

    def refine_with_gpt(self, image_path: str, feedback: str) -> Optional[str]:
        """Refine image based on Grok's feedback"""
        self.log(f"Refining based on QC feedback...")

        # Clean up feedback - extract only the improvement suggestions
        import re
        feedback_match = re.search(r'(?:improvement|issues?)(?:.*?:)(.*?)(?:\n\n|\Z)', feedback, re.DOTALL | re.IGNORECASE)

        if feedback_match:
            clean_feedback = feedback_match.group(1).strip()
        else:
            clean_feedback = feedback

        # Use the image as reference and request improvements
        refine_prompt = f"""Looking at this image, create an improved version that addresses these issues:

{clean_feedback}

Keep the same subject and composition, but fix the quality problems."""

        cmd = [
            "hermes", "chat",
            "--provider", "openai-codex",
            "-m", "gpt-5.4",
            "--image", image_path,
            "-q", refine_prompt
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr

        # Find the new image
        output_single_line = output.replace('\n', '').replace('\r', '')
        matches = re.findall(r'(/[^\s]+gpt-image-2[^\s]*\.png)', output_single_line)

        if matches:
            refined_path = matches[0]
            self.log(f"✅ Refined image: {refined_path}")
            return refined_path

        return None

    def qc_single_image(self, image_path: str, auto_refine: bool = True) -> Dict:
        """QC a single image with optional auto-refinement"""
        self.log(f"QC Processing: {image_path}")

        # Step 1: Grok analysis
        qc_result = self.analyze_with_grok(image_path)

        if qc_result["passes_qc"]:
            self.log(f"✅ PASSED QC (score: {qc_result['score']}/10)")
            return {
                "status": "passed",
                "original_image": image_path,
                "final_image": image_path,
                "qc_result": qc_result
            }

        # Step 2: Refine if failed QC
        if auto_refine and qc_result["score"] and qc_result["score"] < self.min_quality_score:
            self.log(f"⚠️  FAILED QC (score: {qc_result['score']}/10) - Refining...")

            # Extract just the improvement section from Grok's analysis
            import re
            improvement_match = re.search(
                r'(?:3\)|What needs improvement|improvement|issues?)(?:.*?:)(.*?)(?:(?:\n\s*\d+\)|\n\s*─|Color palette|Overall score))',
                qc_result["analysis"],
                re.DOTALL | re.IGNORECASE
            )

            if improvement_match:
                feedback = improvement_match.group(1).strip()[:500]  # Limit to 500 chars
            else:
                # Fallback: use analysis after the score
                feedback = qc_result["analysis"][:500]

            refined_path = self.refine_with_gpt(image_path, feedback)

            if refined_path:
                # QC the refined version
                refined_qc = self.analyze_with_grok(refined_path)
                self.log(f"Refined score: {refined_qc['score']}/10")

                return {
                    "status": "refined",
                    "original_image": image_path,
                    "final_image": refined_path,
                    "original_qc": qc_result,
                    "refined_qc": refined_qc
                }

        return {
            "status": "failed",
            "original_image": image_path,
            "final_image": image_path,
            "qc_result": qc_result
        }

    def batch_qc(self, directory: str, pattern: str = "*.png") -> List[Dict]:
        """Batch QC all images in a directory"""
        self.log(f"Batch QC: {directory}")

        dir_path = Path(directory)
        images = sorted(dir_path.glob(pattern))

        self.log(f"Found {len(images)} images to QC")

        results = []
        for i, image_path in enumerate(images, 1):
            self.log(f"Processing {i}/{len(images)}: {image_path.name}")
            result = self.qc_single_image(str(image_path))
            results.append(result)

        return results

    def integrate_sgflix_run(self, run_id: str):
        """Integrate QC into existing SGFLIX run"""
        self.log(f"Integrating QC into SGFLIX run: {run_id}")

        # Find run directory
        run_path = Path(f"./sgflix_runs/{run_id}")
        if not run_path.exists():
            self.log(f"❌ Run not found: {run_id}")
            return

        # Find all generated images
        image_dirs = [
            run_path / "frames",
            run_path / "RUN_003_MASTER_PACKAGE" / "factory_outputs"
        ]

        all_images = []
        for img_dir in image_dirs:
            if img_dir.exists():
                all_images.extend(img_dir.glob("**/*.png"))

        self.log(f"Found {len(all_images)} images in run")

        # Create QC output directory
        qc_output = run_path / "QC_RESULTS"
        qc_output.mkdir(exist_ok=True)

        # Process each image
        results = []
        for i, image_path in enumerate(all_images[:10], 1):  # Limit to 10 for testing
            self.log(f"QC {i}/{min(10, len(all_images))}: {image_path.name}")
            result = self.qc_single_image(str(image_path))
            results.append(result)

        # Save QC report
        report = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "total_images": len(all_images),
            "processed_images": len(results),
            "min_quality_score": self.min_quality_score,
            "results": results
        }

        report_file = qc_output / "qc_report.json"
        report_file.write_text(json.dumps(report, indent=2))

        self.log(f"✅ QC report saved: {report_file}")

        # Print summary
        passed = sum(1 for r in results if r["status"] == "passed")
        refined = sum(1 for r in results if r["status"] == "refined")
        failed = sum(1 for r in results if r["status"] == "failed")

        self.log(f"\n📊 QC Summary:")
        self.log(f"  Passed: {passed}")
        self.log(f"  Refined: {refined}")
        self.log(f"  Failed: {failed}")

        return report


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    qc = SGFLIXQCLayer(min_quality_score=8)

    if command == "qc":
        if not args:
            print("Usage: sgflix_qc_layer.py qc <image_path>")
            sys.exit(1)

        result = qc.qc_single_image(args[0])
        print(json.dumps(result, indent=2))

    elif command == "batch":
        if not args:
            print("Usage: sgflix_qc_layer.py batch <directory>")
            sys.exit(1)

        results = qc.batch_qc(args[0])
        print(json.dumps(results, indent=2))

    elif command == "integrate":
        if not args:
            print("Usage: sgflix_qc_layer.py integrate <run_id>")
            print("Example: sgflix_qc_layer.py integrate run_003_dana_white_gala")
            sys.exit(1)

        qc.integrate_sgflix_run(args[0])

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
