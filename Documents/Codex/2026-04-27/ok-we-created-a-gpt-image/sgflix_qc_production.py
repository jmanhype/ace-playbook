#!/usr/bin/env python3
"""
SGFLIX QC Production Integration
=================================

Production-ready QC integration for SGFLIX factory:
- Character bible quality control
- Storyboard frame quality control
- Full run quality reports
- Automated refinement loop

Usage:
    # QC all character bibles
    python3 sgflix_qc_production.py characters

    # QC specific SGFLIX run
    python3 sgflix_qc_production.py run run_003_dana_white_gala

    # QC all runs (batch)
    python3 sgflix_qc_production.py all-runs
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

class SGFLIXProductionQC:
    def __init__(self, min_score: int = 8, max_iterations: int = 2):
        self.min_score = min_score
        self.max_iterations = max_iterations
        self.results = []

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def qc_image(self, image_path: str) -> Dict:
        """Single QC check with Grok 4.3"""
        qc_prompt = "Rate this image 1-10 for production quality. Just say the number."

        cmd = [
            "hermes", "chat",
            "--provider", "xai-oauth",
            "-m", "grok-4.3",
            "--image", image_path,
            "-q", qc_prompt
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        output = result.stdout + result.stderr

        # Extract score - look for the score in Hermes output box
        import re
        # The score appears alone in the Hermes response box
        # Pattern: lines between the box markers with just a number
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if '─' in line and i > 5:  # After the box starts
                # Check next few lines for just a number
                for j in range(i+1, min(i+5, len(lines))):
                    test_line = lines[j].strip()
                    # Remove box markers
                    test_line = test_line.replace('│', '').strip()
                    # Check if it's just a number 1-10
                    if test_line in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                        score = int(test_line)
                        break
                if score:
                    break

        if not score:
            # Fallback: try to find any number 1-10
            score_match = re.search(r'\b(10|[1-9])\b', output)
            score = int(score_match.group(1)) if score_match else None

        return {
            "image_path": image_path,
            "score": score,
            "passes": score is not None and score >= self.min_score
        }

    def refine_image(self, image_path: str, feedback: str) -> Optional[str]:
        """Refine image with GPT-5.4"""
        self.log(f"Refining: {Path(image_path).name}")

        refine_prompt = f"Looking at the image, create an improved version that addresses these issues: {feedback}. Keep same subject and composition."

        cmd = [
            "hermes", "chat",
            "--provider", "openai-codex",
            "-m", "gpt-5.4",
            "--image", image_path,
            "-q", refine_prompt
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr

        # Find new image
        import re
        output_single_line = output.replace('\n', '').replace('\r', '')
        matches = re.findall(r'(/[^\s]+gpt-image-2[^\s]*\.png)', output_single_line)

        if matches:
            return matches[0]
        return None

    def qc_with_refinement(self, image_path: str) -> Dict:
        """QC with automatic refinement loop"""
        self.log(f"QC Processing: {Path(image_path).name}")

        current_image = image_path
        iteration = 0
        qc_history = []

        while iteration <= self.max_iterations:
            # QC check
            qc_result = self.qc_image(current_image)
            qc_result["iteration"] = iteration
            qc_history.append(qc_result)

            if qc_result["passes"]:
                self.log(f"✅ PASS (score: {qc_result['score']}/10)")
                return {
                    "original_image": image_path,
                    "final_image": current_image,
                    "status": "passed",
                    "final_score": qc_result["score"],
                    "iterations": iteration + 1,
                    "qc_history": qc_history
                }

            # Need refinement
            if iteration < self.max_iterations:
                self.log(f"⚠️  Score {qc_result['score']}/10 - Refining...")

                # Get feedback
                feedback_cmd = [
                    "hermes", "chat",
                    "--provider", "xai-oauth",
                    "-m", "grok-4.3",
                    "--image", current_image,
                    "-q", "What are the main issues? List them in one sentence."
                ]
                feedback_result = subprocess.run(feedback_cmd, capture_output=True, text=True, timeout=90)
                feedback = feedback_result.stdout + feedback_result.stderr

                # Refine
                refined = self.refine_image(current_image, feedback[:200])  # Limit feedback length
                if refined:
                    current_image = refined
                else:
                    self.log(f"❌ Refinement failed")
                    break

            iteration += 1

        # Max iterations reached or refinement failed
        final_qc = self.qc_image(current_image)
        self.log(f"Final score: {final_qc['score']}/10")

        return {
            "original_image": image_path,
            "final_image": current_image,
            "status": "failed" if not final_qc["passes"] else "refined",
            "final_score": final_qc["score"],
            "iterations": iteration,
            "qc_history": qc_history
        }

    def process_character_bible(self, bible_path: str) -> Dict:
        """QC entire character bible"""
        self.log(f"\n{'='*60}")
        self.log(f"PROCESSING CHARACTER BIBLE: {bible_path}")
        self.log(f"{'='*60}")

        bible_dir = Path(bible_path)
        characters = [d for d in bible_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        results = {
            "bible_path": bible_path,
            "timestamp": datetime.now().isoformat(),
            "total_characters": len(characters),
            "characters": []
        }

        for char_dir in sorted(characters):
            self.log(f"\n🎭 Character: {char_dir.name}")

            # Find images
            images = list(char_dir.glob("**/*.png"))
            if not images:
                self.log(f"  No images found")
                continue

            # QC first image (usually the primary reference)
            primary_image = images[0]
            result = self.qc_with_refinement(str(primary_image))
            result["character_name"] = char_dir.name
            results["characters"].append(result)

        # Save report
        report_path = bible_dir / f"QC_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(json.dumps(results, indent=2))
        self.log(f"\n✅ Report saved: {report_path}")

        return results

    def process_sgflix_run(self, run_id: str) -> Dict:
        """QC entire SGFLIX run"""
        self.log(f"\n{'='*60}")
        self.log(f"PROCESSING SGFLIX RUN: {run_id}")
        self.log(f"{'='*60}")

        run_path = Path(f"./sgflix_runs/{run_id}")
        if not run_path.exists():
            self.log(f"❌ Run not found: {run_id}")
            return {}

        # Find all images
        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_paths.extend(run_path.glob(f"**/{ext}"))

        # Limit to first 20 for testing
        images_to_qc = image_paths[:20] if len(image_paths) > 20 else image_paths

        self.log(f"Found {len(image_paths)} total images")
        self.log(f"QC processing {len(images_to_qc)} images")

        results = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "total_images_found": len(image_paths),
            "images_processed": len(images_to_qc),
            "min_quality_score": self.min_score,
            "max_refinement_iterations": self.max_iterations,
            "results": []
        }

        for i, image_path in enumerate(images_to_qc, 1):
            self.log(f"\n[{i}/{len(images_to_qc)}] {image_path.name}")

            try:
                result = self.qc_with_refinement(str(image_path))
                result["relative_path"] = str(image_path.relative_to(run_path))
                results["results"].append(result)
            except Exception as e:
                self.log(f"❌ Error: {e}")
                results["results"].append({
                    "image_path": str(image_path),
                    "status": "error",
                    "error": str(e)
                })

        # Save report
        qc_dir = run_path / "QC_RESULTS"
        qc_dir.mkdir(exist_ok=True)
        report_path = qc_dir / f"qc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(json.dumps(results, indent=2))
        self.log(f"\n✅ Report saved: {report_path}")

        # Print summary
        passed = sum(1 for r in results["results"] if r.get("status") == "passed")
        refined = sum(1 for r in results["results"] if r.get("status") == "refined")
        failed = sum(1 for r in results["results"] if r.get("status") == "failed")

        self.log(f"\n📊 SUMMARY:")
        self.log(f"  Passed: {passed}")
        self.log(f"  Refined: {refined}")
        self.log(f"  Failed: {failed}")
        self.log(f"  Success rate: {(passed + refined) / len(results['results']) * 100:.1f}%")

        return results

    def process_all_runs(self):
        """QC all SGFLIX runs"""
        runs_dir = Path("./sgflix_runs")
        runs = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

        self.log(f"Found {len(runs)} runs to process")

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(runs),
            "runs": {}
        }

        for run_id in sorted(runs):
            self.log(f"\n\n{'#'*60}")
            self.log(f"# RUN: {run_id}")
            self.log(f"{'#'*60}")

            try:
                result = self.process_sgflix_run(run_id)
                all_results["runs"][run_id] = result
            except Exception as e:
                self.log(f"❌ Error processing {run_id}: {e}")
                all_results["runs"][run_id] = {"error": str(e)}

        # Master report
        master_report = runs_dir / f"MASTER_QC_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        master_report.write_text(json.dumps(all_results, indent=2))
        self.log(f"\n✅ Master report saved: {master_report}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands:")
        print("  characters     - QC all character bibles")
        print("  run <run_id>   - QC specific SGFLIX run")
        print("  all-runs       - QC all SGFLIX runs")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    qc = SGFLIXProductionQC(min_score=8, max_iterations=2)

    if command == "characters":
        # Process character bibles
        bible_path = args[0] if args else "./character_bibles/yu_yu_hakusho_canon_gpt_image_2"
        qc.process_character_bible(bible_path)

    elif command == "run":
        if not args:
            print("Usage: sgflix_qc_production.py run <run_id>")
            print("Example: sgflix_qc_production.py run run_003_dana_white_gala")
            sys.exit(1)

        qc.process_sgflix_run(args[0])

    elif command == "all-runs":
        qc.process_all_runs()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
