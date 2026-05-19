"""
SGFLIX Run Script with QC Integration
========================================

Modified version of create_run_024_rock_tint_meter.py
with automatic QC integration for Phase 7 (first frame generation)

Changes:
- Import QC wrapper
- Replace generate_image() with generate_first_frame_with_qc()
- Handle QC failures gracefully
- Log QC results
"""

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Import QC wrapper
import sys
sys.path.insert(0, str(Path(__file__).parent))
from qc_wrapper import generate_first_frame_with_qc


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN = "024"
SLUG = "rock_tint_meter"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

# ... rest of the original script ...
# (SOURCES, CANDIDATES, WINNER selection code omitted for brevity)

def write_text(path, content):
    """Write text to file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")

def write_json(path, data):
    """Write JSON to file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

# ... PROMPT definitions omitted for brevity ...

def main_with_qc():
    """
    Main function with QC integration
    """
    print(f"🎬 SGFLIX Run {RUN} - WITH QC")
    print("="*60)
    
    # Create package directory
    PKG.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate first frame WITH QC
    print(f"\n📸 Step 1: Generate first frame with QC...")
    
    first_frame_result = generate_first_frame_with_qc(
        prompt=FIRST_FRAME_PROMPT,  # Would need to be defined
        output_path=PKG / "frames/gpt_image_2/first_frame_v01.png",
        chai_spec=None,  # Would load CHAI spec if available
        min_score=8,
        max_iterations=2
    )
    
    print(f"   Result: {'✅ PASSED' if first_frame_result['ok'] else '⚠️  FAILED'}")
    print(f"   QC Score: {first_frame_result.get('qc_score', 'N/A')}/10")
    print(f"   QC Status: {first_frame_result.get('qc_status', 'N/A')}")
    print(f"   QC Report: {first_frame_result.get('qc_report', 'N/A')}")
    
    # Step 2: Generate shared choices WITH QC
    print(f"\n📸 Step 2: Generate shared choices with QC...")
    
    shared_choices_result = generate_first_frame_with_qc(
        prompt=SHARED_CHOICES_PROMPT,  # Would need to be defined
        output_path=PKG / "storyboards/shared_choices/shared_choices_v01.png",
        chai_spec=None,
        min_score=8,
        max_iterations=2
    )
    
    print(f"   Result: {'✅ PASSED' if shared_choices_result['ok'] else '⚠️  FAILED'}")
    print(f"   QC Score: {shared_choices_result.get('qc_score', 'N/A')}/10")
    
    # Step 3: Overall verdict
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT")
    print(f"{'='*60}")
    
    if first_frame_result['ok'] and shared_choices_result['ok']:
        print(f"✅ Both images passed QC - Ready for Phase 8")
        verdict = "passed_qc"
    elif first_frame_result['ok'] or shared_choices_result['ok']:
        print(f"⚠️  Partial pass - Review before Phase 8")
        verdict = "partial_qc"
    else:
        print(f"❌ Both images failed QC - Manual review required")
        verdict = "failed_qc"
    
    # Save QC summary
    qc_summary = {
        "run_id": RUN,
        "slug": SLUG,
        "timestamp": NOW,
        "qc_enabled": True,
        "min_score": 8,
        "first_frame_qc": first_frame_result,
        "shared_choices_qc": shared_choices_result,
        "overall_verdict": verdict
    }
    
    write_json(PKG / "qc_summary.json", qc_summary)
    print(f"\n✅ QC summary saved: qc_summary.json")
    
    return qc_summary


if __name__ == "__main__":
    # This would be the full script with QC integration
    print("Note: This is a demonstration of QC integration.")
    print("The full script would include all the original code plus QC wrapper calls.")
    print("\nKey changes:")
    print("1. Import: from qc_wrapper import generate_first_frame_with_qc")
    print("2. Replace: generate_image() → generate_first_frame_with_qc()")
    print("3. Handle: QC results and failures")
    print("4. Save: QC summary report")
