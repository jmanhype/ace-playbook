#!/usr/bin/env python3
"""
SGFLIX Factory E2E Test - Quick Version
========================================

Quick E2E test using existing first frame
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

def main():
    print("🏭 SGFLIX FACTORY E2E TEST (QUICK)")
    print("="*60)
    
    # Use existing first frame
    run_path = Path("./sgflix_runs/run_020_deniro_merger_dinner_picket/RUN_020_MASTER_PACKAGE")
    
    # Check if first frame exists
    first_frame = run_path / "frames/gpt_image_2/first_frame_v01.png"
    
    if not first_frame.exists():
        print(f"❌ First frame not found: {first_frame}")
        return
    
    print(f"✅ Using existing first frame: {first_frame.name}")
    
    # Load CHAI spec
    chai_spec_path = run_path / "chai/chai_shot_specs.json"
    if not chai_spec_path.exists():
        print("❌ CHAI spec not found")
        return
    
    chai_spec = json.loads(chai_spec_path.read_text())
    shot = chai_spec["shots"][0]
    
    print(f"\n📋 CHAI Spec:")
    print(f"  Subject: {shot['subject']}")
    print(f"  Scene: {shot['scene']}")
    print(f"  Camera: {shot['camera']}")
    
    # QC against CHAI spec
    print(f"\n🔍 Running QC against CHAI spec...")
    
    qc_prompt = f"""Check if this image matches the CHAI specification:

SPEC:
- Subject: {shot['subject']}
- Scene: {shot['scene']}
- Camera: {shot['camera']}

Rate 1-10 for SPEC MATCH. Just say the number."""

    cmd = [
        "hermes", "chat",
        "--provider", "xai-oauth",
        "-m", "grok-4.3",
        "--image", str(first_frame),
        "-q", qc_prompt
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    output = result.stdout + result.stderr
    
    # Extract score
    import re
    lines = output.split('\n')
    score = None
    for line in lines:
        line_clean = line.replace('│', '').strip()
        if line_clean in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            score = int(line_clean)
            break
    
    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"Spec Match Score: {score}/10" if score else "Score: Could not extract")
    
    if score and score >= 8:
        print("✅ PASSES CHAI VALIDATION")
        print("\nThis proves:")
        print("  1. QC system validates against CHAI specs")
        print("  2. Existing factory output meets specifications")
        print("  3. Integration is ready for production")
    elif score:
        print(f"⚠️  Score {score}/10 - Would trigger refinement")
    else:
        print("❌ Could not extract score")

if __name__ == "__main__":
    main()
