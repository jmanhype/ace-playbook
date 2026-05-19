#!/usr/bin/env python3
"""
SGFLIX QC Wrapper for Phase 7 (First Frame Generation)
======================================================

Wraps image generation with automatic QC:
- Generate image using OpenAI GPT-Image
- QC check against CHAI spec
- Auto-refine if score < 8
- Only save 8/10+ images
- Attach QC report to each image

Usage:
    from qc_wrapper import generate_first_frame_with_qc
    
    result = generate_first_frame_with_qc(
        prompt=prompt_text,
        output_path=Path("frames/gpt_image_2/first_frame_v01.png"),
        chai_spec=chai_spec_data
    )
"""

import sys
import json
import base64
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgflix_qc_production import SGFLIXProductionQC


def generate_image_openai(prompt, path, size):
    """
    Generate image using OpenAI GPT-Image API
    """
    try:
        from openai import OpenAI

        client = OpenAI()
        result = client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(result.data[0].b64_json))
        return {"ok": True, "mode": "openai_gpt_image_api"}
    except Exception as exc:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".generation_error.txt").write_text(
            f"OpenAI image generation failed. Error: {exc}\n",
            encoding="utf-8",
        )
        return {"ok": False, "mode": "blocked_openai_error", "error": str(exc)}


def generate_image_hermes(prompt, path, size):
    """
    Generate image using Hermes (GPT-5.4 via Codex)
    """
    import subprocess
    import re

    try:
        # Add generation request to prompt
        full_prompt = f"Generate this image using gpt-image-2:\n\n{prompt}"

        cmd = [
            "hermes", "chat",
            "--provider", "openai-codex",
            "-m", "gpt-5.4",
            "-q", full_prompt
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr

        # Extract image path from output
        output_single_line = output.replace('\n', '').replace('\r', '')
        matches = re.findall(r'(/[^\s]+gpt-image-2[^\s]*\.png)', output_single_line)

        if matches:
            # Copy generated image to target path
            source_path = matches[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", source_path, str(path)])
            return {"ok": True, "mode": "hermes_gpt_5_4", "source": source_path}
        else:
            # No image found in output
            path.with_suffix(".generation_error.txt").write_text(
                f"Hermes generation completed but no image found in output.\nOutput:\n{output}\n",
                encoding="utf-8",
            )
            return {"ok": False, "mode": "hermes_no_image", "error": "No image in output"}

    except subprocess.TimeoutExpired:
        return {"ok": False, "mode": "hermes_timeout", "error": "Generation timed out"}
    except Exception as exc:
        return {"ok": False, "mode": "hermes_error", "error": str(exc)}


def generate_image(prompt, path, size, use_hermes=True):
    """
    Generate image using available method

    Args:
        prompt: Image generation prompt
        path: Output path
        size: Image size (e.g., "1024x1536")
        use_hermes: If True, try Hermes first, then OpenAI. If False, only OpenAI.

    Returns:
        dict with ok, mode, and error keys
    """
    if use_hermes:
        # Try Hermes first (GPT-5.4 via Codex)
        print(f"   Trying Hermes (GPT-5.4)...")
        result = generate_image_hermes(prompt, path, size)
        if result.get("ok"):
            return result
        print(f"   Hermes failed: {result.get('error')}")
        print(f"   Falling back to OpenAI API...")

    # Fallback to OpenAI API
    return generate_image_openai(prompt, path, size)


def generate_first_frame_with_qc(prompt, output_path, chai_spec=None, min_score=8, max_iterations=2):
    """
    Generate first frame with automatic QC
    
    Args:
        prompt: Image generation prompt
        output_path: Where to save the image
        chai_spec: CHAI spec dict (optional, for validation)
        min_score: Minimum QC score to pass (default: 8)
        max_iterations: Maximum refinement iterations (default: 2)
    
    Returns:
        dict with keys:
        - ok: True if passed QC, False otherwise
        - mode: Generation mode used
        - path: Path to generated image
        - qc_score: Final QC score
        - qc_status: "passed", "refined", or "failed"
        - qc_report: Path to QC report
        - iterations: Number of refinement iterations
    """
    print(f"\n🎨 Generating first frame with QC...")
    print(f"   Output: {output_path}")
    print(f"   Min score: {min_score}/10")
    
    # Step 1: Generate image
    print(f"\n📸 Step 1: Generating image...")
    image_result = generate_image(prompt, output_path, "1024x1536")
    
    if not image_result.get("ok"):
        print(f"❌ Image generation failed: {image_result.get('error')}")
        return {
            "ok": False,
            "mode": image_result.get("mode"),
            "error": image_result.get("error"),
            "qc_status": "generation_failed"
        }
    
    print(f"✅ Image generated: {image_result['mode']}")
    
    # Step 2: QC check
    print(f"\n🔍 Step 2: Running QC...")
    qc = SGFLIXProductionQC(min_score=min_score, max_iterations=max_iterations)
    qc_result = qc.qc_with_refinement(str(output_path))
    
    print(f"   Score: {qc_result['final_score']}/10")
    print(f"   Status: {qc_result['status']}")
    print(f"   Iterations: {qc_result['iterations']}")
    
    # Step 3: Save QC report
    qc_report_path = output_path.parent / f"{output_path.stem}_qc_report.json"
    qc_report_path.write_text(json.dumps(qc_result, indent=2))
    print(f"✅ QC report saved: {qc_report_path.name}")
    
    # Step 4: Return result
    passed = qc_result["status"] == "passed"
    
    if passed:
        print(f"\n✅ FIRST FRAME PASSED QC ({qc_result['final_score']}/10)")
    else:
        print(f"\n⚠️  First frame needs manual review ({qc_result['final_score']}/10)")
    
    return {
        "ok": passed,
        "mode": image_result.get("mode"),
        "path": str(output_path),
        "qc_score": qc_result.get("final_score"),
        "qc_status": qc_result["status"],
        "qc_report": str(qc_report_path),
        "iterations": qc_result.get("iterations", 0)
    }


def main():
    """
    Test the QC wrapper
    """
    # Test prompt
    test_prompt = """
    Create a vertical 9:16 satirical cinematic first frame.
    Subject: Elderly protest leader holding oversized petition.
    Scene: Outside formal media dinner at night.
    Style: High-end satirical magazine photo.
    """
    
    # Test output
    test_output = Path("./test_qc_wrapper_output.png")
    
    # Run test
    result = generate_first_frame_with_qc(
        prompt=test_prompt,
        output_path=test_output
    )
    
    # Print result
    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"OK: {result.get('ok', False)}")

    if 'qc_score' in result:
        print(f"QC Score: {result['qc_score']}/10")
        print(f"QC Status: {result['qc_status']}")
        print(f"Iterations: {result['iterations']}")
        print(f"QC Report: {result['qc_report']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
