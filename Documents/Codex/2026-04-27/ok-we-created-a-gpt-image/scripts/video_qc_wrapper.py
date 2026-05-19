#!/usr/bin/env python3
"""
SGFLIX QC Wrapper for Phase 11 (Video Rendering)
================================================

Wraps video rendering with automatic QC:
- Extract frames from rendered video
- QC each frame against CHAI spec
- Check motion consistency
- Only approve videos with avg ≥ 8/10
- Attach QC report to each video

Usage:
    from video_qc_wrapper import render_video_with_qc
    
    result = render_video_with_qc(
        video_path=Path("renders/output.mp4"),
        chai_spec=chai_spec_data,
        min_score=8
    )
"""

import sys
import json
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_frames(video_path, output_dir, num_frames=5):
    """
    Extract key frames from video for QC analysis
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (default: 5)
    
    Returns:
        List of Paths to extracted frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video duration
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    
    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    
    # Calculate frame intervals
    interval = duration / (num_frames + 1)
    
    frames = []
    for i in range(num_frames):
        timestamp = interval * (i + 1)
        output_path = output_dir / f"frame_{i+1:03d}_t{timestamp:.1f}s.png"
        
        ffmpeg_cmd = [
            "ffmpeg", "-i", str(video_path),
            "-ss", str(timestamp),
            "-vframes", "1",
            "-q:v", "2",
            "-y", str(output_path)
        ]
        
        subprocess.run(ffmpeg_cmd, capture_output=True)
        frames.append(output_path)
    
    return frames


def qc_motion_consistency(frames):
    """
    Check motion consistency across frames
    
    Args:
        frames: List of frame paths
    
    Returns:
        Motion consistency score (1-10)
    """
    if len(frames) < 2:
        return {"score": None, "note": "Need at least 2 frames"}
    
    # Compare adjacent frames
    comparison_prompt = f"""Compare these two frames from a video sequence:

Frame 1: {frames[0].name}
Frame 2: {frames[1].name}

Questions:
1. Is there smooth motion between frames?
2. Are there any jumps or discontinuities?
3. Is the camera movement consistent?

Rate 1-10 for MOTION SMOOTHNESS. Just say the number."""

    cmd = [
        "hermes", "chat",
        "--provider", "xai-oauth",
        "-m", "grok-4.3",
        "--image", str(frames[0]),
        "--image", str(frames[1]),
        "-q", comparison_prompt
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
    
    return {"score": score, "note": "Motion smoothness validated"}


def render_video_with_qc(video_path, chai_spec=None, min_score=8, num_frames=5):
    """
    Render video with automatic QC
    
    Args:
        video_path: Path to rendered video
        chai_spec: CHAI spec dict (optional, for validation)
        min_score: Minimum average QC score to pass (default: 8)
        num_frames: Number of frames to QC (default: 5)
    
    Returns:
        dict with keys:
        - ok: True if passed QC, False otherwise
        - video_path: Path to video
        - avg_score: Average frame score
        - motion_score: Motion consistency score
        - passes: True if avg ≥ min_score AND motion ≥ 7
        - qc_report: Path to QC report
        - frame_scores: List of individual frame scores
    """
    print(f"\n🎬 Running video QC...")
    print(f"   Video: {video_path.name}")
    print(f"   Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Min score: {min_score}/10")
    
    # Step 1: Extract frames
    print(f"\n📸 Step 1: Extracting frames...")
    frames_dir = video_path.parent / "qc_frames"
    frames = extract_frames(video_path, frames_dir, num_frames=num_frames)
    
    print(f"✅ Extracted {len(frames)} frames:")
    for frame in frames:
        size_kb = frame.stat().st_size / 1024
        print(f"   {frame.name}: {size_kb:.1f} KB")
    
    # Step 2: QC each frame
    print(f"\n🔍 Step 2: QCing frames...")
    from sgflix_qc_production import SGFLIXProductionQC
    
    qc = SGFLIXProductionQC(min_score=min_score, max_iterations=0)  # No refinement for videos yet
    frame_scores = []
    
    for i, frame in enumerate(frames, 1):
        print(f"[{i}/{len(frames)}] {frame.name}...", end=" ")
        result = qc.qc_image(str(frame))
        score = result.get("score")
        
        if score:
            print(f"{score}/10 {'✅' if score >= 8 else '⚠️'}")
            frame_scores.append(score)
        else:
            print("N/A ⚠️")
    
    # Step 3: Check motion consistency
    print(f"\n🎬 Step 3: Checking motion consistency...")
    motion_result = qc_motion_consistency(frames[:2])  # Check first 2 frames
    motion_score = motion_result["score"]
    
    if motion_score:
        print(f"   Motion smoothness: {motion_score}/10 {'✅' if motion_score >= 7 else '⚠️'}")
    else:
        print(f"   Motion smoothness: Could not determine")
    
    # Step 4: Calculate average
    if frame_scores:
        avg_score = sum(frame_scores) / len(frame_scores)
        print(f"\n📊 Frame QC Average: {avg_score:.1f}/10")
    else:
        avg_score = None
        print(f"\n⚠️  Could not calculate average score")
    
    # Step 5: Determine pass/fail
    passes = False
    if avg_score and motion_score:
        passes = avg_score >= min_score and motion_score >= 7
        print(f"\n{'='*60}")
        if passes:
            print(f"✅ VIDEO PASSES QC")
            print(f"   Frame avg: {avg_score:.1f}/10 ≥ {min_score}")
            print(f"   Motion: {motion_score}/10 ≥ 7")
        else:
            print(f"⚠️  VIDEO NEEDS REVIEW")
            if avg_score < min_score:
                print(f"   Frame avg: {avg_score:.1f}/10 < {min_score}")
            if motion_score < 7:
                print(f"   Motion: {motion_score}/10 < 7")
        print(f"{'='*60}")
    
    # Step 6: Save QC report
    qc_report = {
        "video_path": str(video_path),
        "num_frames_qc": len(frames),
        "frame_scores": frame_scores,
        "avg_score": avg_score,
        "motion_score": motion_score,
        "min_score_required": min_score,
        "passes": passes,
        "chai_spec": chai_spec
    }
    
    qc_report_path = video_path.parent / f"{video_path.stem}_qc_report.json"
    qc_report_path.write_text(json.dumps(qc_report, indent=2))
    print(f"\n✅ QC report saved: {qc_report_path.name}")
    
    return {
        "ok": passes,
        "video_path": str(video_path),
        "avg_score": avg_score,
        "motion_score": motion_score,
        "passes": passes,
        "qc_report": str(qc_report_path),
        "frame_scores": frame_scores
    }


def main():
    """
    Test the video QC wrapper
    """
    # Test with existing video
    test_video = Path("./sgflix_runs/run_007_doctor_donald_miracle_ward/RUN_007_MASTER_PACKAGE/renders/temporal_reveals/temporal_reveal_099_doctor_donald.mp4")
    
    if not test_video.exists():
        print(f"❌ Test video not found: {test_video}")
        return
    
    # Run test
    result = render_video_with_qc(test_video)
    
    # Print result
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"OK: {result['ok']}")
    print(f"Avg Score: {result['avg_score']}/10" if result['avg_score'] else "Avg Score: N/A")
    print(f"Motion Score: {result['motion_score']}/10" if result['motion_score'] else "Motion Score: N/A")
    print(f"QC Report: {result['qc_report']}")


if __name__ == "__main__":
    main()
