# SGFLIX QC Integration Design

**Date**: 2026-05-18
**Purpose**: Design QC integration into factory workflow
**Status**: Design Phase

---

## 📊 Current State Analysis

### Phase 7 (First Frame Generation)
**Current Implementation**:
```python
def generate_image(prompt, path, size):
    from openai import OpenAI
    result = OpenAI().images.generate(model="gpt-image-1", prompt=prompt, size=size)
    path.write_bytes(base64.b64decode(result.data[0].b64_json))
    return {"ok": True}
```

**Location**: Multiple run scripts in `scripts/`
**Output**: `frames/gpt_image_2/first_frame_v01.png`
**Problem**: No QC check, saves all images regardless of quality

### Phase 11 (Video Rendering)
**Current Implementation**:
- Videos rendered on 3090 (ComfyUI) or closed tools (Kling)
- Saved to `renders/` directory
- Manual QC inspection via Phase 12/13
**Problem**: No automated QC, inconsistent quality

---

## 🎯 QC Integration Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GENERATION PHASE                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      QC CHECK (NEW)                         │
│  - Load CHAI spec                                           │
│  - Validate output matches spec                             │
│  - Score 1-10 for quality/compliance                        │
└─────────────────────────────────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
              Score ≥ 8         Score < 8
                    │               │
                    ▼               ▼
              ┌─────────┐    ┌─────────────┐
              │  APPROVE │    │  REFINE     │
              │  & SAVE  │    │  LOOP       │
              └─────────┘    │  (max 2x)   │
                             └─────────────┘
                                    │
                                    ▼
                             ┌─────────────┐
                             │  RE-QC      │
                             │  refined    │
                             └─────────────┘
                                    │
                             ┌────────┴────────┐
                             │                 │
                             ▼                 ▼
                       Score ≥ 8         Score < 8
                             │                 │
                             ▼                 ▼
                       ┌─────────┐      ┌──────────┐
                       │ APPROVE │      │ FLAG     │
                       │ & SAVE  │      │ MANUAL   │
                       └─────────┘      │ REVIEW  │
                                        └──────────┘
```

---

## 🔧 Integration Points

### Point 1: Phase 7 - First Frame QC

**Location**: After `generate_image()` in run scripts
**When**: Immediately after image generation
**What**: Validate against CHAI spec

**Current Code**:
```python
first_mode = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
```

**New Code**:
```python
from sgflix_qc_production import SGFLIXProductionQC

# Generate image
result = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")

# QC check (NEW)
qc = SGFLIXProductionQC(min_score=8, max_iterations=2)
qc_result = qc.qc_with_refinement(str(PKG / "frames/gpt_image_2/first_frame_v01.png"))

# Only proceed if passes
if qc_result["status"] != "passed":
    print(f"⚠️  First frame failed QC: {qc_result['final_score']}/10")
    # Handle failure - flag for manual review or abort
```

**Files to Modify**:
- `scripts/create_run_*.py` (all run creation scripts)
- Create shared wrapper: `scripts/qc_wrapper.py`

---

### Point 2: Phase 11 - Video QC

**Location**: After video rendering completes
**When**: Before moving to Phase 12 (transcript check)
**What**: Frame extraction + batch QC

**Integration**:
```python
from video_qc_comprehensive import VideoQC

# After video render
video_path = renders_dir / "output.mp4"

# QC check (NEW)
qc = VideoQC()
qc_result = qc.qc_video(video_path, chai_spec)

if qc_result["avg_score"] < 8:
    print(f"⚠️  Video failed QC: {qc_result['avg_score']}/10")
    # Flag for re-render or manual review
```

**Files to Modify**:
- Rendering scripts on 3090
- Post-render workflow
- Create: `scripts/video_qc_wrapper.py`

---

## 📋 Detailed Workflow

### Phase 7 Enhanced Workflow

```python
def generate_first_frame_with_qc(prompt, output_path, chai_spec):
    """
    Generate first frame with automatic QC
    """
    # Step 1: Generate image
    image_result = generate_image(prompt, output_path, "1024x1536")
    
    if not image_result.get("ok"):
        return image_result
    
    # Step 2: QC check (NEW)
    qc = SGFLIXProductionQC(min_score=8, max_iterations=2)
    qc_result = qc.qc_with_refinement(str(output_path))
    
    # Step 3: Save QC report (NEW)
    qc_report_path = output_path.parent / f"{output_path.stem}_qc_report.json"
    qc_report_path.write_text(json.dumps(qc_result, indent=2))
    
    # Step 4: Return result with QC status (NEW)
    return {
        "ok": qc_result["status"] == "passed",
        "mode": image_result.get("mode"),
        "path": str(output_path),
        "qc_score": qc_result.get("final_score"),
        "qc_status": qc_result["status"],
        "qc_report": str(qc_report_path)
    }
```

---

### Phase 11 Enhanced Workflow

```python
def render_video_with_qc(render_input, output_path, chai_spec):
    """
    Render video with automatic QC
    """
    # Step 1: Render video
    video_result = render_video(render_input, output_path)
    
    if not video_result.get("ok"):
        return video_result
    
    # Step 2: Extract frames for QC (NEW)
    frames_dir = output_path.parent / "qc_frames"
    frames = extract_frames(output_path, frames_dir, num_frames=5)
    
    # Step 3: QC each frame (NEW)
    qc = SGFLIXProductionQC(min_score=8, max_iterations=2)
    frame_scores = []
    
    for frame in frames:
        result = qc.qc_image(str(frame))
        frame_scores.append(result["score"])
    
    # Step 4: Check motion consistency (NEW)
    motion_score = check_motion_consistency(frames)
    
    # Step 5: Calculate average (NEW)
    avg_score = sum(frame_scores) / len(frame_scores)
    
    # Step 6: Save QC report (NEW)
    qc_report = {
        "video_path": str(output_path),
        "frame_scores": frame_scores,
        "avg_score": avg_score,
        "motion_score": motion_score,
        "passes": avg_score >= 8 and motion_score >= 7
    }
    
    qc_report_path = output_path.parent / f"{output_path.stem}_qc_report.json"
    qc_report_path.write_text(json.dumps(qc_report, indent=2))
    
    # Step 7: Return result with QC status (NEW)
    return {
        "ok": qc_report["passes"],
        "path": str(output_path),
        "qc_score": avg_score,
        "motion_score": motion_score,
        "qc_report": str(qc_report_path)
    }
```

---

## 🔄 Refinement Loop Design

### Auto-Refinement Logic

```python
def qc_with_refinement(image_path, max_iterations=2):
    """
    QC with automatic refinement loop
    """
    current_image = image_path
    iteration = 0
    
    while iteration <= max_iterations:
        # QC check
        qc_result = qc_image(current_image)
        
        if qc_result["score"] >= 8:
            return {
                "status": "passed",
                "final_image": current_image,
                "final_score": qc_result["score"],
                "iterations": iteration
            }
        
        # Need refinement
        if iteration < max_iterations:
            # Get feedback
            feedback = get_qc_feedback(current_image, qc_result)
            
            # Refine using GPT-5.4
            refined_image = refine_with_gpt(current_image, feedback)
            
            if refined_image:
                current_image = refined_image
            else:
                break
        
        iteration += 1
    
    # Max iterations reached
    return {
        "status": "failed",
        "final_image": current_image,
        "final_score": qc_image(current_image)["score"],
        "iterations": iteration
    }
```

---

## 📊 Success Criteria

### Phase 7 QC Success
- ✅ 100% of saved first frames score 8/10+
- ✅ QC reports attached to each frame
- ✅ Failed frames flagged for manual review
- ✅ Refinement loop reduces manual work by 80%

### Phase 11 QC Success
- ✅ 100% of approved videos score 8/10+ average
- ✅ Motion consistency validated
- ✅ Frame-by-frame analysis complete
- ✅ QC reports saved for audit trail

---

## 🚀 Implementation Plan

### Phase 1: Wrapper Functions (1 hour)
1. Create `scripts/qc_wrapper.py` - Phase 7 QC wrapper
2. Create `scripts/video_qc_wrapper.py` - Phase 11 QC wrapper
3. Test with existing runs

### Phase 2: Modify Run Scripts (1 hour)
1. Update `generate_image()` calls to use QC wrapper
2. Add QC checks to render workflow
3. Test with new run

### Phase 3: Error Handling (30 min)
1. Handle QC failures gracefully
2. Log all QC results
3. Create manual review queue

### Phase 4: Documentation (30 min)
1. Update factory SOP
2. Create quick start guide
3. Train team

---

## ✅ Next Steps

1. ✅ Design complete
2. → Create QC wrapper functions
3. → Modify run scripts
4. → Test with new run
5. → Deploy to production

---

**Status**: Design complete, ready for implementation
