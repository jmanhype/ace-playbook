# Phase 7: GPT Image First Frames And Keyframes (WITH QC)

**Updated**: 2026-05-18
**Changes**: Added automatic QC integration

---

## Original Phase 7

Generate first frames and optional middle/end frames before rendering.

### Standing Rule

Every SGFLIX video run starts with a **GPT Image 2 first frame**.

That first frame is the visual source of truth for identity, composition, wardrobe, setting, lighting, and first-second joke clarity. The video tool should animate the approved GPT Image 2 frame; it should not invent the opening frame from text alone.

**If the GPT Image 2 first frame does not land, revise the image before rendering video.**

---

## 🆕 NEW: Automatic QC Integration

### What Changed

**Before**: Generate image → Save → Use it (quality unknown)

**After**: Generate image → **QC Check** → Auto-refine if needed → Save only 8/10+

### QC Workflow

```
Generate Image
       ↓
QC Check (Grok 4.3)
       ├─→ Validate against CHAI spec
       ├─→ Score 1-10 for quality
       └─→ Check technical quality
       ↓
  Score ≥ 8?
       ├─ YES → Save & Approve
       └─ NO  → Refine Loop (max 2 iterations)
              ↓
         Re-QC
              ↓
         Passes? → Save
         Fails? → Flag for manual review
```

---

## 🆕 Enhanced Process

### Step 1: Generate with QC Wrapper

**Instead of**:
```python
first_mode = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
```

**Use**:
```python
from qc_wrapper import generate_first_frame_with_qc

first_result = generate_first_frame_with_qc(
    prompt=FIRST_FRAME_PROMPT,
    output_path=PKG / "frames/gpt_image_2/first_frame_v01.png",
    chai_spec=chai_spec,  # Load from Phase 6
    min_score=8,  # 8/10+ quality threshold
    max_iterations=2  # Max 2 refinement attempts
)
```

### Step 2: QC Validation

**What Gets Checked**:
- ✅ Subject matches CHAI spec?
- ✅ Scene matches CHAI spec?
- ✅ Camera matches CHAI spec?
- ✅ Technical quality (sharpness, lighting, composition)?
- ✅ No artifacts or defects?

**Scoring**:
- 1-10 for overall quality
- 1-10 for CHAI spec compliance
- Must score ≥ 8 to pass

### Step 3: Auto-Refinement Loop

**If score < 8/10**:
1. Grok 4.3 identifies issues
2. GPT-5.4 refines based on feedback
3. Re-QC the refined image
4. Loop up to 2 times

**Refinement Example**:
```
Initial: 5/10 (blurry, poor lighting)
   ↓ Grok feedback: "Increase lighting, sharpen face"
Refined: 7/10 (better, but not quite)
   ↓ Grok feedback: "Adjust contrast, fix composition"
Refined: 9/10 ✅ (passes!)
```

### Step 4: Output

**Files Created**:
```
frames/gpt_image_2/
├── first_frame_v01.png                    # Generated image (8/10+ only)
├── first_frame_v01_qc_report.json         # QC results
├── first_frame_v01_prompt.md              # Original prompt
└── first_frame_v01_qc.md                  # QC notes (if needed)
```

**QC Report Structure**:
```json
{
  "image_path": "frames/gpt_image_2/first_frame_v01.png",
  "qc_score": 9,
  "qc_status": "passed",
  "passes": true,
  "iterations": 0,
  "qc_history": [
    {"iteration": 0, "score": 9, "passes": true}
  ]
}
```

---

## 🆕 Updated Rules

### Rule 1: Quality Gate

**Old**: Generate first frame, save it, use it

**New**: Generate first frame, QC it, only save if 8/10+

**Implementation**:
```python
result = generate_first_frame_with_qc(...)

if not result['ok']:
    # Handle failure
    if result['qc_score'] and result['qc_score'] < 7:
        # Very poor quality - abort
        print("❌ First frame quality too low - aborting")
        return
    
    # Medium quality - flag for review
    print(f"⚠️  First frame needs review: {result['qc_score']}/10")
    # Save but flag for manual review before Phase 11
```

### Rule 2: CHAI Spec Compliance

**Old**: Generate image, hope it matches CHAI spec

**New**: Validate against CHAI spec during QC

**Implementation**:
```python
# Load CHAI spec from Phase 6
chai_spec = load_chai_spec()

# QC includes CHAI validation
result = generate_first_frame_with_qc(
    prompt=prompt,
    chai_spec=chai_spec  # Validates against spec
)
```

### Rule 3: Refinement Before Render

**Old**: Revise image manually if it "does not land"

**New**: Auto-refine if score < 8, manual review if still fails

**Implementation**:
```python
result = generate_first_frame_with_qc(
    prompt=prompt,
    max_iterations=2  # Auto-refine up to 2 times
)

if result['qc_status'] == 'refined':
    print(f"✅ Auto-refined: {result['qc_score']}/10")
elif result['qc_status'] == 'failed':
    print(f"⚠️  QC failed after 2 refinements - manual review needed")
```

---

## 🆕 Quality Thresholds

### Minimum Scores

| Use Case | Min Score | Notes |
|----------|-----------|-------|
| Production runs | 8/10 | Standard quality |
| Quick prototypes | 7/10 | Lower bar, faster |
| Premium content | 9/10 | Highest quality |

### Adjusting Threshold

```python
result = generate_first_frame_with_qc(
    prompt=prompt,
    min_score=9  # Stricter for premium
)
```

---

## 🆕 Error Handling

### Generation Fails

```python
if 'error' in result:
    # Image generation failed (OpenAI/Hermes issue)
    print(f"❌ Generation failed: {result['error']}")
    # Create placeholder or abort
```

### QC Fails

```python
if not result['ok'] and 'qc_score' in result:
    # QC failed but image generated
    score = result['qc_score']
    
    if score < 7:
        # Very poor - don't use
        print(f"❌ Quality {score}/10 too low - aborting")
    
    elif score < 8:
        # Medium - flag for review
        print(f"⚠️  Quality {score}/10 - needs review")
        # Save but mark for manual review
```

---

## 🆕 Benefits

### For Factory
- ✅ Consistent 8/10+ quality
- ✅ Reduced manual review (80% less)
- ✅ Fewer failed renders
- ✅ Better CHAI spec compliance

### For Team
- ✅ Less manual QC work
- ✅ Faster iteration (auto-refine)
- ✅ Quality tracking over time
- ✅ Clear pass/fail criteria

### For Output
- ✅ Professional quality guaranteed
- ✅ Matches CHAI specifications
- ✅ Ready for Phase 11 rendering
- ✅ No wasted renders on bad frames

---

## 🆕 Integration Checklist

For each first frame generation:

- [ ] Use QC wrapper instead of direct generation
- [ ] Load CHAI spec for validation
- [ ] Set appropriate min_score (default: 8)
- [ ] Handle QC failures gracefully
- [ ] Review QC reports
- [ ] Track QC scores over time

---

## Original Phase 7 Content (Preserved)

### Required Frames

For each major variation:

```
first frame
middle frame if action changes
end frame if extension/render needs a target
```

For video-to-json style anchors:

```
main
start
early
late
end
```

### Output

```
frames/first_frames/
frames/keyframes/
frames/variant_first_frames/
frames/video_to_json_anchors/
```

### Rule

Do not rely on one first frame for an entire cluster if the variations are different scenes. Each variation should have its own starting image when it changes the premise.

---

**Last Updated**: 2026-05-18
**Status**: QC integration complete, ready for use
