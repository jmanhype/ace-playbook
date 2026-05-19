# QC Integration Guide for Run Scripts

**Date**: 2026-05-18
**Purpose**: How to update run scripts to use QC wrapper

---

## 🎯 Overview

This guide shows how to modify existing SGFLIX run scripts to use the automatic QC system.

---

## 📋 Before vs After

### Before (Current Code)

```python
# Import
from pathlib import Path

# Generate images
first_mode = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
board_mode = generate_image(SHARED_CHOICES_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")

# Check if blocked
blocked = first_mode.startswith("blocked") or board_mode.startswith("blocked")
```

### After (With QC)

```python
# Import
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from qc_wrapper import generate_first_frame_with_qc

# Generate images WITH QC
first_result = generate_first_frame_with_qc(
    prompt=FIRST_FRAME_PROMPT,
    output_path=PKG / "frames/gpt_image_2/first_frame_v01.png",
    chai_spec=None,
    min_score=8,
    max_iterations=2
)

board_result = generate_first_frame_with_qc(
    prompt=SHARED_CHOICES_PROMPT,
    output_path=PKG / "storyboards/shared_choices/shared_choices_v01.png",
    chai_spec=None,
    min_score=8,
    max_iterations=2
)

# Check results
blocked = not first_result['ok'] or not board_result['ok']
```

---

## 🔧 Step-by-Step Integration

### Step 1: Add Import

Add at the top of the script after existing imports:

```python
import sys
from pathlib import Path

# Add QC wrapper import
sys.path.insert(0, str(Path(__file__).parent))
from qc_wrapper import generate_first_frame_with_qc
```

### Step 2: Replace generate_image() Calls

**Find**:
```python
first_mode = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
```

**Replace with**:
```python
first_result = generate_first_frame_with_qc(
    prompt=FIRST_FRAME_PROMPT,
    output_path=PKG / "frames/gpt_image_2/first_frame_v01.png",
    min_score=8,
    max_iterations=2
)
```

### Step 3: Handle QC Results

**Add QC logging**:
```python
print(f"First frame QC: {first_result['qc_score']}/10 - {first_result['qc_status']}")
```

**Check pass/fail**:
```python
if not first_result['ok']:
    print(f"⚠️  First frame failed QC: {first_result.get('error', 'Quality check failed')}")
    # Handle failure - log, abort, or flag for manual review
```

### Step 4: Save QC Summary

**Add to script output**:
```python
qc_summary = {
    "run_id": RUN,
    "first_frame_qc": {
        "ok": first_result['ok'],
        "score": first_result.get('qc_score'),
        "status": first_result.get('qc_status'),
        "report": first_result.get('qc_report')
    },
    "shared_choices_qc": {
        "ok": board_result['ok'],
        "score": board_result.get('qc_score'),
        "status": board_result.get('qc_status'),
        "report": board_result.get('qc_report')
    }
}

write_json(PKG / "qc_summary.json", qc_summary)
```

---

## 📊 Return Value Changes

### Old Return Format

```python
{
    "ok": True,
    "mode": "openai_gpt_image_api"
}
```

### New Return Format

```python
{
    "ok": True,  # True if passed QC, False otherwise
    "mode": "openai_gpt_image_api",
    "path": "path/to/image.png",
    "qc_score": 9,  # Final QC score
    "qc_status": "passed",  # "passed", "refined", or "failed"
    "qc_report": "path/to/qc_report.json",
    "iterations": 0  # Number of refinement iterations
}
```

---

## ⚠️ Error Handling

### Generation Failure

```python
if not result['ok']:
    if 'error' in result:
        # Image generation failed
        print(f"❌ Generation failed: {result['error']}")
        # Handle same as before - create placeholder or abort
    else:
        # QC failed
        print(f"⚠️  QC failed: {result['qc_score']}/10 < 8")
        # Option 1: Flag for manual review
        # Option 2: Save but mark as failed
        # Option 3: Abort run
```

### QC Failure Options

**Option 1: Flag and Continue**
```python
if not result['ok']:
    # Save image but mark for review
    write_text(PKG / "frames/gpt_image_2/needs_review.txt", 
              f"QC failed with score {result['qc_score']}/10")
```

**Option 2: Abort Run**
```python
if not result['ok']:
    print("❌ QC failed - aborting run")
    sys.exit(1)
```

**Option 3: Manual Review Queue**
```python
if not result['ok']:
    # Add to manual review queue
    review_queue.append({
        "image": result['path'],
        "score": result['qc_score'],
        "reason": "QC failed"
    })
```

---

## 🧪 Testing

### Test QC Integration

1. **Run modified script**:
   ```bash
   python3 scripts/create_run_024_rock_tint_meter_WITH_QC.py
   ```

2. **Check outputs**:
   - Images generated: `frames/gpt_image_2/first_frame_v01.png`
   - QC reports: `frames/gpt_image_2/first_frame_v01_qc_report.json`
   - QC summary: `qc_summary.json`

3. **Verify QC scores**:
   - All saved images should have 8/10+ scores
   - QC reports show detailed results
   - Failed images flagged appropriately

---

## 📁 Files to Modify

### Priority 1 (Test First)
- `scripts/create_run_024_rock_tint_meter.py`
- `scripts/create_run_070_amazon_price_match_court.py`

### Priority 2 (Active Runs)
- Any run scripts actively being used
- Recent run scripts (last 30 days)

### Priority 3 (Backlog)
- All remaining run scripts
- Archive scripts (historical reference)

---

## ✅ Checklist

For each run script:

- [ ] Add QC wrapper import
- [ ] Replace `generate_image()` with `generate_first_frame_with_qc()`
- [ ] Update error handling
- [ ] Add QC logging
- [ ] Save QC summary
- [ ] Test with actual run
- [ ] Verify QC reports created
- [ ] Document any issues

---

## 🚀 Rollout Plan

### Phase 1: Test (1 day)
1. Modify 1-2 run scripts
2. Test with actual runs
3. Verify QC works correctly
4. Fix any issues

### Phase 2: Deploy (1 week)
1. Update all active run scripts
2. Monitor QC results
3. Gather feedback
4. Refine as needed

### Phase 3: Standardize (ongoing)
1. Make QC wrapper default in new run scripts
2. Update factory SOP
3. Train team on new workflow
4. QC becomes standard practice

---

## 💡 Tips

### Tip 1: Gradual Rollout
Start with non-critical runs to test QC integration before using on production runs.

### Tip 2: Monitor QC Scores
Track QC scores over time to identify patterns:
- Which prompts score highest?
- Which CHAI specs are hardest to meet?
- Average QC score per run?

### Tip 3: Adjust Thresholds
If 8/10 is too strict, adjust `min_score` parameter:
- 7/10 for more lenient QC
- 9/10 for stricter QC
- Match to your quality standards

---

**Status**: Guide complete, ready for implementation
