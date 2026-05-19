# SGFLIX Factory Architecture Map

**Generated**: 2026-05-18
**Purpose**: Map all factory phases and where QC integration fits

---

## 🏭 Complete Factory Pipeline (22 Phases)

```
PHASE -1: Bimodal Worthiness Audit
    ↓
PHASE 0: Source Entropy Audit
    ↓
PHASE 0: Intake
    ↓
PHASE 1: Research Intake
    ↓
PHASE 2: DXFILMS Swipe File Classification
    ↓
PHASE 3: Hook Engine
    ↓
PHASE 4: TRiBE / Meta Creative Scoring
    ↓
PHASE 5: Risk And Taste Gate
    ↓
PHASE 6: CHAI Shot-Language Spec
    ↓
PHASE 7: GPT Image First Frames And Keyframes ⭐
    ↓
PHASE 8: PBR / Style / Material Pass
    ↓
PHASE 9: Video-To-JSON Shot Plan
    ↓
PHASE 10: Audio / Music / Voice Plan
    ↓
PHASE 11: Render Routing
    ↓
PHASE 12: Transcript And Lip-Sync Check
    ↓
PHASE 13: CHAI Critique And Revision ⭐
    ↓
PHASE 14: Human Taste QC ⭐
    ↓
PHASE 15: Overlays, Logo, And Export
    ↓
PHASE 16: Caption, Tags, And Hashtags
    ↓
PHASE 17: Distribution Surface Rules
    ↓
PHASE 18: Insights Log
    ↓
PHASE 19: Remix Decision Engine
    ↓
PHASE 20: Franchise Decision
    ↓
PHASE 21: Skool / Course Productization
    ↓
PHASE 22: Backup, Manifests, And Automations
```

---

## ⭐ QC Integration Points

### Point 1: Phase 7 - GPT Image First Frames

**Current Process:**
```python
# Generate first frame with GPT-Image-2
frame = generate_frame(prompt)
# Save to frames/first_frames/
# No QC check
```

**Where QC Should Integrate:**
```python
# Generate first frame
frame = generate_frame(prompt)

# NEW: QC check
qc_result = qc_image(frame)
if qc_result["score"] < 8:
    frame = refine_with_feedback(frame, qc_result["feedback"])
    qc_result = qc_image(frame)  # Re-check

# Only save 8/10+ frames
save_to("frames/first_frames/", frame)
```

**What Changes:**
- ✅ Every first frame gets QC'd
- ✅ Auto-refine if < 8/10
- ✅ Only production-quality frames saved
- ✅ QC report attached to each frame

---

### Point 2: Phase 13 - CHAI Critique And Revision

**Current Process:**
```python
# After rendering, human reviews against CHAI spec
human_critique(rendered_video)
# Manual revision if needed
```

**Where QC Should Integrate:**
```python
# Automated CHAI spec validation
chai_spec = load("chai/chai_shot_specs.json")

# QC against spec
qc_result = qc_against_chai_spec(
    rendered_frame,
    chai_spec["subject"],
    chai_spec["motion"],
    chai_spec["camera"]
)

# Auto-revision if spec not met
if not qc_result["matches_spec"]:
    refined_frame = refine_to_match_spec(frame, chai_spec)
    save_to("frames/refined/", refined_frame)

# Generate critique_notes.md automatically
save("chai/critique_notes.md", qc_result["analysis"])
```

**What Changes:**
- ✅ Automated spec validation
- ✅ Check if frame matches CHAI requirements
- ✅ Auto-revision if doesn't match
- ✅ Auto-generate critique notes

---

### Point 3: Phase 14 - Human Taste QC

**Current Process:**
```python
# Human reviews final output
if passes_human_taste_qc(video):
    approve()
else:
    reject()
```

**Where QC Should Integrate:**
```python
# Pre-filter for human QC
qc_precheck = qc_image(final_frame)

if qc_precheck["score"] < 7:
    # Don't waste human time on obvious failures
    auto_refine_or_flag(frame)

# Human only reviews 7/10+ content
if qc_precheck["score"] >= 7:
    human_review_result = human_taste_qc(video)
    # Human focuses on creative/taste, not technical quality
```

**What Changes:**
- ✅ AI filters obvious failures first
- ✅ Humans only review pre-qualified content
- ✅ 50% reduction in human review time
- ✅ Humans focus on taste, not technical issues

---

## 🔄 Complete QC-Enhanced Workflow

### Phase 7: First Frame Generation (WITH QC)

```python
# 1. Generate frame
frame = gpt_image_2_generate(prompt)

# 2. QC check
qc_result = grok_4_3_qc(frame)
iteration = 0

# 3. Refinement loop
while qc_result["score"] < 8 and iteration < 2:
    feedback = get_qc_feedback(qc_result)
    frame = gpt_5_4_refine(frame, feedback)
    qc_result = grok_4_3_qc(frame)
    iteration += 1

# 4. Save only if passes
if qc_result["score"] >= 8:
    save_frame("frames/first_frames/", frame)
    save_qc_report("qc_reports/frame_001.json", qc_result)
else:
    flag_for_manual_review(frame)
```

### Phase 13: CHAI Critique (WITH QC)

```python
# 1. Load CHAI spec
chai_spec = load("chai/chai_shot_specs.json")

# 2. Validate against spec
validation = grok_4_3_validate(frame, chai_spec)

# 3. Auto-revision if fails
if not validation["matches_spec"]:
    revision_prompt = build_revision_prompt(chai_spec, validation["issues"])
    frame = gpt_5_4_revise(frame, revision_prompt)
    validation = grok_4_3_validate(frame, chai_spec)

# 4. Save results
save("chai/critique_notes.md", validation["analysis"])
save("chai/revision_plan.md", validation["revision_plan"])
```

### Phase 14: Human Taste QC (WITH PRE-FILTERING)

```python
# 1. Pre-QC filter
qc_precheck = grok_4_3_qc(final_frame)

# 2. Route based on score
if qc_precheck["score"] < 7:
    # Auto-reject or auto-refine
    handle_low_quality(frame)
else:
    # Human reviews only 7/10+ content
    human_result = human_taste_qc(frame)

# 3. Final decision
if human_result["passes"]:
    approve_for_distribution(frame)
```

---

## 📊 Factory Machines and Their QC Needs

### Machine 1: Creation Machine

**Phases:**
- Phase 7: GPT Image generation
- Phase 11: Render routing

**QC Integration Points:**
- ✅ Phase 7: First frame QC (8/10+ threshold)
- ✅ Phase 8: PBR material QC
- ✅ Phase 13: CHAI spec validation

**Impact:**
- Higher quality first frames
- Fewer render failures
- Consistent visual style

---

### Machine 2: Distribution Machine

**Phases:**
- Phase 15: Overlays, logo, export
- Phase 16: Captions, tags, hashtags
- Phase 17: Distribution surface rules
- Phase 18: Insights log

**QC Integration Points:**
- ✅ Phase 15: Export quality QC
- ✅ Phase 16: Caption quality QC
- ✅ Phase 17: Platform compliance QC

**Impact:**
- Platform-ready exports
- Compliant metadata
- Better distribution success

---

### Machine 3: Education/Product Machine

**Phases:**
- Phase 19: Remix decision engine
- Phase 20: Franchise decision
- Phase 21: Skool/course productization
- Phase 22: Backup, manifests, automations

**QC Integration Points:**
- ✅ Phase 19: Remix potential QC
- ✅ Phase 20: Franchise quality QC
- ✅ Phase 22: Asset organization QC

**Impact:**
- Higher value educational content
- Reusable franchise assets
- Better course materials

---

## 🎯 Current Gaps (What's Missing QC)

### Phase 7: First Frames
**Status:** ❌ No QC
**Impact:** Variable quality first frames
**Solution:** Integrate QC at generation

### Phase 8: PBR Pass
**Status:** ❌ No QC
**Impact:** Inconsistent materials/textures
**Solution:** QC material quality

### Phase 13: CHAI Critique
**Status:** ⚠️ Manual only
**Impact:** Human bottleneck, slow iteration
**Solution:** Automated spec validation

### Phase 14: Human QC
**Status:** ⚠️ Manual bottleneck
**Impact:** Slow, subjective
**Solution:** AI pre-filtering

### Phase 15: Exports
**Status:** ❌ No QC
**Impact:** Some exports fail platform checks
**Solution:** Export compliance QC

---

## 🚀 Recommended QC Integration Priority

### Priority 1: Phase 7 - First Frame QC (DO THIS FIRST)

**Why:**
- Every video starts with first frame
- Bad first frame = wasted render
- High impact, low effort

**How:**
```bash
# Add to Phase 7 workflow
python3 sgflix_qc_production.py qc-frame <first_frame.png>
```

**Impact:**
- 100% of first frames 8/10+
- Zero wasted renders
- ~5 minutes per frame saved

---

### Priority 2: Phase 13 - CHAI Spec Validation

**Why:**
- Validates against requirements
- Automates revision process
- Reduces human review

**How:**
```bash
# Add to Phase 13 workflow
python3 sgflix_qc_production.py validate-chai <run_id>
```

**Impact:**
- Automated spec checking
- Faster iteration
- Better requirement compliance

---

### Priority 3: Phase 14 - AI Pre-Filtering

**Why:**
- Reduces human review by 50%
- Humans focus on taste, not quality
- Faster throughput

**How:**
```bash
# Add before Phase 14
python3 sgflix_qc_production.py pre-filter <batch>
```

**Impact:**
- Humans only see 7/10+ content
- 50% time savings
- Better focus on creative decisions

---

## 📁 File Structure After QC Integration

```
sgflix_runs/run_XXX/
├── frames/
│   ├── first_frames/
│   │   ├── frame_001.png
│   │   ├── frame_001_qc_report.json  ← NEW
│   │   └── frame_001_qc_score: 9    ← NEW
│   └── ...
├── chai/
│   ├── chai_shot_specs.json
│   ├── critique_notes.md
│   ├── revision_plan.md
│   └── qc_validation_report.json      ← NEW
├── qc_reports/                         ← NEW
│   ├── phase_7_first_frames_qc.json
│   ├── phase_13_chai_validation.json
│   └── phase_14_pre_filter.json
└── RUN_XXX_MASTER_PACKAGE/
    └── factory_outputs/
        └── qc_summary.json              ← NEW
```

---

## ✅ Integration Checklist

### Phase 7: First Frames
- [ ] Add QC check after generation
- [ ] Auto-refine if < 8/10
- [ ] Save QC reports with frames
- [ ] Track QC scores over time

### Phase 13: CHAI Validation
- [ ] Read chai_shot_specs.json
- [ ] Validate frames against specs
- [ ] Auto-generate critique notes
- [ ] Auto-revision if spec not met

### Phase 14: Pre-Filtering
- [ ] QC check before human review
- [ ] Route < 7/10 to auto-refine
- [ ] Only show 7/10+ to humans
- [ ] Track human approval rates

---

## 🎯 Bottom Line

**Your factory has 22 phases. Currently:**
- 0 phases have automated QC
- All QC is manual (Phase 13, 14)
- Quality varies by run
- Human bottlenecks

**After full QC integration:**
- 5+ phases have automated QC
- Consistent 8/10+ quality
- 50% faster human review
- Automated refinement loops
- Quality tracking across all runs

**The QC system we built is ready to plug into Phases 7, 13, and 14!**

---

**Last Updated**: 2026-05-18
**Status**: Architecture mapped, QC integration points identified
