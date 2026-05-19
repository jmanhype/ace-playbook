# SGFLIX Factory - Actual Inputs/Outputs Map

**Date**: 2026-05-18
**What I actually found by investigating run_020**

---

## 📊 Complete Factory I/O Map

### Phase -1: Worthiness Audit
**Inputs:**
- Trend research (external)
- Entropy analysis (external)

**Outputs:**
```
strategy/
├── phase_minus_one_worthiness_audit.json
├── tribe_meta_score.json
├── winner_decision.md
└── risk_taste_score.json
```

---

### Phase 6: CHAI Shot-Language Spec
**Inputs:**
- Strategy decisions (Phase -1/0)
- Creative direction

**Outputs:**
```
chai/
└── chai_shot_specs.json
    {
      "subject": "elderly protest leader with petition",
      "scene": "private media dinner entrance",
      "motion": "slow push-in, waiter lifts cloche",
      "camera": "vertical 9:16, low angle, 35mm",
      "critique": "Must read as satire, not news",
      "revision": "If text messy, crop tighter"
    }
```

**Used by:**
- Phase 7 (first frame generation)
- Phase 13 (validation)

---

### Phase 7: GPT Image First Frames
**Inputs:**
- `chai/chai_shot_specs.json`

**Outputs:**
```
frames/gpt_image_2/
├── first_frame_v01_prompt.md          # Prompt for GPT-Image-2
├── first_frame_v01.png                # Generated image
├── first_frame_v01_qc.md               # QC notes
├── first_frame_v02_repair_prompt.md   # If repair needed
└── first_frame_v01.png (final)        # Approved version
```

**Used by:**
- Phase 9: scene_json (references first_frame.png)
- Phase 13: validation against CHAI spec
- Phase 15: overlay export
- Phase 16: caption generation

---

### Phase 8: Storyboards (Shared Choices)
**Inputs:**
- Creative direction
- Character bibles

**Outputs:**
```
storyboards/shared_choices/
├── shared_choices_v01_prompt.md      # Prompt for storyboard
├── shared_choices_v01.png             # Storyboard image (2.3MB)
└── shared_choices_v01_qc.md           # QC notes
```

**Used by:**
- Phase 13: visual reference
- Handoffs: grok_agent_prompt.md, closed_tool_handoff.json
- Phase 14: human QC reference

---

### Phase 9: Scene JSON
**Inputs:**
- `frames/gpt_image_2/first_frame_v01.png`

**Outputs:**
```
scene_json/
└── shot_001.json
    {
      "shot_id": "shot_001",
      "source_frame": "frames/gpt_image_2/first_frame_v01.png",
      "duration_seconds": 10,
      "beats": ["Placard holds frame", "Camera reveals", "Waiter lifts cloche"],
      "overlay_plan": ["MERGER DINNER HAD A MENU", "AUDIENCE CHOICE WAS NOT ON IT"]
    }
```

**Used by:**
- Phase 11: render instructions
- Phase 13: motion validation

---

### Phase 13: CHAI Critique
**Inputs:**
- `chai/chai_shot_specs.json` (original spec)
- `frames/gpt_image_2/first_frame_v01.png` (actual output)
- Phase 11 render output

**Outputs:**
```
qc/
├── first_frame_v01_qc.md              # Already created in Phase 7
└── (could add:)
    └── phase_13_critique_notes.md      # Validates against CHAI spec
```

**What it checks:**
- Does output match CHAI spec?
- Are critique items addressed?
- Is revision plan needed?

---

### Phase 14-16: Distribution Package
**Inputs:**
- `frames/gpt_image_2/first_frame_v01.png`
- `scene_json/shot_001.json`

**Outputs:**
```
captions/
└── instagram_caption.md
    "The merger had a private dinner before the audience had a choice.
    Satire based on reported Hollywood opposition to Paramount-Warner.
    #filmindustry #hollywood #satire"

distribution/
└── post_plan.md
    "Hook frame: use first_frame_v01.png, cropped to preserve placard.
     Include 'satire' in caption.
     Do not tag real companies."
```

---

### Handoffs: Cross-System Integration
**Inputs:**
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`
- CHAI specs
- Scene JSON

**Outputs:**
```
handoffs/
├── grok_agent_prompt.md              # For Grok analysis
│   "Use approved stills as references:
│    - frames/gpt_image_2/first_frame_v01.png
│    - storyboards/shared_choices/shared_choices_v01.png
│    Concept: Merger Dinner Picket satire...
│    Hard constraints: No official logos, no fake news..."
│
└── closed_tool_handoff.json          # For other tools (Kling, etc.)
    {
      "visual_references": ["first_frame_v01.png", "shared_choices_v01.png"],
      "avoid": ["exact studio logos", "fake news framing"],
      "video_generation_allowed": false
    }
```

**Used by:**
- Grok agent (for analysis)
- Kling/runway (for video generation)
- Other closed tools

---

## 🔄 What Connects to What

```
Strategy (Phase -1/0)
    ↓
CHAI Spec (Phase 6)
    ↓
    ┌─────────────────────────────────┐
    │                                 │
    ├─→ First Frame Prompt (Phase 7)   │
    │   "elderly protest leader"       │
    │   "private dinner entrance"     │
    │                                 │
    ├─→ Scene JSON (Phase 9)          │
    │   References: first_frame.png   │
    │                                 │
    ├─→ Grok Agent Handoff            │
    │   References: first_frame.png   │
    │   References: shared_choices.png│
    │                                 │
    └─→ Caption Generator (Phase 16)  │
        "Based on reported opposition"
        "Satire of Paramount-Warner"
```

---

## 🎯 What Could Be Reused

### **Prompts as Context:**
- `first_frame_v01_prompt.md` → Could inform character Bible generation
- `shared_choices_v01_prompt.md` → Could inform style guide
- `grok_agent_prompt.md` → Could inform other AI analysis

### **Images as References:**
- `first_frame_v01.png` → Referenced by scene_json, handoffs, captions
- `shared_choices_v01.png` → Referenced by handoffs, Grok agent

### **Specs as Validation:**
- `chai_shot_specs.json` → Validates Phase 7 output
- `scene_json/shot_001.json` → Validates Phase 11 motion
- `closed_tool_handoff.json` → Validates Phase 11 render

### **QC as Feedback Loop:**
- `first_frame_v01_qc.md` → Informs `first_frame_v02_repair_prompt.md`
- QC notes → Repair prompts → Better output

---

## 🚀 What This Means for QC Integration

### **Current Factory Workflow:**
```
Generate V01 → QC发现问题 → 修复提示 → Generate V02 → 手动批准
```

### **With QC System:**
```
Generate V01 → 自动QC → 如果< 8分 → 自动修复 → QC again → 只有8+通过
```

### **What Should Use What:**

**Phase 7 QC should check:**
- ✅ Does first_frame.png match chai_shot_specs.json?
- ✅ Is quality 8/10+?

**Phase 13 QC should check:**
- ✅ Does output match CHAI spec from Phase 6?
- ✅ Does it match first_frame from Phase 7?
- ✅ Are all QC issues resolved?

**Phase 14 QC should check:**
- ✅ Does final package honor all constraints?
- ✅ Are captions consistent with visual?
- ✅ Is it safe for distribution?

---

## 📝 Key Insight

**The factory is a CONNECTED SYSTEM where:**

1. **Prompts** (md files) instruct generation
2. **Images** (png files) serve as visual references
3. **Specs** (json files) define requirements
4. **QC** (md files) validates against specs
5. **Handoffs** (md/json) pass to other systems

**Each output becomes input for the next phase!**

---

**Last Updated**: 2025-05-18
**Status**: Actual factory I/O mapped, connections identified
