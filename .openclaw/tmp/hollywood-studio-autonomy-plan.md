# Plan: Hollywood Studio Autonomy

**Generated**: 2026-04-18
**Estimated Complexity**: High

## Overview
The current system is a functioning proto-studio: Paperclip manages issues, proof reports, reusable lane playbooks, and real media generation. What it does not yet provide is a finished autonomous Hollywood-grade production system. The gap is not raw rendering capability; it is repeatability, continuity, editorial judgment, audio finish, studio operations, and predictable quality under load.

The plan below turns `HOL-33` through `HOL-39` into a dependency-ordered program:
- first prove repeatable multi-scene storyboard execution
- then solve cross-scene continuity
- then add editorial and audio intelligence
- then harden studio operations
- finally prove output quality under pressure

## Current Baseline
- Proven: Paperclip control layer, proof ledger, lane templates, and one canonical storyboard scene (`HARBOR_NIGHT_S01`)
- Proven: generation, finishing, stitch, captions, interpolation, lip-sync, ACE-Step music lanes
- Not yet proven: repeatable second-scene storyboard execution, actor continuity across scenes, robust location continuity across scenes, taste-level editorial decisions, film-grade audio post, studio scheduling/review operations, and stress-tested autonomy

## Prerequisites
- Live Paperclip Hollywood Studio workspace on the 3090 remains the operational source of truth
- ComfyUI/Wan/LTX/ACE-Step lanes remain accessible and stable
- Proof reporting format remains mandatory for every issue
- Shot-package YAML and scene proof conventions from `HOL-33` remain canonical

## Sprint 1: Repeatable Storyboard Execution
**Goal**: Turn one-scene capability into a repeatable scene-production loop.
**Related Epics**: `HOL-33`
**Demo/Validation**:
- Produce a second storyboarded scene with ordered shot packages, QC, and stitched proof
- Show that Scene 02 follows the same contract as Scene 01
- Verify both scenes can be reconstructed from their package docs and proof reports alone

### Task 1.1: Run Second-Scene Reliability Proof
- **Location**: Paperclip `HOL-45`
- **Description**: Choose a second scene, define its shot package, run first proof shot, expand on pass, stitch the scene, reconcile results in Paperclip
- **Complexity**: 6/10
- **Dependencies**: Existing `HOL-33` Scene 01 package and proof flow
- **Acceptance Criteria**:
  - Scene 02 has at least 3 ordered shots
  - Each shot has prompt, pass criteria, output path, and proof status
  - Final stitched scene artifact exists and is logged in Paperclip
- **Validation**:
  - `ffprobe` on all outputs
  - QC verdict in Paperclip proof comment

### Task 1.2: Canonicalize Shot-Package Schema
- **Location**: Live Paperclip Hollywood Studio shared docs
- **Description**: Lock down one schema for scene metadata, continuity targets, QC questions, per-shot state, and stitched-scene outputs
- **Complexity**: 4/10
- **Dependencies**: Scene 01 and Scene 02 examples
- **Acceptance Criteria**:
  - No ad hoc fields between scenes
  - One template can scaffold any new scene package
- **Validation**:
  - Diff Scene 01 and Scene 02 packages for schema consistency

### Task 1.3: Add Batch Continuation Rules
- **Location**: `HOL-33` playbook / executor docs
- **Description**: Encode the exact rule for when a first proof shot is enough to continue, when the sequence must pause, and how a failed detail shot is iterated
- **Complexity**: 4/10
- **Dependencies**: Scene 02 run data
- **Acceptance Criteria**:
  - Continuation logic is explicit and reusable
  - Retry behavior is standardized
- **Validation**:
  - Manual walkthrough of both scenes against the written rules

## Sprint 2: Cross-Scene Continuity Stack
**Goal**: Make scenes consistent with each other, not only internally coherent.
**Related Epics**: `HOL-34`, `HOL-35`
**Demo/Validation**:
- Produce two or more scenes with preserved character/location identity
- Run continuity review across scenes rather than only per-shot QC

### Task 2.1: Actor Identity Bible
- **Location**: New Paperclip issue under `HOL-34`
- **Description**: Define canonical reference pack, do-not-drift constraints, acceptable variation bands, and identity QC checklist
- **Complexity**: 7/10
- **Dependencies**: Repeatable scene production from Sprint 1
- **Acceptance Criteria**:
  - Same actor reads as the same actor across multiple scenes
  - Identity failures are diagnosable against a written contract
- **Validation**:
  - Side-by-side scene comparison with explicit QC

### Task 2.2: Location Continuity Bible
- **Location**: New Paperclip issue under `HOL-35`
- **Description**: Define location anchor assets, lighting palette, environmental constraints, and continuity inheritance rules across shots/scenes
- **Complexity**: 7/10
- **Dependencies**: Repeatable scene production from Sprint 1
- **Acceptance Criteria**:
  - Same harbor/street/interior family remains recognizable across scenes
  - Drift is caught before stitch/export
- **Validation**:
  - Continuity checklist on at least two scenes in the same setting

### Task 2.3: Continuity Scoring
- **Location**: New Paperclip issue under `HOL-34` or `HOL-35`
- **Description**: Add a lightweight continuity scorecard covering actor, wardrobe/props, lighting, and location consistency
- **Complexity**: 5/10
- **Dependencies**: Actor and location bibles
- **Acceptance Criteria**:
  - Every scene receives a comparable continuity score
  - Scores determine pass / iterate / reject
- **Validation**:
  - Scorecard attached to at least two multi-scene proofs

## Sprint 3: Editorial Intelligence
**Goal**: Move from “technical concat” to editorially defensible sequence assembly.
**Related Epics**: `HOL-36`
**Demo/Validation**:
- Produce an edited sequence whose shot order, pacing, and transitions are justified by a written beat or editorial intent

### Task 3.1: Editorial Beat-Sheet Contract
- **Location**: New Paperclip issue under `HOL-36`
- **Description**: Define beat types, coverage needs, shot purpose labels, and allowed transitions
- **Complexity**: 6/10
- **Dependencies**: Sprint 1 repeatability
- **Acceptance Criteria**:
  - Every shot serves a declared editorial purpose
  - Stitch choices are grounded in the beat sheet
- **Validation**:
  - One edited scene sequence with beat annotations

### Task 3.2: Sequence Assembly Rules
- **Location**: New Paperclip issue under `HOL-36`
- **Description**: Encode cut/hold/xfade rules, acceptable rhythm changes, and escalation conditions when a scene lacks usable coverage
- **Complexity**: 6/10
- **Dependencies**: Beat-sheet contract
- **Acceptance Criteria**:
  - Sequence assembly is reproducible
  - The system knows when to request another shot instead of forcing a weak edit
- **Validation**:
  - Re-edit of one scene with explicit rationale per transition

## Sprint 4: Audio Post Stack
**Goal**: Upgrade from isolated audio lanes to a coordinated sound pipeline.
**Related Epics**: `HOL-37`
**Demo/Validation**:
- Produce one scene with score/dialogue/sound-bed/mix/master chain and a proof report for each stage

### Task 4.1: Audio Stage Order and Handoff Spec
- **Location**: New Paperclip issue under `HOL-37`
- **Description**: Define the canonical order for dialogue, lip-sync, score, ambience, SFX, mix, and final mastering
- **Complexity**: 5/10
- **Dependencies**: Existing ACE-Step and lip-sync lanes
- **Acceptance Criteria**:
  - A single scene can run through the whole audio stack without ambiguity
- **Validation**:
  - One end-to-end audio-finished scene proof

### Task 4.2: Mix and Master QC
- **Location**: New Paperclip issue under `HOL-37`
- **Description**: Add loudness, clarity, ducking, and sync checks so audio quality is reviewed systematically
- **Complexity**: 6/10
- **Dependencies**: Audio stage order
- **Acceptance Criteria**:
  - Audio failures are described as objective QC misses, not taste-only complaints
- **Validation**:
  - QC checklist attached to finished scene

## Sprint 5: Studio Operations Layer
**Goal**: Add the human/studio management layer required for reliable production.
**Related Epics**: `HOL-38`
**Demo/Validation**:
- Show one production cycle with asset tracking, review gates, explicit approvals, ownership, and schedule state

### Task 5.1: Asset and Review Ledger
- **Location**: New Paperclip issue under `HOL-38`
- **Description**: Normalize how scenes, shots, reference packs, outputs, review notes, and approvals are linked
- **Complexity**: 6/10
- **Dependencies**: Repeatable scene execution
- **Acceptance Criteria**:
  - No important artifact lives only in chat history
  - Review state is queryable from Paperclip
- **Validation**:
  - One scene package with linked assets and review history

### Task 5.2: Staffing and Scheduling Model
- **Location**: New Paperclip issue under `HOL-38`
- **Description**: Define which work belongs to lead, worker, or local-board/manual takeover and how queued work is prioritized
- **Complexity**: 5/10
- **Dependencies**: Asset/review ledger
- **Acceptance Criteria**:
  - Ownership and handoff rules are explicit
  - Queue sequencing is visible and auditable
- **Validation**:
  - One multi-issue production cycle using the model

## Sprint 6: Predictable Quality Under Pressure
**Goal**: Prove the system can sustain acceptable results under multiple queued scenes and retries.
**Related Epics**: `HOL-39`
**Demo/Validation**:
- Run a small batch of scenes and show stable completion, consistent proof reporting, and acceptable QC hit rate

### Task 6.1: Define Studio SLOs
- **Location**: New Paperclip issue under `HOL-39`
- **Description**: Set measurable thresholds for completion rate, retry rate, average render time, QC pass rate, and manual intervention count
- **Complexity**: 5/10
- **Dependencies**: Prior sprints
- **Acceptance Criteria**:
  - Quality claims are measurable
  - “Autonomous” has operational meaning
- **Validation**:
  - SLO table attached to batch run

### Task 6.2: Batch Stress Proof
- **Location**: New Paperclip issue under `HOL-39`
- **Description**: Queue multiple scene jobs and measure whether the control plane and proof ledger stay coherent without constant babysitting
- **Complexity**: 8/10
- **Dependencies**: SLOs and all prior sprints
- **Acceptance Criteria**:
  - Batch completes with bounded manual intervention
  - Failures are isolated and recoverable
- **Validation**:
  - Batch report with pass/fail against SLOs

## Dependency Order
1. `HOL-33` repeatability
2. `HOL-34` actor continuity + `HOL-35` location continuity
3. `HOL-36` editorial intelligence
4. `HOL-37` integrated audio post
5. `HOL-38` studio operations
6. `HOL-39` quality under pressure

## Testing Strategy
- Every sprint ends with a real media artifact, not documentation-only output
- Every artifact gets a proof report in Paperclip
- Every scene gets both technical checks (`ffprobe`, render completion, file paths) and QC judgment
- Higher-order sprints add scorecards and SLOs rather than replacing proof reports

## Potential Risks
- Scene 01 success may not transfer cleanly to Scene 02
  - Mitigation: treat `HOL-45` as the hard gate before expanding roadmap claims
- Wan/LTX lanes may be too weak for some continuity or motion targets
  - Mitigation: explicitly distinguish pipeline limitation from prompt-tuning miss
- Editorial and audio “quality” can drift into vague taste debates
  - Mitigation: define beat sheets, scorecards, and measurable QC rules
- Paperclip control-plane bugs could reappear under batch load
  - Mitigation: keep proof reporting and run-state checks mandatory in every sprint

## Rollback Plan
- If a later sprint destabilizes the process, fall back to the last proven scene-production contract:
  - shot-package YAML
  - single-scene proof flow
  - proof-first QC gate
- Keep Scene 01 as the canonical baseline artifact until a better baseline is explicitly promoted

## Immediate Next Move
Start `HOL-45`: repeat the Scene 01 process on a second scene. That is the shortest path from “working proto-studio” to “repeatable studio system.”
