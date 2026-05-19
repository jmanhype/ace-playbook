# SGFLIX AI Content Factory SOP

Version: 1.0  
Date: 2026-04-30  
Workspace: `/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image`

## Purpose

This SOP defines the full SGFLIX AI content factory: how one idea, song, video, event, product, or trend becomes finished short-form content, distribution assets, reusable templates, and Skool/course material.

The factory is three connected machines:

```text
Creation machine
Distribution machine
Education/product machine
```

The goal is not just to make one video. The goal is to create a repeatable system where every run leaves behind reusable assets, scoring data, prompts, JSON handoffs, QC notes, distribution learnings, and teachable workflows.

## Operating Principle

Every serious run must answer four questions:

1. What is the hook?
2. What are we making?
3. Where will it be rendered or handed off?
4. What reusable asset, lesson, or franchise does it leave behind?

The factory should always preserve two paths:

```text
Automatic/local path: Mac / 3090 / Paperclip / Comfy / audio stack
Manual/closed-tool path: copy-paste handoff JSON for Kling / Grok / other tools
```

If the 3090 is training, overloaded, or low on disk, do not force local rendering. Generate the handoff package and route the render elsewhere.

## Phase -1: Bimodal Worthiness Audit

Before Source Entropy Audit, decide whether the idea is worth compute at all.

The point is to avoid wasting the factory on culturally sterile material while still preserving two kinds of virality:

```text
Track A: Newsjack Vector - reactive, currently trending, high urgency
Track B: Archetype Resonance - proactive, evergreen, strong cultural stereotype
```

An idea only needs to pass one track.

### Required JSON

Every serious run should create:

```text
strategy/phase_minus_one_worthiness_audit.json
```

Required structure:

```json
{
  "phase_minus_one_audit": {
    "source_target": "short target name",
    "track_a_newsjack_velocity": {
      "active_trend_score": "1-10",
      "algorithmic_slipstream": "current attention level",
      "polarization_factor": "1-10",
      "track_a_total": "number",
      "track_a_verdict": "PASS | FAIL"
    },
    "track_b_archetype_resonance": {
      "iconography_strength": "1-10",
      "stereotype_rigidity": "Low | Medium | High",
      "subversion_potential": "1-10",
      "track_b_total": "number",
      "track_b_verdict": "PASS | FAIL"
    },
    "system_verdict": {
      "final_decision": "PROCEED_TO_ENTROPY_AUDIT | DISCARD | HOLD",
      "primary_vector": "TRACK_A_NEWSJACK | TRACK_B_EVERGREEN_ARCHETYPE",
      "urgency_class": "Max | High | Medium | Low",
      "strategic_directive": "what the rest of the factory should optimize for"
    }
  }
}
```

### Track A: Newsjack Vector

Use when the world is already talking about it.

Examples:

- Ice Spice McDonald's conflict
- breaking celebrity beef
- live TV blunder
- fresh court/drama moment

This path optimizes for speed. Shelf life is usually 24-48 hours.

### Track B: Latent Archetype Vector

Use when the subject is not necessarily trending, but the cultural archetype is rigid and instantly understood.

Examples:

- Balenciaga Pope
- Kid Rock + Apache air-support estate
- DJ Khaled strongman myth
- Trump ballroom mythology

This path optimizes for visual subversion and can be evergreen.

### Discard Rule

If an idea fails both tracks, do not send it to the Humor Engine.

## Phase 0: Source Entropy Audit

Before Hook Engine, Humor Engine, CHAI, first frames, or video generation, audit the source material.

The point is to avoid **vector cancellation**: adding absurdity to a source that is already absurd can make the output feel loud, fake, and try-hard.

### Required JSON

Every serious run should create:

```text
strategy/source_entropy_audit.json
```

Required fields:

```json
{
  "source_entropy_audit": {
    "baseline_reality_check": "What is the intended context of this source?",
    "detected_anomalies": ["What already does not fit the baseline?"],
    "native_entropy_score": "1-10",
    "subject_self_awareness": "trying_to_be_funny | trying_to_look_cool | unaware | deadpan | staged",
    "comedic_vector_recommendation": "blank_canvas | native_absurdity | cringe_vector",
    "recommended_strategy": "inject_high_anomaly | straight_man_framing | micro_spotlight"
  }
}
```

### Vector Balance Rules

**Low entropy, 1-3: Blank Canvas**

The source is mundane or flat. Inject a high anomaly.

Example:

`boring paparazzi estate sighting -> Apache air-support reveal`

**High entropy, 7-10: Native Absurdity**

The source is already weird. Do not add more chaos. Use straight-man framing.

Example:

`wild source clip -> sterile HR/corporate/legal overlay`

**Mid entropy, 4-6: Cringe Vector**

The source is trying to look cool or serious but has a weak point. Do not add new objects. Micro-spotlight the failure.

Example:

`try-hard celebrity pose -> dry caption pointing at the exact pose`

### Kill Rule

If the source is high-entropy and the planned intervention is also high-entropy, stop. Reframe before generating.

## Current System Map

### Operational Now

- PaperclipAI local server on the 3090
- Hollywood Studio Paperclip board
- Web Intel Stack Paperclip board
- Media Sourcing Paperclip board
- Taste Engine Paperclip board
- Kling Hollywood Studio Paperclip board
- ComfyUI / Wan / LTX / Facefusion / LatentSync stack
- ACE-Step music generation
- OmniVoice / VoxCPM2 / Qwen3-TTS / Chatterbox voice stack
- `video-to-json-i2v` repo
- TRiBE files and quality scoring files
- Media sourcing / stem / transcription / dataset pipeline
- SGFLIX run folders and master packages
- Codex skills for SGFLIX, render QC, and overlay compositing

### Partially Wired

- SGFLIX asset vault
- Hook Engine handoffs
- Closed-tool JSON handoffs
- CHAI-style shot specs
- DXFILMS swipe file
- Skill system expansion
- Distribution/insights database

### Blocked Or Not Finished

- 3090 local 3D camera-control rescue lane
- Wan 2.2 FP8 + SCAIL + VACE reference lane
- Comfy runtime model matrix/profiles
- Overnight queue/proof receipt automation
- Full Kling browser render/download/QC operating loop

## Standard Run Folder

Every run should use this structure:

```text
run_###_short_name/
├── research/
│   ├── last30days_report.md
│   ├── sources.json
│   └── source_notes.md
├── strategy/
│   ├── hook_engine_brief.md
│   ├── tribe_meta_score.json
│   ├── risk_taste_score.json
│   └── franchise_decision.md
├── frames/
│   ├── first_frames/
│   ├── keyframes/
│   ├── variant_first_frames/
│   └── video_to_json_anchors/
├── scene_json/
│   ├── shot_001.json
│   ├── shot_002.json
│   └── manifest.json
├── chai/
│   ├── shot_language_specs.json
│   ├── critique_notes.md
│   └── revision_plan.md
├── audio/
│   ├── source_audio/
│   ├── stems/
│   ├── lyrics_or_transcript/
│   ├── music_handoff.json
│   └── voice_lipsync_notes.md
├── handoffs/
│   ├── closed_tool_handoff.json
│   ├── kling_handoff.json
│   ├── grok_handoff.json
│   └── comfy_handoff.json
├── renders/
│   ├── raw/
│   ├── keepers/
│   └── rejected/
├── qc/
│   ├── transcript_check.md
│   ├── visual_qc.md
│   ├── taste_qc.md
│   └── final_verdict.md
├── captions/
│   ├── instagram_caption.md
│   ├── tiktok_caption.md
│   ├── youtube_shorts_caption.md
│   └── tags_hashtags.json
├── exports/
│   ├── instagram_ready/
│   ├── tiktok_ready/
│   └── youtube_shorts_ready/
├── insights/
│   ├── post_log.json
│   ├── performance_snapshot.md
│   └── remix_decision.md
└── skool/
    ├── case_study.md
    ├── prompt_pack.md
    ├── student_assignment.md
    └── sop_extract.md
```

## Phase 0: Intake

### Input Types

Accepted inputs:

- Real event or trend
- Music track
- Video reference
- Screenshot
- Product or affiliate offer
- Character/UGC persona
- Old SGFLIX post to remix
- Google Drive/course source material
- Random concept or joke

### Required Output

Create a run name and decide the run type:

```text
viral entertainment cluster
music video
UGC/ad creative
closed-tool handoff
Skool case study
franchise/series development
research-only
```

## Phase 1: Research Intake

Use Web Intel / last30days when the idea depends on recent culture, trends, public events, creator behavior, or platform practice.

### Required Actions

1. Gather recent public context.
2. Separate confirmed facts from allegations, jokes, reactions, and rumors.
3. Identify what people are actually reacting to.
4. Save a source log.
5. Write the grounded creative angle.

### Output

```text
research/last30days_report.md
research/sources.json
research/source_notes.md
```

### Rule

Never let generic AI imagination override the research. The joke must come from what people are actually talking about.

## Phase 2: DXFILMS Swipe File Classification

Compare the idea against the DXFILMS-style content patterns.

### Fields To Capture

```text
post link
celebrity/person/context
event
format
caption type
hashtags
tags
audio/song
cluster size
visual pattern
SGFLIX adaptation
```

### Common Formats

- Fake article/lore caption
- Live performance
- Music video
- Outtakes
- Fake interview
- Product/ad crossover
- Trailer
- Photo carousel
- Newsroom parody
- Reality TV meltdown
- Documentary recap

### Output

```text
strategy/swipe_classification.md
```

## Phase 3: Hook Engine

The Hook Engine turns the research into a creative package.

### Required Hook Questions

1. What is the first-frame contradiction?
2. What is the meme?
3. What is the phrase people will repeat?
4. What is the scene, not the summary?
5. Can this become 3+ posts?
6. Can it become a franchise?

### Required Output

```text
strategy/hook_engine_brief.md
handoffs/closed_tool_handoff.json
```

### Rule

If the plan sounds like a narrator summarizing the event, reject it. The video must show a scene.

## Phase 4: TRiBE / Meta Creative Scoring

Judge content like creative testing, not like a normal script.

### Score These Layers

```text
concept
format
hook
body/story
caption/CTA
```

### Score 1-10

- Concept clarity
- Scroll-stop first frame
- Humor or emotional charge
- Recognizable context
- Share/comment potential
- Rewatch potential
- Format fit
- Series potential
- Asset reuse
- Platform risk
- Factory value

### Output

```text
strategy/tribe_meta_score.json
```

## Phase 5: Risk And Taste Gate

This is internal. Public captions do not need to say "fictional AI parody" unless the specific post needs it.

### Risk Checks

```text
defamation risk
real violence risk
platform risk
likeness risk
brand/logo risk
too fake-newsy risk
graphic content risk
misleading-news risk
```

### Taste Checks

Ask:

1. Does this feel human?
2. Is it a scene or a summary?
3. Is the joke clear in one second?
4. Would DXFILMS post something in this family?
5. Would people comment on the premise?
6. Is the voice/caption too AI?
7. Is the character doing something, or just explaining?

### Output

```text
strategy/risk_taste_score.json
qc/taste_qc.md
```

## Phase 6: CHAI Shot-Language Spec

CHAI is the required formal shot language layer.

Every serious shot JSON needs:

```text
Subject
Scene
Motion
Spatial
Camera
Critique
Revision
```

### CHAI Fields

Subject:
- Who/what is visible?
- What identity must stay stable?
- What facial expression or body language matters?

Scene:
- Where are we?
- What objects, props, and background details matter?
- What should not appear?

Motion:
- What moves?
- What stays still?
- What is the action beat by beat?

Spatial:
- Foreground/midground/background relationship
- Negative space
- Object positions
- Character blocking

Camera:
- Shot size
- Lens feel
- Camera movement
- Focus behavior
- Stability or handheld feel

Critique:
- What could go wrong?
- What would make this look AI/sloppy?
- What would break the joke?

Revision:
- Exact changes to fix the critique
- Stronger wording for the render prompt
- Clearer constraints

### Output

```text
chai/shot_language_specs.json
chai/critique_notes.md
chai/revision_plan.md
```

## Phase 7: GPT Image First Frames And Keyframes

Generate first frames and optional middle/end frames before rendering.

### Standing Rule

Every SGFLIX video run starts with a **GPT Image 2 first frame**.

That first frame is the visual source of truth for identity, composition, wardrobe, setting, lighting, and first-second joke clarity. The video tool should animate the approved GPT Image 2 frame; it should not invent the opening frame from text alone.

If the GPT Image 2 first frame does not land, revise the image before rendering video.

### Required Frames

For each major variation:

```text
first frame
middle frame if action changes
end frame if extension/render needs a target
```

For video-to-json style anchors:

```text
main
start
early
late
end
```

### Output

```text
frames/first_frames/
frames/keyframes/
frames/variant_first_frames/
frames/video_to_json_anchors/
```

### Rule

Do not rely on one first frame for an entire cluster if the variations are different scenes. Each variation should have its own starting image when it changes the premise.

## Phase 8: PBR / Style / Material Pass

PBR is not CHAI. PBR is the realism/material layer.

Use it to specify:

- Skin texture
- Fabric detail
- Metal/glass/plastic behavior
- Lighting response
- Wetness/smoke/dust
- Product realism
- Set dressing materials

### Output

```text
strategy/pbr_style_notes.md
```

## Phase 9: Video-To-JSON Shot Plan

Use video-to-json style when converting references or planning complex video generation.

### Required Shot JSON

Each `shot_###.json` should include:

- Source frame
- Reference video if applicable
- Duration
- Aspect ratio
- Prompt
- Negative prompt
- Timeline
- Camera
- Subject motion
- Background motion
- Dialogue or implied line
- Text safe space
- QC checklist
- Workflow spec
- CHAI block

### Output

```text
scene_json/shot_001.json
scene_json/manifest.json
```

## Phase 10: Audio / Music / Voice Plan

Audio is a first-class layer, not an afterthought.

### Default SGFLIX Run Rule

Every new SGFLIX entertainment run should create an audio lane by default.

That does **not** mean every post must become a full music video. It means every run must leave behind at least one usable audio artifact or a clear blocked report:

```text
audio concept
-> ACE-Step payload
-> generated short hook/song candidate when 3090 is available
-> proxy QC
-> keeper/reject decision
-> music-video timing brief if usable
```

Minimum audio outputs for every new run:

```text
audio/audio_concept.md
audio/ace_step_payload.json
audio/music_handoff.json
qc/audio_qc.md
```

If the 3090/ACE-Step lane is available, also create:

```text
audio/generated_candidates/
audio/audio_scorecard.json
audio/keeper_manifest.json
qc/hook_timing_qc.md
```

If audio generation cannot run, create:

```text
qc/AUDIO_GENERATION_BLOCKED_REPORT.md
```

The factory should not silently skip audio. It should either generate a short candidate, package an approved existing keeper, or explain exactly why audio was blocked.

### Required Audio Intelligence Order

For song cloning/remaking, do not let measurements lead the creative call. The order is:

```text
rights/ownership confirmation
-> producer taste gate
-> genre/subgenre call
-> arrangement and drop map
-> stem/transcript analysis
-> BPM/key/energy measurements
-> lyrics split by role
-> generator prompt and QC
```

The producer taste gate must identify drum language, vocal delivery, half-time versus double-time feel, and whether the audio is assembled from interview/sample + ducked song bed.

### Run 012 Audio Factory Pattern

The working local audio lane is now:

```text
ACE-Step generation
-> Demucs stem split
-> Whisper word timing
-> Librosa/ffmpeg measurements
-> proxy critic scorecard
-> payload mutation
-> keeper packaging
-> visual run handoff
```

This is the Tier-1 proxy factory. It does not pretend that a text model can "hear" soul. It uses deterministic audio facts to filter structural trash before human producer review.

Use this lane when:

- a song, clone, parody track, music bed, or audio hook is the core asset
- the user owns or has provided the source song
- ACE-Step output needs iteration
- a generated song needs QC before becoming a visual SGFLIX run
- cover mode needs to be tested against text-to-music/reference-text generation

The current working package is:

```text
audio_intelligence/run_012_tier1_proxy_factory/
```

The 3090 bulk workspace is:

```text
/mnt/bulk/home/straughter/sgflix_audio_factory
```

Required 3090 run pattern:

```text
short-lived ACE-Step subprocess
-> release VRAM
-> Demucs / Whisper / Librosa
-> deterministic or LM Studio critique
```

Do not keep ACE-Step, Demucs, Whisper, and a critic model loaded in one long-lived GPU process. The stable pattern is hard subprocess isolation.

### ACE-Step Lanes

Use both ACE-Step lanes. They solve different problems.

**Cover mode lane**

Cover mode works when we want a controlled repaint of an existing song while preserving structure, bounce, and performance feel. It should not be dismissed just because a max-strength cover can return something too close to the source.

The working pattern from Run 011 is a cover strength sweep:

```text
0.35 = looser repaint, more transformed, higher drift risk
0.50 = balanced repaint, usually the first serious review point
0.65 = closest/source-preserving repaint, useful when the target feel is already right
```

Run cover sweeps when the original track is strong and we want to keep its timing, groove, and general song identity while changing the generated result.

**Reference/text-to-music lane**

Reference/text-to-music is the companion lane when we need a new performance that follows our lyrics and style without simply replaying the source. This was the better route for the Run 009C-style fast trap parity test, especially when cover mode was too literal.

The factory should test both lanes when cloning/remaking a user-owned target track:

```text
cover sweep: 0.35 / 0.50 / 0.65
reference/text-to-music: structured lyrics + style + source reference
proxy QC: Demucs + Whisper + Librosa + keeper scorecard
human producer review: choose cover keeper, reference keeper, or next mutation
```

For trap/hip-hop parity:

- do not put the chorus first unless the target does
- explicitly control half-time versus double-time
- use the original structure map before writing lyrics
- watch for mid-song voice drift
- save every payload beside the MP3

### Audio Keeper Package

When an audio run produces a keeper, package it before using it visually:

```text
audio/<keeper>.mp3
audio/<keeper>_metrics.json
audio/<keeper>_critique.json
audio/audio_scorecard.json
audio/music_handoff.json
```

If the keeper becomes a visual run, copy the MP3 and scorecard into that SGFLIX run package and build the first-frame/CHAI/scene JSON around the audio drop timing.

### Audio Capabilities

- Song analysis
- Stem analysis
- Hook timing
- Transcript timing
- Lyric-safe references
- Voice planning
- Lip-sync checking
- OmniVoice / VoxCPM / Qwen3-TTS / Chatterbox routing
- ACE-Step music generation
- Demucs stem extraction
- Whisper transcription

### Required Output

```text
audio/music_handoff.json
audio/voice_lipsync_notes.md
qc/transcript_check.md
audio/role_split_report.md
audio/style_prompt_suno_1000.txt
audio/audio_scorecard.json
audio/ace_step_payload.json
audio/keeper_manifest.json
```

### Rule

If dialogue matters, script the dialogue for the full extension. Do not leave the renderer to invent unsynced filler.

If music cloning matters, first separate interview/dialogue from song lyrics. For trap and hip-hop, check whether a measured slow BPM is actually a half-time reading of a double-time grid.

If an audio keeper becomes the source of a visual concept, the run must include a music-video timing brief: intro, first drop, verse start, hook start, bridge, final hit, and any places where on-screen graphics should appear.

## Phase 11: Render Routing

Decide where the work goes.

### Routes

Mac:
- overlays
- ffmpeg edits
- caption packaging
- lightweight QC
- frame extraction

3090:
- ComfyUI
- Wan/LTX
- ACE-Step
- voice
- stem analysis
- transcription
- local model work

Closed tools:
- Kling
- Grok
- Seedance
- other manual tools

Paperclip:
- task tracking
- run ownership
- issue history
- project board
- proof reports

### 3090 Preflight

Before routing to 3090, check:

```text
GPU memory
disk space
active training
Comfy status
ACE-Step status
Paperclip status
running jobs
```

Important: the 3090 is a remote machine, not the local Mac workspace. Do not declare the 3090 unavailable because local `nvidia-smi` is missing or local `/mnt/bulk` is not mounted.

Use the remote preflight:

```bash
ssh 3090 'hostname; command -v nvidia-smi; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; test -d /mnt/bulk/home/straughter/sgflix_audio_factory && echo BULK_AUDIO_OK'
```

For SGFLIX audio generation, use the remote Run 012 route:

```bash
scp audio/ace_step_payload.json 3090:/mnt/bulk/home/straughter/sgflix_audio_factory/payloads/run_###_<slug>_payload_001.json
ssh 3090 'ACE_CONFIG_PATH=acestep-v15-xl-sft ACE_OFFLOAD_TO_CPU=false /home/straughter/ComfyUI/venv/bin/python /mnt/bulk/home/straughter/sgflix_audio_factory/scripts/auto_producer_loop.py --initial-payload /mnt/bulk/home/straughter/sgflix_audio_factory/payloads/run_###_<slug>_payload_001.json --iterations 1 --run-label run_###_<slug>'
```

Then copy back the generated MP3, metrics JSON, critique JSON, and manifest into the run package.

### Rule

If the 3090 is training, low on disk, or VRAM is occupied, do not start renders. Produce the handoff JSON instead.

## Phase 12: Transcript And Lip-Sync Check

After a video render, always inspect:

- What was actually said?
- Is it synced?
- Did the model skip words?
- Did it invent extra dialogue?
- Did the mouth form the key phrase?
- Does the audio land at the correct time?

### Output

```text
qc/transcript_check.md
```

### Pass/Fail

Pass if:

- Key phrase is understandable
- Lip sync is believable
- No unwanted extra line ruins the scene
- Dialogue supports the meme

Fail if:

- It becomes a generic summary
- Lip sync is visibly wrong
- A missing word breaks the hook
- The subject says something out of character

## Phase 13: CHAI Critique And Revision

Use CHAI again after rendering.

### Critique Questions

1. Did the subject stay stable?
2. Did the scene remain coherent?
3. Did the motion follow the plan?
4. Did foreground/background blocking work?
5. Did the camera behave as specified?
6. What was missed?
7. What exact prompt revision fixes it?

### Output

```text
chai/critique_notes.md
chai/revision_plan.md
```

## Phase 14: Human Taste QC

This is the anti-slop gate.

### Reject If

- It sounds like AI wrote a news summary.
- The character explains the whole event.
- The joke needs too much context.
- The text overlay covers the face.
- The video has no scene.
- The caption feels generic.
- The first second does not communicate the premise.

### Approve If

- The first frame stops the scroll.
- The premise is clear.
- The character feels alive.
- The scene is funny or tense without explanation.
- The caption adds lore.
- The asset can become another post, remix, or lesson.

### Output

```text
qc/final_verdict.md
```

## Phase 15: Overlays, Logo, And Export

Create overlay graphics only after the video itself passes taste QC.

### Overlay Rules

- Do not block faces.
- Do not cover the entire lower half unless intentionally making a title card.
- Use transparent PNG/webm-style assets where needed.
- Text must not be cut off.
- Effects should be slow enough to read.
- End card may fade to black with centered SGFLIX logo.

### Output

```text
exports/instagram_ready/
exports/tiktok_ready/
exports/youtube_shorts_ready/
```

## Phase 16: Caption, Tags, And Hashtags

Captions should follow the platform and brand.

### SGFLIX Instagram

Use DXFILMS-style lore captions when appropriate:

- longer story caption
- recognized names, not overly formal names
- tag related people
- simple hashtags

Example hashtag style:

```text
#ai #comedy #trending #memes #sgflix
```

### Do Not Default To

```text
This is fictional AI parody...
```

Use that only when internal risk says the post needs visible clarification.

### Output

```text
captions/instagram_caption.md
captions/tags_hashtags.json
```

## Phase 17: Distribution Surface Rules

### SGFLIX / Batman Osama Instagram

Purpose:
- public entertainment
- viral AI cinema
- DXFILMS-style clusters
- SGFLIX watermark/brand

Use for:
- celebrity/event memes
- music videos
- fake interviews
- skits
- trailers
- remix clusters

### AI Video Factory Facebook Group

Purpose:
- conversation
- behind the scenes
- creator education
- funnel into Skool

Use for:
- process posts
- tool breakdowns
- prompt screenshots
- before/after
- "what worked / what failed" posts

### AI Video Factory Skool

Purpose:
- structured lessons
- templates
- workflows
- paid product path

Use for:
- SOPs
- prompt packs
- JSON templates
- case studies
- assignments
- critique checklists

### UGC Profiles

Purpose:
- persona + offer matching
- affiliate/product consistency

Rule:
- Do not post SGFLIX chaos content unless it matches that persona and offer.
- If an Inuit herbal wellness persona promotes natural products, do not mix in random Ice Spice/McDonald's content.

### Google Drive / Research Archive

Purpose:
- source material
- later extraction
- remixing into original courses

Rule:
- Extract and rewrite into your own frameworks before turning into course content.

## Phase 18: Insights Log

After posting, save:

- post URL
- time posted
- account
- caption
- tags
- first-frame image
- video file
- views
- reach
- non-follower percentage
- likes
- comments
- saves
- shares
- follows
- watch/retention if available

### Output

```text
insights/post_log.json
insights/performance_snapshot.md
```

## Phase 19: Remix Decision Engine

After insights, choose one:

```text
kill
repost later
make shorter
make longer
make song
make fake interview
make sequel
make photo carousel
make behind-the-scenes breakdown
turn into Skool lesson only
turn into franchise
```

### Rule

Small-account posts need enough time and enough surface area before judgment. Do not kill a premise because one new-account Reel got low early reach.

## Phase 20: Franchise Decision

Ask:

1. Is this a one-off?
2. Is this a mini-series?
3. Does it fit an existing SGFLIX show?
4. Can this character/world/joke come back next week?
5. Did it create reusable assets?
6. Is the risk low enough to keep repeating?
7. Can it become a Skool lesson?

### Decision Matrix

```text
High virality + high reuse + low risk = franchise
High virality + low reuse = one-off cluster
Low virality + high learning value = Skool case study
High risk = reframe or kill
```

### Output

```text
strategy/franchise_decision.md
```

## Phase 21: Skool / Course Productization

Every public experiment should be turned into at least one education asset.

### Possible Products

- case study
- SOP
- prompt pack
- JSON template
- QC checklist
- student assignment
- before/after breakdown
- "mistakes we fixed" lesson

### Output

```text
skool/case_study.md
skool/prompt_pack.md
skool/student_assignment.md
skool/sop_extract.md
```

## Phase 22: Backup, Manifests, And Automations

Each run should include:

- run manifest
- asset manifest
- caption database entry
- post database entry
- insights log
- automation registry entry when used
- backup/snapshot

### Required Files

```text
RUN_MANIFEST.json
ASSET_MANIFEST.json
AUTOMATION_LOG.md
BACKUP_NOTES.md
```

## Daily Operating Loop

### Morning

1. Check posted content from yesterday.
2. Log insights.
3. Decide whether to remix, kill, or extend.
4. Run last30days/Web Intel if needed.
5. Pick one primary SGFLIX concept and one education/product asset.

### Production Block

1. Create Hook Engine brief.
2. Score with TRiBE/Meta logic.
3. Run risk/taste gate.
4. Create CHAI spec.
5. Generate first frame/keyframes.
6. Create shot JSON and closed-tool handoff.
7. Route render.
8. QC transcript, lip sync, visuals, and taste.
9. Export platform-ready version.

### Posting Block

1. Write long lore caption if SGFLIX.
2. Tag relevant public accounts when appropriate.
3. Use simple hashtags.
4. Post.
5. Save the post URL and initial metadata.

### Evening

1. Record performance snapshot.
2. Choose remix decision.
3. Create or update Skool case study.
4. Back up run artifacts.

## Required Artifact Checklist

Before a run is considered complete:

- [ ] Research saved
- [ ] Hook Engine brief saved
- [ ] TRiBE/Meta score saved
- [ ] Risk/taste score saved
- [ ] CHAI spec saved
- [ ] First frame/keyframes saved
- [ ] Shot JSON saved
- [ ] Closed-tool handoff saved
- [ ] Audio/voice plan saved when relevant
- [ ] Render saved
- [ ] Transcript/lip-sync checked
- [ ] Human taste QC saved
- [ ] Platform export saved
- [ ] Caption/tags saved
- [ ] Post logged
- [ ] Insights logged
- [ ] Remix decision saved
- [ ] Skool/productization artifact created
- [ ] Backup/manifest updated

## Non-Negotiable Quality Rules

1. The video must show a scene, not summarize a scene.
2. The first second must communicate the premise.
3. Dialogue must be intentional if lip sync matters.
4. Every serious shot needs CHAI structure.
5. Every serious run needs a closed-tool handoff.
6. Every posted asset needs an insights log.
7. Every useful run becomes a teaching asset.
8. Do not overload the 3090 when it is training or VRAM/disk constrained.
9. Do not mix UGC persona/offer accounts with unrelated SGFLIX chaos.
10. Do not let generic AI wording replace human taste.

## Quick Start: New SGFLIX Entertainment Run

```text
1. Create run folder
2. Research the event/topic
3. Classify against DXFILMS patterns
4. Write Hook Engine brief
5. Score with TRiBE/Meta logic
6. Run risk/taste gate
7. Write CHAI shot spec
8. Generate first frame/keyframes
9. Build shot_001.json and closed_tool_handoff.json
10. Render or hand off
11. QC transcript/lip sync/visual/taste
12. Export IG/TikTok/Shorts version
13. Write long SGFLIX caption
14. Post and log
15. Decide remix/franchise/course next step
```

## Quick Start: Music Or Song Run

```text
1. Import source audio
2. Confirm rights/ownership or permitted source use
3. Listen like a producer: genre, subgenre, drums, vocal delivery, arrangement, drop map
4. Split interview/dialogue from actual song lyric
5. Extract stems if needed with Demucs
6. Measure BPM/key/energy and Whisper timing with the 3090 proxy lane
7. Create or mutate ACE-Step/Suno/Udio prompts under the relevant character limits
8. Generate or QC audio with Run 012-style scorecard
9. Package keepers with MP3, payload, metrics, critique, and manifest
10. Identify visual premise from the best audio moment
11. Build Hook Engine music-video brief
12. Build CHAI shot plan
13. Generate first frames
14. Build scene JSONs and closed-tool handoff
15. Render or hand off manually
16. Check sync and hook impact
17. Export cluster: teaser, full, remix, caption package
18. Productize as Skool prompt/workflow case study
```

## Quick Start: UGC Persona / Offer Run

```text
1. Confirm persona
2. Confirm product/offer
3. Confirm brand fit
4. Research objections and audience
5. Write first-frame hook
6. Script short creator performance
7. Generate persona-consistent first frame
8. Render or hand off
9. QC authenticity and offer clarity
10. Post only on matching UGC profile
11. Log conversion/proxy performance
```

## Known System Paths

Local SGFLIX workspace:

```text
/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image
```

Run 003 master package:

```text
/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image/sgflix_runs/run_003_dana_white_gala/RUN_003_MASTER_PACKAGE
```

3090 Paperclip:

```text
ssh 3090
Paperclip local UI/API: 127.0.0.1:3100 on 3090
```

3090 video-to-json:

```text
/home/straughter/video-to-json-i2v
```

3090 Paperclip scene imports:

```text
/home/straughter/paperclip_factory_imports
```

3090 ComfyUI:

```text
/home/straughter/ComfyUI
```

## Summary

The factory is not one tool. It is a loop:

```text
Research
→ Hook
→ Score
→ CHAI
→ Frames
→ JSON
→ Audio
→ Render/handoff
→ QC
→ Post
→ Measure
→ Remix
→ Productize
→ Archive
```

The important rule: every run must leave behind something reusable.
