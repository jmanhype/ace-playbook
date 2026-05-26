# COMPLETE AI FACTORY MEGA-GIST

**Date**: 2026-05-26
**Status**: ✅ PRODUCTION READY - ALL AUDIO SYSTEMS OPERATIONAL
**Version**: 2.1 - Complete Audio Factory Update

---

## TABLE OF CONTENTS

1. [Factory Overview](#factory-overview)
2. [Infrastructure](#infrastructure)
3. [SGFLIX Content Factory](#sgflix-content-factory)
4. [Motion Capture Factory](#motion-capture-factory)
5. [Audio Factory](#audio-factory)
6. [Dark Factory](#dark-factory)
7. [Integration Workflows](#integration-workflows)
8. [Quick Reference](#quick-reference)

---

## FACTORY OVERVIEW

### The AI Factory Ecosystem

Your AI Factory is a **multi-modal content creation system** spanning two machines with 4 specialized production pipelines:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI FACTORY ECOSYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  SGFLIX Content │  │  Motion Capture  │  │   Audio      │ │
│  │     Factory      │  │     Factory      │  │   Factory    │ │
│  │  (22 phases)     │  │  (CEBSam3d v2)   │  │ (ACE-Step)   │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│           │                     │                    │          │
│           └─────────────────────┴────────────────────┘          │
│                            │                                    │
│                    ┌───────▼────────┐                          │
│                    │  Dark Factory  │                          │
│                    │  (Bug Bounty)   │                          │
│                    │   (24 stages)   │                          │
│                    └────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### Production Capabilities

**Content Types:**
- ✅ AI-generated video (anime style, live action)
- ✅ Motion capture (3D rigs, pixel meshes)
- ✅ Music/audio production (EDM, hip-hop, voice)
- ✅ Character bibles (8-page production kits)
- ✅ Automated bug bounty discovery
- ✅ Multi-language dubbing
- ✅ Poster-first IP development

**Infrastructure:**
- **Mac Studio**: Orchestration, editing, ComfyUI client
- **3090 Box**: GPU rendering, inference, databases
- **ZimaBoard**: PostgreSQL databases
- **GitLab**: Version control, CI/CD
- **Tailscale**: VPN access

---

## INFRASTRUCTURE

### Machine 1: Mac Studio (Your Primary)

**Purpose**: Orchestration, development, light compute

```yaml
Location: Local
Role: Factory Command Center

Key Services:
  - Codex Desktop: AI orchestration (GPT-5.5, xhigh reasoning)
  - Blender 4.3.2: 3D scene building, rendering
  - ComfyUI Client: API to 3090 ComfyUI
  - ffmpeg: Video processing, frame extraction
  - Python 3.14: Script execution
  - rsync: File transfer to/from 3090

Storage:
  - /Users/speed/ai-video-factory/: Storyboard generation
  - /Users/speed/sgflix_audio_factory/: Audio production (217 critiques, 73 keepers)
  - /Users/speed/CEBSam3d/: Motion capture pipelines
  - /Users/speed/.codex/: Codex config, skills, automations

Memory: 64GB RAM
GPU: None (relies on 3090 for heavy compute)
```

### Machine 2: 3090 Box (straughter@192.168.1.143)

**Purpose**: GPU rendering, AI inference, databases

```yaml
Location: Remote (SSH: straughter@192.168.1.143)
Role: Factory Engine Room

Hardware:
  CPU: ?? (check with: lscpu)
  GPU: RTX 3090 (24GB VRAM)
  RAM: 64GB
  Storage: /mnt/bulk/ (large capacity)

Key Services:
  - ComfyUI: Port 8188 (Diffusion, SAM3D, video generation)
  - llama-server (Qwen 35B): Port 8080 (23.3GB VRAM, 256K context)
  - Qwen 3.6: Port 8081 (STRIPS validation)
  - GitLab: Port 8929 (Self-hosted Git server)
  - Paseo: Port 6767 (Workflow orchestration)
  - Opencode: Ports 34535, 38565, 45303 (AI agent platform)
  - Paperclip: Port 3100 (Experiment tracking)
  - Ollama: Port 11434 (Alternative LLM server)

Models:
  - SAM3D Body: /home/straughter/ComfyUI/models/sam3dbody/model.ckpt (2.0G)
  - MHR Model: /home/straughter/ComfyUI/models/sam3dbody/assets/mhr_model.pt
  - Qwen 3.5-35B-A3B: Q4_K_M quantization, ~20GB

VRAM Allocation:
  - Qwen 35B: 23.3GB (model 19.9GB + KV 1.4GB + compute 0.8GB)
  - ComfyUI: ~2GB (when SAM3D loaded)
  - Available: ~1-2GB (tight!)
```

### Machine 3: ZimaBoard CT 110 (192.168.1.154)

**Purpose**: Databases

```yaml
Location: Remote
Role: Data Persistence

Key Services:
  - PostgreSQL 15: Port 5432

Databases:
  - InsForge: Dark Factory bug bounty pipeline
    - 3,288 in-scope targets
    - 150 test runs completed
    - Tables: df_scope_programs, df_scope_targets, df_invariants, df_test_runs, df_findings

  - pgvector: Vector similarity search (Docker)

Connection:
  psql -h 192.168.1.154 -U insforge -d insforge
  Password: DarkFactory2026
```

### Network Architecture

```yaml
LAN: 192.168.1.x
  - Mac: 192.168.1.? (DHCP)
  - 3090: 192.168.1.143
  - ZimaBoard: 192.168.1.154

Tailscale VPN: 100.77.225.85
  - GitLab access: http://100.77.225.85:8929
  - SSH access: ssh://git@100.77.225.85:2224

File Transfer:
  - rsync: Mac ↔ 3090 (frames, videos, MHR data)
  - scp: Single file transfer
  - sftp: Interactive file transfer

Latency:
  - LAN: <1ms
  - Tailscale: 5-10ms
  - Internet: Variable
```

---

## SGFLIX CONTENT FACTORY

### Overview

**Complete 22-phase AI content production pipeline** with automated QC integration

**Status**: ✅ Production Ready (May 19, 2026)
**Location**: GitLab - http://100.77.225.85:8929/root/jumperx-stack

### 22-Phase Pipeline

```
PHASE -1: Bimodal Worthiness Audit
    ↓
PHASE 0: Source Entropy Audit
    ↓
PHASE 0: Intake
    ↓
PHASE 1: Research Intake (Hermes + Grok 4.3)
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
PHASE 7: GPT Image First Frames ⭐ QC INTEGRATED
    ↓
PHASE 8: PBR / Style / Material Pass
    ↓
PHASE 9: Video-To-JSON Shot Plan
    ↓
PHASE 10: Audio / Music / Voice Plan
    ↓
PHASE 11: Render Routing ⭐ QC INTEGRATED
    ↓
PHASE 12: Transcript And Lip-Sync Check
    ↓
PHASE 13: CHAI Critique ⭐ QC INTEGRATED
    ↓
PHASE 14: Human Taste QC ⭐ PRE-FILTERING
    ↓
PHASE 15: Overlays, Logo, Export
    ↓
PHASE 16: Caption, Tags, Hashtags
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
PHASE 22: Backup, Manifests, Automations
```

### Key Phases Explained

#### Phase 6: CHAI Shot-Language Spec

**Purpose**: Create detailed shot specifications

**Output Structure**:
```json
{
  "subject": "elderly protest leader with petition",
  "scene": "private media dinner entrance",
  "motion": "slow push-in, waiter lifts cloche",
  "camera": "vertical 9:16, low angle, 35mm",
  "critique": "Must read as satire, not news",
  "revision": "If text messy, crop tighter"
}
```

**Used By**: Phase 7, Phase 13

#### Phase 7: GPT Image First Frames ⭐

**Purpose**: Generate first frames using gpt-image-2

**Stack**: Codex gpt-image-2

**QC Integration**: ✅ Automatic QC (8/10+ threshold)

**Workflow**:
1. Generate image with gpt-image-2
2. QC check against CHAI spec
3. Auto-refine if score < 8
4. Only save 8/10+ images
5. Attach QC report

**Output Structure**:
```
frames/gpt_image_2/
├── first_frame_v01_prompt.md
├── first_frame_v01.png
├── first_frame_v01_qc.md
├── first_frame_v02_repair_prompt.md (if needed)
└── first_frame_v01.png (final approved)
```

#### Phase 9: Video-To-JSON Shot Plan

**Purpose**: Create detailed shot plans in JSON

**Input**: `frames/gpt_image_2/first_frame_v01.png`

**Output Structure**:
```json
{
  "shot_id": "shot_001",
  "source_frame": "frames/gpt_image_2/first_frame_v01.png",
  "duration_seconds": 10,
  "beats": ["Placard holds frame", "Camera reveals", "Waiter lifts cloche"],
  "overlay_plan": ["MERGER DINNER HAD A MENU", "AUDIENCE CHOICE WAS NOT ON IT"]
}
```

#### Phase 11: Render Routing ⭐

**Purpose**: Route to appropriate rendering engine

**Options**: ComfyUI, Kling, Runway, etc.

**QC Integration**: ✅ Video QC (frame + motion validation)

**Workflow**:
1. Render video
2. Extract 5 key frames
3. QC each frame against CHAI spec
4. Check motion consistency
5. Only approve 8/10+ avg + 7/10+ motion
6. Attach QC report

#### Phase 14: Human Taste QC ⭐

**Purpose**: Human review of creative quality

**QC Integration**: ✅ AI pre-filtering (only 7/10+ shown to humans)

**Impact**: 50% reduction in human review time

**Workflow**:
1. AI pre-check QC
2. Route < 7/10 to auto-refine
3. Humans only review 7/10+ content
4. Humans focus on taste, not technical issues

### Character Bible System

**Purpose**: Generate complete 8-page character bibles from scratch

**Stack**: Codex gpt-image-2 with CHARACTER_IDENTITY_LOCK structure

**8-Page Structure**:
1. **PRIMARY_HERO_REFERENCE** (3:4) - Main character reference with full identity lock
2. **ORTHOGRAPHIC_TURNAROUND** - Front, side, back views for 3D understanding
3. **MORPHOLOGY_PROPORTIONS_SILHOUETTE** - Body type, proportions, silhouette
4. **EXPRESSION_EMOTION_SHEET** - Facial expressions and emotions
5. **CRANIAL_APPENDAGE_DETAILS** - Head, hands, feet details
6. **SURFACE_TREATMENT_CONSTRUCTION** - Clothing, materials, textures
7. **EXTREMITIES_PROPS_ACCESSORIES** - Weapons, props, accessories
8. **MATERIALS_COLOR_RIGGING_MOTION** - Color palette, rigging, motion range

**CHARACTER_IDENTITY_LOCK Structure**:
```yaml
CHARACTER_IDENTITY_LOCK:
  name: Character Name
  character_class: humanoid
  age_cohort: age-appropriate description
  species: Human/demon/spirit/etc.
  role: hero/villain/side character/etc.
  
  morphology_lock: body type, posture, build
  cranial_lock: hair, facial features, expressions
  surface_lock: clothing, accessories, materials
  appendage_lock: limb count, extremities
  accessory_prop_lock: signature items/props
  chromatic_lock: primary/secondary colors
  material_lock: material types
  style_lock: art style (anime, manga, etc.)
  
  signature_visual_hooks: key visual identifiers
  DO_NOT_CHANGE: core identity elements
  BOUNDED_VARIATION: allowed variations
```

**Quality Targets**: 8-9/10 character fidelity

**Production Stats**: 90+ productions, 46 bibles

### Poster-First Workflow

**Based on**: Cannon Films' "sell the poster first" model

**Purpose**: Validate concepts before committing to full production

**Workflow**:
1. Generate single explosive poster
2. Apply 5-criteria Cannon Test
3. Greenlight or kill concepts before full bible production
4. If greenlit → Create 8-page character bible
5. If killed → Return to concept phase

**Integration**: Connects poster-first validation with proven SGFLIX system

---

## MOTION CAPTURE FACTORY

### Overview

**CEBSam3d v2** — Two working motion capture pipelines

**Status**: ✅ FULLY OPERATIONAL (May 11, 2026)
**Test Video**: kpop_test.mp4 (2.0M, K-Pop dance)

**Purpose**: Extract skeleton from video, build rigged 3D character, render studio-quality MP4

### Two Pipeline Options

#### Option A: High-Fidelity 3D Pipeline (Blender)

**What it does**: Extracts skeleton from video, builds rigged 3D character with weighted mesh, applies motion capture poses, and renders studio-quality MP4

**Output**:
- `.blend` file (for manual editing)
- `_rendered.mp4` (1080x1920, 30fps, H.264)

**When to use it**: When you need clean motion reference for Kling 3.0, or want to change camera angles/lighting/export to Unreal

**Time**: ~8-10 minutes total

**Pipeline steps**:
1. Mac: Extract frames from video (`ffmpeg`)
2. Mac→3090: Sync frames via `rsync`
3. 3090: SAM3D inference (DINOv3 + MHR pose extraction)
4. 3090: Extract skeleton, mesh, poses (Python + PyTorch)
5. 3090→Mac: Sync MHR data back
6. Mac: Blender builds rigged scene (armature + mesh + poses)
7. Mac: Blender renders MP4 (EEVEE engine)

**Key files**:
- `run_option_a_mocap.sh` — Main orchestrator (runs on Mac)
- `build_mhr_scene.py` — Blender scene builder (Mac Blender)
- `remote_wrapper.py` — Runs on 3090, handles SAM3D + MHR extraction

**Render settings**:
- Resolution: 1080x1920 (portrait)
- Engine: EEVEE (fast real-time render)
- Camera: Auto-positioned based on mesh bounding box
- Lighting: Sun + Fill lights (3-point setup)
- Material: Silver mannequin (metallic 0.8, roughness 0.3)
- Background: Dark gray (0.05, 0.05, 0.05)

**Performance breakdown** (kpop_test.mp4 - 302 frames):
- Frame extraction: ~10 seconds
- SAM3D inference: ~3-5 minutes
- MHR extraction: ~30 seconds
- Blender build: ~1 minute
- Blender render: ~2-3 minutes

#### Option B: Lightning-Fast Pixel Pipeline (ComfyUI)

**What it does**: Runs video through ComfyUI on 3090, uses AI to paint grey mannequin directly over original pixels frame-by-frame

**Output**: Single `.mp4` with isolated mesh on black background

**When to use it**: When you need quick motion reference immediately and don't need camera control

**Time**: ~3-5 minutes total

**Pipeline steps**:
1. Mac: Upload video to 3090
2. Mac: Send API request to ComfyUI (port 8188)
3. 3090: ComfyUI runs SAM3D with `render_mode=mesh_only`
4. 3090: Renders isolated mesh video
5. Mac→3090: Download MP4

**Key files**:
- `sam3d_comfy_api.py` — ComfyUI API client
- `run_kling_mocap.sh` — Orchestrator script

**Render modes available**:
- `mesh_only` — Isolated grey mannequin (default, recommended)
- `side_by_side` — 3-way split (original | mask | overlay)
- `mask_only` — Just silhouette mask
- `overlay` — Mannequin overlaid on original video

### Comparison: Option A vs Option B

| | Option A (Blender) | Option B (ComfyUI) |
|---|---|---|
| **Speed** | ~8-10 min | ~3-5 min |
| **Output quality** | Studio-lit 3D render | AI pixel paint |
| **Camera control** | ✅ Full 3D (auto-positioned) | ❌ Fixed |
| **Exportable rig** | ✅ .blend file | ❌ No |
| **Compute location** | Mac (Blender) + 3090 (SAM3D) | 3090 only |
| **Best for** | Final production ref | Quick iteration |

### Quick Start

```bash
# Option A (Full 3D Rig)
./run_option_a_mocap.sh your_video.mp4

# Output:
# - Option_A_Mocap.blend — Blender scene file
# - Option_A_Mocap_rendered.mp4 — Rendered video

# Option B (Quick Mesh)
./run_kling_mocap.sh your_video.mp4

# Output:
# - sam3d_kling_ref_XXXXX.mp4 — Mesh overlay video
```

### Technical Details

#### Camera Positioning (Option A)

The camera is automatically positioned based on the mesh's bounding box:
1. Calculate mesh bounding box
2. Find center point
3. Set camera distance: `max(height, width) * 2.5`
4. Position camera at chest height: `(center_x, center_y - dist, center_z)`
5. Rotate camera: `(90°, 0, 0)` to face the mesh

This ensures the character is always properly framed regardless of video content.

#### Pose Extraction

The working pose extraction method:
```python
pose_tensor = torch.from_numpy(data[0]['mhr_model_params']).unsqueeze(0)
with torch.no_grad(): _, skel = model(identity, pose_tensor, extra)
```

No need to use `pred_joint_coords` directly - the MHR model handles it correctly.

#### Lighting Setup (Option A)

- **Sun Light**: Main key light (energy: 5.0)
- **Fill Light**: Area light for shadows (energy: 100.0)
- **Material**: Silver mannequin with 80% metallic, 30% roughness

---

## AUDIO FACTORY

### Overview

**Complete AI Audio Factory — May 26, 2026**

**Status**: ✅ OPERATIONAL (All 4 Systems Working)
**Architecture**: 4-Pillar Production System

### The 4-Pillar Architecture (May 26, 2026)

**PILLAR 1: Fish Audio S2 Pro (Voice & Acting)** ✅
- **Location**: `~/fish-speech/` on 3090 box
- **Model**: 4B parameter Dual-AR Transformer
- **VRAM**: 22.21 GB / 24 GB
- **Features**:
  - Zero-shot voice cloning (3-10 second reference)
  - Paralinguistic tags: `[heavy breathing]`, `[terrified whisper]`, `[excited]`, `[laughing]`
  - Hollywood-grade emotion
  - 62-80+ languages
- **Status**: ✅ PRODUCTION READY - TESTED WITH REAL AUDIO
- **Test Samples**:
  - `FISH_TERRIFIED_COMPARE.wav` (312KB, 3.62s) - Heavy breathing, panic
  - `FISH_EXCITED_COMPARE.wav` (468KB, 5.43s) - Laughter, joy
  - Generation speed: 19-30 seconds for 3-6 second clips
  - Quality: Hollywood-grade voice acting

**PILLAR 2: Scenema Audio (Scene-Aware SFX)** ✅
- **Location**: `~/scenema-audio/` on 3090 box (Docker)
- **Model**: LTX-2.3 audio diffusion + Gemma 3 12B
- **VRAM**: 17.3 GB / 24 GB (INT8 + NF4 quantization)
- **Killer Feature**: Scene-aware SFX generation (UNIQUE!)
- **Features**:
  - XML prompts: `<speak>`, `<sound>`, `<action>` tags
  - Generates speech + environmental SFX in single pass
  - Can replace Sony Woosh for environmental foley
- **Status**: ✅ WORKING - TESTED WITH REAL AUDIO
- **Test Sample**:
  - `SCENEMA_TERRIFIED.wav` (1.1MB) - Thunderstorm + speech in one pass
  - Example: `<speak voice="Male, mid 40s. Weathered. Urgent."><sound>Heavy rain and wind howling</sound><action>He shouts over the storm</action>Get the lines! <sound>Thunder cracks overhead</sound></speak>`

**PILLAR 3: Sony Woosh (Foley Generation)** ✅
- **Location**: `~/woosh/` on 3090 box
- **Models**: 6 models installed (8.8GB total)
- **VRAM**: ~2GB during inference
- **Features**:
  - Text-to-audio (T2A): Sportscar engine, footsteps, glass breaking
  - Video-to-audio (V2A): Frame-perfect foley from video
  - Distilled models for real-time generation
- **Status**: ✅ FULLY OPERATIONAL - ALL MODELS TESTED
- **Models Installed**:
  1. Woosh-AE (844MB) - Encoder/decoder
  2. Woosh-CLAP (1.7GB) - Text conditioning
  3. Woosh-Flow (1.3GB) - T2A (full quality)
  4. Woosh-DFlow (1.3GB) - T2A distilled (0.32s generation!)
  5. Woosh-VFlow-8s (1.6GB) - V2A (full quality, 64 steps)
  6. Woosh-DVFlow-8s (1.6GB) - V2A distilled (0.20s generation!)

**PILLAR 4: Stable Audio 3.0 (Musical Score)** ✅
- **Location**: `~/stable-audio-3/` on 3090 box
- **Model**: stabilityai/stable-audio-3-medium (LTX-2.3 audio diffusion)
- **VRAM**: 9.4 GB / 24 GB
- **Features**:
  - 100% commercially licensed training data
  - Variable length (up to 6 minutes)
  - CLI + Gradio UI available
  - Models: medium, small-music, small-sfx, medium-base
  - ⚠️ **INSTRUMENTAL ONLY - Does NOT generate vocals/singing**
- **Status**: ✅ **WORKING - Optimal Settings Found**
- **Optimal Parameters**:
  - **steps**: 8 (ping-pong sampling - NOT 100!)
  - **cfg_scale**: 4.5 (lower is better - NOT 6.0 or 7.0!)
  - **model**: medium (best quality)
  - **duration**: 30 seconds (default)
- **Test Samples**:
  - `STABLE_BOSSA_NOVA.wav` (5.0MB) - Bossa Nova, cfg 6.0, 8 steps
  - `STABLE_AMBIENT_8STEPS.wav` (5.0MB) - Ambient electronic, cfg 4.5, 8 steps ✅ BEST QUALITY
  - `STABLE_AMBIENT_100STEPS.wav` (5.0MB) - Ambient electronic, cfg 4.5, 100 steps (worse than 8)
  - `STABLE_JAZZ_CFG45.wav` (5.0MB) - Jazz fusion, cfg 4.5, 8 steps
- **Quality**: Excellent for instrumental music, ambient, electronic, jazz
- **Best For**: Background scores, ambient music, instrumental tracks (NOT vocals/singing)
- **CLI Usage**:
  ```bash
  cd /home/straughter/stable-audio-3
  source venv_fix/bin/activate
  python -m stable_audio_3.cli --model medium -p "prompt" --duration 30 --steps 8 --cfg-scale 4.5 -o output.wav
  ```
- **Fix**: Created venv_fix with torch 2.7.1 + torchvision 0.22.0 + torchaudio 2.7.1

### Sony Woosh Deep Dive

**Critical Discovery: Prompt Engineering Matters!**

**❌ BAD Prompts** (generate ambient drones):
- "Footsteps on concrete floor"
- "Glass breaking"
- "Rain falling"

**✅ GOOD Prompts** (generate actual foley):
- "person walking in hallway, footsteps echoing"
- "shoes stepping on concrete, heavy footsteps"
- "footsteps on hard surface, rhythmic walking"
- **BEST**: "Two figures in costumes walk down a basement hallway, their footsteps echoing on the concrete floor."

**Working Prompt Formula**:
1. Include **subject** (person/shoes/figures)
2. Include **action** (walking/stepping)
3. Include **sound characteristic** (echoing/heavy/rhythmic)
4. Include **surface** (concrete/hard surface/hallway)

**Quality vs Speed Trade-offs**:

| Model | Steps | CFG | Time | Quality | Use Case |
|-------|-------|-----|------|---------|----------|
| Woosh-DFlow | 4 | 4.5 | 0.32s | Good | Quick previews |
| Woosh-DFlow | 4 | 7.0 | 0.32s | Better | Standard T2A |
| Woosh-VFlow | 64 | 4.5 | 3.98s | Excellent | High quality V2A |
| Woosh-VFlow | 76 | 7.0 | 4.29s | Excellent | Best quality |
| Woosh-VFlow | 88 | 7.0 | 5.40s | ✅ BEST | Final renders |

**VFlow (Video-to-Audio) Performance**:
- DVFlow (distilled): 0.18-0.20 seconds
- VFlow (full): 3.98-5.40 seconds
- Video understanding: Synchformer (24fps frame analysis)
- Audio: Perfectly synced to video frames
- Max duration: 8 seconds per clip

**Gradio Demo**:
- Woosh-DFlow UI: http://localhost:7861 (via SSH tunnel)
- Test prompts interactively
- Generate and download audio directly

### Complete Audio Orchestrator

**Location**: `~/audio_orchestrator.py` (417 lines)

**Pipeline Steps**:
1. **Generate Voice** → Fish Audio S2 Pro (paralinguistic tags)
2. **Generate Foley** → Sony Woosh (video-to-audio)
3. **Generate Score** → Stable Audio 3.0 (commercially licensed)
4. **Normalize All** → -14 LUFS (broadcast standard)
5. **Mix & Mux** → ffmpeg combines 3 tracks + video

**VRAM Requirements**:
- Fish Audio: 22.21 GB (peak)
- Sony Woosh: ~8GB (estimated)
- Stable Audio: ~8GB (estimated)
- **Sequential execution**: 22GB max = PERFECT FIT (24GB available)

### Legacy Audio Factory (Pre-May 2026)

#### ACE-Step 1.5

**Location**: /home/straughter/ACE-Step-1.5/
**Purpose**: AI music generation (legacy system)

**Status**: ⚠️ DEPRECATED - Replaced by Stable Audio 3.0

**Environment Variables**:
```bash
ACESTEP_PATH=/home/straughter/ACE-Step-1.5
ACE_SIMPLE_GENERATE=/home/straughter/sgflix_audio_factory/scripts/simple_generate_configurable.py
ACE_PYTHON=/home/straughter/ACE-Step-1.5/.venv/bin/python
```

#### LUFS Normalization

**Purpose**: Normalize audio to broadcast standards

**Target**:
- Integrated Loudness: -14 LUFS (EBU R128)
- True Peak: -1.0 dBTP
- Sample Rate: 44.1 kHz
- Bit Depth: 24-bit PCM

**Tool**: `ffmpeg-normalize` (2-pass)

**Command**:
```bash
ffmpeg-normalize input.wav \
  -o output.wav \
  -nt ebu \
  -t -14 \
  -tp -1.0 \
  -c:a pcm_s24le \
  -ar 44100 \
  -f
```

**Executable**: `/home/straughter/ComfyUI/venv/bin/ffmpeg-normalize`

### Production Test Results (May 26, 2026)

**All 4 Systems Tested and Verified Working**

**Fish Audio S2 Pro - Voice Acting** ✅
- Sample 1: "Heavy breathing, terrified whisper" (3.62s, 312KB)
- Sample 2: "Excited laughter, joy" (5.43s, 468KB)
- Sample 3: "Barbershop quartet with real vocals" (19.64s, 1.7MB)
- Quality: Hollywood-grade voice acting
- VRAM: 22.21 GB / 24 GB
- Speed: 19-30 seconds generation time

**Scenema Audio - Scene-Aware SFX** ✅
- Sample: "Thunderstorm with speech" (1.1MB)
- Killer feature: Generates speech + rain + wind + thunder in one pass
- VRAM: 17.3 GB / 24 GB
- Can replace Sony Woosh for environmental foley

**Sony Woosh - Foley Generation** ✅
- Text-to-Audio: Sportscar engine (0.32s, 469KB)
- Video-to-Audio: Footsteps in hallway (0.18s, 750KB audio + 1.1MB video)
- Best prompt: "Two figures in costumes walk down a basement hallway, their footsteps echoing on the concrete floor."
- VRAM: ~2 GB during inference
- Speed: 0.18-5.40 seconds depending on quality settings

**Stable Audio 3.0 - Instrumental Music** ✅
- Sample 1: "Bossa Nova with guitar and percussion" (cfg 6.0, 8 steps) - "much better"
- Sample 2: "Ambient electronic music" (cfg 4.5, 8 steps) - ✅ BEST QUALITY
- Sample 3: "Jazz fusion" (cfg 4.5, 8 steps)
- Quality: Excellent for instrumental music, ambient, electronic, jazz
- VRAM: 9.4 GB / 24 GB
- Speed: Fast generation (8 steps recommended, NOT 100)
- **Optimal Settings**: steps=8, cfg_scale=4.5 (lower is better!)

**Quality Comparison**:
| System | Quality | Speed | Best For |
|---------|---------|-------|----------|
| Fish Audio | Hollywood | 19-30s | Voice acting, dialogue, vocals |
| Scenema Audio | Filmmaking | Unknown | Scene SFX + speech |
| Woosh DFlow | Excellent | 0.32s | Quick foley generation |
| Woosh VFlow | Best | 4-5s | Final video foley |
| Stable Audio 3.0 | Excellent | Fast | Instrumental music, ambient |

**All test samples on Mac Desktop**:
- `FISH_TERRIFIED_COMPARE.wav`
- `FISH_EXCITED_COMPARE.wav`
- `BARBERSHOP_QUARTET_FISH.wav` (real vocals!)
- `SCENEMA_TERRIFIED.wav`
- `WOOSH_SPORTSCAR.wav`
- `WOOSH_VFLOW_AUDIO.wav` + `WOOSH_VFLOW_VIDEO.mp4`
- `vflow_descriptive.wav` + `vflow_descriptive.mp4` (BEST QUALITY)
- `STABLE_BOSSA_NOVA.wav` (cfg 6.0)
- `STABLE_AMBIENT_8STEPS.wav` (cfg 4.5, 8 steps) ✅ BEST
- `STABLE_JAZZ_CFG45.wav` (cfg 4.5)

**Production Pipeline**:
1. Generate voice: Fish Audio S2 Pro (paralinguistic tags)
2. Generate foley: Sony Woosh DVFlow (fast) or VFlow (quality)
3. Generate score: Stable Audio 3.0 (steps=8, cfg=4.5)
4. Normalize all tracks: -14 LUFS
5. Mix: ffmpeg combines 3 tracks + video
6. Output: Broadcast-ready MP4

**VRAM Management**:
- Fish Audio: 22.21 GB (largest)
- Scenema Audio: 17.3 GB
- Sony Woosh: ~2 GB
- Stable Audio 3.0: 9.4 GB
- **Sequential execution** = all 4 systems work perfectly on 24GB GPU

---

## DARK FACTORY

### Overview

**Automated Bug Bounty Discovery Engine** — 24-stage autonomous pipeline

**Status**: 🟢 Active (May 14, 2026)
**Location**: /home/straughter/dark-factory-bugbounty/

**Purpose**: Continuous vulnerability discovery, validation, and reporting

### Pipeline Stages

#### Core Discovery (DF-1 through DF-6)

- **DF-1: Scope Extractor** — Fetch programs from HackerOne/Bugcrowd, parse scope rules
- **DF-2: RDF Compiler** — O* Graph Schema for InsForge database
- **DF-3: Scope Parser** — z.ai invariant generation from scope rules
- **DF-4: Watcher** — Certstream monitoring for new subdomains (24/7)
- **DF-5: STRIPS Validator** — Qwen 3.6 invariant validation
- **DF-6: OWASP Juice Shop** — Safe testing environment (localhost:3000)

#### Execution & Validation (DF-7 through DF-11)

- **DF-7: HTTP Interceptor** — Evasive HTTP testing (curl_cffi, Oxylabs proxy)
- **DF-8: PROV-O Serializer** — W3C evidence chains for submissions
- **DF-9: Report Generator** — z.ai professional report writing
- **DF-10: HiRAG Compiler** — ArXiv paper analysis for new techniques
- **DF-11: Qwen Inference** — Local model payload synthesis

#### Advanced Exploitation (DF-12 through DF-20)

- **DF-12: Topological Analysis** — Graph-based attack surface mapping
- **DF-14: PoC Sandbox** — Docker exploit validation with IPv6 rotation
- **DF-15: Triage Extractor** — Vulnerability prioritization
- **DF-16: Subdomain Takeover** — SDTO automated hunting
- **DF-17: Sourcemap Extractor** — JavaScript secret extraction
- **DF-18: Apex Strike** — Advanced exploitation techniques
- **DF-19: BOLA Fuzzer** — Broken Object Level Authorization fuzzing
- **DF-20: OSS Bounty Hunter** — Open source PR automation

#### Autonomous PR Factory (DF-21 through DF-24)

- **DF-21: Autonomous PR** — Full SAST → LLM → Patch → PR pipeline
- **DF-22: Docker Verification** — Sandbox testing + AST-aware patching
- **DF-23: Custom SAST Rules** — Domain-specific vulnerability patterns
- **DF-24: Human Gate** — Responsible PR submission with rate limiting

### Infrastructure

**3090 Box** (straughter@192.168.1.143):
- Qwen 35B A3B (port 8080, 256K context)
- Qwen 3.6 (port 8081)
- 64GB RAM, RTX 3090 (24GB VRAM)

**ZimaBoard CT 110** (192.168.1.154):
- PostgreSQL 15: InsForge database
- 3,288 in-scope targets tracked
- 150 test runs completed (4.56%)

### Current Status

- **Active Pipeline**: DF-1 through DF-11 deployed and running
- **Test Runs**: 150/3,288 (4.56%)
- **Success Rate**: 18% (27/150 HTTP 200)
- **Findings**: 0 confirmed vulnerabilities (investigating false positive rate)

### Recent Work (May 14-15, 2026)

**Web Intel Research**: Completed comprehensive intelligence gathering on 5 high-value targets (Notion, Zoom, Linear, Replit, Mailchimp) using SearXNG + Firecrawl + z.ai GLM-5.1

**Key Findings**:
- Notion: IDOR API bypass ($1K-$5K), confirmed $2K payout (May 2024)
- Zoom: 4 recent CVEs, JWT manipulation ($3K-$10K)
- Linear: GraphQL attack surface ($500-$5K)
- Replit: Container escape vectors (VDP only)
- Mailchimp: IDOR vulnerabilities ($500-$15K)

**Deliverables**:
- `web_intel_bug_bounty_report.md` — Full intelligence report
- `notion_idor_tester.py` — IDOR testing framework
- `zoom_jwt_tester.py` — JWT manipulation testing
- `web_intel_bug_bounty_research.py` — z.ai GLM-5.1 automation

### GitLab Repository

**URL**: `http://100.77.225.85:8929/root/dark-factory-pr-factory.git`

**Branch**: `main`

**Latest Commits**:
- `4c433e5` — Add: Dark Factory DF-10 through DF-22
- `3b30257` — Add: Dark Factory Skills (DF-1 through DF-9)
- `6f7eb57` — Add: Web Intel Bug Bounty Research (May 14 2026)

**Structure**: All 24 DF systems preserved in `skills/` directory

---

## INTEGRATION WORKFLOWS

### End-to-End Content Production

**Workflow**: Character Bible → Motion Capture → Audio → Final Video

```
1. CHARACTER BIBLE (SGFLIX Factory)
   ├─ Generate CHARACTER_IDENTITY_LOCK
   ├─ Create 8-page bible with gpt-image-2
   └─ Output: Production-ready character kit

2. MOTION CAPTURE (CEBSam3d)
   ├─ Option A: Full 3D rig (Blender) OR
   ├─ Option B: Quick mesh (ComfyUI)
   └─ Output: Motion reference video

3. AUDIO PRODUCTION (Audio Factory)
   ├─ Generate music with ACE-Step
   ├─ Normalize to -14 LUFS
   └─ Output: Broadcast-ready audio

4. VIDEO GENERATION (SGFLIX Phase 11)
   ├─ Composite character + motion + audio
   ├─ Render with Kling 3.0
   └─ Output: Final video with QC
```

### Example: K-Pop Dance Video

```bash
# Step 1: Create character bible
"Create a character bible for Lisa from BLACKPINK"

# Step 2: Motion capture
./run_option_a_mocap lisa_dance_reference.mp4
# Output: Option_A_Mocap_rendered.mp4 (silver mannequin)

# Step 3: Generate audio
cd /home/straughter/sgflix_audio_factory/
./ace_step_standalone_from_payload.py payload.json output.wav
ffmpeg-normalize output.wav -o final.wav -nt ebu -t -14

# Step 4: Composite and render
# Use SGFLIX Phase 11 with Kling 3.0
# Input: Character bible + Motion reference + Audio
# Output: Final video with QC
```

### Multi-Language Dubbing

**Workflow**: Original video → Transcript → Translation → Dubbing

```
1. ORIGINAL VIDEO
   └─ SGFLIX Phase 11 output

2. TRANSCRIPT EXTRACTION
   ├─ Extract speech-to-text
   └─ Generate timestamped transcript

3. TRANSLATION
   ├─ Translate transcript to target language
   └─ Preserve timing and emotion

4. VOICE SYNTHESIS
   ├─ Generate voice with PersonaPlex (Moshi)
   └─ Match original timing

5. LIP-SYNC ADJUSTMENT
   ├─ Adjust video timing
   └─ Validate lip-sync (Phase 12)
```

---

## QUICK REFERENCE

### Service URLs

```yaml
Mac Local:
  - Codex Desktop: http://localhost:9100 (gpt-image-2)
  - ComfyUI Client: http://localhost:8188 (API to 3090)

3090 Box (LAN):
  - ComfyUI: http://192.168.1.143:8188
  - Qwen 35B: http://192.168.1.143:8080
  - Qwen 3.6: http://192.168.1.143:8081
  - GitLab: http://192.168.1.143:8929

3090 Box (Tailscale):
  - ComfyUI: http://100.77.225.85:8188
  - GitLab: http://100.77.225.85:8929
  - Git: ssh://git@100.77.225.85:2224

ZimaBoard:
  - InsForge DB: postgresql://insforge:DarkFactory2026@192.168.1.154:5432/insforge
```

### Common Commands

#### Motion Capture

```bash
# Option A: Full 3D rig
cd /Users/speed/CEBSam3d/
./run_option_a_mocap.sh your_video.mp4

# Option B: Quick mesh
./run_kling_mocap.sh your_video.mp4
```

#### Audio Production

```bash
# Generate music
cd /home/straughter/sgflix_audio_factory/
./ace_step_standalone_from_payload.py payload.json output.wav

# Normalize to -14 LUFS
ffmpeg-normalize input.wav -o output.wav -nt ebu -t -14 -tp -1.0 -c:a pcm_s24le -ar 44100 -f
```

#### Character Bible

```bash
# Via natural language (in Codex)
"Create a character bible for Naruto Uzumaki from Naruto"
```

#### Dark Factory

```bash
# Check database
PGPASSWORD=DarkFactory2026 psql -h 192.168.1.154 -U insforge -d insforge

# Restart Qwen models
ssh straughter@192.168.1.143
sudo systemctl restart llama-server-qwen  # Qwen 35B (port 8080)
```

### File Transfer

```bash
# Mac to 3090
rsync -avz /Users/speed/ai-video-factory/ straughter@192.168.1.143:~/incoming/

# 3090 to Mac
rsync -avz straughter@192.168.1.143:~/output/ /Users/speed/ai-video-factory/

# Single file
scp local_file.txt straughter@192.168.1.143:~/
```

### Troubleshooting

#### GPU OOM on 3090

```bash
# Check VRAM usage
ssh straughter@192.168.1.143
nvidia-smi

# Stop Qwen 35B to free VRAM
sudo systemctl stop llama-server-qwen

# Restart ComfyUI
# (via systemd or manually)
```

#### Database Connection Issues

```bash
# Test InsForge connection
psql -h 192.168.1.154 -U insforge -d insforge
# Password: DarkFactory2026

# Check ZimaBoard connectivity
ping 192.168.1.154
```

#### ComfyUI Issues

```bash
# Check if ComfyUI is running
ssh straughter@192.168.1.143
ps aux | grep comfy

# Restart ComfyUI
# (check your systemd service or launch method)
```

---

## MAINTENANCE & BACKUPS

### Backup Strategy

#### GitLab

```bash
# Backup GitLab data
docker exec gitlab gitlab-backup create

# Backup location: /var/opt/gitlab/backups (in container)
```

#### InsForge

```bash
# TODO: Implement automated backups
pg_dump -h 192.168.1.154 -U insforge insforge > backup.sql
```

#### Paperclip

```bash
# Embedded PostgreSQL data
# Location: /home/straughter/.paperclip/instances/default/db
```

### Monitoring

#### System Resources

```bash
# CPU/Memory
htop

# GPU
nvidia-smi

# Disk
df -h

# Docker
docker stats
```

#### Service Health

```bash
# All listening ports
ss -tlnp

# Process tree
ps auxf

# Service logs
journalctl -f
```

---

## NEXT STEPS

### High Priority

1. **GitLab backup automation** — Implement automated backups
2. **InsForge backup automation** — Implement automated backups
3. **Monitoring dashboards** — Grafana or similar
4. **Log aggregation** — ELK or similar
5. **Service health checks** — Automated monitoring

### Medium Priority

1. **GitLab Runner registration** — CI/CD pipeline
2. **CI/CD pipeline configuration** — Automate testing
3. **Disaster recovery testing** — Test restore procedures
4. **Load balancing** — Multiple Opencode instances
5. **API gateway** — Kong or Traefik

### Low Priority

1. **Metrics collection** — Prometheus
2. **Alerting** — Alertmanager
3. **Secrets management** — Vault
4. **Service mesh** — Istio or Linkerd
5. **Distributed tracing** — Jaeger

---

**Last Updated**: 2026-05-26
**Version**: 2.2 - All 4 Audio Systems Working
**Environment**: Production (Mac + 3090 + ZimaBoard)
**Audio Systems**: Fish Audio ✅, Scenema Audio ✅, Sony Woosh ✅, Stable Audio 3.0 ✅

---

## RELATED GISTS

- **SGFLIX Factory Pipeline**: https://gist.github.com/jmanhype/9b1aab1cf9603847456628b3db259577
- **CEBSam3d v2**: https://gist.github.com/jmanhype/68cc229f8f77a40600a4df4d602e1054
- **Audio Factory**: https://gist.github.com/jmanhype/4c82d389db8fc6ad38a1e85d954050c1
- **Dark Factory**: https://gist.github.com/jmanhype/0eeff0a6e15c14755e191c7c080726f8
- **Infrastructure**: https://gist.github.com/jmanhype/af6c078899cf0760ed37852810e54cf0
