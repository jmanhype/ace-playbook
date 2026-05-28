# MAGNITUDE Episode 1 — Production SOP

Version: 1.0  
Date: 2026-05-28  
Status: READY FOR FIRST FRAME GENERATION  
Project: **MAGNITUDE Episode 1 Last Known Footage**  
Episode: **1 — Last Known Footage**  
Observer: **PFC Tanaka**  
Format: **4 × 15s shots = 60s vertical recovered military reel**

---

## Quick Reference

| Parameter | Value |
|---|---|
| Aspect | 9:16 vertical |
| Resolution | 1080×1920 |
| Duration | 60 seconds |
| Shots | 4 × 15 seconds |
| Primary Render | Kling 3.0 with **@Magnitude** entity anchor |
| First Frames | Codex image generation using entity + Ektachrome references |
| Entity Refs | `../character_bible/ek_page_01_hero.png`, `ek_page_02_turnaround.png`, `ek_page_04_expressions.png`, `ek_page_05_details.png` |
| Film Stock Ref | `../camera_tests/test_ektachrome.png` |
| Audio | Fish S2 Pro radio texture + Sony Woosh pixel-follow foley + Stable Audio 3.0 score |
| End Card | `PFC TANAKAS CAMERA WAS RECOVERED 3 DAYS LATER 200 METERS FROM GROUND ZERO` |

---

## Required Package Files

```text
manifest.json
shot_001.json
shot_002.json
shot_003.json
shot_004.json
kling_handoff.json
audio_concept.md
MAGNITUDE_EP01_SOP.md
frames/
renders/
```

Recommended working outputs:

```text
frames/first_frames/MAG_EP01_001_the_harbor.png
frames/first_frames/MAG_EP01_002_the_water_recedes.png
frames/first_frames/MAG_EP01_003_the_emergence.png
frames/first_frames/MAG_EP01_004_the_last_frame.png
renders/raw/MAG_EP01_001_v01.mp4
renders/raw/MAG_EP01_002_v01.mp4
renders/raw/MAG_EP01_003_v01.mp4
renders/raw/MAG_EP01_004_v01.mp4
renders/stitched/MAGNITUDE_EP01_TANAKA_picture_lock.mp4
renders/audio_mix/MAGNITUDE_EP01_TANAKA_audio_mix.wav
renders/keepers/MAGNITUDE_EP01_TANAKA_final_1080x1920.mp4
```

---

## STEP 1 — Generate First Frames with Codex

Use Codex image generation for the first frame of each shot.

Reference images to attach/use:

1. **Entity primary:** `../character_bible/ek_page_01_hero.png`
2. **Film stock / camera look:** `../camera_tests/test_ektachrome.png`
3. Optional continuity pages: `../character_bible/ek_page_02_turnaround.png`, `../character_bible/ek_page_04_expressions.png`, `../character_bible/ek_page_05_details.png`

For each shot, paste the `first_frame_prompt` from `shot_00X.json`.

Rules:
- Generate vertical 9:16 compositions or 1080×1920-ready frames.
- Keep the result as a single recovered 16mm frame, not poster art.
- Enforce anti-tiling guardrails: no cloned windows, repeated crates, duplicated soldiers, mirror rows, or tiled water.
- Do **not** allow @Magnitude to appear before shot 003.
- Do **not** allow dorsal plate glow before shot 004.

---

## STEP 2 — QC First Frames

QC each first frame before upload.

Minimum gate:

- [ ] 9:16 vertical composition works at 1080×1920.
- [ ] Kodak Ektachrome / 1966 military archive look is visible.
- [ ] The harbor geography is coherent across shots.
- [ ] No modern vehicles, phones, digital signs, modern shipping containers, or non-period uniforms.
- [ ] Shot 001 = normalcy, no threat.
- [ ] Shot 002 = receding water, no visible monster.
- [ ] Shot 003 = @Magnitude visible, plates dark/non-glowing.
- [ ] Shot 004 = plates glowing blue, no beam in first frame yet.
- [ ] @Magnitude matches character bible pages 01/02/04/05.
- [ ] No AI tiling, duplicate soldiers, malformed cranes, or repeated buildings.

If a frame fails, revise the prompt and regenerate before video. The first frame is the source of truth.

---

## STEP 3 — Upload Entity to Kling

In Kling 3.0:

1. Create/open the project: **MAGNITUDE Episode 1 — Last Known Footage**.
2. Upload **@Magnitude** entity references:
   - `../character_bible/ek_page_01_hero.png`
   - `../character_bible/ek_page_02_turnaround.png`
   - `../character_bible/ek_page_04_expressions.png`
   - `../character_bible/ek_page_05_details.png`
3. Upload style/camera reference:
   - `../camera_tests/test_ektachrome.png`
4. Name the entity handle exactly: **@Magnitude**.
5. Confirm Kling preserves the textured hide, dorsal silhouette, cranial details, and massive scale.

---

## STEP 4 — Paste Parametric Syntax from SOTA GPT

Use `kling_handoff.json` for each shot. Paste the shot's `parametric_prompt` into Kling.

Required syntax pattern:

```text
[ENTITY:@Magnitude locked to character_bible pages 01,02,04,05]
[PERIOD_STOCK:16mm Kodak Ektachrome 1966 classified military observation reel]
[OBSERVER:PFC Tanaka disciplined military cameraman]
[ASPECT:9:16 vertical 1080x1920]
[CAMERA:<shot camera vector>]
[SHOT:<shot name / narrative phase / timestamp>]
<video prompt>
```

Also paste the shot-specific `negative_prompt`.

Critical continuity:
- Shot 001: locked tripod, normal harbor.
- Shot 002: controlled pan to water receding.
- Shot 003: @Magnitude emergence; dark plates only.
- Shot 004: blue dorsal charge; atomic breath; film burn.

---

## STEP 5 — Render in Kling 3.0

For each shot:

1. Upload corresponding first frame from `frames/first_frames/`.
2. Select **Kling 3.0 I2V**.
3. Set duration to **15 seconds**.
4. Set aspect to **9:16**.
5. Apply @Magnitude entity anchor where visible.
6. Paste parametric prompt + negative prompt from `kling_handoff.json`.
7. Render.
8. Save output to `renders/raw/MAG_EP01_00X_v01.mp4`.

Do not proceed to assembly until each shot passes its JSON `qc_checklist`.

---

## STEP 6 — Visual QC Rendered Shots

Per-shot QC:

- [ ] Does the shot fulfill its narrative phase?
- [ ] Is the camera behavior correct for PFC Tanaka?
- [ ] Does the image remain 1966 Ektachrome and not modern digital?
- [ ] Are there any tiled buildings, repeated soldiers, duplicate monsters, or warped signs?
- [ ] Is @Magnitude consistent and correctly scaled?
- [ ] Does shot 004 destroy the film image convincingly?

Reject and rerender if the tool invents extra monsters, makes the plates glow in shot 003, turns the image into glossy CGI, or makes Tanaka panic too early.

---

## STEP 7 — Stitch with FFmpeg

Create concat list:

```bash
cat > renders/stitched/concat.txt <<'EOF'
file '../raw/MAG_EP01_001_v01.mp4'
file '../raw/MAG_EP01_002_v01.mp4'
file '../raw/MAG_EP01_003_v01.mp4'
file '../raw/MAG_EP01_004_v01.mp4'
EOF
```

Stitch picture:

```bash
ffmpeg -y -f concat -safe 0 -i renders/stitched/concat.txt   -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24"   -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow   renders/stitched/MAGNITUDE_EP01_TANAKA_picture_lock.mp4
```

If the end card is not baked into shot 004, generate a 3-second card and append it after the 60s picture lock or replace the final 3 seconds of shot 004 depending on pacing.

---

## STEP 8 — Audio Orchestration on 3090

Use the plan in `audio_concept.md`.

### Stable Audio 3.0 Score
Generate a 60s dark military tension drone:

```text
dark military tension drone, 1960s classified archive horror, sub-bass pressure, bowed metal, distant siren smear, low industrial rumble, restrained catastrophe, no melody, no drums, no heroic theme, building from quiet harbor unease to overwhelming nuclear blast, silence before impact, damaged tape hiss, cinematic but documentary
```

### Fish S2 Pro Radio Texture
Generate/prepare degraded background military radio chatter:
- narrow-band
- low intelligibility
- buried behind harbor/siren layers
- no story dialogue
- no clean modern radio beeps

### Sony Woosh Foley
Use rendered pixels to drive foley timing:
- ropes and dock creaks
- water suction
- @Magnitude water cascade
- tripod shock hits
- dorsal charge air displacement
- film burn/static at impact

Store audio stems in `renders/audio_mix/`.

---

## STEP 9 — Final Mix

Suggested stem layout:

```text
A1 archive_hiss_projector.wav
A2 harbor_ambience.wav
A3 water_recede_suction.wav
A4 creature_low_frequency.wav
A5 sirens_radio_background.wav
A6 stable_audio_score.wav
A7 sony_woosh_pixel_foley.wav
A8 blast_film_burn_static.wav
```

Mix rules:
- No intelligible dialogue.
- Keep normalcy quiet.
- Let the low-frequency rise sell scale.
- Mandatory silence pocket immediately before atomic breath.
- Final blast may distort/clip tastefully as recovered evidence.
- End on black/static silence.

Mux final:

```bash
ffmpeg -y -i renders/stitched/MAGNITUDE_EP01_TANAKA_picture_lock.mp4   -i renders/audio_mix/MAGNITUDE_EP01_TANAKA_audio_mix.wav   -c:v copy -c:a aac -b:a 320k -shortest   renders/keepers/MAGNITUDE_EP01_TANAKA_final_1080x1920.mp4
```

---

## STEP 10 — Final Acceptance Gate

- [ ] 60 seconds total, 4 shots × 15s.
- [ ] 9:16 1080×1920 export.
- [ ] Narrative arc reads: NORMALCY → UNEASE → EMERGENCE → ATOMIC BREATH.
- [ ] @Magnitude consistency locked to character bible.
- [ ] Ektachrome classified military archive look consistent.
- [ ] PFC Tanaka camera behavior remains disciplined until final failure.
- [ ] No dialogue.
- [ ] Atomic breath destroys the image and sound.
- [ ] End card text exact:

```text
PFC TANAKAS CAMERA WAS RECOVERED 3 DAYS LATER 200 METERS FROM GROUND ZERO
```
