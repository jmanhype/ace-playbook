# MAGNITUDE Episode 2 — Drydock Incident

**Status:** frame generation in progress  
**Date:** 2026-06-07  
**Format:** 1967 classified naval 16mm Kodak Ektachrome observation reel  
**Picture:** horizontal 4:3, target 1536×1152, 24 fps  
**Runtime:** four 15-second shots, 60 seconds total

## Visual premise

A routine drydock inspection escalates from impossible water flow and an unaided destroyer shift to the isolated `@Magnitude_Hand` rising beyond the dock wall. Unlike Episode 1, this episode never conditions the render on a full-body creature reference and never uses atomic breath. The incident destroys the film through floodlight halation, spray, heat refraction, emulsion burn and a splice tear.

## Reference hierarchy

1. `frames/MAG_EP02_001_start.png` — source of truth for drydock geography, destroyer, scaffolding, cranes, camera height and daylight.
2. Previous approved frame in the sequence — immediate continuity reference.
3. `../character_bible/magnitude_hand/view_01_palm.png` — primary clean palm identity reference.
4. `../character_bible/magnitude_hand/view_02_side.png` — side silhouette and articulation reference.
5. `../character_bible/magnitude_hand/view_03_wrist_detail.png` — wrist and scale-material reference.
6. `../character_bible/magnitude_hand/view_04_claw_detail.png` — approved claw-detail crop.
7. `../camera_tests/test_ektachrome.png` — Ektachrome stock response and analog gate texture only.

All four hand references are empty and contain no prop. The archived concrete-gripping draft, full-body hero, turnaround, and detail sheets are provenance only. Do not upload them to the video model for Shots 3–4.

## Frame-generation order

1. Shot 1 start — generate the master location anchor.
2. Shot 1 end — direct continuity edit of Shot 1 start.
3. Shot 2 start and end — preserve the master location while progressing the anomaly.
4. Shot 3 start and end — add only `@Magnitude_Hand` and a short forearm.
5. Shot 4 start and end — preserve hand-only isolation and destroy the film optically.

## Hard continuity rules

- Horizontal 4:3; never 9:16 or phone footage.
- The same destroyer, scaffolding, cranes and drydock geometry persist throughout.
- Shot 1 contains no anomaly.
- Shot 2 contains no visible kaiju.
- Shots 3–4 show `@Magnitude_Hand` and only a short cropped forearm.
- The hand has exactly four total digits: three forward fingers and one opposing inner thumb.
- The hand begins empty; it never grips a concrete block or any other prop.
- No shoulder, elbow, torso, head, dorsal line, tail, leg, or body silhouette may be inferred from the creature reference.
- No atomic breath, heroic pose, Tokyo harbor staging or repeated water-recession plot.
- The destroyer remains the principal scale marker.
- Film failure is physical analog damage, not digital glitch effects.

## Render handoff

Use each approved start/end pair with the corresponding entry in `render_handoff.json`. Upload the four clean `@Magnitude_Hand` references plus the Ektachrome stock reference for Shots 3–4. Never upload the rejected gripping-block draft. Reject any render that adds a prop to the hand, changes the four-digit anatomy, redesigns the destroyer, changes drydock geometry, invents a shoulder or body, introduces modern digital imagery, or turns the ending into a clean VFX shot.

The existing Shot 3 and Shot 4 still frames were generated before the isolated hand anchor existed. Treat them as composition studies and regenerate them with `@Magnitude_Hand` before final video rendering.
