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
3. `../character_bible/ek_magnitude_hand_anchor.png` — the only creature image uploaded for Shots 3–4.
4. `../camera_tests/test_ektachrome.png` — Ektachrome stock response and analog gate texture only.

The full-body hero, turnaround, and detail sheets are design provenance only. Do not upload them to the video model for Shots 3–4.

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
- No shoulder, elbow, torso, head, dorsal line, tail, leg, or body silhouette may be inferred from the creature reference.
- No atomic breath, heroic pose, Tokyo harbor staging or repeated water-recession plot.
- The destroyer remains the principal scale marker.
- Film failure is physical analog damage, not digital glitch effects.

## Render handoff

Use each approved start/end pair with the corresponding entry in `render_handoff.json`. Upload only `@Magnitude_Hand` plus the Ektachrome stock reference for Shots 3–4. Reject any render that redesigns the destroyer, changes drydock geometry, invents a shoulder or body, introduces modern digital imagery, or turns the ending into a clean VFX shot.

The existing Shot 3 and Shot 4 still frames were generated before the isolated hand anchor existed. Treat them as composition studies and regenerate them with `@Magnitude_Hand` before final video rendering.
