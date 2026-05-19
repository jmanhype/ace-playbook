from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "084"
SLUG = "banks_wig_trial_clock"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
MASTER = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


def write_text(rel: str, text: str) -> None:
    path = RUN_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_master_text(rel: str, text: str) -> None:
    path = MASTER / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(rel: str, data: object) -> None:
    path = RUN_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_master_json(rel: str, data: object) -> None:
    path = MASTER / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, fill: str, bold: bool = False, width: int | None = None) -> int:
    fnt = font(size, bold)
    x, y = xy
    lines = wrap(draw, text, fnt, width) if width else text.splitlines()
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(size * 1.22)
    return y


def first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 864), "#171512")
    d = ImageDraw.Draw(img)
    d.rectangle((58, 56, 1478, 808), fill="#f6efe3", outline="#111111", width=6)
    d.rectangle((88, 88, 1448, 188), fill="#111111")
    label(d, (126, 116), "WIG TRIAL CLOCK", 58, "#f6efe3", True)
    label(d, (1048, 132), "RUN 084", 30, "#ff4f79", True)

    d.rectangle((116, 248, 702, 720), fill="#fffaf1", outline="#111111", width=5)
    label(d, (154, 286), "DAMAGES TRIAL", 42, "#111111", True)
    d.rectangle((164, 372, 650, 462), fill="#fbcc4b", outline="#111111", width=4)
    label(d, (188, 398), "PROCEEDS WITH\nOR WITHOUT YOU", 27, "#111111", True)
    d.rectangle((208, 530, 596, 646), fill="#111111", outline="#111111", width=3)
    label(d, (232, 562), "EMPTY MICROPHONE\nON THE STAND", 26, "#f6efe3", True)

    d.rectangle((804, 248, 1372, 720), fill="#252a31", outline="#111111", width=5)
    label(d, (846, 286), "GREEN ROOM EVIDENCE", 40, "#f6efe3", True)
    d.ellipse((920, 398, 1254, 666), fill="#ff4f79", outline="#111111", width=6)
    label(d, (990, 470), "2:00\nLATE", 54, "#111111", True)
    d.rectangle((852, 602, 1328, 660), fill="#f6efe3", outline="#111111", width=3)
    label(d, (878, 616), "WIG STRAIGHTENER: STILL ON", 25, "#111111", True)
    d.line((702, 480, 804, 480), fill="#ff4f79", width=12)
    d.polygon([(804, 480), (760, 452), (760, 508)], fill="#ff4f79")
    label(d, (118, 760), "Satire target: court-clock theatrics and no-show process. Allegation-framed; no exact likeness, no attack on protected traits.", 26, "#111111", True, 1260)
    img.save(path)


def storyboard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1800, 1200), "#ebe2d2")
    d = ImageDraw.Draw(img)
    label(d, (54, 34), "RUN 084 SHARED CHOICES: BANKS WIG TRIAL CLOCK", 38, "#111111", True)
    label(d, (56, 82), "Director bible board: empty courtroom performance booth, legal-clock gag, original club-court hook.", 22, "#343434")
    for i, color in enumerate(["#111111", "#f6efe3", "#ff4f79", "#fbcc4b", "#2d6cdf", "#4bbf86", "#252a31"]):
        d.rectangle((60 + i * 78, 140, 120 + i * 78, 200), fill=color, outline="#111111")
    cards = [
        ("Character", "Fictional absent pop-rapper archetype implied by empty mic, wig stand, and green-room props; no exact face."),
        ("Hero Props", "Bench-trial clock, wig straightener, notice envelope, empty witness mic, metronome gavel."),
        ("Environment", "Half courtroom, half backstage vanity room. The judge bench doubles as a DJ booth with sober legal lighting."),
        ("Blocking", "Clerk starts the trial clock; spotlight hits empty mic; gavel/metronome keeps time; notice slides under green-room door."),
        ("Visual Rules", "No real court seals, no defamatory copy, no direct depiction of private parties, keep all claims source-framed."),
        ("Camera", "24mm wide first contradiction, 70mm insert on clock, top-down desk evidence shot, final 50mm empty-mic hero."),
        ("Audio", "136 BPM ballroom/club court chant: clock in, clock out, trial keeps moving. Gavel kicks and hair-iron hiss."),
        ("Production", "Generated text must stay large and clean: PROCEEDS WITH OR WITHOUT YOU, 2:00 LATE, ALLEGED/REPORTED."),
    ]
    for i, (title, body) in enumerate(cards):
        x = 60 + (i % 4) * 430
        y = 236 + (i // 4) * 218
        d.rectangle((x, y, x + 392, y + 184), fill="#fffaf3", outline="#111111", width=3)
        d.rectangle((x, y, x + 392, y + 42), fill="#111111")
        label(d, (x + 14, y + 10), title, 21, "#f6efe3", True)
        label(d, (x + 16, y + 60), body, 20, "#343434", False, 350)
    panels = [
        "Wide: courtroom bench opens onto a backstage vanity station.",
        "Insert: clock reads two hours late; gavel clicks like a metronome.",
        "Hook: empty mic gets sworn in while notice envelope slides in.",
        "Button: judge-DJ booth stamps WITHOUT YOU as bass drops.",
    ]
    for i, body in enumerate(panels):
        x = 60 + i * 430
        y = 730
        d.rectangle((x, y, x + 392, y + 318), fill="#202329", outline="#fbcc4b", width=4)
        label(d, (x + 18, y + 18), f"Panel {i + 1}", 25, "#ff4f79", True)
        label(d, (x + 18, y + 64), body, 23, "#f6efe3", False, 335)
    img.save(path)


def main() -> None:
    for rel in [
        "research",
        "strategy",
        "audio/generated_candidates",
        "qc",
        "chai",
        "scene_json",
        "handoffs",
        "captions",
        "distribution",
        "skool",
        "manifests",
        "frames/gpt_image_2",
        "storyboards/shared_choices",
    ]:
        (RUN_DIR / rel).mkdir(parents=True, exist_ok=True)
    MASTER.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "id": "rollingstone_banks_no_show_2026_04_22",
            "title": "Azealia Banks Skips Hearing in Ex-Manager 'Stalking' Case",
            "publisher": "Rolling Stone",
            "url": "https://au.rollingstone.com/music/music-news/azealia-banks-damages-trial-stalking-manager-jeff-kwatinetz-93964/",
            "published": "2026-04-22",
            "used_for": "Winner source: no-show hearing, May 5 damages trial, reported two-hour deposition wig-straightening detail.",
            "fact_status": "reported court/update; liability and allegations must stay source-framed",
        },
        {
            "id": "rollingstone_hendrix_heirs_2026_04_29",
            "title": "Jimi Hendrix Bandmates' Heirs Lose Royalties Fight Against Sony, Hendrix Estate",
            "publisher": "Rolling Stone",
            "url": "https://ca.rollingstone.com/music/music-news/jimi-hendrix-bandmates-heirs-lose-royalties-fight-sony/",
            "published": "2026-04-29",
            "used_for": "Candidate scan: royalties archive/court premise.",
            "fact_status": "reported court ruling",
        },
        {
            "id": "rollingstone_apollonia_settlement_2026_04_10",
            "title": "Apollonia 'Very Pleased' After Trademark War Settlement With Prince Estate",
            "publisher": "Rolling Stone",
            "url": "https://au.rollingstone.com/music/music-news/prince-estate-apollonia-trademark-settlement-paisley-park-93427/",
            "published": "2026-04-10",
            "used_for": "Candidate scan: trademark/name-control settlement.",
            "fact_status": "reported confidential settlement",
        },
        {
            "id": "rollingstone_chappell_security_2026_03_28",
            "title": "The Chappell Roan-Security Controversy Explained",
            "publisher": "Rolling Stone",
            "url": "https://au.rollingstone.com/music/music-features/chappell-roan-controversy-hotel-security-incident-explained-1235537530/",
            "published": "2026-03-28",
            "used_for": "Candidate scan: hotel/security boundary controversy.",
            "fact_status": "reported conflicting accounts",
        },
        {
            "id": "rollingstone_bad_bunny_fees_2026_03_24",
            "title": "Bad Bunny Seeks $465,000 Legal Bill Reimbursement After Winning Copyright Case",
            "publisher": "Rolling Stone",
            "url": "https://au.rollingstone.com/music/music-news/bad-bunny-copyright-case-legal-bill-reimbursement-92856/",
            "published": "2026-03-24",
            "used_for": "Candidate scan: legal bill/sample dispute.",
            "fact_status": "reported legal motion",
        },
    ]
    write_json("research/sources.json", {"generated_at": NOW, "sources": sources})
    write_text(
        "research/last30days_report.md",
        """
# Step 1: Research Intake

This cycle began with current web/source research on 2026-05-02. No premise was selected from nearby image files, old storyboards, prior handoffs, generated audio, or existing local assets.

## Scan Frame

The factory looked for music/celebrity legal stories from the last 30 days that still had a usable public hook, first-frame contradiction, and a clean audio lane.

## Candidate Notes

- Azealia Banks no-show/damages-trial update: Rolling Stone reported an April 22 final-status-hearing no-show ahead of a May 5 damages trial and included the earlier deposition detail that she said she was delayed while straightening a wig. Strong court-clock/performance-booth gag.
- Hendrix heirs royalty ruling: serious legacy/catalog story with clear source, but lower meme velocity and weaker first-frame humor.
- Apollonia/Prince estate trademark settlement: strong name-control theme, but the story resolved amicably and the tone wants respect more than roast.
- Chappell Roan hotel-security controversy: high online heat, but child/fan-conflict taste risk and too many disputed accounts.
- Bad Bunny legal-fee motion: famous and clean, but older and less visually strange than the no-show clock.

## Winner

Selected premise: **Banks Wig Trial Clock**.

Fictionalized scene: a courtroom becomes a backstage green room. The trial clock keeps moving, an empty microphone gets sworn in, and a wig straightener glows beside a notice envelope. The joke targets public court-process theatrics and timing, not the underlying allegations.
""",
    )

    candidates = [
        {
            "slug": "banks_wig_trial_clock",
            "source_ids": ["rollingstone_banks_no_show_2026_04_22"],
            "freshness": 16,
            "famous_face_or_music_signal": 15,
            "public_conflict": 16,
            "absurd_source_detail": 19,
            "first_frame": 18,
            "audio_hook": 17,
            "duplicate_penalty": 0,
            "taste_risk_penalty": -8,
            "total": 93,
            "decision": "winner",
        },
        {
            "slug": "hendrix_royalty_vault_clock",
            "source_ids": ["rollingstone_hendrix_heirs_2026_04_29"],
            "freshness": 17,
            "famous_face_or_music_signal": 18,
            "public_conflict": 12,
            "absurd_source_detail": 7,
            "first_frame": 12,
            "audio_hook": 14,
            "duplicate_penalty": 0,
            "taste_risk_penalty": -6,
            "total": 74,
            "decision": "reject_less_comic",
        },
        {
            "slug": "apollonia_name_tag_paisley_counter",
            "source_ids": ["rollingstone_apollonia_settlement_2026_04_10"],
            "freshness": 10,
            "famous_face_or_music_signal": 15,
            "public_conflict": 12,
            "absurd_source_detail": 13,
            "first_frame": 16,
            "audio_hook": 14,
            "duplicate_penalty": -3,
            "taste_risk_penalty": -4,
            "total": 73,
            "decision": "reject_resolved_soft_conflict",
        },
        {
            "slug": "chappell_security_breakfast_gate",
            "source_ids": ["rollingstone_chappell_security_2026_03_28"],
            "freshness": 8,
            "famous_face_or_music_signal": 17,
            "public_conflict": 15,
            "absurd_source_detail": 13,
            "first_frame": 15,
            "audio_hook": 12,
            "duplicate_penalty": 0,
            "taste_risk_penalty": -18,
            "total": 62,
            "decision": "reject_child_fan_taste_risk",
        },
        {
            "slug": "bad_bunny_legal_bill_receipt_booth",
            "source_ids": ["rollingstone_bad_bunny_fees_2026_03_24"],
            "freshness": 7,
            "famous_face_or_music_signal": 19,
            "public_conflict": 11,
            "absurd_source_detail": 12,
            "first_frame": 12,
            "audio_hook": 15,
            "duplicate_penalty": -4,
            "taste_risk_penalty": -3,
            "total": 69,
            "decision": "reject_less_fresh",
        },
    ]
    write_json(
        "strategy/candidate_board.json",
        {"schema": "sgflix.candidate_board.v1", "created_at": NOW, "scoring_max": 100, "winner": "banks_wig_trial_clock", "candidates": candidates},
    )
    write_text(
        "strategy/winner_decision.md",
        """
# Winner Decision

Winner: `banks_wig_trial_clock`, score 93/100.

The winner has the strongest source-native absurd detail and a clear visual grammar: an empty court microphone, a moving trial clock, and a backstage wig-straightener clock. It also has a viable original audio chant without requiring source-audio cloning.

Rejected candidates were either more respectful/low-conflict legacy disputes, already resolved, older, or had avoidable taste risk around a child/fan encounter.

Risk boundary: keep every legal detail reported/alleged, avoid exact likeness, and keep the target on timing/process theatrics rather than the underlying case claims.
""",
    )
    write_json(
        "strategy/phase_minus_one_worthiness_audit.json",
        {
            "phase_minus_one_audit": {
                "source_target": "Azealia Banks final-status-hearing no-show / damages trial",
                "track_a_newsjack_velocity": {
                    "active_trend_score": 7,
                    "algorithmic_slipstream": "May 5 trial-forward legal calendar, reported April 22",
                    "polarization_factor": 8,
                    "track_a_total": 15,
                    "track_a_verdict": "PASS",
                },
                "track_b_archetype_resonance": {
                    "iconography_strength": 9,
                    "stereotype_rigidity": "High",
                    "subversion_potential": 9,
                    "track_b_total": 18,
                    "track_b_verdict": "PASS",
                },
                "system_verdict": {
                    "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                    "primary_vector": "TRACK_B_EVERGREEN_ARCHETYPE",
                    "urgency_class": "Medium",
                    "strategic_directive": "Literalize lateness and court process as a club-performance clock gag.",
                },
            }
        },
    )
    write_json(
        "strategy/source_entropy_audit.json",
        {
            "source_entropy_audit": {
                "baseline_reality_check": "A court update about a no-show hearing before a damages trial.",
                "detected_anomalies": ["reported no-show", "bench trial can proceed without participation", "reported wig-straightening deposition delay detail"],
                "native_entropy_score": 7,
                "subject_self_awareness": "trying_to_look_cool",
                "comedic_vector_recommendation": "native_absurdity",
                "recommended_strategy": "straight_man_framing",
            }
        },
    )
    write_json(
        "strategy/humor_logic_bridge.json",
        {
            "premise": "The performance is missing, but the trial clock performs anyway.",
            "truth": "Court calendars keep moving even when celebrity chaos tries to become the main event.",
            "turn": "An empty microphone gets sworn in like the absent star.",
            "button": "Proceeding with or without you.",
        },
    )
    write_json(
        "strategy/tribe_meta_score.json",
        {
            "TRiBE": {"truth": 8, "recognition": 7, "inversion": 9, "boldness": 8, "execution_clarity": 9},
            "meta": {"scroll_stop": 8, "share_captionability": 8, "comment_trigger": 8, "audio_reuse": 9},
            "summary": "Music/legal-watch audiences understand the 212 reference and the court-clock gag without needing underlying claims repeated.",
        },
    )
    write_json(
        "strategy/risk_taste_score.json",
        {
            "overall_risk": "medium",
            "taste_score": 7,
            "legal_sensitivity": 7,
            "mitigations": ["reported/alleged language", "fictionalized absent-performer archetype", "no exact likeness", "avoid quoting slurs or repeating inflammatory claims"],
            "reject_if": ["caption asserts liability beyond court ruling", "visual mocks protected traits", "image uses real court seal or private-party likeness"],
        },
    )
    write_text(
        "strategy/franchise_decision.md",
        """
# Franchise Decision

Decision: one-off, with reusable `courtroom-as-performance-booth` grammar.

Do not turn this into an Azealia-only franchise. The reusable template is broader: when a public figure misses process, the process becomes the performer.
""",
    )

    lyrics = """[Intro]
Clock in, clock out
Gavel on the two and four
Empty mic standing
Notice at the door

[Hook]
With you, without you, trial still moves
Clock keeps clicking in the green-room groove
Two hours late but the bass came through
With you, without you, stamp that proof

[Verse]
Bench light hot and the chair stays cold
Straightener hiss where the story gets told
No borrowed voice and no famous flow
Just a clerk on rhythm saying ready, set, go

[Hook]
With you, without you, trial still moves
Clock keeps clicking in the green-room groove
Two hours late but the bass came through
With you, without you, stamp that proof"""
    style = (
        "Original satirical club-court hook, 136 BPM, ballroom house pulse with clipped gavel kicks, "
        "receipt-printer hi hats, hair-straightener hiss FX, dry clerk chant, generic alto lead vocal, "
        "call-and-response hook, no imitation of Azealia Banks or any real artist, legally sterile comedy reel energy."
    )
    payload = {
        "schema": "sgflix.ace_step_payload.v1",
        "run_id": RUN_ID,
        "title": "Banks Wig Trial Clock",
        "trackName": "Banks Wig Trial Clock",
        "customMode": True,
        "taskType": "text2music",
        "instrumental": False,
        "lyrics": lyrics,
        "style": style,
        "instruction": "Create an original SGFLIX short hook for a fictional court/green-room timing gag. Keep the voice generic; do not imitate Azealia Banks, '212', or any real song.",
        "duration": 60,
        "bpm": 136,
        "keyScale": "",
        "timeSignature": "4/4",
        "batchSize": 1,
        "randomSeed": False,
        "seed": 84084,
        "thinking": False,
        "audioFormat": "mp3",
        "inferMethod": "ode",
        "guidanceScale": 7.0,
        "inferenceSteps": 27,
        "lmTemperature": 0.62,
        "lmCfgScale": 1.7,
        "lmTopP": 0.78,
        "lmTopK": 0,
        "shift": 3.0,
        "vocalLanguage": "en",
        "lmNegativePrompt": "Azealia Banks voice, 212 melody, celebrity impersonation, copyrighted melody, slurs, interview dialogue, diss track, aggressive harassment, muddy vocals, mumbled lyrics, second singer takeover, slow ballad",
        "useCotMetas": False,
        "useCotCaption": False,
        "useCotLanguage": False,
        "allowLmBatch": True,
        "getScores": False,
        "getLrc": False,
        "scoreScale": 0.5,
        "lmBatchChunkSize": 8,
        "completeTrackClasses": ["drums", "bass", "synth", "fx", "vocals", "backing_vocals"],
        "factory_notes": {
            "lane": "ACE-Step text-to-music, no source clone",
            "rights_guardrail": "No real-artist voice imitation, no source-song remake.",
            "visual_sync_target": "Use clock/gavel hits at 0.0-3.5s and hook landing for the empty-mic oath gag.",
        },
    }
    write_json("audio/ace_step_payload.json", payload)
    write_text(
        "audio/audio_concept.md",
        """
# Audio Concept

Audio role: original hook, not clone/remake.

The track should feel like a courtroom calendar converted into a club clock: gavel kicks, clerk-stamp snares, hair-straightener hiss, and a deadpan chant. The lead vocal must be generic and synthetic enough to avoid any real-artist impression.

Hook target: `With you, without you, trial still moves`.

Timing target: first gavel at 0.0s, trial-clock tick build from 0.8s, hook lands around 3.2s as the empty microphone is sworn in.
""",
    )
    write_json(
        "audio/music_handoff.json",
        {
            "run_id": RUN_ID,
            "usable_audio": "pending_remote_generation",
            "generated_candidates": [],
            "best_current_candidate": None,
            "hook_target": "With you, without you, trial still moves",
            "intended_drop_or_hook_moment": {"time_sec": 3.2, "visual": "empty microphone gets sworn in as gavel turns into kick drum"},
            "sections": [
                {"start": 0.0, "end": 0.8, "description": "single gavel kick and courtroom/green-room reveal"},
                {"start": 0.8, "end": 3.2, "description": "clock ticks and hair-iron hiss build under clerk chant"},
                {"start": 3.2, "end": 6.0, "description": "hook lands; trial stamp reads WITH OR WITHOUT YOU"},
            ],
            "payload_path": "audio/ace_step_payload.json",
            "qc_status": "pending",
        },
    )
    write_text(
        "qc/audio_qc.md",
        """
# Audio QC

Status: pending remote ACE-Step generation.

Preflight requirements:
- Original hook only; no source clone.
- Do not imitate Azealia Banks or reference `212` melody/flow.
- Keep lyrics about court-clock/process timing, not underlying allegations.
- Check half-time vs double-time as 136 BPM club pulse, not 68 BPM drag.
""",
    )

    write_json(
        "chai/chai_shot_specs.json",
        {
            "run_id": RUN_ID,
            "title": "Banks Wig Trial Clock",
            "shots": [
                {
                    "shot": "001",
                    "duration_sec": 6,
                    "subject": "fictional absent pop-rapper archetype represented by empty microphone and green-room props",
                    "scene": "courtroom merged with backstage vanity room",
                    "motion": "slow push from bench-trial clock to empty mic",
                    "spatial": "clock center, empty mic left, notice envelope right, wig straightener background",
                    "camera": "vertical 35mm, low dolly push, mild parallax",
                    "critique": "joke must read as process/timing, not as a factual accusation montage",
                    "revision": "if text drifts, crop to clock/mic and add clean overlay later",
                }
            ],
        },
    )
    shot = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "video_generation_allowed": False,
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "prompt": "No video generation in this automation. Manual closed-tool handoff only after human review.",
        "camera": {"lens": "35mm", "move": "slow push-in", "duration": "6s"},
        "visual_rules": ["fictionalized likeness", "generic court seals only", "large clean text only"],
    }
    write_json("scene_json/shot_0001.json", shot)
    write_json("scene_json/shot_001.json", shot)
    write_json(
        "handoffs/closed_tool_handoff.json",
        {
            "run_id": RUN_ID,
            "status": "stills_ready_audio_pending",
            "video_generation_tools_called": False,
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
            "human_next_step": "Review stills and audio candidate, then manually decide whether to hand off to a video tool.",
        },
    )
    write_text(
        "handoffs/grok_agent_prompt.md",
        """
# Grok Agent Prompt

Use the stills and JSON only as a manual handoff. Do not start a video render automatically.

Create a 6-second satirical court/green-room scene from `first_frame_v01.png`: slow push toward the trial clock, empty microphone gets lit by a witness-stand spotlight, and the notice envelope slides under the door on the hook. Keep all seals generic and all characters fictionalized.
""",
    )
    write_text(
        "captions/instagram_caption.md",
        """
POV: the calendar becomes the performer.

Reported setup: a music/court update, a no-show hearing, and one trial clock that refuses to wait.

Satire target: timing, process, and empty-mic theatrics. Allegation-framed; no verdict cosplay.

#sgflix #satire #musicnews #courtroom #aivideoart
""",
    )
    write_text(
        "distribution/post_plan.md",
        """
# Post Plan

Primary surface: Instagram Reels / TikTok vertical.

Hook overlay: `WITH YOU OR WITHOUT YOU`

Caption guardrail: say `reported` and keep the joke on hearing timing/process. Do not repeat inflammatory claims from trial briefs.

Post-ready status: stills ready; audio generation/QC pending remote result; no video generated.
""",
    )
    write_text(
        "skool/case_study.md",
        """
# Skool Case Study

Lesson: Use a source-native procedural detail as the comedy engine.

The run did not need to restage the legal dispute. The usable meme object is the clock: when a public figure misses process, make the process perform. That creates a first frame, hook lyric, and closed-tool scene without inventing new facts.
""",
    )

    first_frame_path = RUN_DIR / "frames/gpt_image_2/first_frame_v01.png"
    board_path = RUN_DIR / "storyboards/shared_choices/shared_choices_v01.png"
    first_frame(first_frame_path)
    storyboard(board_path)
    write_text(
        "frames/gpt_image_2/first_frame_v01_prompt.md",
        """
# First Frame Prompt

Vertical 16:9 satirical first-frame still, courtroom merged with backstage green room. Center: oversized bench-trial clock and empty witness microphone under a performance spotlight. Right: vanity table with a glowing hair straightener and notice envelope. Left: clerk desk stamping `PROCEEDS WITH OR WITHOUT YOU`. Pop-art legal comedy, premium editorial lighting, generic court symbols only, no exact celebrity likeness, no real court seal, no defamatory text, clean legible typography.
""",
    )
    write_text(
        "storyboards/shared_choices/shared_choices_v01_prompt.md",
        """
# Shared Choices Prompt

Create a director-bible storyboard board for `Banks Wig Trial Clock`: include character archetype represented by props, hero props, color palette, courtroom/green-room environment, floor plan/blocking, four storyboard panels with lens/camera notes, lighting/mood/style notes, visual safety rules, and production notes. Use generic legal symbols, avoid exact likeness, keep all facts reported/alleged.
""",
    )
    write_text(
        "qc/first_frame_v01_qc.md",
        """
# First Frame QC

Verdict: usable internal still; public-quality GPT Image repair recommended if exact editorial polish is required.

Passes:
- The clock/empty-mic contradiction is readable.
- No exact celebrity likeness is used.
- No real court seal or private-party depiction is present.
- No video generation was called.

Watchouts:
- Treat local schematic text as layout guide; use clean overlay if posting.
- Keep caption language reported/alleged.
""",
    )
    write_text(
        "qc/shared_choices_v01_qc.md",
        """
# Shared Choices QC

Verdict: usable director-bible board.

Passes:
- Includes character/hero props, palette, environment, floor plan, storyboard panels, lighting notes, visual rules, and production notes.
- Visual target stays on process/timing rather than underlying claims.
- No video generation was called.

Watchouts:
- Generated board is schematic; rerun prompt through GPT Image 2 for public-quality art if needed.
""",
    )

    required = [
        f"RUN_{RUN_ID}_MASTER_PACKAGE/README.md",
        f"RUN_{RUN_ID}_MASTER_PACKAGE/RUN_{RUN_ID}_MASTER_PACKAGE.json",
        "research/last30days_report.md",
        "research/sources.json",
        "strategy/candidate_board.json",
        "strategy/winner_decision.md",
        "strategy/phase_minus_one_worthiness_audit.json",
        "strategy/source_entropy_audit.json",
        "strategy/humor_logic_bridge.json",
        "strategy/tribe_meta_score.json",
        "strategy/risk_taste_score.json",
        "strategy/franchise_decision.md",
        "audio/audio_concept.md",
        "audio/ace_step_payload.json",
        "audio/music_handoff.json",
        "qc/audio_qc.md",
        "chai/chai_shot_specs.json",
        "scene_json/shot_0001.json",
        "scene_json/shot_001.json",
        "handoffs/closed_tool_handoff.json",
        "handoffs/grok_agent_prompt.md",
        "captions/instagram_caption.md",
        "distribution/post_plan.md",
        "skool/case_study.md",
        "manifests/asset_manifest.json",
        "FACTORY_RUN_STATUS.md",
        "frames/gpt_image_2/first_frame_v01.png",
        "frames/gpt_image_2/first_frame_v01_prompt.md",
        "storyboards/shared_choices/shared_choices_v01.png",
        "storyboards/shared_choices/shared_choices_v01_prompt.md",
        "qc/first_frame_v01_qc.md",
        "qc/shared_choices_v01_qc.md",
    ]
    manifest = {
        "run_id": RUN_ID,
        "slug": SLUG,
        "created_at": NOW,
        "required_files": required,
        "missing_files_pre_audio": [rel for rel in required if not (RUN_DIR / rel).exists()],
        "assets": [
            {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "local_schematic_equivalent_first_frame", "exists": first_frame_path.exists()},
            {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "local_schematic_equivalent_shared_choices_board", "exists": board_path.exists()},
        ],
        "audio_generation_status": "pending_remote_3090",
        "video_generation_tools_called": False,
    }
    write_json("manifests/asset_manifest.json", manifest)
    write_master_text(
        "README.md",
        f"""
# RUN {RUN_ID} MASTER PACKAGE - Banks Wig Trial Clock

Status: `STILLS_READY_AUDIO_PENDING_REMOTE_QC`.

Research query/topic: last-30-days music/legal stories with source-native absurdity and a clean original audio hook.

Selected premise: a courtroom becomes a backstage green room where the trial clock keeps moving and an empty microphone gets sworn in.

Score summary: winner 93/100; Hendrix royalty vault 74; Apollonia name-tag counter 73; Bad Bunny legal bill 69; Chappell security gate 62.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Audio status: ACE-Step payload created; remote generation/QC pending.

No video footage was generated or requested.
""",
    )
    write_master_json(
        f"RUN_{RUN_ID}_MASTER_PACKAGE.json",
        {
            "run_id": RUN_ID,
            "slug": SLUG,
            "created_at": NOW,
            "status": "STILLS_READY_AUDIO_PENDING_REMOTE_QC",
            "selected_premise": candidates[0],
            "sources": sources,
            "generated_stills": {
                "first_frame": "frames/gpt_image_2/first_frame_v01.png",
                "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
            },
            "audio": {
                "payload": "audio/ace_step_payload.json",
                "status": "pending_remote_3090",
            },
            "video_generation_tools_called": False,
        },
    )
    write_text(
        "FACTORY_RUN_STATUS.md",
        f"""
# FACTORY RUN STATUS - RUN {RUN_ID}

Status: `STILLS_READY_AUDIO_PENDING_REMOTE_QC`

Order of operations:
1. Research intake completed first.
2. Fresh candidate board created from current web/source context.
3. Candidates scored before winner selection.
4. Run package created only after winner selection.
5. Default audio lane created with concept, ACE-Step payload, handoff, and QC placeholder.
6. Still-image artifacts generated as local schematic equivalents with saved GPT Image 2 prompts.
7. No video-generation tools were called.

Research query/topic: last-30-days music/legal stories with source-native absurdity and clean audio-hook potential.

Selected premise: `banks_wig_trial_clock`.

Score summary:
- Banks Wig Trial Clock: 93, selected.
- Hendrix Royalty Vault Clock: 74, rejected less comic.
- Apollonia Name-Tag Paisley Counter: 73, rejected resolved/soft conflict.
- Bad Bunny Legal Bill Receipt Booth: 69, rejected less fresh.
- Chappell Security Breakfast Gate: 62, rejected child/fan taste risk.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Generated audio candidate paths: pending remote ACE-Step generation.

Keeper/reject decision: pending remote audio candidate and proxy QC.

Missing files: audio candidate/scorecard/keeper/hook timing are pending until remote generation completes.

Post-ready exports: none.

QC failures: public-quality GPT Image repair recommended for final art; audio pending.

High-risk issues:
- Keep legal details reported/alleged.
- Avoid exact celebrity likeness and any real court seal.
- Do not quote or reuse inflammatory language from litigation briefs.

Exact next human action: review stills and, after remote audio returns, judge whether the generated hook is a keeper.
""",
    )


if __name__ == "__main__":
    main()
