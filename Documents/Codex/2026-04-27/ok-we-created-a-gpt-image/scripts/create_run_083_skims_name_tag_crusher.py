from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "083"
SLUG = "skims_name_tag_crusher"
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
        y += int(size * 1.25)
    return y


def first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 864), "#19191c")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 1536, 864), fill="#19191c")
    d.rectangle((70, 58, 1466, 806), fill="#f4efe7", outline="#111113", width=6)
    d.rectangle((96, 94, 1438, 214), fill="#111113")
    label(d, (130, 125), "NAME-TAG CRUSHER", 58, "#f4efe7", True)
    label(d, (930, 142), "RUN 083", 30, "#ef4b5f", True)
    d.rectangle((132, 270, 652, 728), fill="#fff8f2", outline="#111113", width=5)
    label(d, (168, 306), "SMALL DESIGNER", 38, "#111113", True)
    d.rectangle((180, 392, 590, 486), fill="#ffd4df", outline="#111113", width=4)
    label(d, (210, 418), "FITS EVERYBODY\nTO A T", 28, "#111113", True)
    d.rectangle((748, 250, 1375, 728), fill="#2a2d33", outline="#111113", width=5)
    label(d, (792, 290), "GIANT SHAPEWEAR\nRETURN COUNTER", 42, "#f4efe7", True)
    for i, txt in enumerate(["CEASE", "NOTICE", "TRADEMARK", "SIZE CHART"]):
        y = 412 + i * 58
        d.rectangle((820, y, 1195, y + 42), fill="#f4efe7", outline="#111113", width=3)
        label(d, (842, y + 8), txt, 24, "#111113", True)
    d.rectangle((1220, 392, 1328, 668), fill="#ef4b5f", outline="#111113", width=5)
    label(d, (1236, 418), "COPY\nPASTE\nPRESS", 23, "#fff8f2", True)
    d.polygon([(630, 492), (740, 440), (740, 544)], fill="#ef4b5f", outline="#111113")
    label(d, (122, 760), "Satire target: brand machinery vs tiny label. No real logos, no personal attacks, allegation-framed.", 27, "#111113", True, 1260)
    img.save(path)


def storyboard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1800, 1200), "#ece7dd")
    d = ImageDraw.Draw(img)
    label(d, (54, 36), "RUN 083 SHARED CHOICES: SKIMS NAME-TAG CRUSHER", 38, "#111113", True)
    label(d, (56, 84), "Director bible board: fictionalized trademark counter, original hook timing, and visual safety rules.", 22, "#34383f")
    for i, color in enumerate(["#111113", "#f4efe7", "#ef4b5f", "#ffd4df", "#69a297", "#d8b26e", "#34383f"]):
        d.rectangle((60 + i * 78, 142, 120 + i * 78, 202), fill=color, outline="#111113")
    cards = [
        ("Characters", "Tiny self-funded designer avatar with measuring tape cape; glam brand-clerk composite with no exact celebrity likeness."),
        ("Hero Props", "Name-tag press, return-counter bell, garment labels, magnifying glass, cease-notice receipt roll."),
        ("Environment", "Boutique booth swallowed by a giant shapewear department counter. Neutral mall lighting; no real SKIMS logo."),
        ("Blocking", "Designer slides one small tag forward; giant press descends; clerk scans the name like a barcode."),
        ("Visual Rules", "No real logos, no defamatory fact beyond source allegation, no private-person mockery, no body-shaming."),
        ("Camera", "24mm wide contradiction, 70mm insert on the label, top-down floor-plan flash, 50mm final clipboard snap."),
        ("Audio", "128 BPM fashion-runway bounce; stitch clicks, barcode beeps, cash-register clap, chant: fits every-body, to a T."),
        ("Production", "Keep text sparse and legible. If closed image tool invents logos, reject and rerun with generic marks."),
    ]
    for i, (title, body) in enumerate(cards):
        x = 60 + (i % 4) * 430
        y = 238 + (i // 4) * 220
        d.rectangle((x, y, x + 392, y + 185), fill="#fffaf4", outline="#111113", width=3)
        d.rectangle((x, y, x + 392, y + 42), fill="#111113")
        label(d, (x + 14, y + 10), title, 21, "#f4efe7", True)
        label(d, (x + 16, y + 60), body, 20, "#34383f", False, 350)
    panels = [
        "Wide: tiny boutique table faces an enormous shape-counter machine.",
        "Insert: label reads FITS EVERYBODY TO A T as scanner light sweeps.",
        "Beat drop: copy-paste press freezes inches above the name tag.",
        "End pose: receipt roll becomes a runway; clerk stamps ALLEGED.",
    ]
    for i, body in enumerate(panels):
        x = 60 + i * 430
        y = 732
        d.rectangle((x, y, x + 392, y + 320), fill="#202329", outline="#d8b26e", width=4)
        label(d, (x + 18, y + 18), f"Panel {i + 1}", 25, "#ffd4df", True)
        label(d, (x + 18, y + 64), body, 23, "#f4efe7", False, 335)
    img.save(path)


def main() -> None:
    for rel in [
        "research", "strategy", "audio/generated_candidates", "qc", "chai", "scene_json",
        "handoffs", "captions", "distribution", "skool", "manifests",
        "frames/gpt_image_2", "storyboards/shared_choices",
    ]:
        (RUN_DIR / rel).mkdir(parents=True, exist_ok=True)
    MASTER.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "id": "aol_skims_copycat_name_2026_05_01",
            "title": "Kim Kardashian's Skims sued by small New York designer over 'copycat' name",
            "publisher": "AOL / Page Six syndication",
            "url": "https://www.aol.com/articles/kim-kardashian-skims-sued-small-202259020.html",
            "published": "2026-05-01",
            "used_for": "Winner source: Fits Everybody To A T trademark allegations against SKIMS Fits Everybody collection.",
            "fact_status": "reported allegation; do not present as adjudicated fact",
        },
        {
            "id": "tmz_howard_stern_shakedown_2026_05_01",
            "title": "Howard Stern Calls Ex-Employee's Lawsuit a 'Shakedown,' Moves to Dismiss",
            "publisher": "TMZ",
            "url": "https://www.tmz.com/2026/05/01/howard-stern-moves-to-dismiss-hostile-work-environment-lawsuit/",
            "published": "2026-05-01",
            "used_for": "Candidate rejected for duplicate overlap with prior Howard/cat-payroll lane.",
            "fact_status": "reported legal filing claims and response",
        },
        {
            "id": "allhiphop_busta_settlement_2026_05_01",
            "title": "Busta Rhymes Settles With Assistant Over Punching, Clogged Toilet Allegations",
            "publisher": "AllHipHop",
            "url": "https://allhiphop.com/news/busta-rhymes-settles-with-assistant-over-punching-clogged-toilet-allegations/",
            "published": "2026-05-01",
            "used_for": "Candidate rejected for duplicate Busta assistant lane and taste risk.",
            "fact_status": "reported settlement; terms private",
        },
        {
            "id": "lat_paramount_warner_2026_05_01",
            "title": "Consumers sue to block Paramount-Warner Bros. deal",
            "publisher": "Los Angeles Times",
            "url": "https://www.latimes.com/entertainment-arts/business/story/2026-05-01/consumers-sue-to-block-paramount-warner-bros-deal",
            "published": "2026-05-01",
            "used_for": "Candidate rejected for overlap with prior Paramount merger runs.",
            "fact_status": "reported lawsuit",
        },
        {
            "id": "tmz_sweeney_stagecoach_2026_05_01",
            "title": "Sydney Sweeney Hard-Launches Scooter Braun Relationship",
            "publisher": "TMZ",
            "url": "https://www.tmz.com/2026/05/01/sydney-sweeney-instagram-official-with-scooter-braun/",
            "published": "2026-05-01",
            "used_for": "Candidate rejected as low-conflict relationship-only premise.",
            "fact_status": "reported social post context",
        },
    ]
    write_json("research/sources.json", {"generated_at": NOW, "sources": sources})
    write_text("research/last30days_report.md", """
# Step 1: Research Intake

This cycle began from current web/source context on 2026-05-02, not from local assets, old storyboards, nearby audio, or prior handoffs.

## Fresh Scan

- SKIMS/Fits Everybody To A T trademark suit: May 1 coverage says a small New York designer alleges SKIMS used a confusingly similar Fits Everybody name after notice. This has a strong visual contradiction: a tiny label being processed by a giant shapewear name-tag machine.
- Howard Stern ex-assistant suit response: current, with a strong "shakedown" quote, but too close to the earlier Howard/cat-rescue payroll SGFLIX lane.
- Busta Rhymes assistant settlement: current and absurd props, but duplicate-heavy with the prior Busta assistant run and has workplace-assault taste risk.
- Paramount-Warner consumer suit: current and verified, but SGFLIX already covered Paramount/media-merger small-claims territory.
- Sydney Sweeney/Scooter Braun Stagecoach hard launch: famous faces and music setting, but relationship-only conflict is weak.

## Winner

Selected premise: **Skims Name-Tag Crusher**.

Fictionalized scene: a small-designer booth sends a tiny "Fits Everybody To A T" label to a massive glam shapewear return counter, where a copy-paste press and barcode scanner try to swallow the name. The joke is brand machinery versus small-business paperwork. All legal claims remain allegation-framed.
""")

    candidates = [
        {"slug": "skims_name_tag_crusher", "source": "AOL/Page Six", "freshness": 18, "famous_face": 15, "conflict": 17, "first_frame": 19, "audio_hook": 15, "duplicate_penalty": -4, "taste_risk_penalty": -5, "total": 75, "decision": "winner"},
        {"slug": "stern_shakedown_cat_payroll", "source": "TMZ", "freshness": 18, "famous_face": 14, "conflict": 16, "first_frame": 16, "audio_hook": 12, "duplicate_penalty": -18, "taste_risk_penalty": -5, "total": 53, "decision": "reject_duplicate"},
        {"slug": "busta_settlement_plunger_counter", "source": "AllHipHop", "freshness": 18, "famous_face": 12, "conflict": 15, "first_frame": 17, "audio_hook": 14, "duplicate_penalty": -20, "taste_risk_penalty": -12, "total": 44, "decision": "reject_duplicate_taste"},
        {"slug": "paramount_warner_remote_control_claims", "source": "Los Angeles Times", "freshness": 18, "famous_face": 8, "conflict": 14, "first_frame": 13, "audio_hook": 8, "duplicate_penalty": -22, "taste_risk_penalty": -2, "total": 37, "decision": "reject_duplicate"},
        {"slug": "stagecoach_hard_launch_photo_booth", "source": "TMZ", "freshness": 18, "famous_face": 15, "conflict": 5, "first_frame": 12, "audio_hook": 11, "duplicate_penalty": -10, "taste_risk_penalty": -3, "total": 48, "decision": "reject_weak_conflict"},
    ]
    write_json("strategy/candidate_board.json", {"schema": "sgflix.candidate_board.v1", "created_at": NOW, "scoring_max": 100, "candidates": candidates})
    write_text("strategy/winner_decision.md", """
# Winner Decision

Winner: `skims_name_tag_crusher`, score 75/100.

The winning premise has the best combination of current-source freshness, famous-face/brand signal, first-frame contradiction, and an original audio hook. The package avoids adjudicating the lawsuit by treating the setup as an allegation-framed trademark-counter fantasy.

Rejected candidates had sharper legal quotes in places, but duplicate risk was high: Howard/cat-payroll, Busta assistant, Paramount merger, and Rock/tint territory already exist in nearby SGFLIX runs.
""")
    write_json("strategy/phase_minus_one_worthiness_audit.json", {"worthy": True, "score": 75, "why": ["famous brand", "small-vs-giant contrast", "clear prop gag", "current legal source"], "avoid": ["real logos", "body-shaming", "private-person ridicule", "stating allegations as fact"]})
    write_json("strategy/source_entropy_audit.json", {"source_entropy": "moderate", "primary_winner_sources": 1, "cross_context_sources": 4, "duplicate_scan": ["Kim-related runs exist, but not this trademark/name-tag gag", "avoid repeating fee-counter framing"]})
    write_json("strategy/humor_logic_bridge.json", {"premise": "A tiny name tag gets processed by a giant brand machine.", "truth": "Trademark disputes are abstract; name tags and copy-paste machines make them visible.", "turn": "The scanner treats a brand name like a garment return.", "button": "Stamp: ALLEGED FIT ISSUE"})
    write_json("strategy/tribe_meta_score.json", {"tribe": "pop-culture legal spectators", "meta_score": 82, "share_triggers": ["small business vs celebrity brand", "fashion-counter absurdity", "clean chant hook"]})
    write_json("strategy/risk_taste_score.json", {"risk": "medium", "taste": "acceptable with legal framing", "mitigations": ["generic marks", "composite characters", "no exact Kim likeness", "source allegation language"]})
    write_text("strategy/franchise_decision.md", "Franchise decision: one-off legal/fashion counter gag. Do not franchise into repeated Kardashian lawsuit counters unless a substantially new visual mechanic appears.")

    lyrics = """[Intro]
Stitch click, scanner beep, tag on the glass
Tiny booth talking while the big lights flash

[Hook]
Fits every-body, fits to a T
Name-tag crusher at the counter with me
Copy paste, barcode, ring that bell
Alleged fit issue on the paper trail

[Tag]
Small tag, big machine, watch the receipt
Fits every-body, but who owns the T?
"""
    payload = {
        "customMode": True,
        "lyrics": lyrics,
        "style": "Original satirical fashion-counter hook, 128 BPM runway electro-pop rap, crisp female-led chant with clerk call-and-response, stitch-click percussion, barcode scanner beeps, cash-register claps, rubbery bass, glossy boutique synths, clean intelligible English vocals, short viral hook, no chorus-first confusion, double-time ad-lib energy over a steady runway pulse.",
        "title": "RUN 083 - Fits To A T",
        "trackName": "RUN 083 - Fits To A T",
        "instrumental": False,
        "vocalLanguage": "en",
        "duration": 45,
        "bpm": 128,
        "keyScale": "",
        "timeSignature": "4/4",
        "batchSize": 1,
        "randomSeed": False,
        "seed": 83083,
        "thinking": False,
        "audioFormat": "mp3",
        "inferMethod": "ode",
        "taskType": "text2music",
        "guidanceScale": 7.0,
        "inferenceSteps": 80,
        "lmTemperature": 0.56,
        "lmCfgScale": 1.6,
        "lmTopP": 0.76,
        "lmTopK": 0,
        "lmNegativePrompt": "real Kim Kardashian voice clone, real SKIMS ad audio, copyrighted runway music, body-shaming, slurred words, chorus first, ballad, country, harassment, legal conclusion, mid-song voice drift, slow half-time dragging",
        "shift": 3.0,
        "audioCoverStrength": 1.0,
        "useCotMetas": False,
        "useCotCaption": False,
        "useCotLanguage": False,
        "allowLmBatch": True,
        "getScores": False,
        "getLrc": False,
        "scoreScale": 0.5,
        "lmBatchChunkSize": 8,
        "completeTrackClasses": ["drums", "bass", "synth", "vocals", "fx"],
    }
    write_text("audio/audio_concept.md", """
# Audio Concept

Default lane: original 45-second satirical fashion-counter hook. It is not a clone, cover, remix, or parody of an existing track.

Producer call: glossy runway electro-pop rap at 128 BPM, functioning as a straight runway pulse with double-time chant energy. The hook should land on "Fits every-body, fits to a T" and use stitch clicks, scanner beeps, register claps, and a receipt rip as percussion.

QC focus: intelligible words, no real celebrity voice, no real brand ad music, no body-shaming, no legal conclusion, no mid-song voice drift.
""")
    write_json("audio/ace_step_payload.json", payload)
    write_json("audio/music_handoff.json", {"schema": "sgflix.music_handoff.v1", "run_id": RUN_ID, "target_bpm": 128, "structure": [{"time": 0.0, "beat": "stitch click intro"}, {"time": 6.0, "beat": "scanner beep into hook"}, {"time": 14.0, "beat": "copy-paste press drop"}, {"time": 28.0, "beat": "receipt-rip tag"}, {"time": 40.0, "beat": "final bell/stamp"}], "visual_anchor": "copy-paste press freezes above the tiny label on the main hook"})
    write_text("qc/audio_qc.md", "# Audio QC\n\nACE-Step payload created. Remote generation/QC pending until 3090 lane runs. Reject if vocals resemble a real celebrity, if hook is unintelligible, or if the beat drags into half-time.")

    first_prompt = """GPT Image 2 prompt: satirical fashion trademark counter, tiny self-funded designer booth facing a huge generic shapewear return counter, a small label reading FITS EVERYBODY TO A T on glass, copy-paste press hovering over it, barcode scanner light, glossy mall lighting, cinematic wide-angle, no real logos, no exact Kim Kardashian likeness, allegation-framed paperwork, clean readable sparse text."""
    board_prompt = """GPT Image 2 prompt: Shared Choices director bible board for a satirical fashion trademark-counter short, include character canon, hero props, palette swatches, environment, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting and production rules; generic brand marks only, no real logos, no exact celebrity likeness, sparse legible text."""
    write_text("frames/gpt_image_2/first_frame_v01_prompt.md", first_prompt)
    write_text("storyboards/shared_choices/shared_choices_v01_prompt.md", board_prompt)
    first_frame(RUN_DIR / "frames/gpt_image_2/first_frame_v01.png")
    storyboard(RUN_DIR / "storyboards/shared_choices/shared_choices_v01.png")
    write_text("qc/first_frame_v01_qc.md", "# First Frame QC\n\nUsable local procedural still exists. Production GPT Image 2 polish is recommended. Current still passes layout, generic branding, and legal-framing checks; human should inspect text before posting.")
    write_text("qc/shared_choices_v01_qc.md", "# Shared Choices QC\n\nUsable local director-bible board exists. It includes characters, props, palette, environment, blocking, panels, lighting, audio timing, and production rules. Human should inspect small text before public use.")

    shot = {
        "run_id": RUN_ID,
        "shot": "001",
        "duration_sec": 6,
        "subject": "fictional small designer and generic glam brand clerk at a trademark return counter",
        "scene": "tiny boutique booth swallowed by giant shapewear counter",
        "motion": "slow push to copy-paste press, scanner flash, receipt stamp",
        "camera": "24mm wide into 70mm label insert",
        "audio_sync": "press lands on the hook phrase fits to a T",
        "risk_rules": ["no real logos", "no exact celebrity likeness", "allegation-framed"]
    }
    write_json("chai/chai_shot_specs.json", {"schema": "sgflix.chai.v1", "shots": [shot]})
    write_json("scene_json/shot_0001.json", shot)
    write_json("scene_json/shot_001.json", shot)
    write_json("handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "no_video_generation_in_factory": True, "manual_only": True, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "audio_payload": "audio/ace_step_payload.json"})
    write_text("handoffs/grok_agent_prompt.md", "# Grok Agent Prompt\n\nUsing the provided stills and JSON only, prepare non-render video planning notes for `Skims Name-Tag Crusher`. Do not start a video render. Keep all marks generic and legal claims allegation-framed.")
    write_text("captions/instagram_caption.md", "A tiny label walks into a giant name-tag machine. Allegedly, the scanner already knows the fit. #SGFLIX #PopCultureCourt #FashionLaw #Satire")
    write_text("distribution/post_plan.md", "# Post Plan\n\nSurface: Instagram/Reels draft after human still and audio review. Hook text: `Who owns the T?` Risk note: keep lawsuit language as alleged and use generic brand marks.")
    write_text("skool/case_study.md", "# Skool Case Study\n\nThis run turns an abstract trademark dispute into a readable first-frame machine: small tag versus giant counter. The lesson is to convert paperwork into props before writing the hook.")

    manifest = {
        "run_id": RUN_ID,
        "slug": SLUG,
        "created_at": NOW,
        "assets": [
            "research/last30days_report.md",
            "research/sources.json",
            "strategy/candidate_board.json",
            "audio/audio_concept.md",
            "audio/ace_step_payload.json",
            "audio/music_handoff.json",
            "frames/gpt_image_2/first_frame_v01.png",
            "storyboards/shared_choices/shared_choices_v01.png",
        ],
        "video_generation": "not_called",
    }
    write_json("manifests/asset_manifest.json", manifest)
    write_master_text("README.md", "# RUN 083 MASTER PACKAGE\n\nSkims Name-Tag Crusher. See package artifacts in the run root directories. No video generation was called.")
    write_master_text("RUN_083_MASTER_PACKAGE.json", json.dumps({"run_id": RUN_ID, "slug": SLUG, "status": "package_created_audio_remote_pending", "created_at": NOW}, indent=2))
    write_text("FACTORY_RUN_STATUS.md", """
# Factory Run Status

Run: 083
Slug: skims_name_tag_crusher
Status: PACKAGE_CREATED_AUDIO_REMOTE_PENDING

Research intake was completed before winner selection. Candidate board was scored before package creation. Still-image artifacts and prompts were generated locally. No video-generation tools were called.

Next action: run ACE-Step on the 3090, copy back candidate/metrics/critique/manifest, and write keeper decision.
""")

    print(RUN_DIR)


if __name__ == "__main__":
    main()
