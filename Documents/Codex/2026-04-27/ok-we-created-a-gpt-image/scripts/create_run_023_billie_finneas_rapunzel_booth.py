import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN = "023"
SLUG = "billie_finneas_rapunzel_booth"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

SOURCES = [
    {
        "id": "elle_billie_finneas_2026",
        "title": "Billie Eilish Lights Up the Big Screen",
        "url": "https://www.elle.com/culture/music/a71028922/billie-eilish-james-cameron-concert-film-interview-2026/",
        "publisher": "ELLE",
        "published": "2026-04-28",
        "used_for": "winner source; primary interview context",
        "verification_status": "primary_interview_report",
        "notes": "Eilish discussed the May 8, 2026 concert film, denied a falling-out with Finneas, compared his prior stage placement to Rapunzel, and described massive sibling fights that quickly turn back into music-making."
    },
    {
        "id": "musicradar_billie_finneas_recap",
        "title": "Billie Eilish explains why her brother had become a Rapunzel figure",
        "url": "https://www.musicradar.com/artists/i-heard-somebody-say-did-you-guys-hear-finneas-and-billie-had-a-falling-out-billie-eilish-explains-why-her-brother-had-become-a-rapunzel-figure-in-her-touring-band-and-says-it-was-a-miracle-that-she-won-her-latest-grammy-for-wildflower",
        "publisher": "MusicRadar",
        "published": "2026-04-28",
        "used_for": "winner confirmation and phrasing context",
        "verification_status": "secondary_music_press_recap"
    },
    {
        "id": "dailybeast_guy_fieri_tate_backlash",
        "title": "Food Network Star Guy Fieri Breaks Silence After Being Caught Out With Tate Brothers",
        "url": "https://www.thedailybeast.com/food-network-star-guy-fieri-breaks-silence-after-being-caught-out-with-tate-brothers/",
        "publisher": "The Daily Beast",
        "published": "2026-04-14",
        "used_for": "candidate board alternative",
        "verification_status": "reported_backlash_context"
    },
    {
        "id": "tmz_nancy_mace_email_judge",
        "title": "Nancy Mace's Email to Judge Name-Dropping Trump Revealed in Court",
        "url": "https://www.tmz.com/2026/05/02/nancy-mace-name-dropping-the-president-revealed-in-court/",
        "publisher": "TMZ",
        "published": "2026-05-02",
        "used_for": "candidate board alternative",
        "verification_status": "tabloid_legal_docs_report"
    },
    {
        "id": "variety_paramount_wbd_subscriber_suit",
        "title": "Paramount Faces Suit From Streaming Subscribers Seeking to Block Warner Bros. Deal",
        "url": "https://au.variety.com/2026/film/news/paramount-streaming-subscribers-private-antitrust-36122/",
        "publisher": "Variety Australia",
        "published": "2026-05-01",
        "used_for": "candidate board alternative",
        "verification_status": "trade_legal_report"
    },
    {
        "id": "nbc_lively_baldoni_pretrial",
        "title": "Justin Baldoni and Blake Lively's legal teams hash out details ahead of trial",
        "url": "https://www.nbclosangeles.com/entertainment/entertainment-news/justin-baldoni-blake-lively-legal-teams-hash-out-details-ahead-of-trial/3882863/",
        "publisher": "NBC Los Angeles",
        "published": "2026-04-28",
        "used_for": "candidate board alternative",
        "verification_status": "mainstream_legal_report"
    }
]

CANDIDATES = [
    {
        "rank": 1,
        "slug": SLUG,
        "premise": "A pop-star sibling duo is sent to a five-minute studio arbitration booth: the producer brother is stuck in a tiny Rapunzel-style sound tower, the singer storms in wearing 3D glasses, and by the end of the countdown they are laughing with a finished guitar ballad.",
        "source_basis": ["elle_billie_finneas_2026", "musicradar_billie_finneas_recap"],
        "scores": {
            "famous_face_or_public_recognition": 9,
            "public_conflict": 5,
            "ego_or_status_pressure": 6,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 9,
            "first_frame_visual_contradiction": 10,
            "taste_safety": 9,
            "freshness": 9
        },
        "total": 65,
        "risk_notes": "Low-harm sibling quote and concert-film context. Avoid implying a real feud; make it an affectionate production-system joke."
    },
    {
        "rank": 2,
        "slug": "fieri_flavortown_decon_booth",
        "premise": "A flame-shirt celebrity chef enters a tiny Flavortown PR decontamination booth after a viral UFC handshake while every sauce bottle asks if he knows them.",
        "source_basis": ["dailybeast_guy_fieri_tate_backlash"],
        "scores": {
            "famous_face_or_public_recognition": 9,
            "public_conflict": 8,
            "ego_or_status_pressure": 7,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 10,
            "first_frame_visual_contradiction": 9,
            "taste_safety": 3,
            "freshness": 7
        },
        "total": 61,
        "risk_notes": "Rejected: source is slur-adjacent and tied to serious criminal allegations; only viable with heavy reframing."
    },
    {
        "rank": 3,
        "slug": "nancy_mace_kangaroo_court_email",
        "premise": "A congressional email thread turns into a literal courthouse inbox where a continuance request is stamped by a tiny White House invitation printer.",
        "source_basis": ["tmz_nancy_mace_email_judge"],
        "scores": {
            "famous_face_or_public_recognition": 6,
            "public_conflict": 8,
            "ego_or_status_pressure": 8,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 7,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 5,
            "freshness": 10
        },
        "total": 60,
        "risk_notes": "Fresh and absurd, but politically charged and tied to active abuse/defamation allegations."
    },
    {
        "rank": 4,
        "slug": "paramount_subscriber_merger_remote",
        "premise": "Five streaming subscribers enter antitrust court carrying one remote control that keeps merging every app button into a single expensive button.",
        "source_basis": ["variety_paramount_wbd_subscriber_suit"],
        "scores": {
            "famous_face_or_public_recognition": 5,
            "public_conflict": 8,
            "ego_or_status_pressure": 6,
            "humiliation_or_absurd_defense": 7,
            "brand_or_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 8,
            "freshness": 9
        },
        "total": 59,
        "risk_notes": "Good systems satire, but weaker famous-face signal."
    },
    {
        "rank": 5,
        "slug": "lively_baldoni_brand_receipts",
        "premise": "A pretrial damages hearing turns into a luxury-brand returns counter where every receipt asks whether public reputation or product-market fit caused the refund.",
        "source_basis": ["nbc_lively_baldoni_pretrial"],
        "scores": {
            "famous_face_or_public_recognition": 9,
            "public_conflict": 9,
            "ego_or_status_pressure": 8,
            "humiliation_or_absurd_defense": 7,
            "brand_or_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 2,
            "freshness": 8
        },
        "total": 59,
        "risk_notes": "Rejected: active sexual-harassment/retaliation litigation and private damages issues make comedy taste risk too high."
    }
]

FIRST_FRAME_PROMPT = """Vertical 9:16 satirical cinematic first frame for SGFLIX. A fictional young pop-star archetype, not a photoreal likeness, stands in a private Hollywood screening room wearing oversized 3D glasses and a black-green tour jacket, pointing at a five-minute countdown timer. Across the room, a fictional producer-brother archetype sits in a tiny Rapunzel-style sound booth tower built out of acoustic foam, guitar cables, and studio monitors, holding an acoustic guitar like a peace treaty. The contradiction is affectionate sibling conflict turned into absurd music-industry infrastructure: giant 3D concert-film energy, tiny arbitration tower, finished guitar ballad on a clipboard. Premium editorial lighting, 35mm lens, shallow depth, rich black, neon green, theater red, brushed chrome, no real logos, no exact celebrity likeness, no readable copyrighted song titles, no tabloid cruelty."""

SHARED_CHOICES_PROMPT = """Create one 16:9 SGFLIX Shared Choices director-bible board for Run 023, 'five-minute sibling arbitration tower'. Include: fictional pop-star sibling duo character canon, hero props of 3D glasses, acoustic guitar, five-minute countdown clock, Rapunzel-style acoustic foam booth tower, color palette swatches black / neon green / theater red / chrome / warm guitar wood, Hollywood screening room and studio set design, floor plan and blocking, six storyboard panels with camera/lens/movement notes, lighting and mood notes, visual rules, and production notes. Keep text minimal and clean. Do not use exact Billie Eilish or Finneas likenesses, real logos, real album art, readable song titles, or defamatory feud language."""


def write_text(rel, text):
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(rel, data):
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def make_placeholder(path, title, subtitle, board=False):
    size = (1536, 1024) if board else (1024, 1536)
    img = Image.new("RGB", size, "#101215")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 54)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except Exception:
        title_font = body_font = None
    draw.rectangle([36, 36, size[0] - 36, size[1] - 36], outline="#38f58a", width=6)
    draw.text((76, 88), title, fill="#f8f8f2", font=title_font)
    y = 186
    for line in subtitle.split("\n"):
        draw.text((76, y), line, fill="#e0e0d8", font=body_font)
        y += 45
    if board:
        colors = ["#050505", "#38f58a", "#a31621", "#bcc2c7", "#9b6a3c"]
        for idx, color in enumerate(colors):
            x = 86 + idx * 116
            draw.rectangle([x, size[1] - 168, x + 84, size[1] - 88], fill=color, outline="#f8f8f2")
        for i in range(6):
            x = 86 + (i % 3) * 460
            y = 360 + (i // 3) * 220
            draw.rectangle([x, y, x + 360, y + 145], outline="#a31621", width=4)
    else:
        draw.rectangle([342, 650, 682, 1090], outline="#a31621", width=7)
        draw.text((410, 820), "5:00\nBOOTH", fill="#38f58a", font=title_font)
    img.save(path)


def generate_image(prompt, path, size):
    try:
        from openai import OpenAI

        client = OpenAI()
        result = client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
        path.write_bytes(base64.b64decode(result.data[0].b64_json))
        return "openai_gpt_image_api"
    except Exception as exc:
        path.with_suffix(".generation_error.txt").write_text(
            f"OpenAI image generation failed; placeholder created. Error: {exc}\n",
            encoding="utf-8",
        )
        make_placeholder(
            path,
            "RUN 023 IMAGE BLOCKED",
            "GPT image generation failed.\nPrompt is preserved beside this placeholder.\nDo not route to video until repaired.",
            board=("shared_choices" in str(path)),
        )
        return "blocked_placeholder_after_openai_error"


def main():
    for sub in [
        "research",
        "strategy",
        "frames/gpt_image_2",
        "storyboards/shared_choices",
        "qc",
        "chai",
        "scene_json",
        "handoffs",
        "captions",
        "distribution",
        "skool",
        "manifests",
    ]:
        (PKG / sub).mkdir(parents=True, exist_ok=True)

    previous = ROOT / "sgflix_runs" / "run_022_kathy_jello_scam_pantry" / "NEXT_STEP_REPORT_2026-05-02.md"
    previous.write_text(
        "# Next-Step Report - Run 022\n\n"
        "`run_022_kathy_jello_scam_pantry` remains blocked at GPT Image generation. "
        "The package has research, strategy, prompts, placeholders, and QC, but its saved image prompts must be rerun through GPT Image 2 after billing/access is restored before any video workflow.\n",
        encoding="utf-8",
    )

    write_text("research/last30days_report.md", f"""
# Step 1: Research Intake - Run {RUN}

Created: {NOW}

Research query/topic: late-April and May 2, 2026 entertainment/source scan for fresh SGFLIX premises with famous-face signal, public conflict, absurd quote or defense, brand/location contrast, first-frame contradiction, and manageable taste risk.

## Current Source Context

- ELLE published a Billie Eilish interview on April 28, 2026 around her May 8 concert film with James Cameron. The useful comic source is not a scandal: it is her denial of a falling-out with Finneas, her "Rapunzel" framing for his prior stage placement, and her description of huge sibling fights resolving quickly back into music.
- MusicRadar recapped the same interview context and reinforced the Rapunzel/stage-tower image.
- Alternatives scanned: Guy Fieri's UFC/Tate backlash, Nancy Mace's email to a judge, Paramount/WBD subscriber antitrust lawsuit, and Blake Lively/Justin Baldoni pretrial issues.

## Winner Logic

The winner is an affectionate, low-harm music-industry satire: a five-minute sibling arbitration tower inside a 3D concert-film screening room. The visual contradiction is strong, the facts are limited and clearly sourced, and the package avoids falsely implying an actual feud.

## Verification Notes

All real-world context is treated as reported by the cited outlets. The creative output uses fictional archetypes, not exact likenesses or defamatory claims.
""")
    write_json("research/sources.json", {"created": NOW, "sources": SOURCES})
    write_json("strategy/candidate_board.json", {
        "run": RUN,
        "created": NOW,
        "scoring_scale": "0-10",
        "criteria": list(CANDIDATES[0]["scores"].keys()),
        "candidates": CANDIDATES,
        "winner": SLUG,
    })
    write_text("strategy/winner_decision.md", f"""
# Winner Decision - Run {RUN}

Winner: `{SLUG}`

Selected premise: a fictional pop-star sibling duo is forced into a five-minute studio arbitration system: the producer brother in a tiny Rapunzel-style sound booth tower, the singer in oversized 3D glasses, and a guitar ballad appearing as the peace treaty.

Score summary:
- Billie/Finneas Rapunzel Booth: 65
- Fieri Flavortown Decon Booth: 61
- Nancy Mace Kangaroo Court Email: 60
- Paramount Subscriber Merger Remote: 59
- Lively/Baldoni Brand Receipts: 59

Decision: proceed with still-image package and manual closed-tool handoff only. The humor must read as affectionate sibling-production absurdity, not a real feud accusation.
""")
    write_json("strategy/phase_minus_one_worthiness_audit.json", {
        "phase_minus_one_audit": {
            "source_target": "Billie Eilish / Finneas ELLE interview context",
            "track_a_newsjack_velocity": {
                "active_trend_score": 7,
                "algorithmic_slipstream": "Fresh Apr. 28 interview and May 8 concert-film rollout context",
                "polarization_factor": 3,
                "track_a_total": 10,
                "track_a_verdict": "PASS",
            },
            "track_b_archetype_resonance": {
                "iconography_strength": 9,
                "stereotype_rigidity": "Medium",
                "subversion_potential": 9,
                "track_b_total": 18,
                "track_b_verdict": "PASS",
            },
            "system_verdict": {
                "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                "primary_vector": "TRACK_B_EVERGREEN_ARCHETYPE",
                "urgency_class": "Medium",
                "strategic_directive": "Make sibling creative conflict physical through a tiny studio arbitration tower.",
            },
        }
    })
    write_json("strategy/source_entropy_audit.json", {
        "source_entropy_audit": {
            "baseline_reality_check": "A pop star denies a sibling falling-out and reframes normal creative fights as quickly resolved family dynamics.",
            "detected_anomalies": ["Rapunzel tower metaphor", "Concert film with 3D theater scale", "Five-minute pivot from fight to music-making"],
            "native_entropy_score": 4,
            "subject_self_awareness": "deadpan",
            "comedic_vector_recommendation": "cringe_vector",
            "recommended_strategy": "micro_spotlight",
        }
    })
    write_json("strategy/humor_logic_bridge.json", {
        "premise": SLUG,
        "straight_reality": "Sibling collaborators can have intense fights without a public falling-out.",
        "comic_inversion": "The music industry installs a literal five-minute arbitration tower to convert arguments into ballads.",
        "first_frame_joke": "3D concert-film scale meets a tiny Rapunzel-style producer booth.",
        "overlay_options": ["FIVE MINUTES LATER: A GUITAR BALLAD", "NO FEUD, JUST A TOWER", "SIBLING ARBITRATION BOOTH"],
        "do_not_do": ["Do not imply a real feud", "Do not use exact likenesses", "Do not use real album art or song titles"],
    })
    write_json("strategy/tribe_meta_score.json", {
        "tribe_meta_score": {
            "shareability": 8,
            "comment_prompt": 8,
            "remixability": 9,
            "first_frame_scroll_stop": 9,
            "caption_lore": 8,
            "total": 42,
            "notes": "Works for pop fans, sibling-collab jokes, and behind-the-scenes creative process discourse.",
        }
    })
    write_json("strategy/risk_taste_score.json", {
        "risk_taste_score": {
            "legal_defamation_risk": 2,
            "likeness_risk": 5,
            "harassment_or_abuse_context_risk": 1,
            "fanbase_misread_risk": 4,
            "brand_logo_risk": 4,
            "taste_verdict": "Proceed with fictionalized archetypes and explicit no-feud framing.",
        }
    })
    write_text("strategy/franchise_decision.md", """
# Franchise Decision

Verdict: `FORMAT_CANDIDATE`

This can become a repeatable SGFLIX format: famous creative partners placed into literal conflict-resolution infrastructure. Keep the tone affectionate and process-oriented.
""")
    write_text("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write_text("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)

    first_mode = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board_mode = generate_image(SHARED_CHOICES_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")
    blocked = first_mode.startswith("blocked") or board_mode.startswith("blocked")

    write_json("chai/chai_shot_specs.json", {
        "run": RUN,
        "title": "Five-Minute Sibling Arbitration Tower",
        "shots": [{
            "id": "shot_0001",
            "duration_sec": 6,
            "subject": "fictional pop-star sibling duo",
            "scene": "private Hollywood screening room with tiny acoustic-foam tower",
            "motion": "slow push from 3D glasses and countdown timer to producer booth tower",
            "spatial": "singer foreground, countdown midground, tower background",
            "camera": "vertical 9:16, 35mm, shallow depth, editorial theater lighting",
            "critique": "Must read as no-feud affectionate satire, not tabloid accusation.",
            "revision": "If likeness is too exact, hide faces through glasses, silhouette, or stylized wardrobe changes.",
        }],
    })
    shot = {
        "run": RUN,
        "shot_id": "shot_0001",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "duration_sec": 6,
        "camera": {"format": "9:16", "move": "slow dolly push", "lens": "35mm"},
        "action": "Countdown timer blinks while the guitar is raised like a peace treaty from the acoustic tower.",
        "overlay": "FIVE MINUTES LATER: A GUITAR BALLAD",
        "hard_stop": "No video generation requested by this package.",
    }
    write_json("scene_json/shot_0001.json", shot)
    shot_alt = dict(shot)
    shot_alt["shot_id"] = "shot_001"
    shot_alt["duration_sec"] = 10
    shot_alt["action"] = "The singer lowers the 3D glasses, points at the timer, and the producer slides a finished chorus under the booth door."
    write_json("scene_json/shot_001.json", shot_alt)
    write_json("handoffs/closed_tool_handoff.json", {
        "run": RUN,
        "title": "Five-Minute Sibling Arbitration Tower",
        "video_generation_allowed": False,
        "input_assets": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
        },
        "manual_only_note": "Do not generate video inside this automation. Human must approve or repair stills first.",
        "prompt_summary": "Affectionate sibling-collaboration satire inside a Hollywood 3D screening room and tiny acoustic Rapunzel booth.",
    })
    write_text("handoffs/grok_agent_prompt.md", """
# Grok Agent Prompt - Run 023

Use the approved first frame and Shared Choices board only after human review. Do not treat this package as permission to generate video inside the automation.

Core idea: a fictional pop-star sibling duo turns a huge argument into a finished guitar ballad through a five-minute arbitration tower.

Rules: no exact Billie Eilish or Finneas likenesses, no real logos, no album art, no readable copyrighted song titles, no implication of a real feud.
""")
    write_text("captions/instagram_caption.md", """
No feud. Just a five-minute sibling arbitration tower and a guitar ballad sliding under the door.

Source context: Billie Eilish told ELLE that she and Finneas have never had a falling-out, even if huge sibling fights can flip back into music fast. This is affectionate production satire.

#SGFLIX #BillieEilish #Finneas #MusicTok #ConcertFilm #Satire #SiblingEnergy
""")
    write_text("distribution/post_plan.md", """
# Post Plan

Primary surface: Instagram Reels / TikTok after still approval and manual render.

Hook overlay: FIVE MINUTES LATER: A GUITAR BALLAD

Angle: make the no-feud denial visual and affectionate. Avoid tabloid framing.

Do not auto-post. Do not start video generation from this automation.
""")
    write_text("skool/case_study.md", """
# Skool Case Study - Turning A Quote Into Infrastructure

Lesson: a non-scandal can still work if the quote has a physical metaphor. "Rapunzel" and "five minutes later" become production design: a tiny booth tower, a timer, and a guitar-as-peace-treaty.
""")
    write_json("manifests/asset_manifest.json", {
        "run": RUN,
        "created": NOW,
        "assets": [
            {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode},
            {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt"},
            {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode},
            {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt"},
            {"path": "qc/first_frame_v01_qc.md", "type": "qc"},
            {"path": "qc/shared_choices_v01_qc.md", "type": "qc"},
        ],
        "video_generation": "not_performed",
    })
    if blocked:
        write_text("qc/IMAGE_GENERATION_BLOCKED_REPORT.md", f"""
# Image Generation Blocked Report - Run {RUN}

GPT image generation was attempted after research, scoring, winner selection, and package creation.

Attempted assets:
- `frames/gpt_image_2/first_frame_v01.png` via `{first_mode}`
- `storyboards/shared_choices/shared_choices_v01.png` via `{board_mode}`

The PNGs currently present are placeholders if their generation mode says blocked. Do not route this run to video until GPT Image 2 access is restored, both saved prompts are generated, placeholders are replaced, and QC is rerun.
""")
    write_text("qc/first_frame_v01_qc.md", f"""
# First Frame QC - Run {RUN}

Asset: `frames/gpt_image_2/first_frame_v01.png`

Generation mode: `{first_mode}`

Verdict: {"blocked for production still use; placeholder only" if first_mode.startswith("blocked") else "usable for internal concept review pending human likeness/text inspection"}.

QC notes:
- Must not be an exact Billie Eilish or Finneas likeness.
- No real logos, album art, or readable song titles.
- Preserve the no-feud affectionate framing.
""")
    write_text("qc/shared_choices_v01_qc.md", f"""
# Shared Choices QC - Run {RUN}

Asset: `storyboards/shared_choices/shared_choices_v01.png`

Generation mode: `{board_mode}`

Verdict: {"blocked for production storyboard use; placeholder only" if board_mode.startswith("blocked") else "usable as private director-bible reference pending text inspection"}.

QC notes:
- Treat any microtext as non-final.
- Use the board for palette, blocking, set design, and prop logic.
- Repair if exact celebrity faces, real logos, or feud language appears.
""")
    status = "blocked_at_gpt_image_generation" if blocked else "complete_for_factory_cycle"
    write_json(f"RUN_{RUN}_MASTER_PACKAGE.json", {
        "run": RUN,
        "slug": SLUG,
        "title": "Five-Minute Sibling Arbitration Tower",
        "status": status,
        "created": NOW,
        "selected_after_research": True,
        "premise": CANDIDATES[0]["premise"],
        "generated_stills": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
        },
        "hard_stop_compliance": {
            "called_video_generation_tool": False,
            "requested_video_render": False,
            "auto_posted": False,
            "overwrote_approved_assets": False,
        },
        "next_human_action": "Review or repair the stills before any manual closed-tool video workflow.",
    })
    write_text("README.md", f"""
# RUN {RUN} MASTER PACKAGE - Five-Minute Sibling Arbitration Tower

Status: {status}; no video footage generated.

Research query/topic: late-April/May 2026 entertainment scan around Billie Eilish's ELLE interview, Finneas collaboration context, and adjacent fresh celebrity/legal candidates.

Selected premise: a fictional pop-star sibling duo uses a tiny Rapunzel-style acoustic booth and a five-minute timer to convert a huge argument into a finished guitar ballad.

Still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Next human action: inspect the stills for exact likeness/text/logo risk, or rerun the saved prompts in GPT Image 2 if this package is blocked.
""")
    write_text("FACTORY_RUN_STATUS.md", f"""
# Factory Run Status

Run: {RUN}
Slug: `{SLUG}`
Status: {status}; no video footage generated.
Created: {NOW}

## Order Of Operations

1. Research intake completed first.
2. Candidate board built from current web/source context.
3. Candidates scored before selection.
4. Winner selected: `{SLUG}`.
5. `run_{RUN}_{SLUG}` package created after winner selection.
6. Still artifacts attempted/generated and QC files created.

## Score Summary

- Billie/Finneas Rapunzel Booth: 65
- Fieri Flavortown Decon Booth: 61
- Nancy Mace Kangaroo Court Email: 60
- Paramount Subscriber Merger Remote: 59
- Lively/Baldoni Brand Receipts: 59

## Generated Still Paths

- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

## Missing Files

None from the required package checklist. If status is blocked, the PNGs are placeholders and production-ready GPT Image stills are missing.

## Post-Ready Exports

Caption and post plan are drafted. No rendered video export exists.

## QC Failures Or Cautions

- Exact public-figure likeness must be avoided.
- Generated text is not final copy.
- Framing must say no real feud; affectionate sibling-production satire only.

## High-Risk Issues

- Fanbase misread if the premise is captioned like tabloid conflict.
- Likeness risk if future image repair gets too photoreal.

## Exact Next Human Action

Review `frames/gpt_image_2/first_frame_v01.png` and `storyboards/shared_choices/shared_choices_v01.png`; approve, or rerun the saved prompts through GPT Image 2 and redo QC before any manual closed-tool video workflow.
""")


if __name__ == "__main__":
    main()
