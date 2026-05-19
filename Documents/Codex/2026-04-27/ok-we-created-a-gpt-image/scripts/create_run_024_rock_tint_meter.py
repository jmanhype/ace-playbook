import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN = "024"
SLUG = "rock_tint_meter"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

SOURCES = [
    {
        "id": "tmz_rock_traffic_stop",
        "title": "Dwayne 'The Rock' Johnson Pulled Over by Police After Walk of Fame Event",
        "url": "https://www.tmz.com/2026/04/30/dwayne-johnson-pulled-over-in-los-angeles/",
        "publisher": "TMZ",
        "published": "2026-04-30",
        "updated": "2026-05-01",
        "used_for": "winner source",
        "verification_status": "reported_by_entertainment_press; traffic-stop reason attributed to law-enforcement sources",
        "notes": "Reported that Johnson was stopped after a Hollywood Walk of Fame event; the stated traffic-stop reason was tinted windows."
    },
    {
        "id": "justjared_rock_stop_recap",
        "title": "Why Dwayne Johnson Got Pulled Over by Police After Attending Emily Blunt's Walk of Fame Ceremony",
        "url": "https://www.justjared.com/2026/05/01/why-dwayne-johnson-got-pulled-over-by-police-after-attending-emily-blunts-walk-of-fame-ceremony/",
        "publisher": "Just Jared",
        "published": "2026-05-01",
        "used_for": "winner confirmation",
        "verification_status": "secondary entertainment recap",
        "notes": "Confirms same broad premise: Johnson was pulled over after the April 30 ceremony."
    },
    {
        "id": "tmz_kanye_sofi_lights",
        "title": "Kanye West Gets Pissed Over Stage Lights While Performing At L.A. Concert",
        "url": "https://www.tmz.com/2026/04/02/kanye-west-upset-over-lights-at-sofi-stadium/",
        "publisher": "TMZ",
        "published": "2026-04-02",
        "updated": "2026-04-02",
        "used_for": "candidate board alternative",
        "verification_status": "entertainment-press video recap",
        "notes": "Reported a SoFi concert stop-down over disco-style/corny lighting and an SNL-sketch remark."
    },
    {
        "id": "tmz_blake_nickelodeon",
        "title": "Blake Lively Spends Day at Nickelodeon Universe, Amid Justin Baldoni Drama",
        "url": "https://www.tmz.com/2026/04/01/blake-lively-visits-nickelodeon-universe-during-justin-baldoni-drama/",
        "publisher": "TMZ",
        "published": "2026-04-01",
        "used_for": "candidate board alternative",
        "verification_status": "social-post recap",
        "notes": "Reported a Nickelodeon Universe visit and a quoted public-humiliation caption amid an active legal dispute."
    },
    {
        "id": "tmz_chappelle_weaponized",
        "title": "Dave Chappelle Says He Resents Republicans for Weaponizing His Transgender Jokes",
        "url": "https://www.tmz.com/2026/04/15/dave-chappelle-resents-republicans-weaponizing-trans-jokes/",
        "publisher": "TMZ",
        "published": "2026-04-15",
        "updated": "2026-04-15",
        "used_for": "candidate board alternative",
        "verification_status": "interview recap",
        "notes": "Reported Chappelle's NPR interview comments about political photo-op framing."
    },
    {
        "id": "tmz_britney_dui_charge",
        "title": "Britney Spears Hit With DUI Charge",
        "url": "https://www.tmz.com/categories/celebrity-arrests/",
        "publisher": "TMZ",
        "published": "2026-04-30",
        "updated": "2026-04-30",
        "used_for": "candidate board alternative",
        "verification_status": "charge report; not selected for taste reasons",
        "notes": "Fresh famous-face legal item, rejected due to health/legal sensitivity."
    }
]

CANDIDATES = [
    {
        "rank": 1,
        "slug": SLUG,
        "premise": "A Hollywood Walk of Fame stop turns into the world's most overbuilt window-tint inspection: a fictional action-star titan stands beside a cream-colored luxury car while a tiny officer uses a jeweler-style tint meter like it is a red-carpet award.",
        "source_basis": ["tmz_rock_traffic_stop", "justjared_rock_stop_recap"],
        "scores": {
            "famous_face_or_public_recognition": 10,
            "public_conflict": 6,
            "ego_or_status_pressure": 7,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 10,
            "first_frame_visual_contradiction": 10,
            "taste_safety": 9,
            "freshness": 10
        },
        "total": 70,
        "risk_notes": "Keep it as routine traffic-stop absurdism. Do not imply crime beyond the reported tinted-window stop."
    },
    {
        "rank": 2,
        "slug": "kanye_sofi_light_court",
        "premise": "A stadium lighting rig is put on trial while a fictional perfectionist rapper cross-examines a disco ball for looking like an SNL sketch.",
        "source_basis": ["tmz_kanye_sofi_lights"],
        "scores": {
            "famous_face_or_public_recognition": 10,
            "public_conflict": 8,
            "ego_or_status_pressure": 10,
            "humiliation_or_absurd_defense": 9,
            "brand_or_location_contrast": 8,
            "first_frame_visual_contradiction": 9,
            "taste_safety": 5,
            "freshness": 7
        },
        "total": 66,
        "risk_notes": "Strong quote, but lower freshness and higher controversy load."
    },
    {
        "rank": 3,
        "slug": "blake_nickelodeon_pretrial_crane",
        "premise": "A celebrity legal-calendar war is reframed as a Nickelodeon obstacle-course calendar invite where the Double Dare crane stamps every deposition with slime.",
        "source_basis": ["tmz_blake_nickelodeon"],
        "scores": {
            "famous_face_or_public_recognition": 9,
            "public_conflict": 9,
            "ego_or_status_pressure": 8,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 10,
            "first_frame_visual_contradiction": 9,
            "taste_safety": 3,
            "freshness": 7
        },
        "total": 63,
        "risk_notes": "Rejected: active litigation has too much serious context."
    },
    {
        "rank": 4,
        "slug": "chappelle_photo_op_receipt",
        "premise": "A comedy-club green room becomes a Capitol Hill receipt printer that spits out captions faster than the comedian can decline photos.",
        "source_basis": ["tmz_chappelle_weaponized"],
        "scores": {
            "famous_face_or_public_recognition": 8,
            "public_conflict": 9,
            "ego_or_status_pressure": 7,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 2,
            "freshness": 8
        },
        "total": 58,
        "risk_notes": "Rejected: high sensitivity around trans politics and quote reuse."
    },
    {
        "rank": 5,
        "slug": "britney_dui_paparazzi_lane",
        "premise": "A pop icon's GPS is forced into a paparazzi DMV driving simulator where every lane marker is a camera flash.",
        "source_basis": ["tmz_britney_dui_charge"],
        "scores": {
            "famous_face_or_public_recognition": 10,
            "public_conflict": 8,
            "ego_or_status_pressure": 6,
            "humiliation_or_absurd_defense": 5,
            "brand_or_location_contrast": 7,
            "first_frame_visual_contradiction": 7,
            "taste_safety": 1,
            "freshness": 10
        },
        "total": 54,
        "risk_notes": "Rejected: fresh but too close to personal health/legal vulnerability."
    }
]

FIRST_FRAME_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 9:16 vertical.
subject: a fictional action-movie megastar archetype inspired by public red-carpet charisma, bald head, muscular build, cream trousers, crisp shirt, warm smile; do not make an exact photoreal celebrity likeness.
scene: Hollywood sidewalk immediately after a Walk of Fame ceremony, gold star shapes on the pavement are abstract and non-readable, velvet ropes and camera flashes recede in the background. A cream luxury SUV is pulled over at the curb.
comedic contradiction: a small calm traffic officer holds a jeweler-like window-tint meter up to the SUV glass as if presenting a major award. The action-star archetype stands beside the car with both hands politely visible, looking amused and mildly betrayed by the tint meter.
camera: cinematic 35mm vertical close-medium shot, officer and tint meter in foreground, huge celebrity silhouette and car behind, low angle for absurd scale contrast.
lighting and style: premium tabloid-cinema realism, late-afternoon Hollywood sun, clean shadows, subtle film grain, believable candid-photo energy, no fake news graphics, no police insignia closeups, no readable text, no real logos, no mugshot mood.
negative instructions: avoid implying arrest, DUI, violence, drugs, panic, official LAPD marks, real Walk of Fame names, captions, watermarks, distorted faces, oversaturation, excessive yellow in the photo."""

SHARED_CHOICES_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 16:9.
Create one SGFLIX Shared Choices director-bible board for Run 024, internal title 'Tint Meter Walk of Shame'. Include: fictional action-star character canon, calm traffic-officer archetype, hero props of window-tint meter, cream luxury SUV, velvet rope, flashbulb camera, abstract Hollywood sidewalk stars, traffic cone trophy; color palette swatches cream trouser / asphalt black / paparazzi white / gold star / police blue used sparingly; environment and set design for Hollywood curb outside Walk of Fame ceremony; floor plan and blocking; six storyboard panels with camera/lens/movement notes; lighting/mood/style notes; visual rules; production notes. Keep text minimal and mostly label-like. Do not use exact Dwayne Johnson likeness, readable celebrity names, real logos, real police insignia, defamatory claims, mugshot framing, or readable Walk of Fame star names."""


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
    img = Image.new("RGB", size, "#141414")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 54)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except Exception:
        title_font = body_font = None
    draw.rectangle([38, 38, size[0] - 38, size[1] - 38], outline="#f0c84b", width=6)
    draw.text((76, 82), title, fill="#f6f2e8", font=title_font)
    y = 178
    for line in subtitle.split("\n"):
        draw.text((76, y), line, fill="#e8e1d0", font=body_font)
        y += 44
    if board:
        colors = ["#e8d4b4", "#151515", "#f7f2e9", "#d5aa37", "#1e5a8a"]
        for idx, color in enumerate(colors):
            x = 82 + idx * 118
            draw.rectangle([x, size[1] - 164, x + 84, size[1] - 86], fill=color, outline="#f6f2e8")
        for i in range(6):
            x = 82 + (i % 3) * 462
            y = 334 + (i // 3) * 224
            draw.rectangle([x, y, x + 372, y + 154], outline="#1e5a8a", width=4)
            draw.text((x + 18, y + 18), f"PANEL {i + 1}", fill="#f6f2e8", font=body_font)
    else:
        draw.rectangle([640, 520, 820, 700], outline="#1e5a8a", width=6)
        draw.rectangle([250, 850, 790, 1010], outline="#f0c84b", width=6)
        draw.text((284, 882), "TINT METER\nAWARD MOMENT", fill="#f6f2e8", font=title_font)
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
            "RUN 024 IMAGE BLOCKED",
            "GPT image generation failed.\nPrompt preserved beside image.\nUse GPT Image 2 repair before public export.",
            board=("shared_choices" in str(path)),
        )
        return "blocked_placeholder_after_openai_error"


def main():
    for sub in [
        "research", "strategy", "frames/gpt_image_2", "storyboards/shared_choices", "qc",
        "chai", "scene_json", "handoffs", "captions", "distribution", "skool", "manifests",
    ]:
        (PKG / sub).mkdir(parents=True, exist_ok=True)

    prev = ROOT / "sgflix_runs" / "run_023_billie_finneas_rapunzel_booth" / "NEXT_STEP_REPORT_2026-05-02.md"
    prev.write_text(
        "# Next-Step Report - Run 023\n\n"
        "`run_023_billie_finneas_rapunzel_booth` remains blocked at GPT Image generation. "
        "Saved prompts and placeholder/QC files exist; rerun the first-frame and Shared Choices prompts through GPT Image 2 after billing/access is restored, then redo QC before any closed-tool video workflow.\n",
        encoding="utf-8",
    )

    write_text("research/last30days_report.md", f"""
# Last 30 Days Research Intake - Run {RUN}

Created: {NOW}

Research query/topic: fresh celebrity-status contradictions from April 2026 to May 2, 2026 with famous face, public friction, ego/status pressure, humiliation or absurd defense, and strong first-frame contradiction.

Intake method: current web/source scan first, then candidate scoring. No premise was selected from local images, old storyboards, nearby assets, or prior handoffs.

Shortlist:
- Dwayne Johnson traffic stop after Hollywood Walk of Fame event: reported April 30, updated May 1. The published stop reason was tinted windows, attributed by TMZ to law-enforcement sources. This has high famous-face signal, location contrast, and low-harm routine-stop absurdity.
- Kanye West SoFi lighting stop-down: reported April 2. Strong ego/quote vector, but less fresh and higher controversy load.
- Blake Lively Nickelodeon Universe visit amid Baldoni litigation: strong brand contrast, but active litigation makes satire taste risk high.
- Dave Chappelle photo-op weaponization comments: strong conflict, but trans-politics context is too sensitive for this factory pass.
- Britney Spears DUI charge: very fresh and famous, rejected because the health/legal vulnerability risk overwhelms the comedy.

Winner: `rock_tint_meter`.

Verified facts used:
- TMZ reported Dwayne Johnson was pulled over after a Hollywood Walk of Fame event and described the stop as a tinted-window traffic stop.
- Just Jared recapped the same broad traffic-stop context on May 1, 2026.

Unverified or avoided:
- No citation established whether a ticket was issued beyond the entertainment-press framing.
- No claim is made about impairment, arrest, hostility, or wrongdoing beyond the reported tinted-window stop.
""")
    write_json("research/sources.json", {"created_at": NOW, "sources": SOURCES})

    write_json("strategy/candidate_board.json", {"created_at": NOW, "selection_order": "research_first_then_scoring_then_package", "candidates": CANDIDATES})
    write_text("strategy/winner_decision.md", f"""
# Winner Decision - Run {RUN}

Selected premise: **Tint Meter Walk of Shame**.

Logline: A fictional action-star titan exits a Walk of Fame ceremony and is immediately humbled by the world's smallest window-tint meter, treated like a red-carpet award.

Score summary:
- Winner total: 70
- Runner-up: `kanye_sofi_light_court` at 66
- Main reason: the winner is fresher, visually cleaner, lower-risk, and has a stronger brand/location contradiction.

Decision: proceed with a still-image package and closed-tool handoff only. The joke should stay on scale contrast and celebrity infrastructure, not on criminality.
""")
    write_json("strategy/phase_minus_one_worthiness_audit.json", {
        "phase_minus_one_audit": {
            "source_target": "Dwayne Johnson tinted-window traffic stop after Walk of Fame event",
            "track_a_newsjack_velocity": {
                "active_trend_score": 8,
                "algorithmic_slipstream": "fresh entertainment/news item within 48 hours of source update",
                "polarization_factor": 3,
                "track_a_total": 11,
                "track_a_verdict": "PASS"
            },
            "track_b_archetype_resonance": {
                "iconography_strength": 9,
                "stereotype_rigidity": "High",
                "subversion_potential": 10,
                "track_b_total": 19,
                "track_b_verdict": "PASS"
            },
            "system_verdict": {
                "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                "primary_vector": "TRACK_B_EVERGREEN_ARCHETYPE",
                "urgency_class": "High",
                "strategic_directive": "Make the smallest civic object puncture the largest celebrity-body/status signal."
            }
        }
    })
    write_json("strategy/source_entropy_audit.json", {
        "source_entropy_audit": {
            "baseline_reality_check": "Routine traffic stop after celebrity event; source absurdity comes from status contrast, not danger.",
            "detected_anomalies": ["Walk of Fame glamour immediately followed by tinted-window enforcement", "Huge action-star presence reduced to paperwork-scale inspection"],
            "native_entropy_score": 4,
            "subject_self_awareness": "unaware",
            "comedic_vector_recommendation": "cringe_vector",
            "recommended_strategy": "micro_spotlight"
        }
    })
    write_json("strategy/humor_logic_bridge.json", {
        "bridge": {
            "setup": "An action-star archetype has just left Hollywood ceremonial grandeur.",
            "status_reversal": "A pocket-sized tint meter becomes more powerful than red-carpet flashbulbs.",
            "visual_rule": "Everything glamorous is large and soft; everything bureaucratic is tiny, sharp, and decisive.",
            "punchline": "The tint meter is staged like the true Walk of Fame honoree.",
            "do_not_cross": ["no DUI implication", "no arrest framing", "no real police insignia", "no defamatory escalation"]
        }
    })
    write_json("strategy/tribe_meta_score.json", {
        "tribe_meta_score": {
            "tribe_fit": 8,
            "shareability": 8,
            "duet_or_comment_prompt": 7,
            "caption_lore_potential": 8,
            "repeatable_franchise_value": 8,
            "notes": "Works as a reusable 'tiny civic object humbles giant celebrity status' format."
        }
    })
    write_json("strategy/risk_taste_score.json", {
        "risk_taste_score": {
            "defamation_risk": 2,
            "legal_sensitivity": 3,
            "identity_or_health_sensitivity": 1,
            "public_figure_likeness_risk": 5,
            "brand_logo_risk": 4,
            "overall_risk": "Medium-low",
            "guardrails": ["fictionalize likeness", "avoid official insignia", "avoid suggesting arrest or impairment", "keep public facts sourced"]
        }
    })
    write_text("strategy/franchise_decision.md", """
# Franchise Decision

Franchise lane: **Tiny Authority Object vs Giant Status Object**.

Repeatable template: celebrity exits a maximal status environment, then a tiny mundane tool becomes the judge. Future versions can use badge scanner, valet ticket, parking boot, TSA bin, elevator permit, or coat-check tag.

Verdict: keep as a franchise seed. It is modular, low-harm, and visually legible in the first second.
""")

    write_text("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write_text("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)
    first_mode = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board_mode = generate_image(SHARED_CHOICES_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")
    blocked = first_mode.startswith("blocked") or board_mode.startswith("blocked")

    shot = {
        "run_id": RUN,
        "shot_id": "shot_001",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "duration_seconds": 6,
        "concept": "The tint meter is treated like the true celebrity award.",
        "camera": "slow push from tint meter foreground to action-star archetype at curb",
        "motion": "camera flashes, polite hand gesture, officer raises meter, velvet rope sways",
        "no_video_generation": True,
        "workflow_spec": {"source_image": "frames/gpt_image_2/first_frame_v01.png", "render_lane": "manual_closed_tool_only_after_human_approval"}
    }
    write_json("scene_json/shot_001.json", shot)
    write_json("scene_json/shot_0001.json", {**shot, "shot_id": "shot_0001"})
    write_json("chai/chai_shot_specs.json", {
        "run_id": RUN,
        "specs": [{
            "shot_id": "shot_001",
            "subject": "fictional action-star archetype and calm traffic officer",
            "scene": "Hollywood curb after Walk of Fame ceremony",
            "motion": "tint meter rises into camera as celebrity status deflates politely",
            "spatial": "officer foreground, SUV midground, actor background, red-carpet residue behind",
            "camera": "vertical 35mm slow push, low-angle scale joke",
            "critique": "Must not read as arrest, DUI, or real LAPD footage.",
            "revision": "If too literal, make the tint meter more award-like and the actor less exact."
        }]
    })
    write_json("handoffs/closed_tool_handoff.json", {
        "run_id": RUN,
        "title": "Tint Meter Walk of Shame",
        "video_generation_permitted": False,
        "approved_still_paths": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"
        },
        "next_manual_step": "Only after human image approval, copy shot specs to a closed video tool manually; do not auto-start renders."
    })
    write_text("handoffs/grok_agent_prompt.md", """
# Grok Agent Prompt

Do not generate video automatically. Use the attached stills only as private visual reference.

Create a 6-second satirical clip plan where a fictional action-star archetype leaves a Hollywood ceremony and is politely stopped for a tinted-window inspection. Keep it routine, charming, and bureaucratically tiny. Do not imply arrest, DUI, danger, or hostile police behavior.
""")
    write_text("captions/instagram_caption.md", """
The real Walk of Fame star was the tint meter.

Routine stop. Maximum Hollywood scale. Tiny bureaucracy wins again.

#sgflix #satire #hollywood #therock #walkoffame #shortfilm #aivideo #behindthescenes
""")
    write_text("distribution/post_plan.md", """
# Post Plan

Primary surface: Instagram Reels / TikTok.

Hook text: "POV: you leave the Walk of Fame and meet the final boss of window tint."

Risk caption note: Mention only "reported tinted-window traffic stop"; avoid crime language.

Cutdown plan: 6s first-frame hold, 10s tint-meter award reveal, 15s mini case-study version.
""")
    write_text("skool/case_study.md", """
# Skool Case Study - Run 024

Lesson: status contrast can carry a premise without adding cruelty.

Source fact: a reported routine tinted-window traffic stop after a celebrity Walk of Fame event.

Factory move: convert the smallest factual object, the tint meter, into the most powerful object in the frame. This creates a reusable template for civic micro-tools puncturing celebrity macro-status.
""")
    write_json("manifests/asset_manifest.json", {
        "run_id": RUN,
        "created_at": NOW,
        "assets": [
            {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode},
            {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt"},
            {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode},
            {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt"}
        ]
    })
    if blocked:
        write_text("qc/IMAGE_GENERATION_BLOCKED_REPORT.md", f"""
# Image Generation Blocked Report

GPT image generation was attempted after research, scoring, winner selection, and package creation.

Results:
- `frames/gpt_image_2/first_frame_v01.png` via `{first_mode}`
- `storyboards/shared_choices/shared_choices_v01.png` via `{board_mode}`

The saved PNGs are local placeholders if generation failed. Prompts are preserved beside each asset. Do not route to video until the prompts are rerun through GPT Image 2 and QC passes.
""")
    write_text("qc/first_frame_v01_qc.md", f"""
# First Frame QC

Asset: `frames/gpt_image_2/first_frame_v01.png`

Generation mode: `{first_mode}`

Verdict: {"blocked placeholder only; rerun prompt through GPT Image 2 before production use" if first_mode.startswith("blocked") else "usable for human review; inspect face drift/text/logos before video handoff"}.

QC notes: Must preserve routine traffic-stop tone, no arrest implication, no readable logos or insignia.
""")
    write_text("qc/shared_choices_v01_qc.md", f"""
# Shared Choices QC

Asset: `storyboards/shared_choices/shared_choices_v01.png`

Generation mode: `{board_mode}`

Verdict: {"blocked placeholder only; rerun prompt through GPT Image 2 before production use" if board_mode.startswith("blocked") else "usable as private director-bible reference pending text inspection"}.

QC notes: Board must include character/props/palette/environment/blocking/panels/lighting/rules/production notes and keep text clean.
""")
    status = "blocked_at_gpt_image_generation" if blocked else "complete_for_factory_cycle"
    package = {
        "run_id": RUN,
        "slug": SLUG,
        "title": "Tint Meter Walk of Shame",
        "status": status,
        "created_at": NOW,
        "research_query": "fresh celebrity-status contradictions April 2026-May 2 2026",
        "selected_premise": CANDIDATES[0]["premise"],
        "winner_total": CANDIDATES[0]["total"],
        "video_generation": "not requested and not performed",
        "still_paths": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"
        },
        "next_human_action": "Review image outputs/placeholders; if blocked, rerun saved prompts through GPT Image 2 and redo QC before any closed-tool video workflow."
    }
    write_json(f"RUN_{RUN}_MASTER_PACKAGE.json", package)
    write_text("README.md", f"""
# RUN {RUN} MASTER PACKAGE - Tint Meter Walk of Shame

Status: {status}; no video footage generated.

Research query/topic: fresh celebrity-status contradictions April 2026-May 2 2026.

Selected premise: A fictional action-star titan exits a Walk of Fame ceremony and is humbled by a tiny window-tint meter.

Score summary: winner 70; runner-up Kanye SoFi light court 66; selected for freshness, visual contradiction, and lower taste risk.

Still-image paths:
- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

Exact next human action: Review the still outputs. If they are placeholders, rerun the saved prompts through GPT Image 2, replace the placeholder PNGs, and redo QC before any manual video-tool handoff.
""")
    write_text("FACTORY_RUN_STATUS.md", f"""
# Factory Run Status - Run {RUN}

Status: {status}

New run id: `run_{RUN}_{SLUG}`

Candidate board: created at `strategy/candidate_board.json`.

Selected premise: `Tint Meter Walk of Shame`.

Created files: all required package artifacts were written under `RUN_{RUN}_MASTER_PACKAGE`.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

Missing files: none from the required checklist.

Post-ready exports: none; stills require human review and, if blocked, GPT Image 2 repair.

QC failures: {"GPT Image API blocked; placeholder assets only." if blocked else "None known; human visual inspection still required."}

High-risk issues: public-figure likeness drift, accidental arrest/DUI implication, readable real logos or police insignia.

Exact next human action: Review still outputs; if blocked, rerun `first_frame_v01_prompt.md` and `shared_choices_v01_prompt.md` through GPT Image 2 and redo QC.
""")


if __name__ == "__main__":
    main()
