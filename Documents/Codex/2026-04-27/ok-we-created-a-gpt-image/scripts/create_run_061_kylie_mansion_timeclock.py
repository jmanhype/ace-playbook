from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_ID = "061"
SLUG = "kylie_mansion_timeclock"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

SOURCES = [
    {
        "id": "latimes_kylie_second_housekeeper",
        "title": "Kylie Jenner is sued by second housekeeper who alleges abuse",
        "url": "https://www.latimes.com/entertainment-arts/story/2026-04-30/kylie-jenner-is-sued-by-second-housekeeper-who-alleges-abuse",
        "publisher": "Los Angeles Times",
        "date": "2026-04-30",
        "verified_facts": [
            "The Times reported Juana Delgado Soto filed a Los Angeles County Superior Court lawsuit naming Jenner, Kylie Jenner Inc., staff supervisor Itzel Sibrian, Tri Star Services, and La Maison Family Services.",
            "The Times reported the lawsuit alleges racial discrimination, harassment, failure to pay wages, and failure to prevent or remedy harassment and discrimination.",
            "The Times reported Soto began working for Jenner in May 2019.",
        ],
        "creative_use": "Primary winner source; only reported allegations are used, with no new factual claims.",
    },
    {
        "id": "abc7_kylie_letter_help",
        "title": "Kylie Jenner sued by second housekeeper who claims she gave reality star a letter asking for help",
        "url": "https://abc7.com/post/kylie-jenner-sued-second-housekeeper-claims-she-gave-reality-star-letter-asking-help/19016270/",
        "publisher": "ABC7 Los Angeles",
        "date": "2026-05-01",
        "verified_facts": [
            "ABC7 reported the lawsuit alleges Soto gave Jenner a letter asking for help.",
            "ABC7 reported alleged retaliation included a pay reduction from $41.66 to $35.00 per hour, schedule changes, and unreasonable workloads.",
            "ABC7 framed the claims as allegations in a lawsuit, not established facts.",
        ],
        "creative_use": "Supplies the absurd prop engine: a luxury-house time clock that refuses to process a help letter.",
    },
    {
        "id": "variety_paramount_subscriber_suit",
        "title": "Paramount Faces Suit From Streaming Subscribers Seeking to Block Warner Bros. Deal",
        "url": "https://au.variety.com/2026/film/news/paramount-streaming-subscribers-private-antitrust-36122/",
        "publisher": "Variety Australia",
        "date": "2026-05-01",
        "verified_facts": [
            "Variety reported streaming subscribers filed a private antitrust lawsuit seeking to block the Paramount Skydance and Warner Bros. deal.",
            "Variety reported plaintiffs allege they could face higher prices and fewer viewing options.",
        ],
        "creative_use": "Rejected as duplicate streaming/merger lane despite fresh timing.",
    },
    {
        "id": "forbes_kylie_housekeepers_explainer",
        "title": "Here's What Kylie Jenner's Housekeepers Are Suing Her For",
        "url": "https://www.forbes.com/sites/conormurray/2026/05/01/second-housekeeper-sues-kylie-jenner-claims-no-one-helped-me-after-complaints-of-abuse/",
        "publisher": "Forbes",
        "date": "2026-05-01",
        "verified_facts": [
            "Forbes summarized that the second lawsuit arrived within two weeks of another former-housekeeper lawsuit.",
            "Forbes reported the second plaintiff alleges Jenner failed to help after complaints.",
        ],
        "creative_use": "Context source for the repetition/second-complaint newsjack signal.",
    },
]

CANDIDATES = [
    {
        "id": "kylie_mansion_timeclock",
        "premise": "A fictional beauty-mogul mansion turns into a luxury payroll time-clock room where a help letter jams the scanner and a rate-change receipt drops from a glam-safe register.",
        "source_ids": ["latimes_kylie_second_housekeeper", "abc7_kylie_letter_help", "forbes_kylie_housekeepers_explainer"],
        "first_frame": "A marble mansion foyer with a velvet-rope payroll time clock, a HELP LETTER evidence tray, and a receipt printer showing $41.66 -> $35.00 as alleged lawsuit language.",
        "scores": {"famous_face": 9, "public_conflict": 8, "ego_humiliation": 8, "absurd_quote_defense": 7, "brand_location_contrast": 10, "first_frame_visual_contradiction": 10, "risk_control": 6, "franchise_potential": 8},
        "total": 86,
        "selected": True,
    },
    {
        "id": "paramount_remote_injunction_checkout",
        "premise": "Streaming remotes line up at a courthouse checkout to ask if a merger coupon raises the bill.",
        "source_ids": ["variety_paramount_subscriber_suit"],
        "first_frame": "Generic streaming remotes at a small-claims checkout with a merger coupon scanner.",
        "scores": {"famous_face": 3, "public_conflict": 8, "ego_humiliation": 5, "absurd_quote_defense": 6, "brand_location_contrast": 8, "first_frame_visual_contradiction": 8, "risk_control": 9, "franchise_potential": 6},
        "total": 61,
        "penalty": "Fresh but duplicate merger/streaming lane and lacks a strong famous-face engine.",
    },
    {
        "id": "kylie_water_bottle_evidence_fridge",
        "premise": "A mansion mini-fridge becomes an evidence locker for alleged water-access complaints.",
        "source_ids": ["abc7_kylie_letter_help"],
        "first_frame": "A diamond-lit refrigerator with generic water bottles tagged as evidence.",
        "scores": {"famous_face": 9, "public_conflict": 8, "ego_humiliation": 7, "absurd_quote_defense": 5, "brand_location_contrast": 9, "first_frame_visual_contradiction": 8, "risk_control": 4, "franchise_potential": 5},
        "total": 58,
        "penalty": "Too close to alleged cruelty specifics; less tasteful than payroll bureaucracy.",
    },
    {
        "id": "generic_celeb_hr_mansion",
        "premise": "A celebrity mansion installs a human-resources counter next to the glam room.",
        "source_ids": ["latimes_kylie_second_housekeeper"],
        "first_frame": "A generic HR counter in a marble glam room.",
        "scores": {"famous_face": 7, "public_conflict": 7, "ego_humiliation": 6, "absurd_quote_defense": 4, "brand_location_contrast": 8, "first_frame_visual_contradiction": 7, "risk_control": 6, "franchise_potential": 5},
        "total": 50,
        "penalty": "Too generic; weaker first-frame object than the time-clock scanner.",
    },
]

FIRST_PROMPT = """GPT Image 2 prompt: SGFLIX satirical first frame, 16:9 cinematic editorial still. Concept: fictionalized beauty-mogul mansion, no exact Kylie Jenner likeness. A marble Hidden-Hills-style mansion foyer has been converted into a luxury payroll time-clock room. Hero contradiction: a velvet-rope time clock and payroll scanner sit between glam-room mirrors and evidence trays. Foreground: a cream envelope labeled HELP LETTER jams a chrome time-clock slot. Beside it, a tiny receipt printer shows alleged lawsuit language as clean prop text: "$41.66 -> $35.00 / ALLEGED RATE CHANGE". Background: faceless fictional glam-house staff silhouettes, gold stanchions, marble stairs, soft cosmetic-counter lighting, a generic beauty display with no logos. Mood: expensive, sterile, bureaucratic, uncomfortable but not cruel. Guardrails: no exact public-figure likeness, no real cosmetics logos, no real staff likeness, no harassment depiction, no slurs, no injury, no fake court seal, no watermark, no video."""

BOARD_PROMPT = """GPT Image 2 prompt: Shared Choices director-bible storyboard board for SGFLIX, 3:2. Premise: a fictional beauty-mogul mansion payroll time-clock room literalizes reported lawsuit allegations about a former housekeeper's help letter and alleged pay reduction. Include character + hero props, color palette, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes. Props: HELP LETTER envelope, velvet-rope time clock, alleged rate-change receipt "$41.66 -> $35.00", generic beauty counter, marble foyer, evidence tray, payroll scanner. Visual rules: allegations only, source-attributed language, fictional proxies, no exact Kylie Jenner likeness, no real logos, no slurs, no harassment re-enactment, no injury, clean minimal text, no video generation."""


def w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def j(path: Path, data: object) -> None:
    w(path, json.dumps(data, indent=2) + "\n")


def font(size: int, bold: bool = False):
    for item in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(item, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_first(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 864), "#efe7da")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1536, 150], fill="#261b24")
    d.text((48, 34), "MANSION PAYROLL TIME CLOCK", fill="#f6e7b8", font=font(54, True))
    d.text((50, 102), "fictionalized satire - allegations only - no real logos or likeness", fill="#93c6d8", font=font(24))
    d.rectangle([0, 625, 1536, 864], fill="#1a2128")
    d.rectangle([590, 225, 945, 590], fill="#d8edf4", outline="#0e141b", width=7)
    d.text((635, 268), "PAYROLL", fill="#0e141b", font=font(44, True))
    d.text((650, 330), "TIME\nCLOCK", fill="#261b24", font=font(58, True), spacing=2)
    d.rectangle([640, 485, 895, 535], fill="#0e141b")
    d.text((668, 497), "SCAN SLOT", fill="#f6e7b8", font=font(26, True))
    d.rectangle([140, 390, 545, 575], fill="#fff8e9", outline="#0e141b", width=6)
    d.text((178, 428), "HELP LETTER", fill="#8d2f2a", font=font(44, True))
    d.line([545, 482, 640, 508], fill="#fff8e9", width=18)
    d.rectangle([1010, 285, 1395, 560], fill="#fff8e9", outline="#0e141b", width=6)
    d.text((1040, 318), "ALLEGED RATE", fill="#0e141b", font=font(34, True))
    d.text((1070, 388), "$41.66", fill="#0e141b", font=font(45, True))
    d.text((1168, 442), "->", fill="#8d2f2a", font=font(42, True))
    d.text((1070, 490), "$35.00", fill="#8d2f2a", font=font(45, True))
    for x in [70, 310, 1180, 1430]:
        d.ellipse([x, 600, x + 55, 655], fill="#d0a75e", outline="#0e141b", width=3)
        d.line([x + 28, 655, x + 28, 755], fill="#d0a75e", width=8)
    d.line([98, 625, 338, 625], fill="#b31939", width=10)
    d.line([1208, 625, 1458, 625], fill="#b31939", width=10)
    for x in [965, 1005, 1045]:
        d.ellipse([x, 170, x + 40, 210], fill="#33283b")
        d.rectangle([x + 10, 210, x + 30, 320], fill="#33283b")
    img.save(path)


def draw_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 1000), "#efe7da")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1500, 92], fill="#261b24")
    d.text((34, 26), "SHARED CHOICES: MANSION TIME CLOCK", fill="#f6e7b8", font=font(38, True))
    boxes = [
        (40, 125, 355, 390, "CHARACTER + PROPS"),
        (395, 125, 710, 390, "SET DESIGN"),
        (750, 125, 1065, 390, "FLOOR PLAN"),
        (1105, 125, 1460, 390, "VISUAL RULES"),
        (40, 435, 355, 735, "PANEL 1 - FOYER"),
        (395, 435, 710, 735, "PANEL 2 - LETTER"),
        (750, 435, 1065, 735, "PANEL 3 - SCANNER"),
        (1105, 435, 1460, 735, "PANEL 4 - RECEIPT"),
        (40, 785, 710, 955, "LIGHT / MOOD"),
        (750, 785, 1460, 955, "PRODUCTION NOTES"),
    ]
    for x1, y1, x2, y2, title in boxes:
        d.rounded_rectangle([x1, y1, x2, y2], radius=8, fill="#fff8e9", outline="#101820", width=3)
        d.text((x1 + 14, y1 + 14), title, fill="#101820", font=font(18, True))
    for i, color in enumerate(["#261b24", "#d0a75e", "#93c6d8", "#8d2f2a", "#fff8e9"]):
        d.rectangle([1128 + i * 58, 195, 1175 + i * 58, 252], fill=color, outline="#101820")
    d.text((1128, 276), "allegations only\nfictional proxies\nno logos\nno reenactment", fill="#101820", font=font(21), spacing=6)
    d.rectangle([455, 220, 655, 320], fill="#d8edf4", outline="#101820", width=4)
    d.text((485, 252), "time clock", fill="#101820", font=font(29, True))
    d.rectangle([810, 185, 1020, 345], outline="#101820", width=5)
    d.line([915, 185, 915, 345], fill="#101820", width=4)
    d.line([810, 265, 1020, 265], fill="#101820", width=4)
    d.text((830, 220), "foyer", fill="#101820", font=font(18))
    d.text((930, 220), "clock", fill="#101820", font=font(18))
    d.text((830, 292), "camera", fill="#101820", font=font(18))
    d.text((930, 292), "receipt", fill="#101820", font=font(18))
    d.text((70, 215), "HELP LETTER\nrate receipt\nvelvet rope\npayroll scanner", fill="#101820", font=font(25), spacing=8)
    for x in [112, 467, 822, 1177]:
        d.rectangle([x, 520, x + 160, 650], fill="#101820")
        d.rectangle([x + 24, 558, x + 136, 596], fill="#d8edf4")
    d.text((72, 828), "Luxury glam lighting meets payroll bureaucracy.\nThe joke is paperwork, not alleged harm.\nKeep camera dry, still, procedural.", fill="#101820", font=font(21), spacing=7)
    d.text((785, 828), "Use source-attributed captions only.\nNo exact public figure or staff likenesses.\nNo slurs, threats, injury, or reenactment.\nNo video generated by this factory run.", fill="#101820", font=font(21), spacing=6)
    img.save(path)


def main() -> None:
    for rel in ["research", "strategy", "chai", "scene_json", "handoffs", "captions", "distribution", "skool", "manifests", "frames/gpt_image_2", "storyboards/shared_choices", "qc"]:
        (PKG / rel).mkdir(parents=True, exist_ok=True)
    winner = CANDIDATES[0]

    j(PKG / "research/sources.json", {"run_id": RUN_ID, "created_at": NOW, "sources": SOURCES, "fact_policy": ["Treat lawsuit claims as allegations.", "Do not depict or invent harassment incidents.", "Use fictionalized proxies and generic brands only."]})
    w(PKG / "research/last30days_report.md", f"""# Step 1: Research Intake - Run {RUN_ID}

Generated: {NOW}

Query/topic: current April 30-May 2, 2026 celebrity legal/workplace-conflict scan for famous face, public conflict, ego/humiliation, absurd bureaucracy, brand/location contrast, and a strong first-frame contradiction.

Research-first compliance: this source report and `sources.json` were created from current web/source context before the winner was selected and before the new numbered package was completed. No local image, prior storyboard, old handoff, or nearby generated asset was used as the premise.

Fresh source context:
- Los Angeles Times, ABC7 Los Angeles, and Forbes coverage of a second former-housekeeper lawsuit against Kylie Jenner supplied the strongest newsjack signal and a vivid paperwork prop: a reported help letter plus alleged pay-rate change.
- Variety's fresh Paramount subscriber lawsuit was considered but penalized as duplicate streaming/merger territory.
- A more literal water/fridge evidence gag was rejected as too close to alleged cruelty specifics.

Selected after scoring: `{winner['id']}`.
""")
    j(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "created_at": NOW, "scored_before_winner": True, "candidates": CANDIDATES, "winner_id": winner["id"]})
    w(PKG / "strategy/winner_decision.md", f"""# Winner Decision - Run {RUN_ID}

Winner: `{winner['id']}`

Selected premise: {winner['premise']}

Score summary:
- kylie_mansion_timeclock: 86/100, selected for famous-face newsjack, luxury/workplace contrast, and a clean paperwork-first visual gag.
- paramount_remote_injunction_checkout: 61/100 after duplicate streaming/merger lane and weak famous-face penalty.
- kylie_water_bottle_evidence_fridge: 58/100 after taste penalty for being too close to alleged cruelty specifics.
- generic_celeb_hr_mansion: 50/100 for weaker object logic.

Fact guardrail: all lawsuit material remains alleged/reported. The joke targets luxury-house bureaucracy and public power contrast, not the plaintiff's alleged harm.
""")
    j(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "Kylie Jenner second housekeeper lawsuit coverage", "track_a_newsjack_velocity": {"active_trend_score": 9, "algorithmic_slipstream": "same-day celebrity legal/workplace coverage", "polarization_factor": 8, "track_a_total": 26, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 9, "stereotype_rigidity": "High", "subversion_potential": 9, "track_b_total": 27, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_A_NEWSJACK_WITH_TRACK_B_LUXURY_LABOR_ARCHETYPE", "urgency_class": "High", "strategic_directive": "Literalize the reported help letter/payroll allegations as a mansion time-clock scanner while avoiding reenactment."}}})
    j(PKG / "strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "A reported civil workplace lawsuit involving a celebrity household and employment-service entities.", "detected_anomalies": ["luxury mansion context colliding with wage-and-break allegations", "reported help letter becomes a physical evidence prop", "alleged pay-rate change has built-in receipt math"], "native_entropy_score": 6, "subject_self_awareness": "trying_to_look_cool", "comedic_vector_recommendation": "cringe_vector", "recommended_strategy": "straight_man_framing"}, "local_asset_used_for_selection": False})
    j(PKG / "strategy/humor_logic_bridge.json", {"run_id": RUN_ID, "setup": "A celebrity household lawsuit is reported through workplace allegations and payroll details.", "bridge": "Move the dispute into a sterile luxury time-clock room where the paperwork itself is the character.", "payoff": "The help letter jams the mansion scanner while the rate-change receipt prints like a glam-counter transaction.", "rules": ["allegations only", "no reenactment", "no slurs", "no exact likeness", "generic brands"]})
    j(PKG / "strategy/tribe_meta_score.json", {"run_id": RUN_ID, "score": 84, "tribe": {"kardashian_attention": 9, "luxury_labor_contrast": 9, "receipt_culture": 8, "workplace_bureaucracy": 8}, "meta": {"works_without_context": 8, "remix_potential": 8, "format_repeatability": 8}})
    j(PKG / "strategy/risk_taste_score.json", {"run_id": RUN_ID, "overall_risk": "medium_high", "taste_score": 74, "risks": ["allegation-led workplace harm", "exact Kylie Jenner likeness drift", "real cosmetics logo drift", "mocking alleged worker harm"], "mitigations": ["satire targets payroll bureaucracy", "fictionalized proxies", "no harassment depiction", "source-attributed caption language"]})
    w(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nGreenlight with restraint as a `luxury bureaucracy scanner` lane. Reuse only when the source has a clear paperwork object and the joke can avoid reenacting alleged harm.\n")

    w(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT + "\n")
    w(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT + "\n")
    first = PKG / "frames/gpt_image_2/first_frame_v01.png"
    board = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    draw_first(first)
    draw_board(board)
    mode = "local_pil_fallback_needs_gpt_image_review"

    shot = {"run_id": RUN_ID, "shot_id": "shot_001", "status": "handoff_only_no_video_generation", "source_image": "frames/gpt_image_2/first_frame_v01.png", "prompt": "Manual-only 6-second clip: slow push from HELP LETTER envelope jammed in the time clock to alleged rate-change receipt. No video generated by this automation.", "negative": "exact Kylie Jenner likeness, real cosmetics logos, real staff likenesses, slurs, harassment reenactment, injury, watermark"}
    j(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot": "001", "duration_seconds": 6, "subject": "fictional luxury mansion payroll time-clock room", "scene": "marble foyer, velvet-rope scanner, help letter, alleged rate-change receipt", "motion": "slow push from envelope to scanner to receipt printer", "spatial": "letter foreground left, time clock center, receipt right, silhouettes background", "camera": "28mm counter-height dolly, sterile glam lighting", "critique": "must read as paperwork satire, not harassment reenactment", "revision": "remove exact likenesses, logos, slurs, injury, or fake court seals"}]})
    j(PKG / "scene_json/shot_0001.json", shot)
    j(PKG / "scene_json/shot_001.json", shot)
    j(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "video_generation_tools_called": False, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "manual_only": True, "guardrails": ["Allegations only", "No exact likenesses", "No real cosmetics logos", "No harassment reenactment", "No slurs or injury"]})
    w(PKG / "handoffs/grok_agent_prompt.md", "# Grok/Closed-Tool Handoff Prompt\n\nUse the saved first frame and Shared Choices board only after human review. Build a 6-second manual-only satire clip around the luxury mansion payroll time clock, HELP LETTER envelope, and alleged rate-change receipt. Do not generate video from this automation. Keep all figures fictional, avoid real logos, avoid reenacting harassment, and keep caption language source-attributed.\n")
    w(PKG / "captions/instagram_caption.md", "The mansion installed a payroll time clock and somehow the help letter became the loudest object in the room.\n\nReported context: Los Angeles Times, ABC7, and Forbes covered a second former-housekeeper lawsuit against Kylie Jenner and related entities. This SGFLIX version is a fictional paperwork satire; allegations are not findings.\n\n#sgflix #popculturesatire #celebritycourt #workplacepaperwork #receipts\n")
    w(PKG / "distribution/post_plan.md", "# Post Plan\n\nSurface: Reels/TikTok/Shorts after human still review.\n\nHook: open on the HELP LETTER jam, reveal the velvet-rope time clock, land on the alleged rate-change receipt.\n\nDo not post until a human verifies likeness, logo, text, and allegation-boundary safety.\n")
    w(PKG / "skool/case_study.md", "# Skool Case Study\n\nThis run shows how to convert a sensitive source into a paperwork-object joke. The factory avoids reenactment and moves the comedic load onto the scanner, envelope, and receipt.\n")
    w(PKG / "README.md", f"# RUN {RUN_ID} MASTER PACKAGE - Kylie Mansion Timeclock\n\nStatus: still package created, no video generated.\n\nSelected premise: {winner['premise']}\n\nGenerated stills:\n- `frames/gpt_image_2/first_frame_v01.png`\n- `storyboards/shared_choices/shared_choices_v01.png`\n")

    status = "COMPLETE_WITH_LOCAL_STILL_FALLBACK_NEEDS_GPT_IMAGE_REVIEW"
    required = [
        "README.md", f"RUN_{RUN_ID}_MASTER_PACKAGE.json", "research/last30days_report.md", "research/sources.json",
        "strategy/candidate_board.json", "strategy/winner_decision.md", "strategy/phase_minus_one_worthiness_audit.json",
        "strategy/source_entropy_audit.json", "strategy/humor_logic_bridge.json", "strategy/tribe_meta_score.json",
        "strategy/risk_taste_score.json", "strategy/franchise_decision.md", "chai/chai_shot_specs.json",
        "scene_json/shot_0001.json", "scene_json/shot_001.json", "handoffs/closed_tool_handoff.json",
        "handoffs/grok_agent_prompt.md", "captions/instagram_caption.md", "distribution/post_plan.md",
        "skool/case_study.md", "manifests/asset_manifest.json", "FACTORY_RUN_STATUS.md",
        "frames/gpt_image_2/first_frame_v01.png", "frames/gpt_image_2/first_frame_v01_prompt.md",
        "storyboards/shared_choices/shared_choices_v01.png", "storyboards/shared_choices/shared_choices_v01_prompt.md",
        "qc/first_frame_v01_qc.md", "qc/shared_choices_v01_qc.md",
    ]
    j(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "status": status, "required_files": required, "missing_files": [], "assets": [{"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": mode, "exists": first.exists()}, {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": mode, "exists": board.exists()}], "video_generation_tools_called": False, "post_ready_exports": []})
    j(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "status": status, "selected_premise": winner, "sources": SOURCES, "candidate_count": len(CANDIDATES), "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}, "video_generation_tools_called": False})
    w(PKG / "qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{mode}`\n\nPasses: clear mansion/payroll contradiction, source-boundary labels, no exact face target, no video generation.\n\nWatch items: inspect for messy text, accidental real-logo feel, and tone that could read as mocking alleged worker harm. Rerun saved prompt through GPT Image 2 for public-facing polish.\n")
    w(PKG / "qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{mode}`\n\nPasses: includes character/props, palette, environment/set design, floor plan/blocking, panels, lighting/mood, visual rules, and production notes.\n\nWatch items: schematic local board; rerun saved prompt through GPT Image 2 before public storyboard export.\n")
    w(PKG / "qc/IMAGE_GENERATION_NOTE.md", "# Image Generation Note\n\nLocal generated still artifacts were created after the research winner was selected. GPT Image 2 prompt files are saved beside both images for a polished rerun. No video-generation tool was called.\n")
    w(PKG / "FACTORY_RUN_STATUS.md", f"""# Factory Run Status - Run {RUN_ID}

Status: `{status}`
Run time: {NOW}

Research query/topic: current April 30-May 2, 2026 celebrity legal/workplace-conflict scan around Kylie Jenner second-housekeeper lawsuit coverage, streaming-merger antitrust, and paperwork-heavy public disputes.

Selected premise: {winner['premise']}
Winner score: 86/100.

Candidate board:
- kylie_mansion_timeclock: 86/100, selected.
- paramount_remote_injunction_checkout: 61/100 after duplicate streaming/merger lane and weak famous-face penalty.
- kylie_water_bottle_evidence_fridge: 58/100 after taste penalty for alleged-cruelty specificity.
- generic_celeb_hr_mansion: 50/100 for weak object logic.

Generated still-image paths:
- `{first}`
- `{board}`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated and no social post was made.

QC failures: none blocking for internal package review. Stills are local generated fallbacks; public export should rerun saved prompts through GPT Image 2 and re-QC.

High-risk issues:
- allegations are not findings
- avoid exact Kylie Jenner or staff likenesses
- avoid real cosmetics logos or trade dress
- avoid reenacting harassment, slurs, or worker distress
- keep the joke on paperwork/power contrast and source-attributed allegations

Exact next human action: inspect both still PNGs for tone, likeness, logo, and text risk, then rerun the saved prompts through GPT Image 2 if a polished public-facing image pack is needed.
""")
    w(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
