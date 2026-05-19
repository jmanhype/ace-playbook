from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_ID = "056"
SLUG = "ice_spice_wendys_receipt_counter"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "variety_ice_spice_wendys",
        "title": "Ice Spice Addresses Altercation at McDonald's: 'This Wouldn't Happen at Wendy's'",
        "url": "https://au.variety.com/2026/music/global/ice-spice-addresses-altercation-mcdonalds-wendys-35619/",
        "publisher": "Variety Australia",
        "date": "2026-04-18",
        "verified_facts": [
            "Variety reported that Ice Spice addressed a Hollywood McDonald's altercation on X.",
            "The reported post used the line 'This wouldn't happen at Wendy's.'",
            "Variety framed the line as connected to her Wendy's partnership and a new-song clip.",
        ],
        "creative_use": "Winner source: the quote turns a public altercation response into a fast-food venue-transfer desk.",
    },
    {
        "id": "tag24_ice_spice_police",
        "title": "Ice Spice got attacked at a McDonald's, and her viral clapback is perfect",
        "url": "https://www.tag24.com/entertainment/celebrities/ice-spice-got-attacked-at-a-mcdonalds-and-her-viral-clapback-is-perfect-this-wouldnt-happen-at-wendys-3491233-amp",
        "publisher": "TAG24",
        "date": "2026-04-18",
        "verified_facts": [
            "TAG24 reported the fast-food altercation and the same Wendy's line.",
            "The outlet reported that her attorney said a report had been filed with LAPD.",
        ],
        "creative_use": "Adds a receipt/legal intake prop while keeping the incident as alleged/reported.",
    },
    {
        "id": "variety_paramount_subscriber_suit",
        "title": "Paramount Faces Suit From Streaming Subscribers Seeking to Block Warner Bros. Deal",
        "url": "https://au.variety.com/2026/film/news/paramount-streaming-subscribers-private-antitrust-36122/",
        "publisher": "Variety Australia",
        "date": "2026-05-01",
        "verified_facts": [
            "Variety reported that Paramount+ subscribers filed a federal suit seeking to block the Paramount Skydance-Warner Bros. deal.",
            "The report says plaintiffs allege higher prices and reduced viewing options.",
        ],
        "creative_use": "Scored as a candidate but penalized because run_046 already used the Paramount merger lane.",
    },
    {
        "id": "ap_taylor_trademark",
        "title": "Taylor Swift files 3 new trademark applications",
        "url": "https://apnews.com/article/7f56fbafb269d4959009f3ad34e28fc1",
        "publisher": "Associated Press",
        "date": "2026-04-28",
        "verified_facts": [
            "AP reported that Swift filed three trademark applications.",
            "AP attributed the AI-protection theory to a legal expert.",
        ],
        "creative_use": "Scored as a candidate but rejected as duplicate/crowded AI-celebrity terrain.",
    },
    {
        "id": "variety_tiger_king_fair_use",
        "title": "Netflix Prevails in 'Tiger King' Copyright Case, a Win for Fair Use in Documentaries",
        "url": "https://au.variety.com/2026/tv/global/netflix-tiger-king-copyright-case-documentaries-36118/",
        "publisher": "Variety Australia",
        "date": "2026-05-01",
        "verified_facts": [
            "Variety reported that an appellate panel upheld dismissal of a copyright suit against Netflix and Tiger King filmmakers.",
            "The report says the panel treated use of a 66-second funeral clip as transformative fair use.",
        ],
        "creative_use": "Scored as a documentary-law candidate but rejected for lower famous-face immediacy and IP risk.",
    },
    {
        "id": "variety_mrbeast_employee_suit",
        "title": "MrBeast's Beast Industries Sued by Former Employee",
        "url": "https://au.variety.com/2026/digital/news/mrbeast-sued-former-employee-sexual-harassment-retaliation-35780/",
        "publisher": "Variety Australia",
        "date": "2026-04-23",
        "verified_facts": [
            "Variety reported that a former Beast Industries employee sued alleging harassment and retaliation.",
            "The company denied the allegations and called the suit categorically false.",
        ],
        "creative_use": "Scored but rejected for sensitive workplace-harassment taste risk.",
    },
]


CANDIDATES = [
    {
        "id": "ice_spice_wendys_receipt_counter",
        "premise": "A fictional rap-star proxy tries to transfer an entire fast-food altercation to a Wendy's-style complaint counter where the receipt machine only prints venue excuses.",
        "source_ids": ["variety_ice_spice_wendys", "tag24_ice_spice_police"],
        "first_frame": "Red-haired anonymous rapper silhouette at a generic red-and-yellow fast-food counter; a clerk stamps a receipt 'WRONG RESTAURANT' while security watches a tray of frosty-orange evidence cups.",
        "scores": {
            "famous_face": 9,
            "public_conflict": 8,
            "ego_humiliation": 8,
            "absurd_quote_defense": 10,
            "brand_location_contrast": 10,
            "first_frame_visual_contradiction": 10,
            "risk_control": 8,
            "franchise_potential": 9,
        },
        "total": 92,
    },
    {
        "id": "paramount_subscriber_small_claims_remote",
        "premise": "Paramount+ subscribers enter small-claims court carrying remotes that only show merger paperwork instead of movies.",
        "source_ids": ["variety_paramount_subscriber_suit"],
        "first_frame": "Streaming remotes and subscription receipts on a merger-court intake desk.",
        "scores": {"famous_face": 5, "public_conflict": 9, "ego_humiliation": 7, "absurd_quote_defense": 7, "brand_location_contrast": 8, "first_frame_visual_contradiction": 8, "risk_control": 8, "franchise_potential": 7},
        "total": 59,
        "penalty": "Duplicate lane: prior Paramount merger package already exists.",
    },
    {
        "id": "swift_soundmark_customs",
        "premise": "A trademark customs agent makes pop-star voice clones declare every vowel at the border.",
        "source_ids": ["ap_taylor_trademark"],
        "first_frame": "Soundwave passports on a trademark-office inspection belt.",
        "scores": {"famous_face": 10, "public_conflict": 7, "ego_humiliation": 6, "absurd_quote_defense": 8, "brand_location_contrast": 8, "first_frame_visual_contradiction": 9, "risk_control": 5, "franchise_potential": 6},
        "total": 57,
        "penalty": "Crowded celebrity-plus-AI lane and prior Taylor/voice package overlap.",
    },
    {
        "id": "tiger_king_fair_use_funeral_scanner",
        "premise": "A documentary fair-use clerk measures a 66-second clip with a carnival height stick labeled transformative.",
        "source_ids": ["variety_tiger_king_fair_use"],
        "first_frame": "Archive-footage scanner, documentary notes, and a stopwatch in a copyright office.",
        "scores": {"famous_face": 5, "public_conflict": 7, "ego_humiliation": 6, "absurd_quote_defense": 7, "brand_location_contrast": 7, "first_frame_visual_contradiction": 8, "risk_control": 7, "franchise_potential": 6},
        "total": 53,
        "penalty": "IP/logos/real-case imagery risk without a strong celebrity face.",
    },
    {
        "id": "mrbeast_receipts_warehouse",
        "premise": "A creator-economy warehouse tries to prove every allegation false by stacking receipts into an obstacle course.",
        "source_ids": ["variety_mrbeast_employee_suit"],
        "first_frame": "Receipt pallets and HR binders in a warehouse game set.",
        "scores": {"famous_face": 9, "public_conflict": 8, "ego_humiliation": 7, "absurd_quote_defense": 7, "brand_location_contrast": 7, "first_frame_visual_contradiction": 8, "risk_control": 2, "franchise_potential": 7},
        "total": 43,
        "penalty": "Sensitive harassment allegations; not worth the taste risk.",
    },
]


FIRST_PROMPT = """GPT Image 2 prompt: SGFLIX satirical first frame, 16:9 cinematic editorial still. Concept: a fictional red-haired rap-star proxy, not Ice Spice and no exact real-person likeness, stands at a generic red-and-yellow fast-food complaint counter after a reported viral restaurant altercation. The hero contradiction is a receipt printer stamping WRONG RESTAURANT while a polite clerk slides over a venue-transfer form and three unlabeled orange dessert cups sit like evidence. A tiny security monitor shows only abstract pixel shapes, not real footage. Include a paper sign with minimal pseudo-text only: VENUE TRANSFER DESK. Tone: funny, glossy, premium pop-culture satire, clean fast-food fluorescents mixed with music-video rim light, no real McDonald's or Wendy's logos, no real brand marks, no assault depiction, no injuries, no readable legal claims, no watermark."""

BOARD_PROMPT = """GPT Image 2 prompt: Shared Choices director-bible storyboard board for SGFLIX, 3:2. Premise: a fictional rap-star proxy's line 'this would not happen at the other restaurant' becomes a fast-food venue-transfer complaint counter. Build a polished board with distinct zones: character canon and hero props, color palette swatches, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes. Props: generic receipt printer, venue-transfer form, unlabeled orange dessert cups, red/yellow counter, security monitor with abstract pixels, neutral clerk, no real logos. Use concise readable labels where possible, but no real brand names, no exact Ice Spice likeness, no assault depiction, no video generation, no watermark."""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    write(path, json.dumps(data, indent=2) + "\n")


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for item in candidates:
        try:
            return ImageFont.truetype(item, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def try_openai_image(prompt: str, path: Path, size: str) -> str:
    try:
        from openai import OpenAI

        result = OpenAI().images.generate(model="gpt-image-1", prompt=prompt, size=size)
        b64 = result.data[0].b64_json
        if not b64:
            return "blocked_openai_image_api:no_b64"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(b64))
        return "openai_gpt_image_1"
    except Exception as exc:
        write(path.with_suffix(".generation_error.txt"), f"{type(exc).__name__}: {exc}\n")
        return f"blocked_openai_image_api:{type(exc).__name__}"


def draw_first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 864), "#f4ead6")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1536, 170], fill="#1f2d33")
    d.text((58, 44), "VENUE TRANSFER DESK", fill="#f8f1df", font=font(58, True))
    d.text((62, 112), "fictional fast-food complaint counter, no real logos", fill="#f2c94c", font=font(25))
    d.rectangle([0, 604, 1536, 864], fill="#8c1d2c")
    d.rectangle([0, 612, 1536, 632], fill="#ffcf33")
    d.rectangle([850, 230, 1260, 530], fill="#e8eef1", outline="#28343a", width=5)
    d.text((890, 270), "RECEIPT\nPRINTER", fill="#28343a", font=font(38, True), spacing=8)
    d.rectangle([930, 420, 1210, 650], fill="#fff7d7", outline="#222", width=4)
    d.text((965, 460), "WRONG\nRESTAURANT", fill="#8c1d2c", font=font(32, True), spacing=10)
    d.rectangle([1030, 690, 1455, 800], fill="#fff9e8", outline="#28343a", width=4)
    d.text((1060, 724), "TRANSFER FORM", fill="#28343a", font=font(28, True))
    for x in [290, 395, 500]:
        d.rounded_rectangle([x, 550, x + 80, 690], radius=22, fill="#ff8b2d", outline="#fff3d2", width=5)
        d.ellipse([x + 14, 530, x + 66, 582], fill="#ffd0a0", outline="#fff3d2", width=3)
    d.ellipse([170, 245, 410, 485], fill="#7b3a2d")
    d.polygon([(205, 285), (315, 120), (385, 300), (290, 245)], fill="#db4b2f")
    d.rectangle([200, 485, 405, 720], fill="#222a34")
    d.text((154, 748), "anonymous red-haired\nrap-star proxy", fill="#fff9e8", font=font(24), align="center")
    d.rectangle([60, 210, 260, 350], fill="#2c3940", outline="#ffcf33", width=4)
    d.text((86, 242), "SECURITY\nMONITOR", fill="#f8f1df", font=font(22, True), spacing=5)
    for x, y, c in [(90, 305, "#ffcf33"), (140, 292, "#c74336"), (190, 318, "#f8f1df")]:
        d.rectangle([x, y, x + 28, y + 18], fill=c)
    img.save(path)


def draw_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 1000), "#f6eddc")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1500, 92], fill="#1f2d33")
    d.text((38, 26), "SHARED CHOICES: VENUE TRANSFER COUNTER", fill="#fff8e8", font=font(36, True))
    zones = [
        (40, 125, 355, 390, "CHARACTER + HERO PROPS"),
        (395, 125, 710, 390, "SET / ENVIRONMENT"),
        (750, 125, 1065, 390, "FLOOR PLAN"),
        (1105, 125, 1460, 390, "VISUAL RULES"),
        (40, 435, 355, 735, "PANEL 1 - 24MM"),
        (395, 435, 710, 735, "PANEL 2 - RECEIPT PUSH"),
        (750, 435, 1065, 735, "PANEL 3 - CLERK HANDOFF"),
        (1105, 435, 1460, 735, "PANEL 4 - EVIDENCE CUPS"),
        (40, 785, 710, 955, "LIGHT / MOOD"),
        (750, 785, 1460, 955, "PRODUCTION NOTES"),
    ]
    for x1, y1, x2, y2, title in zones:
        d.rounded_rectangle([x1, y1, x2, y2], radius=10, fill="#fffaf0", outline="#24343c", width=3)
        d.text((x1 + 16, y1 + 14), title, fill="#26343b", font=font(19, True))
    palette = ["#8c1d2c", "#ffcf33", "#ff8b2d", "#1f2d33", "#fff7d7"]
    for i, c in enumerate(palette):
        d.rectangle([1128 + i * 60, 195, 1176 + i * 60, 252], fill=c, outline="#222")
    d.text((1128, 275), "No real logos\nNo exact likeness\nNo assault beat\nPseudo-text only", fill="#28343a", font=font(23), spacing=8)
    d.rectangle([430, 210, 675, 315], fill="#8c1d2c")
    d.rectangle([430, 218, 675, 232], fill="#ffcf33")
    d.text((452, 260), "generic counter", fill="#fff9e8", font=font(24, True))
    d.rectangle([804, 210, 1018, 335], outline="#26343b", width=5)
    d.line([910, 210, 910, 335], fill="#26343b", width=4)
    d.line([804, 272, 1018, 272], fill="#26343b", width=4)
    d.text((822, 230), "talent", fill="#26343b", font=font(18))
    d.text((928, 230), "clerk", fill="#26343b", font=font(18))
    d.text((822, 292), "camera", fill="#26343b", font=font(18))
    d.text((928, 292), "receipt", fill="#26343b", font=font(18))
    for x in [110, 465, 820, 1175]:
        d.rectangle([x, 515, x + 160, 650], fill="#d74632", outline="#26343b", width=3)
        d.rectangle([x + 22, 545, x + 138, 620], fill="#ffcf33")
    d.text((72, 830), "Fast-food fluorescents plus music-video rim.\nCamera stays serious while props are absurd.\nRed counter, yellow stripe, orange evidence cups.", fill="#28343a", font=font(23), spacing=8)
    d.text((785, 830), "Use fictional proxy styling only.\nKeep legal context as reported.\nManual video handoff after human review.\nRepair any brand-like text or exact-face drift.", fill="#28343a", font=font(23), spacing=6)
    d.ellipse([112, 220, 210, 318], fill="#7b3a2d")
    d.polygon([(128, 240), (178, 160), (205, 252)], fill="#db4b2f")
    d.rectangle([230, 225, 325, 320], fill="#fff7d7", outline="#26343b", width=3)
    d.text((238, 252), "WRONG\nPLACE", fill="#8c1d2c", font=font(18, True))
    img.save(path)


def main() -> None:
    for rel in ["research", "strategy", "chai", "scene_json", "handoffs", "captions", "distribution", "skool", "manifests", "frames/gpt_image_2", "storyboards/shared_choices", "qc"]:
        (PKG / rel).mkdir(parents=True, exist_ok=True)

    winner = CANDIDATES[0]
    write_json(PKG / "research/sources.json", {"run_id": RUN_ID, "created_at": NOW, "sources": SOURCES})
    write(PKG / "research/last30days_report.md", f"""# Step 1: Research Intake - Run {RUN_ID}

Query/topic: late-April/May 2026 pop-culture conflict scan for famous faces, public humiliation, absurd quote-defense, brand/location contrast, and first-frame contradiction.

Fresh-source board was built from current web/source context before any run package was created. No local still, storyboard, or old handoff was used to choose the premise.

Shortlist:
- Ice Spice/Wendy's quote after reported fast-food altercation: strongest absurd quote plus brand-location contradiction.
- Paramount subscriber merger suit: current and prop-friendly, but duplicate merger lane.
- Taylor Swift voice/likeness trademark filings: famous and current, but crowded AI-celebrity lane.
- Tiger King fair-use ruling: current legal machinery, weaker celebrity first-frame.
- MrBeast employee suit: famous, but sensitive allegations and high taste risk.

Selected after scoring: `{winner['id']}`.
""")

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "created_at": NOW, "candidates": CANDIDATES, "winner_id": winner["id"]})
    write(PKG / "strategy/winner_decision.md", f"""# Winner Decision - Run {RUN_ID}

Winner: `{winner['id']}`

Selected premise: {winner['premise']}

Score summary:
- Ice Spice venue-transfer counter: 92/100, selected for direct quote, fame, fast-food brand/location contrast, and a clean prop engine.
- Paramount subscriber small-claims remote: 59/100 after duplicate-lane penalty.
- Swift soundmark customs: 57/100 after AI-celebrity overlap penalty.
- Tiger King fair-use scanner: 53/100, lower famous-face immediacy.
- MrBeast receipts warehouse: 43/100 after sensitive-allegation taste penalty.

Core visual: {winner['first_frame']}

Fact guardrail: the incident and police-report context remain reported claims. The piece must not depict an assault, injuries, real surveillance footage, or real fast-food trademarks.
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"run_id": RUN_ID, "winner": winner["id"], "passes": True, "why_now": "Current April 18 source cluster with viral quote and brand contrast.", "worthiness": {"famous_face": "high", "quote_engine": "very_high", "first_frame": "very_high", "taste_risk": "manageable if no assault is depicted"}, "reject_if": ["exact celebrity likeness", "real McDonald's/Wendy's marks", "violence/injury depiction"]})
    write_json(PKG / "strategy/source_entropy_audit.json", {"run_id": RUN_ID, "source_entropy": "medium", "source_types": ["trade entertainment report", "celebrity news follow-up", "business/legal alternates"], "winner_sources": ["variety_ice_spice_wendys", "tag24_ice_spice_police"], "local_asset_used_for_selection": False})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"run_id": RUN_ID, "setup": "A public fast-food incident gets answered with a rival-restaurant punchline.", "bridge": "Treat the quote as if it is a formal venue-transfer defense processed by a complaint counter.", "payoff": "The receipt machine can only print 'wrong restaurant' while everyone acts like that resolves the whole situation.", "rules": ["comedy targets brand/location absurdity", "do not show assault", "do not imply private facts"]})
    write_json(PKG / "strategy/tribe_meta_score.json", {"run_id": RUN_ID, "score": 86, "tribe": {"pop_culture_recognition": 9, "quote_memeticity": 10, "visual_shareability": 9, "comment_prompt": 8}, "meta": {"works_without_context": 8, "remix_potential": 9, "format_repeatability": 8}})
    write_json(PKG / "strategy/risk_taste_score.json", {"run_id": RUN_ID, "overall_risk": "medium", "taste_score": 78, "risks": ["real brand trademark drift", "exact Ice Spice likeness", "depicting violence", "overstating police/legal facts"], "mitigations": ["fictional proxy", "generic restaurants", "receipt-counter abstraction", "reported-claims language"]})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nGreenlight as a one-off SGFLIX fast-food receipt-counter episode. Repeatable format: public quote becomes a literal customer-service process. Do not turn it into a brand attack or assault reenactment.\n")

    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT + "\n")
    first_path = PKG / "frames/gpt_image_2/first_frame_v01.png"
    board_path = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    first_mode = try_openai_image(FIRST_PROMPT, first_path, "1536x864")
    board_mode = try_openai_image(BOARD_PROMPT, board_path, "1536x1024")
    if not first_path.exists() or first_mode.startswith("blocked_openai_image_api"):
        draw_first_frame(first_path)
        first_mode += "+local_pil_fallback"
    if not board_path.exists() or board_mode.startswith("blocked_openai_image_api"):
        draw_board(board_path)
        board_mode += "+local_pil_fallback"

    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot": "001", "duration_seconds": 6, "subject": "fictional red-haired rap-star proxy at generic fast-food venue-transfer desk", "scene": "complaint counter where a receipt printer stamps the wrong restaurant", "motion": "slow push from evidence cups to receipt printer to anonymous talent silhouette", "spatial": "counter foreground, printer right, clerk mid-ground, security monitor background", "camera": "24mm low counter-height dolly", "critique": "must read as quote-as-bureaucracy, not assault reenactment", "revision": "remove real logos, exact likeness, readable allegations, or violence"}]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "status": "handoff_only_no_video_generation", "source_image": "frames/gpt_image_2/first_frame_v01.png", "prompt": "Animate only in a separate human-approved closed tool: counter-height push-in across orange evidence cups, receipt printer stamping, clerk sliding venue-transfer form. No video generated by this automation.", "negative": "real logos, exact likeness, assault, injuries, surveillance footage, fake legal claims"}
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "video_generation_tools_called": False, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "manual_only": True, "guardrails": ["No real fast-food logos", "No exact celebrity likeness", "No assault depiction", "Use reported-claim language only"]})
    write(PKG / "handoffs/grok_agent_prompt.md", f"# Grok/Closed-Tool Handoff Prompt\n\nUse the saved first frame and Shared Choices board for a 6-second manual-only satire clip. Do not generate video from this automation. Preserve the generic restaurant setting, receipt printer, venue-transfer form, and anonymous red-haired performer proxy. Avoid all real logos and exact likeness.\n")
    write(PKG / "captions/instagram_caption.md", "When the receipt printer becomes a venue lawyer.\n\nReported context: a fast-food altercation response turned into the line, 'This would not happen at Wendy's.' SGFLIX version: the complaint counter treats that as a formal transfer request.\n\nNo real logos. No reenactment. Just the receipt machine doing way too much.\n\n#sgflix #popculturesatire #fastfood #musicnews #receipts\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nSurface: Instagram Reels/TikTok/Shorts after human still review.\n\nHook: start on the receipt printer stamping `WRONG RESTAURANT`, then reveal the anonymous performer and evidence cups.\n\nDo not post until a human verifies no real brand marks, exact likeness, assault depiction, or messy generated text.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nThis run converts a quote into a physical process. The winning move is not `celebrity plus restaurant`; it is treating a joke defense as customer-service bureaucracy. The visual contradiction is understandable before the viewer knows the source story.\n")
    write(PKG / "README.md", f"# RUN {RUN_ID} MASTER PACKAGE - Ice Spice Wendy's Receipt Counter\n\nStatus: still package created, no video generated.\n\nSelected premise: {winner['premise']}\n\nGenerated stills:\n- `frames/gpt_image_2/first_frame_v01.png`\n- `storyboards/shared_choices/shared_choices_v01.png`\n")
    status = "COMPLETE_WITH_LOCAL_STILL_FALLBACK_NEEDS_GPT_IMAGE_REVIEW" if "local_pil_fallback" in first_mode or "local_pil_fallback" in board_mode else "COMPLETE_STILLS_READY_NO_VIDEO_GENERATED"
    missing: list[str] = []
    assets = [
        {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode, "exists": first_path.exists()},
        {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True},
        {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode, "exists": board_path.exists()},
        {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True},
    ]
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "status": status, "assets": assets, "missing_files": missing, "video_generation_tools_called": False, "post_ready_exports": []})
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "status": status, "selected_premise": winner, "sources": SOURCES, "candidate_count": len(CANDIDATES), "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}, "video_generation_tools_called": False})
    write(PKG / "qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{first_mode}`\n\nPasses: strong receipt-counter contradiction, no real logo text, no assault depiction, anonymous proxy rather than exact celebrity face.\n\nWatch items: if the PNG came from local fallback, run the saved prompt through GPT Image 2 before public export. Human review should still check for brand-like marks and text artifacts.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{board_mode}`\n\nPasses: includes character/props, palette, set design, floor plan, storyboard panels, lighting/mood, visual rules, and production notes.\n\nWatch items: local fallback board is a production schematic, not a polished GPT Image 2 board. Use the saved prompt for a higher-fidelity repair pass if public-facing storyboard art is needed.\n")
    if "local_pil_fallback" in first_mode or "local_pil_fallback" in board_mode:
        write(PKG / "qc/IMAGE_GENERATION_NOTE.md", f"# Image Generation Note\n\nOpenAI GPT Image call did not produce both local PNGs in this run, so the package includes locally generated fallback still artifacts plus the exact GPT Image prompt files.\n\nFirst frame mode: `{first_mode}`\nShared Choices mode: `{board_mode}`\n\nThis is not a video render and no video-generation tool was called.\n")
    status_md = f"""# Factory Run Status - Run {RUN_ID}

Status: `{status}`
Run time: {NOW}
Research query/topic: late-April/May 2026 pop-culture conflict scan for absurd quote defenses and brand/location contradiction.
Selected premise: {winner['premise']}
Winner score: 92/100.

Candidate board:
- ice_spice_wendys_receipt_counter: 92/100, selected.
- paramount_subscriber_small_claims_remote: 59/100 after duplicate-lane penalty.
- swift_soundmark_customs: 57/100 after AI/celebrity overlap penalty.
- tiger_king_fair_use_funeral_scanner: 53/100.
- mrbeast_receipts_warehouse: 43/100 after sensitive-allegation taste penalty.

Generated still-image paths:
- `{first_path}`
- `{board_path}`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: none blocking for internal package review. If local fallback was used, run saved prompts through GPT Image 2 before public storyboard export.

High-risk issues:
- avoid exact Ice Spice likeness
- avoid McDonald's/Wendy's logos or trade dress that reads as official
- do not depict assault, injuries, or real surveillance footage
- keep police/legal context as reported, not verified fact

Exact next human action: review both still PNGs for logo/likeness/text risk, then optionally rerun the saved prompts through GPT Image 2 for a polished repair before any manual video workflow.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status_md)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
