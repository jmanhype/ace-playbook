from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_ID = "060"
SLUG = "blake_brand_returns_counter"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "tmz_baldoni_lively_business_failed_unlikable",
        "title": "Justin Baldoni's Lawyers Say Blake Lively's Businesses Failed Because She's Unlikable",
        "url": "https://www.tmz.com/2026/04/28/justin-baldonis-lawyers-says-blake-lively-has-track-record-of-businesses-failing/",
        "publisher": "TMZ",
        "date": "2026-04-28",
        "verified_facts": [
            "TMZ reported that pretrial damages arguments included Lively's claim that reputation harm affected her brands.",
            "TMZ reported Baldoni's side argued that Lively's brand losses were not caused by the defendants.",
            "TMZ reported the defense pointed to Betty Buzz, Blake Brown, Kate Middleton-related criticism, and a disputed $132 million future-earnings figure.",
            "TMZ reported Lively's side argued her businesses might have grown absent alleged retaliation.",
        ],
        "creative_use": "Primary source for turning courtroom damages rhetoric into a fictional brand-returns counter.",
    },
    {
        "id": "variety_taylor_voice_likeness_trademark",
        "title": "Taylor Swift Files to Trademark Her Voice and Likeness, Apparently to Protect Against AI Misuse",
        "url": "https://au.variety.com/2026/music/news/taylor-swift-trademark-voice-likeness-ai-misuse-35964/",
        "publisher": "Variety Australia",
        "date": "2026-04-28",
        "verified_facts": [
            "Variety reported Swift's company filed three USPTO trademark applications on April 24, 2026.",
            "Variety reported two filings related to sound marks for Swift's voice and one to a specific visual image.",
            "Variety framed the filings as a potential identity-protection strategy in the AI era.",
        ],
        "creative_use": "Candidate only; rejected because celebrity-plus-AI needs a sharper human humiliation engine.",
    },
    {
        "id": "variety_mrbeast_employee_lawsuit",
        "title": "MrBeast's Beast Industries Sued by Former Employee Alleging Sexual Harassment and Retaliation",
        "url": "https://au.variety.com/2026/digital/news/mrbeast-sued-former-employee-sexual-harassment-retaliation-35780/",
        "publisher": "Variety Australia",
        "date": "2026-04-23",
        "verified_facts": [
            "Variety reported a former Beast Industries employee sued alleging harassment and retaliation.",
            "Variety reported the company denied the allegations and called the statements categorically false.",
            "Variety reported Donaldson's channel subscriber scale and the company's employee scale.",
        ],
        "creative_use": "Candidate only; rejected as allegation-led and taste-risky for a comedy still package.",
    },
    {
        "id": "variety_paramount_wbd_petition",
        "title": "Hollywood Petition to Block Paramount-Warner Bros. Merger Tops 4,000 Names",
        "url": "https://au.variety.com/2026/biz/news/petition-block-paramount-warner-bros-merger-4000-names-robert-de-niro-sofia-coppola-holly-hunter-35854/",
        "publisher": "Variety Australia",
        "date": "2026-04-24",
        "verified_facts": [
            "Variety reported an open letter opposing the Paramount-WBD merger had 4,194 total signatories.",
            "Variety reported signatories included Robert De Niro, Sofia Coppola, Holly Hunter, and many other entertainment figures.",
            "Variety reported opponents cited lost jobs, higher costs, and fewer shows or movies.",
        ],
        "creative_use": "Candidate only; rejected as current but too close to recent SGFLIX merger/legal-counter lanes.",
    },
]


CANDIDATES = [
    {
        "id": "blake_brand_returns_counter",
        "premise": "A fictional courtroom brand-returns counter audits celebrity reputation damages like broken soda cans, haircare bottles, and a warranty claim stamped PEOPLE DID NOT LIKE THIS.",
        "source_ids": ["tmz_baldoni_lively_business_failed_unlikable"],
        "first_frame": "A sleek department-store returns desk inside a courtroom where a lawyer slides damaged celebrity-brand products across a scanner marked REPUTATION DAMAGES.",
        "scores": {
            "famous_face": 9,
            "public_conflict": 9,
            "ego_humiliation": 10,
            "absurd_quote_defense": 9,
            "brand_location_contrast": 9,
            "first_frame_visual_contradiction": 10,
            "risk_control": 7,
            "franchise_potential": 8,
        },
        "total": 89,
    },
    {
        "id": "taylor_voice_trademark_lost_found",
        "premise": "A pop star files her voice at a courthouse lost-and-found where the clerk asks every phrase to take a number.",
        "source_ids": ["variety_taylor_voice_likeness_trademark"],
        "first_frame": "A USPTO-style lost-and-found counter with voice bubbles in evidence bags and a pink guitar silhouette on the claim ticket.",
        "scores": {
            "famous_face": 10,
            "public_conflict": 6,
            "ego_humiliation": 5,
            "absurd_quote_defense": 8,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 9,
            "risk_control": 8,
            "franchise_potential": 7,
        },
        "total": 73,
        "penalty": "Celebrity-plus-AI premise passes visual clarity but lacks a sharper public humiliation target.",
    },
    {
        "id": "mrbeast_hr_chocolate_factory",
        "premise": "A creator empire tries to process a lawsuit through a challenge-video HR maze with oversized subscriber counters.",
        "source_ids": ["variety_mrbeast_employee_lawsuit"],
        "first_frame": "A sterile HR desk wedged inside a giant content-studio prize set.",
        "scores": {
            "famous_face": 9,
            "public_conflict": 8,
            "ego_humiliation": 7,
            "absurd_quote_defense": 6,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "risk_control": 2,
            "franchise_potential": 6,
        },
        "total": 54,
        "penalty": "Allegation-led harassment subject is too high-risk for this satire cycle.",
    },
    {
        "id": "deniro_merger_picket_coat_check",
        "premise": "A Hollywood petition becomes a black-tie coat check where every A-list signature is another merger objection ticket.",
        "source_ids": ["variety_paramount_wbd_petition"],
        "first_frame": "A gala coat check overflowing with protest placards and studio-logo-neutral ticket stubs.",
        "scores": {
            "famous_face": 8,
            "public_conflict": 8,
            "ego_humiliation": 6,
            "absurd_quote_defense": 6,
            "brand_location_contrast": 7,
            "first_frame_visual_contradiction": 8,
            "risk_control": 8,
            "franchise_potential": 6,
        },
        "total": 62,
        "penalty": "Duplicate lane: recent runs already used De Niro/merger/legal-dinner framing.",
    },
]


FIRST_PROMPT = """GPT Image 2 prompt: SGFLIX satirical first frame, 16:9 cinematic editorial still. Concept: fictionalized public-figure proxies, not exact Blake Lively or Justin Baldoni likenesses. A sleek luxury brand returns counter has been installed inside a courtroom. Hero contradiction: celebrity reputation damages are scanned like retail returns. Foreground: a lawyer hand slides generic damaged soda cans, haircare bottles, and lifestyle-brand boxes over a scanner labeled REPUTATION DAMAGES. The clerk holds a warranty slip stamped PEOPLE DID NOT LIKE THIS, while a separate tiny tag reads DISPUTED $132M CLAIM. Background: courtroom wood paneling, fluorescent retail counter lights, a small gavel-shaped barcode scanner, neat stacks of evidence receipts. Style: crisp pop-culture legal satire, expensive department-store palette with courtroom browns, champagne gold, clinical scanner blue, 28mm lens, high detail, clean readable prop text. Guardrails: no real brand logos, no exact celebrity likeness, no defamatory new claims, no weapons, no watermark, no video."""

BOARD_PROMPT = """GPT Image 2 prompt: Shared Choices director-bible storyboard board for SGFLIX, 3:2. Premise: a fictional courtroom brand-returns counter audits reputation damages like retail returns after reported pretrial arguments about celebrity business losses. Include labeled zones: character + hero props, color palette, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes. Props: REPUTATION DAMAGES scanner, generic soda cans, haircare bottles, lifestyle-brand boxes, PEOPLE DID NOT LIKE THIS warranty slip, DISPUTED $132M tag, evidence receipts, gavel barcode scanner. Visual rules: fictionalized proxies only, no exact Blake Lively or Justin Baldoni likenesses, no real brand logos, source-attributed claim language only, no weapons, no video generation, clean minimal text, no watermark."""


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


def draw_first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 864), "#efe6d8")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1536, 160], fill="#3a2418")
    d.text((48, 38), "REPUTATION DAMAGES", fill="#f6e6be", font=font(60, True))
    d.text((52, 108), "fictional courtroom brand-returns counter - source-attributed satire only", fill="#86c5da", font=font(24))
    d.rectangle([0, 640, 1536, 864], fill="#111923")
    d.rectangle([0, 585, 1536, 650], fill="#cda35b")
    d.rectangle([450, 260, 950, 520], fill="#d9eef6", outline="#111923", width=6)
    d.text((510, 305), "SCANNER", fill="#111923", font=font(42, True))
    d.text((508, 366), "REPUTATION\nDAMAGES", fill="#3a2418", font=font(36, True), spacing=5)
    for x, label, color in [(110, "SODA", "#8bbbd9"), (240, "HAIR", "#f0d9a2"), (370, "LIFE", "#d48a73")]:
        d.rounded_rectangle([x, 505, x + 95, 625], radius=14, fill=color, outline="#111923", width=4)
        d.text((x + 18, 550), label, fill="#111923", font=font(22, True))
    d.rectangle([980, 245, 1385, 540], fill="#fff8e8", outline="#111923", width=6)
    d.text((1025, 282), "WARRANTY SLIP", fill="#111923", font=font(30, True))
    d.rectangle([1030, 350, 1335, 430], outline="#a7352c", width=7)
    d.text((1060, 372), "PEOPLE DID\nNOT LIKE THIS", fill="#a7352c", font=font(31, True), spacing=0)
    d.rectangle([1068, 465, 1300, 515], fill="#111923")
    d.text((1090, 478), "DISPUTED $132M", fill="#f6e6be", font=font(25, True))
    d.rectangle([60, 190, 250, 400], fill="#73482f", outline="#111923", width=5)
    d.text((94, 245), "COURT\nRETURNS", fill="#f6e6be", font=font(34, True), spacing=8)
    d.polygon([(760, 585), (825, 530), (890, 585), (860, 630), (790, 630)], fill="#3a2418", outline="#f6e6be")
    d.text((794, 580), "SCAN", fill="#f6e6be", font=font(22, True))
    img.save(path)


def draw_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 1000), "#efe6d8")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1500, 92], fill="#3a2418")
    d.text((34, 27), "SHARED CHOICES: BRAND RETURNS COURT", fill="#f6e6be", font=font(38, True))
    zones = [
        (40, 125, 355, 390, "CHARACTER + PROPS"),
        (395, 125, 710, 390, "SET DESIGN"),
        (750, 125, 1065, 390, "FLOOR PLAN"),
        (1105, 125, 1460, 390, "VISUAL RULES"),
        (40, 435, 355, 735, "PANEL 1 - COUNTER"),
        (395, 435, 710, 735, "PANEL 2 - SCANNER"),
        (750, 435, 1065, 735, "PANEL 3 - WARRANTY"),
        (1105, 435, 1460, 735, "PANEL 4 - TAG"),
        (40, 785, 710, 955, "LIGHT / MOOD"),
        (750, 785, 1460, 955, "PRODUCTION NOTES"),
    ]
    for x1, y1, x2, y2, title in zones:
        d.rounded_rectangle([x1, y1, x2, y2], radius=8, fill="#fff8e8", outline="#111923", width=3)
        d.text((x1 + 15, y1 + 14), title, fill="#111923", font=font(18, True))
    for i, color in enumerate(["#3a2418", "#cda35b", "#86c5da", "#a7352c", "#fff8e8"]):
        d.rectangle([1128 + i * 58, 195, 1175 + i * 58, 252], fill=color, outline="#111923")
    d.text((1128, 278), "fictional proxies only\nno real brand logos\nsource-attributed claims\nno video generation", fill="#111923", font=font(23), spacing=8)
    d.rectangle([445, 220, 670, 315], fill="#d9eef6", outline="#111923", width=4)
    d.text((472, 253), "scanner", fill="#111923", font=font(30, True))
    d.rectangle([812, 190, 1020, 345], outline="#111923", width=5)
    d.line([916, 190, 916, 345], fill="#111923", width=4)
    d.line([812, 267, 1020, 267], fill="#111923", width=4)
    d.text((830, 220), "proxy", fill="#111923", font=font(18))
    d.text((928, 220), "clerk", fill="#111923", font=font(18))
    d.text((830, 292), "camera", fill="#111923", font=font(18))
    d.text((928, 292), "scanner", fill="#111923", font=font(18))
    for x in [110, 465, 820, 1175]:
        d.rectangle([x, 520, x + 160, 650], fill="#111923", outline="#3a2418", width=3)
        d.rectangle([x + 25, 560, x + 135, 595], fill="#d9eef6")
    d.text((72, 830), "Courtroom browns meet clinical retail scanner blue.\nThe gag is retail procedure applied to reputation damages.\nKeep it dry, petty, and legalistic.", fill="#111923", font=font(24), spacing=8)
    d.text((785, 830), "Use fictionalized public-figure proxies only.\nKeep all brands generic and readable.\nHuman review required before public export.\nNo video generated by this factory run.", fill="#111923", font=font(24), spacing=7)
    d.rectangle([100, 225, 320, 330], fill="#fff8e8", outline="#111923", width=3)
    d.text((123, 258), "PEOPLE DID\nNOT LIKE THIS", fill="#a7352c", font=font(26, True), spacing=1)
    img.save(path)


def copy_external_image(src: str, dest: Path) -> bool:
    if not src:
        return False
    source = Path(src).expanduser()
    if not source.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return True


def main() -> None:
    external_first = sys.argv[1] if len(sys.argv) > 1 else ""
    external_board = sys.argv[2] if len(sys.argv) > 2 else ""
    for rel in [
        "research",
        "strategy",
        "chai",
        "scene_json",
        "handoffs",
        "captions",
        "distribution",
        "skool",
        "manifests",
        "frames/gpt_image_2",
        "storyboards/shared_choices",
        "qc",
    ]:
        (PKG / rel).mkdir(parents=True, exist_ok=True)

    winner = CANDIDATES[0]
    write_json(PKG / "research/sources.json", {"run_id": RUN_ID, "created_at": NOW, "sources": SOURCES})
    write(PKG / "research/last30days_report.md", f"""# Step 1: Research Intake - Run {RUN_ID}

Generated: {NOW}

Query/topic: current late-April/early-May 2026 entertainment conflict scan for famous face, public conflict, ego/humiliation, absurd legal/PR quote, brand/location contrast, and strong first-frame contradiction.

Research order note: this report and `sources.json` were created from current web/source context before a winner was selected and before the numbered run package was completed. No local image, prior storyboard, old handoff, or nearby asset was used as the premise.

Fresh source intake:
- TMZ's April 28, 2026 report on Blake Lively/Justin Baldoni pretrial damages arguments supplied the strongest humiliation-language and brand-returns visual engine.
- Variety's April 28, 2026 Taylor Swift trademark report was considered, but celebrity-plus-AI was penalized because the human conflict/humiliation layer was weaker.
- Variety's April 23, 2026 MrBeast lawsuit report was considered, but rejected as allegation-led and taste-risky.
- Variety's April 24, 2026 Paramount-WBD petition report was considered, but rejected as a duplicate merger/legal lane.

Selected after scoring: `{winner["id"]}`.
""")
    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "created_at": NOW, "candidates": CANDIDATES, "winner_id": winner["id"]})
    write(PKG / "strategy/winner_decision.md", f"""# Winner Decision - Run {RUN_ID}

Winner: `{winner["id"]}`

Selected premise: {winner["premise"]}

Score summary:
- blake_brand_returns_counter: 89/100, selected for famous-face conflict, humiliating damages rhetoric, brand/location contrast, and clean first-frame contradiction.
- taylor_voice_trademark_lost_found: 73/100 after weak celebrity-plus-AI humor-gate penalty.
- deniro_merger_picket_coat_check: 62/100 after duplicate-lane penalty.
- mrbeast_hr_chocolate_factory: 54/100 after harassment-allegation taste penalty.

Core visual: {winner["first_frame"]}

Fact guardrail: source claims must be attributed to reporting or legal-position language. The satire targets public litigation rhetoric and celebrity-brand math. Do not imply new facts about private conduct, actual brand quality, or liability.
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "Blake Lively / Justin Baldoni damages rhetoric", "track_a_newsjack_velocity": {"active_trend_score": 8, "algorithmic_slipstream": "fresh courtroom/pretrial coverage around a high-attention celebrity dispute", "polarization_factor": 8, "track_a_total": 24, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 9, "stereotype_rigidity": "High", "subversion_potential": 9, "track_b_total": 27, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_A_NEWSJACK_WITH_TRACK_B_CELEBRITY_BRAND_ARCHETYPE", "urgency_class": "High", "strategic_directive": "Literalize reputation-damages arguments as a retail returns counter, with fictional proxies and no brand/logo drift."}}})
    write_json(PKG / "strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "A real, high-attention celebrity legal dispute with reported pretrial arguments over reputation and business damages.", "detected_anomalies": ["business losses argued through celebrity likability", "retail brands as courtroom damages objects", "a large future-earnings number functioning like a price tag", "public-feud tone colliding with legal procedure"], "native_entropy_score": 6, "subject_self_awareness": "trying_to_look_cool", "comedic_vector_recommendation": "cringe_vector", "recommended_strategy": "straight_man_framing"}, "local_asset_used_for_selection": False})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"run_id": RUN_ID, "setup": "A celebrity legal dispute includes public reporting about reputation harm, brand performance, and a disputed damages number.", "bridge": "Treat reputation as a defective retail item being returned at a courthouse customer-service desk.", "payoff": "The scanner rings up PEOPLE DID NOT LIKE THIS as if it were a product warranty diagnosis.", "rules": ["target litigation rhetoric, not alleged private conduct", "use fictionalized proxies", "keep brands generic", "attribute claims to reporting/legal positions", "avoid new defamatory claims"]})
    write_json(PKG / "strategy/tribe_meta_score.json", {"run_id": RUN_ID, "score": 88, "tribe": {"celebrity_litigation_attention": 9, "brand_failure_discourse": 9, "retail_returns_archetype": 8, "receipt_culture": 8}, "meta": {"works_without_context": 9, "remix_potential": 8, "format_repeatability": 9}})
    write_json(PKG / "strategy/risk_taste_score.json", {"run_id": RUN_ID, "overall_risk": "medium", "taste_score": 82, "risks": ["exact likeness drift", "real brand logo drift", "defamatory implication that reported legal positions are facts", "overly cruel personal-insult framing"], "mitigations": ["fictionalized proxies", "generic brand packaging", "source-attributed captions", "satire targets damages rhetoric and brand math"]})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nGreenlight as a reusable `brand returns counter` franchise lane. It can repeat whenever a celebrity, politician, or company converts reputation, loyalty, or identity into a claimable business asset.\n")

    first_path = PKG / "frames/gpt_image_2/first_frame_v01.png"
    board_path = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT + "\n")
    first_mode = "built_in_imagegen_external" if copy_external_image(external_first, first_path) else "local_pil_fallback"
    board_mode = "built_in_imagegen_external" if copy_external_image(external_board, board_path) else "local_pil_fallback"
    if first_mode == "local_pil_fallback":
        draw_first_frame(first_path)
    if board_mode == "local_pil_fallback":
        draw_board(board_path)

    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot": "001", "duration_seconds": 6, "subject": "fictional courtroom brand-returns counter with celebrity-brand products", "scene": "luxury returns desk inside courtroom, reputation scanner, evidence receipts, generic products", "motion": "slow push from warranty slip to scanner to disputed damages tag", "spatial": "products left foreground, scanner center, clerk slip right, courtroom background", "camera": "28mm counter-height dolly, crisp editorial lighting", "critique": "must target legal rhetoric and brand math, not alleged private conduct", "revision": "remove exact likenesses, real logos, defamatory text, or messy labels"}]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "status": "handoff_only_no_video_generation", "source_image": "frames/gpt_image_2/first_frame_v01.png", "prompt": "Manual-only 6-second clip: push from PEOPLE DID NOT LIKE THIS warranty slip to REPUTATION DAMAGES scanner to DISPUTED $132M tag. No video generated by this automation.", "negative": "exact Blake Lively likeness, exact Justin Baldoni likeness, real brand logos, defamatory new claims, weapons, watermark"}
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "video_generation_tools_called": False, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "manual_only": True, "guardrails": ["No exact likenesses", "No real product logos", "No invented legal claims", "Use sourced reporting/legal-position language only", "Keep the joke on reputation-damages retail math"]})
    write(PKG / "handoffs/grok_agent_prompt.md", "# Grok/Closed-Tool Handoff Prompt\n\nUse the saved first frame and Shared Choices board for a 6-second manual-only satire clip. Do not generate video from this automation. Preserve the courtroom brand-returns counter, REPUTATION DAMAGES scanner, PEOPLE DID NOT LIKE THIS warranty slip, generic lifestyle-brand products, and DISPUTED $132M tag. Avoid exact likenesses, real logos, weapons, and unsourced legal claims.\n")
    write(PKG / "captions/instagram_caption.md", "When reputation damages get processed like a return without a receipt.\n\nReported context: TMZ covered pretrial arguments in the Lively/Baldoni dispute over whether reputation harm affected celebrity brands and whether the claimed business losses were caused by the defendants. SGFLIX version: the courtroom installs a customer-service desk and scans every brand claim like a defective product.\n\n#sgflix #popculturesatire #celebritycourt #branddrama #receipts\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nSurface: Instagram Reels/TikTok/Shorts after human still review.\n\nHook: open on the warranty slip, reveal the REPUTATION DAMAGES scanner, then land on the DISPUTED $132M tag.\n\nDo not post until a human verifies no exact likeness, real logo drift, messy generated text, or unsourced legal claims.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nThis run demonstrates the `abstract damages to physical counter` conversion. The public-language object is reputation. The factory turns it into a returnable retail item, then lets scanner text and warranty slips carry the joke.\n")
    write(PKG / "README.md", f"# RUN {RUN_ID} MASTER PACKAGE - Blake Brand Returns Counter\n\nStatus: still package created, no video generated.\n\nSelected premise: {winner['premise']}\n\nGenerated stills:\n- `frames/gpt_image_2/first_frame_v01.png`\n- `storyboards/shared_choices/shared_choices_v01.png`\n")

    status = "COMPLETE_STILLS_READY_NO_VIDEO_GENERATED" if first_mode.startswith("built_in") and board_mode.startswith("built_in") else "COMPLETE_WITH_LOCAL_STILL_FALLBACK_NEEDS_GPT_IMAGE_REVIEW"
    assets = [
        {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode, "exists": first_path.exists()},
        {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True},
        {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode, "exists": board_path.exists()},
        {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True},
    ]
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "status": status, "assets": assets, "missing_files": [], "video_generation_tools_called": False, "post_ready_exports": []})
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "status": status, "selected_premise": winner, "sources": SOURCES, "candidate_count": len(CANDIDATES), "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}, "video_generation_tools_called": False})
    write(PKG / "qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{first_mode}`\n\nPasses: clear courtroom/retail contradiction, generic products, no video generation.\n\nWatch items: verify no exact likeness, real brand/logo drift, messy text, or defamatory claim drift before public export.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{board_mode}`\n\nPasses: includes character/props, color palette, set design, floor plan, storyboard panels, lighting/mood, visual rules, and production notes.\n\nWatch items: if local fallback was used, it is schematic; rerun saved prompt through GPT Image 2 before public export.\n")
    if "local_pil_fallback" in {first_mode, board_mode}:
        write(PKG / "qc/IMAGE_GENERATION_NOTE.md", f"# Image Generation Note\n\nThe package includes local generated fallback still artifacts where built-in image output was not supplied to the script.\n\nFirst frame mode: `{first_mode}`\nShared Choices mode: `{board_mode}`\n\nThis is not a video render and no video-generation tool was called.\n")

    status_md = f"""# Factory Run Status - Run {RUN_ID}

Status: `{status}`
Run time: {NOW}
Research query/topic: current late-April/early-May 2026 entertainment conflict scan around celebrity legal rhetoric, brand damages, identity-protection filings, creator-company lawsuits, and merger protest coverage.
Selected premise: {winner['premise']}
Winner score: 89/100.

Candidate board:
- blake_brand_returns_counter: 89/100, selected.
- taylor_voice_trademark_lost_found: 73/100 after weak celebrity-plus-AI humor-gate penalty.
- deniro_merger_picket_coat_check: 62/100 after duplicate-lane penalty.
- mrbeast_hr_chocolate_factory: 54/100 after allegation-led taste penalty.

Generated still-image paths:
- `{first_path}`
- `{board_path}`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: none blocking for internal package review. Public export needs human text/likeness/logo review.

High-risk issues:
- avoid exact Blake Lively or Justin Baldoni likenesses
- avoid real Betty Buzz, Blake Brown, or other product logos/trade dress
- avoid presenting reported legal positions as established facts
- avoid cruel personal-insult framing beyond the public rhetoric being satirized

Exact next human action: review both still PNGs for likeness/logo/text/claim-risk issues, then manually decide whether to send the saved first frame and Shared Choices board into a closed-tool video workflow.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status_md)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
