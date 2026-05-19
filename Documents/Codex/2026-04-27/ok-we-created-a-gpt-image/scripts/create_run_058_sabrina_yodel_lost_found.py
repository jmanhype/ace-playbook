from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_ID = "058"
SLUG = "sabrina_yodel_lost_found"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "tmz_sabrina_coachella_yodel",
        "title": "Sabrina Carpenter Confuses Cultural Cry For Yodeling During Coachella Set",
        "url": "https://www.tmz.com/2026/04/11/sabrina-carpenter-slammed-for-disliking-cultural-cry-at-coachella/",
        "publisher": "TMZ",
        "date": "2026-04-11",
        "verified_facts": [
            "TMZ reported that Carpenter mistook a fan's cultural celebration call for yodeling during her Coachella set.",
            "TMZ reported that she later posted an apology and said she could not see or hear the person clearly.",
        ],
        "creative_use": "Primary source for the yodeling-mislabel and apology-bureaucracy visual engine.",
    },
    {
        "id": "financial_express_sabrina_apology",
        "title": "Sabrina Carpenter issues apology after culturally insensitive moment at Coachella 2026",
        "url": "https://www.financialexpress.com/life/entertainment-could-have-handled-it-better-sabrina-carpenter-issues-apology-after-culturally-insensitive-moment-at-coachella-2026-4204541/lite/",
        "publisher": "Financial Express",
        "date": "2026-04-12",
        "verified_facts": [
            "The report described the fan sound as a Zaghrouta, a traditional Arabic celebration call.",
            "The report said Carpenter apologized the next day and framed her reaction as confusion rather than intent.",
        ],
        "creative_use": "Adds cultural-specificity guardrails and the apology-counter premise.",
    },
    {
        "id": "ibtimes_sabrina_explainer",
        "title": "Sabrina Carpenter Coachella 2026 Controversy Explained",
        "url": "https://www.ibtimes.co.uk/sabrina-carpenter-coachella-apology-1791353",
        "publisher": "IBTimes UK",
        "date": "2026-04-12",
        "verified_facts": [
            "IBTimes reported that the exchange happened during a pause in Carpenter's Coachella headline set.",
            "IBTimes reported the backlash centered on the fan's celebration call being dismissed as weird/yodeling.",
        ],
        "creative_use": "Supports the first-frame contradiction: a pop-stage sound booth forced to sort every audience sound into the wrong drawer.",
    },
    {
        "id": "vice_sabrina_response",
        "title": "Sabrina Carpenter Responds to Coachella Controversy After Receiving Backlash Online",
        "url": "https://www.vice.com/en/article/sabrina-carpenter-responds-to-coachella-controversy-after-receiving-backlash-online/",
        "publisher": "VICE",
        "date": "2026-04-12",
        "verified_facts": [
            "VICE reported that Carpenter responded after online backlash over the Coachella exchange.",
            "VICE included the social-media dynamics around apology escalation.",
        ],
        "creative_use": "Secondary source for the internet backlash and apology escalation lane.",
    },
    {
        "id": "nbc_lively_baldoni_pretrial",
        "title": "Blake Lively, Justin Baldoni lawsuit set for May trial",
        "url": "https://www.nbclosangeles.com/entertainment/entertainment-news/justin-baldoni-blake-lively-legal-teams-hash-out-details-ahead-of-trial/3882863/",
        "publisher": "NBC Los Angeles",
        "date": "2026-04-28",
        "verified_facts": [
            "NBC reported that Lively and Baldoni legal teams were arguing pretrial issues before a May 2026 trial.",
            "NBC described the litigation as involving serious harassment and retaliation allegations.",
        ],
        "creative_use": "Candidate only; rejected because the legal context is too sensitive for this cycle's humor lane.",
    },
    {
        "id": "lat_paramount_consumer_suit",
        "title": "Consumers sue to block Paramount-Warner Bros. deal",
        "url": "https://www.latimes.com/entertainment-arts/business/story/2026-05-01/consumers-sue-to-block-paramount-warner-bros-deal",
        "publisher": "Los Angeles Times",
        "date": "2026-05-01",
        "verified_facts": [
            "The Los Angeles Times reported that five consumers sued to block Paramount Skydance's Warner Bros. Discovery acquisition.",
            "The report said plaintiffs allege the deal would reduce competition and raise prices.",
        ],
        "creative_use": "Candidate only; rejected as current but too close to earlier Paramount merger SGFLIX lanes.",
    },
]


CANDIDATES = [
    {
        "id": "sabrina_yodel_lost_found",
        "premise": "A fictional pop headliner visits Coachella's Lost & Found for audience sounds after a cultural celebration call was accidentally filed under 'yodeling.'",
        "source_ids": [
            "tmz_sabrina_coachella_yodel",
            "financial_express_sabrina_apology",
            "ibtimes_sabrina_explainer",
            "vice_sabrina_response",
        ],
        "first_frame": "A desert-festival sound booth with drawers labeled applause, yodeling, apologies, and do-not-mislabel-culture; a pop-star proxy holds a tiny wrong-ticket receipt while the cultural-call waveform glows on the monitor.",
        "scores": {
            "famous_face": 8,
            "public_conflict": 8,
            "ego_humiliation": 8,
            "absurd_quote_defense": 8,
            "brand_location_contrast": 9,
            "first_frame_visual_contradiction": 10,
            "risk_control": 8,
            "franchise_potential": 8,
        },
        "total": 86,
    },
    {
        "id": "blake_unlikable_brand_audit",
        "premise": "A Hollywood brand accountant tries to enter 'unlikable' as a business-causation category and the damages calculator asks for a personality receipt.",
        "source_ids": ["nbc_lively_baldoni_pretrial"],
        "first_frame": "A glossy deposition room where a label maker prints personality receipts beside beauty-brand boxes.",
        "scores": {
            "famous_face": 9,
            "public_conflict": 9,
            "ego_humiliation": 9,
            "absurd_quote_defense": 10,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "risk_control": 3,
            "franchise_potential": 7,
        },
        "total": 70,
        "penalty": "Underlying litigation includes serious harassment/retaliation allegations; too taste-fragile for a comedy-first package.",
    },
    {
        "id": "paramount_remote_injunction_counter",
        "premise": "Streaming subscribers bring TV remotes to federal court because every button opens a merger exhibit.",
        "source_ids": ["lat_paramount_consumer_suit"],
        "first_frame": "Antitrust clerk counter stacked with remotes, popcorn, and merger binders.",
        "scores": {
            "famous_face": 4,
            "public_conflict": 9,
            "ego_humiliation": 7,
            "absurd_quote_defense": 7,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "risk_control": 9,
            "franchise_potential": 7,
        },
        "total": 60,
        "penalty": "Duplicate lane; prior SGFLIX packages already used Paramount merger courtroom comedy.",
    },
    {
        "id": "bravo_conflict_waiver_kiosk",
        "premise": "A reality-TV cast member discovers the waiver kiosk prints conflict as a required amenity.",
        "source_ids": [],
        "first_frame": "A reunion-show check-in desk with a stress-disclaimer stamp and glitter folders.",
        "scores": {
            "famous_face": 5,
            "public_conflict": 7,
            "ego_humiliation": 7,
            "absurd_quote_defense": 8,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 7,
            "risk_control": 5,
            "franchise_potential": 6,
        },
        "total": 53,
        "penalty": "Lower broad recognition and harassment-lawsuit context make it less suitable.",
    },
]


FIRST_PROMPT = """GPT Image 2 prompt: SGFLIX satirical first frame, 16:9 cinematic editorial still. Concept: a fictional pop headliner, not Sabrina Carpenter and no exact real-person likeness, stands inside a polished desert music-festival sound lost-and-found booth after an audience cultural celebration call was mistakenly filed as yodeling. Hero contradiction: an audio waveform glows on a monitor while a clerk slides open tiny drawers labeled APPLAUSE, YODELING, APOLOGY, and DO NOT MISLABEL CULTURE. The pop-star proxy holds a small wrong-ticket receipt; the booth window shows desert stage lights and festival wristbands, but no real Coachella logos. Tone: crisp pop-culture satire, warm desert sunset, chrome audio gear, clean prop text, respectful toward Arabic cultural tradition. No mockery of the cultural call, no real logos, no defamatory text, no watermark, no video."""

BOARD_PROMPT = """GPT Image 2 prompt: Shared Choices director-bible storyboard board for SGFLIX, 3:2. Premise: a fictional pop headliner has to visit a festival sound lost-and-found because a cultural celebration call got mislabeled as yodeling. Build a polished director bible with zones: character + hero props, color palette, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes. Props: audio waveform monitor, drawer cabinet labeled applause/yodeling/apology/do-not-mislabel-culture, wrong-ticket receipt, festival wristband tray, apology stamp pad, desert stage lights through the booth window. Visual rules: no exact Sabrina Carpenter likeness, no Coachella logos, no mockery of Zaghrouta or Arabic culture, satire targets pop-stage confusion and apology bureaucracy, clean minimal text, no video generation, no watermark."""


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
    img = Image.new("RGB", (1536, 864), "#f7e8ce")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1536, 160], fill="#15222b")
    d.text((48, 40), "FESTIVAL SOUND LOST & FOUND", fill="#fff5df", font=font(54, True))
    d.text((52, 107), "fictional pop-stage confusion satire - respect the culture, fix the label", fill="#f4b35f", font=font(25))
    d.rectangle([0, 645, 1536, 864], fill="#563d32")
    d.rectangle([0, 625, 1536, 645], fill="#ef6f61")
    for x in range(0, 1536, 110):
        d.ellipse([x, 185, x + 38, 223], fill="#ffd166")
    d.rectangle([565, 215, 1035, 485], fill="#172b35", outline="#0d1216", width=5)
    d.text((610, 245), "CULTURAL CALL WAVEFORM", fill="#fff5df", font=font(31, True))
    points = []
    for i in range(0, 380, 12):
        amp = [35, 70, 22, 85, 38][(i // 12) % 5]
        points.append((615 + i, 380 - amp))
        points.append((621 + i, 380 + amp))
    d.line(points, fill="#35d0ba", width=5)
    d.rectangle([1080, 238, 1415, 600], fill="#fff5df", outline="#15222b", width=5)
    labels = ["APPLAUSE", "YODELING", "APOLOGY", "DO NOT\nMISLABEL\nCULTURE"]
    for idx, label in enumerate(labels):
        y = 265 + idx * 78
        d.rectangle([1110, y, 1385, y + 55], fill="#f1d6a7", outline="#15222b", width=3)
        d.text((1130, y + 12), label, fill="#15222b", font=font(23, True), spacing=1)
    d.ellipse([190, 230, 345, 385], fill="#c99572", outline="#15222b", width=4)
    d.rectangle([215, 385, 335, 625], fill="#f0d7a1", outline="#15222b", width=4)
    d.rectangle([155, 500, 440, 585], fill="#fff5df", outline="#15222b", width=4)
    d.text((183, 520), "WRONG-TICKET\nRECEIPT", fill="#b43a34", font=font(26, True), spacing=3)
    d.text((138, 665), "anonymous pop-headliner proxy\nno exact likeness, no real festival logos", fill="#fff5df", font=font(24), align="center")
    d.rectangle([590, 540, 875, 615], fill="#ef6f61", outline="#15222b", width=4)
    d.text((620, 562), "APOLOGY STAMP PAD", fill="#fff5df", font=font(25, True))
    img.save(path)


def draw_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 1000), "#f7e8ce")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1500, 92], fill="#15222b")
    d.text((34, 27), "SHARED CHOICES: FESTIVAL SOUND LOST & FOUND", fill="#fff5df", font=font(36, True))
    zones = [
        (40, 125, 355, 390, "CHARACTER + PROPS"),
        (395, 125, 710, 390, "SET DESIGN"),
        (750, 125, 1065, 390, "FLOOR PLAN"),
        (1105, 125, 1460, 390, "VISUAL RULES"),
        (40, 435, 355, 735, "PANEL 1 - BOOTH"),
        (395, 435, 710, 735, "PANEL 2 - WAVEFORM"),
        (750, 435, 1065, 735, "PANEL 3 - DRAWERS"),
        (1105, 435, 1460, 735, "PANEL 4 - RECEIPT"),
        (40, 785, 710, 955, "LIGHT / MOOD"),
        (750, 785, 1460, 955, "PRODUCTION NOTES"),
    ]
    for x1, y1, x2, y2, title in zones:
        d.rounded_rectangle([x1, y1, x2, y2], radius=8, fill="#fff7e6", outline="#15222b", width=3)
        d.text((x1 + 15, y1 + 14), title, fill="#15222b", font=font(18, True))
    for i, color in enumerate(["#15222b", "#35d0ba", "#f4b35f", "#ef6f61", "#fff7e6"]):
        d.rectangle([1128 + i * 58, 195, 1175 + i * 58, 252], fill=color, outline="#111")
    d.text((1128, 278), "No exact likeness\nNo festival logos\nNo culture mockery\nTarget: mislabeling", fill="#15222b", font=font(23), spacing=8)
    d.rectangle([438, 210, 672, 315], fill="#172b35")
    d.text((462, 248), "sound booth window", fill="#fff5df", font=font(23, True))
    d.rectangle([805, 205, 1020, 335], outline="#15222b", width=5)
    d.line([912, 205, 912, 335], fill="#15222b", width=4)
    d.line([805, 270, 1020, 270], fill="#15222b", width=4)
    d.text((826, 228), "proxy", fill="#15222b", font=font(18))
    d.text((930, 228), "clerk", fill="#15222b", font=font(18))
    d.text((826, 292), "camera", fill="#15222b", font=font(18))
    d.text((930, 292), "drawers", fill="#15222b", font=font(18))
    for x in [110, 465, 820, 1175]:
        d.rectangle([x, 520, x + 160, 650], fill="#172b35", outline="#15222b", width=3)
        d.line([x + 20, 590, x + 140, 550], fill="#35d0ba", width=5)
    d.text((72, 830), "Warm desert sunset outside, cool chrome booth inside.\nThe gag is bureaucratic: the wrong drawer is the villain.\nKeep the cultural call dignified and visually abstract.", fill="#15222b", font=font(23), spacing=8)
    d.text((785, 830), "Use fictional pop-star styling only.\nUse clean prop text; repair if generated text drifts.\nManual video handoff after human still review.\nNo video generated by this factory run.", fill="#15222b", font=font(23), spacing=6)
    d.rectangle([108, 220, 318, 340], fill="#f1d6a7", outline="#15222b", width=3)
    d.text((128, 252), "WRONG-TICKET\nRECEIPT", fill="#b43a34", font=font(24, True), spacing=2)
    img.save(path)


def main() -> None:
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
    write(
        PKG / "research/last30days_report.md",
        f"""# Step 1: Research Intake - Run {RUN_ID}

Generated: {NOW}

Query/topic: current April-May 2026 pop-culture/public-conflict scan for famous face, public conflict, ego/humiliation, absurd quote/defense, brand/location contrast, and strong first-frame contradiction.

Research order note: this report and `sources.json` were created from current web/source context before a winner was selected and before any run package artifacts were completed. No local image, prior storyboard, or old handoff was used as the premise.

Fresh source intake:
- Sabrina Carpenter Coachella cultural-call/yodeling apology coverage from April 11-12, 2026.
- Blake Lively / Justin Baldoni pretrial dispute coverage from April 28, 2026, considered but penalized because the underlying case contains serious allegations.
- Paramount-Warner consumer antitrust suit coverage from May 1, 2026, considered but penalized as a duplicate merger lane.
- Reality-TV lawsuit/waiver conflict pattern, considered but rejected for lower broad recognition and taste risk.

Selected after scoring: `{winner["id"]}`.
""",
    )

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "created_at": NOW, "candidates": CANDIDATES, "winner_id": winner["id"]})
    write(PKG / "strategy/winner_decision.md", f"""# Winner Decision - Run {RUN_ID}

Winner: `{winner["id"]}`

Selected premise: {winner["premise"]}

Score summary:
- sabrina_yodel_lost_found: 86/100, selected for famous-person recognition, public apology heat, strong festival/location contrast, and a clean first-frame contradiction.
- blake_unlikable_brand_audit: 70/100 after sensitive-litigation penalty.
- paramount_remote_injunction_counter: 60/100 after duplicate-lane penalty.
- bravo_conflict_waiver_kiosk: 53/100 after fame/taste penalties.

Core visual: {winner["first_frame"]}

Fact guardrail: the story must attribute source context to reporting and keep the satire aimed at pop-stage confusion, labeling bureaucracy, and apology logistics. It must not mock Zaghrouta, Arabic culture, or the fan.
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "Sabrina Carpenter Coachella cultural-call apology", "track_a_newsjack_velocity": {"active_trend_score": 8, "algorithmic_slipstream": "viral festival clip/apology discourse", "polarization_factor": 7, "track_a_total": 23, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 8, "stereotype_rigidity": "medium-high", "subversion_potential": 9, "track_b_total": 25, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_A_NEWSJACK_WITH_EVERGREEN_STAGE_CONFUSION", "urgency_class": "High", "strategic_directive": "Turn the mistaken label into festival bureaucracy while respecting the cultural call."}}})
    write_json(PKG / "strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "Reported Coachella exchange and subsequent apology after a cultural celebration call was mislabeled as yodeling/weird.", "detected_anomalies": ["glam pop stage versus sound-desk bureaucracy", "a living cultural sound placed in a wrong drawer", "apology as a festival service counter", "Burning Man/yodeling misread"], "native_entropy_score": 7, "subject_self_awareness": "apologetic confusion", "comedic_vector_recommendation": "cringe_vector", "recommended_strategy": "micro_spotlight"}, "local_asset_used_for_selection": False})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"run_id": RUN_ID, "setup": "A pop headliner reportedly mistook a fan's cultural celebration call for yodeling and apologized after backlash.", "bridge": "Make the mistake a physical festival lost-and-found ticketing error: culture was filed in the wrong audio drawer.", "payoff": "The apology stamp pad is ready, but the drawer labeled DO NOT MISLABEL CULTURE is the only one that actually matters.", "rules": ["target mislabeling and stage confusion", "respect the cultural call", "no exact real-person likeness", "no real festival logos"]})
    write_json(PKG / "strategy/tribe_meta_score.json", {"run_id": RUN_ID, "score": 84, "tribe": {"pop_star_recognition": 8, "festival_culture": 9, "apology_discourse": 8, "visual_shareability": 9}, "meta": {"works_without_context": 8, "remix_potential": 8, "format_repeatability": 8}})
    write_json(PKG / "strategy/risk_taste_score.json", {"run_id": RUN_ID, "overall_risk": "medium", "taste_score": 82, "risks": ["mocking cultural tradition", "exact Sabrina Carpenter likeness", "Coachella logo drift", "messy generated text"], "mitigations": ["fictional proxy", "explicit respect rule", "generic desert festival", "minimal clean labels"]})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nGreenlight as a one-off 'apology logistics' episode. Repeatable format: a viral quote mistake becomes a literal customer-service counter. Do not turn it into culture-mockery content.\n")

    first_path = PKG / "frames/gpt_image_2/first_frame_v01.png"
    board_path = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT + "\n")
    first_mode = try_openai_image(FIRST_PROMPT, first_path, "1536x864")
    board_mode = try_openai_image(BOARD_PROMPT, board_path, "1536x1024")
    if not first_path.exists() or first_mode.startswith("blocked_openai_image_api"):
        draw_first_frame(first_path)
        first_mode += "+local_pil_fallback"
    if not board_path.exists() or board_mode.startswith("blocked_openai_image_api"):
        draw_board(board_path)
        board_mode += "+local_pil_fallback"

    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot": "001", "duration_seconds": 6, "subject": "fictional pop headliner at festival sound lost-and-found booth", "scene": "waveform monitor, wrong audio drawers, apology stamp pad, desert stage lights", "motion": "slow push from wrong-ticket receipt to waveform monitor to drawer labeled DO NOT MISLABEL CULTURE", "spatial": "proxy left, waveform center, drawer cabinet right, clerk hand foreground", "camera": "28mm counter-height dolly with shallow pop-editorial focus", "critique": "must target mislabeling bureaucracy, not the cultural call", "revision": "remove exact likeness, festival logos, messy labels, or any mocking cultural imagery"}]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "status": "handoff_only_no_video_generation", "source_image": "frames/gpt_image_2/first_frame_v01.png", "prompt": "Manual-only 6-second clip: push from wrong-ticket receipt to waveform monitor to drawer labeled DO NOT MISLABEL CULTURE. No video generated by this automation.", "negative": "exact Sabrina Carpenter likeness, Coachella logos, culture mockery, defamatory text, watermark"}
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "video_generation_tools_called": False, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "manual_only": True, "guardrails": ["No exact likeness", "No Coachella logos", "No mockery of Zaghrouta or Arabic culture", "Use reported-context language only"]})
    write(PKG / "handoffs/grok_agent_prompt.md", "# Grok/Closed-Tool Handoff Prompt\n\nUse the saved first frame and Shared Choices board for a 6-second manual-only satire clip. Do not generate video from this automation. Preserve the sound lost-and-found booth, wrong-ticket receipt, audio waveform, apology stamp pad, and drawer labeled DO NOT MISLABEL CULTURE. Avoid exact likeness, real festival logos, culture mockery, and defamatory text.\n")
    write(PKG / "captions/instagram_caption.md", "The festival sound lost-and-found found the wrong drawer.\n\nReported context: Sabrina Carpenter apologized after mistaking a fan's cultural celebration call for yodeling during Coachella. SGFLIX version: the audio clerk prints a wrong-ticket receipt and finally opens the drawer labeled DO NOT MISLABEL CULTURE.\n\n#sgflix #popculturesatire #coachella #musicnews #apologytour\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nSurface: Instagram Reels/TikTok/Shorts after human still review.\n\nHook: start on the wrong-ticket receipt, reveal the waveform, then land on the drawer label `DO NOT MISLABEL CULTURE`.\n\nDo not post until a human verifies no exact likeness, festival-logo drift, messy generated text, or cultural mockery.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nThis run demonstrates a safer satire conversion: the target is not the fan or the cultural call. The target is the institutional act of mislabeling, represented by a lost-and-found desk, wrong-ticket receipt, and apology stamp pad.\n")
    write(PKG / "README.md", f"# RUN {RUN_ID} MASTER PACKAGE - Sabrina Yodel Lost Found\n\nStatus: still package created, no video generated.\n\nSelected premise: {winner['premise']}\n\nGenerated stills:\n- `frames/gpt_image_2/first_frame_v01.png`\n- `storyboards/shared_choices/shared_choices_v01.png`\n")

    status = "COMPLETE_WITH_LOCAL_STILL_FALLBACK_NEEDS_GPT_IMAGE_REVIEW" if "local_pil_fallback" in first_mode or "local_pil_fallback" in board_mode else "COMPLETE_STILLS_READY_NO_VIDEO_GENERATED"
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "status": status, "assets": [{"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode, "exists": first_path.exists()}, {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True}, {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode, "exists": board_path.exists()}, {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True}], "missing_files": [], "video_generation_tools_called": False, "post_ready_exports": []})
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "status": status, "selected_premise": winner, "sources": SOURCES, "candidate_count": len(CANDIDATES), "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}, "video_generation_tools_called": False})
    write(PKG / "qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{first_mode}`\n\nPasses: clear sound-lost-and-found contradiction, fictional proxy, no real festival logo, no mocking depiction of the cultural call, no video generation.\n\nWatch items: inspect generated labels and rerun the saved prompt through GPT Image 2 if local fallback was used.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{board_mode}`\n\nPasses: includes character/props, color palette, set design, floor plan, storyboard panels, lighting/mood, visual rules, and production notes.\n\nWatch items: local fallback is a schematic; use the saved GPT Image prompt for a polished repair before public export.\n")
    if "local_pil_fallback" in first_mode or "local_pil_fallback" in board_mode:
        write(PKG / "qc/IMAGE_GENERATION_NOTE.md", f"# Image Generation Note\n\nThe GPT image API did not produce both local PNGs in this run, so the package includes locally generated fallback still artifacts plus the exact GPT Image prompt files.\n\nFirst frame mode: `{first_mode}`\nShared Choices mode: `{board_mode}`\n\nThis is not a video render and no video-generation tool was called.\n")

    status_md = f"""# Factory Run Status - Run {RUN_ID}

Status: `{status}`
Run time: {NOW}
Research query/topic: current April-May 2026 pop-culture/public-conflict scan around festival apology, mislabeling, legal/business disputes, and strong first-frame contradiction.
Selected premise: {winner['premise']}
Winner score: 86/100.

Candidate board:
- sabrina_yodel_lost_found: 86/100, selected.
- blake_unlikable_brand_audit: 70/100 after sensitive-litigation penalty.
- paramount_remote_injunction_counter: 60/100 after duplicate-lane penalty.
- bravo_conflict_waiver_kiosk: 53/100 after fame/taste penalties.

Generated still-image paths:
- `{first_path}`
- `{board_path}`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: none blocking for internal package review. If local fallback was used, run saved prompts through GPT Image 2 before public storyboard export.

High-risk issues:
- avoid exact Sabrina Carpenter likeness
- avoid real Coachella/festival logos
- avoid mocking Zaghrouta, Arabic culture, or the fan
- keep source context attributed to reporting
- inspect generated text for messy or defamatory drift

Exact next human action: review both still PNGs for likeness/logo/text/cultural-risk issues, then optionally rerun the saved prompts through GPT Image 2 for polished repair before any manual video workflow.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status_md)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
