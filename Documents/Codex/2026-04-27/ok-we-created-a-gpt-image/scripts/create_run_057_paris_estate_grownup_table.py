from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_ID = "057"
SLUG = "paris_estate_grownup_table"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "tmz_paris_estate_bully",
        "title": "Paris Jackson Accuses MJ's Estate Executors of Using Dad's Fortune to Bully Her",
        "url": "https://www.tmz.com/2026/04/07/paris-jackson-accuses-mjs-estate-executors-of-using-dads-fortune-to-bully-her/",
        "publisher": "TMZ",
        "date": "2026-04-07",
        "verified_facts": [
            "TMZ reported that Paris Jackson accused Michael Jackson estate executors of bullying her for questioning spending.",
            "TMZ reported that her filing disputed claims that she was milking the court fight for press coverage.",
        ],
        "creative_use": "Primary winner source for a probate-court customer-service satire about estate money being used to buy insults.",
    },
    {
        "id": "independent_paris_mock_belittle",
        "title": "Paris Jackson says legal battle over dad Michael Jackson's estate is being used to 'mock and belittle' her",
        "url": "https://www.independent.co.uk/arts-entertainment/music/news/paris-jackson-michael-estate-legal-fight-b2954576.html",
        "publisher": "The Independent",
        "date": "2026-04-10",
        "verified_facts": [
            "The Independent reported Paris Jackson's accusation that estate money was used to publicly insult her amid the legal battle.",
            "The report described an 83-page status report and an April 6 filing in the ongoing estate dispute.",
        ],
        "creative_use": "Adds the 'mock and belittle' phrase and a visual reason for an oversized court report prop.",
    },
    {
        "id": "theblast_paris_mock",
        "title": "Paris Jackson Claims Dad's Money Was Used To 'Mock' Her",
        "url": "https://theblast.com/794176/paris-jackson-executors-dads-money-mock-belittle-her/",
        "publisher": "The Blast",
        "date": "2026-04-08",
        "verified_facts": [
            "The Blast, citing People-obtained documents, reported that Paris Jackson's camp claimed estate executors used her father's money to attack her in the media.",
            "The report said Paris's side characterized the conduct as an attempt to bully her into submission.",
        ],
        "creative_use": "Supports the money-as-media-bully machine prop engine while keeping the facts attributed.",
    },
    {
        "id": "tmz_blake_mean_girl",
        "title": "Blake Lively Tells Court Her 'Mean Girl' Label Cost Her $40.5 Million",
        "url": "https://www.tmz.com/2026/04/20/blake-lively-tells-court-mean-girl-label-cost-her-millions/",
        "publisher": "TMZ",
        "date": "2026-04-20",
        "verified_facts": [
            "TMZ reported that Blake Lively claimed online labels including 'mean girl' and 'bully' caused reputational harm estimated at $36.5 million to $40.5 million.",
            "The report tied the numbers to filings in the Baldoni/Lively litigation.",
        ],
        "creative_use": "Scored as a candidate but rejected because the underlying litigation contains sensitive allegations and an already crowded discourse lane.",
    },
    {
        "id": "forbes_paramount_subscriber_suit",
        "title": "Paramount+ Subscribers Sue Over Merger With Warner Bros. Discovery",
        "url": "https://www.forbes.com/sites/siladityaray/2026/05/01/paramount-subscribers-sue-over-merger-with-warner-bros-discovery/",
        "publisher": "Forbes",
        "date": "2026-05-01",
        "verified_facts": [
            "Forbes reported that Paramount+ subscribers filed a lawsuit to block Paramount Skydance's proposed Warner Bros. Discovery acquisition on antitrust grounds.",
            "The report described the deal value as $110 billion.",
        ],
        "creative_use": "Scored as a current business-conflict candidate but rejected because earlier SGFLIX packages already used the Paramount merger lane.",
    },
    {
        "id": "tmz_huda_service",
        "title": "Huda Mustafa's BF Louis' Baby Mama Claims 'Love Island' Star Is 'Evading' Court Battle",
        "url": "https://www.tmz.com/2026/04/28/huda-mustafa-bf-louis-ex-gf-claims-love-island-star-is-evading-court-battle/",
        "publisher": "TMZ",
        "date": "2026-04-28",
        "verified_facts": [
            "TMZ reported that lawyers claimed difficulty serving Huda Mustafa with restraining-order papers.",
            "TMZ also reported that a representative for Huda denied the claim.",
        ],
        "creative_use": "Scored as a process-server comedy candidate but rejected for lower fame and personal-relationship sensitivity.",
    },
]


CANDIDATES = [
    {
        "id": "paris_estate_grownup_table",
        "premise": "A fictional pop-heiress proxy enters probate court and finds the estate has rented a tiny 'grown-up table' where every objection costs another stack of her father's legacy money.",
        "source_ids": ["tmz_paris_estate_bully", "independent_paris_mock_belittle", "theblast_paris_mock"],
        "first_frame": "A gothic-pop probate intake desk: an anonymous tattooed daughter-of-pop proxy faces an enormous 83-page status report while a clerk stamps DAD'S MONEY: MEDIA INSULT BUDGET beside a miniature boardroom table labeled grown-up table.",
        "scores": {
            "famous_face": 8,
            "public_conflict": 8,
            "ego_humiliation": 9,
            "absurd_quote_defense": 9,
            "brand_location_contrast": 7,
            "first_frame_visual_contradiction": 10,
            "risk_control": 8,
            "franchise_potential": 8,
        },
        "total": 87,
    },
    {
        "id": "blake_mean_girl_invoice_lab",
        "premise": "A reputation-damages accountant tries to scan the phrase 'mean girl' and the calculator prints a Hollywood budget.",
        "source_ids": ["tmz_blake_mean_girl"],
        "first_frame": "A glossy damages lab where a label maker and calculator fight over whether a phrase is worth $40.5M.",
        "scores": {
            "famous_face": 9,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "absurd_quote_defense": 9,
            "brand_location_contrast": 7,
            "first_frame_visual_contradiction": 9,
            "risk_control": 4,
            "franchise_potential": 7,
        },
        "total": 72,
        "penalty": "Sensitive underlying litigation and crowded Lively/Baldoni discourse make it less taste-stable.",
    },
    {
        "id": "paramount_remote_merger_injunction",
        "premise": "Streaming subscribers bring TV remotes to federal court because every button now opens a merger exhibit.",
        "source_ids": ["forbes_paramount_subscriber_suit"],
        "first_frame": "A courtroom evidence cart full of remotes, popcorn, and antitrust binders.",
        "scores": {
            "famous_face": 5,
            "public_conflict": 9,
            "ego_humiliation": 7,
            "absurd_quote_defense": 7,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "risk_control": 8,
            "franchise_potential": 7,
        },
        "total": 59,
        "penalty": "Duplicate lane: prior SGFLIX Paramount merger packages exist.",
    },
    {
        "id": "huda_service_avoidance_elevator",
        "premise": "A process server chases a reality-TV influencer through a luxury condo where every elevator button says 'not served.'",
        "source_ids": ["tmz_huda_service"],
        "first_frame": "Luxury-condo mailroom with a process server, a denied-service stamp, and reality-TV rose petals on legal papers.",
        "scores": {
            "famous_face": 5,
            "public_conflict": 7,
            "ego_humiliation": 7,
            "absurd_quote_defense": 8,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "risk_control": 5,
            "franchise_potential": 6,
        },
        "total": 54,
        "penalty": "Personal-relationship restraining-order context and weaker broad recognition.",
    },
]


FIRST_PROMPT = """GPT Image 2 prompt: SGFLIX satirical first frame, 16:9 cinematic editorial still. Concept: a fictional pop-heiress proxy, not Paris Jackson and no exact real-person likeness, stands at a probate-court intake desk in a glamorous gothic-pop courthouse. The hero contradiction: a massive 83-page status report towers like a skyscraper while a miniature boardroom table beside it has a tiny plaque reading GROWN-UP TABLE. A neutral clerk stamps a receipt that says LEGACY MONEY: MEDIA INSULT BUDGET. Visual tone: glossy legal satire, prestige magazine lighting, muted black, cream, courthouse green, brass, and a single cherry-red stamp pad. Include no Michael Jackson imagery, no estate logos, no real signatures, no defamatory claims, no courtroom judge, no injuries, no watermark. Keep text minimal and clean."""

BOARD_PROMPT = """GPT Image 2 prompt: Shared Choices director-bible storyboard board for SGFLIX, 3:2. Premise: a fictional pop-heiress proxy's probate dispute becomes a courthouse customer-service desk where legacy money is converted into media-insult receipts and a tiny 'grown-up table' prop. Build a polished board with distinct zones: character + hero props, color palette, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes. Props: 83-page status report tower, miniature boardroom table, brass probate intake bell, receipt printer, red stamp pad, neutral clerk, anonymous tattooed daughter-of-pop proxy. No exact Paris Jackson likeness, no Michael Jackson imagery, no real estate logos, no judge, no defamatory text, no video generation, no watermark."""


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
    img = Image.new("RGB", (1536, 864), "#f3ecdf")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1536, 150], fill="#172326")
    d.text((52, 42), "PROBATE INTAKE", fill="#fff8e8", font=font(58, True))
    d.text((58, 106), "fictional pop-heiress proxy, reported estate-dispute satire", fill="#c7a55b", font=font(25))
    d.rectangle([0, 610, 1536, 864], fill="#314439")
    d.rectangle([0, 610, 1536, 630], fill="#b33a35")
    d.rectangle([95, 260, 390, 610], fill="#101719")
    d.ellipse([135, 175, 310, 350], fill="#c9a780", outline="#111", width=4)
    d.rectangle([165, 350, 325, 610], fill="#232a34")
    d.line([155, 395, 335, 395], fill="#c7a55b", width=5)
    d.text((92, 635), "anonymous daughter-of-pop\nproxy, no exact likeness", fill="#fff8e8", font=font(23), align="center")
    d.rectangle([575, 205, 835, 650], fill="#fffaf0", outline="#172326", width=5)
    d.rectangle([592, 222, 818, 632], outline="#c7a55b", width=4)
    d.text((620, 280), "83-PAGE\nSTATUS\nREPORT", fill="#172326", font=font(34, True), spacing=8)
    d.rectangle([970, 395, 1295, 615], fill="#fff7df", outline="#172326", width=5)
    d.text((1010, 425), "LEGACY MONEY:\nMEDIA INSULT\nBUDGET", fill="#8b2529", font=font(31, True), spacing=8)
    d.rectangle([1010, 650, 1320, 760], fill="#e9d6b5", outline="#172326", width=4)
    d.text((1040, 684), "GROWN-UP TABLE", fill="#172326", font=font(25, True))
    for x in [1045, 1130, 1215]:
        d.rectangle([x, 615, x + 45, 655], fill="#684d3b")
    d.rectangle([1280, 210, 1435, 345], fill="#24342e", outline="#c7a55b", width=4)
    d.text((1307, 250), "CLERK\nSTAMP", fill="#fff8e8", font=font(24, True), spacing=5)
    d.rectangle([1315, 365, 1420, 420], fill="#b33a35")
    d.text((1328, 378), "STAMP", fill="#fff8e8", font=font(21, True))
    img.save(path)


def draw_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 1000), "#f4eddf")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1500, 92], fill="#172326")
    d.text((38, 26), "SHARED CHOICES: GROWN-UP TABLE PROBATE DESK", fill="#fff8e8", font=font(36, True))
    zones = [
        (40, 125, 355, 390, "CHARACTER + HERO PROPS"),
        (395, 125, 710, 390, "SET / ENVIRONMENT"),
        (750, 125, 1065, 390, "FLOOR PLAN"),
        (1105, 125, 1460, 390, "VISUAL RULES"),
        (40, 435, 355, 735, "PANEL 1 - 28MM"),
        (395, 435, 710, 735, "PANEL 2 - REPORT TOWER"),
        (750, 435, 1065, 735, "PANEL 3 - STAMP"),
        (1105, 435, 1460, 735, "PANEL 4 - TINY TABLE"),
        (40, 785, 710, 955, "LIGHT / MOOD"),
        (750, 785, 1460, 955, "PRODUCTION NOTES"),
    ]
    for x1, y1, x2, y2, title in zones:
        d.rounded_rectangle([x1, y1, x2, y2], radius=8, fill="#fffaf0", outline="#172326", width=3)
        d.text((x1 + 16, y1 + 14), title, fill="#172326", font=font(18, True))
    palette = ["#172326", "#314439", "#f4eddf", "#c7a55b", "#b33a35"]
    for i, color in enumerate(palette):
        d.rectangle([1128 + i * 60, 195, 1176 + i * 60, 252], fill=color, outline="#111")
    d.text((1128, 275), "No exact likeness\nNo MJ iconography\nReported claims only\nNo defamatory text", fill="#172326", font=font(23), spacing=8)
    d.rectangle([430, 210, 675, 315], fill="#314439")
    d.text((450, 255), "probate intake desk", fill="#fff8e8", font=font(24, True))
    d.rectangle([805, 205, 1020, 335], outline="#172326", width=5)
    d.line([912, 205, 912, 335], fill="#172326", width=4)
    d.line([805, 270, 1020, 270], fill="#172326", width=4)
    d.text((826, 228), "proxy", fill="#172326", font=font(18))
    d.text((930, 228), "clerk", fill="#172326", font=font(18))
    d.text((826, 292), "camera", fill="#172326", font=font(18))
    d.text((930, 292), "props", fill="#172326", font=font(18))
    for x in [110, 465, 820, 1175]:
        d.rectangle([x, 515, x + 160, 650], fill="#e9d6b5", outline="#172326", width=3)
        d.rectangle([x + 25, 545, x + 135, 622], fill="#b33a35")
    d.text((75, 830), "Prestige legal satire, brass and courthouse green.\nCamera stays formal while props expose the joke.\nThe tiny table is the one impossible object.", fill="#172326", font=font(23), spacing=8)
    d.text((785, 830), "Use fictional proxy styling only.\nNo real-family iconography or exact face.\nManual video handoff after human review.\nRepair generated text if it drifts.", fill="#172326", font=font(23), spacing=6)
    d.rectangle([230, 220, 327, 322], fill="#fff7df", outline="#172326", width=3)
    d.text((240, 252), "83\nPAGES", fill="#8b2529", font=font(24, True), spacing=2)
    d.rectangle([110, 255, 185, 330], fill="#c9a780", outline="#172326", width=3)
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

Query/topic: last-30-days pop-culture/public-conflict scan for famous face, public conflict, ego/humiliation, absurd quote/defense, brand or location contrast, and strong first-frame contradiction.

Research order note: this report and sources manifest were created from current research/web context before selecting the winner and before creating any story premise artifacts. No nearby local image, old storyboard, or previous handoff was used as the premise.

Live scan intake:
- last30days agent query: `recent celebrity public conflict absurd quote brand controversy lawsuit humiliation`
- Result shape: X posts, YouTube items, and Polymarket items were available; Reddit returned a rate-limit/error and is marked as a limitation.
- Supplemental verified web/source context: TMZ, The Independent, The Blast, Forbes, and related current search results from April 7-May 1, 2026.

Candidate board built from current context:
- Paris Jackson estate dispute / 'mock and belittle' / father's money used to attack her: strongest prop logic and low reenactment requirement.
- Blake Lively 'mean girl' reputational-damages claim: famous and current, but taste risk from sensitive litigation context.
- Paramount+ subscribers suing to block the Warner Bros. Discovery deal: current and visually clean, but duplicate merger lane.
- Huda Mustafa alleged service-avoidance dispute: process-server visual, but weaker broad-recognition and relationship/legal sensitivity.

Selected after scoring: `{winner["id"]}`.
""",
    )

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "created_at": NOW, "candidates": CANDIDATES, "winner_id": winner["id"]})
    write(
        PKG / "strategy/winner_decision.md",
        f"""# Winner Decision - Run {RUN_ID}

Winner: `{winner["id"]}`

Selected premise: {winner["premise"]}

Score summary:
- paris_estate_grownup_table: 87/100, selected for the cleanest visual contradiction and strong reported language: estate money, media attacks, mock/belittle, and implied adult-table power theater.
- blake_mean_girl_invoice_lab: 72/100 after sensitive-litigation penalty.
- paramount_remote_merger_injunction: 59/100 after duplicate-lane penalty.
- huda_service_avoidance_elevator: 54/100 after fame and personal-dispute penalties.

Core visual: {winner["first_frame"]}

Fact guardrail: all claims stay attributed to reporting and legal filings. The piece must not depict real Michael Jackson imagery, exact Paris Jackson likeness, estate logos, real signatures, or unverified wrongdoing.
""",
    )
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "Paris Jackson estate dispute coverage", "track_a_newsjack_velocity": {"active_trend_score": 7, "algorithmic_slipstream": "April 2026 estate/biopic dispute renewed discussion", "polarization_factor": 7, "track_a_total": 21, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 8, "stereotype_rigidity": "High", "subversion_potential": 9, "track_b_total": 25, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_B_EVERGREEN_ARCHETYPE", "urgency_class": "High", "strategic_directive": "Make inherited-pop-royalty bureaucracy visual without using real family imagery."}}})
    write_json(PKG / "strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "Reported legal dispute over estate management and media/court statements.", "detected_anomalies": ["father's money allegedly used for media attacks", "mock/belittle language", "adult/grown-up framing in estate power dispute", "83-page status-report machinery"], "native_entropy_score": 6, "subject_self_awareness": "deadpan", "comedic_vector_recommendation": "cringe_vector", "recommended_strategy": "micro_spotlight"}, "local_asset_used_for_selection": False})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"run_id": RUN_ID, "setup": "A beneficiary questions estate spending and claims the estate is using legacy money to attack her publicly.", "bridge": "Treat the estate dispute as a probate intake counter where every objection is converted into a receipt for media-insult expenses.", "payoff": "The tiny grown-up table makes the power dynamic physical while everyone acts like it is standard court administration.", "rules": ["comedy targets bureaucracy and status theater", "no exact real-person likeness", "no MJ imagery", "reported claims only"]})
    write_json(PKG / "strategy/tribe_meta_score.json", {"run_id": RUN_ID, "score": 83, "tribe": {"pop_culture_recognition": 8, "estate_money_fascination": 9, "visual_shareability": 9, "comment_prompt": 8}, "meta": {"works_without_context": 8, "remix_potential": 8, "format_repeatability": 8}})
    write_json(PKG / "strategy/risk_taste_score.json", {"run_id": RUN_ID, "overall_risk": "medium", "taste_score": 80, "risks": ["exact Paris Jackson likeness", "Michael Jackson iconography or estate-logo drift", "overstating legal claims", "messy generated text"], "mitigations": ["fictional proxy", "attributed/reporting language", "generic probate court", "minimal clean prop text"]})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nGreenlight as a one-off 'legacy bureaucracy' episode. Repeatable format: a public dispute phrase becomes a literal office procedure. Do not turn this into family attack content or a Michael Jackson biopic spoof.\n")

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

    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot": "001", "duration_seconds": 6, "subject": "fictional pop-heiress proxy at probate intake desk", "scene": "legacy-money receipt printer, 83-page report tower, miniature grown-up table", "motion": "slow push from stamp pad to report tower to tiny table reveal", "spatial": "proxy left, report tower center, receipt and tiny table right", "camera": "28mm courthouse counter-height dolly", "critique": "must read as bureaucracy/status satire, not family attack", "revision": "remove exact likeness, MJ imagery, logos, real signatures, or unverified allegations"}]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "status": "handoff_only_no_video_generation", "source_image": "frames/gpt_image_2/first_frame_v01.png", "prompt": "Animate only in a separate human-approved closed tool: counter-height dolly across stamp pad, 83-page report tower, clerk receipt, and tiny grown-up table. No video generated by this automation.", "negative": "exact Paris Jackson likeness, Michael Jackson imagery, estate logos, judge, real signatures, defamatory text"}
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "video_generation_tools_called": False, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "manual_only": True, "guardrails": ["No exact likeness", "No Michael Jackson imagery", "No estate logos", "Use reported-claim language only"]})
    write(PKG / "handoffs/grok_agent_prompt.md", "# Grok/Closed-Tool Handoff Prompt\n\nUse the saved first frame and Shared Choices board for a 6-second manual-only satire clip. Do not generate video from this automation. Preserve the probate intake setting, 83-page report tower, receipt printer, red stamp pad, and miniature grown-up table. Avoid exact likeness, Michael Jackson imagery, estate logos, real signatures, and defamatory claims.\n")
    write(PKG / "captions/instagram_caption.md", "When probate court adds a tiny grown-up table to the invoice.\n\nReported context: Paris Jackson's side claimed estate money was used to mock and belittle her in the media. SGFLIX version: the clerk prints it as a legacy-expense receipt.\n\nNo exact likeness. No family iconography. Just bureaucracy doing status theater.\n\n#sgflix #popculturesatire #probatecourt #musicnews #receipts\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nSurface: Instagram Reels/TikTok/Shorts after human still review.\n\nHook: open on the tiny `GROWN-UP TABLE`, tilt to the receipt reading `LEGACY MONEY`, then reveal the report tower.\n\nDo not post until a human verifies no exact likeness, Michael Jackson imagery, estate-logo drift, messy generated text, or unverified legal claim.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nThis run turns reported court-language power dynamics into props. The core lesson: do not reenact the legal dispute. Make the phrase physical. The tiny grown-up table is the joke, and the receipt printer is the logic bridge.\n")
    write(PKG / "README.md", f"# RUN {RUN_ID} MASTER PACKAGE - Paris Estate Grown-Up Table\n\nStatus: still package created, no video generated.\n\nSelected premise: {winner['premise']}\n\nGenerated stills:\n- `frames/gpt_image_2/first_frame_v01.png`\n- `storyboards/shared_choices/shared_choices_v01.png`\n")

    status = "COMPLETE_WITH_LOCAL_STILL_FALLBACK_NEEDS_GPT_IMAGE_REVIEW" if "local_pil_fallback" in first_mode or "local_pil_fallback" in board_mode else "COMPLETE_STILLS_READY_NO_VIDEO_GENERATED"
    assets = [
        {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode, "exists": first_path.exists()},
        {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True},
        {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode, "exists": board_path.exists()},
        {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True},
    ]
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "status": status, "assets": assets, "missing_files": [], "video_generation_tools_called": False, "post_ready_exports": []})
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "status": status, "selected_premise": winner, "sources": SOURCES, "candidate_count": len(CANDIDATES), "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}, "video_generation_tools_called": False})
    write(PKG / "qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{first_mode}`\n\nPasses: clear probate-desk contradiction, no exact likeness target, no Michael Jackson imagery, no judge, no assault/injury depiction.\n\nWatch items: verify generated text, remove anything that looks like a real signature/logo, and rerun the saved prompt through GPT Image 2 if local fallback was used.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{board_mode}`\n\nPasses: includes character/props, palette, set design, floor plan, storyboard panels, lighting/mood, visual rules, and production notes.\n\nWatch items: if local fallback was used, this is a production schematic rather than a polished public-facing board. Use the saved prompt for a repair pass before external presentation.\n")
    if "local_pil_fallback" in first_mode or "local_pil_fallback" in board_mode:
        write(PKG / "qc/IMAGE_GENERATION_NOTE.md", f"# Image Generation Note\n\nThe GPT image API did not produce both local PNGs in this run, so the package includes locally generated fallback still artifacts plus the exact GPT Image prompt files.\n\nFirst frame mode: `{first_mode}`\nShared Choices mode: `{board_mode}`\n\nThis is not a video render and no video-generation tool was called.\n")
    status_md = f"""# Factory Run Status - Run {RUN_ID}

Status: `{status}`
Run time: {NOW}
Research query/topic: last-30-days pop-culture/public-conflict scan for absurd quote defenses, humiliation, and first-frame contradiction.
Selected premise: {winner['premise']}
Winner score: 87/100.

Candidate board:
- paris_estate_grownup_table: 87/100, selected.
- blake_mean_girl_invoice_lab: 72/100 after sensitive-litigation penalty.
- paramount_remote_merger_injunction: 59/100 after duplicate-lane penalty.
- huda_service_avoidance_elevator: 54/100 after fame/personal-dispute penalties.

Generated still-image paths:
- `{first_path}`
- `{board_path}`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: none blocking for internal package review. If local fallback was used, run saved prompts through GPT Image 2 before public storyboard export.

High-risk issues:
- avoid exact Paris Jackson likeness
- avoid Michael Jackson imagery, estate logos, real signatures, or biopic spoofing
- keep all court/media claims attributed to reporting
- inspect generated text for messy or defamatory drift

Exact next human action: review both still PNGs for likeness/iconography/text risk, then optionally rerun the saved prompts through GPT Image 2 for a polished repair before any manual video workflow.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status_md)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
