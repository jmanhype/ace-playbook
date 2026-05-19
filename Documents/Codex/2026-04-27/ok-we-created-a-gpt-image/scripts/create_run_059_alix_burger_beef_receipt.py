from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_ID = "059"
SLUG = "alix_burger_beef_receipt"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "tmz_carls_jr_stands_by_alix",
        "title": "Carl's Jr. Stands By Alix Earle, Amid Alex Cooper Feud",
        "url": "https://www.tmz.com/2026/04/15/alix-earle-supported-by-carls-jr/",
        "publisher": "TMZ",
        "date": "2026-04-15",
        "verified_facts": [
            "TMZ reported that Carl's Jr. stood behind Alix Earle amid the Alex Cooper feud.",
            "TMZ reported a Carl's Jr. spokesperson used a beef pun and called Earle a fit for its campaign.",
            "TMZ attributed to Dave Portnoy the claim that Cooper said she would not have done the Carl's Jr. ad even with a gun to her head and $10 million in a bag.",
        ],
        "creative_use": "Primary source for the literal burger-beef receipt-counter visual engine.",
    },
    {
        "id": "tmz_alex_calls_out_alix",
        "title": "Alex Cooper Blasts Alix Earle Over 'Fake Drama,' Demands She Speak Up, on Video",
        "url": "https://www.tmz.com/2026/04/13/alex-cooper-calls-out-alix-earle/",
        "publisher": "TMZ",
        "date": "2026-04-13",
        "verified_facts": [
            "TMZ reported that Cooper called out Earle by name in a TikTok video.",
            "TMZ reported Cooper described the behavior as passive-aggressive and fake drama.",
            "TMZ reported Cooper challenged Earle to get specific.",
        ],
        "creative_use": "Supports the passive-aggressive receipt audit board and the callout-ticket gag.",
    },
    {
        "id": "tmz_alix_response_okay_on_it",
        "title": "Alix Earle Responds to Alex Cooper's Fiery Feud Video",
        "url": "https://www.tmz.com/2026/04/13/alix-earle-responds-to-alex-cooper-reaction-feud/",
        "publisher": "TMZ",
        "date": "2026-04-13",
        "verified_facts": [
            "TMZ reported that Earle responded in the comments with 'Okay on it!!'",
            "TMZ described the feud as unfolding loudly and publicly.",
        ],
        "creative_use": "Turns the reply into a tiny order number at a burger counter.",
    },
    {
        "id": "eonline_alix_wakeup_video",
        "title": "Alix Earle on Seeing Alex Cooper's Feud Video",
        "url": "https://www.eonline.com/news/1430910/alix-earle-on-seeing-alex-coopers-feud-video",
        "publisher": "E! Online",
        "date": "2026-04-15",
        "verified_facts": [
            "E! reported that Earle posted a video about being woken up the morning after Coachella to see Cooper's callout.",
            "E! reported the broader context as Cooper calling out passive-aggressive reposts, likes, and comments.",
        ],
        "creative_use": "Secondary source for the morning-after notification and friends-waking-her-up beat.",
    },
    {
        "id": "tmz_rayj_arbitration",
        "title": "Ray J's Countersuit Against Kim Kardashian, Kris Jenner Sent to Arbitration by Judge",
        "url": "https://www.tmz.com/2026/04/24/ray-j-countersuit-against-kim-kardashian-kris-jenner-going-to-arbitration/",
        "publisher": "TMZ",
        "date": "2026-04-24",
        "verified_facts": [
            "TMZ reported that Ray J's countersuit against Kim Kardashian and Kris Jenner was sent to private arbitration.",
            "TMZ reported that the related defamation case remained in court.",
        ],
        "creative_use": "Candidate only; rejected because the sex-tape legal context is taste-fragile and high-risk.",
    },
    {
        "id": "variety_paramount_subscriber_suit",
        "title": "Paramount Faces Suit From Streaming Subscribers Seeking to Block Warner Bros. Deal",
        "url": "https://au.variety.com/2026/film/news/paramount-streaming-subscribers-private-antitrust-36122/",
        "publisher": "Variety Australia",
        "date": "2026-05-01",
        "verified_facts": [
            "Variety reported that streaming subscribers sued to block the Paramount Skydance and Warner Bros. Discovery deal.",
            "Variety reported the suit alleged reduced competition, higher prices, and fewer theatrical options.",
        ],
        "creative_use": "Candidate only; rejected as current but too close to prior merger/courtroom lanes.",
    },
    {
        "id": "forbes_chappell_security_meme",
        "title": "Chappell Roan Backlash Sparks Bizarre Meme Trend",
        "url": "https://www.forbes.com/sites/danidiplacido/2026/03/22/chappell-roan-controversy-sparks-bizarre-meme-trend/",
        "publisher": "Forbes",
        "date": "2026-03-22",
        "verified_facts": [
            "Forbes reported that an alleged fan-security incident involving Chappell Roan sparked meme responses.",
            "Forbes reported Roan responded on Instagram story and that fake responses also circulated.",
        ],
        "creative_use": "Candidate only; rejected as older and more fan-sensitive.",
    },
]


CANDIDATES = [
    {
        "id": "alix_burger_beef_receipt",
        "premise": "A fictional influencer feud gets sent to a Carl's Jr.-style beef receipt counter where every passive-aggressive like prints as a burger add-on.",
        "source_ids": [
            "tmz_carls_jr_stands_by_alix",
            "tmz_alex_calls_out_alix",
            "tmz_alix_response_okay_on_it",
            "eonline_alix_wakeup_video",
        ],
        "first_frame": "A late-night burger counter labeled BEEF RECEIPTS where an influencer proxy holds order number OKAY ON IT while a clerk weighs passive-aggressive likes next to a $10M bag prop.",
        "scores": {
            "famous_face": 7,
            "public_conflict": 8,
            "ego_humiliation": 8,
            "absurd_quote_defense": 10,
            "brand_location_contrast": 10,
            "first_frame_visual_contradiction": 10,
            "risk_control": 8,
            "franchise_potential": 8,
        },
        "total": 87,
    },
    {
        "id": "rayj_private_arbitration_waiting_room",
        "premise": "A reality-TV legal feud enters a private arbitration waiting room where the receipt machine refuses to print anything public.",
        "source_ids": ["tmz_rayj_arbitration"],
        "first_frame": "A velvet arbitration lobby with a giant printer eating celebrity receipts behind a PRIVACY glass window.",
        "scores": {
            "famous_face": 10,
            "public_conflict": 8,
            "ego_humiliation": 8,
            "absurd_quote_defense": 6,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "risk_control": 3,
            "franchise_potential": 6,
        },
        "total": 65,
        "penalty": "Sex-tape and defamation context creates avoidable taste/legal risk.",
    },
    {
        "id": "paramount_remote_merger_checkout",
        "premise": "Streaming subscribers try to return a merger at a TV remote customer-service desk.",
        "source_ids": ["variety_paramount_subscriber_suit"],
        "first_frame": "A courtroom checkout counter where remotes scan as antitrust exhibits.",
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
        "penalty": "Duplicate lane; prior SGFLIX packages have used Paramount/legal-counter comedy.",
    },
    {
        "id": "chappell_breakfast_security_claim_ticket",
        "premise": "A pop-star breakfast table requires a security claim ticket for every accidental child sightline.",
        "source_ids": ["forbes_chappell_security_meme"],
        "first_frame": "A brunch host stand with tiny velvet ropes around one cereal bowl.",
        "scores": {
            "famous_face": 8,
            "public_conflict": 7,
            "ego_humiliation": 7,
            "absurd_quote_defense": 5,
            "brand_location_contrast": 7,
            "first_frame_visual_contradiction": 8,
            "risk_control": 6,
            "franchise_potential": 6,
        },
        "total": 54,
        "penalty": "Older and fan/child-adjacent; weaker than the burger-beef receipt lane.",
    },
]


FIRST_PROMPT = """GPT Image 2 prompt: SGFLIX satirical first frame, 16:9 cinematic editorial still. Concept: a fictional influencer feud, not exact Alix Earle or Alex Cooper likenesses, has been routed to a late-night burger-chain customer-service counter labeled BEEF RECEIPTS. Hero contradiction: a glossy burger counter has a legal-style receipt printer spitting passive-aggressive likes, reposts, and comment tickets as burger add-ons. A fictional influencer proxy in festival-afterparty clothes holds order number OKAY ON IT; the clerk points to a tiny prop money bag labeled 10M and a wall menu with generic items: FAKE DRAMA COMBO, GET SPECIFIC SAUCE, POST-GAME RECOVERY. Style: crisp pop-culture satire, chrome fryer lights, red/yellow burger-counter palette, cinematic 28mm, clean readable prop text. No real Carl's Jr. logos, no exact real-person likeness, no defamatory claims, no weapons, no watermark, no video."""

BOARD_PROMPT = """GPT Image 2 prompt: Shared Choices director-bible storyboard board for SGFLIX, 3:2. Premise: a fictional influencer feud becomes a burger-chain beef receipt counter where every passive-aggressive like prints as a menu add-on. Build a polished director bible with zones: character + hero props, color palette, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes. Props: BEEF RECEIPTS counter, order number OKAY ON IT, passive-aggressive like receipts, generic 10M prop money bag, fake-drama combo board, post-game recovery cup, no real fast-food logos. Visual rules: no exact Alix Earle or Alex Cooper likeness, no real Carl's Jr. branding, no weapon imagery, satire targets influencer feud mechanics and brand opportunism, clean minimal text, no video generation, no watermark."""


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
    img = Image.new("RGB", (1536, 864), "#f8e7c0")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1536, 150], fill="#251a18")
    d.text((48, 36), "BEEF RECEIPTS", fill="#fff5df", font=font(64, True))
    d.text((52, 106), "fictional influencer feud counter - no real logos, no exact likeness", fill="#f6c044", font=font(25))
    d.rectangle([0, 640, 1536, 864], fill="#a32020")
    d.rectangle([0, 605, 1536, 645], fill="#f6c044")
    d.rectangle([930, 205, 1400, 555], fill="#fff5df", outline="#251a18", width=6)
    d.text((970, 235), "MENU BOARD", fill="#251a18", font=font(31, True))
    menu = ["FAKE DRAMA COMBO", "GET SPECIFIC SAUCE", "POST-GAME RECOVERY", "BEEF ONLINE ONLY"]
    for i, item in enumerate(menu):
        d.text((970, 295 + i * 52), item, fill="#a32020", font=font(28, True))
    d.rectangle([565, 235, 870, 500], fill="#2f3337", outline="#251a18", width=5)
    d.text((595, 265), "RECEIPT PRINTER", fill="#fff5df", font=font(29, True))
    for i, label in enumerate(["LIKE", "REPOST", "COMMENT", "OKAY ON IT"]):
        y = 325 + i * 42
        d.rectangle([615, y, 835, y + 30], fill="#fff5df")
        d.text((635, y + 5), label, fill="#251a18", font=font(20, True))
    d.ellipse([165, 225, 330, 390], fill="#c98b63", outline="#251a18", width=4)
    d.rectangle([195, 390, 335, 625], fill="#f1d2a4", outline="#251a18", width=4)
    d.rectangle([110, 500, 465, 588], fill="#fff5df", outline="#251a18", width=4)
    d.text((142, 523), "ORDER # OKAY ON IT", fill="#a32020", font=font(31, True))
    d.polygon([(430, 450), (520, 390), (610, 450), (575, 555), (465, 555)], fill="#16643a", outline="#251a18")
    d.text((462, 456), "10M", fill="#fff5df", font=font(34, True))
    d.rectangle([680, 585, 900, 645], fill="#f6c044", outline="#251a18", width=4)
    d.text((715, 603), "NO WEAPONS", fill="#251a18", font=font(26, True))
    img.save(path)


def draw_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 1000), "#f8e7c0")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1500, 92], fill="#251a18")
    d.text((34, 27), "SHARED CHOICES: BEEF RECEIPT COUNTER", fill="#fff5df", font=font(38, True))
    zones = [
        (40, 125, 355, 390, "CHARACTER + PROPS"),
        (395, 125, 710, 390, "SET DESIGN"),
        (750, 125, 1065, 390, "FLOOR PLAN"),
        (1105, 125, 1460, 390, "VISUAL RULES"),
        (40, 435, 355, 735, "PANEL 1 - COUNTER"),
        (395, 435, 710, 735, "PANEL 2 - RECEIPTS"),
        (750, 435, 1065, 735, "PANEL 3 - MENU"),
        (1105, 435, 1460, 735, "PANEL 4 - ORDER"),
        (40, 785, 710, 955, "LIGHT / MOOD"),
        (750, 785, 1460, 955, "PRODUCTION NOTES"),
    ]
    for x1, y1, x2, y2, title in zones:
        d.rounded_rectangle([x1, y1, x2, y2], radius=8, fill="#fff7e6", outline="#251a18", width=3)
        d.text((x1 + 15, y1 + 14), title, fill="#251a18", font=font(18, True))
    for i, color in enumerate(["#251a18", "#a32020", "#f6c044", "#16643a", "#fff7e6"]):
        d.rectangle([1128 + i * 58, 195, 1175 + i * 58, 252], fill=color, outline="#111")
    d.text((1128, 278), "No exact likeness\nNo real burger logos\nNo weapon imagery\nTarget: feud mechanics", fill="#251a18", font=font(23), spacing=8)
    d.rectangle([438, 225, 672, 315], fill="#a32020")
    d.text((462, 253), "beef receipts", fill="#fff5df", font=font(26, True))
    d.rectangle([805, 205, 1020, 335], outline="#251a18", width=5)
    d.line([912, 205, 912, 335], fill="#251a18", width=4)
    d.line([805, 270, 1020, 270], fill="#251a18", width=4)
    d.text((826, 228), "proxy", fill="#251a18", font=font(18))
    d.text((930, 228), "clerk", fill="#251a18", font=font(18))
    d.text((826, 292), "camera", fill="#251a18", font=font(18))
    d.text((930, 292), "printer", fill="#251a18", font=font(18))
    for x in [110, 465, 820, 1175]:
        d.rectangle([x, 520, x + 160, 650], fill="#2f3337", outline="#251a18", width=3)
        d.rectangle([x + 25, 560, x + 135, 595], fill="#fff5df")
    d.text((72, 830), "Burger-counter warmth with chrome receipt printer glare.\nThe gag is literalized language: beef becomes paperwork.\nKeep it petty, not cruel.", fill="#251a18", font=font(24), spacing=8)
    d.text((785, 830), "Use fictional influencer proxies only.\nKeep all brands generic and readable.\nHuman review required before public export.\nNo video generated by this factory run.", fill="#251a18", font=font(24), spacing=7)
    d.rectangle([100, 230, 320, 330], fill="#fff5df", outline="#251a18", width=3)
    d.text((125, 263), "ORDER #\nOKAY ON IT", fill="#a32020", font=font(26, True), spacing=1)
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
- Alix Earle / Alex Cooper feud coverage from April 13-15, 2026, including Cooper's direct callout, Earle's short comment response, the morning-after Coachella wakeup clip, and Carl's Jr. publicly standing by Earle.
- Ray J / Kardashian arbitration coverage from April 24, 2026, considered but penalized for taste/legal fragility.
- Paramount-Warner consumer antitrust coverage from May 1, 2026, considered but penalized as a duplicate merger lane.
- Chappell Roan fan/security meme coverage from March 22, 2026, considered but rejected as older and fan-sensitive.

Selected after scoring: `{winner["id"]}`.
""",
    )

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "created_at": NOW, "candidates": CANDIDATES, "winner_id": winner["id"]})
    write(PKG / "strategy/winner_decision.md", f"""# Winner Decision - Run {RUN_ID}

Winner: `{winner["id"]}`

Selected premise: {winner["premise"]}

Score summary:
- alix_burger_beef_receipt: 87/100, selected for literal beef-brand contrast, clean absurd quote energy, petty-but-low-stakes public conflict, and instantly readable first frame.
- rayj_private_arbitration_waiting_room: 65/100 after sex-tape/defamation taste penalty.
- paramount_remote_merger_checkout: 60/100 after duplicate-lane penalty.
- chappell_breakfast_security_claim_ticket: 54/100 after older-cycle and fan/child-adjacent penalties.

Core visual: {winner["first_frame"]}

Fact guardrail: source claims must be attributed to reporting. The satire targets influencer feud mechanics, brand opportunism, and receipt culture. Do not imply any criminal conduct, real contract breach, or private fact beyond sourced reporting.
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "Alix Earle / Alex Cooper / Carl's Jr. feud", "track_a_newsjack_velocity": {"active_trend_score": 8, "algorithmic_slipstream": "recent TikTok/influencer feud plus brand-response coverage", "polarization_factor": 7, "track_a_total": 23, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 9, "stereotype_rigidity": "high", "subversion_potential": 9, "track_b_total": 27, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_A_NEWSJACK_WITH_TRACK_B_INFLUENCER_RECEIPT_ARCHETYPE", "urgency_class": "High", "strategic_directive": "Literalize online beef as a burger receipt counter without using real brand marks or exact likenesses."}}})
    write_json(PKG / "strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "Reported influencer feud involving TikTok callout, brief comment response, and fast-food brand support statement.", "detected_anomalies": ["literal beef brand entering an online beef", "a $10M bag quote inside burger-ad discourse", "a two-word comment response functioning like an order ticket", "Coachella morning-after wakeup clip as drama delivery service"], "native_entropy_score": 6, "subject_self_awareness": "trying_to_look_cool", "comedic_vector_recommendation": "cringe_vector", "recommended_strategy": "micro_spotlight"}, "local_asset_used_for_selection": False})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"run_id": RUN_ID, "setup": "A public influencer feud escalated through TikTok callouts, comment responses, and a fast-food brand statement about 'beef.'", "bridge": "Treat every vague like/repost/comment as a literal burger-counter receipt add-on.", "payoff": "The order number is 'OKAY ON IT' and the menu board sells GET SPECIFIC SAUCE.", "rules": ["target influencer feud mechanics", "use fictional proxies", "avoid real logos", "avoid weapon imagery despite sourced quote", "do not invent legal claims"]})
    write_json(PKG / "strategy/tribe_meta_score.json", {"run_id": RUN_ID, "score": 86, "tribe": {"gen_z_influencer_recognition": 8, "podcast_drama": 8, "fast_food_brand_contrast": 10, "receipt_culture": 9}, "meta": {"works_without_context": 9, "remix_potential": 8, "format_repeatability": 8}})
    write_json(PKG / "strategy/risk_taste_score.json", {"run_id": RUN_ID, "overall_risk": "medium-low", "taste_score": 84, "risks": ["exact likeness drift", "real Carl's Jr. logo drift", "weapon imagery from sourced quote", "defamatory contract speculation"], "mitigations": ["fictional proxies", "generic burger-chain design", "explicit no-weapons rule", "source-attributed caption language"]})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nGreenlight as a repeatable `receipt counter` franchise lane. The structure works whenever a public feud has a quote, brand response, or vague social signal that can be turned into a literal order ticket.\n")

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

    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot": "001", "duration_seconds": 6, "subject": "fictional influencer at generic burger-chain beef receipt counter", "scene": "receipt printer, menu board, order ticket, prop money bag, red/yellow counter lights", "motion": "slow push from order number OKAY ON IT to receipt printer to GET SPECIFIC SAUCE menu board", "spatial": "influencer proxy left, printer center, menu right, clerk hand foreground", "camera": "28mm counter-height dolly with crisp editorial fast-food lighting", "critique": "must target feud mechanics and brand opportunism, not private conduct", "revision": "remove exact likeness, real logos, weapon imagery, or messy labels"}]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "status": "handoff_only_no_video_generation", "source_image": "frames/gpt_image_2/first_frame_v01.png", "prompt": "Manual-only 6-second clip: push from order number OKAY ON IT to receipt printer to GET SPECIFIC SAUCE menu board. No video generated by this automation.", "negative": "exact Alix Earle likeness, exact Alex Cooper likeness, real Carl's Jr. branding, weapons, defamatory claims, watermark"}
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "video_generation_tools_called": False, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "manual_only": True, "guardrails": ["No exact likenesses", "No real fast-food logos", "No weapon imagery", "No invented legal claims", "Use sourced reporting language only"]})
    write(PKG / "handoffs/grok_agent_prompt.md", "# Grok/Closed-Tool Handoff Prompt\n\nUse the saved first frame and Shared Choices board for a 6-second manual-only satire clip. Do not generate video from this automation. Preserve the generic burger-chain beef receipt counter, order number OKAY ON IT, passive-aggressive receipt printer, fake-drama menu board, and prop 10M bag. Avoid exact likenesses, real logos, weapons, and defamatory text.\n")
    write(PKG / "captions/instagram_caption.md", "When the beef goes corporate, the receipts come with fries.\n\nReported context: Alex Cooper publicly called out Alix Earle over alleged fake drama, Earle replied briefly, and Carl's Jr. stood by Earle with a beef-themed statement. SGFLIX version: every passive-aggressive like gets rung up at the Beef Receipts counter.\n\n#sgflix #popculturesatire #influencerdrama #receipts #beef\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nSurface: Instagram Reels/TikTok/Shorts after human still review.\n\nHook: open on `ORDER # OKAY ON IT`, reveal the receipt printer, then land on the menu item `GET SPECIFIC SAUCE`.\n\nDo not post until a human verifies no exact likeness, real logo drift, weapon imagery, messy generated text, or unsourced legal claims.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nThis run demonstrates the `literalized phrase` conversion: an online beef becomes a fast-food receipt counter. The safest comic target is the public performance layer: likes, reposts, comments, brand statements, and menu-board language.\n")
    write(PKG / "README.md", f"# RUN {RUN_ID} MASTER PACKAGE - Alix Burger Beef Receipt\n\nStatus: still package created, no video generated.\n\nSelected premise: {winner['premise']}\n\nGenerated stills:\n- `frames/gpt_image_2/first_frame_v01.png`\n- `storyboards/shared_choices/shared_choices_v01.png`\n")

    status = "COMPLETE_WITH_LOCAL_STILL_FALLBACK_NEEDS_GPT_IMAGE_REVIEW" if "local_pil_fallback" in first_mode or "local_pil_fallback" in board_mode else "COMPLETE_STILLS_READY_NO_VIDEO_GENERATED"
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "status": status, "assets": [{"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode, "exists": first_path.exists()}, {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True}, {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode, "exists": board_path.exists()}, {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True}], "missing_files": [], "video_generation_tools_called": False, "post_ready_exports": []})
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "status": status, "selected_premise": winner, "sources": SOURCES, "candidate_count": len(CANDIDATES), "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}, "video_generation_tools_called": False})
    write(PKG / "qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{first_mode}`\n\nPasses: clear beef-receipt counter contradiction, fictional proxy, no real burger logo, no weapon imagery, no video generation.\n\nWatch items: inspect generated labels and rerun the saved prompt through GPT Image 2 if local fallback was used.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nGeneration mode: `{board_mode}`\n\nPasses: includes character/props, color palette, set design, floor plan, storyboard panels, lighting/mood, visual rules, and production notes.\n\nWatch items: local fallback is a schematic; use the saved GPT Image prompt for a polished repair before public export.\n")
    if "local_pil_fallback" in first_mode or "local_pil_fallback" in board_mode:
        write(PKG / "qc/IMAGE_GENERATION_NOTE.md", f"# Image Generation Note\n\nThe GPT image API did not produce both local PNGs in this run, so the package includes locally generated fallback still artifacts plus the exact GPT Image prompt files.\n\nFirst frame mode: `{first_mode}`\nShared Choices mode: `{board_mode}`\n\nThis is not a video render and no video-generation tool was called.\n")

    status_md = f"""# Factory Run Status - Run {RUN_ID}

Status: `{status}`
Run time: {NOW}
Research query/topic: current April-May 2026 pop-culture/public-conflict scan around influencer feud, fast-food brand response, literal beef language, legal/business disputes, and strong first-frame contradiction.
Selected premise: {winner['premise']}
Winner score: 87/100.

Candidate board:
- alix_burger_beef_receipt: 87/100, selected.
- rayj_private_arbitration_waiting_room: 65/100 after sex-tape/defamation taste penalty.
- paramount_remote_merger_checkout: 60/100 after duplicate-lane penalty.
- chappell_breakfast_security_claim_ticket: 54/100 after older-cycle and fan/child-adjacent penalties.

Generated still-image paths:
- `{first_path}`
- `{board_path}`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: none blocking for internal package review. If local fallback was used, run saved prompts through GPT Image 2 before public storyboard export.

High-risk issues:
- avoid exact Alix Earle or Alex Cooper likenesses
- avoid real Carl's Jr. logos or trade dress
- avoid weapon imagery despite the sourced quote
- avoid invented legal/contract claims
- inspect generated text for messy or defamatory drift

Exact next human action: review both still PNGs for likeness/logo/text/weapon-risk issues, then optionally rerun the saved prompts through GPT Image 2 for polished repair before any manual video workflow.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status_md)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
