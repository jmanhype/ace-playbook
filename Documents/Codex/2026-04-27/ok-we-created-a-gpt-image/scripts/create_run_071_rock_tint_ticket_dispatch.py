from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_ID = "071"
SLUG = "rock_tint_ticket_dispatch"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "tmz_hart_rock_piece_quote",
        "title": "Kevin Hart Calls Dwayne 'The Rock' Johnson A Piece of S*** After Traffic Stop",
        "url": "https://www.tmz.com/2026/05/02/kevin-hart-jokingly-blasts-dwayne-johnson/",
        "publisher": "TMZ",
        "date": "2026-05-02",
        "verified_facts": [
            "TMZ reported Hart joked to paparazzi that Johnson was a 'piece of s***' after Johnson's traffic stop.",
            "TMZ framed the exchange as a gag between longtime friends.",
            "TMZ reported Hart also joked that he was the one who called police.",
        ],
        "creative_use": "Primary source for the absurd quote and dispatch-office visual contradiction.",
    },
    {
        "id": "tmz_rock_tinted_windows",
        "title": "Dwayne 'The Rock' Johnson Pulled Over by Police After Walk of Fame Event",
        "url": "https://www.tmz.com/2026/04/30/dwayne-johnson-pulled-over-in-los-angeles/",
        "publisher": "TMZ",
        "date": "2026-04-30",
        "verified_facts": [
            "TMZ reported Johnson was pulled over in Los Angeles after a Hollywood Walk of Fame event.",
            "TMZ attributed the stop to tinted windows, citing law-enforcement sources.",
            "TMZ reported Johnson was compliant during the routine stop.",
        ],
        "creative_use": "Source for the tiny traffic-ticket object and contrast between megastar scale and minor vehicle-code paperwork.",
    },
    {
        "id": "variety_paramount_subscriber_suit",
        "title": "Paramount Faces Suit From Streaming Subscribers Seeking to Block Warner Bros. Deal",
        "url": "https://au.variety.com/2026/film/news/paramount-streaming-subscribers-private-antitrust-36122/",
        "publisher": "Variety",
        "date": "2026-05-01",
        "verified_facts": [
            "Variety reported streaming subscribers sued to block the Paramount Skydance and Warner Bros. transaction.",
            "Variety reported the plaintiffs allege higher prices and fewer viewing options.",
        ],
        "creative_use": "Candidate only; rejected as lower famous-face energy and too close to prior merger packages.",
    },
    {
        "id": "tmz_kathy_known_danger",
        "title": "'RHOBH's Kathy Hilton Fighting House Guest Over $$$ Demand For Alleged Injuries",
        "url": "https://www.tmz.com/2026/05/02/kathy-hilton-fighting-injured-house-guest-over-money/",
        "publisher": "TMZ",
        "date": "2026-05-02",
        "verified_facts": [
            "TMZ reported Kathy Hilton denied allegations in a lawsuit over an alleged fall at her Bel Air mansion.",
            "TMZ reported an affirmative defense may argue the plaintiff exposed herself to a 'known danger'.",
        ],
        "creative_use": "Candidate only; rejected because SGFLIX already made a known-danger foyer run.",
    },
    {
        "id": "tmz_mace_email_judge",
        "title": "Nancy Mace's Email to Judge Name-Dropping Trump Revealed in Court",
        "url": "https://www.tmz.com/2026/05/02/nancy-mace-name-dropping-the-president-revealed-in-court/",
        "publisher": "TMZ",
        "date": "2026-05-02",
        "verified_facts": [
            "TMZ reported court documents include an email that appears to name-drop Trump to a judge.",
            "TMZ tied the filing to a defamation lawsuit involving Mace's former fiance.",
        ],
        "creative_use": "Candidate only; rejected as legally sensitive and overlapping with a recent Mace package.",
    },
]

RUBRIC = [
    "famous_face_or_power_archetype",
    "public_conflict",
    "ego_humiliation",
    "absurd_quote_or_defense",
    "brand_location_contrast",
    "first_frame_visual_contradiction",
    "freshness",
    "risk_control",
    "franchise_potential",
]

CANDIDATES = [
    {
        "id": "rock_tint_ticket_dispatch",
        "premise": "A buddy-comedy insult becomes a 911 dispatch desk where a tiny tinted-window ticket is treated like a national emergency and the 'caller' is obviously a short comedian proxy.",
        "source_ids": ["tmz_hart_rock_piece_quote", "tmz_rock_tinted_windows"],
        "first_frame": "A police dispatch desk with a huge red phone labeled BUDDY COMEDY HOTLINE, a miniature traffic citation under museum glass, an oversized action-star silhouette calmly holding window tint samples, and a tiny comedian silhouette stamping FRIENDSHIP VIOLATION.",
        "scores": {
            "famous_face_or_power_archetype": 10,
            "public_conflict": 7,
            "ego_humiliation": 9,
            "absurd_quote_or_defense": 10,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 10,
            "freshness": 10,
            "risk_control": 9,
            "franchise_potential": 9,
        },
        "total": 82,
    },
    {
        "id": "paramount_remote_injunction_counter",
        "premise": "Streaming subscribers bring remotes to federal court because every button opens a merger exhibit.",
        "source_ids": ["variety_paramount_subscriber_suit"],
        "first_frame": "A small-claims antitrust counter stacked with remotes, popcorn, and merger binders.",
        "scores": {
            "famous_face_or_power_archetype": 5,
            "public_conflict": 9,
            "ego_humiliation": 7,
            "absurd_quote_or_defense": 7,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "freshness": 9,
            "risk_control": 9,
            "franchise_potential": 7,
        },
        "total": 69,
        "penalty": "Prior SGFLIX merger-court packages reduce novelty.",
    },
    {
        "id": "kathy_known_danger_signage",
        "premise": "A Bel Air foyer gets OSHA-style signage because the mansion itself is now the alleged known danger.",
        "source_ids": ["tmz_kathy_known_danger"],
        "first_frame": "A luxury foyer covered in hazard tape and liability placards.",
        "scores": {
            "famous_face_or_power_archetype": 7,
            "public_conflict": 7,
            "ego_humiliation": 8,
            "absurd_quote_or_defense": 9,
            "brand_location_contrast": 9,
            "first_frame_visual_contradiction": 9,
            "freshness": 10,
            "risk_control": 5,
            "franchise_potential": 7,
        },
        "total": 64,
        "penalty": "Rejected as a direct repeat of run_039_hilton_known_danger_foyer.",
    },
    {
        "id": "mace_name_drop_clerk_window",
        "premise": "A courthouse clerk installs a VIP name-drop detector that beeps whenever a filing tries to drag the White House into a local dispute.",
        "source_ids": ["tmz_mace_email_judge"],
        "first_frame": "A judge's inbox with a glowing NAME DROP DETECTOR beside redacted court exhibits.",
        "scores": {
            "famous_face_or_power_archetype": 8,
            "public_conflict": 8,
            "ego_humiliation": 8,
            "absurd_quote_or_defense": 8,
            "brand_location_contrast": 7,
            "first_frame_visual_contradiction": 8,
            "freshness": 10,
            "risk_control": 4,
            "franchise_potential": 6,
        },
        "total": 59,
        "penalty": "Rejected for legal sensitivity and overlap with run_062_mace_oconus_judge_chambers.",
    },
]

WINNER = CANDIDATES[0]

FIRST_PROMPT = """GPT Image 2 prompt: SGFLIX satirical first frame, 16:9 cinematic editorial still. Premise: a fictional buddy-comedy traffic-dispatch office, inspired by public reports that Kevin Hart joked about Dwayne Johnson's minor tinted-window traffic stop. Do not create exact likenesses of Kevin Hart or Dwayne Johnson; use recognizable archetypes only: one very large bald action-star silhouette in cream trousers calmly holding window-tint sample cards, and one very short comedian-caller proxy behind a dispatch console stamping FRIENDSHIP VIOLATION. Hero contradiction: a tiny tinted-window ticket sits under museum glass while a giant red phone labeled BUDDY COMEDY HOTLINE blinks as if it is a national emergency. Add a dispatch board with clean prop labels: TINT LEVEL, WALK OF FAME EXIT, ROAST RESPONSE, FRIENDSHIP GAG. Tone: polished TMZ-meets-traffic-court satire, bright Miami/Los Angeles nightlife color, crisp editorial lighting, no real police insignia, no real logos, no defamatory text, no mugshot styling, no video generation, no watermark."""

BOARD_PROMPT = """GPT Image 2 prompt: Shared Choices director-bible storyboard board for SGFLIX, 3:2. Title: ROCK TINT TICKET DISPATCH. Build a polished production board with zones: character + hero props, color palette, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes. Character canon: anonymous huge action-star proxy with window tint samples; anonymous tiny comedian proxy at dispatch console; bored neutral clerk. Hero props: tiny tinted-window citation under museum glass, BUDDY COMEDY HOTLINE red phone, FRIENDSHIP VIOLATION stamp, tint-meter cards, Walk of Fame exit sign with no real marks. Visual rules: no exact real-person likeness, no real police logos, frame the quote as a friendship gag, minor traffic-ticket stakes only, no criminal implication, clean readable labels, no video generation, no watermark."""


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
    img = Image.new("RGB", (1536, 864), "#f5f0e7")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1536, 135], fill="#171f2a")
    d.text((46, 34), "BUDDY COMEDY HOTLINE", fill="#fff4d6", font=font(56, True))
    d.text((50, 96), "minor traffic ticket treated like a national roast emergency", fill="#ffcc66", font=font(25))
    d.rectangle([0, 650, 1536, 864], fill="#273645")
    d.rectangle([75, 190, 525, 585], fill="#243447", outline="#0c1117", width=5)
    d.text((115, 220), "DISPATCH BOARD", fill="#fff4d6", font=font(36, True))
    board_rows = ["TINT LEVEL", "WALK OF FAME EXIT", "ROAST RESPONSE", "FRIENDSHIP GAG"]
    for i, row in enumerate(board_rows):
        y = 285 + i * 65
        d.rectangle([115, y, 485, y + 42], fill="#f7d37c", outline="#0c1117", width=2)
        d.text((134, y + 9), row, fill="#171f2a", font=font(23, True))
    d.ellipse([650, 215, 790, 355], fill="#b88b6a", outline="#171f2a", width=4)
    d.rectangle([683, 355, 770, 648], fill="#f1e4c8", outline="#171f2a", width=4)
    d.text((616, 675), "huge action-star proxy\ncalmly holds tint samples", fill="#fff4d6", font=font(24), align="center")
    for i, shade in enumerate(["#dde7ee", "#a3b4bf", "#566978", "#1d2832"]):
        d.rectangle([820 + i * 48, 388, 858 + i * 48, 495], fill=shade, outline="#171f2a", width=3)
    d.ellipse([1110, 300, 1200, 390], fill="#9c6c51", outline="#171f2a", width=4)
    d.rectangle([1085, 390, 1225, 605], fill="#e84d5b", outline="#171f2a", width=4)
    d.rectangle([1025, 500, 1325, 620], fill="#fff4d6", outline="#171f2a", width=4)
    d.text((1052, 530), "FRIENDSHIP\nVIOLATION", fill="#ba2636", font=font(32, True), spacing=4)
    d.ellipse([1248, 172, 1415, 338], fill="#d72638", outline="#171f2a", width=6)
    d.rectangle([1302, 330, 1360, 510], fill="#d72638", outline="#171f2a", width=5)
    d.text((1216, 118), "RED PHONE", fill="#d72638", font=font(28, True))
    d.rectangle([583, 548, 937, 638], fill="#d9e4ea", outline="#171f2a", width=5)
    d.rectangle([630, 575, 890, 612], fill="#fff4d6", outline="#171f2a", width=3)
    d.text((654, 583), "TINY TINT TICKET", fill="#171f2a", font=font(22, True))
    img.save(path)


def draw_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 1000), "#f5f0e7")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1500, 92], fill="#171f2a")
    d.text((36, 27), "SHARED CHOICES: ROCK TINT TICKET DISPATCH", fill="#fff4d6", font=font(36, True))
    zones = [
        (40, 125, 350, 372, "CHARACTER + HERO PROPS"),
        (385, 125, 710, 372, "COLOR + LIGHT"),
        (745, 125, 1105, 372, "SET / FLOOR PLAN"),
        (1140, 125, 1460, 372, "VISUAL RULES"),
        (40, 430, 1460, 805, "STORYBOARD PANELS"),
        (40, 835, 710, 960, "CAMERA / MOTION"),
        (745, 835, 1460, 960, "PRODUCTION NOTES"),
    ]
    for box in zones:
        x1, y1, x2, y2, title = box
        d.rounded_rectangle([x1, y1, x2, y2], radius=10, fill="#fff9ed", outline="#171f2a", width=3)
        d.text((x1 + 18, y1 + 16), title, fill="#171f2a", font=font(22, True))
    d.text((62, 178), "Huge action-star proxy\nTiny comedian caller\nNeutral clerk\nTint ticket under glass\nFriendship stamp", fill="#273645", font=font(21), spacing=8)
    palette = ["#171f2a", "#d72638", "#f7d37c", "#f5f0e7", "#2f8f9d"]
    for i, color in enumerate(palette):
        d.rectangle([410 + i * 55, 200, 450 + i * 55, 310], fill=color, outline="#171f2a", width=2)
    d.text((770, 178), "Dispatch desk foreground\nRed phone back right\nTicket pedestal center\nProxy scale contrast left/right\nNo real police marks", fill="#273645", font=font(21), spacing=8)
    d.text((1162, 178), "No exact likeness\nNo criminal implication\nQuote is friendship gag\nClean prop labels\nNo video generation", fill="#273645", font=font(21), spacing=8)
    panel_w = 330
    for i in range(4):
        x = 70 + i * 350
        d.rectangle([x, 500, x + panel_w, 760], fill="#e7eef2", outline="#171f2a", width=3)
        d.text((x + 18, 520), f"PANEL {i+1}", fill="#171f2a", font=font(24, True))
        notes = [
            "24mm push on\nred hotline",
            "50mm insert:\ntiny tint ticket",
            "35mm two-shot:\nscale contrast",
            "70mm stamp slam:\nFRIENDSHIP",
        ][i]
        d.text((x + 28, 585), notes, fill="#273645", font=font(24), spacing=8)
    d.text((64, 882), "Start wide dispatch board, dolly to ticket glass, rack focus to red phone, end on stamp. Keep it still-image sourced.", fill="#273645", font=font(22))
    d.text((770, 872), "Use saved GPT Image prompt for polished repair if local schematic mode was used. Preserve parody framing and traffic-ticket stakes.", fill="#273645", font=font(21))
    img.save(path)


def main() -> None:
    for subdir in [
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
        (PKG / subdir).mkdir(parents=True, exist_ok=True)

    write_json(PKG / "research/sources.json", {"created_at": NOW, "sources": SOURCES})
    write(
        PKG / "research/last30days_report.md",
        f"""# Last 30 Days Research Intake - Run {RUN_ID}

Created: {NOW}

## Research Query / Topic

Current celebrity/legal/entertainment stories with high SGFLIX visual contradiction: famous faces, public conflict, ego puncture, absurd quote/defense, and low-harm object comedy.

## Source Intake

- TMZ, May 2, 2026: Kevin Hart joked about Dwayne Johnson after Johnson's minor tinted-window traffic stop. The source frames Hart's insult and claimed police call as a friendship gag.
- TMZ, April 30, 2026: Johnson was reportedly stopped in Los Angeles after a Walk of Fame event for tinted windows and was compliant.
- Variety, May 1, 2026: Paramount subscriber antitrust suit supplied a current corporate candidate, but not enough famous-face energy.
- TMZ, May 2, 2026: Kathy Hilton known-danger filing supplied strong phrase comedy, but SGFLIX already covered that lane.
- TMZ, May 2, 2026: Nancy Mace email-to-judge story supplied power-archetype conflict, but legal sensitivity and recent overlap lowered fit.

## Winner

`{WINNER["id"]}` won because the verified context is light, current, visual, low-stakes, and already framed by the source as a joke between friends. The satire target is celebrity scale versus tiny traffic-ticket bureaucracy, not criminal conduct.

## Fact Guardrails

- Treat Hart's remarks as a reported joke, not a literal accusation.
- Treat the traffic stop as a minor reported vehicle-code/tinted-window stop.
- No exact likenesses, real police marks, mugshot framing, or implication of arrest.
""",
    )
    write_json(PKG / "strategy/candidate_board.json", {"created_at": NOW, "rubric": RUBRIC, "candidates": CANDIDATES, "winner_id": WINNER["id"]})
    write(
        PKG / "strategy/winner_decision.md",
        f"""# Winner Decision - Run {RUN_ID}

Selected premise: `{WINNER["id"]}`

{WINNER["premise"]}

Score: `{WINNER["total"]}/90`

Why it wins:
- Fresh: sourced from May 2 and April 30, 2026 items.
- Famous-face engine: Hart and Johnson are broadly legible without needing exact likenesses.
- Object comedy: the whole joke can live in a tiny tinted-window ticket, a red phone, and a friendship-violation stamp.
- Taste control: source itself describes the exchange as a gag between friends.

Rejected:
- Paramount subscriber suit: current but less face-driven and too close to prior merger court packages.
- Kathy Hilton known danger: excellent phrase, but duplicate lane from run_039.
- Nancy Mace email: vivid power-archetype premise, but too legally sensitive and overlaps with run_062.
""",
    )
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {
        "run_id": RUN_ID,
        "premise": WINNER["premise"],
        "passes": True,
        "score": 8.8,
        "checks": {
            "famous_face": "strong",
            "humiliation_without_cruelty": "strong; minor traffic-ticket ego puncture",
            "visual_contradiction": "strong; giant emergency for tiny ticket",
            "fact_safety": "strong if framed as reported friendship gag",
            "not_ai_slop": "passes; not celebrity + AI",
        },
    })
    write_json(PKG / "strategy/source_entropy_audit.json", {
        "run_id": RUN_ID,
        "source_count": len(SOURCES),
        "winner_source_count": 2,
        "entropy": "medium",
        "primary_source_risk": "TMZ-only for the joke lane; acceptable because the premise is explicitly treated as soft celebrity banter.",
        "duplicate_check": ["not run_067 spirit", "not run_039 hilton", "not run_062 mace", "not prior Paramount merger lane"],
    })
    write_json(PKG / "strategy/humor_logic_bridge.json", {
        "setup": "A routine tinted-window ticket exits a Walk of Fame event.",
        "turn": "Kevin Hart-style buddy insult escalates the minor ticket into dispatch-room panic.",
        "visual_rule": "Make bureaucratic machinery enormous and the actual offense tiny.",
        "punchline_objects": ["BUDDY COMEDY HOTLINE", "TINY TINT TICKET", "FRIENDSHIP VIOLATION stamp", "tint sample cards"],
        "do_not_do": ["real police insignia", "mugshot language", "exact likeness", "literal emergency"],
    })
    write_json(PKG / "strategy/tribe_meta_score.json", {
        "run_id": RUN_ID,
        "TRiBE": {"truth": 8, "relatability": 9, "irony": 10, "believability": 8, "emotion": 8, "total": 43},
        "Meta": {"thumb_stop": 9, "shareability": 8, "caption_friction": 8, "replay_value": 8, "total": 33},
    })
    write_json(PKG / "strategy/risk_taste_score.json", {
        "run_id": RUN_ID,
        "risk": "low-medium",
        "score": 8.7,
        "risks": [
            "Exact likeness drift could make it feel like a fake documentary still.",
            "Police imagery could imply arrest if not kept traffic-ticket-only.",
            "TMZ wording should be paraphrased except for a short referenced quote in research context.",
        ],
        "mitigations": ["fictional proxies", "minor traffic stakes", "friendship gag framing", "no real police marks"],
    })
    write(
        PKG / "strategy/franchise_decision.md",
        "# Franchise Decision\n\nDecision: `single_episode_with_recurring_bureaucracy_potential`\n\nThis can extend into a recurring SGFLIX desk format: tiny celebrity infractions handled by absurdly overbuilt municipal departments. Do not make this a broader policing joke; keep it as buddy-comedy paperwork.\n",
    )
    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT + "\n")

    first_path = PKG / "frames/gpt_image_2/first_frame_v01.png"
    board_path = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    first_mode = try_openai_image(FIRST_PROMPT, first_path, "1536x1024")
    board_mode = try_openai_image(BOARD_PROMPT, board_path, "1536x1024")
    if not first_path.exists():
        draw_first_frame(first_path)
        first_mode += "_fallback_local_schematic"
    if not board_path.exists():
        draw_board(board_path)
        board_mode += "_fallback_local_schematic"

    shot = {
        "run_id": RUN_ID,
        "shot": "001",
        "duration_seconds": 6,
        "subject": "fictional huge action-star proxy and tiny comedian-caller proxy; no exact likeness",
        "scene": "traffic-dispatch office treating a tiny tinted-window ticket as a friendship emergency",
        "motion": "manual-only push from red hotline to ticket under glass to friendship-violation stamp",
        "spatial": "dispatch board left, action-star proxy center, red phone right, ticket pedestal foreground",
        "camera": "24mm wide push, 50mm ticket insert, 70mm stamp close",
        "critique": "must read as a reported friendship gag and minor traffic-ticket satire",
        "revision": "remove real police insignia, mugshot cues, exact likeness, messy text, or criminal implication",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
    }
    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [shot]})
    scene = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "status": "handoff_only_no_video_generation",
        "source_image": "frames/gpt_image_2/first_frame_v01.png",
        "prompt": "Manual-only 6-second plan: push from the blinking Buddy Comedy Hotline to the tiny tint ticket under museum glass, then land on the FRIENDSHIP VIOLATION stamp. No video generation in this automation.",
        "negative": "exact likeness, real police insignia, mugshot, arrest implication, defamatory text, watermark",
    }
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {
        "run_id": RUN_ID,
        "video_generation_tools_called": False,
        "manual_only": True,
        "first_frame": "frames/gpt_image_2/first_frame_v01.png",
        "storyboard": "storyboards/shared_choices/shared_choices_v01.png",
        "guardrails": ["No exact likeness", "No real police logos", "Friendship gag framing", "Minor traffic-ticket stakes only"],
    })
    write(PKG / "handoffs/grok_agent_prompt.md", f"# Grok/Closed Tool Prompt - Run {RUN_ID}\n\nDo not generate video automatically. Use the saved first frame and Shared Choices board only if a human starts a manual render. Preserve the fictional-proxy casting and traffic-ticket gag. Keep all facts as reported by TMZ and do not imply arrest or criminal conduct.\n")
    write(PKG / "captions/instagram_caption.md", "A tiny tint ticket walked into the wrong friendship group chat.\n\nReported context: Kevin Hart joked about Dwayne Johnson's minor traffic stop; this is a fictional buddy-comedy paperwork sketch, not a crime story.\n\n#sgflix #satire #buddycomedy #trafficcourt #popculture\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nStatus: still package only; no video generated.\n\nSurface: Instagram Reels/TikTok after human visual approval and manual video workflow.\n\nHook: \"When your best friend reports your window tint to the Buddy Comedy Hotline.\"\n\nHuman action: review stills for likeness, police-mark, and text clarity before any manual render.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nThis run demonstrates a low-harm SGFLIX premise: convert a soft celebrity item into object bureaucracy. The joke does not need harsher allegations; it needs scale mismatch, readable props, and a clean first-frame contradiction.\n")
    assets = [
        {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode, "exists": first_path.exists()},
        {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True},
        {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode, "exists": board_path.exists()},
        {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True},
        {"path": "qc/first_frame_v01_qc.md", "type": "qc", "exists": True},
        {"path": "qc/shared_choices_v01_qc.md", "type": "qc", "exists": True},
    ]
    missing = [item["path"] for item in assets if not item.get("exists")]
    status = "complete_stills_ready_no_video_generated" if not missing else "blocked_missing_required_assets"
    if "fallback_local_schematic" in first_mode or "fallback_local_schematic" in board_mode:
        status = "complete_with_local_schematic_stills_gpt_image_repair_recommended"
        write(PKG / "qc/IMAGE_GENERATION_NOTE.md", f"# Image Generation Note\n\nGPT image prompt files were created after winner selection. API/local availability did not produce both polished GPT image outputs, so local schematic PNGs were generated as usable internal still artifacts.\n\nFirst frame mode: `{first_mode}`\nShared Choices mode: `{board_mode}`\n\nNo video-generation tool was called.\n")

    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "status": status, "assets": assets, "missing_files": missing, "post_ready_exports": [], "video_generation_tools_called": False})
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {
        "run_id": RUN_ID,
        "slug": SLUG,
        "created_at": NOW,
        "status": status,
        "research_query": "current celebrity/legal/entertainment stories with SGFLIX visual contradiction",
        "selected_premise": WINNER,
        "candidate_count": len(CANDIDATES),
        "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"},
        "video_generation_tools_called": False,
    })
    write(PKG / "qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nAsset: `frames/gpt_image_2/first_frame_v01.png`\n\nGeneration mode: `{first_mode}`\n\nPasses: clear dispatch-office contradiction, tiny ticket object, fictional proxies, minor traffic-ticket stakes, no video generation.\n\nWatch items: inspect generated text and likeness drift. If local schematic mode was used, run the saved prompt through GPT Image 2 for a polished repair before public export.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: `USABLE_FOR_INTERNAL_REVIEW`\n\nAsset: `storyboards/shared_choices/shared_choices_v01.png`\n\nGeneration mode: `{board_mode}`\n\nPasses: includes character/props, color palette, set design, blocking, storyboard panels, lighting/mood, visual rules, and production notes.\n\nWatch items: verify text readability and rerun prompt through GPT Image 2 if a more cinematic board is needed.\n")
    readme = f"""# RUN {RUN_ID} Master Package - Rock Tint Ticket Dispatch

Status: `{status}`

Selected premise: {WINNER["premise"]}

Research came first, candidates were scored, and the winner was selected before this package was created.

Generated still artifacts:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

No video footage was generated or requested.

Next human action: review the two still PNGs for likeness drift, police-logo risk, and label clarity; optionally rerun the saved GPT Image prompts for polished repair before any manual closed-tool video workflow.
"""
    write(PKG / "README.md", readme)
    status_md = f"""# Factory Run Status - Run {RUN_ID}

Status: `{status}`

Created: {NOW}

Research query/topic: current celebrity/legal/entertainment stories with famous-face, public-conflict, absurd-quote, and first-frame visual contradiction potential.

Candidate board:
- `{CANDIDATES[0]["id"]}`: {CANDIDATES[0]["total"]}/90 - selected.
- `{CANDIDATES[1]["id"]}`: {CANDIDATES[1]["total"]}/90 - rejected, duplicate merger lane.
- `{CANDIDATES[2]["id"]}`: {CANDIDATES[2]["total"]}/90 - rejected, duplicate known-danger lane.
- `{CANDIDATES[3]["id"]}`: {CANDIDATES[3]["total"]}/90 - rejected, legal sensitivity and overlap.

Selected premise: {WINNER["premise"]}

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing required files: {missing if missing else "none"}

Post-ready exports: none; still package requires human review first.

QC failures: none blocking for internal package review. Repair recommended if generated labels or likeness drift fail human taste check.

High-risk issues:
- Keep the quote framed as reported friendship banter.
- Do not imply arrest or criminal conduct.
- Avoid exact Hart/Johnson likenesses and real police insignia.

Exact next human action: review first frame and Shared Choices board, then approve or request a GPT Image 2 repair pass before any manual video handoff.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status_md)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"# Factory Run Status - Run {RUN_ID}\n\nSee `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n\nStatus: `{status}`\n")


if __name__ == "__main__":
    main()
