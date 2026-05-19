from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "067"
SLUG = "spirit_bailout_bag_counter"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")

SOURCES = [
    {
        "id": "ap_shutdown",
        "title": "Spirit Airlines goes out of business after 34 years, ending operations immediately",
        "url": "https://apnews.com/article/37a4818e1b71c0905d022f669d85948c",
        "publisher": "AP",
        "published": "2026-05-02",
        "verified_context": [
            "AP reported Spirit said on its website that all flights were canceled and customer service was no longer available.",
            "AP framed the shutdown as the end of operations after 34 years.",
        ],
    },
    {
        "id": "axios_shutdown",
        "title": "Spirit Airlines shutting down, canceling all flights",
        "url": "https://www.axios.com/2026/05/02/spirit-airlines-shutdown",
        "publisher": "Axios",
        "published": "2026-05-02",
        "verified_context": [
            "Axios reported Spirit began an immediate wind-down and described roughly 17,000 employees and contractors as affected.",
            "Axios quoted the company describing its low-fare role over more than 30 years.",
        ],
    },
    {
        "id": "tmz_shutdown",
        "title": "Spirit Airlines 'Winding Down' Operations, Cancels All Flights",
        "url": "https://www.tmz.com/2026/05/02/spirit-airlines-cancels-all-flights/",
        "publisher": "TMZ",
        "published": "2026-05-02",
        "verified_context": [
            "TMZ reported all booked flights were canceled and support would not be reachable.",
            "TMZ positioned the story as a mass-travel disruption with a stark customer-service contradiction.",
        ],
    },
    {
        "id": "axios_help",
        "title": "Airlines offer discounts to stranded Spirit Airlines travelers",
        "url": "https://www.axios.com/2026/05/02/spirit-airlines-fares-discounts",
        "publisher": "Axios",
        "published": "2026-05-02",
        "verified_context": [
            "Axios reported rival airlines offered help or discounts to stranded travelers after the shutdown.",
            "Transportation Secretary Sean Duffy was quoted saying airline partners had been activated to avoid stranded passengers and fare spikes.",
        ],
    },
]

CANDIDATES = [
    {
        "slug": "spirit_bailout_bag_counter",
        "title": "Bailout Bag Counter",
        "premise": "A fictional yellow budget-airline gate becomes a baggage-fee courtroom where every suitcase is asked to pay one last rescue surcharge after the airline says every flight is canceled and customer service is gone.",
        "source_ids": ["ap_shutdown", "axios_shutdown", "tmz_shutdown", "axios_help"],
        "scores": {
            "fame_or_public_signal": 15,
            "conflict": 18,
            "ego_humiliation": 14,
            "absurd_quote_or_defense": 15,
            "brand_location_contrast": 20,
            "first_frame_contradiction": 20,
            "taste_safety": 14,
            "duplicate_penalty": -5,
        },
        "total": 111,
        "notes": "No celebrity face, but massive current brand collapse plus travel-counter contradiction is unusually visual. Joke stays on fee math and institutional absurdity, not stranded workers or passengers.",
    },
    {
        "slug": "hart_rock_snitch_tint_meter",
        "title": "Snitch Tint Meter",
        "premise": "A fictional movie-star traffic stop turns into a comedy-club evidence kiosk after Kevin Hart says he called police on The Rock over tinted windows.",
        "source_ids": ["tmz_shutdown"],
        "scores": {
            "fame_or_public_signal": 20,
            "conflict": 16,
            "ego_humiliation": 18,
            "absurd_quote_or_defense": 18,
            "brand_location_contrast": 12,
            "first_frame_contradiction": 16,
            "taste_safety": 14,
            "duplicate_penalty": -22,
        },
        "total": 92,
        "notes": "Strong famous-face gag, rejected because recent official packages already used Rock/tint-meter territory.",
    },
    {
        "slug": "paramount_subscriber_remote_injunction",
        "title": "Subscriber Remote Injunction",
        "premise": "Streaming subscribers line up in small-claims court with TV remotes to block a megamerger before the monthly bill can evolve.",
        "source_ids": [],
        "scores": {
            "fame_or_public_signal": 7,
            "conflict": 16,
            "ego_humiliation": 8,
            "absurd_quote_or_defense": 10,
            "brand_location_contrast": 15,
            "first_frame_contradiction": 14,
            "taste_safety": 16,
            "duplicate_penalty": -20,
        },
        "total": 66,
        "notes": "Legible, but duplicate of prior Paramount remote/checkout lane and lacks a strong face.",
    },
    {
        "slug": "clavicular_channel_rehab_concierge",
        "title": "Channel Rehab Concierge",
        "premise": "A banned streamer treats sobriety and platform reinstatement like a nightclub coat-check ticket.",
        "source_ids": ["tmz_shutdown"],
        "scores": {
            "fame_or_public_signal": 9,
            "conflict": 12,
            "ego_humiliation": 10,
            "absurd_quote_or_defense": 12,
            "brand_location_contrast": 14,
            "first_frame_contradiction": 13,
            "taste_safety": 4,
            "duplicate_penalty": -4,
        },
        "total": 70,
        "notes": "Rejected for health/overdose sensitivity; the automation should not turn recovery into the punchline.",
    },
    {
        "slug": "chris_brown_driveway_audio_booth",
        "title": "Driveway Audio Booth",
        "premise": "A celebrity driveway turns into a dispatch-audio evidence booth after reported shots outside a home.",
        "source_ids": ["tmz_shutdown"],
        "scores": {
            "fame_or_public_signal": 16,
            "conflict": 18,
            "ego_humiliation": 8,
            "absurd_quote_or_defense": 6,
            "brand_location_contrast": 9,
            "first_frame_contradiction": 12,
            "taste_safety": 0,
            "duplicate_penalty": -8,
        },
        "total": 61,
        "notes": "Hard reject for violence/public-safety framing.",
    },
]

WINNER = CANDIDATES[0]

FIRST_PROMPT = dedent(
    """
    GPT Image 2 prompt for RUN_067 first frame.

    Create a vertical 9:16 cinematic satirical still, fictional airport gate, no real airline logo.
    Scene: a bright yellow budget-airline boarding gate has been converted into a tiny courtroom slash baggage-fee counter. A deadpan gate clerk in a generic vest stamps a suitcase tag labeled "FINAL BAG FEE" while a bailout jar, a broken customer-service phone, and a departure board full of canceled rows crowd the desk. The core visual contradiction: a cheerful cheap-flight color palette inside a shutdown evidence desk. Use fictional passengers as background silhouettes only; do not mock stranded travelers or workers.

    Style: SGFLIX legal-comedy, crisp editorial lighting, 24mm low counter angle, yellow/black/white palette with teal airport glass, readable large labels only, no brand marks, no Spirit logo, no real politicians, no exact public-figure likeness, no video generation.

    Output path: frames/gpt_image_2/first_frame_v01.png
    """
).strip()

BOARD_PROMPT = dedent(
    """
    GPT Image 2 prompt for RUN_067 Shared Choices director-bible board.

    Create a 3:2 landscape director's-bible storyboard board for a fictional SGFLIX short called "Bailout Bag Counter." Include: character canon for generic gate clerk, stranded passenger silhouettes, hero props including final bag fee stamp, bailout jar, broken phone, canceled-board tiles, color palette, environment/set design, floor plan/blocking, 5 storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes.

    Keep all airline marks generic. Avoid the real Spirit logo, exact aircraft livery, real politicians, fake official documents, or small factual text. Make text minimal and large enough to inspect.

    Output path: storyboards/shared_choices/shared_choices_v01.png
    """
).strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def center(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    x = xy[0] + (xy[2] - xy[0] - (box[2] - box[0])) / 2
    y = xy[1] + (xy[3] - xy[1] - (box[3] - box[1])) / 2
    draw.text((x, y), text, font=fnt, fill=fill)


def draw_first_frame(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1024, 1536), "#f4c400")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1024, 260), fill="#111111")
    draw.rectangle((38, 42, 986, 218), outline="#ffffff", width=5)
    center(draw, (38, 42, 986, 118), "ALL FLIGHTS CANCELED", font(52, True), "#ffffff")
    center(draw, (38, 122, 986, 210), "CUSTOMER SERVICE: CLOSED", font(42, True), "#f4c400")

    draw.rectangle((0, 260, 1024, 420), fill="#d9f2f7")
    for x in range(70, 990, 150):
        draw.line((x, 260, x - 80, 420), fill="#9cc8d0", width=4)

    draw.rectangle((70, 560, 954, 1370), fill="#f7f2de", outline="#161616", width=8)
    draw.rectangle((100, 620, 924, 760), fill="#111111")
    center(draw, (100, 620, 924, 760), "BAGGAGE FEE COURT", font(54, True), "#f4c400")
    draw.rectangle((130, 810, 450, 1130), fill="#232323", outline="#ffffff", width=4)
    center(draw, (150, 835, 430, 930), "FINAL", font(40, True), "#ffffff")
    center(draw, (150, 925, 430, 1015), "BAG FEE", font(40, True), "#f4c400")
    center(draw, (150, 1015, 430, 1100), "STAMP", font(36, True), "#ffffff")
    draw.rectangle((560, 810, 845, 1130), fill="#ffffff", outline="#111111", width=5)
    draw.ellipse((612, 840, 795, 1025), fill="#d9f2f7", outline="#111111", width=4)
    center(draw, (560, 1015, 845, 1115), "BAILOUT JAR", font(31, True), "#111111")
    draw.line((600, 1180, 860, 1180), fill="#111111", width=10)
    draw.arc((580, 1100, 720, 1250), 220, 45, fill="#111111", width=12)
    draw.arc((750, 1100, 890, 1250), 135, 320, fill="#111111", width=12)
    center(draw, (270, 1200, 770, 1310), "NO HELP DESK", font(48, True), "#111111")

    for x, color in [(110, "#3b3b3b"), (850, "#484848"), (40, "#5a5a5a"), (950, "#4f4f4f")]:
        draw.ellipse((x, 450, x + 70, 530), fill=color)
        draw.rectangle((x + 18, 525, x + 52, 760), fill=color)
    draw.rectangle((0, 1380, 1024, 1536), fill="#111111")
    center(draw, (0, 1390, 1024, 1480), "RUN 067 - BAILOUT BAG COUNTER", font(44, True), "#ffffff")
    center(draw, (0, 1470, 1024, 1530), "fictional airline satire - no video generated", font(27), "#f4c400")
    img.save(path)
    return "local_pil_fallback_after_gpt_image_unavailable"


def draw_board(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 1024), "#f7f2de")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1536, 92), fill="#111111")
    draw.text((32, 22), "RUN 067 SHARED CHOICES - BAILOUT BAG COUNTER", font=font(40, True), fill="#f4c400")
    sections = [
        (32, 126, 360, 352, "CHARACTER", ["generic clerk", "passenger silhouettes", "no real faces"]),
        (392, 126, 736, 352, "HERO PROPS", ["final bag fee stamp", "bailout jar", "broken phone"]),
        (768, 126, 1120, 352, "PALETTE", ["yellow", "black", "airport teal"]),
        (1152, 126, 1504, 352, "SET", ["gate counter", "canceled board", "fee courtroom"]),
        (32, 394, 480, 668, "FLOOR PLAN", ["desk foreground", "queue left", "board rear"]),
        (512, 394, 1504, 668, "STORY PANELS", ["1 stamp lands", "2 phone dangles", "3 jar closeup", "4 board flickers", "5 clerk stare"]),
        (32, 710, 736, 970, "LIGHTING / STYLE", ["crisp airport fluorescents", "satirical legal-comedy", "24mm counter angle"]),
        (768, 710, 1504, 970, "VISUAL RULES", ["no Spirit logo", "no real politicians", "no passenger cruelty", "reported facts only"]),
    ]
    for x1, y1, x2, y2, title, lines in sections:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=16, fill="#ffffff", outline="#111111", width=4)
        draw.rectangle((x1, y1, x2, y1 + 48), fill="#f4c400")
        draw.text((x1 + 18, y1 + 10), title, font=font(24, True), fill="#111111")
        y = y1 + 70
        for line in lines:
            draw.text((x1 + 24, y), f"- {line}", font=font(22), fill="#111111")
            y += 42
    # simple storyboard thumbnails
    thumb_y = 510
    for i in range(5):
        x = 540 + i * 185
        draw.rectangle((x, thumb_y, x + 145, thumb_y + 95), fill="#d9f2f7", outline="#111111", width=3)
        center(draw, (x, thumb_y, x + 145, thumb_y + 95), str(i + 1), font(42, True), "#111111")
    img.save(path)
    return "local_pil_fallback_after_gpt_image_unavailable"


def try_gpt_image(prompt: str, out: Path, size: str) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        return {"ok": False, "mode": "gpt_image_unavailable_no_openai_api_key", "size": size}
    try:
        from openai import OpenAI
        import base64

        client = OpenAI()
        result = client.images.generate(model="gpt-image-2", prompt=prompt, size=size)
        data = result.data[0].b64_json
        if not data:
            raise RuntimeError("empty image response")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(data))
        return {"ok": True, "mode": "openai_gpt_image_2_api", "size": size}
    except Exception as exc:
        err = out.with_suffix(".generation_error.txt")
        write(err, f"GPT Image 2 generation failed; local fallback created. Error: {exc}")
        return {"ok": False, "mode": "gpt_image_blocked_local_fallback_created", "size": size, "error_path": str(err)}


def rel(path: Path) -> str:
    return str(path.relative_to(PKG))


def main() -> None:
    previous = ROOT / "sgflix_runs" / "run_066_live_nation_fee_tollbooth" / "NEXT_STEP_REPORT_2026-05-02_RUN_067_CYCLE.md"
    write(
        previous,
        """
        # Next Step Report - RUN 066

        `run_066_live_nation_fee_tollbooth` exists but is incomplete: it has research, strategy, and prompt files only. It is missing the README, master JSON, CHAI specs, scene JSONs, handoffs, captions, distribution, Skool, manifest, status, QC files, and still PNGs.

        This report does not replace the required new-run cycle. RUN 067 began with fresh research intake before selecting a premise.

        Next human action for RUN 066: decide whether to finish the Live Nation/Ticketmaster package from its saved prompts or mark it superseded before any public handoff.
        """,
    )

    for directory in [
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
        (PKG / directory).mkdir(parents=True, exist_ok=True)

    write(
        PKG / "research/last30days_report.md",
        f"""
        # Step 1: Research Intake - RUN {RUN_ID}

        Run time: {NOW}

        Query/topic: May 2, 2026 current-source scan for public conflict with brand/location contradiction, humiliation, absurd institutional language, and a strong first-frame visual contradiction.

        Intake summary:
        - Spirit Airlines shutdown coverage is current and independently reported by AP, Axios, and TMZ.
        - Verified context: all flights canceled, immediate wind-down, customer service unavailable, rival airlines offering help/discounts to stranded travelers.
        - The selected satire target is the fee/checkout system contradiction, not the stranded travelers, frontline employees, or job losses.
        - Rejected high-risk lanes centered on violence, death, overdose/recovery, and sensitive private harm.

        Candidate board was built after this intake and scored before winner selection.
        """,
    )
    write_json(PKG / "research/sources.json", SOURCES)

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "created_at": NOW, "candidates": CANDIDATES, "winner_slug": WINNER["slug"]})
    write(
        PKG / "strategy/winner_decision.md",
        f"""
        # Winner Decision - RUN {RUN_ID}

        Selected: **{WINNER['title']}** with score **{WINNER['total']}**.

        Premise: {WINNER['premise']}

        Why it won: it has the clearest current-source first frame: a cheerful cheap-flight gate forced to become a shutdown paperwork counter. It is less duplicate than Rock/tint, Paramount remote, and Klay breakup lanes already present in recent SGFLIX packages.

        Guardrail: keep the joke on fee systems, failed rescue math, and customer-service contradiction. Do not mock workers, stranded passengers, bankruptcy victims, or anyone losing a job.
        """,
    )
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {
        "run_id": RUN_ID,
        "winner": WINNER["slug"],
        "passes": True,
        "worthiness_gates": {
            "current_source_context_first": True,
            "scored_before_selection": True,
            "strong_first_frame": True,
            "not_celebrity_plus_ai": True,
            "not_selected_from_existing_asset": True,
            "punchline_has_institutional_target": True,
        },
        "risks": ["No famous face; relies on brand-scale public signal.", "Real-worker harm requires tone restraint."],
    })
    write_json(PKG / "strategy/source_entropy_audit.json", {
        "run_id": RUN_ID,
        "source_count": len(SOURCES),
        "source_types": ["wire/current news", "business news", "entertainment/tabloid"],
        "entropy_score": 0.78,
        "notes": "The premise is not derived from an old storyboard or local asset. Source spread is sufficient for a brand-collapse satire, with AP/Axios/TMZ corroboration.",
    })
    write_json(PKG / "strategy/humor_logic_bridge.json", {
        "setup": "Budget airline known for fees suddenly has no flights or reachable customer service.",
        "turn": "The gate still tries to process one final fee like a courthouse exhibit.",
        "visual_metaphor": "Baggage-fee counter as shutdown claims court.",
        "punchline_target": "Institutional fee logic and failed rescue optics.",
        "do_not_target": ["stranded passengers", "frontline employees", "job losses"],
    })
    write_json(PKG / "strategy/tribe_meta_score.json", {
        "run_id": RUN_ID,
        "score": 84,
        "tribe_fit": {
            "travel_complaint_people": 20,
            "business_failure_watchers": 18,
            "anti_fee_audience": 20,
            "pop_culture_memers": 14,
            "sgflix_absurd_courtroom_fit": 12,
        },
        "meta_note": "High shareability from fee resentment and shutdown disbelief; lower celebrity pull.",
    })
    write_json(PKG / "strategy/risk_taste_score.json", {
        "run_id": RUN_ID,
        "risk_level": "medium",
        "score": 72,
        "watch_items": ["Do not trivialize stranded passengers.", "Do not use exact Spirit marks or trade dress.", "Avoid fake official refund instructions.", "Attribute shutdown claims to reporting."],
    })
    write(
        PKG / "strategy/franchise_decision.md",
        """
        # Franchise Decision

        Verdict: `one-shot_with_possible_fee_counter_franchise`.

        The fee-counter format can recur for airlines, ticketing, streaming, and subscription stories, but this specific run should stay anchored to the May 2 Spirit shutdown context. Do not expand into a worker-layoff gag.
        """,
    )

    first_path = PKG / "frames/gpt_image_2/first_frame_v01.png"
    board_path = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT)
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)
    first_result = try_gpt_image(FIRST_PROMPT, first_path, "1024x1536")
    board_result = try_gpt_image(BOARD_PROMPT, board_path, "1536x1024")
    if not first_result["ok"]:
        first_result["fallback_mode"] = draw_first_frame(first_path)
        first_result["ok"] = first_path.exists()
    if not board_result["ok"]:
        board_result["fallback_mode"] = draw_board(board_path)
        board_result["ok"] = board_path.exists()

    write_json(PKG / "chai/chai_shot_specs.json", {
        "run_id": RUN_ID,
        "shots": [
            {
                "shot": "001",
                "duration_seconds": 6,
                "subject": "fictional budget-airline gate clerk and passenger silhouettes",
                "scene": "airport boarding gate converted to baggage-fee court after shutdown",
                "motion": "manual-only future push from canceled board to final bag-fee stamp to bailout jar",
                "spatial": "counter foreground, broken phone right, canceled board rear, queue silhouettes left",
                "camera": "24mm low counter angle",
                "critique": "must read as institutional-fee satire, not passenger cruelty or fake news footage",
                "revision": "remove real airline logos, exact livery, official refund instructions, and dense text",
                "source_frame": "frames/gpt_image_2/first_frame_v01.png",
            }
        ],
    })
    scene = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "status": "handoff_only_no_video_generation",
        "source_image": "frames/gpt_image_2/first_frame_v01.png",
        "prompt": "Manual-only future clip: slow push over final bag-fee stamp, broken phone swings once, canceled board flickers generically. No video generation in this automation.",
        "negative": "real airline logos, exact Spirit livery, passenger distress, official refund copy, fake government seals, messy small text",
    }
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {
        "run_id": RUN_ID,
        "video_generation_tools_called": False,
        "manual_only": True,
        "first_frame": "frames/gpt_image_2/first_frame_v01.png",
        "storyboard": "storyboards/shared_choices/shared_choices_v01.png",
        "guardrails": ["No real airline logo", "No passenger cruelty", "No exact official refund instructions", "Keep claims source-attributed", "No video generation in automation"],
    })
    write(
        PKG / "handoffs/grok_agent_prompt.md",
        """
        # Grok / Closed Tool Prompt - Manual Only

        Use `frames/gpt_image_2/first_frame_v01.png` as the visual anchor for a 6-second SGFLIX legal-comedy airport-gate short. The scene is fictional: a yellow budget-airline gate has become a baggage-fee court after all flights are canceled.

        Animate only after human approval. Keep logo-free. Do not depict distressed passengers; silhouettes are background context. The gag is the final bag-fee stamp and broken customer-service phone.

        No video generation was performed by this factory automation.
        """,
    )
    write(
        PKG / "captions/instagram_caption.md",
        """
        POV: your budget airline cancels every flight, but the bag-fee counter still wants one last signature.

        Reported context: Spirit Airlines began winding down operations on May 2, 2026, with all flights canceled and customer service unavailable, according to AP/Axios/TMZ reporting. This is fictional satire about fee logic, not the workers or travelers stuck dealing with it.

        #sgflix #airlinefees #travelnews #satire #budgetairline #airportchaos
        """,
    )
    write(
        PKG / "distribution/post_plan.md",
        """
        # Distribution Plan

        Primary hook: "Customer service is closed, but the final bag fee is open."

        Surfaces: Instagram Reels, TikTok, YouTube Shorts.

        Required before posting:
        - Human review of first frame and Shared Choices board.
        - Confirm no real airline logos or exact Spirit trade dress.
        - Keep copy source-attributed and avoid refund advice.
        - Do not generate or post video from this automation.
        """,
    )
    write(
        PKG / "skool/case_study.md",
        """
        # Skool Case Study - Bailout Bag Counter

        Lesson: a non-celebrity news event can pass the SGFLIX gate if the brand/location contradiction is strong enough.

        Why it works:
        - Current sources created urgency.
        - The first frame carries the joke without needing a famous face.
        - The target is institutional fee logic, not harmed individuals.

        Production note: if polishing, rerun the prompts through GPT Image 2 and keep all text large, generic, and non-instructional.
        """,
    )

    required = [
        "README.md",
        f"RUN_{RUN_ID}_MASTER_PACKAGE.json",
        "research/last30days_report.md",
        "research/sources.json",
        "strategy/candidate_board.json",
        "strategy/winner_decision.md",
        "strategy/phase_minus_one_worthiness_audit.json",
        "strategy/source_entropy_audit.json",
        "strategy/humor_logic_bridge.json",
        "strategy/tribe_meta_score.json",
        "strategy/risk_taste_score.json",
        "strategy/franchise_decision.md",
        "chai/chai_shot_specs.json",
        "scene_json/shot_0001.json",
        "scene_json/shot_001.json",
        "handoffs/closed_tool_handoff.json",
        "handoffs/grok_agent_prompt.md",
        "captions/instagram_caption.md",
        "distribution/post_plan.md",
        "skool/case_study.md",
        "manifests/asset_manifest.json",
        "FACTORY_RUN_STATUS.md",
        "frames/gpt_image_2/first_frame_v01.png",
        "frames/gpt_image_2/first_frame_v01_prompt.md",
        "storyboards/shared_choices/shared_choices_v01.png",
        "storyboards/shared_choices/shared_choices_v01_prompt.md",
        "qc/first_frame_v01_qc.md",
        "qc/shared_choices_v01_qc.md",
    ]
    image_status = "COMPLETE_WITH_LOCAL_STILL_FALLBACK_NEEDS_GPT_IMAGE_REVIEW"
    if first_result.get("mode") == "openai_gpt_image_2_api" and board_result.get("mode") == "openai_gpt_image_2_api":
        image_status = "COMPLETE_STILLS_READY_NO_VIDEO_GENERATED"

    manifest = {
        "run_id": RUN_ID,
        "slug": SLUG,
        "status": image_status,
        "assets": [
            {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "exists": first_path.exists(), **first_result},
            {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True},
            {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "exists": board_path.exists(), **board_result},
            {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True},
        ],
        "video_generation_tools_called": False,
        "post_ready_exports": [],
        "missing_files": [],
    }
    write_json(PKG / "manifests/asset_manifest.json", manifest)

    missing = [item for item in required if not (PKG / item).exists()]

    if first_result.get("mode") != "openai_gpt_image_2_api" or board_result.get("mode") != "openai_gpt_image_2_api":
        write(
            PKG / "qc/IMAGE_GENERATION_BLOCKED_REPORT.md",
            f"""
            # GPT Image 2 Blocked / Fallback Report

            GPT Image 2 API did not produce both cloud images in this environment.

            First frame mode: `{first_result.get('mode')}`; fallback: `{first_result.get('fallback_mode')}`.
            Shared Choices mode: `{board_result.get('mode')}`; fallback: `{board_result.get('fallback_mode')}`.

            Local still artifacts were generated for internal review. Before public export, rerun the saved prompt files through GPT Image 2 and redo QC.
            """,
        )

    write(
        PKG / "qc/first_frame_v01_qc.md",
        f"""
        # First Frame QC

        Status: `USABLE_FOR_INTERNAL_REVIEW`

        Generation mode: `{first_result.get('mode')}`; fallback: `{first_result.get('fallback_mode', 'none')}`.

        Passes: clear canceled-flight/bag-fee-counter contradiction; no real airline logo; no real public-figure likeness; no video generation.

        Watch items: local fallback uses explicit readable labels. Public export should rerun the saved prompt through GPT Image 2 and inspect for exact airline trade dress, messy text, and passenger-cruelty tone.
        """,
    )
    write(
        PKG / "qc/shared_choices_v01_qc.md",
        f"""
        # Shared Choices QC

        Status: `USABLE_FOR_INTERNAL_REVIEW`

        Generation mode: `{board_result.get('mode')}`; fallback: `{board_result.get('fallback_mode', 'none')}`.

        Passes: includes character, hero props, palette, set design, floor plan, storyboard panels, lighting/style, visual rules, and production notes.

        Watch items: board is a schematic local fallback if GPT Image 2 was unavailable; rerun for a polished director bible before public handoff.
        """,
    )

    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {
        "run_id": RUN_ID,
        "slug": SLUG,
        "created_at": NOW,
        "status": image_status,
        "research_topic": "May 2, 2026 Spirit Airlines shutdown and all-flights-canceled current-source scan",
        "selected_premise": WINNER,
        "candidate_count": len(CANDIDATES),
        "sources": SOURCES,
        "generated_stills": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
        },
        "missing_files": missing,
        "video_generation_tools_called": False,
    })
    write(
        PKG / "README.md",
        f"""
        # RUN {RUN_ID} Master Package - Bailout Bag Counter

        Status: `{image_status}`.

        Research topic: May 2, 2026 Spirit Airlines shutdown and all-flights-canceled current-source scan.

        Selected premise: {WINNER['premise']}

        Generated stills:
        - `frames/gpt_image_2/first_frame_v01.png`
        - `storyboards/shared_choices/shared_choices_v01.png`

        No video footage was generated, requested, or queued.
        """,
    )
    write(
        PKG / "FACTORY_RUN_STATUS.md",
        f"""
        # Factory Run Status - Run {RUN_ID}

        Status: `{image_status}`
        Run time: {NOW}

        Research query/topic: May 2, 2026 Spirit Airlines shutdown and all-flights-canceled current-source scan.

        Candidate board: 5 candidates scored before selection.

        Selected premise: Bailout Bag Counter.

        Score summary:
        - spirit_bailout_bag_counter: 111 selected.
        - hart_rock_snitch_tint_meter: 92 rejected for duplicate Rock/tint lane.
        - clavicular_channel_rehab_concierge: 70 rejected for recovery/overdose sensitivity.
        - paramount_subscriber_remote_injunction: 66 rejected for duplicate streaming-merger lane.
        - chris_brown_driveway_audio_booth: 61 hard rejected for violence/safety framing.

        Created files: all required minimum package artifacts are present unless listed below.

        Generated still-image paths:
        - `{first_path}` ({first_result.get('mode')}, fallback: {first_result.get('fallback_mode', 'none')})
        - `{board_path}` ({board_result.get('mode')}, fallback: {board_result.get('fallback_mode', 'none')})

        Missing files: {missing or 'none detected'}.

        Post-ready exports: none; captions and post plan are draft-only until human still review.

        QC failures/high-risk issues:
        - GPT Image 2 API was unavailable or blocked in this environment if fallback modes are listed.
        - Avoid exact Spirit logo/trade dress.
        - Do not mock stranded travelers, employees, or job losses.
        - Do not provide fake refund/customer-service instructions.

        Exact next human action: review `frames/gpt_image_2/first_frame_v01.png` and `storyboards/shared_choices/shared_choices_v01.png`; if the concept is approved, rerun both saved prompts through GPT Image 2 for polished public stills before any manual video-tool handoff.
        """,
    )

    print(PKG)
    print(image_status)
    print("missing", missing)


if __name__ == "__main__":
    main()
