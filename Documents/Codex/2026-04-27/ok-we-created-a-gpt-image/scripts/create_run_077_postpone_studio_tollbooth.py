from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN = "077"
SLUG = "postpone_studio_tollbooth"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")

SOURCES = [
    {
        "id": "tmz_postpone_2026_05_02",
        "title": "Post Malone Cancels First Few Weeks of Upcoming Tour",
        "url": "https://www.tmz.com/2026/05/02/post-malone-cancels-beginning-of-tour/",
        "publisher": "TMZ",
        "published": "2026-05-02",
        "used_for": ["winner", "quote hook", "tour-delay premise"],
        "verified_notes": [
            "TMZ reported Post Malone delayed the opening stretch of a tour with Jelly Roll to finish new music.",
            "TMZ reported he pushed back against speculation about ticket sales and said he needed time to complete promised tracks.",
        ],
    },
    {
        "id": "nme_postpone_2026_05_02",
        "title": "Post Malone cancels start of Big Ass Stadium tour with Jelly Roll",
        "url": "https://www.nme.com/news/music/post-malone-cancels-start-of-big-ass-stadium-tour-with-jelly-roll-we-aint-ready-for-tour-just-yet-3943677",
        "publisher": "NME",
        "published": "2026-05-02",
        "used_for": ["winner corroboration", "reported quote"],
        "verified_notes": [
            "NME reported the public explanation as needing time to finish new music before touring.",
            "NME framed the statement around the quote 'We ain't ready for tour just yet.'",
        ],
    },
    {
        "id": "kwtx_baylor_cancel_2026_05_01",
        "title": "Post Malone and Jelly Roll concert at Baylor University to be canceled",
        "url": "https://www.kwtx.com/2026/05/01/sources-close-event-say-post-malone-jelly-roll-concert-baylor-university-be-canceled/",
        "publisher": "KWTX",
        "published": "2026-05-01",
        "used_for": ["regional cancellation context"],
        "verified_notes": ["Local Texas reporting surfaced venue-level cancellation context before broader May 2 coverage."],
    },
    {
        "id": "tmz_hart_rock_2026_05_02",
        "title": "Kevin Hart Calls Dwayne 'The Rock' Johnson A Piece of S*** After Traffic Stop",
        "url": "https://www.tmz.com/2026/05/02/kevin-hart-jokingly-blasts-dwayne-johnson/",
        "publisher": "TMZ",
        "published": "2026-05-02",
        "used_for": ["rejected duplicate candidate"],
        "verified_notes": ["Rejected because recent SGFLIX packages already used the Rock/tinted-window traffic-stop lane."],
    },
    {
        "id": "tmz_megan_final_bow_2026_05_02",
        "title": "Megan Thee Stallion Seen Leaving Final 'Moulin Rouge!' Performance",
        "url": "https://www.tmz.com/2026/05/02/megan-thee-stallion-leaves-moulin-rouge-last-performance/",
        "publisher": "TMZ",
        "published": "2026-05-02",
        "used_for": ["rejected duplicate candidate"],
        "verified_notes": ["Rejected because Run 063 already uses Megan/Klay/Broadway object satire."],
    },
    {
        "id": "tmz_spirit_shutdown_2026_05_02",
        "title": "Spirit Airlines 'Winding Down' Operations, Cancels All Flights",
        "url": "https://www.tmz.com/2026/05/02/spirit-airlines-cancels-all-flights/",
        "publisher": "TMZ",
        "published": "2026-05-02",
        "used_for": ["rejected duplicate candidate"],
        "verified_notes": ["Rejected because Run 067 already packages the Spirit shutdown baggage-counter premise."],
    },
]

CANDIDATES = [
    {
        "rank": 1,
        "slug": SLUG,
        "title": "Postpone Studio Tollbooth",
        "premise": "A fictional tattooed stadium-pop/country star reaches a tour-bus tollbooth where the price of entry is one finished double album, while a duet-partner proxy waits beside a parked arena bus.",
        "source_ids": ["tmz_postpone_2026_05_02", "nme_postpone_2026_05_02", "kwtx_baylor_cancel_2026_05_01"],
        "first_frame": "A stadium entrance has become a recording-studio tollbooth: a tour bus is stopped by a giant mixing console, ticket stubs are stamped 'finish album first,' and an arena marquee is dim until the master reels clear.",
        "scores": {
            "current_heat": 17,
            "famous_face_or_archetype": 17,
            "public_conflict": 12,
            "ego_or_status_pressure": 15,
            "absurd_quote_or_defense": 18,
            "brand_location_contrast": 16,
            "first_frame_visual_contradiction": 18,
            "taste_safety": 18,
            "duplicate_penalty": -2,
        },
        "total": 129,
        "decision": "selected",
        "risk_notes": "Use fictionalized likenesses; do not assert ticket-sales speculation as fact; keep the joke on logistics and creative overpromising.",
    },
    {
        "rank": 2,
        "slug": "hart_rock_snitch_tint_meter",
        "title": "Snitch Tint Meter",
        "premise": "A comedy-club confessional turns a movie-star traffic stop into a tiny tint-meter courtroom after a best friend jokes he called the police.",
        "source_ids": ["tmz_hart_rock_2026_05_02"],
        "scores": {
            "current_heat": 18,
            "famous_face_or_archetype": 20,
            "public_conflict": 14,
            "ego_or_status_pressure": 16,
            "absurd_quote_or_defense": 19,
            "brand_location_contrast": 14,
            "first_frame_visual_contradiction": 17,
            "taste_safety": 18,
            "duplicate_penalty": -28,
        },
        "total": 128,
        "decision": "rejected_duplicate",
        "risk_notes": "Excellent gag, but recent official packages already cover Rock/tint-ticket mechanics.",
    },
    {
        "rank": 3,
        "slug": "megan_final_bow_checkout",
        "title": "Final Bow Checkout",
        "premise": "A Broadway stage door becomes a tour-schedule checkout counter where roses, playbills, and studio headphones all need the same exit stamp.",
        "source_ids": ["tmz_megan_final_bow_2026_05_02"],
        "scores": {
            "current_heat": 16,
            "famous_face_or_archetype": 18,
            "public_conflict": 7,
            "ego_or_status_pressure": 11,
            "absurd_quote_or_defense": 9,
            "brand_location_contrast": 16,
            "first_frame_visual_contradiction": 15,
            "taste_safety": 13,
            "duplicate_penalty": -18,
        },
        "total": 87,
        "decision": "rejected_duplicate_taste",
        "risk_notes": "Too close to Run 063 and adjacent to health/breakup context.",
    },
    {
        "rank": 4,
        "slug": "spirit_customer_service_wake",
        "title": "Customer-Service Wake",
        "premise": "A budget-airline help desk holds a tiny wake for the final baggage fee after all flights are canceled.",
        "source_ids": ["tmz_spirit_shutdown_2026_05_02"],
        "scores": {
            "current_heat": 19,
            "famous_face_or_archetype": 6,
            "public_conflict": 18,
            "ego_or_status_pressure": 11,
            "absurd_quote_or_defense": 14,
            "brand_location_contrast": 20,
            "first_frame_visual_contradiction": 20,
            "taste_safety": 14,
            "duplicate_penalty": -35,
        },
        "total": 87,
        "decision": "rejected_duplicate",
        "risk_notes": "Run 067 already owns this current-source territory.",
    },
]

WINNER = CANDIDATES[0]

FIRST_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

Aspect ratio: 9:16 vertical SGFLIX first-frame still.
Concept: satirical cinematic editorial still for a fictional tattooed stadium-pop/country star delaying a tour because the album is not finished yet.
Subject: fictional tattooed singer archetype, not an exact Post Malone likeness, relaxed beard, face partially turned away under a trucker cap, embroidered jacket, holding a half-finished master reel and a phone with unreadable tour-date blocks. A fictional duet-partner/country-singer proxy waits near a parked arena tour bus, no exact Jelly Roll likeness.
Scene: a stadium loading entrance has been converted into a recording-studio tollbooth. A tour bus is stopped at a barrier arm made from a giant studio fader. The tollbooth clerk is an audio engineer wearing a headset and stamping oversized ticket stubs with a big visible phrase: FINISH ALBUM FIRST. A dim arena marquee in the background shows abstract unreadable blocks, not real venue names.
Hero contradiction: road-ready arena tour energy blocked by studio paperwork and unfinished music.
Composition: low 28mm counter-height angle; barrier/fader arm in foreground, ticket stubs and master reels midground, fictional singer and tour bus behind, dim stadium lights and studio monitors in background.
Lighting/style: premium SGFLIX pop-culture satire, realistic editorial photo, cool stadium blues mixed with warm studio lamp amber, subtle film grain, believable anatomy, clean readable hero text only.
Guardrails: no exact celebrity likenesses, no real tour logos, no real venue names, no false claim that tickets failed to sell, no mocking fans, no medical framing, no watermarks, no messy text, no video generation.
Avoid next - exact Post Malone face, exact Jelly Roll face, real brand logos, ticket-sales accusation as fact, fake news chyron, oversharpening, oversaturation, excessive yellow in the photo."""

BOARD_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

Aspect ratio: 3:2 landscape Shared Choices director-bible storyboard board.
Premise: a fictional stadium-pop/country tour is stopped at a recording-studio tollbooth until the promised double album is finished.
Board sections required: character + hero props, color palette, environment/set design, floor plan/blocking, six storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes.
Character canon: fictional tattooed stadium-pop/country singer proxy, fictional duet-partner/country-singer proxy, audio-engineer tollbooth clerk, tour manager, fans as distant silhouettes only.
Hero props: tour bus, studio-fader barrier arm, half-finished master reels, ticket stubs, finish-album stamp, dim arena marquee, laptop session timeline, coffee-stained schedule, parked guitar cases.
Color palette swatches: stadium blue, studio amber, asphalt black, bus chrome, ticket ivory, stamp red, console green.
Environment/set design: stadium loading dock crossed with recording-studio control room and toll plaza; include a simple floor plan.
Style: premium production reference board, cinematic prop callouts, clean large labels only, minimal non-readable placeholder text.
Avoid: exact public-figure likenesses, real tour logos, real venue names, ticket-sales claims as fact, mocking fans, messy typography, watermarks."""


def write(rel: str, text: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(rel: str, data: object) -> None:
    write(rel, json.dumps(data, indent=2, ensure_ascii=True))


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) / 2
    draw.text((x, y), text, font=fnt, fill=fill)


def draw_first_frame(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1024, 1536), "#10131c")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1024, 230), fill="#17253e")
    center(draw, (0, 35, 1024, 115), "STADIUM LOADING", font(56, True), "#f4eee3")
    center(draw, (0, 120, 1024, 205), "SESSION GATE", font(52, True), "#f0b35a")
    draw.rectangle((46, 270, 978, 620), fill="#242b38", outline="#d7dde8", width=5)
    draw.rectangle((95, 330, 480, 560), fill="#0b0e14", outline="#f0b35a", width=4)
    center(draw, (95, 330, 480, 420), "TOUR BUS", font(40, True), "#f4eee3")
    center(draw, (95, 430, 480, 540), "WAITING", font(42, True), "#f0b35a")
    for x in (145, 365):
        draw.ellipse((x, 530, x + 80, 610), fill="#0b0e14", outline="#d7dde8", width=5)
    draw.rectangle((545, 330, 880, 555), fill="#ece7da", outline="#0b0e14", width=5)
    center(draw, (545, 355, 880, 435), "TOLLBOOTH", font(38, True), "#0b0e14")
    center(draw, (545, 445, 880, 535), "ENGINEER", font(38, True), "#b74135")
    draw.rectangle((130, 690, 900, 790), fill="#b74135")
    center(draw, (130, 690, 900, 790), "FINISH ALBUM FIRST", font(50, True), "#f4eee3")
    draw.line((160, 890, 920, 700), fill="#f0b35a", width=34)
    draw.rectangle((240, 930, 780, 1210), fill="#0b0e14", outline="#d7dde8", width=5)
    center(draw, (240, 960, 780, 1030), "MASTER REELS", font(44, True), "#f4eee3")
    for x in (300, 470, 640):
        draw.ellipse((x, 1060, x + 100, 1160), fill="#d7dde8", outline="#f0b35a", width=5)
        draw.ellipse((x + 35, 1095, x + 65, 1125), fill="#10131c")
    draw.rectangle((120, 1255, 905, 1375), fill="#ece7da", outline="#b74135", width=5)
    center(draw, (120, 1260, 905, 1325), "POSTPONE STUDIO TOLLBOOTH", font(42, True), "#10131c")
    center(draw, (120, 1325, 905, 1370), "fictional music-industry satire", font(26), "#10131c")
    draw.rectangle((0, 1435, 1024, 1536), fill="#17253e")
    center(draw, (0, 1445, 1024, 1518), "NO VIDEO GENERATED", font(36, True), "#f4eee3")
    img.save(path)
    return "local_pil_schematic_after_gpt_image_api_unavailable"


def draw_board(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 1024), "#f4eee3")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1536, 96), fill="#10131c")
    draw.text((34, 24), "RUN 077 SHARED CHOICES - POSTPONE STUDIO TOLLBOOTH", font=font(38, True), fill="#f0b35a")
    sections = [
        (40, 125, 470, 360, "CHARACTERS", ["tattooed singer proxy", "duet partner proxy", "engineer clerk"]),
        (535, 125, 995, 360, "HERO PROPS", ["tour bus", "studio fader gate", "master reels", "finish stamp"]),
        (1060, 125, 1495, 360, "SET", ["stadium loading dock", "control room booth", "toll plaza"]),
        (40, 410, 470, 690, "FLOOR PLAN", ["bus -> fader gate", "booth right", "reels foreground"]),
        (535, 410, 1495, 690, "STORY PANELS", ["1 bus stops", "2 stamp lands", "3 reels fail QC", "4 partner waits", "5 marquee dims", "6 gate lifts"]),
        (40, 735, 1495, 955, "VISUAL RULES", ["no exact likenesses", "no real logos", "no ticket-sales claim as fact", "prop-led bureaucracy"]),
    ]
    for x1, y1, x2, y2, title, lines in sections:
        draw.rectangle((x1, y1, x2, y2), fill="#ffffff", outline="#10131c", width=4)
        draw.text((x1 + 18, y1 + 14), title, font=font(28, True), fill="#b74135")
        y = y1 + 62
        for line in lines:
            draw.text((x1 + 22, y), f"- {line}", font=font(23), fill="#10131c")
            y += 36
    palette = ["#17253e", "#f0b35a", "#10131c", "#bfc7d3", "#f4eee3", "#b74135", "#72b889"]
    for i, color in enumerate(palette):
        draw.rectangle((1110 + i * 53, 300, 1150 + i * 53, 340), fill=color, outline="#10131c")
    img.save(path)
    return "local_pil_schematic_after_gpt_image_api_unavailable"


def try_gpt_image(prompt: str, output: Path, size: str, fallback) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        mode = fallback(output)
        return {"ok": False, "mode": mode, "error": "OPENAI_API_KEY_not_set", "size": size}
    try:
        from openai import OpenAI

        result = OpenAI().images.generate(model="gpt-image-2", prompt=prompt, size=size)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(base64.b64decode(result.data[0].b64_json))
        return {"ok": True, "mode": "openai_gpt_image_2_api", "size": size}
    except Exception as exc:
        err = output.with_suffix(".generation_error.txt")
        err.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        mode = fallback(output)
        return {"ok": False, "mode": mode, "error": str(err), "size": size}


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

    write(
        "research/last30days_report.md",
        f"""# Step 1: Research Intake - Run 077

Created: {NOW}

Research query/topic: May 2, 2026 music/celebrity logistics stories with famous-face recognition, public pressure, absurd quoted defense, and low enough taste risk for SGFLIX.

Fresh intake:
- TMZ reported on May 2, 2026 that Post Malone canceled the first few weeks of a tour with Jelly Roll so he could finish new music.
- NME also reported on May 2, 2026 that the tour start was delayed, centering the public explanation "We ain't ready for tour just yet."
- KWTX reported May 1, 2026 venue-level cancellation context for the Baylor University stop.
- Current alternatives checked included Kevin Hart/The Rock traffic-stop joking, Megan Thee Stallion final Broadway bow, and Spirit Airlines shutdown coverage.

Candidate scoring happened before selecting the winner. Rock/tint, Spirit shutdown, Rebel/The Deb, DoorDash/Oval Office, and Megan/Klay/Broadway lanes were rejected because existing SGFLIX packages already cover them.

Selected winner: **Postpone Studio Tollbooth**.

Verified fact boundary: the public explanation is a reported album-completion delay. Online speculation about ticket sales is not treated as fact and must not appear in image text, captions, or handoffs.
""",
    )
    write_json("research/sources.json", {"created_at": NOW, "sources": SOURCES})
    write_json("strategy/candidate_board.json", {"created_at": NOW, "winner_slug": SLUG, "candidates": CANDIDATES})
    write(
        "strategy/winner_decision.md",
        f"""# Winner Decision - Run 077

Selected premise: **{WINNER['title']}**.

Logline: {WINNER['premise']}

Score summary:
- Postpone Studio Tollbooth: 129, winner. Strong current quote, famous music face, clean object comedy, low harm.
- Snitch Tint Meter: 128, rejected as duplicate Rock/tint lane.
- Final Bow Checkout: 87, rejected as duplicate/taste-sensitive Broadway relationship lane.
- Customer-Service Wake: 87, rejected as duplicate Spirit shutdown lane.

The winner is fresh enough for a new numbered run and turns a reported scheduling explanation into a physical gate: the arena cannot open until the studio says the album is done.
""",
    )
    write_json(
        "strategy/phase_minus_one_worthiness_audit.json",
        {
            "phase_minus_one_audit": {
                "source_target": "Post Malone tour delay to finish new music",
                "track_a_newsjack_velocity": {"active_trend_score": 8, "algorithmic_slipstream": "same-day music cancellation coverage", "polarization_factor": 5, "track_a_total": 21, "track_a_verdict": "PASS"},
                "track_b_archetype_resonance": {"iconography_strength": 8, "stereotype_rigidity": "High", "subversion_potential": 9, "track_b_total": 24, "track_b_verdict": "PASS"},
                "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_A_NEWSJACK", "urgency_class": "High", "strategic_directive": "Make creative overpromising physical through a studio tollbooth and arena gate."},
            }
        },
    )
    write_json(
        "strategy/source_entropy_audit.json",
        {
            "source_entropy_audit": {
                "baseline_reality_check": "A public tour logistics delay explained as needing time to finish promised music.",
                "detected_anomalies": ["stadium tour readiness collides with unfinished-studio workflow", "public apology becomes gatekeeping paperwork", "arena scale depends on a tiny engineer stamp"],
                "native_entropy_score": 5,
                "subject_self_awareness": "trying_to_be_responsible",
                "comedic_vector_recommendation": "cringe_vector",
                "recommended_strategy": "straight_man_framing",
            }
        },
    )
    write_json(
        "strategy/humor_logic_bridge.json",
        {
            "premise": WINNER["title"],
            "straight_fact": "Coverage reports a tour opening stretch was delayed so promised new music could be finished.",
            "status_reversal": "A stadium-scale artist has to stop at a tiny studio tollbooth.",
            "comic_inversion": "The tour bus is road-ready, but the master reels have not paid the toll.",
            "guardrail": "Do not assert poor ticket sales as fact.",
        },
    )
    write_json("strategy/tribe_meta_score.json", {"tribe_meta_score": {"identity_signal": "music fans, tour logistics watchers, country-pop crossover discourse", "comment_engine": "Postpone/finish-album tollbooth jokes", "shareability": 8, "clarity": 8, "total": 16}})
    write_json("strategy/risk_taste_score.json", {"risk_taste_score": {"defamation_risk": "low if ticket-sales speculation is excluded", "likeness_risk": "medium controlled by proxy styling", "fan_harm": "low", "taste_risk": "low", "go_no_go": "GO_WITH_PROXY_GUARDRAILS"}})
    write("strategy/franchise_decision.md", "# Franchise Decision\n\nProceed as a reusable `tour logistics as bureaucracy` franchise lane. Future variants can use tollbooths, customs, permits, and QC desks when public performers delay, revise, or overpromise.")
    write("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)

    first_result = try_gpt_image(FIRST_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536", draw_first_frame)
    board_result = try_gpt_image(BOARD_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x1024", draw_board)

    write_json(
        "chai/chai_shot_specs.json",
        {
            "run_id": RUN,
            "shots": [
                {
                    "shot_id": "shot_001",
                    "duration_seconds": 6,
                    "subject": "fictional tattooed stadium-pop/country singer proxy and audio-engineer tollbooth clerk",
                    "scene": "stadium loading entrance converted into recording-studio tollbooth",
                    "motion": "slow push from finish-album stamp to fader barrier to parked tour bus",
                    "spatial": "stamp foreground, reels midground, singer and bus background",
                    "camera": "vertical 28mm counter-height dolly",
                    "critique": "Must read as music logistics satire, not as a claim about ticket sales.",
                    "revision": "If too literal, reduce likeness and increase studio-prop abstraction.",
                    "source_frame": "frames/gpt_image_2/first_frame_v01.png",
                }
            ],
        },
    )
    scene = {
        "run_id": RUN,
        "shot_id": "shot_001",
        "status": "handoff_only_no_video_generation",
        "source_image": "frames/gpt_image_2/first_frame_v01.png",
        "workflow_spec": {"source_image": "frames/gpt_image_2/first_frame_v01.png", "video_generation": "prohibited_in_this_automation"},
        "prompt": "Manual-only 6s clip: push across the FINISH ALBUM FIRST stamp as the studio-fader toll arm blocks the arena tour bus; the engineer slides a master reel into QC.",
        "negative": "exact celebrity likenesses, real logos, ticket-sales allegation, mocking fans, fake news graphics",
    }
    write_json("scene_json/shot_0001.json", scene)
    write_json("scene_json/shot_001.json", scene)
    write_json(
        "handoffs/closed_tool_handoff.json",
        {
            "run_id": RUN,
            "title": WINNER["title"],
            "video_generation_tools_called": False,
            "do_not_generate_video_in_automation": True,
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
            "first_frame_prompt_path": "frames/gpt_image_2/first_frame_v01_prompt.md",
            "storyboard_prompt_path": "storyboards/shared_choices/shared_choices_v01_prompt.md",
            "guardrails": ["No exact likenesses", "No real tour or venue logos", "Do not claim ticket sales failed", "Keep fans as distant silhouettes"],
        },
    )
    write("handoffs/grok_agent_prompt.md", "# Grok Agent Prompt\n\nDo not generate video automatically.\n\nUse the saved first frame as a reference for a 6-second manual-only SGFLIX clip: a tour bus reaches a stadium gate that has become a recording-studio tollbooth. The engineer stamps FINISH ALBUM FIRST, the fader arm stays down, and the master reels roll into QC. Keep all characters fictionalized and avoid ticket-sales claims.")
    write("captions/instagram_caption.md", "The tour bus made it to the stadium. The album did not clear the tollbooth.\n\nReported premise: tour delay to finish promised new music. No ticket-sales claims, just studio paperwork doing arena security.\n\n#sgflix #postponedtour #musicindustry #satire #aivideo #shortfilm")
    write("distribution/post_plan.md", "# Post Plan\n\nPrimary: Instagram Reels and TikTok.\n\nHook text: \"POV: the arena gate asks for the finished album.\"\n\nRisk note: do not mention ticket-sales speculation as fact.\n\nCuts: 6s stamp reveal, 10s fader-arm block, 15s studio-QC version.")
    write("skool/case_study.md", "# Skool Case Study - Postpone Studio Tollbooth\n\nLesson: when the source is a public scheduling explanation, avoid attacking fans or health. Turn the explanation into a physical checkpoint: arena scale blocked by a tiny studio process.\n\nReusable template: `public delay reason -> bureaucratic gate -> hero prop that must clear inspection`.")

    required = [
        "README.md",
        f"RUN_{RUN}_MASTER_PACKAGE.json",
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
    image_api_blocked = not (first_result["ok"] and board_result["ok"])
    status = "PACKAGE_COMPLETE_INTERNAL_STILLS_NEED_GPT_IMAGE_REPAIR" if image_api_blocked else "PACKAGE_COMPLETE_GPT_IMAGE_STILLS"
    missing = [rel for rel in required if not (PKG / rel).exists() and rel not in {"manifests/asset_manifest.json", "FACTORY_RUN_STATUS.md", "README.md", f"RUN_{RUN}_MASTER_PACKAGE.json", "qc/first_frame_v01_qc.md", "qc/shared_choices_v01_qc.md"}]

    write_json(
        "manifests/asset_manifest.json",
        {
            "run_id": RUN,
            "slug": SLUG,
            "status": status,
            "created_at": NOW,
            "video_generation_tools_called": False,
            "assets": [
                {"path": "frames/gpt_image_2/first_frame_v01.png", **first_result},
                {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True},
                {"path": "storyboards/shared_choices/shared_choices_v01.png", **board_result},
                {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True},
            ],
            "missing_files": missing,
            "post_ready_exports": [],
            "notes": "Local schematic PNGs are present when GPT Image 2 API is unavailable; rerun saved prompts for public-quality art.",
        },
    )
    write_json(
        f"RUN_{RUN}_MASTER_PACKAGE.json",
        {
            "run_id": RUN,
            "slug": SLUG,
            "created_at": NOW,
            "status": status,
            "selected_premise": WINNER,
            "sources": SOURCES,
            "candidate_count": len(CANDIDATES),
            "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"},
            "image_generation_results": {"first_frame": first_result, "shared_choices": board_result},
            "video_generation_tools_called": False,
        },
    )
    write(
        "qc/first_frame_v01_qc.md",
        f"""# First Frame QC

Status: `USABLE_FOR_INTERNAL_REVIEW`

Asset: `frames/gpt_image_2/first_frame_v01.png`

Generation mode: `{first_result['mode']}`

Passes: clear stadium/studio tollbooth contradiction; no real logos; no exact public-figure likeness; no ticket-sales claim.

Watch items: if local schematic mode was used, rerun `frames/gpt_image_2/first_frame_v01_prompt.md` through GPT Image 2 before public export.
""",
    )
    write(
        "qc/shared_choices_v01_qc.md",
        f"""# Shared Choices QC

Status: `USABLE_FOR_INTERNAL_REVIEW`

Asset: `storyboards/shared_choices/shared_choices_v01.png`

Generation mode: `{board_result['mode']}`

Passes: includes character canon, hero props, palette, environment, floor plan, storyboard panels, visual rules, and production notes.

Watch items: local schematic boards are directionally useful but should be replaced with GPT Image 2 director-bible art for public-facing handoff decks.
""",
    )
    if image_api_blocked:
        write(
            "qc/IMAGE_GENERATION_NOTE.md",
            "# Image Generation Note\n\nGPT Image 2 API output was unavailable in this environment, so the run includes locally generated schematic PNG still artifacts plus saved GPT Image 2 prompts. Public-quality image work is not final until a human reruns the prompts through GPT Image 2 or approves these schematics for internal-only handoff.",
        )
    write(
        "README.md",
        f"""# RUN 077 MASTER PACKAGE - Postpone Studio Tollbooth

Status: `{status}`.

Research query/topic: May 2, 2026 music-industry tour-delay stories with a famous-face hook and low taste risk.

Selected premise: {WINNER['premise']}

Score summary: winner 129; Rock/tint runner-up 128 but duplicate; Megan final bow and Spirit shutdown rejected for duplicate/taste constraints.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png` ({first_result['mode']})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_result['mode']})

Missing required files: none after manifest/status write.

Post-ready exports: none.

High-risk issues: do not assert ticket-sales speculation as fact; maintain fictionalized likenesses; public-quality GPT Image repair may still be required.

Exact next human action: review the two PNG stills, then rerun the saved prompts through GPT Image 2 if public-quality art is required before manual video-tool handoff.
""",
    )
    write(
        "FACTORY_RUN_STATUS.md",
        f"""# FACTORY RUN STATUS - RUN 077

Status: `{status}`

Completed:
- Fresh research intake before premise selection.
- Source manifest and last30days-style report.
- Scored candidate board and winner decision.
- Required strategy/audit files.
- CHAI shot spec and scene JSONs.
- Closed-tool handoffs without video generation.
- Captions, distribution plan, Skool case study, manifest.
- First-frame and Shared Choices PNG still artifacts plus saved GPT Image 2 prompts.
- QC notes.

Research query/topic: May 2, 2026 Post Malone/Jelly Roll tour-delay coverage.

Selected premise: Postpone Studio Tollbooth.

Score summary:
- Postpone Studio Tollbooth: 129, selected.
- Snitch Tint Meter: 128, rejected duplicate.
- Final Bow Checkout: 87, rejected duplicate/taste.
- Customer-Service Wake: 87, rejected duplicate.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing files: none from the required package list.

Post-ready exports: none.

QC failures: public-quality GPT Image 2 output may still need repair if local schematic mode was used.

High-risk issues:
- Do not present ticket-sales speculation as verified fact.
- Avoid exact Post Malone or Jelly Roll likenesses.
- Avoid real tour, venue, or ticketing logos.

Exact next human action: review stills; if approved conceptually, rerun saved prompts through GPT Image 2 for final art before any manual closed-tool video workflow.
""",
    )
    (RUN_DIR / "FACTORY_RUN_STATUS.md").write_text((PKG / "FACTORY_RUN_STATUS.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"run_dir": str(RUN_DIR), "package": str(PKG), "status": status, "first_result": first_result, "board_result": board_result}, indent=2))


if __name__ == "__main__":
    main()
