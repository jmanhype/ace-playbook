import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN = "025"
SLUG = "mariah_rock_hall_coat_check"
TITLE = "Rock Hall Coat-Check Snub"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

SOURCES = [
    {
        "id": "rockhall_2026_nominees",
        "title": "Rock & Roll Hall of Fame Foundation Reveals 2026 Performer Nominees List",
        "url": "https://rockhall.com/press-release/rock-roll-hall-of-fame-foundation-reveals-2026-performer-nominees-list/",
        "publisher": "Rock & Roll Hall of Fame",
        "published": "2026-02-25",
        "used_for": "verification that Mariah Carey was a 2026 performer nominee",
        "verification_status": "official source",
    },
    {
        "id": "rockhall_2026_inductees",
        "title": "The Rock & Roll Hall Of Fame Reveals 2026 Inductees",
        "url": "https://rockhall.com/press-release/the-rock-roll-hall-of-fame-reveals-2026-inductees/",
        "publisher": "Rock & Roll Hall of Fame",
        "published": "2026-04-13",
        "used_for": "verification of 2026 inductee class; Carey absent from announced inductees",
        "verification_status": "official source",
    },
    {
        "id": "tmz_mariah_who_cares",
        "title": "Mariah Carey Says She Doesn't Care About Rock & Roll Hall of Fame Snub",
        "url": "https://www.tmz.com/2026/04/20/mariah-carey-does-not-care-about-rock-hall-snub/",
        "publisher": "TMZ",
        "published": "2026-04-20",
        "used_for": "winner quote and public reaction framing",
        "verification_status": "entertainment-press street video recap",
    },
    {
        "id": "vg_mariah_quote_recap",
        "title": "Mariah Carey snytt - bryr seg ikke",
        "url": "https://www.vg.no/rampelys/i/xrlKOB/mariah-carey-droppet-av-rock-hall-of-fame-bryr-seg-ikke",
        "publisher": "VG",
        "published": "2026-04-21",
        "used_for": "secondary quote recap",
        "verification_status": "secondary entertainment recap",
    },
    {
        "id": "guardian_rebel_defamation",
        "title": "Actor in feud with Rebel Wilson signed $150,000 record deal, court told",
        "url": "https://www.theguardian.com/film/2026/apr/22/actor-in-feud-rebel-wilson-signed-150000-record-deal-ntwnfb",
        "publisher": "The Guardian",
        "published": "2026-04-22",
        "used_for": "candidate board alternative",
        "verification_status": "reported court coverage",
    },
    {
        "id": "tmz_depp_vampires_lawsuit",
        "title": "Johnny Depp's Hollywood Vampires Fighting Promoter Over Axed Slovakia Concert",
        "url": "https://www.tmz.com/2026/04/01/johnny-depp-hollywood-vampires-fight-lawsuit-over-canceled-concert/",
        "publisher": "TMZ",
        "published": "2026-04-01",
        "used_for": "candidate board alternative",
        "verification_status": "entertainment-press lawsuit recap",
    },
    {
        "id": "justjared_howie_regret_apology",
        "title": "Howie Mandel Now 'Regrets' Apologizing to Kelly Ripa Following On-Air Exchange",
        "url": "https://www.justjared.com/2026/04/02/howie-mandel-now-regrets-apologizing-to-kelly-ripa-following-on-air-exchange/",
        "publisher": "Just Jared",
        "published": "2026-04-02",
        "used_for": "candidate board alternative",
        "verification_status": "entertainment recap",
    },
    {
        "id": "tmz_rock_tint_duplicate",
        "title": "Dwayne 'The Rock' Johnson Pulled Over by Police After Walk of Fame Event",
        "url": "https://www.tmz.com/2026/04/30/dwayne-johnson-pulled-over-in-los-angeles/",
        "publisher": "TMZ",
        "published": "2026-04-30",
        "updated": "2026-05-01",
        "used_for": "candidate board alternative rejected as already packaged in run 024",
        "verification_status": "entertainment-press report",
    },
]

CANDIDATES = [
    {
        "rank": 1,
        "slug": SLUG,
        "premise": "A fictional pop-diva archetype calmly checks a glittering vocal-coach coat at a Rock Hall cloakroom while a tiny velvet-rope attendant treats a non-induction like a lost claim ticket.",
        "source_basis": ["rockhall_2026_nominees", "rockhall_2026_inductees", "tmz_mariah_who_cares", "vg_mariah_quote_recap"],
        "scores": {
            "famous_face_or_public_recognition": 10,
            "public_conflict": 7,
            "ego_or_status_pressure": 10,
            "humiliation_or_absurd_defense": 9,
            "brand_or_location_contrast": 9,
            "first_frame_visual_contradiction": 10,
            "taste_safety": 8,
            "freshness": 8,
        },
        "total": 71,
        "risk_notes": "Use a fictional diva archetype and official nomination/inductee facts. Do not imply she literally campaigned beyond sourced public framing.",
    },
    {
        "rank": 2,
        "slug": "rock_tint_meter_duplicate_penalty",
        "premise": "The Walk of Fame tint-meter stop remains a strong visual joke, but it was already packaged as run 024 and is not eligible as a new winner.",
        "source_basis": ["tmz_rock_tint_duplicate"],
        "scores": {
            "famous_face_or_public_recognition": 10,
            "public_conflict": 6,
            "ego_or_status_pressure": 7,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 10,
            "first_frame_visual_contradiction": 10,
            "taste_safety": 9,
            "freshness": 10,
            "duplicate_penalty": -12,
        },
        "total": 58,
        "risk_notes": "Rejected because factory must create a new numbered package with a fresh selected premise.",
    },
    {
        "rank": 3,
        "slug": "rebel_deb_subpoena_musical",
        "premise": "A musical-comedy courtroom where Instagram Stories are wheeled in as tap-dancing exhibits during the Rebel Wilson/The Deb defamation trial.",
        "source_basis": ["guardian_rebel_defamation"],
        "scores": {
            "famous_face_or_public_recognition": 8,
            "public_conflict": 10,
            "ego_or_status_pressure": 9,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 9,
            "first_frame_visual_contradiction": 9,
            "taste_safety": 3,
            "freshness": 9,
        },
        "total": 65,
        "risk_notes": "Rejected for active litigation and sexual-harassment-adjacent claims; too easy to misstate facts.",
    },
    {
        "rank": 4,
        "slug": "depp_vampires_venue_receipt",
        "premise": "A gothic supergroup fights a concert invoice while a tiny venue-inspection clipboard blocks the castle gate.",
        "source_basis": ["tmz_depp_vampires_lawsuit"],
        "scores": {
            "famous_face_or_public_recognition": 9,
            "public_conflict": 7,
            "ego_or_status_pressure": 7,
            "humiliation_or_absurd_defense": 6,
            "brand_or_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 6,
            "freshness": 5,
        },
        "total": 56,
        "risk_notes": "Good props, but less fresh and more contract-specific than the winner.",
    },
    {
        "rank": 5,
        "slug": "howie_apology_return_desk",
        "premise": "A daytime-TV apology is returned at customer service because the comedian later regrets issuing it.",
        "source_basis": ["justjared_howie_regret_apology"],
        "scores": {
            "famous_face_or_public_recognition": 6,
            "public_conflict": 6,
            "ego_or_status_pressure": 6,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 7,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 8,
            "freshness": 5,
        },
        "total": 54,
        "risk_notes": "Clean but lower famous-face signal.",
    },
]

FIRST_FRAME_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 9:16 vertical.
subject: a fictional legendary pop-diva archetype inspired by champagne glamour and vocal-superstar poise, not an exact real celebrity likeness; huge sunglasses, glossy curls, sparkling black-gold evening coat, calm amused expression.
scene: an elegant Rock and Roll Hall of Fame style cloakroom, with velvet ropes, guitar-shaped coat hooks, museum lighting, abstract gold records on the wall, and a tiny claim-ticket window. No real logos, no readable names.
comedic contradiction: the diva is checking a shimmering vocal-coach coat labeled only by a blank claim ticket while a tiny coat-check attendant solemnly stamps a giant card that says nothing readable. The induction trophy display is visible far behind the ropes, deliberately out of reach, like airport lost luggage.
camera: cinematic vertical 35mm, low angle from the coat-check counter, diva in midground, tiny stamp and claim ticket in foreground, trophy case background bokeh.
lighting and style: premium tabloid-cinema realism, warm museum gold, glossy black accents, paparazzi-white highlights, subtle film grain, calm deadpan celebrity energy.
negative instructions: no exact Mariah Carey likeness, no readable Rock Hall branding, no real award logos, no mocking body features, no cruel rejection poster, no distorted hands, no extra text, no watermark, no fake news lower-third."""

SHARED_CHOICES_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 16:9.
Create one SGFLIX Shared Choices director-bible board for Run 025, internal title 'Rock Hall Coat-Check Snub'. Include: fictional pop-diva character canon, tiny cloakroom attendant archetype, hero props of blank claim ticket, stamp pad, guitar-shaped coat hook, velvet rope, trophy case, gold-record wall, diva sunglasses, glitter coat; color palette swatches museum gold / piano black / champagne ivory / velvet red / paparazzi white; environment and set design for a Rock Hall style museum cloakroom; floor plan and blocking; six storyboard panels with camera/lens/movement notes; lighting/mood/style notes; visual rules; production notes. Keep text minimal and mostly label-like. Do not use exact Mariah Carey likeness, readable celebrity names, real logos, real award marks, defamatory claims, or readable museum branding."""


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
    img = Image.new("RGB", size, "#111111")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 54)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except Exception:
        title_font = body_font = None
    draw.rectangle([38, 38, size[0] - 38, size[1] - 38], outline="#d7b45d", width=6)
    draw.text((76, 82), title, fill="#f8f2e4", font=title_font)
    y = 178
    for line in subtitle.split("\n"):
        draw.text((76, y), line, fill="#e5d6b5", font=body_font)
        y += 44
    if board:
        colors = ["#d7b45d", "#0e0e10", "#efe2c7", "#7b1420", "#f7f3ea"]
        for idx, color in enumerate(colors):
            x = 82 + idx * 118
            draw.rectangle([x, size[1] - 164, x + 84, size[1] - 86], fill=color, outline="#f8f2e4")
        for i in range(6):
            x = 82 + (i % 3) * 462
            y = 334 + (i // 3) * 224
            draw.rectangle([x, y, x + 372, y + 154], outline="#7b1420", width=4)
            draw.text((x + 18, y + 18), f"PANEL {i + 1}", fill="#f8f2e4", font=body_font)
    else:
        draw.rectangle([220, 780, 820, 970], outline="#7b1420", width=6)
        draw.text((250, 824), "BLANK CLAIM TICKET\nOUTRANKS TROPHY CASE", fill="#f8f2e4", font=title_font)
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
            f"RUN {RUN} IMAGE BLOCKED",
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

    prev = ROOT / "sgflix_runs" / "run_024_rock_tint_meter" / "NEXT_STEP_REPORT_2026-05-02.md"
    prev.write_text(
        "# Next-Step Report - Run 024\n\n"
        "`run_024_rock_tint_meter` appears package-complete by checklist, but its QC includes image-generation-blocked artifacts. "
        "Next step: rerun the saved first-frame and Shared Choices prompts through GPT Image 2 if the current PNGs are placeholders, then redo image QC before any closed-tool video workflow.\n",
        encoding="utf-8",
    )

    write_text("research/last30days_report.md", f"""
# Last 30 Days Research Intake - Run {RUN}

Created: {NOW}

Research query/topic: fresh celebrity-status contradictions from April 2026 to May 2, 2026, emphasizing famous face, public status friction, quotable ego/shrug, low sensitivity, and strong first-frame contradiction.

Intake method: current web/source scan first, then candidate scoring. No premise was selected from local images, old storyboards, nearby assets, or prior handoffs.

Shortlist:
- Mariah Carey / Rock & Roll Hall of Fame snub response: official sources verify Carey was a 2026 performer nominee and not listed in the announced inductees. TMZ and VG recapped a street-video response where she dismissed the snub. This has famous-face signal, status pressure, a quotable shrug, and a clean museum/cloakroom visual.
- Dwayne Johnson Walk of Fame tinted-window stop: very strong, but rejected because it was already packaged as Run 024.
- Rebel Wilson / The Deb defamation hearing: fresh and dramatic, but rejected due to active litigation and sexual-harassment-adjacent factual risk.
- Johnny Depp Hollywood Vampires canceled-concert lawsuit: usable gothic invoice visual, but less fresh and more contract-specific.
- Howie Mandel apology-regret item: clean apology-return-desk visual, but lower fame/status impact.

Winner: `{SLUG}`.

Verified facts used:
- Rock Hall official nominee list included Mariah Carey among 2026 performer nominees.
- Rock Hall official April 13, 2026 inductee announcement did not list Carey among performer inductees.
- TMZ reported on April 20, 2026 that Carey said she did not care about the snub.
- VG recapped the same public response on April 21, 2026.

Unverified or avoided:
- No claim is made about private feelings, campaigning, vote totals, or Rock Hall internal deliberations.
- The creative frame uses a fictional diva archetype, not an exact public-figure likeness.
""")
    write_json("research/sources.json", {"created_at": NOW, "sources": SOURCES})
    write_json("strategy/candidate_board.json", {"created_at": NOW, "selection_order": "research_first_then_scoring_then_package", "candidates": CANDIDATES})
    write_text("strategy/winner_decision.md", f"""
# Winner Decision - Run {RUN}

Selected premise: **{TITLE}**.

Logline: A fictional pop-diva archetype treats a Rock Hall non-induction like a lost coat-check ticket, while a tiny cloakroom stamp becomes more powerful than a trophy case.

Score summary:
- Winner total: 71
- Runner-up by raw fit was the Rebel Wilson musical-courtroom premise at 65, but it failed taste/legal risk gates.
- The Dwayne tint-meter item would remain strong, but it was rejected as a duplicate because Run 024 already packaged it.

Decision: proceed with still-image package and closed-tool handoff only. The joke lives in status bureaucracy, not in cruelty toward the artist or invented Rock Hall facts.
""")
    write_json("strategy/phase_minus_one_worthiness_audit.json", {
        "phase_minus_one_audit": {
            "source_target": "Mariah Carey 2026 Rock Hall nomination without induction plus public shrug response",
            "track_a_newsjack_velocity": {
                "active_trend_score": 7,
                "algorithmic_slipstream": "April 2026 inductees announcement followed by entertainment press reaction coverage",
                "polarization_factor": 4,
                "track_a_total": 11,
                "track_a_verdict": "PASS"
            },
            "track_b_archetype_resonance": {
                "iconography_strength": 10,
                "stereotype_rigidity": "High",
                "subversion_potential": 9,
                "track_b_total": 19,
                "track_b_verdict": "PASS"
            },
            "system_verdict": {
                "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                "primary_vector": "TRACK_B_EVERGREEN_ARCHETYPE",
                "urgency_class": "Medium-high",
                "strategic_directive": "Make institutional gatekeeping look like petty cloakroom bureaucracy."
            }
        }
    })
    write_json("strategy/source_entropy_audit.json", {
        "source_entropy_audit": {
            "baseline_reality_check": "An iconic singer was nominated for a music institution and did not appear in the announced inductee class; entertainment press framed her response as dismissive.",
            "detected_anomalies": ["maximum vocal legacy meets claim-ticket bureaucracy", "a trophy institution becomes a coat-check counter", "public snub is answered with elite calm"],
            "native_entropy_score": 5,
            "subject_self_awareness": "high",
            "comedic_vector_recommendation": "status_shrug_vector",
            "recommended_strategy": "museum_bureaucracy_as_antagonist"
        }
    })
    write_json("strategy/humor_logic_bridge.json", {
        "bridge": {
            "setup": "A legendary pop-diva archetype approaches a hall-of-fame institution.",
            "status_reversal": "The induction becomes a blank claim ticket controlled by a tiny cloakroom attendant.",
            "visual_rule": "The diva stays composed and glamorous; the institution looks small, fussy, and literal.",
            "punchline": "The lost coat ticket has more ceremony than the trophy case.",
            "do_not_cross": ["no exact likeness", "no invented private quote", "no real logos", "no misogynistic age/body joke", "no claim of corruption"]
        }
    })
    write_json("strategy/tribe_meta_score.json", {
        "tribe_meta_score": {
            "tribe_fit": 8,
            "shareability": 8,
            "duet_or_comment_prompt": 8,
            "caption_lore_potential": 9,
            "repeatable_franchise_value": 8,
            "notes": "Works for music fans, awards snub discourse, and diva-status memes without leaning on scandal."
        }
    })
    write_json("strategy/risk_taste_score.json", {
        "risk_taste_score": {
            "defamation_risk": 1,
            "legal_sensitivity": 1,
            "identity_or_health_sensitivity": 1,
            "public_figure_likeness_risk": 5,
            "brand_logo_risk": 5,
            "overall_risk": "Medium-low",
            "guardrails": ["fictionalize likeness", "use official facts only", "avoid readable Rock Hall branding", "make institution bureaucracy the butt of the joke"]
        }
    })
    write_text("strategy/franchise_decision.md", """
# Franchise Decision

Franchise lane: **Institutional Snub as Petty Desk Job**.

Repeatable template: a famous person encounters a grand institution, but the visible antagonist is a tiny clerk, claim ticket, barcode scanner, velvet rope, or waiting-room number.

Verdict: keep as a franchise seed. It is low-harm, highly legible, and can travel across music, sports, film, and award-show snub discourse.
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
        "concept": "A claim-ticket stamp turns a Rock Hall snub into cloakroom bureaucracy.",
        "camera": "slow push from blank claim ticket and stamp to composed diva at the counter",
        "motion": "stamp descends, velvet rope twitches, gold records shimmer, diva lowers sunglasses slightly",
        "no_video_generation": True,
        "workflow_spec": {"source_image": "frames/gpt_image_2/first_frame_v01.png", "render_lane": "manual_closed_tool_only_after_human_approval"}
    }
    write_json("scene_json/shot_001.json", shot)
    write_json("scene_json/shot_0001.json", {**shot, "shot_id": "shot_0001"})
    write_json("chai/chai_shot_specs.json", {
        "run_id": RUN,
        "specs": [{
            "shot_id": "shot_001",
            "subject": "fictional pop-diva archetype and tiny cloakroom attendant",
            "scene": "Rock Hall style museum cloakroom with trophy case beyond velvet rope",
            "motion": "claim ticket is stamped with absurd ceremony while diva remains calm",
            "spatial": "stamp and ticket foreground, counter midground, diva and trophy case background",
            "camera": "vertical 35mm slow push, low counter angle for status contrast",
            "critique": "Must not use exact celebrity likeness, readable museum marks, or cruel rejection signage.",
            "revision": "If the joke reads too mean, make the attendant and claim ticket more absurd and the diva more unbothered."
        }]
    })
    write_json("handoffs/closed_tool_handoff.json", {
        "run_id": RUN,
        "title": TITLE,
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

Create a 6-second satirical clip plan where a fictional pop-diva archetype handles a hall-of-fame non-induction like a lost cloakroom ticket. The institution should feel tiny and bureaucratic; the diva should feel composed and above it. Do not use exact likeness, readable logos, or invented factual claims.
""")
    write_text("captions/instagram_caption.md", """
The trophy case said no.

The coat-check ticket said "who cares?"

Routine museum bureaucracy. Maximum diva altitude.

#sgflix #satire #mariahcarey #rockhall #musicmemes #awardseason #aivideo #shortfilm
""")
    write_text("distribution/post_plan.md", """
# Post Plan

Primary surface: Instagram Reels / TikTok.

Hook text: "POV: the Hall of Fame turns into coat check."

Risk caption note: Use only official nomination/non-induction facts and entertainment-press reaction framing. Avoid claiming private feelings or vote reasons.

Cutdown plan: 6s claim-ticket stamp, 10s trophy-case reveal, 15s mini case-study with source-safe caption.
""")
    write_text("skool/case_study.md", """
# Skool Case Study - Run 025

Lesson: a public snub can be made funny by shrinking the institution instead of attacking the person.

Source fact: an official 2026 Rock Hall nominee did not appear in the announced inductee class, and entertainment press recapped a dismissive public response.

Factory move: convert grand institutional validation into a tiny coat-check ritual. The visual joke is bureaucracy outranking glamour.
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

QC notes: Must preserve composed-diva tone, no exact public-figure likeness, no readable museum branding, no cruel rejection poster.
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
        "title": TITLE,
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
# RUN {RUN} MASTER PACKAGE - {TITLE}

Status: {status}; no video footage generated.

Research query/topic: fresh celebrity-status contradictions April 2026-May 2 2026.

Selected premise: A fictional pop-diva archetype handles a Rock Hall non-induction like a lost coat-check ticket.

Score summary: winner 71; Rebel Wilson musical-courtroom alternative 65 but failed active-litigation risk; Run 024 tint-meter idea rejected as duplicate.

Still-image paths:
- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

Exact next human action: Review the still outputs. If they are placeholders, rerun the saved prompts through GPT Image 2, replace the placeholder PNGs, and redo QC before any manual video-tool handoff.
""")
    write_text("FACTORY_RUN_STATUS.md", f"""
# Factory Run Status - Run {RUN}

Status: {status}

New run id: `run_{RUN}_{SLUG}`

Research query/topic: fresh celebrity-status contradictions April 2026-May 2 2026.

Candidate board: created at `strategy/candidate_board.json`.

Selected premise: `{TITLE}`.

Score summary: winner 71; active-litigation runner-up rejected for taste/legal risk; duplicate Run 024 premise rejected.

Created files: all required package artifacts were written under `RUN_{RUN}_MASTER_PACKAGE`.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

Missing files: none from the required checklist.

Post-ready exports: none; stills require human review and, if blocked, GPT Image 2 repair.

QC failures: {"GPT Image API blocked; placeholder assets only." if blocked else "None known; human visual inspection still required."}

High-risk issues: exact public-figure likeness drift, readable Rock Hall branding, invented vote/internal-deliberation claims.

Exact next human action: Review still outputs; if blocked, rerun `first_frame_v01_prompt.md` and `shared_choices_v01_prompt.md` through GPT Image 2 and redo QC.
""")


if __name__ == "__main__":
    main()
