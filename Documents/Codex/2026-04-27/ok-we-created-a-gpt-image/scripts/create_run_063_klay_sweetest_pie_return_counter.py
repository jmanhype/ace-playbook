import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN = "063"
SLUG = "klay_sweetest_pie_return_counter"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

SOURCES = [
    {
        "id": "tmz_megan_klay_split",
        "title": "Megan Thee Stallion and Klay Thompson Split",
        "url": "https://www.tmz.com/2026/04/25/megan-thee-stallion-klay-thompson-break-up/",
        "publisher": "TMZ",
        "published": "2026-04-25",
        "updated": "2026-04-27",
        "used_for": "winner source",
        "verification_status": "entertainment press report; relationship statements attributed to TMZ/rep",
        "notes": "Search result and TMZ people page describe split, the 'Sweetest Pie' hook, and a statement about trust/fidelity/respect."
    },
    {
        "id": "tmz_megan_broadway_exit",
        "title": "Megan Thee Stallion Exiting Broadway Show Early After Klay Thompson Split",
        "url": "https://www.tmz.com/2026/04/27/megan-thee-stallion-leaving-broadway-after-klay-thompson-split/",
        "publisher": "TMZ",
        "published": "2026-04-27",
        "used_for": "winner context",
        "verification_status": "entertainment press report",
        "notes": "Used only for Broadway calendar/object contrast; do not turn private emotion into the joke."
    },
    {
        "id": "tmz_klay_people_page",
        "title": "Klay Thompson Latest News and Updates",
        "url": "https://www.tmz.com/people/klay-thompson/",
        "publisher": "TMZ",
        "published": "2026-05-02 crawl",
        "used_for": "source index confirmation",
        "verification_status": "current topic index",
        "notes": "Lists April 25 split, April 27 Broadway exit, April 30 rumor-harassment story, and prior Bentley/boat beats."
    },
    {
        "id": "ap_megan_broadway_health_context",
        "title": "Megan Thee Stallion takes 2 Broadway shows off after illness during Moulin Rouge!",
        "url": "https://apnews.com/article/d6aef0f4620366efa4df3a35926916d4",
        "publisher": "Associated Press",
        "published": "2026-04-01",
        "used_for": "context only",
        "verification_status": "wire-service report",
        "notes": "Confirms Broadway role context and early-April health episode; rejected as joke target for taste reasons."
    },
    {
        "id": "tmz_howard_stern_shakedown",
        "title": "Howard Stern Calls Ex-Employee's Lawsuit a Shakedown, Moves to Dismiss",
        "url": "https://www.tmz.com/2026/05/01/howard-stern-moves-to-dismiss-hostile-work-environment-lawsuit/",
        "publisher": "TMZ",
        "published": "2026-05-01",
        "used_for": "candidate alternative",
        "verification_status": "entertainment press court-doc recap",
        "notes": "Rejected for overlap with earlier Stern/cat-rescue package and legal/workplace sensitivity."
    },
    {
        "id": "variety_paramount_subscriber_suit",
        "title": "Paramount Faces Suit From Streaming Subscribers Seeking to Block Warner Bros. Deal",
        "url": "https://au.variety.com/2026/film/news/paramount-streaming-subscribers-private-antitrust-36122/",
        "publisher": "Variety",
        "published": "2026-05-01",
        "used_for": "candidate alternative",
        "verification_status": "trade press legal/business report",
        "notes": "Rejected for overlap with prior Paramount merger package and weaker famous-face signal."
    }
]

CANDIDATES = [
    {
        "rank": 1,
        "slug": SLUG,
        "premise": "A fictional NBA champion archetype arrives at a Broadway lost-and-found counter trying to return a heart-shaped 'Sweetest Pie' trophy, a toy boat nameplate, and a luxury-car gift receipt while a stage manager stamps everything NON-NEGOTIABLE.",
        "source_basis": ["tmz_megan_klay_split", "tmz_megan_broadway_exit", "tmz_klay_people_page"],
        "scores": {
            "famous_face_or_public_recognition": 9,
            "public_conflict": 8,
            "ego_or_status_pressure": 8,
            "humiliation_or_absurd_defense": 9,
            "brand_or_location_contrast": 10,
            "first_frame_visual_contradiction": 10,
            "taste_safety": 7,
            "freshness": 9
        },
        "total": 70,
        "risk_notes": "Use fictional likenesses and object satire. Do not depict cheating as fact beyond sourced/disputed public breakup framing; do not include Lexie Brown harassment rumors."
    },
    {
        "rank": 2,
        "slug": "stern_hush_money_mansion_hr",
        "premise": "A shock-jock mansion HR desk turns a 'shakedown' motion into a hush-money vending machine guarded by payroll folders and cat-rescue clipboards.",
        "source_basis": ["tmz_howard_stern_shakedown"],
        "scores": {
            "famous_face_or_public_recognition": 8,
            "public_conflict": 9,
            "ego_or_status_pressure": 8,
            "humiliation_or_absurd_defense": 9,
            "brand_or_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 4,
            "freshness": 10
        },
        "total": 64,
        "risk_notes": "Rejected: overlaps with prior Stern/cat-rescue package and active workplace allegations."
    },
    {
        "rank": 3,
        "slug": "paramount_subscriber_small_claims",
        "premise": "Five streaming subscribers drag a $110B studio merger into a small-claims theater lobby where every popcorn bucket asks for triple damages.",
        "source_basis": ["variety_paramount_subscriber_suit"],
        "scores": {
            "famous_face_or_public_recognition": 5,
            "public_conflict": 8,
            "ego_or_status_pressure": 7,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 9,
            "first_frame_visual_contradiction": 9,
            "taste_safety": 8,
            "freshness": 10
        },
        "total": 64,
        "risk_notes": "Rejected: prior Paramount merger package exists and no clean famous-face center."
    },
    {
        "rank": 4,
        "slug": "rayj_private_arbitration_lobby",
        "premise": "A public courtroom turns into a silent private-arbitration elevator where every button is an NDA clause.",
        "source_basis": ["tmz_rayj_arbitration_search"],
        "scores": {
            "famous_face_or_public_recognition": 8,
            "public_conflict": 9,
            "ego_or_status_pressure": 8,
            "humiliation_or_absurd_defense": 7,
            "brand_or_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 3,
            "freshness": 7
        },
        "total": 58,
        "risk_notes": "Rejected: adult tape context and overlap with prior Kardashian/Ray J receipt package."
    },
    {
        "rank": 5,
        "slug": "sydney_stagecoach_pda_checkpoint",
        "premise": "A Stagecoach VIP gate becomes a relationship milestone checkpoint with a private-jet dog carrier and cowboy boots on a clipboard.",
        "source_basis": ["tmz_sydney_stagecoach_gallery"],
        "scores": {
            "famous_face_or_public_recognition": 8,
            "public_conflict": 3,
            "ego_or_status_pressure": 6,
            "humiliation_or_absurd_defense": 5,
            "brand_or_location_contrast": 8,
            "first_frame_visual_contradiction": 7,
            "taste_safety": 8,
            "freshness": 9
        },
        "total": 54,
        "risk_notes": "Rejected: fresh but too soft; not enough conflict or humiliation."
    }
]

FIRST_FRAME_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 9:16 vertical.
subject: fictional NBA champion archetype, tall athletic silhouette, calm embarrassed posture, tasteful dark travel suit, carrying a sealed basketball gear bag; do not create an exact likeness of Klay Thompson or any real athlete.
scene: Broadway theater lost-and-found / returns counter after a glamorous musical performance. Red velvet ropes, warm marquee bulbs, backstage callboard, and a small counter sign that uses only simple unreadable marks. No real show logos, no readable celebrity names.
hero props: a heart-shaped dessert trophy labeled only with abstract icing marks, a tiny toy boat nameplate with unreadable scribbles, a luxury-car gift receipt printed as fake blocks, a basketball, and a rubber stamp reading visually as NON-NEGOTIABLE but with no legible small legal text.
comedic contradiction: the giant athlete waits politely while a tiny stern stage manager weighs the heart-shaped pie trophy on a ticket-office scale like it is official breakup evidence.
camera: vertical 35mm cinematic tabloid-realism, counter-height foreground with stamp and pie trophy huge, athlete midground, Broadway lights and backstage curtains background, clean first-second joke clarity.
lighting and style: premium satire still, warm tungsten theater glow, paparazzi flash rim light, believable human anatomy, no cruelty, no tears, no medical emergency, no cheating scene, no harassment rumors.
negative instructions: no exact Megan Thee Stallion likeness, no exact Klay Thompson likeness, no real logos, no readable show names, no sexualized framing, no courtroom, no police, no violence, no messy text, no watermark."""

SHARED_CHOICES_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 16:9.
Create one SGFLIX Shared Choices director-bible board for Run 063, internal title 'Sweetest Pie Return Counter'. Include: fictional NBA champion character canon, fictional Broadway stage-manager canon, hero props of heart-shaped pie trophy, fake luxury-car receipt, toy boat nameplate, basketball gear bag, velvet ropes, rubber stamp, ticket-office scale; color palette swatches theater red / warm marquee amber / black suit / chrome receipt gray / dessert pink used sparingly; environment/set design for Broadway lost-and-found counter; floor plan and blocking; six storyboard panels with camera/lens/movement notes; lighting/mood/style notes; visual rules; production notes. Keep text minimal and label-like. Do not use exact celebrity likenesses, real show logos, readable names, sexualized imagery, harassment rumors, medical-emergency imagery, or defamatory proof claims."""


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
    img = Image.new("RGB", size, "#171417")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 54)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except Exception:
        title_font = body_font = None
    draw.rectangle([38, 38, size[0] - 38, size[1] - 38], outline="#d1a14a", width=6)
    draw.text((76, 82), title, fill="#f6efe5", font=title_font)
    y = 178
    for line in subtitle.split("\n"):
        draw.text((76, y), line, fill="#e8ded2", font=body_font)
        y += 44
    if board:
        colors = ["#8e1f2f", "#d99b45", "#111111", "#b8b5ae", "#e89ab6"]
        for idx, color in enumerate(colors):
            x = 82 + idx * 118
            draw.rectangle([x, size[1] - 164, x + 84, size[1] - 86], fill=color, outline="#f6efe5")
        for i in range(6):
            x = 82 + (i % 3) * 462
            y = 318 + (i // 3) * 224
            draw.rectangle([x, y, x + 372, y + 154], outline="#d99b45", width=4)
            draw.text((x + 18, y + 18), f"PANEL {i + 1}", fill="#f6efe5", font=body_font)
    else:
        draw.ellipse([340, 560, 680, 880], outline="#e89ab6", width=8)
        draw.rectangle([190, 950, 834, 1120], outline="#d99b45", width=6)
        draw.text((235, 985), "PIE TROPHY\nRETURN COUNTER", fill="#f6efe5", font=title_font)
    img.save(path)


def generate_image(prompt, path, size, board=False):
    try:
        from openai import OpenAI

        client = OpenAI()
        result = client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
        path.write_bytes(base64.b64decode(result.data[0].b64_json))
        return "openai_gpt_image_api"
    except Exception as exc:
        path.with_suffix(".generation_error.txt").write_text(
            f"OpenAI image generation failed; blocked placeholder created. Error: {exc}\n",
            encoding="utf-8",
        )
        make_placeholder(
            path,
            "RUN 063 IMAGE BLOCKED",
            "GPT Image generation failed.\nPrompt preserved beside image.\nRerun through GPT Image 2 before public use.",
            board=board,
        )
        return "blocked_placeholder_after_openai_error"


def main():
    for sub in [
        "research", "strategy", "frames/gpt_image_2", "storyboards/shared_choices", "qc",
        "chai", "scene_json", "handoffs", "captions", "distribution", "skool", "manifests",
    ]:
        (PKG / sub).mkdir(parents=True, exist_ok=True)

    prev = RUN_DIR / "PREVIOUS_RUN_CHECK.md"
    prev.write_text(
        "# Previous Run Check\n\nHighest official non-aborted run found before this cycle: `run_062_asset_locker_confessional`.\n\n"
        "Status read: complete still-image package, waiting on human video decision. No next-step report was needed to replace this new research-first cycle.\n",
        encoding="utf-8",
    )

    write_text("research/last30days_report.md", f"""
# Last 30 Days Research Intake - Run {RUN}

Created: {NOW}

Research query/topic: fresh celebrity/public-status contradictions from April 2026 through May 2, 2026 with famous face, public conflict, humiliation or absurd object language, brand/location contrast, and a clean first-frame visual.

Intake method: current web/source scan first, then candidate scoring. No premise was selected from local images, old storyboards, nearby assets, or prior handoffs.

Shortlist:
- Megan Thee Stallion / Klay Thompson split plus Broadway calendar context: high famous-face signal, a clean quoted object hook around "Sweetest Pie", and strong Broadway/NBA/luxury-gift contrast. Selected only as fictional object satire; do not visualize private pain or allegations as proven fact.
- Howard Stern lawsuit-dismissal filing: very fresh and quote-rich ("shakedown", "hush-money"), but rejected for overlap with prior Stern/cat-rescue package and active workplace-allegation sensitivity.
- Paramount/WBD subscriber antitrust suit: fresh, visually workable through popcorn/triple-damages objects, but rejected for low famous-face signal and overlap with prior Paramount merger run.
- Ray J/Kim/Kris arbitration: conflict-rich, but rejected for adult-tape context and overlap with prior Kardashian/Ray J package.
- Sydney Sweeney/Scooter Braun Stagecoach PDA: fresh and visually clean, but too soft on conflict/humiliation gates.

Verified facts used:
- TMZ's current Klay Thompson topic page lists the April 25 breakup item, the April 27 Broadway exit item, and prior Bentley/boat relationship beats.
- TMZ search context describes the April 25 split report and the "Sweetest Pie" headline framing.
- AP confirmed Megan's Broadway role context earlier in April; used only to ground the theater setting.

Unverified or avoided:
- Cheating details are treated as disputed public-relationship context, not as established fact.
- Harassment/death-threat rumor fallout is excluded from the premise.
- No medical incident, crying scene, or exact celebrity likeness is used as the joke target.
""")
    write_json("research/sources.json", {"created_at": NOW, "sources": SOURCES})

    write_json("strategy/candidate_board.json", {"created_at": NOW, "selection_order": "research_first_then_scoring_then_package", "candidates": CANDIDATES})
    write_text("strategy/winner_decision.md", f"""
# Winner Decision - Run {RUN}

Selected premise: **Sweetest Pie Return Counter**.

Logline: A fictional NBA champion archetype tries to return a heart-shaped "Sweetest Pie" trophy, toy boat nameplate, and luxury gift receipt at a Broadway lost-and-found counter where the stage manager stamps everything non-negotiable.

Score summary:
- Winner total: 70
- Runner-up: `stern_hush_money_mansion_hr` at 64
- Main reason: the winner has a stronger first-frame contradiction, avoids duplicate Stern/Paramount/Kardashian lanes, and can stay on object comedy rather than litigated allegations.

Decision: proceed with still-image package and closed-tool handoff only. Video generation was not called.
""")
    write_json("strategy/phase_minus_one_worthiness_audit.json", {
        "source_target": "Megan Thee Stallion / Klay Thompson split as object-satire return-counter setup",
        "track_a_newsjack_velocity": {"active_trend_score": 8, "algorithmic_slipstream": "fresh within one week and still indexed on TMZ topic pages", "polarization_factor": 5, "track_a_total": 13, "verdict": "PASS_WITH_TASTE_GUARDRAILS"},
        "track_b_archetype_resonance": {"iconography_strength": 9, "stereotype_rigidity": "high: athlete, Broadway, luxury gifts, breakup paperwork", "subversion_potential": 10, "track_b_total": 19, "verdict": "PASS"},
        "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "OBJECT_RETURN_COUNTER", "strategic_directive": "Make love-story trophies behave like retail returns without mocking private vulnerability."}
    })
    write_json("strategy/source_entropy_audit.json", {
        "baseline_reality_check": "Public breakup coverage plus Broadway schedule context; object beats include song-title framing, prior Bentley gift, and boat-name relationship lore.",
        "detected_anomalies": ["Broadway glamour colliding with NBA/luxury relationship objects", "romantic symbols treated as accountable inventory", "NON-NEGOTIABLE values turned into a rubber stamp"],
        "native_entropy_score": 6,
        "subject_self_awareness": "mixed",
        "comedic_vector_recommendation": "object-led status reversal",
        "recommended_strategy": "stage-door returns desk"
    })
    write_json("strategy/humor_logic_bridge.json", {
        "setup": "A very public relationship has left behind oversized romantic status objects.",
        "status_reversal": "A tiny stage-manager stamp has more authority than celebrity gifts, arena status, or Broadway lights.",
        "visual_rule": "Everything emotional becomes inventory; everything glamorous has to wait in line.",
        "punchline": "The heart-shaped pie trophy cannot be returned because the values stamp says NON-NEGOTIABLE.",
        "do_not_cross": ["no exact likeness", "no cheating proof scene", "no harassment rumors", "no medical/crying scene", "no sexualized framing"]
    })
    write_json("strategy/tribe_meta_score.json", {"tribe_meta_score": {"tribe_fit": 8, "shareability": 8, "comment_prompt": 8, "caption_lore_potential": 8, "repeatable_franchise_value": 9, "notes": "Reusable breakup-object returns counter format with strong prop comedy."}})
    write_json("strategy/risk_taste_score.json", {"risk_taste_score": {"defamation_risk": 5, "relationship_sensitivity": 6, "identity_or_health_sensitivity": 2, "public_figure_likeness_risk": 5, "brand_logo_risk": 4, "overall_risk": "Medium", "guardrails": ["fictionalize everyone", "avoid proving allegations", "avoid harassment side plot", "no real show logos", "center props and bureaucracy"]}})
    write_text("strategy/franchise_decision.md", "# Franchise Decision\n\nFranchise lane: **Celebrity Object Returns Counter**.\n\nRepeatable template: public relationship or status scandal leaves behind symbolic objects; a mundane returns desk turns them into inventory with one blunt stamp.\n\nVerdict: franchise seed approved with taste guardrails. The format is prop-forward, low-video-complexity, and easy to adapt without cruelty.\n")

    write_text("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write_text("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)
    first_mode = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536", board=False)
    board_mode = generate_image(SHARED_CHOICES_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x1024", board=True)
    blocked = first_mode.startswith("blocked") or board_mode.startswith("blocked")

    shot = {
        "run_id": RUN,
        "shot_id": "shot_001",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "duration_seconds": 6,
        "concept": "The Broadway returns counter treats relationship symbols as inventory.",
        "camera": "counter-height slow push from rubber stamp and pie trophy to embarrassed athlete archetype",
        "motion": "marquee bulbs flicker, receipt curls, stage manager stamps once, athlete lowers gear bag",
        "no_video_generation": True,
        "workflow_spec": {"source_image": "frames/gpt_image_2/first_frame_v01.png", "render_lane": "manual_closed_tool_only_after_human_approval"}
    }
    write_json("scene_json/shot_001.json", shot)
    write_json("scene_json/shot_0001.json", {**shot, "shot_id": "shot_0001"})
    write_json("chai/chai_shot_specs.json", {"run_id": RUN, "specs": [{"shot_id": "shot_001", "subject": "fictional NBA champion archetype and fictional Broadway stage manager", "scene": "Broadway lost-and-found returns counter", "motion": "rubber stamp lands as relationship trophies become inventory", "spatial": "stamp/pie foreground, stage manager counter left, athlete midground, velvet theater background", "camera": "vertical 35mm counter-height push", "critique": "Must not read as mocking a real person's distress or proving infidelity.", "revision": "If too literal, reduce likeness and increase prop abstraction."}]})
    write_json("handoffs/closed_tool_handoff.json", {"run_id": RUN, "title": "Sweetest Pie Return Counter", "video_generation_permitted": False, "approved_still_paths": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}, "next_manual_step": "After human still approval only, use shot specs for a manual closed-tool video handoff. Do not auto-render."})
    write_text("handoffs/grok_agent_prompt.md", "# Grok Agent Prompt\n\nDo not generate video automatically. Use attached stills only as private visual reference.\n\nPlan a 6-second satirical clip where a fictional NBA champion archetype tries to return symbolic relationship props at a Broadway lost-and-found counter. Keep it prop-led and bureaucratic. No exact celebrity likenesses, no harassment rumors, no medical/crying scene, and no assertion that disputed allegations are fact.\n")
    write_text("captions/instagram_caption.md", "Broadway lost-and-found said the Sweetest Pie trophy was final sale.\n\nTrust, respect, and the rubber stamp are non-negotiable.\n\n#sgflix #satire #broadway #nba #popculture #shortfilm #aivideo\n")
    write_text("distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels / TikTok.\n\nHook text: \"POV: the relationship trophies hit the Broadway returns counter.\"\n\nRisk caption note: Keep language to public breakup/object satire. Do not mention unverified affair rumors or harassment.\n\nCutdown plan: 6s stamp reveal, 10s prop inventory, 15s case-study version.\n")
    write_text("skool/case_study.md", "# Skool Case Study - Run 063\n\nLesson: turn a sensitive celebrity breakup into safer object comedy by moving the joke away from pain and toward symbolic inventory.\n\nFactory move: extract public objects from source context, then stage them inside a bureaucratic returns counter. The punchline comes from the stamp, not from personal humiliation.\n")
    write_json("manifests/asset_manifest.json", {"run_id": RUN, "created_at": NOW, "assets": [{"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode}, {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt"}, {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode}, {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt"}]})
    if blocked:
        write_text("qc/IMAGE_GENERATION_BLOCKED_REPORT.md", f"# Image Generation Blocked Report\n\nGPT image generation was attempted after research, scoring, winner selection, and package creation.\n\nResults:\n- `frames/gpt_image_2/first_frame_v01.png` via `{first_mode}`\n- `storyboards/shared_choices/shared_choices_v01.png` via `{board_mode}`\n\nThe saved PNGs are blocked placeholders if generation failed. Do not route to video until the prompts are rerun through GPT Image 2 and QC passes.\n")
    write_text("qc/first_frame_v01_qc.md", f"# First Frame QC\n\nAsset: `frames/gpt_image_2/first_frame_v01.png`\n\nGeneration mode: `{first_mode}`\n\nVerdict: {'blocked placeholder only; rerun prompt through GPT Image 2 before production use' if first_mode.startswith('blocked') else 'usable for human review; inspect face drift/text/logos before video handoff'}.\n\nQC notes: Must avoid exact likeness, harassment rumors, and readable show/brand text.\n")
    write_text("qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nAsset: `storyboards/shared_choices/shared_choices_v01.png`\n\nGeneration mode: `{board_mode}`\n\nVerdict: {'blocked placeholder only; rerun prompt through GPT Image 2 before production use' if board_mode.startswith('blocked') else 'usable as private director-bible reference pending text inspection'}.\n\nQC notes: Board must include character/props/palette/environment/blocking/panels/lighting/rules/production notes.\n")

    status = "blocked_at_gpt_image_generation" if blocked else "complete_for_factory_cycle"
    package = {"run_id": RUN, "slug": SLUG, "title": "Sweetest Pie Return Counter", "status": status, "created_at": NOW, "research_query": "fresh celebrity-status contradictions April 2026-May 2 2026", "selected_premise": CANDIDATES[0]["premise"], "winner_total": CANDIDATES[0]["total"], "video_generation": "not requested and not performed", "still_paths": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}, "next_human_action": "Review still outputs; if blocked, rerun saved prompts through GPT Image 2 and redo QC before any closed-tool video workflow."}
    write_json(f"RUN_{RUN}_MASTER_PACKAGE.json", package)
    write_text("README.md", f"# RUN {RUN} MASTER PACKAGE - Sweetest Pie Return Counter\n\nStatus: {status}; no video footage generated.\n\nResearch query/topic: fresh celebrity-status contradictions April 2026-May 2 2026.\n\nSelected premise: A fictional NBA champion archetype tries to return a heart-shaped Sweetest Pie trophy and relationship props at a Broadway lost-and-found counter.\n\nScore summary: winner 70; runner-up Stern hush-money mansion HR 64; selected for freshness, visual contradiction, and lower duplicate risk.\n\nStill-image paths:\n- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})\n- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})\n\nExact next human action: Review still outputs. If placeholders, rerun the saved prompts through GPT Image 2, replace the PNGs, and redo QC before manual video-tool handoff.\n")
    write_text("FACTORY_RUN_STATUS.md", f"# Factory Run Status - Run {RUN}\n\nStatus: {status}\n\nNew run id: `run_{RUN}_{SLUG}`\n\nResearch query/topic: fresh celebrity-status contradictions April 2026-May 2 2026.\n\nCandidate board: created at `strategy/candidate_board.json`.\n\nSelected premise: `Sweetest Pie Return Counter`.\n\nScore summary: winner 70; runner-up `stern_hush_money_mansion_hr` 64; selected for stronger first-frame contradiction and lower duplicate/taste risk.\n\nCreated files: all required package artifacts were written under `RUN_{RUN}_MASTER_PACKAGE`.\n\nGenerated still-image paths:\n- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})\n- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})\n\nMissing files: none from the required checklist.\n\nPost-ready exports: none; stills require human review and, if blocked, GPT Image 2 repair.\n\nQC failures: {'GPT Image API blocked; placeholder assets only.' if blocked else 'None known; human visual inspection still required.'}\n\nHigh-risk issues: public-figure likeness drift, accidentally proving disputed relationship claims, harassment-rumor contamination, readable real logos/show names.\n\nExact next human action: Review still outputs; if blocked, rerun `first_frame_v01_prompt.md` and `shared_choices_v01_prompt.md` through GPT Image 2 and redo QC.\n")


if __name__ == "__main__":
    main()
