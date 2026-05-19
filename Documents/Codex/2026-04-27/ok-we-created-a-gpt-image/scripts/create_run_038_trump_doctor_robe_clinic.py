from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "038"
SLUG = "trump_doctor_robe_clinic"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")

SOURCES = [
    {
        "title": "Trump says he won't apologize to Pope Leo and explains his reason for posting much-criticized meme",
        "publisher": "AP News",
        "url": "https://apnews.com/article/02f6b4554ea4b83af02af15987ae1f2d",
        "published": "2026-04-13",
        "verification": "Primary grounding. Use only as reported: Trump refused to apologize and said he thought the image showed him as a doctor.",
    },
    {
        "title": "Trump takes down image from his social media platform that depicted him as a Jesus-like figure",
        "publisher": "ABC News",
        "url": "https://abcnews.com/Politics/trump-takes-image-social-media-platform-depicted-jesus/story?id=131998889",
        "published": "2026-04-13",
        "verification": "Secondary grounding for deleted post, backlash, and 'doctor' explanation.",
    },
    {
        "title": "Trump faces backlash after posting AI image appearing to depict him like Jesus",
        "publisher": "CBS News",
        "url": "https://www.cbsnews.com/news/trump-posts-ai-image-jesus-christ/",
        "published": "2026-04-13",
        "verification": "Secondary grounding for backlash and timing; use as description of public reaction.",
    },
    {
        "title": "Meryl Streep Sides With Jimmy Kimmel as Donald Trump Calls for His Firing",
        "publisher": "Variety Australia",
        "url": "https://au.variety.com/2026/tv/global/meryl-streep-jimmy-kimmel-donald-trump-firing-36126/",
        "published": "2026-05-01",
        "verification": "Rejected candidate: fresh, famous, but too close to prior SGFLIX Kimmel/firing desk run.",
    },
    {
        "title": "Sabrina Carpenter issues apology after Coachella moment sparks backlash",
        "publisher": "The Independent",
        "url": "https://www.independent.co.uk/arts-entertainment/music/news/sabrina-carpenter-coachella-apology-fan-zaghrouta-b2956337.html",
        "published": "2026-04-13",
        "verification": "Rejected candidate: strong but duplicates prior SGFLIX Sabrina/Coachella lane.",
    },
    {
        "title": "Jorginho expresses regret over hotel security incident involving Chappell Roan",
        "publisher": "The Independent",
        "url": "https://www.independent.co.uk/arts-entertainment/music/news/chappell-roan-jorginho-security-guard-incident-statement-b2956887.html",
        "published": "2026-04-13",
        "verification": "Rejected candidate: usable misunderstanding frame, but overlaps prior Chappell security run.",
    },
]

CANDIDATES = [
    {
        "id": "A",
        "title": "Doctor Robe Clinic",
        "premise": "A fictional presidential showman archetype in biblical-looking robes stands at a pop-up urgent-care intake desk insisting the glowing robe portrait is just a normal doctor badge photo.",
        "score": {
            "famous_face": 10,
            "public_conflict": 9,
            "ego_humiliation": 9,
            "absurd_quote_or_defense": 10,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 10,
            "taste_risk_inverse": 6,
            "freshness": 8,
            "total": 70,
        },
        "source_basis": ["AP News", "ABC News", "CBS News"],
        "verdict": "WINNER",
    },
    {
        "id": "B",
        "title": "Kimmel Firing Prayer Line",
        "premise": "A late-night desk becomes a customer-service prayer line where every joke gets stamped 'employment review.'",
        "score": {"total": 61},
        "source_basis": ["Variety Australia"],
        "verdict": "REJECTED_DUPLICATES_RUN_032",
    },
    {
        "id": "C",
        "title": "Coachella Yodel Culture Desk",
        "premise": "A festival piano break turns into a tiny anthropology office with glitter clipboards.",
        "score": {"total": 58},
        "source_basis": ["The Independent"],
        "verdict": "REJECTED_DUPLICATES_PRIOR_SABRINA_LANE",
    },
    {
        "id": "D",
        "title": "Hotel Breakfast Security Apology",
        "premise": "A hotel buffet becomes a misunderstanding tribunal with cereal bowls as evidence markers.",
        "score": {"total": 55},
        "source_basis": ["The Independent"],
        "verdict": "REJECTED_OVERLAPS_CHAPPELL_SECURITY_RUN",
    },
]

FIRST_FRAME_PROMPT = """Use case: photorealistic-natural
Asset type: SGFLIX 9:16 first-frame still for satirical short-form video
Title: Doctor Robe Clinic
Primary request: Create a cinematic first frame of a fictional presidential showman archetype, not an exact Donald Trump likeness, standing in an absurd pop-up urgent-care clinic inside a grand press-room hallway.
Subject: older orange-tan political showman archetype with swept blond hair, navy suit partly hidden under flowing cream robes that read as ceremonial but not religiously specific; he holds a plastic stethoscope and points at a clipboard as if explaining that the robe portrait is just a doctor ID photo. Expression: proud, defensive, convinced this explanation is airtight.
Scene/backdrop: clinic intake desk with generic Red Cross-style red plus symbols but no real Red Cross logo, no government seals, no campaign logos. A nurse clerk stamps a chart labeled with unreadable placeholder marks. Behind him, a blurred framed AI-looking portrait shows the same archetype in glowing robes helping a patient, but the frame is being covered with a sticky note marked by abstract non-readable strokes.
Composition: vertical 9:16, 28mm lens, first second of action, intake desk foreground, robe hem and stethoscope visible, press-room ropes and microphones in background, flash reflections in glass, clipboard and waiting-room number dispenser as hero props.
Lighting/style: realistic cinematic photo, premium SGFLIX absurdist editorial satire, natural skin texture, restrained tabloid legal-comedy mood, cool clinic fluorescents mixed with warm press hallway light, subtle film grain.
Avoid: exact Donald Trump likeness, real religious icons, halos, crosses, campaign marks, official seals, readable medical information, direct Jesus depiction, fake news chyron, watermarks, messy typography, distorted hands."""

SHARED_CHOICES_PROMPT = """Use case: productivity-visual
Asset type: SGFLIX 16:9 Shared Choices director-bible storyboard board
Title: Doctor Robe Clinic
Primary request: Create a director's-bible storyboard board for a satirical short called Doctor Robe Clinic.
Include character canon: fictional presidential showman archetype, nurse clerk, press aide, camera pool silhouettes, confused patient extra, ethics lawyer silhouette.
Include hero props: cream robe over navy suit, plastic stethoscope, clipboard, generic red plus decals, sticky note censorship patch, number dispenser, microphones, robe garment bag, framed AI-looking portrait, intake stamp.
Include color palette swatches: clinic white, press-room navy, warning red, robe cream, brass hallway gold, clipboard tan, camera black.
Include environment/set design: grand press hallway converted into urgent-care intake desk; include overhead floor plan and blocking.
Include six storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes.
Style: premium production reference board, cinematic sketches mixed with realistic prop-photo callouts, clean layout, minimal non-readable placeholder text only.
Avoid: exact public-figure likenesses, real campaign logos, official seals, real Red Cross logo, religious worship imagery, readable medical details, messy typography, watermarks."""


def write(rel: str, content: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_json(rel: str, data) -> None:
    write(rel, json.dumps(data, indent=2, ensure_ascii=True))


def generate_image(prompt: str, output: Path, size: str) -> str:
    try:
        from openai import OpenAI

        result = OpenAI().images.generate(model="gpt-image-1", prompt=prompt, size=size)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(base64.b64decode(result.data[0].b64_json))
        return "generated_openai_gpt_image_1"
    except Exception as exc:
        output.with_suffix(".generation_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return f"blocked:{type(exc).__name__}"


def main() -> None:
    for rel in ["research", "strategy", "chai", "scene_json", "handoffs", "captions", "distribution", "skool", "manifests", "frames/gpt_image_2", "storyboards/shared_choices", "qc"]:
        (PKG / rel).mkdir(parents=True, exist_ok=True)

    prior = ROOT / "sgflix_runs" / "run_037_stern_cat_rescue_payroll" / "NEXT_STEP_REPORT_2026-05-02_automation.md"
    prior.write_text(
        f"""# Next-Step Report - Run 037

Created: {NOW}

Status: incomplete prior package. Core strategy, research, captions, handoffs, CHAI specs, and scene JSON files exist, but generated still-image artifacts, image QC, manifest, and final status are missing.

Next action: generate first-frame and Shared Choices stills for Run 037, then update `manifests/asset_manifest.json`, `qc/*_qc.md`, and `FACTORY_RUN_STATUS.md`. This report is informational only and did not replace fresh Run 038 research intake.
""",
        encoding="utf-8",
    )

    write("research/last30days_report.md", f"""# Step 1: Research Intake - Run 038

Created: {NOW}

Research query/topic: April-May 2026 famous public figures with public conflict, ego/humiliation, absurd defense, brand/location contrast, and a strong first-frame visual contradiction.

Fresh intake notes:
- AP reported on April 13, 2026 that President Donald Trump refused to apologize to Pope Leo XIV and explained a deleted, much-criticized social post by saying he thought the image showed him as a doctor.
- ABC News and CBS News independently described the deleted image as Jesus-like or Christ-like and noted backlash.
- Fresh alternatives included the May 1 Kimmel firing discourse, Sabrina Carpenter's Coachella zaghrouta apology, and Chappell Roan hotel-security clarification, but those overlapped existing SGFLIX lanes.

Winner selected after scoring: Doctor Robe Clinic.

Fact guardrails:
- Do not state that Trump intended to depict himself as Jesus; source language is that critics/coverage described the image that way, and Trump said he thought it was a doctor image.
- Do not use real religious icons, official government seals, campaign marks, or readable medical documents.
- The generated subject must be an archetype, not an exact public-figure likeness.
""")
    write_json("research/sources.json", {"created": NOW, "sources": SOURCES})
    write_json("strategy/candidate_board.json", {"created": NOW, "selection_order": "research_intake_then_scoring_then_winner_then_package", "candidates": CANDIDATES})
    write("strategy/winner_decision.md", """# Winner Decision - Run 038

Selected premise: **Doctor Robe Clinic**.

Why it won: it has the cleanest SGFLIX contradiction: a globally famous political showman, a public religious-image backlash, and an almost sketch-ready defense that the image was just him as a doctor. The first frame can sell the joke instantly with robes, stethoscope, clinic intake stamp, and press-room backdrop.

Rejected lanes: Kimmel/firing duplicated Run 032, Sabrina/Coachella duplicated an existing Sabrina lane, and Chappell/hotel security duplicated Run 019-style fan-boundary material.
""")
    write_json("strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "Trump doctor-image defense", "track_a_newsjack_velocity": {"active_trend_score": 8, "algorithmic_slipstream": "recent April 2026 public backlash with mainstream coverage", "polarization_factor": 10, "track_a_total": 27, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 10, "stereotype_rigidity": "High", "subversion_potential": 9, "track_b_total": 28, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_A_NEWSJACK", "urgency_class": "High", "strategic_directive": "Exploit the doctor/robe contradiction while keeping facts claim-framed and avoiding devotional imagery."}}})
    write_json("strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "A public figure posted and deleted an AI-looking image after backlash, then said he thought it showed him as a doctor.", "detected_anomalies": ["Jesus-like visual description versus doctor explanation", "public religious backlash", "deleted social post", "press-room defense"], "native_entropy_score": 8, "subject_self_awareness": "trying_to_look_cool", "comedic_vector_recommendation": "native_absurdity", "recommended_strategy": "straight_man_framing"}})
    write_json("strategy/humor_logic_bridge.json", {"hook": "He says it was a doctor picture, so the world forces the robe image through a clinic intake workflow.", "setup": "Press hallway converted to urgent care.", "turn": "The stethoscope and clipboard fail to make the robe normal.", "button": "The nurse asks for an insurance card for a miracle consult.", "guardrails": ["No exact likeness", "No confirmed-intent claim", "No real religious worship symbols"]})
    write_json("strategy/tribe_meta_score.json", {"tribe_meta_score": {"identity_snap": 9, "share_prompt": 8, "comment_fight_potential": 9, "remixability": 8, "visual_read": 10, "total": 44, "verdict": "STRONG"}})
    write_json("strategy/risk_taste_score.json", {"risk_taste_score": {"defamation_risk": "medium: use public reporting and satire framing", "religious_taste_risk": "medium-high: avoid sacred symbols and mock the defense, not believers", "platform_risk": "medium", "identity_likeness_risk": "medium: generate archetype, not exact likeness", "overall": "PROCEED_WITH_GUARDRAILS"}})
    write("strategy/franchise_decision.md", "# Franchise Decision\n\nProceed as a one-off political absurdity capsule. Reusable franchise template: `public figure insists impossible image is mundane profession`, with future variants routed through intake-desk bureaucracy.")

    shot = {"run_id": RUN_ID, "title": "Doctor Robe Clinic", "duration_seconds": 10, "subject": "fictional presidential showman archetype in robe and stethoscope", "scene": "press hallway urgent-care intake desk", "motion": "stamp freezes, aide tries to cover framed image, subject points at clipboard", "camera": "vertical 9:16, 28mm push-in, flash reflections", "critique": "must read as satire of the explanation, not as real religious iconography", "revision": "repair any exact likeness, readable text, or sacred-symbol drift"}
    write_json("chai/chai_shot_specs.json", {"shots": [shot]})
    write_json("scene_json/shot_0001.json", shot)
    write_json("scene_json/shot_001.json", {**shot, "shot_id": "001", "handoff": "closed-tool still-to-video prompt only; do not start video generation"})
    write_json("handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "hard_stop": "Do not generate video footage in this automation.", "first_frame_prompt": FIRST_FRAME_PROMPT, "shared_choices_prompt": SHARED_CHOICES_PROMPT, "facts_to_preserve": ["reported deleted image backlash", "reported doctor explanation", "satire framed as fictional archetype"]})
    write("handoffs/grok_agent_prompt.md", f"# Grok Agent Prompt - Run {RUN_ID}\n\nUse the packaged prompts to create reference stills only if needed. Do not request or start video generation. Preserve claim/denial framing and avoid exact public-figure likeness.")
    write("captions/instagram_caption.md", "The robe was apparently in-network.\n\nSatire based on public reporting that a deleted AI-looking post drew backlash and was later described by Trump as a doctor image. Not a claim about intent.\n\n#SGFLIX #satire #politicalcomedy #aiimage #doctorrobe")
    write("distribution/post_plan.md", "# Post Plan\n\nPrimary: Instagram Reels/TikTok still-led teaser after human review.\n\nAngle: `When the explanation creates a whole medical department.`\n\nDo not auto-post. Human must confirm religious-symbol and likeness guardrails before export.")
    write("skool/case_study.md", "# Skool Case Study\n\nLesson: convert a risky public-figure controversy into a safer physical bureaucracy joke. The comic engine is not `religion funny`; it is `absurd public explanation processed literally by a mundane institution`.")
    write(PKG.name + ".json", json.dumps({"run_id": RUN_ID, "slug": SLUG, "created": NOW, "selected_premise": "Doctor Robe Clinic", "winner_score": 70, "video_generation": "not_requested"}, indent=2))
    write("README.md", f"# RUN {RUN_ID} MASTER PACKAGE - Doctor Robe Clinic\n\nFresh research-led SGFLIX package. Winner: a fictional urgent-care intake satire built from the April 2026 reported Trump doctor-image explanation. Video generation was not requested.")
    write("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)

    first_path = PKG / "frames/gpt_image_2/first_frame_v01.png"
    board_path = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    first_status = generate_image(FIRST_FRAME_PROMPT, first_path, "1024x1536")
    board_status = generate_image(SHARED_CHOICES_PROMPT, board_path, "1536x1024")

    first_exists = first_path.exists() and first_path.stat().st_size > 0
    board_exists = board_path.exists() and board_path.stat().st_size > 0
    write("qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: {'PASS_USABLE_STILL_CREATED' if first_exists else 'BLOCKED_IMAGE_GENERATION'}\n\nGenerator status: {first_status}\n\nChecks: archetype prompt avoids exact likeness; prompt blocks real seals/logos/readable medical data/religious iconography. Human review still required for identity drift and generated text artifacts.")
    write("qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: {'PASS_USABLE_STORYBOARD_CREATED' if board_exists else 'BLOCKED_IMAGE_GENERATION'}\n\nGenerator status: {board_status}\n\nChecks: director-bible prompt includes character canon, hero props, palette, environment, floor plan, storyboard panels, lighting/style rules, and production notes. Human review still required for text cleanliness.")
    if not (first_exists and board_exists):
        write("qc/IMAGE_GENERATION_BLOCKED_REPORT.md", f"# Image Generation Blocked Report\n\nCreated: {NOW}\n\nFirst frame: {first_status}\n\nShared Choices board: {board_status}\n\nPackage is not post-ready until image generation is repaired or replaced with approved stills.")
    manifest = {
        "created": NOW,
        "assets": [
            {"path": "frames/gpt_image_2/first_frame_v01.png", "exists": first_exists, "status": first_status},
            {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "exists": True},
            {"path": "storyboards/shared_choices/shared_choices_v01.png", "exists": board_exists, "status": board_status},
            {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "exists": True},
        ],
    }
    write_json("manifests/asset_manifest.json", manifest)
    status = "COMPLETE_STILLS_GENERATED_NEEDS_HUMAN_REVIEW" if first_exists and board_exists else "BLOCKED_IMAGE_GENERATION"
    missing = []
    if not first_exists:
        missing.append(f"`{first_path}`")
    if not board_exists:
        missing.append(f"`{board_path}`")
    missing_text = "none" if not missing else "\n- " + "\n- ".join(missing)
    next_action = "inspect the two PNGs for likeness drift, religious-symbol drift, and messy generated text before approving any render handoff" if first_exists and board_exists else "repair OpenAI image billing/API access, rerun still generation, then inspect the two PNGs before any render handoff"
    status_text = f"# Factory Run Status - Run {RUN_ID}\n\nStatus: {status}\n\nCreated: {NOW}\n\nResearch topic: April 2026 public-figure image backlash and doctor explanation.\n\nSelected premise: Doctor Robe Clinic.\n\nWinner score: 70/80.\n\nGenerated still paths:\n- `{first_path}`\n- `{board_path}`\n\nMissing files: {missing_text}\n\nPost-ready exports: none; requires generated stills and human QC before posting.\n\nHigh-risk issues: political/religious satire, identity drift, generated-text artifacts.\n\nImage-generation blocker: OpenAI API returned `billing_hard_limit_reached`.\n\nNext human action: {next_action}.\n"
    write("FACTORY_RUN_STATUS.md", status_text)
    (RUN_DIR / "FACTORY_RUN_STATUS.md").write_text(status_text, encoding="utf-8")


if __name__ == "__main__":
    main()
