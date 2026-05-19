from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "040"
SLUG = "deb_24h_story_evidence"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


SOURCES = [
    {
        "title": "Rebel Wilson says social media posts are true during defamation trial cross-examination",
        "publisher": "ABC News Australia",
        "url": "https://www.abc.net.au/news/2026-04-30/rebel-wilson-court-evidence-defamation-trial-the-deb/106623682",
        "published": "2026-04-30",
        "verification": "Primary winner grounding: Wilson gave evidence in a Sydney Federal Court defamation trial concerning social media posts about The Deb lead actor Charlotte MacInnes. Use claim/denial framing only.",
    },
    {
        "title": "Rebel Wilson tight-lipped arriving at Sydney court ahead of giving evidence in defamation trial",
        "publisher": "ABC News Australia",
        "url": "https://www.abc.net.au/news/2026-04-28/rebel-wilson-gives-evidence-in-court-the-deb-defamation-case/106614508",
        "published": "2026-04-28",
        "verification": "Timeline and courtroom context. Confirms the Federal Court setting and The Deb connection.",
    },
    {
        "title": "Rebel Wilson tells defamation trial she was not behind websites that attacked producer",
        "publisher": "The Guardian",
        "url": "https://www.theguardian.com/film/2026/apr/28/rebel-wilson-gives-evidence-defamation-case-ntwnfb",
        "published": "2026-04-28",
        "verification": "Secondary context for broader online-evidence/campaign allegations. Avoid treating disputed claims as proven.",
    },
    {
        "title": "Rebel Wilson Faces Federal Court in The Deb Defamation Case",
        "publisher": "Variety Australia",
        "url": "https://au.variety.com/2026/film/news/rebel-wilson-the-deb-defamation-dispute-sydney-35659/",
        "published": "2026-04-20",
        "verification": "Entertainment-industry framing and rejected/accepted details: nine-day Sydney hearing, social posts, Australian release timing.",
    },
    {
        "title": "Consumers sue to block Paramount-Warner Bros. deal",
        "publisher": "Los Angeles Times",
        "url": "https://www.latimes.com/entertainment-arts/business/story/2026-05-01/consumers-sue-to-block-paramount-warner-bros-deal",
        "published": "2026-05-01",
        "verification": "Rejected candidate: fresh but overlaps prior SGFLIX merger/picket lanes.",
    },
    {
        "title": "Taylor Swift Files to Trademark Her Voice and Likeness, Apparently to Protect Against AI Misuse",
        "publisher": "Variety Australia",
        "url": "https://au.variety.com/2026/music/news/taylor-swift-trademark-voice-likeness-ai-misuse-35964/",
        "published": "2026-04-28",
        "verification": "Rejected candidate: famous and visual, but duplicates prior voice-vault lane.",
    },
]


CANDIDATES = [
    {
        "id": "A",
        "title": "Deb 24-Hour Story Evidence",
        "premise": "A fictional Australian comedy-star/director archetype stands in a Sydney courthouse prop room where vanishing social stories are sealed in evidence bags before they expire.",
        "score": {
            "famous_face": 7,
            "public_conflict": 8,
            "ego_humiliation": 8,
            "absurd_quote_or_defense": 7,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 9,
            "taste_risk_inverse": 6,
            "freshness": 9,
            "total": 62,
        },
        "source_basis": ["ABC News Australia", "The Guardian", "Variety Australia"],
        "verdict": "WINNER",
        "notes": "Best non-duplicate: current, famous enough, visually specific, and can stay focused on courtroom handling of social media evidence rather than disputed underlying allegations.",
    },
    {
        "id": "B",
        "title": "Paramount Consumer Receipt Maze",
        "premise": "A movie-studio gate turns into a grocery checkout where every streaming bundle prints another antitrust receipt.",
        "score": {
            "famous_face": 4,
            "public_conflict": 9,
            "ego_humiliation": 6,
            "absurd_quote_or_defense": 6,
            "brand_location_contrast": 9,
            "first_frame_visual_contradiction": 8,
            "taste_risk_inverse": 8,
            "freshness": 10,
            "total": 60,
        },
        "source_basis": ["Los Angeles Times"],
        "verdict": "REJECTED_DUPLICATES_MERGER_LANE",
        "notes": "Fresh May 1 lawsuit, but prior runs already used Paramount/WBD merger protest mechanics.",
    },
    {
        "id": "C",
        "title": "Taylor Voice Trademark Booth",
        "premise": "A pop-star archetype records a greeting inside a vault while trademark clerks tag the sound waves like couture.",
        "score": {
            "famous_face": 10,
            "public_conflict": 6,
            "ego_humiliation": 5,
            "absurd_quote_or_defense": 8,
            "brand_location_contrast": 7,
            "first_frame_visual_contradiction": 8,
            "taste_risk_inverse": 7,
            "freshness": 8,
            "total": 59,
        },
        "source_basis": ["Variety Australia"],
        "verdict": "REJECTED_DUPLICATES_RUN_015",
        "notes": "Too close to existing Taylor voice-vault run.",
    },
    {
        "id": "D",
        "title": "BAFTA Duty-of-Care Switchboard",
        "premise": "An awards-show control room becomes a duty-of-care emergency switchboard after a broadcast review.",
        "score": {
            "famous_face": 4,
            "public_conflict": 8,
            "ego_humiliation": 6,
            "absurd_quote_or_defense": 5,
            "brand_location_contrast": 7,
            "first_frame_visual_contradiction": 7,
            "taste_risk_inverse": 3,
            "freshness": 7,
            "total": 47,
        },
        "source_basis": ["Variety Australia"],
        "verdict": "REJECTED_TASTE_RISK",
        "notes": "The core incident involves a racial slur and medical/disability context; poor SGFLIX comedy target.",
    },
]


FIRST_FRAME_PROMPT = """Use case: photorealistic-natural
Asset type: SGFLIX 9:16 first-frame still for satirical short-form video
Title: Deb 24-Hour Story Evidence
Primary request: Create a cinematic first frame inside a Sydney courthouse evidence room that has been absurdly converted into a musical-theater prop check.
Subject: fictional Australian comedy-star/director archetype, not an exact Rebel Wilson likeness, blonde hair in a polished court-day style, black blazer over a bright musical-theater rehearsal shirt, holding a phone with only abstract non-readable social-story blocks on screen. Expression: tight polite smile, trying to look calm while realizing the phone posts are now evidence props.
Scene/backdrop: Federal Court-style evidence counter, theatrical costume racks from a fictional musical called "The Deb" with no real title logo, sealed clear evidence bags containing printed screenshots represented by blurred rectangles, a giant hourglass labeled only with abstract placeholder marks, stage lights packed beside court binders, Sydney sandstone courthouse hints in the background.
Composition: vertical 9:16, 28mm lens, first second of action, evidence bag foreground, phone and hourglass as hero props, clerk's gloved hand sliding a vanishing-story printout into a bag, director archetype centered, costume rack and courtroom door behind, paparazzi flash reflection in glass.
Lighting/style: realistic cinematic photo, premium SGFLIX absurdist editorial satire, natural skin texture, restrained legal-comedy mood, cool court fluorescents mixed with warm stage practicals, subtle film grain.
Avoid: exact Rebel Wilson likeness, exact Charlotte MacInnes likeness, real court seals, real movie logos, readable social posts, sexual-harassment reenactments, nude-photo references, victim mockery, fake verdicts, watermarks, messy typography, distorted hands."""


SHARED_CHOICES_PROMPT = """Use case: productivity-visual
Asset type: SGFLIX 16:9 Shared Choices director-bible storyboard board
Title: Deb 24-Hour Story Evidence
Primary request: Create a director's-bible storyboard board for a satirical short about disappearing social posts becoming courthouse evidence props.
Include character canon: fictional Australian comedy-star/director archetype, Federal Court evidence clerk, theater stage manager, lead-actor silhouette, crisis PR assistant, paparazzi reflections.
Include hero props: smartphone with non-readable story blocks, sealed evidence bags, giant hourglass, musical costume rack, court binders, stage lights, blank screenshot printouts, gavel-shaped prop mallet, sticky evidence tabs.
Include color palette swatches: courthouse sandstone, evidence-bag clear plastic, stage-light amber, legal navy, phone-screen blue, stamp red, costume-rack chrome.
Include environment/set design: Sydney courthouse evidence room crossed with musical-theater prop storage; include floor plan/blocking.
Include six storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes.
Style: premium production reference board, cinematic sketches mixed with realistic prop-photo callouts, clean layout, minimal non-readable placeholder text only.
Avoid: exact public-figure likenesses, real court seals, real movie logos, readable allegations, sexual-harassment reenactments, nude-photo references, messy typography, watermarks."""


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
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".generation_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return f"blocked:{type(exc).__name__}"


def main() -> None:
    prior = ROOT / "sgflix_runs" / "run_039_hilton_known_danger_foyer" / "NEXT_STEP_REPORT_2026-05-02_automation.md"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text(
        f"""# Next-Step Report - Run 039

Created: {NOW}

Status: incomplete prior run folder. It contains generated still PNGs but no `RUN_039_MASTER_PACKAGE`, research intake, strategy files, captions, handoffs, manifest, or status file.

Next action: either wrap the existing stills in a complete Run 039 package with research provenance, or mark Run 039 aborted before relying on it as an official completed package. This report did not replace the fresh Run 040 research intake.
""",
        encoding="utf-8",
    )

    for rel in ["research", "strategy", "chai", "scene_json", "handoffs", "captions", "distribution", "skool", "manifests", "frames/gpt_image_2", "storyboards/shared_choices", "qc"]:
        (PKG / rel).mkdir(parents=True, exist_ok=True)

    write("research/last30days_report.md", f"""# Step 1: Research Intake - Run 040

Created: {NOW}

Research query/topic: late-April/early-May 2026 entertainment disputes with public conflict, famous-face recognition, absurd physicalization potential, and low-enough taste risk for SGFLIX.

Fresh intake notes:
- ABC News Australia reported April 30, 2026 that Rebel Wilson told a Sydney court her social media posts at issue in a defamation trial were true.
- ABC News Australia reported April 28, 2026 that Wilson arrived at Sydney Federal Court to give evidence in the case tied to her directorial debut, The Deb.
- The Guardian reported April 28, 2026 on Wilson's evidence about websites allegedly attacking a producer; all related accusations remain disputed and must be claim-framed.
- Variety Australia reported April 20, 2026 that the defamation dispute headed to Federal Court in Sydney for a nine-day hearing.

Candidate board was scored before winner selection. The selected premise is **Deb 24-Hour Story Evidence**.

Fact guardrails:
- Do not treat any disputed allegation as proven.
- Do not reenact the underlying harassment or image-leak accusations.
- Do not make a fake verdict.
- The comedy target is the absurdity of temporary social posts becoming permanent courtroom props.
""")
    write_json("research/sources.json", {"created": NOW, "sources": SOURCES})

    write_json("strategy/candidate_board.json", {"created": NOW, "selection_order": "research_intake_then_scoring_then_winner_then_package", "candidates": CANDIDATES})
    write("strategy/winner_decision.md", """# Winner Decision - Run 040

Selected premise: **Deb 24-Hour Story Evidence**.

Why it won: it is current, non-duplicate, and has a clean physical joke: temporary social posts, normally gone in 24 hours, are being handled like permanent courtroom evidence inside a musical-theater prop room. The first frame can communicate the contradiction instantly with a phone, hourglass, evidence bags, court binders, and costume racks.

Rejected lanes:
- Paramount consumer antitrust receipt: fresh, but too close to prior merger/picket packages.
- Taylor voice trademark booth: too close to Run 015 voice-vault material.
- BAFTA duty-of-care switchboard: weak famous-face hook and high taste risk.
""")
    write_json("strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "Rebel Wilson/The Deb defamation trial social-post evidence", "track_a_newsjack_velocity": {"active_trend_score": 8, "mainstream_recent_coverage": 9, "duplicate_penalty": 0, "track_a_total": 25, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"archetype": "celebrity director tries to control the narrative, then the narrative enters evidence intake", "subversion_potential": 8, "first_frame_native_absurdity": 9, "track_b_total": 24, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_A_NEWSJACK", "strategic_directive": "Keep the joke on temporary-post permanence and court bureaucracy; claim-frame all case facts."}}})
    write_json("strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "A current Sydney defamation trial concerns social media posts about The Deb and disputed behind-the-scenes claims.", "detected_anomalies": ["ephemeral posts preserved as litigation evidence", "musical-theater brand colliding with Federal Court seriousness", "celebrity-director narrative control becoming cross-examination material"], "native_entropy_score": 7, "subject_self_awareness": "trying_to_control_story", "recommended_strategy": "bureaucracy_literalization"}})
    write_json("strategy/humor_logic_bridge.json", {"hook": "The 24-hour story did not disappear; it got a chain-of-custody tag.", "setup": "Court evidence room shares space with musical props.", "turn": "Every vanished post is printed, bagged, and put on a costume rack.", "button": "The clerk asks whether the story expires before or after cross-examination.", "guardrails": ["No verdict claims", "No exact likeness", "No reenactment of disputed sensitive allegations"]})
    write_json("strategy/tribe_meta_score.json", {"tribe_meta_score": {"identity_snap": 7, "share_prompt": 8, "comment_fight_potential": 8, "remixability": 7, "visual_read": 9, "total": 39, "verdict": "GOOD"}})
    write_json("strategy/risk_taste_score.json", {"risk_taste_score": {"defamation_risk": "medium-high: use public reports and claim framing only", "sensitive_allegation_risk": "high if mishandled; avoid underlying allegation content entirely", "platform_risk": "medium", "identity_likeness_risk": "medium: generate archetype, not exact likeness", "overall": "PROCEED_WITH_STRICT_GUARDRAILS"}})
    write("strategy/franchise_decision.md", "# Franchise Decision\n\nProceed as a one-off legal-media satire capsule. Reusable franchise shape: `ephemeral platform behavior forced into old-world bureaucracy`.")

    shot = {
        "run_id": RUN_ID,
        "title": "Deb 24-Hour Story Evidence",
        "duration_seconds": 10,
        "subject": "fictional Australian comedy-star/director archetype with phone, evidence bags, and courthouse prop-room set",
        "scene": "Sydney courthouse evidence room crossed with musical-theater prop storage",
        "motion": "clerk seals a disappearing-story printout, hourglass flips, stage manager wheels in costume rack",
        "spatial": "phone foreground, evidence counter midground, costume rack and courtroom door background",
        "camera": "vertical 9:16, 28mm push-in, flash reflections in glass",
        "critique": "must read as satire of social-post permanence, not a claim about disputed facts",
        "revision": "repair exact likeness, readable allegations, real court marks, or sensitive-allegation drift",
    }
    write_json("chai/chai_shot_specs.json", {"shots": [shot]})
    write_json("scene_json/shot_0001.json", shot)
    write_json("scene_json/shot_001.json", {**shot, "shot_id": "001", "handoff": "closed-tool still-to-video prompt only; do not start video generation"})
    write_json("handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "hard_stop": "Do not generate video footage in this automation.", "first_frame_prompt": FIRST_FRAME_PROMPT, "shared_choices_prompt": SHARED_CHOICES_PROMPT, "facts_to_preserve": ["Sydney defamation trial", "The Deb social-media-post dispute", "all sensitive allegations remain disputed and claim-framed"]})
    write("handoffs/grok_agent_prompt.md", f"# Grok Agent Prompt - Run {RUN_ID}\n\nUse the packaged prompts to create or repair reference stills only. Do not request or start video generation. Preserve claim/denial framing and avoid exact public-figure likenesses, real court seals, readable allegations, or sensitive reenactments.")
    write("captions/instagram_caption.md", "A 24-hour story walks into court and leaves with an evidence tag.\n\nSatire based on public reporting about a current Sydney defamation trial involving social media posts tied to The Deb. No verdict implied; disputed claims remain disputed.\n\n#SGFLIX #satire #entertainmentlaw #socialmedia #courtroomcomedy")
    write("distribution/post_plan.md", "# Post Plan\n\nPrimary: Instagram Reels/TikTok still-led teaser after human review.\n\nAngle: `The story expired, but the exhibit did not.`\n\nDo not auto-post. Human must confirm no readable allegations, no exact likeness, and no sensitive-claim reenactment before export.")
    write("skool/case_study.md", "# Skool Case Study\n\nLesson: risky celebrity litigation can become usable SGFLIX only when the joke moves away from the alleged conduct and into a safe physical contradiction. Here, ephemeral social UX becomes permanent evidence bureaucracy.")
    write(PKG.name + ".json", json.dumps({"run_id": RUN_ID, "slug": SLUG, "created": NOW, "selected_premise": "Deb 24-Hour Story Evidence", "winner_score": 62, "video_generation": "not_requested"}, indent=2))
    write("README.md", f"# RUN {RUN_ID} MASTER PACKAGE - Deb 24-Hour Story Evidence\n\nFresh research-led SGFLIX package. Winner: a Sydney courthouse/musical-prop satire built from current public reporting about The Deb defamation trial. Video generation was not requested.")
    write("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)

    first_path = PKG / "frames/gpt_image_2/first_frame_v01.png"
    board_path = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    first_status = generate_image(FIRST_FRAME_PROMPT, first_path, "1024x1536")
    board_status = generate_image(SHARED_CHOICES_PROMPT, board_path, "1536x1024")
    first_exists = first_path.exists() and first_path.stat().st_size > 0
    board_exists = board_path.exists() and board_path.stat().st_size > 0

    write("qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: {'PASS_USABLE_STILL_CREATED' if first_exists else 'BLOCKED_IMAGE_GENERATION'}\n\nGenerator status: {first_status}\n\nChecks: prompt avoids exact likeness, real court seals, readable allegations, sensitive reenactments, and fake verdicts. Human review still required for identity drift and generated text artifacts.")
    write("qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: {'PASS_USABLE_STORYBOARD_CREATED' if board_exists else 'BLOCKED_IMAGE_GENERATION'}\n\nGenerator status: {board_status}\n\nChecks: director-bible prompt includes character canon, hero props, palette, environment, floor plan, storyboard panels, lighting/style rules, and production notes. Human review still required for text cleanliness.")
    if not (first_exists and board_exists):
        write("qc/IMAGE_GENERATION_BLOCKED_REPORT.md", f"# Image Generation Blocked Report\n\nCreated: {NOW}\n\nFirst frame: {first_status}\n\nShared Choices board: {board_status}\n\nPackage is not post-ready until image generation is repaired or replaced with approved GPT Image stills.")

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
    next_action = "inspect the two PNGs for likeness drift, readable allegations, real court marks, and sensitive-claim drift before approving any render handoff" if first_exists and board_exists else "repair OpenAI image billing/API access or run the saved prompts through GPT Image 2, then inspect the two PNGs before any render handoff"
    status_text = f"# Factory Run Status - Run {RUN_ID}\n\nStatus: {status}\n\nCreated: {NOW}\n\nResearch topic: late-April 2026 Rebel Wilson / The Deb defamation trial social-media evidence.\n\nSelected premise: Deb 24-Hour Story Evidence.\n\nWinner score: 62/80.\n\nCandidate board summary:\n- Deb 24-Hour Story Evidence: 62, winner.\n- Paramount Consumer Receipt Maze: 60, rejected duplicate merger lane.\n- Taylor Voice Trademark Booth: 59, rejected duplicate voice-vault lane.\n- BAFTA Duty-of-Care Switchboard: 47, rejected taste risk.\n\nGenerated still paths:\n- `{first_path}`\n- `{board_path}`\n\nMissing files: {missing_text}\n\nPost-ready exports: none; requires generated stills and human QC before posting.\n\nQC failures: {'none at automation level; human visual QC still required' if first_exists and board_exists else 'image generation blocked; PNG assets missing'}.\n\nHigh-risk issues: ongoing litigation, disputed sensitive allegations, identity drift, readable generated text.\n\nNext human action: {next_action}.\n"
    write("FACTORY_RUN_STATUS.md", status_text)
    (RUN_DIR / "FACTORY_RUN_STATUS.md").write_text(status_text, encoding="utf-8")


if __name__ == "__main__":
    main()
