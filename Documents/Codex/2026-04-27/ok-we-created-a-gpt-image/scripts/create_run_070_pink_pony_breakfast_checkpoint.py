from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN = "070"
SLUG = "pink_pony_breakfast_checkpoint"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

FIRST_IMAGE_SOURCE = Path(
    "/Users/speed/.codex/generated_images/019de980-aed1-7fd1-91fb-63ec9deed855/"
    "ig_0f84ec0ed4bb16330169f62551e93c819981a9f576d539f213.png"
)
BOARD_IMAGE_SOURCE = Path(
    "/Users/speed/.codex/generated_images/019de980-aed1-7fd1-91fb-63ec9deed855/"
    "ig_0f84ec0ed4bb16330169f62685436081999f6769ee813cc8e3.png"
)

FIRST_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

SGFLIX satirical editorial first frame, vertical 1024x1536. Title concept: Pink Pony Breakfast Checkpoint. A fictional maximalist pop-star archetype, not an exact likeness of any real performer, sits alone at a luxury desert-festival hotel breakfast table in theatrical pink western-glam styling: rhinestone cowboy hat, structured pink jacket, dramatic stage makeup, oversized sunglasses on the table. The absurd visual contradiction: the calm hotel breakfast buffet has been converted into a tiny TSA-style security checkpoint, with velvet ropes, a miniature metal detector arch labeled only with abstract icons, a tray containing a croissant, room key, tiny microphone, and pink feather boa, and a serious bodyguard archetype holding a clipboard labeled GUEST FLOW. No children, no real logos, no readable celebrity names, no defamatory implication. Premium cinematic satire, glossy magazine lighting, 24mm low table-height wide shot, warm hotel morning light, polished marble, palm-shadow texture, Coachella-adjacent desert resort atmosphere, first-second joke clarity, humorous but tasteful, fictionalized public-incident commentary. Avoid messy text, exact celebrity likeness, real hotel/festival logos, child depiction, violence, harassment framing, fake news chyron."""

BOARD_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

Create one SGFLIX Shared Choices director-bible storyboard board for Run 070, internal title Pink Pony Breakfast Checkpoint, horizontal 1536x1024. Use a clean premium production-design board layout with minimal label-like text. Include: character canon for a fictional maximalist pop-star archetype in pink western-glam styling, bodyguard archetype, hotel breakfast staff silhouettes; hero props including croissant tray, room key, tiny microphone, feather boa, clipboard labeled GUEST FLOW, velvet rope, miniature metal detector arch, coffee urn, room-service cloche; color palette swatches of hot pink, butter yellow, chrome gray, hotel marble white, palm green, espresso black; environment/set design for luxury desert-festival hotel breakfast buffet converted into a tiny security checkpoint; floor plan and blocking diagram showing breakfast table foreground, buffet left, checkpoint arch center, guard right, resort windows back; six storyboard panels with camera/lens/movement notes for a 6-10 second handoff: low table-height reveal, rack focus to croissant tray, slow push through metal detector, clipboard stamp, pop-star deadpan sip, final wide reveal; lighting/mood/style notes: warm morning resort light, glossy editorial satire, restrained absurdity; visual rules: no exact likeness, no children, no real logos, no fake allegations, no messy readable text; production notes for manual closed-tool video handoff only, no auto video generation. Make the board legible as a director bible, not a finished poster."""

SOURCES = [
    {
        "id": "ap_chappell_jorginho_breakfast",
        "title": "Chappell Roan pushes back after soccer star Jorginho alleges his daughter was mistreated",
        "url": "https://apnews.com/article/d23631e3b7fe884f9a2efaae282ad41c",
        "publisher": "Associated Press",
        "published": "2026-03-22",
        "notes": "Primary grounding for the hotel breakfast dispute; treats guard conduct and Roan involvement as disputed/alleged.",
    },
    {
        "id": "latimes_guard_statement",
        "title": "Security guard at center of Chappell Roan controversy breaks silence",
        "url": "https://www.latimes.com/entertainment-arts/story/2026-03-25/chappell-roans-security-guard-breaks-silence-i-take-full-responsibility",
        "publisher": "Los Angeles Times",
        "published": "2026-03-25",
        "notes": "Reports Pascal Duvier statement taking responsibility and clarifying the guard lane.",
    },
    {
        "id": "tmz_jorginho_misunderstanding",
        "title": "Jorginho Notes 'Misunderstanding' Over Chappell Roan Bodyguard Confusion",
        "url": "https://www.tmz.com/2026/04/13/jorginho-says-chappell-roan-drama-was-a-misunderstanding/",
        "publisher": "TMZ",
        "published": "2026-04-13",
        "notes": "Source for subsequent de-escalation/misunderstanding framing; used as risk reducer, not sole factual basis.",
    },
    {
        "id": "rollingstone_bad_bunny_fees",
        "title": "Bad Bunny Seeks $465,000 in Legal Fees After Winning 'Un Verano Sin Ti' Copyright Case",
        "url": "https://ca.rollingstone.com/bad-bunny-copyright-case-legal-bill-reimbursement/",
        "publisher": "Rolling Stone Canada",
        "published": "2026-03-24",
        "notes": "Candidate context: dismissed copyright case and requested legal-fee reimbursement.",
    },
    {
        "id": "espn_wizards_scripted_prank",
        "title": "Wizards apologize for April Fools' stunt, say bit was scripted",
        "url": "https://www.espn.com/nba/story/_/id/48378756/wizards-apologize-april-fools-stunt-say-bit-scripted",
        "publisher": "ESPN",
        "published": "2026-04-02",
        "notes": "Candidate context: scripted half-court shot promotion and team apology.",
    },
    {
        "id": "yahoo_wizards_scripted_prank",
        "title": "Wizards apologize after scripted April Fools joke in which 'fan' thought they won $10,000 falls flat",
        "url": "https://sports.yahoo.com/nba/article/wizards-apologize-after-scripted-april-fools-joke-in-which-fan-thought-they-won-10000-falls-flat-170425548.html",
        "publisher": "Yahoo Sports",
        "published": "2026-04-02",
        "notes": "Secondary source for the Wizards candidate and audience reaction context.",
    },
]

CANDIDATES = [
    {
        "id": "pink_pony_breakfast_checkpoint",
        "title": "Pink Pony Breakfast Checkpoint",
        "premise": "A fictional maximalist pop-star archetype finds her quiet hotel breakfast converted into a tiny TSA-style checkpoint after a public security-guard misunderstanding.",
        "source_ids": ["ap_chappell_jorginho_breakfast", "latimes_guard_statement", "tmz_jorginho_misunderstanding"],
        "scores": {
            "famous_face_or_power_archetype": 9,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 9,
            "first_frame_visual_contradiction": 10,
            "freshness": 7,
            "risk_control": 8,
            "franchise_potential": 8,
        },
        "total": 76,
        "selection_notes": "Winner: strongest immediate visual contradiction and can be fictionalized away from child/harassment claims.",
    },
    {
        "id": "bad_bunny_legal_fee_dance_class",
        "title": "Legal Fee Dance Class",
        "premise": "A reggaeton dance studio becomes an invoice-auditing courtroom where every eight-count costs $465,612 in legal fees.",
        "source_ids": ["rollingstone_bad_bunny_fees"],
        "scores": {
            "famous_face_or_power_archetype": 9,
            "public_conflict": 7,
            "ego_humiliation": 7,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "freshness": 7,
            "risk_control": 8,
            "franchise_potential": 8,
        },
        "total": 70,
        "selection_notes": "Strong object comedy, but less first-frame surprise than the breakfast checkpoint.",
    },
    {
        "id": "wizards_halfcourt_prank_refund_desk",
        "title": "Halfcourt Prank Refund Desk",
        "premise": "A struggling arena's halftime skit is rerouted to a fake $10,000 refund counter staffed by performance-team clipboard auditors.",
        "source_ids": ["espn_wizards_scripted_prank", "yahoo_wizards_scripted_prank"],
        "scores": {
            "famous_face_or_power_archetype": 6,
            "public_conflict": 8,
            "ego_humiliation": 8,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 9,
            "freshness": 8,
            "risk_control": 9,
            "franchise_potential": 6,
        },
        "total": 70,
        "selection_notes": "Clean satire but weaker famous-face signal and less SGFLIX celebrity pull.",
    },
    {
        "id": "sabrina_yodel_permit_office",
        "title": "Yodel Permit Office",
        "premise": "A festival pop-performance yodel becomes a municipal permit hearing with a tiny alpine horn on the evidence table.",
        "source_ids": ["prior_current_scan_only"],
        "scores": {
            "famous_face_or_power_archetype": 8,
            "public_conflict": 5,
            "ego_humiliation": 7,
            "absurd_quote_or_object": 9,
            "brand_location_contrast": 7,
            "first_frame_visual_contradiction": 8,
            "freshness": 5,
            "risk_control": 9,
            "franchise_potential": 7,
        },
        "total": 65,
        "penalty": "Rejected because existing run_058_sabrina_yodel_lost_found already covers this lane.",
    },
]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    dirs = [
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
    ]
    for directory in dirs:
        (PKG / directory).mkdir(parents=True, exist_ok=True)

    previous_check = (
        "# Previous Run Check\n\n"
        "Highest official non-aborted run found before this cycle: "
        "`run_069_banksy_flag_optometry`.\n\n"
        "Status read: complete with built-in GPT Image stills and no generated video. "
        "No previous-run repair replaced this new research-first cycle.\n"
    )
    write(RUN_DIR / "PREVIOUS_RUN_CHECK.md", previous_check)

    write(
        PKG / "research/last30days_report.md",
        """
# Last 30 Days Research Report - Run 070

Step 1: Research Intake was performed before premise selection, package creation, or still generation.

Research query/topic: current public-entertainment incidents with famous-face signal, public conflict, ego/humiliation, absurd object logic, and a strong first-frame contradiction as of 2026-05-02.

Intake summary:
- Chappell Roan / Jorginho / hotel breakfast security dispute: AP reported the initial dispute around a hotel breakfast in Sao Paulo and Roan's response. Later reporting covered guard responsibility and a misunderstanding/de-escalation frame. The run uses only the verified public shape: hotel breakfast, security guard confusion, public backlash, and later clarification. It avoids depicting the child, proving blame, or using an exact likeness.
- Bad Bunny legal-fee request: Rolling Stone Canada reported a $465,612 legal-fee request after a dismissed copyright case tied to "Enseñame a Bailar." This has strong invoice comedy but less immediate first-frame contradiction.
- Washington Wizards scripted prank apology: ESPN/Yahoo reported the scripted April Fools half-court-shot stunt and apology. This is visually clean but lacks a famous-face anchor.
- Sabrina/yodel material was treated as a duplicate lane because run_058 already exists.

Winner rationale:
The breakfast checkpoint wins because the reality baseline is mundane and specific: a hotel breakfast table. The satire intervention is one clean object-system inversion: breakfast becomes airport-style guest-flow security. That gives strong first-second comprehension without adding allegations.

Verification/risk notes:
- All claims about who directed or caused the incident remain disputed/alleged.
- This package frames the premise as fictionalized public-incident satire.
- No child appears in stills or prompts.
- No video-generation tool was called.
""",
    )
    write_json(PKG / "research/sources.json", {"created_at": NOW, "sources": SOURCES})
    write_json(PKG / "strategy/candidate_board.json", {"created_at": NOW, "scoring_scale": "1-10 per category; total is sum", "candidates": CANDIDATES})
    write(
        PKG / "strategy/winner_decision.md",
        """
# Winner Decision - Run 070

Selected premise: `Pink Pony Breakfast Checkpoint`.

Winner score: 76.

Score summary:
- Pink Pony Breakfast Checkpoint: 76
- Bad Bunny Legal Fee Dance Class: 70
- Wizards Halfcourt Prank Refund Desk: 70
- Sabrina Yodel Permit Office: 65, rejected for duplicate risk with run_058

Decision:
Proceed with the fictionalized breakfast-checkpoint premise. The source has a famous performer, public conflict, an ego/status contradiction, and a visual system that can be made absurd without inventing facts. The stills must avoid child depiction, exact celebrity likeness, real hotel/festival logos, and any implication that disputed allegations are proven.
""",
    )
    write_json(
        PKG / "strategy/phase_minus_one_worthiness_audit.json",
        {
            "phase_minus_one_audit": {
                "source_target": "hotel breakfast security misunderstanding around famous pop performer",
                "track_a_newsjack_velocity": {
                    "active_trend_score": 7,
                    "algorithmic_slipstream": "recent entertainment-news backlash with follow-up clarification",
                    "polarization_factor": 8,
                    "track_a_total": 22,
                    "track_a_verdict": "PASS",
                },
                "track_b_archetype_resonance": {
                    "iconography_strength": 9,
                    "stereotype_rigidity": "High",
                    "subversion_potential": 9,
                    "track_b_total": 27,
                    "track_b_verdict": "PASS",
                },
                "system_verdict": {
                    "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                    "primary_vector": "TRACK_B_EVERGREEN_ARCHETYPE",
                    "urgency_class": "Medium",
                    "strategic_directive": "Use fictionalized celebrity-status iconography and one bureaucratic object system; do not litigate the child/guard facts.",
                },
            }
        },
    )
    write_json(
        PKG / "strategy/source_entropy_audit.json",
        {
            "source_entropy_audit": {
                "baseline_reality_check": "A public performer, a soccer player's family, and security confusion at a hotel breakfast during Brazil festival travel.",
                "detected_anomalies": [
                    "A low-stakes breakfast setting escalated into global entertainment backlash.",
                    "Security/bodyguard ambiguity became the whole story.",
                    "Later clarification/misunderstanding reduced the need for blame-based satire.",
                ],
                "native_entropy_score": 5,
                "subject_self_awareness": "unaware",
                "comedic_vector_recommendation": "cringe_vector",
                "recommended_strategy": "micro_spotlight",
            }
        },
    )
    write_json(
        PKG / "strategy/humor_logic_bridge.json",
        {
            "bridge": {
                "setup": "A glamorous pop-star breakfast is supposed to be quiet, private, and mundane.",
                "turn": "The room now behaves like an airport checkpoint because everyone is over-managing who may pass a table.",
                "button": "The croissant itself goes through security before the celebrity takes a deadpan sip.",
                "do_not_do": ["do not show a child", "do not prove fault", "do not use exact likeness", "do not add chaos beyond checkpoint logic"],
            }
        },
    )
    write_json(
        PKG / "strategy/tribe_meta_score.json",
        {
            "tribe_meta_score": {
                "instant_read": 9,
                "status_comedy": 9,
                "public_conflict_legibility": 8,
                "caption_flexibility": 8,
                "shareability": 8,
                "overall": 42,
                "notes": "Readable to pop-culture and sports-adjacent audiences without requiring the viewer to know every detail.",
            }
        },
    )
    write_json(
        PKG / "strategy/risk_taste_score.json",
        {
            "risk_taste_score": {
                "defamation_risk": "medium controlled by fictionalization and alleged/disputed labels",
                "minor_depiction_risk": "controlled: no child depiction",
                "likeness_risk": "medium: prompts require fictional archetype, not exact performer",
                "brand_logo_risk": "low if QC confirms no real hotel/festival logos",
                "taste_verdict": "usable after human image inspection",
                "score": 8,
            }
        },
    )
    write(
        PKG / "strategy/franchise_decision.md",
        """
# Franchise Decision

Franchise lane: `Celebrity Privacy Infrastructure`.

Reusable pattern:
Take a public status/privacy conflict and convert it into a tiny overbuilt service checkpoint: breakfast, dressing room, coat check, green room, airport lounge, restaurant host stand.

Decision: keep as a one-off with franchise option. The checkpoint grammar is reusable, but this specific incident should remain tightly fictionalized and not become harassment-bait.
""",
    )

    first_dest = PKG / "frames/gpt_image_2/first_frame_v01.png"
    board_dest = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    shutil.copy2(FIRST_IMAGE_SOURCE, first_dest)
    shutil.copy2(BOARD_IMAGE_SOURCE, board_dest)
    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT)
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)

    shot = {
        "run_id": RUN,
        "shot": "001",
        "duration_seconds": 8,
        "subject": "fictional maximalist pop-star archetype at hotel breakfast checkpoint",
        "scene": "luxury desert-festival hotel breakfast buffet converted into miniature guest-flow security",
        "motion": "slow low table-height push from croissant tray through checkpoint arch to deadpan sip",
        "spatial": "breakfast table foreground; buffet left; checkpoint arch center; guard right; resort windows back",
        "camera": "24mm low wide, rack focus from tray to clipboard, final wider reveal",
        "critique": "must read as fictionalized privacy-status satire, not a fake documentary still",
        "revision": "remove children, real logos, exact likeness, harassment framing, and messy text",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
    }
    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN, "title": "Pink Pony Breakfast Checkpoint", "shots": [shot]})
    scene = {
        "run_id": RUN,
        "shot_id": "shot_001",
        "status": "handoff_only_no_video_generation",
        "source_image": "frames/gpt_image_2/first_frame_v01.png",
        "prompt": "Animate a restrained table-height push through a hotel breakfast security checkpoint: croissant tray, tiny metal detector, clipboard stamp, deadpan sip, warm resort light.",
        "negative": "children, exact celebrity likeness, real logos, proven blame, aggressive confrontation, fake news chyron, messy readable text",
    }
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    handoff = {
        "run_id": RUN,
        "title": "Pink Pony Breakfast Checkpoint",
        "do_not_generate_video_in_automation": True,
        "video_generation_permitted": False,
        "premise": CANDIDATES[0]["premise"],
        "first_frame": "frames/gpt_image_2/first_frame_v01.png",
        "storyboard": "storyboards/shared_choices/shared_choices_v01.png",
        "manual_only_next_step": "Human reviews stills, then may manually route to a closed video tool outside this automation.",
        "guardrails": ["fictional archetype only", "no child depiction", "no real logos", "all source facts treated as disputed/alleged where applicable"],
    }
    write_json(PKG / "handoffs/closed_tool_handoff.json", handoff)
    write(
        PKG / "handoffs/grok_agent_prompt.md",
        """
# Grok Agent Prompt - Manual Only

Use the approved stills and CHAI shot specs for a fictionalized SGFLIX short titled `Pink Pony Breakfast Checkpoint`.

Do not generate video unless a human explicitly starts the render outside this automation. Preserve the first frame composition. Keep the joke to one system: a hotel breakfast buffet has become a miniature guest-flow checkpoint. No child, no real logos, no exact celebrity likeness, no claims of proven wrongdoing.
""",
    )
    write(
        PKG / "captions/instagram_caption.md",
        """
POV: breakfast now has a guest-flow department.

The croissant has cleared security. The room key is still under review.

#sgflix #popculture #satire #festivalweekend #hotelbreakfast #privacyplease
""",
    )
    write(
        PKG / "distribution/post_plan.md",
        """
# Post Plan

Primary surfaces: Instagram Reels, TikTok, YouTube Shorts.

Hook text option: `BREAKFAST CHECKPOINT OPEN`.

Publishing conditions:
- Human approves first-frame and Shared Choices stills.
- No child depiction, exact celebrity likeness, real logos, or messy text.
- Caption remains fictionalized and does not allege proven misconduct.

Post-ready exports: none in this automation; no video footage generated.
""",
    )
    write(
        PKG / "skool/case_study.md",
        """
# Skool Case Study - Pink Pony Breakfast Checkpoint

Teaching point: convert disputed celebrity-drama entropy into a controllable object system.

Instead of replaying allegations, this run isolates the verified cultural shape: a famous performer, breakfast privacy, a security ambiguity, and public backlash. The comedic engine makes the setting enforce itself through tiny checkpoint props. This keeps the satire visual, legible, and less dependent on proving disputed facts.
""",
    )
    manifest = {
        "run_id": RUN,
        "status": "complete_with_builtin_gpt_image_stills_no_video_generated",
        "created_at": NOW,
        "assets": [
            {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": "builtin_gpt_image_2_workflow", "source": str(FIRST_IMAGE_SOURCE)},
            {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt"},
            {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": "builtin_gpt_image_2_workflow", "source": str(BOARD_IMAGE_SOURCE)},
            {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt"},
        ],
        "missing_files": [],
        "video_generation_tools_called": False,
    }
    write_json(PKG / "manifests/asset_manifest.json", manifest)
    write(
        PKG / "qc/first_frame_v01_qc.md",
        """
# First Frame QC

Asset: `frames/gpt_image_2/first_frame_v01.png`

Generation mode: `builtin_gpt_image_2_workflow`.

Verdict: usable for human review and package handoff.

Passes:
- Clear first-second joke: hotel breakfast converted into security checkpoint.
- No child depiction.
- No obvious real hotel/festival logo.
- Fictionalized pop-star archetype rather than exact documentary framing.

Human review focus: inspect for likeness drift, unreadable/messy small labels, and any implied proven blame.
""",
    )
    write(
        PKG / "qc/shared_choices_v01_qc.md",
        """
# Shared Choices QC

Asset: `storyboards/shared_choices/shared_choices_v01.png`

Generation mode: `builtin_gpt_image_2_workflow`.

Verdict: usable director-bible board for human review.

Passes:
- Includes character/prop canon, palette, set design, floor plan/blocking, storyboard panels, lighting/style notes, visual rules, and production notes.
- No video generation performed.

Human review focus: inspect generated microtext, logo drift, and whether any panel resembles a real person too closely.
""",
    )
    package = {
        "run_id": RUN,
        "slug": SLUG,
        "title": "Pink Pony Breakfast Checkpoint",
        "status": "complete_with_builtin_gpt_image_stills_no_video_generated",
        "created_at": NOW,
        "research_query": "current public-entertainment incidents with famous-face signal and first-frame contradiction, scanned 2026-05-02",
        "selected_premise": CANDIDATES[0]["premise"],
        "winner_score": 76,
        "candidate_scores": {candidate["id"]: candidate["total"] for candidate in CANDIDATES},
        "video_generation": "not requested and not performed",
        "still_paths": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
        },
        "next_human_action": "Review the two still PNGs for likeness/logo/text risk, then approve or request a repair prompt before any manual video handoff.",
    }
    write_json(PKG / f"RUN_{RUN}_MASTER_PACKAGE.json", package)
    readme = f"""
# RUN {RUN} MASTER PACKAGE - Pink Pony Breakfast Checkpoint

Status: `complete_with_builtin_gpt_image_stills_no_video_generated`.

Selected premise: {CANDIDATES[0]["premise"]}

Research query/topic: current public-entertainment incidents with famous-face signal and first-frame contradiction, scanned 2026-05-02.

Score summary: winner 76; Bad Bunny Legal Fee Dance Class 70; Wizards Halfcourt Prank Refund Desk 70; Sabrina Yodel Permit Office 65 and rejected for duplicate risk.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing files: none from the required checklist.

Post-ready exports: none; stills require human approval before any manual video workflow.

Exact next human action: Review both still PNGs for likeness/logo/text risk, then approve or request a repair prompt before any manual closed-tool video handoff.
"""
    write(PKG / "README.md", readme)
    status = f"""
# Factory Run Status - Run {RUN}

Status: `complete_with_builtin_gpt_image_stills_no_video_generated`

New run id: `run_{RUN}_{SLUG}`

Research query/topic: current public-entertainment incidents with famous-face signal and first-frame contradiction, scanned 2026-05-02.

Candidate board: created at `strategy/candidate_board.json`.

Selected premise: `Pink Pony Breakfast Checkpoint`.

Score summary: winner 76; Bad Bunny Legal Fee Dance Class 70; Wizards Halfcourt Prank Refund Desk 70; Sabrina Yodel Permit Office 65 and rejected for duplicate risk with run_058.

Created files: all required package artifacts were written under `RUN_{RUN}_MASTER_PACKAGE`.

Generated still-image paths:
- `{first_dest}`
- `{board_dest}`

Missing files: none from the required checklist.

Post-ready exports: none; no video footage generated.

QC failures: none blocking. Human visual inspection still required for microtext, likeness drift, and logo drift.

High-risk issues: disputed source facts, child-related source context, public-figure likeness drift, accidental real festival/hotel branding.

Exact next human action: Review both still PNGs for likeness/logo/text risk, then approve or request a repair prompt before any manual closed-tool video handoff.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", status)


if __name__ == "__main__":
    main()
