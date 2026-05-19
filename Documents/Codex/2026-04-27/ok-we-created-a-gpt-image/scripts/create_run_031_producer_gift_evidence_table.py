from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "031"
SLUG = "producer_gift_evidence_table"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

FIRST_FRAME_SRC = Path(
    "/Users/speed/.codex/generated_images/019de85f-f700-7951-99a0-d04278eb7df2/"
    "ig_02d16dfb0ad24fb30169f5db6699e0819ab8b7a881d22645f3.png"
)
STORYBOARD_SRC = Path(
    "/Users/speed/.codex/generated_images/019de85f-f700-7951-99a0-d04278eb7df2/"
    "ig_02d16dfb0ad24fb30169f5dcd183ac819aa8ed1b583c4fd410.png"
)


def mkdirs() -> None:
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
        "storyboards/shared_choices/rejected",
        "qc",
    ]:
        (PKG / rel).mkdir(parents=True, exist_ok=True)


def write(rel: str, content: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_json(rel: str, data: object) -> None:
    write(rel, json.dumps(data, indent=2, ensure_ascii=False))


sources = [
    {
        "id": "src_001",
        "title": "Jorge Masvidal Explains Why Chael Sonnen Is Beefing With Him",
        "url": "https://www.tmz.com/2026/05/01/jorge-masvidal-explains-chael-sonnen-feud/",
        "publisher": "TMZ Sports",
        "published": "2026-05-01",
        "used_for": "Primary report and quote basis for the selected premise.",
        "verification": "Current public entertainment/sports report. Treat Masvidal's account as an allegation/claim, not a proven fact.",
    },
    {
        "id": "src_002",
        "title": "Feuds: News, Photos, Videos",
        "url": "https://www.tmz.com/categories/feuds/",
        "publisher": "TMZ",
        "published": "2026-05-01",
        "used_for": "Freshness scan for feud-heavy candidate board.",
        "verification": "Category page surfaced the Masvidal/Sonnen item and alternate public-conflict stories.",
    },
    {
        "id": "src_003",
        "title": "Celebrity News | Entertainment News",
        "url": "https://www.tmz.com/",
        "publisher": "TMZ",
        "published": "2026-05-02",
        "used_for": "Current entertainment-news intake and alternate candidate scan.",
        "verification": "Live news index; individual story details need source-level confirmation before use.",
    },
]

candidates = [
    {
        "id": "A",
        "premise": "Producer Gift Evidence Table: coffee, flowers, and chocolates become forensic exhibits in an MMA studio job-feud.",
        "source_basis": ["src_001", "src_002"],
        "famous_face": 8,
        "public_conflict": 9,
        "ego_humiliation": 9,
        "absurd_quote_or_defense": 10,
        "brand_location_contrast": 8,
        "first_frame_contradiction": 10,
        "taste_risk_inverse": 7,
        "freshness": 10,
        "total": 71,
        "notes": "Winner. The gift-table detail is visual, petty, and nonviolent enough to stage as evidence without amplifying serious harm.",
    },
    {
        "id": "B",
        "premise": "Kimmel Upfront Attendance Desk: a talk-show host signs in at a corporate ad upfront like it is a diplomatic incident.",
        "source_basis": ["src_003"],
        "famous_face": 8,
        "public_conflict": 7,
        "ego_humiliation": 6,
        "absurd_quote_or_defense": 5,
        "brand_location_contrast": 7,
        "first_frame_contradiction": 7,
        "taste_risk_inverse": 6,
        "freshness": 9,
        "total": 55,
        "notes": "Timely but more political/media procedural than visually absurd.",
    },
    {
        "id": "C",
        "premise": "Kylie Housekeeper No-Eye-Contact Hallway: luxury glam squad red carpet becomes HR compliance walk-through.",
        "source_basis": ["src_003"],
        "famous_face": 9,
        "public_conflict": 7,
        "ego_humiliation": 6,
        "absurd_quote_or_defense": 6,
        "brand_location_contrast": 8,
        "first_frame_contradiction": 8,
        "taste_risk_inverse": 4,
        "freshness": 9,
        "total": 57,
        "notes": "Strong environment contrast, but employment-law allegations create avoidable taste and defamation risk.",
    },
    {
        "id": "D",
        "premise": "Celebrity Boxing Dribble Warning: basketball court language gets translated into a boxing safety briefing.",
        "source_basis": ["src_002"],
        "famous_face": 6,
        "public_conflict": 8,
        "ego_humiliation": 7,
        "absurd_quote_or_defense": 8,
        "brand_location_contrast": 8,
        "first_frame_contradiction": 8,
        "taste_risk_inverse": 5,
        "freshness": 5,
        "total": 55,
        "notes": "Usable but lower fame/freshness, and the violent quote makes the joke less clean.",
    },
]

winner = candidates[0]


def main() -> None:
    mkdirs()
    shutil.copy2(FIRST_FRAME_SRC, PKG / "frames/gpt_image_2/first_frame_v01.png")
    shutil.copy2(STORYBOARD_SRC, PKG / "storyboards/shared_choices/shared_choices_v01.png")

    write_json("research/sources.json", {"created_at": NOW, "sources": sources})
    write(
        "research/last30days_report.md",
        f"""
# Last 30 Days Research Intake - Run {RUN_ID}

Created: {NOW}

Research query/topic: current public feuds with a famous face, ego wound, absurd defense, and a strong first-frame contradiction.

Intake sequence followed:
1. Current web/source scan before premise selection.
2. Candidate board built from fresh public conflict context.
3. Candidates scored before selecting a winner.
4. Run package created only after winner selection.

Best current source: TMZ Sports reported on 2026-05-01 that Jorge Masvidal explained his feud with Chael Sonnen and claimed Sonnen brought coffee, flowers, and chocolate to producers while pursuing a hosting job. This is treated as Masvidal's claim, not as independently proven fact.

Why it fits SGFLIX: the petty producer-gift detail creates a clean visual contradiction: hard-edged MMA broadcast ego staged like a forensic evidence table. It has famous faces, public conflict, ego/humiliation, and a prop trio that reads instantly in a first frame.

Rejected/held ideas:
- Kimmel upfront attendance: timely, but lower prop absurdity.
- Kylie hired-help lawsuit hallway: visually strong but higher employment-law/taste risk.
- Celebrity boxing dribble warning: colorful quote, but less fresh and more violent.

Fact handling:
- Do not state that Sonnen actually brought gifts as a verified fact.
- Use wording such as "Masvidal claimed," "reported feud," and "producer gift allegation."
- No real logos, no exact likeness cloning, no slurs, no legal conclusions.
""",
    )
    write_json("strategy/candidate_board.json", {"created_at": NOW, "candidates": candidates, "winner_id": winner["id"]})
    write(
        "strategy/winner_decision.md",
        f"""
# Winner Decision - Run {RUN_ID}

Selected premise: {winner["premise"]}

Score summary:
- Total: {winner["total"]}/80
- Freshness: {winner["freshness"]}/10
- First-frame contradiction: {winner["first_frame_contradiction"]}/10
- Absurd quote/defense: {winner["absurd_quote_or_defense"]}/10
- Ego/humiliation: {winner["ego_humiliation"]}/10
- Taste-risk inverse: {winner["taste_risk_inverse"]}/10

Decision: proceed with a fictionalized sports-media green-room satire. The joke lives on the evidence-table treatment of mundane gifts, not on proving any real-world employment or character claim.
""",
    )
    write_json(
        "strategy/phase_minus_one_worthiness_audit.json",
        {
            "phase_minus_one_audit": {
                "source_target": "Masvidal/Sonnen producer gift feud claim",
                "track_a_newsjack_velocity": {
                    "active_trend_score": 8,
                    "algorithmic_slipstream": "Fresh TMZ Sports feud item from 2026-05-01 with MMA personalities and a quoted prop list.",
                    "polarization_factor": 7,
                    "track_a_total": 15,
                    "track_a_verdict": "PASS",
                },
                "track_b_archetype_resonance": {
                    "iconography_strength": 8,
                    "stereotype_rigidity": "High",
                    "subversion_potential": 9,
                    "track_b_total": 17,
                    "track_b_verdict": "PASS",
                },
                "system_verdict": {
                    "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                    "primary_vector": "TRACK_A_NEWSJACK",
                    "urgency_class": "High",
                    "strategic_directive": "Stage the mundane gift list as premium forensic sports-media evidence.",
                },
            }
        },
    )
    write_json(
        "strategy/source_entropy_audit.json",
        {
            "source_entropy_audit": {
                "baseline_reality_check": "A reported public feud between MMA media figures over podcast comments and a hosting role.",
                "detected_anomalies": ["Coffee, flowers, and chocolate allegedly used as producer butter-up props."],
                "native_entropy_score": 6,
                "subject_self_awareness": "trying_to_look_cool",
                "comedic_vector_recommendation": "cringe_vector",
                "recommended_strategy": "micro_spotlight",
            }
        },
    )
    write_json(
        "strategy/humor_logic_bridge.json",
        {
            "setup": "MMA studio feud is framed with maximum masculine seriousness.",
            "turn": "The supposed smoking gun is a gift basket table.",
            "payoff": "Every camera treats coffee, flowers, and chocolate like fight-night evidence.",
            "rules": ["Keep the claim alleged", "No slurs", "No exact real faces", "No real network logos"],
        },
    )
    write_json(
        "strategy/tribe_meta_score.json",
        {
            "tribe_meta_score": {
                "attention_hook": 9,
                "identity_signal": 8,
                "remixability": 8,
                "caption_lift": 8,
                "share_trigger": 8,
                "total": 41,
                "verdict": "Strong sports/internet-culture clip premise.",
            }
        },
    )
    write_json(
        "strategy/risk_taste_score.json",
        {
            "risk_taste_score": {
                "defamation_risk": "Medium if stated as fact; Low-Medium if framed as Masvidal's reported claim.",
                "likeness_risk": "Medium; use fictionalized archetypes.",
                "platform_safety": "Medium; omit slurs and explicit sexual insults from source language.",
                "taste_score": 7,
                "required_mitigations": ["alleged/reported wording", "no exact likenesses", "no logos", "prop comedy only"],
            }
        },
    )
    write(
        "strategy/franchise_decision.md",
        """
# Franchise Decision

Verdict: one-off with franchise option.

Reusable lane: "Evidence Table for Petty Celebrity Claims" can recur when the source has one absurd concrete object list. Do not force a series unless future stories provide equally visual props.
""",
    )
    write_json(
        "chai/chai_shot_specs.json",
        {
            "run_id": RUN_ID,
            "title": "Producer Gift Evidence Table",
            "shots": [
                {
                    "shot": "001",
                    "duration_seconds": 6,
                    "subject": "fictionalized MMA analyst and fighter-host archetypes",
                    "scene": "premium MMA studio green room",
                    "motion": "slow push toward evidence table, then rack focus to defensive hands",
                    "spatial": "coffee left, flowers center, chocolate right; talent behind table",
                    "camera": "24mm vertical first frame, then 50mm insert",
                    "critique": "Avoid exact faces and real logos; keep evidence labels generic.",
                    "revision": "If identity drift resembles real people too closely, use silhouette or crop to hands/props.",
                }
            ],
        },
    )
    shot = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "premise": winner["premise"],
        "no_video_generation": True,
        "visual_prompt": "MMA broadcast green room, producer gift table as forensic evidence, fictionalized archetypes only.",
        "risk_notes": ["Claim must remain alleged/reported.", "No real logos or exact faces."],
    }
    write_json("scene_json/shot_0001.json", shot)
    write_json("scene_json/shot_001.json", shot)
    write_json(
        "handoffs/closed_tool_handoff.json",
        {
            "run_id": RUN_ID,
            "premise": winner["premise"],
            "status": "READY_FOR_MANUAL_STILL_REVIEW_NO_VIDEO_RENDER_REQUESTED",
            "do_not_call": ["Grok Video", "Kling", "Runway", "Luma", "Sora", "Seedance"],
            "manual_next_step": "Review stills and copy handoff into a closed video tool only if a human chooses to render later.",
            "assets": {
                "first_frame": "frames/gpt_image_2/first_frame_v01.png",
                "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
            },
        },
    )
    write(
        "handoffs/grok_agent_prompt.md",
        """
# Grok Agent Prompt - Manual Only

Create no video automatically.

If a human later asks for a visual pack, use the existing first frame and Shared Choices board as references. Build a fictionalized MMA studio green-room satire around coffee, flowers, and chocolate being handled like producer evidence. Keep the claim reported/alleged. Do not use exact real likenesses, real network logos, slurs, or legal conclusions.
""",
    )
    write(
        "captions/instagram_caption.md",
        """
When the alleged producer gifts become Exhibit A, B, and C.

Coffee. Flowers. Chocolate. Fight-night ego has entered discovery.

#sgflix #mma #sportsmedia #satire #creatorworkflow
""",
    )
    write(
        "distribution/post_plan.md",
        """
# Post Plan

Primary hook: "The alleged job audition gifts are now evidence."

Surfaces: Instagram Reels, TikTok, Shorts.

Do not post automatically. Human review required for likeness similarity, source wording, and any accidental logo/text in the generated stills.
""",
    )
    write(
        "skool/case_study.md",
        """
# Skool Case Study

Lesson: convert a single absurd reported object list into a visual system.

This run shows the SGFLIX research-first process: current source intake, candidate scoring, winner selection, still generation, QC, and closed-tool handoff without video generation.
""",
    )
    write(
        "frames/gpt_image_2/first_frame_v01_prompt.md",
        """
Vertical first-frame prompt: premium satirical MMA broadcast green-room evidence table. Fictionalized retired analyst archetype and confrontational fighter-host archetype. Coffee tray, flowers, and chocolate box foregrounded as Exhibit A/B/C. No real logos, no exact faces, no defamatory captions, no slurs.
""",
    )
    write(
        "storyboards/shared_choices/shared_choices_v01_prompt.md",
        """
Shared Choices prompt: director-bible board for Producer Gift Evidence Table. Include character/hero props, color palette, environment, floor plan/blocking, four storyboard panels, lighting/mood/style, visual rules, and production notes. Fictionalized archetypes only; no real names, logos, legal claims, slurs, joke disclaimers, or the phrase "probably true."
""",
    )
    write(
        "qc/first_frame_v01_qc.md",
        """
# First Frame QC

Status: PASS_WITH_NOTES

Passes:
- Strong foreground prop read: coffee, flowers, chocolate.
- Vertical first-frame composition is clear.
- No real network logo is visible.
- Text is limited to generic exhibit labels.

Notes:
- Character archetypes may evoke real MMA personalities; keep any caption copy fictionalized and alleged.
""",
    )
    write(
        "qc/shared_choices_v01_qc.md",
        """
# Shared Choices QC

Status: PASS_WITH_NOTES

Passes:
- Includes character/props, palette, set design, blocking, storyboard panels, lighting, visual rules, and production notes.
- Repaired version removed the risky disclaimer gag from the rejected draft.
- No real names or real logos are used.

Rejected draft retained at: storyboards/shared_choices/rejected/shared_choices_v01_rejected_text_risk.png

Notes:
- Some small board text should be treated as design guidance, not source copy.
""",
    )
    manifest = {
        "run_id": RUN_ID,
        "created_at": NOW,
        "status": "COMPLETE_PACKAGE_NO_VIDEO",
        "assets": [
            "frames/gpt_image_2/first_frame_v01.png",
            "frames/gpt_image_2/first_frame_v01_prompt.md",
            "storyboards/shared_choices/shared_choices_v01.png",
            "storyboards/shared_choices/shared_choices_v01_prompt.md",
            "storyboards/shared_choices/rejected/shared_choices_v01_rejected_text_risk.png",
        ],
        "source_generated_images_dir": "/Users/speed/.codex/generated_images/019de85f-f700-7951-99a0-d04278eb7df2",
        "no_video_generation_tools_called": True,
    }
    write_json("manifests/asset_manifest.json", manifest)
    write_json(
        f"RUN_{RUN_ID}_MASTER_PACKAGE.json",
        {
            "run_id": RUN_ID,
            "slug": SLUG,
            "title": "Producer Gift Evidence Table",
            "created_at": NOW,
            "selected_premise": winner["premise"],
            "research_query": "current public feuds with absurd concrete props and first-frame contradiction",
            "status": "COMPLETE_PACKAGE_NO_VIDEO",
            "required_files_complete": True,
            "no_video_generation_tools_called": True,
        },
    )
    readme = f"""
# RUN {RUN_ID} MASTER PACKAGE - Producer Gift Evidence Table

Status: COMPLETE_PACKAGE_NO_VIDEO
Created: {NOW}

Selected premise: {winner["premise"]}

Research topic: current public feuds with famous faces, ego, absurd defenses, and prop-forward first frames.

Generated stills:
- frames/gpt_image_2/first_frame_v01.png
- storyboards/shared_choices/shared_choices_v01.png

Rejected/repair asset:
- storyboards/shared_choices/rejected/shared_choices_v01_rejected_text_risk.png

High-risk issues:
- The producer-gift detail is a reported claim from Masvidal, not verified fact.
- Avoid exact public-figure likenesses and real broadcast/network logos.
- Do not repeat source slurs or explicit insults in captions.

Next human action: review both stills for likeness/logo/text risk before any optional manual video work.
    """
    write("README.md", readme)
    write("FACTORY_RUN_STATUS.md", readme)
    (RUN_DIR / "FACTORY_RUN_STATUS.md").write_text(readme.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
