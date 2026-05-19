#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "050"
SLUG = "jada_legal_fee_red_table"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

FIRST_FRAME_SRC = Path("/Users/speed/.codex/generated_images/019de8e5-a460-7fd2-b29d-448b5d5aebf0/ig_07a1181ee1eb9d040169f5fdffca8081948e34938ad74fd384.png")
STORYBOARD_SRC = Path("/Users/speed/.codex/generated_images/019de8e5-a460-7fd2-b29d-448b5d5aebf0/ig_07a1181ee1eb9d040169f5fe81e90881948694e05d434cf597.png")

FIRST_FRAME_PROMPT = """Use case: illustration-story
Asset type: SGFLIX first-frame still, vertical 9:16 social video opening frame
Primary request: Create a satirical cinematic courtroom-receipt tableau inspired by a public entertainment legal-fee story, without depicting exact real-person likenesses or real logos.
Scene/backdrop: A glossy Hollywood small-claims courtroom crossed with a daytime talk-show set. A red round table sits where the witness stand should be. Behind it, a giant itemized legal-fee receipt curls down from the ceiling like a banner.
Subject: A fictional poised celebrity host with cropped hair and a structured emerald suit stands at the red table, calmly sliding a comically oversized invoice marked "LEGAL FEES" toward an off-camera former friend. A fictional movie-star husband silhouette is visible only as a cardboard courtroom exhibit in the background, not a recognizable portrait.
Composition: Vertical 9:16, first-frame hook, low wide-angle lens, receipt foreground huge, celebrity host centered, judge bench and red table visible, strong facial expression but not an exact likeness of Jada Pinkett Smith, Will Smith, or Bilaal Salaam.
Visual comedy: The first-frame contradiction is a therapy-talk-show red table operating as a collections desk. Include generic stamps like "MOTION FILED" and "$49K" on paper props only; no brand logos, no actual court seals.
Style: Premium satirical editorial still, high-detail cinematic lighting, rich reds and greens balanced with neutral courthouse wood, sharp production design, realistic but clearly fictionalized.
Safety/accuracy: Do not create a real news photo. Do not imply anyone is guilty of a crime. Avoid exact celebrity likeness, avoid real Red Table Talk branding, avoid real network logos, avoid defamatory signage.
Text constraints: Only simple legible prop text: "LEGAL FEES", "$49K", "MOTION FILED". No other readable text, no watermarks.
"""

STORYBOARD_PROMPT = """Use case: infographic-diagram
Asset type: SGFLIX Shared Choices director-bible storyboard board, landscape production reference sheet
Primary request: Create one premium director's-bible storyboard board for a satirical SGFLIX short about a fictional celebrity legal-fee collection scene inspired by current entertainment reporting. Do not depict exact real-person likenesses or logos.
Canvas/layout: Landscape 16:9 board divided into clean labeled visual zones with minimal legible labels. Use refined production-design layout, not a messy collage.
Required zones: 1) character canon: fictional cropped-hair celebrity host in emerald suit, generic former-friend figure as off-camera hands only, cardboard movie-star silhouette prop; 2) hero props: oversized LEGAL FEES receipt, red round table, court stamp, evidence binder, red mug; 3) palette: courthouse walnut, talk-show red, emerald suit, paper white, brass lights; 4) environment/set design: courtroom blended with daytime talk-show red table set; 5) floor plan/blocking: top-down simple plan showing red table, judge bench, receipt drop, camera path; 6) storyboard panels: four panels showing receipt unroll, invoice slide, former-friend hands freeze, judge reaction; 7) camera/lens/movement notes represented visually with arrows, 24mm low push-in, overhead insert, rack focus; 8) lighting/mood/style notes: glossy courtroom, theatrical red curtains, no real logos; 9) visual rules and production notes: fictionalized faces, no real seals, claims stay as legal motion/request.
Style: Premium cinematic satire board, realistic concept art plus clean production annotations, polished editorial taste, high contrast but not dark, readable enough for director reference.
Text constraints: Use only short labels: "CHARACTER", "PROPS", "PALETTE", "SET", "BLOCKING", "PANELS", "CAMERA", "LIGHT", "RULES" plus prop text "LEGAL FEES" and "$49K". Avoid long paragraphs and avoid gibberish text.
Safety/accuracy: No exact Jada Pinkett Smith, Will Smith, or Bilaal Salaam likeness; no Red Table Talk logo; no official court seal; no defamatory assertions.
"""

SOURCES = [
    {
        "id": "latimes_2026_04_24_jada_legal_fees",
        "title": "Jada Pinkett Smith asks court to make Will Smith's former friend pay her $49,000 legal bills",
        "publisher": "Los Angeles Times",
        "url": "https://www.latimes.com/entertainment-arts/story/2026-04-24/jada-pinkett-smith-asks-court-for-bilaal-salaam-to-pay-legal-bills",
        "published": "2026-04-24",
        "used_for": "Primary factual spine: legal-fee motion, amount, anti-SLAPP context, ongoing case status.",
        "verification": "searched_current_web_2026_05_02",
    },
    {
        "id": "tmz_2026_04_23_jada_49k",
        "title": "Jada Pinkett Smith Asks Judge to Force Will's Ex-Pal to Cough Up $49K in Court",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/23/jada-pinkett-smith-asks-judge-to-force-bilaal-salaam-to-pay-fees/",
        "published": "2026-04-23",
        "used_for": "Secondary entertainment/legal framing and court-doc detail.",
        "verification": "searched_current_web_2026_05_02",
    },
    {
        "id": "bet_2026_04_24_jada_legal_bill",
        "title": "Jada Pinkett Smith Asks a Judge to Make Will Smith's Ex-Friend Pay Her $49,000 Legal Bill",
        "publisher": "BET",
        "url": "https://www.bet.com/article/gbq1p8/jada-pinkett-smith-asks-a-judge-to-make-will-smiths-ex-friend-pay-her-49000-legal-bill",
        "published": "2026-04-24",
        "used_for": "Cross-check of broad narrative and public-facing entertainment angle.",
        "verification": "searched_current_web_2026_05_02",
    },
    {
        "id": "tmz_2026_04_20_lively_40m",
        "title": "Blake Lively Tells Court Her 'Mean Girl' Label Cost Her $40.5 Million",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/20/blake-lively-tells-court-mean-girl-label-cost-her-millions/",
        "published": "2026-04-20",
        "used_for": "Candidate board comparison only; not selected.",
        "verification": "searched_current_web_2026_05_02",
    },
    {
        "id": "tmz_2026_04_17_kardashian_rayj_7m",
        "title": "Kim Kardashian & Kris Jenner's $7 Million Demand to Ray J Revealed in Court",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/17/kim-kardashian-kris-jenner-millions-demand-revealed-in-court/",
        "published": "2026-04-17",
        "used_for": "Candidate board comparison only; partially overlaps prior run 042.",
        "verification": "searched_current_web_2026_05_02",
    },
]

CANDIDATES = [
    {
        "rank": 1,
        "id": "jada_legal_fee_red_table",
        "topic": "Jada Pinkett Smith asks court to make Will Smith's former friend cover roughly $49K in legal fees",
        "premise": "A fictional red-table talk host turns a courtroom into a calm collections desk, sliding a giant legal-fee receipt across the table like therapy homework.",
        "source_ids": ["latimes_2026_04_24_jada_legal_fees", "tmz_2026_04_23_jada_49k", "bet_2026_04_24_jada_legal_bill"],
        "scores": {
            "famous_face": 9,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "absurd_quote_or_defense": 7,
            "brand_location_contrast": 10,
            "first_frame_visual_contradiction": 10,
            "freshness": 9,
            "repeat_risk_penalty": -1,
            "legal_taste_risk_penalty": -2,
            "total": 59,
        },
        "verdict": "WINNER",
    },
    {
        "rank": 2,
        "id": "lively_mean_girl_damages_store",
        "topic": "Blake Lively says a 'mean girl' label caused major reputational/business damages in ongoing litigation",
        "premise": "A luxury haircare aisle becomes a damages calculator where every bottle wears a court exhibit tag.",
        "source_ids": ["tmz_2026_04_20_lively_40m"],
        "scores": {
            "famous_face": 8,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "absurd_quote_or_defense": 9,
            "brand_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "freshness": 9,
            "repeat_risk_penalty": -2,
            "legal_taste_risk_penalty": -4,
            "total": 53,
        },
        "verdict": "HOLD: higher defamation/harassment sensitivity and more active litigation heat.",
    },
    {
        "rank": 3,
        "id": "kardashian_rayj_arbitration_vault",
        "topic": "Kim Kardashian/Kris Jenner and Ray J settlement/countersuit arbitration coverage",
        "premise": "A celebrity vault tries to seal itself while an arbitration clerk keeps printing receipts.",
        "source_ids": ["tmz_2026_04_17_kardashian_rayj_7m"],
        "scores": {
            "famous_face": 10,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "absurd_quote_or_defense": 7,
            "brand_location_contrast": 7,
            "first_frame_visual_contradiction": 8,
            "freshness": 8,
            "repeat_risk_penalty": -8,
            "legal_taste_risk_penalty": -5,
            "total": 44,
        },
        "verdict": "DISCARD: too close to existing Kardashian/Ray J/receipt runs and sex-tape context is taste-risky.",
    },
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if RUN_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing run directory: {RUN_DIR}")
    for src in (FIRST_FRAME_SRC, STORYBOARD_SRC):
        if not src.exists():
            raise SystemExit(f"Missing generated image source: {src}")

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

    first_frame_dst = PKG / "frames/gpt_image_2/first_frame_v01.png"
    storyboard_dst = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    shutil.copy2(FIRST_FRAME_SRC, first_frame_dst)
    shutil.copy2(STORYBOARD_SRC, storyboard_dst)

    write_text(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write_text(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", STORYBOARD_PROMPT)

    write_json(PKG / "research/sources.json", {
        "run_id": RUN_ID,
        "created_at": NOW,
        "research_intake_first": True,
        "sources": SOURCES,
        "fact_policy": [
            "Only sourced court-reporting claims are treated as factual context.",
            "All dialogue, props, red-table court staging, and collections-desk behavior are fictional satire.",
            "No exact real-person likenesses, logos, official court seals, or invented private documents should be used downstream.",
        ],
    })

    write_text(PKG / "research/last30days_report.md", f"""
# Last 30 Days Research Intake - Run {RUN_ID}

Created: {NOW}

## Research Query / Topic

Fresh current-culture scan for entertainment legal conflicts with famous faces, receipt math, public ego, and strong first-frame visual contradiction.

Queries included:
- `Jada Pinkett Smith Bilaal Salaam $49,000 legal fees anti-SLAPP April 2026`
- `May 2026 entertainment lawsuit absurd quote celebrity brand controversy`
- `April 2026 celebrity lawsuit $7 million Kardashian weird court source`
- `Blake Lively mean girl label cost millions court April 2026`

## Fresh Source Context

The winning source lane is April 2026 reporting that Jada Pinkett Smith asked a court to require Bilaal Salaam, described in coverage as Will Smith's former friend, to cover roughly $49K in legal fees after she prevailed on an anti-SLAPP motion against portions of his lawsuit. The case is still active, and the factory should treat the fee request as a legal motion/request, not a final moral verdict.

The comedic object is not the lawsuit itself. The comedic object is the surreal class contrast: a celebrity talk-table confessional grammar becoming a sterile courthouse collections desk with an itemized receipt.

## Candidate Scan

1. Jada legal-fee red table - strongest because it joins famous face, former-friend conflict, a concrete dollar amount, court paperwork, and an instantly recognizable but fictionalizable red-table visual.
2. Blake Lively "mean girl" damages - strong phrase and enormous damages math, but higher sensitivity and more active litigation risk.
3. Kardashian/Ray J settlement/arbitration vault - strong receipts, but repeats prior Kardashian/Ray J/receipt territory and carries unnecessary sex-tape taste drag.

## Verification Notes

- Verified source context comes from current web search on 2026-05-02.
- The package does not invent private facts, screenshots, or court outcomes.
- The visual execution must fictionalize likenesses and marks.
""")

    write_text(PKG / "research/source_notes.md", """
# Source Notes

Primary source spine: LA Times reporting on the April 2026 legal-fee motion.

Secondary cross-check: TMZ and BET entertainment/legal coverage.

Unverified or avoided:
- No claim that any party has been finally ordered to pay the requested amount unless future court records confirm it.
- No invented private conversations, memoir pages, threats, medical records, or sealed settlement terms.
- No real Red Table Talk logo, network logo, court seal, or exact celebrity likeness.
""")

    write_json(PKG / "strategy/candidate_board.json", {
        "run_id": RUN_ID,
        "created_at": NOW,
        "board_method": "research_intake_first_current_web_scan_then_scored_candidates",
        "scoring_scale": "1-10 plus negative risk/repeat penalties",
        "candidates": CANDIDATES,
        "winner": "jada_legal_fee_red_table",
    })

    write_text(PKG / "strategy/winner_decision.md", """
# Winner Decision

Winner: `jada_legal_fee_red_table`

Selected premise: A fictional celebrity red-table host transforms a courtroom into a calm legal-fee collections desk, sliding a giant `$49K` receipt across the table like therapy homework.

Why it wins:
- Concrete object: legal-fee receipt.
- Famous-public-family gravity without needing exact likeness.
- Strong first-frame contradiction: talk-show intimacy versus courthouse fee shifting.
- Good taste route: make the joke about process, props, and reputation theater, not about alleged private distress.

Rejected alternatives:
- Blake Lively damages store: strong but too litigation-sensitive for this cycle.
- Kardashian/Ray J arbitration vault: too repetitive and taste-risky.
""")

    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {
        "phase_minus_one_audit": {
            "source_target": "Jada legal-fee red table",
            "track_a_newsjack_velocity": {
                "active_trend_score": 9,
                "algorithmic_slipstream": "fresh entertainment legal coverage with familiar public figures and receipt math",
                "polarization_factor": 8,
                "track_a_total": 17,
                "track_a_verdict": "PASS",
            },
            "track_b_archetype_resonance": {
                "iconography_strength": 9,
                "stereotype_rigidity": "High",
                "subversion_potential": 10,
                "track_b_total": 19,
                "track_b_verdict": "PASS",
            },
            "system_verdict": {
                "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                "primary_vector": "TRACK_A_NEWSJACK",
                "urgency_class": "High",
                "strategic_directive": "Optimize for courtroom receipt visual and fictionalized red-table set contrast.",
            },
        }
    })

    write_json(PKG / "strategy/source_entropy_audit.json", {
        "source_entropy_audit": {
            "baseline_reality_check": "A legal-fee motion in an active entertainment lawsuit.",
            "detected_anomalies": [
                "A public family/confidant dispute is being converted into fee-shifting math.",
                "The reported amount is specific enough to become a prop.",
                "The Red Table cultural memory creates instant contrast with court procedure.",
            ],
            "native_entropy_score": 5,
            "subject_self_awareness": "deadpan",
            "comedic_vector_recommendation": "cringe_vector",
            "recommended_strategy": "micro_spotlight the legal-fee receipt and make every absurdity procedural, not chaotic",
        }
    })

    write_json(PKG / "strategy/humor_logic_bridge.json", {
        "hook": "What if a celebrity therapy table became a small-claims collections counter?",
        "truth": "Public image management often turns private mess into paperwork, motions, and invoices.",
        "absurd_transfer": "Replace emotional confession beats with court stamps and itemized billing.",
        "first_frame": "A giant LEGAL FEES receipt hangs over a red table inside a courtroom.",
        "escalation": [
            "receipt unrolls like a curtain",
            "host slides invoice calmly",
            "former-friend hands reach in and freeze",
            "judge treats the red table like official courtroom furniture",
        ],
        "taste_guardrails": [
            "No claims beyond sourced legal reporting.",
            "No exact likenesses.",
            "No jokes about alleged emotional distress, medical claims, or threats.",
        ],
    })

    write_json(PKG / "strategy/tribe_meta_score.json", {
        "run_id": RUN_ID,
        "tribe_score": {
            "relatability": 8,
            "share_trigger": 9,
            "caption_friction": 7,
            "visual_comprehension_under_one_second": 10,
            "comment_prompt_strength": 8,
            "total": 42,
        },
        "meta_score": {
            "freshness": 9,
            "specificity": 9,
            "asset_reusability": 8,
            "franchise_expandability": 8,
            "total": 34,
        },
    })

    write_json(PKG / "strategy/risk_taste_score.json", {
        "risk_taste_score": {
            "legal_defamation_risk": 6,
            "likeness_rights_risk": 7,
            "brand_logo_risk": 4,
            "harassment_or_private_fact_risk": 5,
            "mitigations": [
                "fictionalize all faces and names in images",
                "use generic prop text only",
                "state clearly that fee payment is requested in a motion, not adjudicated unless verified",
                "keep humor on paperwork and celebrity process theater",
            ],
            "final_taste_verdict": "PROCEED_WITH_FICTIONALIZED_VISUALS",
        }
    })

    write_text(PKG / "strategy/franchise_decision.md", """
# Franchise Decision

Decision: `PROCEED_AS_RECEIPT_COURT_EPISODE`

Franchise lane: Celebrity Receipt Court.

Reusable template:
- famous public dispute
- one concrete invoice/dollar prop
- sterile courtroom procedure
- luxury/therapy/talk-show setting invaded by billing mechanics

This should not become a direct impersonation lane. It should remain a prop-driven satire lane.
""")

    write_json(PKG / "chai/chai_shot_specs.json", {
        "run_id": RUN_ID,
        "shots": [
            {
                "shot_id": "shot_001",
                "duration_seconds": 6,
                "subject": "fictional cropped-hair celebrity host at red courtroom table",
                "scene": "Hollywood courtroom crossed with daytime talk-show set",
                "motion": "slow push-in as receipt unrolls and invoice slides forward",
                "spatial": "receipt foreground, red table midground, judge bench background",
                "camera": "24mm low-angle vertical, rack focus from receipt to host",
                "critique": "Must read as fictional satire, not real news photo.",
                "revision": "If likeness gets too exact, regenerate with more stylized fictional host and less facial specificity.",
            }
        ],
    })

    shot = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "title": "Legal Fee Red Table",
        "no_video_generation": True,
        "source_context": "April 2026 reporting on Jada Pinkett Smith legal-fee motion",
        "visual_prompt": FIRST_FRAME_PROMPT,
        "negative_prompt": "exact celebrity likeness, real logos, official court seal, real news photo, defamatory text, private documents",
        "duration_seconds": 6,
        "camera_plan": ["receipt foreground insert", "low push-in", "rack focus to fictional host", "judge reaction hold"],
    }
    write_json(PKG / "scene_json/shot_0001.json", shot)
    write_json(PKG / "scene_json/shot_001.json", shot)

    write_json(PKG / "handoffs/closed_tool_handoff.json", {
        "run_id": RUN_ID,
        "status": "STILLS_READY_NO_VIDEO_GENERATED",
        "hard_stop": "Do not call any video generation tool without human approval.",
        "premise": CANDIDATES[0]["premise"],
        "first_frame_path": "frames/gpt_image_2/first_frame_v01.png",
        "shared_choices_path": "storyboards/shared_choices/shared_choices_v01.png",
        "source_guardrails": [
            "Represent this as satire inspired by public reporting.",
            "Do not assert final court outcome on the fee request.",
            "Avoid exact celebrity likenesses and all real logos.",
        ],
        "manual_next_step": "Human reviews stills, then manually chooses whether to hand off to a video tool.",
    })

    write_text(PKG / "handoffs/grok_agent_prompt.md", """
# Grok Agent Prompt

Do not generate video. Build only a closed-tool prep analysis from the provided stills and package.

Premise: A fictional celebrity red-table host turns a courtroom into a legal-fee collections desk, sliding a giant `$49K` receipt across a red table.

Guardrails:
- Fictionalized faces only.
- No real logos or court seals.
- No invented private facts.
- Phrase the real-world context as a reported legal-fee motion/request in an active case.

Needed output if a human later asks: 6-second and 10-second I2V shot prompts that preserve the first-frame composition.
""")

    write_text(PKG / "captions/instagram_caption.md", """
When the red table becomes the receipts table.

Reported context: a public legal-fee motion, not a final verdict. The short keeps the joke on paperwork, image management, and celebrity process theater.

#sgflix #celebritysatire #receiptcourt #aitools #visualstorytelling
""")

    write_text(PKG / "distribution/post_plan.md", """
# Distribution Post Plan

Primary format: 9:16 Reel/TikTok/Short after manual video approval.

Hook overlay:
`WHEN THE RED TABLE SENDS AN INVOICE`

Risk-safe caption angle:
- Say "reported legal-fee motion" instead of implying final payment order.
- Avoid naming private allegations in the on-screen joke.
- Use the generated fictional still, not real press imagery.

Post-ready exports: none. No video was generated.
""")

    write_text(PKG / "skool/case_study.md", """
# Skool Case Study - Run 050

Lesson: Turn a volatile celebrity legal story into a safer visual satire by moving the joke from personal allegations to public paperwork.

Factory pattern:
1. Start with current source intake.
2. Score multiple candidates before selecting.
3. Extract one concrete prop: `$49K` legal-fee receipt.
4. Build a visual contradiction: therapy-talk table inside courthouse collections.
5. Create stills and handoffs only; leave video generation for human approval.

Reusable exercise: Find one current story where a private/public conflict can become an oversized bureaucratic prop.
""")

    write_json(PKG / "manifests/asset_manifest.json", {
        "run_id": RUN_ID,
        "created_at": NOW,
        "assets": [
            {
                "path": "frames/gpt_image_2/first_frame_v01.png",
                "type": "generated_first_frame",
                "source": str(FIRST_FRAME_SRC),
                "status": "usable",
            },
            {
                "path": "frames/gpt_image_2/first_frame_v01_prompt.md",
                "type": "prompt",
                "status": "complete",
            },
            {
                "path": "storyboards/shared_choices/shared_choices_v01.png",
                "type": "generated_shared_choices_board",
                "source": str(STORYBOARD_SRC),
                "status": "usable_with_human_text_review",
            },
            {
                "path": "storyboards/shared_choices/shared_choices_v01_prompt.md",
                "type": "prompt",
                "status": "complete",
            },
        ],
        "preservation": "Original generated images under CODEX_HOME were left in place; package copies were created.",
    })

    write_text(PKG / "qc/first_frame_v01_qc.md", """
# First Frame QC

Status: `PASS_WITH_HUMAN_REVIEW`

Checks:
- Project-bound PNG saved in package.
- Visual contradiction reads: red talk table inside courtroom/collections scene.
- Prop text is limited to legal-fee/fee amount concepts.
- Image is fictionalized and should not be used as a real news image.

Known risks:
- Human reviewer should verify no face reads too close to Jada Pinkett Smith, Will Smith, or Bilaal Salaam.
- Downstream tools must preserve fictionalization.
""")

    write_text(PKG / "qc/shared_choices_v01_qc.md", """
# Shared Choices QC

Status: `PASS_WITH_TEXT_REVIEW`

Checks:
- Project-bound director-bible board saved in package.
- Board includes character, props, palette, set design, blocking, storyboard panels, camera/lens/movement, lighting/style, visual rules, and production notes.
- No video generation was requested.

Known risks:
- Small image text may be imperfect; use the board as director reference, not as a publish-facing graphic.
- Human reviewer should verify no exact real-person likeness or real logo appears before closed-tool handoff.
""")

    package = {
        "run_id": RUN_ID,
        "slug": SLUG,
        "created_at": NOW,
        "status": "STILLS_READY_NO_VIDEO_GENERATED",
        "research_query_topic": "Jada Pinkett Smith legal-fee motion and adjacent entertainment legal receipt stories",
        "selected_premise": CANDIDATES[0]["premise"],
        "candidate_board": CANDIDATES,
        "required_artifacts": "complete",
        "generated_stills": [
            "frames/gpt_image_2/first_frame_v01.png",
            "storyboards/shared_choices/shared_choices_v01.png",
        ],
        "missing_files": [],
        "post_ready_exports": [],
        "qc_failures": [],
        "high_risk_issues": [
            "active legal dispute; avoid implying final court outcome",
            "avoid exact celebrity likenesses",
            "avoid real logos and official court seals",
            "do not invent private facts or documents",
        ],
        "next_human_action": "Review first-frame and Shared Choices PNGs, then approve or request a repair prompt before any manual video workflow.",
    }
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", package)

    readme = f"""
# RUN {RUN_ID} MASTER PACKAGE - Jada Legal Fee Red Table

Status: `STILLS_READY_NO_VIDEO_GENERATED`

Research query/topic: Jada Pinkett Smith legal-fee motion and adjacent entertainment legal receipt stories.

Selected premise: {CANDIDATES[0]["premise"]}

Score summary:
- Jada legal-fee red table: 59 - winner.
- Lively mean-girl damages store: 53 - hold for risk.
- Kardashian/Ray J arbitration vault: 44 - discard for repeat/taste risk.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing files: none from required minimum package list.

Post-ready exports: none. No video footage was generated.

QC failures: none blocking. Human review should check fictional likeness and small storyboard text.

High-risk issues:
- The legal-fee story is an active legal dispute.
- Do not imply a final court outcome unless future court records confirm it.
- Avoid exact Jada Pinkett Smith, Will Smith, or Bilaal Salaam likenesses.
- Avoid real Red Table Talk branding, network logos, and official court seals.

Exact next human action: Review the first-frame and Shared Choices PNGs, then approve or request a repair prompt before any manual video workflow.
"""
    write_text(PKG / "README.md", readme)
    write_text(PKG / "FACTORY_RUN_STATUS.md", f"""
# Factory Run Status - Run {RUN_ID}

Status: `STILLS_READY_NO_VIDEO_GENERATED`

Completed:
- fresh research intake before premise selection
- current-source candidate board and score-first winner selection
- strategy audits and scoring files
- CHAI and scene JSON handoffs
- first-frame prompt plus generated PNG
- Shared Choices prompt plus generated PNG
- caption, distribution, Skool, manifest, and QC files

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: none blocking. Human review should check fictional likeness and small storyboard text.

High-risk issues:
- active legal dispute; avoid implying a final court outcome
- avoid exact celebrity likenesses
- avoid real logos and official court seals
- do not invent private facts or documents

Exact next human action: Review both still PNGs, then approve or request a GPT Image 2 repair pass before any manual video workflow.
""")

    print(PKG)


if __name__ == "__main__":
    main()
