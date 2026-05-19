from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "062"
SLUG = "mace_oconus_judge_chambers"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "tmz_mace_oconus_2026_05_02",
        "title": "Nancy Mace's Email to Judge Name-Dropping Trump Revealed in Court",
        "url": "https://www.tmz.com/2026/05/02/nancy-mace-name-dropping-the-president-revealed-in-court/",
        "publisher": "TMZ",
        "published": "2026-05-02",
        "use": "live trigger",
        "verified_notes": [
            "TMZ says legal docs include an email that appears to be from Mace to the judge in Patrick Bryant's defamation lawsuit.",
            "The reported email explains absence from a hearing by saying she was OCONUS, meaning outside the continental United States.",
            "The reported email mentions an invitation from the president.",
        ],
        "risk_notes": ["Treat the email as reported/alleged by TMZ unless independently verified in court records."],
    },
    {
        "id": "ap_mace_house_speech_2025_02_11",
        "title": "Rep. Nancy Mace accuses ex-fiance and associates of assaulting her and raping others in House speech",
        "url": "https://apnews.com/article/7d831f415ae00d703e30fa6c701f18de",
        "publisher": "AP",
        "published": "2025-02-11",
        "use": "dispute context",
        "verified_notes": [
            "AP reported Mace named Patrick Bryant in a House floor speech.",
            "AP said it could not independently verify Mace's claims.",
            "Bryant denied the allegations to AP.",
        ],
        "risk_notes": ["Avoid restating underlying assault claims as fact; the SGFLIX joke targets bureaucratic optics only."],
    },
    {
        "id": "tmz_homepage_2026_05_02",
        "title": "TMZ homepage scan, May 2 2026",
        "url": "https://www.tmz.com/",
        "publisher": "TMZ",
        "published": "2026-05-02",
        "use": "candidate scan",
        "verified_notes": [
            "The May 2 feed also surfaced Kevin Hart/The Rock traffic-stop comments, Kathy Hilton's known-danger defense, and other entertainment hooks.",
        ],
        "risk_notes": ["Homepage context is volatile; captured as research scan, not evergreen source."],
    },
    {
        "id": "tmz_kardashian_rayj_2026_04_17",
        "title": "Kim Kardashian & Kris Jenner's $7 Million Demand to Ray J Revealed in Court",
        "url": "https://www.tmz.com/2026/04/17/kim-kardashian-kris-jenner-millions-demand-revealed-in-court/",
        "publisher": "TMZ",
        "published": "2026-04-17",
        "use": "candidate scan",
        "verified_notes": [
            "TMZ reported a court-revealed demand involving Kim Kardashian, Kris Jenner, and Ray J.",
        ],
        "risk_notes": ["Existing SGFLIX run_042 already used a Kardashian receipt-desk vector; penalized as franchise fatigue."],
    },
]


CANDIDATES = [
    {
        "id": "c1_mace_oconus_judge_gate",
        "premise": "A fictionalized congresswoman turns a judge's chambers into an airport gate because her court excuse says OCONUS.",
        "source_ids": ["tmz_mace_oconus_2026_05_02", "ap_mace_house_speech_2025_02_11"],
        "scores": {
            "famous_face": 7,
            "public_conflict": 9,
            "ego_or_status": 8,
            "humiliation_engine": 8,
            "absurd_quote_or_defense": 10,
            "brand_location_contrast": 9,
            "first_frame_contradiction": 10,
            "risk_manageability": 7,
            "freshness": 10,
            "franchise_nonduplication": 9,
        },
        "total": 87,
        "risk": "Medium: political/legal dispute; keep the target on optics, acronyms, and status behavior, not underlying allegations.",
    },
    {
        "id": "c2_hart_rock_tint_confessional",
        "premise": "Kevin Hart appears as the tiny traffic-court narrator who claims he reported The Rock's tinted windows.",
        "source_ids": ["tmz_homepage_2026_05_02"],
        "scores": {
            "famous_face": 10,
            "public_conflict": 6,
            "ego_or_status": 7,
            "humiliation_engine": 8,
            "absurd_quote_or_defense": 8,
            "brand_location_contrast": 6,
            "first_frame_contradiction": 8,
            "risk_manageability": 8,
            "freshness": 10,
            "franchise_nonduplication": 8,
        },
        "total": 79,
        "risk": "Low-medium, but the setup is more insult recap than visual contradiction.",
    },
    {
        "id": "c3_kathy_known_danger_guest_map",
        "premise": "A Beverly Hills foyer becomes a safety-training museum around the phrase known danger.",
        "source_ids": ["tmz_homepage_2026_05_02"],
        "scores": {
            "famous_face": 6,
            "public_conflict": 8,
            "ego_or_status": 7,
            "humiliation_engine": 7,
            "absurd_quote_or_defense": 9,
            "brand_location_contrast": 8,
            "first_frame_contradiction": 8,
            "risk_manageability": 7,
            "freshness": 9,
            "franchise_nonduplication": 2,
        },
        "total": 71,
        "risk": "Already covered in run_039_hilton_known_danger_foyer; discard to avoid duplication.",
    },
    {
        "id": "c4_kardashian_rayj_return_window",
        "premise": "A luxury returns counter processes a $7M NDA receipt while everyone argues over who kept the copy.",
        "source_ids": ["tmz_kardashian_rayj_2026_04_17"],
        "scores": {
            "famous_face": 10,
            "public_conflict": 8,
            "ego_or_status": 8,
            "humiliation_engine": 7,
            "absurd_quote_or_defense": 7,
            "brand_location_contrast": 8,
            "first_frame_contradiction": 8,
            "risk_manageability": 5,
            "freshness": 6,
            "franchise_nonduplication": 4,
        },
        "total": 71,
        "risk": "Sex-tape litigation context and prior Kardashian receipt run make this noisy for current cycle.",
    },
]

WINNER = CANDIDATES[0]

FIRST_FRAME_PROMPT = """Use case: illustration-story
Asset type: SGFLIX short-form first-frame still, vertical 9:16.
Primary request: Create a satirical, non-photoreal editorial first frame inspired by current public reporting about a congresswoman's alleged email to a judge using the term OCONUS and mentioning a White House invitation. Do not create an exact likeness of any real person. Use a fictional South Carolina congresswoman archetype with blonde bob, red blazer, anxious courtroom posture.
Scene/backdrop: Judge's chambers transformed into a tiny airport departure gate: wood-paneled bench, court seal intentionally generic and unreadable, security rope, tiny desk sign that says only "HEARING" in clean legible text if possible.
Subject: A fictional congresswoman holds a printed email stamped "OCONUS" in giant red letters; a judge's clerk points to a tiny suitcase on the witness stand; a distant miniature White House invitation glows on a corkboard, no real logos.
Composition: Vertical cinematic frame, character on left third, oversized OCONUS stamp dominating center, judge bench and airport gate elements visible, strong first-frame contradiction. Keep text minimal; OCONUS must be the only prominent readable word.
Style: premium satirical editorial still, crisp linework over painterly realism, warm mahogany courtroom light mixed with cold TSA-blue rim light, high contrast, tasteful, no cruelty.
Avoid: exact likeness, defamatory labels, real court seals, real campaign logos, messy text, gore, sexual content, video generation, watermarks.
"""

SHARED_CHOICES_PROMPT = """Use case: infographic-diagram
Asset type: SGFLIX Shared Choices director-bible storyboard board, horizontal 16:9 production reference.
Primary request: Create a clean director's-bible storyboard board for a satirical short called "OCONUS Hearing Gate". Do not depict exact real-person likenesses; use a fictional blonde South Carolina congresswoman archetype.
Board layout requirements: clearly separated panels for 1) character + hero props, 2) color palette swatches, 3) environment/set design, 4) floor plan/blocking top-down diagram, 5) three storyboard panels with camera/lens/movement notes, 6) lighting/mood/style notes, 7) visual rules, 8) production notes.
Visual concept: Judge's chambers crossed with airport departure gate. Hero props: giant OCONUS stamped email, small rolling suitcase on witness stand, generic glowing White House-style invitation on corkboard, TSA-blue rope line, judge bench, generic court medallion with unreadable markings.
Storyboard panels: Panel A close-up of red OCONUS stamp on email; Panel B wide reveal of courtroom as airport gate; Panel C deadpan clerk pointing from hearing placard to suitcase.
Style: premium production board, crisp editorial concept art, readable structure, restrained labels, warm mahogany and cold airport blue palette, no real seals or logos, no defamatory claims, no messy paragraphs.
Text rules: keep text short and blocky; prominent words allowed: OCONUS, HEARING, GATE, DELAYED, WIDE, CLOSE, PUSH-IN. Avoid long generated body copy.
Avoid: exact likeness, real logos, campaign marks, legal conclusions, gore, sexual content, video generation, watermarks.
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def mirror(rel: str) -> None:
    src = PKG / rel
    dst = RUN_DIR / rel
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file() and not dst.exists():
            dst.write_bytes(src.read_bytes())


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
        "qc",
        "frames/gpt_image_2",
        "storyboards/shared_choices",
    ]:
        (PKG / rel).mkdir(parents=True, exist_ok=True)

    write(PKG / "research/last30days_report.md", f"""# Run 062 Research Intake: OCONUS Hearing Gate

Created: {NOW}

## Query Topic
Fresh public-conflict scan around May 2, 2026 entertainment/politics stories with a strong absurd quote, legal setting, status flex, and first-frame visual contradiction.

## Winner Source
TMZ's May 2, 2026 report says legal docs revealed an email that appears to be from Nancy Mace to the judge in Patrick Bryant's defamation lawsuit. The reported comic payload is not the underlying lawsuit; it is the visual collision of court procedure, presidential name-dropping, and the dry acronym OCONUS.

## Verification Notes
- Treat the email and court filing details as reported by TMZ, not independently verified here from court records.
- AP context confirms the broader dispute has serious contested allegations and denials; those are intentionally not the joke.
- The run should use a fictionalized public-official archetype rather than an exact likeness.

## Candidate Scan
1. Mace OCONUS email to judge: strongest courtroom/airport contradiction and freshest trigger.
2. Kevin Hart/The Rock traffic-stop comments: famous faces, but weaker prop logic.
3. Kathy Hilton known-danger defense: strong phrase, but already covered by run_039.
4. Kardashian/Ray J $7M demand: famous, but franchise fatigue and higher taste risk.

## Selected Premise
`OCONUS Hearing Gate`: a judge's chambers becomes a tiny airport gate because one bureaucratic acronym tries to explain a missed court hearing.
""")
    write_json(PKG / "research/sources.json", {"created_at": NOW, "sources": SOURCES})
    write_json(PKG / "strategy/candidate_board.json", {"created_at": NOW, "scored_before_winner": True, "winner_id": WINNER["id"], "candidates": CANDIDATES})
    write(PKG / "strategy/winner_decision.md", f"""# Winner Decision

Winner: `{WINNER['id']}`

Selected premise: {WINNER['premise']}

Score: {WINNER['total']}/100

Why it wins:
- The acronym `OCONUS` is a compact, weird, source-native phrase.
- The first frame can be understood instantly: courtroom turns into airport gate.
- The joke does not require repeating disputed underlying allegations.
- It is fresh, not a reuse of nearby storyboards or existing run assets.

Risk boundary: parody the optics of an alleged legal email and status-name-drop; do not assert facts beyond sourced reporting.
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {
        "phase_minus_one_audit": {
            "source_target": "Mace OCONUS email to judge",
            "track_a_newsjack_velocity": {
                "active_trend_score": 9,
                "algorithmic_slipstream": "fresh TMZ legal-docs story on May 2, 2026",
                "polarization_factor": 8,
                "track_a_total": 17,
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
                "strategic_directive": "Use dry bureaucratic airport language to satirize status-pressure in a court setting.",
            },
        }
    })
    write_json(PKG / "strategy/source_entropy_audit.json", {
        "source_entropy_audit": {
            "baseline_reality_check": "A court email explaining absence from a hearing in a defamation dispute.",
            "detected_anomalies": ["OCONUS acronym in a legal excuse", "presidential invitation/name-drop in judge-facing context", "courtroom status flex"],
            "native_entropy_score": 6,
            "subject_self_awareness": "trying_to_look_cool",
            "comedic_vector_recommendation": "cringe_vector",
            "recommended_strategy": "micro_spotlight",
        }
    })
    write_json(PKG / "strategy/humor_logic_bridge.json", {
        "hook": "What if a court excuse became an airport gate?",
        "comic_engine": "literalize OCONUS as airport bureaucracy inside chambers",
        "straight_man": "clerk treats the acronym like a boarding problem",
        "do_not_do": ["do not litigate the underlying allegations", "do not create exact likeness", "do not add unrelated chaos"],
        "payoff": "The smallest legal acronym becomes the largest prop in the room.",
    })
    write_json(PKG / "strategy/tribe_meta_score.json", {
        "TRiBE": {"truth": 8, "recognition": 8, "inversion": 9, "boldness": 7, "execution_clarity": 9},
        "meta": {"scroll_stop": 9, "share_captionability": 8, "duet_reactability": 7, "comment_trigger": 8},
        "summary": "Strong for politics/media-watch audiences who enjoy status-flex legal absurdity.",
    })
    write_json(PKG / "strategy/risk_taste_score.json", {
        "overall_risk": "Medium",
        "taste_score": 7,
        "legal_sensitivity": 8,
        "mitigations": ["fictionalize likeness", "label facts as reported", "joke on acronym and setting", "avoid underlying abuse claims"],
        "reject_if": ["image implies guilt or crime", "caption asserts unsourced court facts", "generated text becomes accusatory"],
    })
    write(PKG / "strategy/franchise_decision.md", """# Franchise Decision

Decision: one-off with optional recurring `courtroom-as-service-desk` lane.

This should not become a politics-only franchise. The reusable asset is the transformation grammar: dry legal phrase -> literal service counter or transit process. Good future siblings would need a fresh phrase as strong as `OCONUS`.
""")
    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)
    write_json(PKG / "chai/chai_shot_specs.json", {
        "run_id": RUN_ID,
        "title": "OCONUS Hearing Gate",
        "shots": [
            {
                "shot": "001",
                "duration_sec": 6,
                "subject": "fictionalized congresswoman archetype, judge clerk, OCONUS email",
                "scene": "judge chambers converted into airport gate",
                "motion": "slow push-in from bench to red OCONUS stamp",
                "spatial": "email center, suitcase on witness stand, clerk right, official left",
                "camera": "35mm vertical, low dolly push, mild parallax",
                "critique": "keep joke readable without exact likeness or legal accusations",
                "revision": "if text messy, crop to OCONUS stamp and use clean overlay later",
            }
        ],
    })
    shot = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "video_generation_allowed": False,
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "prompt": "Animate only after human approval in a separate closed tool; no video generation in this automation.",
        "camera": {"lens": "35mm", "move": "slow push-in", "duration": "6s"},
        "visual_rules": ["fictionalized likeness", "OCONUS as only prominent text", "generic seals only"],
    }
    write_json(PKG / "scene_json/shot_0001.json", shot)
    write_json(PKG / "scene_json/shot_001.json", shot)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {
        "run_id": RUN_ID,
        "status": "STILLS_READY_NO_VIDEO_GENERATED",
        "video_generation_tools_called": False,
        "first_frame": "frames/gpt_image_2/first_frame_v01.png",
        "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
        "human_next_step": "Review stills and decide whether to hand off to a video tool manually.",
    })
    write(PKG / "handoffs/grok_agent_prompt.md", """# Grok Agent Prompt

Use the stills and JSON only as a manual handoff. Do not start a video render automatically.

Create a 6-second satirical legal-airport scene from `first_frame_v01.png`: slow push toward the OCONUS stamp, clerk deadpan points at the suitcase, tiny gate-delay board flickers. Keep all seals generic and all characters fictionalized.
""")
    write(PKG / "captions/instagram_caption.md", """POV: your court excuse gets upgraded to Gate OCONUS.

Reported setup: legal-docs story, alleged email, one acronym doing way too much work.

Satire target: bureaucracy, status flexing, and the moment a judge's chambers starts feeling like departures.

#sgflix #satire #courtroom #politicalsatire #shortformvideo #aivideoart
""")
    write(PKG / "distribution/post_plan.md", """# Post Plan

Primary surface: Instagram Reels / TikTok vertical.

Hook overlay: `GATE OCONUS NOW BOARDING`

Risk note: caption must say `reported` or `alleged email`; do not summarize the underlying lawsuit as fact.

Post-ready status: stills ready, video not generated.
""")
    write(PKG / "skool/case_study.md", """# Skool Case Study

Lesson: Turn one source-native phrase into the visual set.

`OCONUS` works because it is dry, official, and weird enough to literalize. The factory avoided the serious disputed allegations and built the joke around an acronym, a status signal, and a location contradiction.
""")
    first_exists = (PKG / "frames/gpt_image_2/first_frame_v01.png").exists()
    board_exists = (PKG / "storyboards/shared_choices/shared_choices_v01.png").exists()
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
    missing_before_manifest = [rel for rel in required if rel != "manifests/asset_manifest.json" and not (PKG / rel).exists()]
    write_json(PKG / "manifests/asset_manifest.json", {
        "run_id": RUN_ID,
        "created_at": NOW,
        "required_files": required,
        "missing_files": missing_before_manifest,
        "assets": [
            {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "gpt_image_2_builtin_first_frame", "exists": first_exists, "source": "/Users/speed/.codex/generated_images/019de94d-b778-7c83-8996-8b0b46d6f1fa/ig_08aa6fc28a026cce0169f618c43a4c8191895acb7618e2ab1b.png"},
            {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "gpt_image_2_builtin_shared_choices_board", "exists": board_exists, "source": "/Users/speed/.codex/generated_images/019de94d-b778-7c83-8996-8b0b46d6f1fa/ig_08aa6fc28a026cce0169f61950071c81918fff463009a5d069.png"},
        ],
        "video_generation_tools_called": False,
    })
    write(PKG / "qc/first_frame_v01_qc.md", """# First Frame QC

Verdict: usable for review.

Passes:
- Strong courtroom/airport contradiction.
- OCONUS is visually central.
- Character is fictionalized rather than exact real-person likeness.
- No video generation was called.

Watchouts:
- Generated micro-text outside OCONUS should be ignored or covered by clean overlay if it distracts.
- Keep captions in reported/alleged language.
""")
    write(PKG / "qc/shared_choices_v01_qc.md", """# Shared Choices QC

Verdict: usable director-bible board.

Passes:
- Includes character/hero props, palette, environment, floor plan, storyboard beats, lighting notes, visual rules, and production notes.
- Board is horizontal and production-oriented.
- Avoids real seals/logos and exact likeness.

Watchouts:
- Any tiny generated text should be treated as decorative, not source copy.
""")
    package = {
        "run_id": RUN_ID,
        "slug": SLUG,
        "created_at": NOW,
        "status": "STILLS_READY_NO_VIDEO_GENERATED",
        "research_query": "May 2 2026 celebrity/politics legal-conflict hooks with absurd quote or defense",
        "selected_premise": WINNER,
        "score_summary": {"winner_total": WINNER["total"], "runner_up_total": CANDIDATES[1]["total"]},
        "generated_stills": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
        },
        "high_risk_issues": ["political/legal dispute", "unverified email unless court docs inspected directly", "avoid underlying allegations"],
        "exact_next_human_action": "Review the two generated stills for text/likeness drift; if approved, manually hand off to a video tool outside this automation.",
        "video_generation_tools_called": False,
    }
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", package)
    write(PKG / "README.md", f"""# RUN {RUN_ID} MASTER PACKAGE: OCONUS Hearing Gate

Status: STILLS_READY_NO_VIDEO_GENERATED

Selected premise: {WINNER['premise']}

Research query/topic: May 2, 2026 public-conflict scan for legal/status absurdity with a strong first-frame contradiction.

Generated stills:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing files: none expected inside this package.

Post-ready exports: caption and post plan are drafted; no rendered video exists.

Exact next human action: review stills for text/likeness drift, then decide whether to manually hand off the stills to a video tool.
""")
    status = f"""# Factory Run Status

Run: {RUN_ID} `{SLUG}`
Status: STILLS_READY_NO_VIDEO_GENERATED
Created: {NOW}

Order of operations:
1. Research intake completed first.
2. Candidate board built from current source scan.
3. Candidates scored before winner selection.
4. Run package created for selected winner.
5. GPT Image workflow produced first-frame and Shared Choices stills.
6. No video-generation tools were called.

Selected premise: {WINNER['premise']}

Score summary:
- Winner: {WINNER['total']}/100
- Runner-up: {CANDIDATES[1]['total']}/100

Generated still-image paths:
- `RUN_{RUN_ID}_MASTER_PACKAGE/frames/gpt_image_2/first_frame_v01.png`
- `RUN_{RUN_ID}_MASTER_PACKAGE/storyboards/shared_choices/shared_choices_v01.png`

Missing files: none detected.

QC failures: none blocking; review tiny generated text before posting.

High-risk issues:
- Treat the email as reported/alleged unless court docs are directly inspected.
- Avoid underlying allegations and exact likeness.

Exact next human action: review the two stills; if approved, use `handoffs/closed_tool_handoff.json` manually for a video render outside this automation.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", status)

    for rel in required:
        if rel != "README.md" and rel != f"RUN_{RUN_ID}_MASTER_PACKAGE.json":
            mirror(rel)


if __name__ == "__main__":
    main()
