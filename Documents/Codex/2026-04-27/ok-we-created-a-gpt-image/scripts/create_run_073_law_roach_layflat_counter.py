from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN_ID = "073"
SLUG = "law_roach_layflat_counter"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

RESEARCH_QUERY = (
    "late-April/early-May 2026 entertainment, travel, and celebrity-status conflicts "
    "with public quote, brand contrast, absurd first-frame prop, and low-enough taste risk"
)

SOURCES = [
    {
        "id": "tmz_law_roach_delta",
        "title": "Law Roach Calls Out Delta Airlines After Not Getting His 'Lay Flat' Seat",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/21/law-roach-blasts-delta-airlines/",
        "published": "2026-04-21",
        "verified_facts": [
            "TMZ reported Law Roach publicly criticized Delta after a New York-to-Los Angeles flight did not have the lay-flat seat he said he paid for.",
            "TMZ reported a Delta spokesperson attributed the issue to a last-minute aircraft change and said customer care had reached out.",
            "TMZ reported Roach referenced Delta 360 and Million Miler status.",
        ],
        "unverified_or_claim_framed": [
            "Exact compensation and private customer-service resolution are not verified in this package.",
            "The package does not assert wrongdoing beyond the public report and Delta's stated aircraft-change explanation.",
        ],
    },
    {
        "id": "tmz_huda_service",
        "title": "Huda Mustafa's BF Louis' Baby Mama Claims 'Love Island' Star Is 'Evading' Court Battle",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/28/huda-mustafa-bf-louis-ex-gf-claims-love-island-star-is-evading-court-battle/",
        "published": "2026-04-28",
        "verified_facts": [
            "TMZ reported a restraining-order service dispute involving Huda Mustafa and denials from her representative.",
        ],
        "unverified_or_claim_framed": ["All alleged threats and service-avoidance claims remain disputed."],
    },
    {
        "id": "tmz_rock_hart",
        "title": "Kevin Hart Calls Dwayne 'The Rock' Johnson A Piece of S*** After Traffic Stop",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/05/02/kevin-hart-jokingly-blasts-dwayne-johnson/",
        "published": "2026-05-02",
        "verified_facts": [
            "Search intake found a current TMZ item about Kevin Hart joking about Dwayne Johnson's traffic stop.",
        ],
        "unverified_or_claim_framed": [
            "Full article details were not used because the premise duplicates prior Rock tinted-window runs.",
        ],
    },
    {
        "id": "variety_swift_trademark",
        "title": "Taylor Swift Files to Trademark Her Voice and Likeness, Apparently to Protect Against AI Misuse",
        "publisher": "Variety Australia/New Zealand",
        "url": "https://au.variety.com/2026/music/news/taylor-swift-trademark-voice-likeness-ai-misuse-35964/",
        "published": "2026-04-28",
        "verified_facts": [
            "Variety reported Taylor Swift's company filed sound and visual trademark applications connected to voice and likeness protection.",
        ],
        "unverified_or_claim_framed": ["Rejected as too close to prior SGFLIX voice-vault runs."],
    },
    {
        "id": "tmz_cardi_tasha",
        "title": "Cardi B Seeks Sanctions Against Tasha K For Talking Offset & Stefon Diggs",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/10/cardi-b-wants-tasha-k-penalized-for-talking-offset-stefon-diggs/",
        "published": "2026-04-10",
        "verified_facts": [
            "TMZ reported Cardi B sought sanctions alleging NDA violations; Tasha K context remains litigation-framed.",
        ],
        "unverified_or_claim_framed": ["Rejected for higher legal risk and overlap with prior Cardi legal material."],
    },
]

CANDIDATES = [
    {
        "id": "law_roach_layflat_counter",
        "premise": "A fictional celebrity stylist arrives at a luxury airline complaint counter carrying a couture spine mannequin and a velvet 'lay-flat' seat that refuses to recline.",
        "source_ids": ["tmz_law_roach_delta"],
        "scores": {
            "famous_face_or_status_proxy": 7,
            "public_conflict": 8,
            "ego_status_pressure": 10,
            "humiliation_or_contradiction": 9,
            "absurd_quote_or_defense": 8,
            "brand_location_contrast": 10,
            "first_frame_clarity": 10,
            "risk_adjustment": 8,
        },
        "total": 70,
        "verdict": "WINNER",
    },
    {
        "id": "huda_process_server_pool_gate",
        "premise": "Reality-star luxury condo security turns a process server into a poolside clipboard ghost while every gate key is labeled 'proper mature way via legal team.'",
        "source_ids": ["tmz_huda_service"],
        "scores": {
            "famous_face_or_status_proxy": 5,
            "public_conflict": 8,
            "ego_status_pressure": 7,
            "humiliation_or_contradiction": 8,
            "absurd_quote_or_defense": 7,
            "brand_location_contrast": 8,
            "first_frame_clarity": 8,
            "risk_adjustment": 5,
        },
        "total": 56,
        "verdict": "REJECTED_LESS_FAME_HIGHER_PERSONAL_RISK",
    },
    {
        "id": "rock_tint_ticket_roast",
        "premise": "Kevin Hart audits a giant tinted-window ticket booth after The Rock's traffic stop.",
        "source_ids": ["tmz_rock_hart"],
        "scores": {
            "famous_face_or_status_proxy": 10,
            "public_conflict": 6,
            "ego_status_pressure": 8,
            "humiliation_or_contradiction": 9,
            "absurd_quote_or_defense": 9,
            "brand_location_contrast": 7,
            "first_frame_clarity": 9,
            "risk_adjustment": 3,
        },
        "total": 61,
        "verdict": "REJECTED_DUPLICATES_RUN_071_ROCK_TINT_TICKET_DISPATCH",
    },
    {
        "id": "swift_voice_trademark_lost_found",
        "premise": "A pop-star voice-tag lost-and-found counter files sound marks in velvet evidence drawers.",
        "source_ids": ["variety_swift_trademark"],
        "scores": {
            "famous_face_or_status_proxy": 10,
            "public_conflict": 6,
            "ego_status_pressure": 8,
            "humiliation_or_contradiction": 6,
            "absurd_quote_or_defense": 8,
            "brand_location_contrast": 8,
            "first_frame_clarity": 8,
            "risk_adjustment": 4,
        },
        "total": 58,
        "verdict": "REJECTED_DUPLICATES_PRIOR_SWIFT_VOICE_RUNS",
    },
    {
        "id": "cardi_nda_cat_mouse_counter",
        "premise": "A blogger's microphone keeps popping out of NDA mousetraps while a judge counts future sanctions tickets.",
        "source_ids": ["tmz_cardi_tasha"],
        "scores": {
            "famous_face_or_status_proxy": 8,
            "public_conflict": 9,
            "ego_status_pressure": 8,
            "humiliation_or_contradiction": 7,
            "absurd_quote_or_defense": 7,
            "brand_location_contrast": 6,
            "first_frame_clarity": 7,
            "risk_adjustment": 2,
        },
        "total": 54,
        "verdict": "REJECTED_HIGHER_LEGAL_RISK_AND_EXISTING_CARDI_LANE",
    },
]

FIRST_FRAME_PROMPT = """Use case: photorealistic-natural
Asset type: SGFLIX vertical first-frame still, 9:16.
Primary request: Create a satirical premium editorial first frame inspired by public reporting that celebrity stylist Law Roach criticized Delta after paying for a Delta One lay-flat seat that did not lie flat after an aircraft swap. Do not depict an exact likeness of Law Roach and do not use Delta logos or trademarks.
Scene/backdrop: an absurd luxury airline complaint counter inside a polished airport lounge, half couture atelier and half gate-service desk.
Subject: a fictional celebrity stylist archetype in sharp monochrome fashion, wearing oversized sunglasses, standing beside a velvet airline seat labeled only with generic pseudo-text, the seat is locked bolt-upright while a couture spine mannequin reclines perfectly on a luggage cart.
Hero props: a gold frequent-flyer medallion reading only "360 / MILLION MILER" in generic typography, a red aircraft-swap stamp, a tape measure checking the seat angle, a garment rack of immaculate suits, a tiny customer-care phone under a glass dome, boarding pass with unreadable pseudo-text.
Composition: vertical 9:16, low desk-height 28mm lens, the upright seat dominates foreground right, the stylist archetype foreground left, complaint counter and aircraft-swap placard behind glass. First read must be: luxury passenger promised flat seat, seat refuses to recline.
Lighting/style: premium satirical magazine photo, crisp airport fluorescents mixed with fashion-show rim light, colors of airline navy, warning red, brushed aluminum, runway cream, tailoring black, and electric customer-service blue. Polished but not glossy.
Text policy: no readable brand logos, no real airline marks, only short generic pseudo-labels. Avoid messy generated paragraphs.
Safety/guardrails: no exact public-figure likeness, no defamation, no claim that the airline intentionally harmed anyone, no real Delta logo, no aircraft crash imagery, no medical injury depiction, no video-generation language.
Negative prompt: distorted hands, real logos, exact celebrity face, fake news chyron, readable legal accusations, bokeh blobs, one-note blue palette, over-saturated terminal, watermark."""

SHARED_CHOICES_PROMPT = """Use case: productivity-visual
Asset type: SGFLIX Shared Choices director's-bible storyboard board, 16:9.
Primary request: Create a premium director's-bible board for the satire premise "Lay-Flat Complaint Counter" inspired by public reporting about a celebrity stylist's lay-flat airline seat complaint. Do not depict exact Law Roach likeness and do not use Delta branding.
Board contents: character canon for fictional celebrity stylist archetype, hero props, color palette, environment/set design, floor plan/blocking, six storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes.
Character/props: fictional stylist silhouette with oversized sunglasses and tailored black outfit; upright velvet premium seat; couture spine mannequin; generic frequent-flyer medallion; aircraft-swap stamp; garment rack; customer-care phone under glass; pseudo boarding pass.
Environment/set design: airport lounge complaint desk crossed with backstage fashion atelier; brushed aluminum counters, runway garment racks, velvet rope queue, generic gate monitor with unreadable pseudo-text.
Storyboard panels: 1) first-frame reveal of upright seat and reclining mannequin, 2) medallion slapped on counter, 3) red aircraft-swap stamp lands, 4) tape measure checks seat angle, 5) customer-care phone under glass rings, 6) stylist exits with the upright seat rolling behind like luggage.
Visual rules: no real airline logos, no exact celebrity likeness, no readable paragraphs, no medical injury gag, no video render request, satire targets status/service contradiction.
Style: clean premium production reference board, cinematic sketches mixed with prop-photo callouts, balanced palette swatches, neat layout, minimal pseudo-text only.
Negative prompt: real Delta mark, exact Law Roach face, dense illegible typography, distorted bodies, defamatory claims, aircraft emergency imagery, one-note blue palette, watermark."""


def write(rel: str, content: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_json(rel: str, data) -> None:
    write(rel, json.dumps(data, indent=2, ensure_ascii=True))


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
        f"""# Step 1: Research Intake - Run {RUN_ID}

Created: {NOW}

Research query/topic: {RESEARCH_QUERY}

Process note: this cycle began with current web/source intake. No local image, old storyboard, old handoff, or nearby asset was used to choose the premise.

Fresh intake summary:
- TMZ reported on April 21, 2026 that Law Roach criticized Delta after a New York-to-Los Angeles flight did not provide the lay-flat seat he said he paid for. TMZ also reported Delta's aircraft-swap explanation and outreach statement.
- TMZ reported on April 28, 2026 on a disputed Love Island service-of-process story involving Huda Mustafa, but that lane scored lower on fame and personal-risk grounds.
- Current search intake also surfaced a Kevin Hart / Dwayne Johnson traffic-stop joke, Taylor Swift voice/likeness trademark filings, and Cardi B/Tasha K sanctions reporting. These were scored but rejected for duplication or risk.

Selected winner after scoring: **Lay-Flat Complaint Counter**.

Fact guardrails:
- Treat the airline issue as a reported customer-service dispute, not proven intentional misconduct.
- Use a fictional celebrity-stylist archetype rather than an exact likeness.
- Avoid real airline logos, trademarks, aircraft emergencies, medical injury claims, or personal attacks.
""",
    )
    write_json("research/sources.json", {"created": NOW, "research_query": RESEARCH_QUERY, "sources": SOURCES})
    write_json(
        "strategy/candidate_board.json",
        {"created": NOW, "selection_order": "research_intake_then_scoring_then_winner_then_package", "candidates": CANDIDATES},
    )
    write(
        "strategy/winner_decision.md",
        f"""# Winner Decision - Run {RUN_ID}

Winner: **Lay-Flat Complaint Counter**.

Why it won: it is current, visual in one frame, lower-stakes than active harassment/abuse litigation, and has a clean SGFLIX contradiction: luxury service status meets a seat that physically refuses the promised posture. The joke can live in props, posture, and customer-service theater without inventing facts.

Score summary:
- Law Roach lay-flat counter: 70/80, winner.
- Rock tint ticket roast: 61/80, rejected as duplicate of prior Run 071 Rock tint lane.
- Swift voice trademark lost found: 58/80, rejected as duplicate of prior voice-vault lanes.
- Huda process-server pool gate: 56/80, rejected for lower fame and higher personal-life risk.
- Cardi NDA cat-mouse counter: 54/80, rejected for legal risk and existing Cardi lane.
""",
    )
    write_json(
        "strategy/phase_minus_one_worthiness_audit.json",
        {
            "phase_minus_one_audit": {
                "source_target": "Law Roach / Delta lay-flat seat complaint",
                "track_a_newsjack_velocity": {
                    "active_trend_score": 7,
                    "mainstream_recent_coverage": 7,
                    "duplicate_penalty": 0,
                    "track_a_total": 21,
                    "track_a_verdict": "PASS",
                },
                "track_b_archetype_resonance": {
                    "archetype": "luxury-status customer forced into ordinary discomfort",
                    "subversion_potential": 9,
                    "first_frame_native_absurdity": 10,
                    "track_b_total": 27,
                    "track_b_verdict": "PASS",
                },
                "system_verdict": {
                    "final_decision": "PROCEED",
                    "primary_vector": "TRACK_B_ARCHETYPE",
                    "strategic_directive": "Physicalize the seat posture contradiction and avoid real airline marks.",
                },
            }
        },
    )
    write_json(
        "strategy/source_entropy_audit.json",
        {
            "source_entropy_audit": {
                "baseline_reality_check": "Reported celebrity stylist customer-service complaint over a premium seat expectation after an aircraft swap.",
                "detected_anomalies": [
                    "a lay-flat product that cannot physically lie flat",
                    "elite frequent-flyer status colliding with ordinary upright posture",
                    "fashion-world perfection measured against airline equipment chaos",
                ],
                "native_entropy_score": 8,
                "recommended_strategy": "turn service downgrade into courtroom-style complaint counter and couture posture lab",
            }
        },
    )
    write_json(
        "strategy/humor_logic_bridge.json",
        {
            "hook": "The lay-flat seat has entered a not-guilty plea.",
            "setup": "A luxury stylist arrives with elite credentials and a seat that remains bolt upright.",
            "turn": "The couture spine mannequin gets the only fully reclined position in the room.",
            "button": "Customer care is under glass, ringing forever.",
            "guardrails": ["No exact likeness", "No real airline logos", "No claim of intentional harm"],
        },
    )
    write_json(
        "strategy/tribe_meta_score.json",
        {
            "tribe_meta_score": {
                "identity_snap": 8,
                "share_prompt": 8,
                "comment_fight_potential": 9,
                "remixability": 8,
                "visual_read": 10,
                "total": 43,
                "verdict": "STRONG",
            }
        },
    )
    write_json(
        "strategy/risk_taste_score.json",
        {
            "risk_taste_score": {
                "defamation_risk": "low-medium; keep to sourced complaint and Delta aircraft-swap explanation",
                "brand_trademark_risk": "medium; avoid Delta logos and exact trade dress",
                "identity_likeness_risk": "medium; use fictional stylist archetype",
                "taste_risk": "low; no severe allegations",
                "overall": "GO_WITH_BRAND_AND_LIKENESS_GUARDRAILS",
            }
        },
    )
    write("strategy/franchise_decision.md", "# Franchise Decision\n\nProceed. Reusable franchise shape: luxury-service complaint transformed into a bureaucratic prop lab.")

    shot = {
        "run_id": RUN_ID,
        "shot_id": "001",
        "title": "Lay-Flat Complaint Counter",
        "duration_seconds": 8,
        "subject": "fictional celebrity stylist archetype, upright premium seat, couture spine mannequin",
        "scene": "airport lounge complaint desk crossed with fashion atelier",
        "motion": "seat angle is measured while the aircraft-swap stamp lands and customer-care phone rings under glass",
        "spatial": "stylist foreground left, upright seat foreground right, complaint counter and garment racks mid/background",
        "camera": "vertical 9:16, 28mm low desk-height push-in",
        "critique": "must read as luxury seat-service satire, not a real ad or real airline accusation",
        "revision": "remove exact likeness, real logos, readable brand text, medical injury framing",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
    }
    write_json("chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [shot]})
    write_json("scene_json/shot_0001.json", shot)
    write_json("scene_json/shot_001.json", {**shot, "handoff_only_no_video_generation": True})
    write_json(
        "handoffs/closed_tool_handoff.json",
        {
            "run_id": RUN_ID,
            "hard_stop": "Do not generate video footage in this automation.",
            "premise": "Lay-Flat Complaint Counter",
            "first_frame_prompt_path": "frames/gpt_image_2/first_frame_v01_prompt.md",
            "storyboard_prompt_path": "storyboards/shared_choices/shared_choices_v01_prompt.md",
            "required_guardrails": ["fictional stylist archetype", "no real airline logos", "no exact Law Roach likeness", "no aircraft emergency imagery"],
        },
    )
    write(
        "handoffs/grok_agent_prompt.md",
        f"""# Grok Agent Prompt - Run {RUN_ID}

Do not generate video. Review and tighten the still-to-video handoff language for the SGFLIX premise `Lay-Flat Complaint Counter`.

Premise: a fictional celebrity stylist at a luxury airline complaint desk presents elite frequent-flyer credentials while a supposedly lay-flat seat remains bolt upright.

Guardrails: no exact Law Roach likeness, no Delta marks, no claim of intentional harm, preserve aircraft-swap/customer-service framing, and keep all text pseudo or generic.
""",
    )
    write(
        "captions/instagram_caption.md",
        """The seat said Delta One. The posture said jury duty.

Satire based on public reporting about a celebrity stylist's airline seat complaint and the airline's aircraft-swap explanation. No brand logo, no exact likeness, no video generated.

#SGFLIX #satire #travel #fashion #customerexperience #airportcomedy""",
    )
    write(
        "distribution/post_plan.md",
        """# Post Plan

Primary surface: Instagram Reels/TikTok after human still QC and separate approved render workflow.

Hook: `The lay-flat seat refused to testify.`

Do not auto-post. Human review must confirm no real airline marks, no exact likeness, and no readable/generated fake claims.
""",
    )
    write(
        "skool/case_study.md",
        """# Skool Case Study

Lesson: lower-stakes status conflicts can outperform heavier scandal when the first-frame contradiction is instantly legible. Here, elite customer status plus an upright premium seat gives a prop-native joke without needing to litigate personal allegations.
""",
    )
    write("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)
    write_json(
        f"RUN_{RUN_ID}_MASTER_PACKAGE.json",
        {
            "run_id": RUN_ID,
            "slug": SLUG,
            "status": "AWAITING_STILL_IMAGE_LOCALIZATION",
            "research_query": RESEARCH_QUERY,
            "selected_premise": "Lay-Flat Complaint Counter",
            "score_summary": {candidate["id"]: candidate["total"] for candidate in CANDIDATES},
            "video_generation": "not_requested",
            "post_ready_exports": [],
            "high_risk_issues": ["brand/trademark similarity", "exact public-figure likeness", "generated text artifacts"],
        },
    )
    write(
        "README.md",
        f"""# RUN {RUN_ID} MASTER PACKAGE - Lay-Flat Complaint Counter

Status: awaiting generated still localization.

Research query: {RESEARCH_QUERY}

Selected premise: a fictional celebrity stylist at a luxury airline complaint counter confronts a premium seat that refuses to lie flat while a couture spine mannequin gets the perfect recline.

Required stills:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Next human action: after generated PNGs are localized, inspect for logo leakage, exact likeness, and messy text before any separate video handoff.
""",
    )
    status_text = f"""# Factory Run Status - Run {RUN_ID}

Status: AWAITING_STILL_IMAGE_LOCALIZATION

Created: {NOW}

Research topic: {RESEARCH_QUERY}

Candidate board: 5 current-source candidates scored before winner selection.

Selected premise: Lay-Flat Complaint Counter.

Winner score: 70/80.

Generated still paths pending:
- `{PKG / 'frames/gpt_image_2/first_frame_v01.png'}`
- `{PKG / 'storyboards/shared_choices/shared_choices_v01.png'}`

Missing files: generated PNGs and QC files pending localization.

Post-ready exports: none.

QC failures: pending image generation/localization.

High-risk issues: no real airline marks, no exact Law Roach likeness, no readable fake claims.

Next human action: localize generated stills, then inspect before any render handoff.
"""
    write("FACTORY_RUN_STATUS.md", status_text)
    (RUN_DIR / "FACTORY_RUN_STATUS.md").write_text(status_text, encoding="utf-8")


if __name__ == "__main__":
    main()
