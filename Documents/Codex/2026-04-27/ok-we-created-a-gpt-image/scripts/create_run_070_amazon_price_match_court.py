from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "070"
SLUG = "amazon_price_match_court"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


SOURCES = [
    {
        "id": "ap_amazon_california_pricing_2026_04_20",
        "publisher": "Associated Press via WSLS",
        "title": "California says Amazon pressured retailers to boost prices on their websites to not undercut it",
        "url": "https://www.wsls.com/news/national/2026/04/20/california-says-amazon-pressured-retailers-to-boost-prices-on-their-websites-to-not-undercut-it/",
        "published": "2026-04-20",
        "verified_points": [
            "AP reported a California filing alleged Amazon used market leverage to push companies to raise prices on other websites so Amazon would not be undercut.",
            "AP reported California Attorney General Rob Bonta sued Amazon in 2022 under state antitrust and unfair competition law.",
            "AP reported the lawsuit is scheduled for trial next year and Bonta is seeking a preliminary injunction.",
            "AP reported Amazon disputed the motion and said it looks forward to responding in court.",
        ],
        "unverified_or_sensitive": [
            "The price-fixing claims are allegations from a court filing, not findings of liability.",
            "Do not imply Walmart, Levi Strauss, or Amazon have been found liable.",
            "Avoid using real logos or exact brand trade dress in generated images.",
        ],
    },
    {
        "id": "ap_trump_ballroom_pressure_2026_04_26",
        "publisher": "Associated Press",
        "title": "Justice Department cites dinner shooting to press preservationists to drop Trump ballroom suit",
        "url": "https://www.ap.org/news-highlights/elections/2026/justice-department-cites-dinner-shooting-to-press-preservationists-to-drop-trump-ballroom-suit/",
        "published": "2026-04-26",
        "verified_points": [
            "AP reported the Justice Department used the White House Correspondents' Dinner shooting to pressure preservationists to drop a ballroom lawsuit.",
            "AP reported the planned ballroom was described as a $400 million project on the former East Wing site.",
        ],
        "unverified_or_sensitive": [
            "Rejected because a prior SGFLIX ballroom lane exists and the source includes a recent shooting context.",
        ],
    },
    {
        "id": "ap_musk_altman_trial_2026_04_27",
        "publisher": "Associated Press via Local10",
        "title": "Elon Musk and OpenAI CEO Sam Altman head to court in high-stakes showdown over AI",
        "url": "https://www.local10.com/tech/2026/04/27/elon-musk-and-openai-ceo-sam-altman-head-to-court-in-high-stakes-showdown-over-ai/",
        "published": "2026-04-27",
        "verified_points": [
            "AP reported jury selection started in Oakland for Musk's lawsuit against OpenAI/Altman.",
            "AP reported the case concerns OpenAI's nonprofit origins and later for-profit structure.",
        ],
        "unverified_or_sensitive": [
            "Rejected as duplicate: run_034 already uses Musk/OpenAI courtroom cake-receipt mechanics.",
        ],
    },
    {
        "id": "ap_swift_showgirl_trademark_2026_03_31",
        "publisher": "Associated Press",
        "title": "Lawsuit says Taylor Swift's 'Showgirl' pose comes too close to the work of a real one",
        "url": "https://apnews.com/article/1e65b44eb6cca03297a712f1d247e3bf",
        "published": "2026-03-31",
        "verified_points": [
            "AP reported a federal trademark lawsuit challenged Swift's Showgirl branding.",
        ],
        "unverified_or_sensitive": [
            "Rejected as duplicate: run_064 already owns a Swift voice/trademark lane and earlier scripts considered Showgirl.",
        ],
    },
]


CANDIDATES = [
    {
        "id": "amazon_price_match_court",
        "title": "Amazon Price-Match Court",
        "premise": "A giant generic marketplace checkout lane turns into small-claims court where a price tag from another store is marched to the witness stand for being too low.",
        "source_ids": ["ap_amazon_california_pricing_2026_04_20"],
        "scores": {
            "freshness": 15,
            "famous_face_or_power_archetype": 16,
            "public_conflict": 18,
            "ego_humiliation": 13,
            "absurd_quote_or_defense": 14,
            "brand_location_contrast": 19,
            "first_frame_visual_contradiction": 20,
            "taste_safety": 17,
            "duplicate_penalty": 0,
        },
        "total": 132,
        "risk": "Use allegations language; no real logos, exact marketplace smile marks, or liability conclusion.",
        "decision": "selected",
    },
    {
        "id": "ballroom_subpoena_bouncer",
        "title": "Ballroom Subpoena Bouncer",
        "premise": "A velvet-rope ballroom bouncer checks preservationist subpoenas like gala wristbands while construction cones wait for table assignments.",
        "source_ids": ["ap_trump_ballroom_pressure_2026_04_26"],
        "scores": {
            "freshness": 17,
            "famous_face_or_power_archetype": 18,
            "public_conflict": 18,
            "ego_humiliation": 14,
            "absurd_quote_or_defense": 17,
            "brand_location_contrast": 18,
            "first_frame_visual_contradiction": 17,
            "taste_safety": 8,
            "duplicate_penalty": -22,
        },
        "total": 105,
        "risk": "Shooting context and prior ballroom package make this less clean.",
        "decision": "rejected_duplicate_and_taste_context",
    },
    {
        "id": "musk_altman_nonprofit_cake_receipt",
        "title": "Nonprofit Cake Receipt",
        "premise": "Two tech-founder silhouettes argue over a charity cake at a courtroom returns counter while the receipt keeps printing valuation zeros.",
        "source_ids": ["ap_musk_altman_trial_2026_04_27"],
        "scores": {
            "freshness": 18,
            "famous_face_or_power_archetype": 20,
            "public_conflict": 20,
            "ego_humiliation": 17,
            "absurd_quote_or_defense": 18,
            "brand_location_contrast": 16,
            "first_frame_visual_contradiction": 18,
            "taste_safety": 14,
            "duplicate_penalty": -35,
        },
        "total": 106,
        "risk": "Direct duplicate of run_034's Musk/OpenAI cake courtroom lane.",
        "decision": "rejected_duplicate_lane",
    },
    {
        "id": "showgirl_trademark_costume_claim",
        "title": "Showgirl Trademark Costume Claim",
        "premise": "A feathered costume is stopped at a trademark lost-and-found counter because three name tags all say showgirl.",
        "source_ids": ["ap_swift_showgirl_trademark_2026_03_31"],
        "scores": {
            "freshness": 11,
            "famous_face_or_power_archetype": 18,
            "public_conflict": 15,
            "ego_humiliation": 13,
            "absurd_quote_or_defense": 13,
            "brand_location_contrast": 17,
            "first_frame_visual_contradiction": 16,
            "taste_safety": 15,
            "duplicate_penalty": -28,
        },
        "total": 90,
        "risk": "Duplicate Taylor/trademark territory and unresolved claims.",
        "decision": "rejected_duplicate_lane",
    },
    {
        "id": "lively_baldoni_witness_parking",
        "title": "Witness Parking Valet",
        "premise": "A courthouse valet lot overflows with celebrity witness name placards while lawyers argue over who validated the ticket.",
        "source_ids": [],
        "scores": {
            "freshness": 13,
            "famous_face_or_power_archetype": 18,
            "public_conflict": 18,
            "ego_humiliation": 15,
            "absurd_quote_or_defense": 10,
            "brand_location_contrast": 12,
            "first_frame_visual_contradiction": 13,
            "taste_safety": 6,
            "duplicate_penalty": -30,
        },
        "total": 75,
        "risk": "Rejected due to repeated lane and serious underlying allegations.",
        "decision": "rejected_taste_and_duplicate",
    },
]


WINNER = CANDIDATES[0]


FIRST_PROMPT = """Generate an image with this exact creative intent.

Aspect ratio: 9:16 vertical cinematic first frame.
Subject: a fictional generic online marketplace checkout lane, not Amazon, no real logos, staged as a tiny small-claims courtroom.
Scene: a massive cardboard delivery box sits behind a judge's bench made from stacked shipping cartons. A price tag from a different generic store is on the witness stand, embarrassed because it is visibly lower. A deadpan clerk holds a rubber stamp reading ALLEGED PRICE MATCH. A shopping cart, barcode scanner, khaki pants on a hanger, home-decor box, garden trowel, and pet bowl form the evidence table.
Visual contradiction: everyday e-commerce checkout becomes courtroom pressure theater; a low price tag is treated like a hostile witness.
Camera: 24mm low-angle counter shot, scroll-stop first frame, foreground scanner and price tag sharp, background marketplace warehouse shelves cinematic.
Lighting/mood: clean retail fluorescents mixed with courtroom spotlight, dry legal-comedy absurdity, premium SGFLIX satire.
Style: realistic editorial production still, subtle film grain, muted palette of cardboard brown, court green, off-white labels, black barcode lines, small accent of generic marketplace blue. Large readable prop labels only.
Avoid: Amazon logo, Walmart logo, Levi's logo, official court seals, real people, politician likenesses, claims stated as proven, fake news graphics, watermarks, dense tiny text, distorted hands, excessive yellow."""


BOARD_PROMPT = """Generate an image with this exact creative intent.

Aspect ratio: 16:9 landscape Shared Choices director-bible board for SGFLIX satire "Amazon Price-Match Court" using a fictional generic marketplace only.
Layout: one polished production-design board with six clear visual zones and minimal readable labels.
Include: character canon for generic marketplace clerk, low-price tag witness, cardboard-box judge silhouette, background shopper silhouettes; hero props including ALLEGED PRICE MATCH stamp, barcode scanner, price tags, khaki pants hanger, pet bowl, garden trowel, shipping cartons; color palette of cardboard brown, court green, barcode black, off-white label paper, generic marketplace blue; environment/set design for warehouse checkout courtroom; floor plan/blocking diagram with judge bench, witness stand, evidence table, camera path; three storyboard panels with lens/camera movement notes; lighting/mood/style notes; visual rules and production guardrails.
Production notes: keep every legal point framed as alleged, no real logos, no official seals, no fake documents, no real executives, no readable dense text.
Style: premium cinematic director's-bible, realistic mixed-media production art, inspectable layout, no video generation.
Avoid: Amazon smile mark, Walmart spark, Levi's red tab, official court seal, real public figures, fake news lower thirds, messy microtext, watermarks, excessive yellow."""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def generate_image(prompt: str, path: Path, size: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("OPENAI_API_KEY"):
        err = path.with_suffix(".blocked.txt")
        write(err, "OPENAI_API_KEY was unavailable, so GPT Image generation could not run.")
        return {"ok": False, "mode": "blocked_no_openai_api_key", "path": str(path), "error_path": str(err)}
    try:
        from openai import OpenAI

        result = OpenAI().images.generate(model="gpt-image-1", prompt=prompt, size=size)
        b64 = result.data[0].b64_json
        if not b64:
            raise RuntimeError("image response did not include b64_json")
        path.write_bytes(base64.b64decode(b64))
        return {"ok": True, "mode": "openai_gpt_image_api", "path": str(path), "size": size}
    except Exception as exc:
        err = path.with_suffix(".blocked.txt")
        write(err, f"GPT Image generation failed: {type(exc).__name__}: {exc}")
        return {"ok": False, "mode": "blocked_openai_gpt_image_api", "path": str(path), "error_path": str(err)}


def main() -> None:
    for sub in [
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
        (PKG / sub).mkdir(parents=True, exist_ok=True)

    write_json(PKG / "research/sources.json", SOURCES)
    write(
        PKG / "research/last30days_report.md",
        f"""# Step 1: Research Intake - Run {RUN_ID}

Created: {NOW}

Research query/topic: current AP/source-context scan for fresh SGFLIX premises with public conflict, ego, humiliation, absurd defense/object, brand/location contrast, and a strong first-frame contradiction.

Intake sources used:
- AP via WSLS, April 20, 2026: California filing alleges Amazon pressured retailers/vendors to raise prices on other sites so Amazon would not be undercut.
- AP, April 26, 2026: Justice Department cited the White House Correspondents' Dinner shooting while pressuring preservationists over the Trump ballroom lawsuit.
- AP via Local10, April 27, 2026: Musk/Altman/OpenAI trial began jury selection in Oakland.
- AP, March 31, 2026: Taylor Swift Showgirl trademark lawsuit coverage.

Research-first compliance:
- No local image, storyboard, handoff, or existing SGFLIX asset was used as the premise source.
- Existing packages were scanned only after research intake to avoid duplicate lanes.
- Candidate scoring happened before creating this run package.

Winner selected after scoring: `{WINNER["id"]}`.

Verified context for winner:
- AP reported California's attorney general filed allegations in an ongoing lawsuit; the claims are not court findings.
- AP reported Amazon disputed the motion and said it would respond in court.
- AP reported examples involved retailers/vendors and product categories including apparel, home decor, garden products, and pet care.

Unverified / do-not-invent notes:
- Treat all pricing pressure details as allegations from a filing unless independently proven.
- Do not state that any named company was found liable.
- Do not use real marketplace, retailer, or clothing logos in image generation.
""",
    )

    write_json(PKG / "strategy/candidate_board.json", {"created": NOW, "candidates": CANDIDATES, "winner_id": WINNER["id"]})
    write(
        PKG / "strategy/winner_decision.md",
        f"""# Winner Decision - Run {RUN_ID}

Selected premise: **{WINNER["title"]}**

Premise: {WINNER["premise"]}

Score summary:
- Amazon Price-Match Court: 132, selected.
- Musk/Altman Nonprofit Cake Receipt: 106, rejected as a duplicate of run_034.
- Ballroom Subpoena Bouncer: 105, rejected for prior ballroom lane plus shooting-context taste risk.
- Showgirl Trademark Costume Claim: 90, rejected as Taylor/trademark duplicate territory.
- Lively/Baldoni Witness Parking Valet: 75, rejected for repeated lane and serious allegations.

Why this won: it turns a current antitrust/pricing allegation into a concrete prop joke that needs no real person likeness: a low price tag on the witness stand. The first frame reads instantly and the risk can be controlled with generic branding and allegation language.
""",
    )

    write_json(
        PKG / "strategy/phase_minus_one_worthiness_audit.json",
        {
            "phase_minus_one_audit": {
                "source_target": "California Amazon pricing filing",
                "track_a_newsjack_velocity": {
                    "active_trend_score": 7,
                    "algorithmic_slipstream": "Medium: current AP business/legal story with everyday affordability hook",
                    "polarization_factor": 7,
                    "track_a_total": 21,
                    "track_a_verdict": "PASS",
                },
                "track_b_archetype_resonance": {
                    "iconography_strength": 8,
                    "stereotype_rigidity": "High",
                    "subversion_potential": 9,
                    "track_b_total": 25,
                    "track_b_verdict": "PASS",
                },
                "system_verdict": {
                    "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                    "primary_vector": "TRACK_B_EVERGREEN_ARCHETYPE",
                    "urgency_class": "Medium",
                    "strategic_directive": "Make the price tag the humiliated witness; keep company names generic and legal framing alleged.",
                },
            }
        },
    )

    write_json(
        PKG / "strategy/source_entropy_audit.json",
        {
            "source_entropy_audit": {
                "baseline_reality_check": "A business/legal filing in an antitrust lawsuit over alleged marketplace pricing pressure.",
                "detected_anomalies": [
                    "A low price on another website is allegedly treated as a problem to be corrected.",
                    "Everyday products become antitrust evidence.",
                ],
                "native_entropy_score": 5,
                "subject_self_awareness": "deadpan",
                "comedic_vector_recommendation": "cringe_vector",
                "recommended_strategy": "micro_spotlight",
            }
        },
    )

    write_json(
        PKG / "strategy/humor_logic_bridge.json",
        {
            "bridge": {
                "real_world_pressure": "A marketplace allegedly dislikes being undercut by outside prices.",
                "comic_translation": "The cheaper price tag is treated like a hostile witness.",
                "prop_logic": ["witness stand", "rubber stamp", "scanner", "evidence table", "shipping-carton judge bench"],
                "first_frame_question": "Why is a price tag testifying in court?",
                "punchline_engine": "The alleged pricing behavior becomes a courtroom etiquette problem.",
            }
        },
    )

    write_json(
        PKG / "strategy/tribe_meta_score.json",
        {
            "tribe_meta_score": {
                "shareability": 8,
                "comment_bait": 8,
                "duet_remix_potential": 7,
                "visual_memorability": 9,
                "brand_safety": 8,
                "overall": 40,
                "notes": "Works for consumer frustration, antitrust watchers, and e-commerce jokes without needing defamation-risk likenesses.",
            }
        },
    )

    write_json(
        PKG / "strategy/risk_taste_score.json",
        {
            "risk_taste_score": {
                "defamation_or_false_claim_risk": "Medium: must keep allegations language.",
                "logo_trademark_risk": "Medium: no real marketplace or retailer logos.",
                "personal_harm_risk": "Low: no private person or health/violence context.",
                "taste_risk": "Low",
                "overall_go": "GO_WITH_GENERIC_BRANDING_AND_ALLEGED_LANGUAGE",
            }
        },
    )

    write(
        PKG / "strategy/franchise_decision.md",
        """# Franchise Decision

Decision: `limited_series_candidate`

This can become a recurring "consumer court" franchise: price tags, fees, receipts, subscriptions, and warranty cards get dragged into tiny courtrooms. Keep it episodic and prop-led, not logo-led.
""",
    )

    shot = {
        "run_id": RUN_ID,
        "title": WINNER["title"],
        "duration_target_seconds": 6,
        "video_generation_requested": False,
        "subject": "fictional marketplace checkout courtroom, low price tag witness",
        "scene": "warehouse checkout transformed into small-claims court",
        "motion": "slow push from scanner foreground to witness price tag; no render requested",
        "spatial": "judge bench rear, witness stand center, evidence table left, clerk right",
        "camera": "24mm low counter angle, vertical 9:16",
        "critique": "Must not use real Amazon/Walmart/Levi logos or imply proven liability.",
        "revision": "If generated text is messy, repair by removing all text except ALLEGED PRICE MATCH.",
    }
    write_json(PKG / "chai/chai_shot_specs.json", {"shots": [shot]})
    write_json(PKG / "scene_json/shot_0001.json", shot)
    write_json(PKG / "scene_json/shot_001.json", shot)

    closed_tool = {
        "run_id": RUN_ID,
        "premise": WINNER["premise"],
        "do_not_generate_video": True,
        "handoff_use": "manual still-to-video planning only after human approval",
        "prompt": FIRST_PROMPT,
        "negative_rules": [
            "no real logos",
            "no fake legal finding",
            "no real executives or politicians",
            "no video render request",
        ],
    }
    write_json(PKG / "handoffs/closed_tool_handoff.json", closed_tool)
    write(
        PKG / "handoffs/grok_agent_prompt.md",
        f"""# Grok Agent Prompt - Run {RUN_ID}

Do not generate video. Review this package as a still/handoff concept only.

Premise: {WINNER["premise"]}

Check whether the satire remains grounded in allegations, avoids real logos, and makes the low price tag witness readable in the first second.
""",
    )

    write(
        PKG / "captions/instagram_caption.md",
        """The cheapest price tag got called to testify.

Fictional SGFLIX satire based on reported allegations in an ongoing pricing lawsuit. No company has been found liable in this short.

#sgflix #satire #ecommerce #antitrust #pricing #consumerhumor #legalcomedy""",
    )
    write(
        PKG / "distribution/post_plan.md",
        """# Post Plan

Primary surface: Instagram Reels after human image approval.

Hook overlay: `THE PRICE TAG WAS TOO LOW`

Safety caption: `Fictional satire. Based on reported allegations, not a court finding.`

Do not post automatically. Do not use real logos. Hold for human review of image text and brand similarity.
""",
    )
    write(
        PKG / "skool/case_study.md",
        """# Skool Case Study

Lesson: turn a business/legal filing into a prop trial.

The strong move is not "company bad." It is object translation: a low price tag becomes a witness, a checkout scanner becomes evidence, and a generic marketplace becomes a courtroom. This keeps the satire visual and reduces personal-attack risk.
""",
    )

    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT)
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)

    first_result = generate_image(FIRST_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board_result = generate_image(BOARD_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")

    first_ok = bool(first_result.get("ok"))
    board_ok = bool(board_result.get("ok"))
    status = "complete_with_gpt_image_stills_no_video_generated" if first_ok and board_ok else "blocked_image_generation_no_video_generated"

    write(
        PKG / "qc/first_frame_v01_qc.md",
        f"""# First Frame QC

Generation result: `{first_result.get("mode")}`
Path: `{first_result.get("path")}`

Status: {"usable_pending_human_review" if first_ok else "blocked"}

Required human checks:
- No real Amazon, Walmart, Levi, or official court marks.
- Text should be limited and readable; reject if dense/messy.
- Frame must imply allegation, not proven liability.
""",
    )
    write(
        PKG / "qc/shared_choices_v01_qc.md",
        f"""# Shared Choices QC

Generation result: `{board_result.get("mode")}`
Path: `{board_result.get("path")}`

Status: {"usable_pending_human_review" if board_ok else "blocked"}

Required human checks:
- Board includes character canon, hero props, palette, environment, blocking, storyboard panels, lighting/style notes, visual rules, and production notes.
- Reject if it contains real logos or unreadable dense text.
""",
    )

    if not first_ok or not board_ok:
        write(
            PKG / "qc/IMAGE_GENERATION_BLOCKED_REPORT.md",
            f"""# Image Generation Blocked Report

Run {RUN_ID} completed research, scoring, and package writing, but GPT image generation did not fully succeed.

First frame result: `{first_result}`
Shared Choices result: `{board_result}`

Do not mark this run post-ready until a human generates or repairs the missing stills.
""",
        )

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
    missing = [p for p in required if not (PKG / p).exists()]

    package_json = {
        "run_id": RUN_ID,
        "slug": SLUG,
        "created": NOW,
        "status": status,
        "research_query_topic": "current AP/source-context scan for consumer/legal/pricing satire",
        "selected_premise": WINNER,
        "image_results": {"first_frame": first_result, "shared_choices": board_result},
        "video_generation_tools_called": False,
        "missing_files": missing,
    }
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", package_json)
    write_json(
        PKG / "manifests/asset_manifest.json",
        {
            "run_id": RUN_ID,
            "status": status,
            "assets": [
                {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", **first_result},
                {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True},
                {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", **board_result},
                {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True},
            ],
            "missing_files": missing,
            "post_ready_exports": [],
            "video_generation_tools_called": False,
        },
    )
    write(
        PKG / "README.md",
        f"""# RUN {RUN_ID} MASTER PACKAGE - {WINNER["title"]}

Status: `{status}`

Selected premise: {WINNER["premise"]}

Research topic: current AP/source-context scan for consumer/legal/pricing satire.

Primary source: AP via WSLS, April 20, 2026, California pricing-pressure filing story.

Still assets:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

No video-generation tool was called or requested.
""",
    )
    write(
        PKG / "FACTORY_RUN_STATUS.md",
        f"""# Factory Run Status - Run {RUN_ID}

Status: `{status}`

Research query/topic: current AP/source-context scan for consumer/legal/pricing satire.

Candidate board: 5 candidates scored before selection.

Selected premise: {WINNER["title"]}.

Score summary: Amazon 132 selected; Musk/OpenAI 106 duplicate rejected; ballroom 105 duplicate/taste-context rejected; Showgirl 90 duplicate rejected; Lively/Baldoni 75 duplicate/taste rejected.

Generated still-image paths:
- `{PKG / "frames/gpt_image_2/first_frame_v01.png"}` ({first_result.get("mode")})
- `{PKG / "storyboards/shared_choices/shared_choices_v01.png"}` ({board_result.get("mode")})

Missing files: {missing if missing else "none observed"}.

Post-ready exports: none; stills require human review before public use.

QC failures/high-risk issues: pricing facts must remain framed as allegations; reject any image with real marketplace/retailer logos, official court seals, or dense/messy text.

Exact next human action: inspect the first frame and Shared Choices board for real-logo drift and legal-overclaiming, then approve or request a repaired still prompt.
""",
    )
    write(
        RUN_DIR / "FACTORY_RUN_STATUS.md",
        f"""# Factory Run Status - Run {RUN_ID}

See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.

Status: `{status}`
""",
    )


if __name__ == "__main__":
    main()
