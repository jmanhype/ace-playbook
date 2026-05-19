from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "034"
SLUG = "musk_cake_receipt_court"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


SOURCES = [
    {
        "title": "In court, Elon Musk accuses OpenAI of trying to 'have your cake and eat it, too'",
        "publisher": "OPB / NPR",
        "url": "https://www.opb.org/article/2026/04/29/in-court-elon-musk-accuses-openai-of-trying-to-have-your-cake-and-eat-it-too/",
        "published": "2026-04-29",
        "used_for": "Primary current-source premise: Musk testified in his lawsuit against OpenAI and used the cake metaphor for nonprofit/for-profit structure.",
        "verification": "Mainstream public-radio report; courtroom paraphrase/quotes should stay attributed.",
    },
    {
        "title": "On the stand, Elon Musk can't escape his own tweets",
        "publisher": "TechCrunch",
        "url": "https://techcrunch.com/2026/04/29/on-the-stand-elon-musk-cant-escape-his-own-tweets/",
        "published": "2026-04-29",
        "used_for": "Contradiction engine: under-oath testimony reportedly clashed with recent public posts about AGI and OpenAI investment figures.",
        "verification": "Technology press courtroom report; use as reported testimony, not independent legal conclusion.",
    },
    {
        "title": "To beat Altman in court, Musk offers to give all damages to OpenAI nonprofit",
        "publisher": "Ars Technica",
        "url": "https://arstechnica.com/tech-policy/2026/04/to-beat-altman-in-court-musk-offers-to-give-all-damages-to-open-ai-nonprofit/",
        "published": "2026-04-08",
        "used_for": "Context for nonprofit/charitable-trust damages posture and the 'charity' framing.",
        "verification": "Technology/legal report; details should be framed as lawsuit posture.",
    },
    {
        "title": "Ice Spice to Be Deposed in Wig Lawsuit After Allegedly Reneging on $20K Deal",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/10/ice-spice-sued-by-thebellabrand-over-wig-deal/",
        "published": "2026-04-10",
        "used_for": "Rejected candidate: celebrity deposition with strong prop engine.",
        "verification": "Entertainment legal report based on court docs TMZ says it obtained; allegations unverified here.",
    },
    {
        "title": "Consumers sue to block Paramount-Warner Bros. deal",
        "publisher": "Los Angeles Times",
        "url": "https://www.latimes.com/entertainment-arts/business/story/2026-05-01/consumers-sue-to-block-paramount-warner-bros-deal",
        "published": "2026-05-01",
        "used_for": "Rejected candidate: media-merger public-airwaves contrast.",
        "verification": "Mainstream entertainment-business report.",
    },
    {
        "title": "Blake Lively Tells Court Her 'Mean Girl' Label Cost Her $40.5 Million",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/20/blake-lively-tells-court-mean-girl-label-cost-her-millions/",
        "published": "2026-04-20",
        "used_for": "Rejected candidate: damages math and reputation labels.",
        "verification": "Entertainment legal report; not selected partly due to repeated SGFLIX litigation lane.",
    },
]


CANDIDATES = [
    {
        "id": "A",
        "title": "Moral High Ground Cake Receipt Court",
        "premise": "A fictional tech-billionaire founder archetype stands at a courthouse bakery counter, trying to return a halo-shaped charity cake while receipts, capped-profit slices, and old tweets are bagged as evidence.",
        "source_basis": ["OPB/NPR 2026-04-29", "TechCrunch 2026-04-29", "Ars Technica 2026-04-08"],
        "famous_face": 10,
        "public_conflict": 10,
        "ego_humiliation": 9,
        "absurd_quote_or_defense": 10,
        "brand_location_contrast": 9,
        "first_frame_contradiction": 10,
        "taste_risk_inverse": 6,
        "freshness": 10,
        "total": 74,
        "notes": "Best winner: famous public trial, clean quote-prop engine, courtroom setting, and visible contradiction between halo charity and bakery/cash-register mechanics.",
    },
    {
        "id": "B",
        "title": "Ice Spice Wig Deposition Salon",
        "premise": "A courthouse deposition table turns into a custom-wig fitting room, with 25 evidence mannequins and a $20K appointment card.",
        "source_basis": ["TMZ 2026-04-10"],
        "famous_face": 8,
        "public_conflict": 7,
        "ego_humiliation": 7,
        "absurd_quote_or_defense": 7,
        "brand_location_contrast": 9,
        "first_frame_contradiction": 9,
        "taste_risk_inverse": 6,
        "freshness": 7,
        "total": 60,
        "notes": "Good prop comedy, but weaker mainstream signal and narrower cultural footprint.",
    },
    {
        "id": "C",
        "title": "Public Airwaves Auction Desk",
        "premise": "A studio backlot auctioneer tries to sell a giant TV antenna while consumers hold streaming bills and movie tickets as evidence.",
        "source_basis": ["Los Angeles Times 2026-05-01"],
        "famous_face": 4,
        "public_conflict": 8,
        "ego_humiliation": 5,
        "absurd_quote_or_defense": 8,
        "brand_location_contrast": 9,
        "first_frame_contradiction": 8,
        "taste_risk_inverse": 7,
        "freshness": 10,
        "total": 59,
        "notes": "Fresh and consequential, but no famous face and less immediate character comedy.",
    },
    {
        "id": "D",
        "title": "Mean Girl Damages Calculator",
        "premise": "A courtroom accountant weighs reputation labels on a jewelers scale while beverage bottles and haircare boxes wear subpoena tags.",
        "source_basis": ["TMZ 2026-04-20"],
        "famous_face": 8,
        "public_conflict": 9,
        "ego_humiliation": 9,
        "absurd_quote_or_defense": 8,
        "brand_location_contrast": 8,
        "first_frame_contradiction": 9,
        "taste_risk_inverse": 5,
        "freshness": 7,
        "total": 63,
        "notes": "Strong damages-math premise, rejected because recent SGFLIX packages have already used this legal-celebrity lane.",
    },
]


FIRST_FRAME_PROMPT = """Use case: photorealistic-natural
Asset type: SGFLIX 9:16 first-frame still
Primary request: Create a cinematic satirical courtroom-bakery first frame for a short-form video called "Moral High Ground Cake Receipt Court."
Subject: fictional tech-billionaire founder archetype, clearly not an exact real-person likeness, in a dark suit at a witness stand shaped like a bakery counter. He holds a half-eaten halo-shaped cake box while a clerk in court attire stamps anonymous receipts. His expression is controlled but visibly cornered.
Scene/backdrop: Oakland federal courthouse imagined as a premium bakery checkout lane, with marble counter, jury box, evidence cart, glass cake case, charity donation jar, capped-profit cake slices under cloches, and printed tweet receipts bagged in transparent evidence sleeves. Use abstract icons and unreadable placeholder marks only.
Composition: vertical 9:16, low-angle 28mm lens, first second of action, cake and receipt evidence centered, witness stand/bakery counter in foreground, judge bench blurred in background, mild handheld documentary energy.
Lighting/style: realistic cinematic photo, SGFLIX absurdist editorial satire, natural skin texture, restrained color palette, cool courthouse shadows with warm bakery practical lights, subtle film grain, strong vignette, high-end tabloid legal drama.
Avoid: exact Elon Musk, Sam Altman, OpenAI logo, Tesla/X/xAI logos, readable text, fake news graphics, official court seals, defamation claims as text, distorted hands, watermarks, captions, oversaturated yellow."""


SHARED_CHOICES_PROMPT = """Use case: productivity-visual
Asset type: SGFLIX 16:9 Shared Choices director-bible storyboard board
Primary request: Create a director's-bible storyboard board for "Moral High Ground Cake Receipt Court."
Include: fictional character canon for tech-billionaire founder archetype, opposing AI-executive silhouette, courthouse bakery clerk, stern judge, evidence-cart paralegal, and bored jurors.
Include hero props: halo-shaped charity cake, capped-profit cake slices, receipt roll, transparent tweet-evidence sleeves, donation jar, bakery tongs, courtroom stamp, glass cake case, witness-stand checkout counter.
Include color palette swatches: marble white, legal red, receipt paper gray, chrome silver, cake-box cream, courthouse navy, bakery warm amber.
Include environment/set design: Oakland federal courthouse transformed into a bakery checkout lane, floor plan/blocking, six storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, production notes.
Style: premium production reference board, cinematic sketches mixed with realistic prop-photo callouts, clean layout, minimal non-readable placeholder text only.
Avoid: exact public-figure likenesses, real company logos, readable names, fake news chyron, official court insignia, messy typography, watermarks, excessive yellow."""


def write(rel: str, content: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_json(rel: str, data) -> None:
    write(rel, json.dumps(data, indent=2, ensure_ascii=True))


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
        "storyboards/shared_choices",
        "qc",
    ]:
        (PKG / rel).mkdir(parents=True, exist_ok=True)


def main() -> None:
    winner = CANDIDATES[0]
    mkdirs()

    write(
        "research/last30days_report.md",
        f"""
# Step 1: Research Intake - Run {RUN_ID}

Created: {NOW}

Research query/topic: fresh April-May 2026 famous-person public conflicts with ego, humiliation, quotable courtroom defenses, brand/location contrast, and first-frame visual contradiction.

Order of operations followed:
1. Fresh web/source scan first.
2. Candidate board built from current reports, not local images or old storyboards.
3. Candidates scored before selecting the winner.
4. New numbered run package created only after selection.
5. Still-image artifacts queued after the winner.

Current source context:
- OPB/NPR reported on April 29, 2026 that Elon Musk testified for a second day in his lawsuit against OpenAI and Sam Altman, including the quote-frame that OpenAI could not "have your cake and eat it too" around nonprofit/for-profit structure.
- TechCrunch reported on April 29, 2026 that cross-examination focused on Musk's prior support for for-profit structures and on testimony that conflicted with recent posts about AGI and OpenAI investment figures.
- Ars Technica reported on April 8, 2026 that Musk's side amended its requested remedy so damages would go to OpenAI's nonprofit arm, strengthening the charity/cake/receipt prop system.
- Ice Spice's wig-deposition lawsuit, Paramount-Warner consumer-merger litigation, and Blake Lively reputation-damages math were considered and rejected after scoring.

Selected winner: {winner["premise"]}

Grounded creative angle:
The joke is not "celebrity plus AI." The joke is a courtroom bakery where nonprofit moral authority, capped-profit math, and old public statements become physical cake slices and receipts. The first frame should be readable as a contradiction before any caption: a famous-founder archetype trying to return a halo cake at a legal checkout counter.

Verification posture:
- Attribute courtroom facts to sources; do not present claims as proven legal findings.
- Avoid exact likeness or logos in generated images.
- Avoid suggesting a trial result; this is a satirical scene built from reported testimony and lawsuit posture.
- Treat legal interpretations as reported arguments, not settled facts.
""",
    )
    write_json("research/sources.json", SOURCES)

    write_json("strategy/candidate_board.json", {"created_at": NOW, "winner_id": winner["id"], "candidates": CANDIDATES})
    write(
        "strategy/winner_decision.md",
        """
# Winner Decision - Run 034

Winner: A - Moral High Ground Cake Receipt Court.

Score summary:
- A Musk/OpenAI cake receipt court: 74. Selected for famous face, current courtroom conflict, direct quote-prop, ego/humiliation, and instant first-frame contradiction.
- D Blake Lively damages calculator: 63. Strong, but too close to recently repeated celebrity-litigation packaging.
- B Ice Spice wig deposition salon: 60. Strong prop world, weaker current mainstream signal.
- C Public airwaves auction desk: 59. Fresh, but lacks a famous face and character engine.

Selection rationale:
This passes the celebrity + AI worthiness gate because the comedic object is not generic AI. The joke is a physical legal contradiction: moral halo, cake, receipts, capped-profit slices, and old tweets all colliding at a courthouse checkout counter.
""",
    )
    write_json(
        "strategy/phase_minus_one_worthiness_audit.json",
        {
            "phase_minus_one_audit": {
                "source_target": "Musk/OpenAI courtroom testimony",
                "track_a_newsjack_velocity": {
                    "active_trend_score": 9,
                    "algorithmic_slipstream": "High: current testimony, major tech/legal conflict, fresh reporting",
                    "polarization_factor": 9,
                    "track_a_total": 27,
                    "track_a_verdict": "PASS",
                },
                "track_b_archetype_resonance": {
                    "iconography_strength": 9,
                    "stereotype_rigidity": "High",
                    "subversion_potential": 10,
                    "track_b_total": 28,
                    "track_b_verdict": "PASS",
                },
                "system_verdict": {
                    "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                    "primary_vector": "TRACK_A_NEWSJACK",
                    "urgency_class": "High",
                    "strategic_directive": "Use the courtroom quote as a prop system; keep legal facts attributed and avoid exact likeness/logos.",
                },
            }
        },
    )
    write_json(
        "strategy/source_entropy_audit.json",
        {
            "source_entropy_audit": {
                "baseline_reality_check": "A real, high-stakes civil trial over OpenAI's nonprofit/for-profit structure and Musk's claims against OpenAI executives.",
                "detected_anomalies": [
                    "Cake metaphor in courtroom testimony",
                    "Charity/nonprofit halo language in a giant commercial AI fight",
                    "Old posts and testimony reportedly colliding under cross-examination",
                ],
                "native_entropy_score": 6,
                "subject_self_awareness": "trying_to_look_principled",
                "comedic_vector_recommendation": "cringe_vector",
                "recommended_strategy": "straight_man_framing with prop literalization",
            }
        },
    )
    write_json(
        "strategy/humor_logic_bridge.json",
        {
            "hook": "What if the 'have your cake and eat it too' argument became a courthouse bakery receipt dispute?",
            "literalized_objects": ["halo cake", "capped-profit slices", "receipt roll", "tweet evidence sleeves", "charity donation jar"],
            "misdirection": "A serious tech trial opens like a luxury bakery return desk.",
            "punchline": "The moral high ground is treated as a returnable pastry with missing receipts.",
            "caption_phrase": "The cake entered evidence before lunch.",
        },
    )
    write_json(
        "strategy/tribe_meta_score.json",
        {
            "scores": {
                "concept_clarity": 9,
                "scroll_stop_first_frame": 10,
                "humor_or_emotional_charge": 9,
                "recognizable_context": 9,
                "share_comment_potential": 8,
                "rewatch_potential": 7,
                "format_fit": 9,
                "series_potential": 8,
                "asset_reuse": 9,
                "platform_risk": 6,
                "factory_value": 9,
            },
            "total": 93,
            "verdict": "MAKE_STILLS_AND_HANDOFF",
        },
    )
    write_json(
        "strategy/risk_taste_score.json",
        {
            "risk": {
                "defamation": "medium: avoid assertions beyond reported testimony",
                "likeness": "medium: use fictional archetype, no exact face",
                "brand_logo": "medium: no real AI/company logos",
                "misleading_news": "medium: captions must read as satire",
                "platform": "medium-low",
            },
            "taste": {
                "scene_not_summary": True,
                "one_second_joke": True,
                "human_feel": True,
                "too_ai": False,
                "dxfilms_family": "fake legal-lore tableau with prop comedy",
            },
            "verdict": "PROCEED_WITH_VISUAL_GUARDRAILS",
        },
    )
    write(
        "strategy/franchise_decision.md",
        """
# Franchise Decision

Franchise lane: Courtroom Object Literalization.

Reusable format:
Turn a public quote or legal argument into a physical checkout-counter object, then stage a first-frame evidence ritual around it.

Potential follow-ups:
- The AGI receipt printer jams.
- The charity halo cake gets sliced into cap-table wedges.
- Old tweets arrive in bakery bags marked as stale inventory.

Decision: keep as a repeatable legal-prop franchise, but avoid overusing AI lawsuits unless each has a concrete visual object.
""",
    )

    write_json(
        "chai/chai_shot_specs.json",
        {
            "run_id": RUN_ID,
            "title": "Moral High Ground Cake Receipt Court",
            "shots": [
                {
                    "shot": "001",
                    "subject": "Fictional tech-billionaire founder archetype at witness-stand bakery counter with halo cake box.",
                    "scene": "Federal courthouse transformed into premium bakery checkout lane.",
                    "motion": "Founder slides a half-eaten halo cake toward clerk; receipt roll spills into evidence cart.",
                    "spatial": "Cake and receipts foreground center; founder midground left; judge bench blurred rear.",
                    "camera": "28mm low-angle vertical, slight handheld, shallow depth.",
                    "critique": "Avoid exact likeness/logos/readable text; hands and cake geometry must stay clean.",
                    "revision": "If identity drifts too close to a real person, push toward generic founder silhouette and prop emphasis.",
                }
            ],
        },
    )
    shot_json = {
        "id": "shot_001",
        "duration_seconds": 6,
        "subject": "Fictional tech-founder archetype in courtroom bakery scene",
        "action": "Attempts to return the moral-high-ground cake while court clerk stamps receipts into evidence.",
        "camera": "Vertical 9:16, 28mm low angle, slow push in.",
        "motion_beats": ["cake box hits counter", "receipt roll unspools", "evidence bag lifts into frame"],
        "negative_prompts": ["exact likeness", "logos", "readable text", "fake news graphics", "official seals"],
    }
    write_json("scene_json/shot_0001.json", shot_json)
    write_json("scene_json/shot_001.json", shot_json)

    write_json(
        "handoffs/closed_tool_handoff.json",
        {
            "run_id": RUN_ID,
            "title": "Moral High Ground Cake Receipt Court",
            "no_video_generation_requested": True,
            "image_prompts": {
                "first_frame_v01": "frames/gpt_image_2/first_frame_v01_prompt.md",
                "shared_choices_v01": "storyboards/shared_choices/shared_choices_v01_prompt.md",
            },
            "video_handoff": {
                "duration": "6-10 seconds",
                "render_tools": "Manual only. Do not start renders in this automation.",
                "action": "Use first frame as anchor; animate receipt roll, cake slide, and evidence bag lift.",
            },
        },
    )
    write(
        "handoffs/grok_agent_prompt.md",
        """
# Grok Agent Prompt

Build a 6-10 second satirical legal-prop scene from the stills for SGFLIX Run 034.

Do not generate or request video from this automation. Human operator only: use this as a manual closed-tool prompt.

Core action: a fictional tech-founder archetype tries to return a halo-shaped charity cake at a courthouse bakery counter while receipts and old post printouts are sealed as evidence.

Guardrails: no exact real-person likeness, no OpenAI/Tesla/X/xAI logos, no readable court claims, no fake news chyron, no verdict implication.
""",
    )
    write("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)

    write(
        "captions/instagram_caption.md",
        """
The cake entered evidence before lunch.

Reported courtroom testimony said the fight was about who gets the moral high ground, who gets the upside, and who still has the receipts. SGFLIX translated that into the only legal setting that made sense: a courthouse bakery return desk.

#sgflix #aiparody #techsatire #courtroomsatire #elonmusk #openai #founderdrama #legalhumor #satire
""",
    )
    write(
        "distribution/post_plan.md",
        """
# Post Plan

Primary surface: Instagram Reels.
Secondary: TikTok, YouTube Shorts.

First-frame copy direction: keep caption minimal; let the cake/receipt contradiction carry the hook.

Risk note: public caption should frame this as satire based on reported testimony. Avoid saying any party won, lied, stole, or breached anything as fact.

Post-ready exports: none; no video footage generated.
""",
    )
    write(
        "skool/case_study.md",
        """
# Skool Case Study - Run 034

Lesson: turn a reported quote into a physical prop system.

Why it works:
- The quote already contains a visual object: cake.
- The case already contains an institutional contradiction: nonprofit halo versus commercial upside.
- The courtroom supplies stakes without needing exposition.
- Receipts/tweets/capped-profit slices give animatable objects for later video work.

Assignment:
Find one public quote from a current story and convert it into three physical props, then design a first frame where the props tell the whole joke before the caption.
""",
    )
    write_json(
        "manifests/asset_manifest.json",
        {
            "run_id": RUN_ID,
            "created_at": NOW,
            "required_assets": [
                "frames/gpt_image_2/first_frame_v01.png",
                "storyboards/shared_choices/shared_choices_v01.png",
            ],
            "status": "awaiting_image_generation",
        },
    )
    write(
        "qc/first_frame_v01_qc.md",
        """
# First Frame QC

Status: pending image generation.

Pass criteria:
- Reads as courtroom bakery in one second.
- Cake and receipts are centered and legible as props without readable legal text.
- Character is fictionalized, not exact public-figure likeness.
- No real logos or official court seals.
""",
    )
    write(
        "qc/shared_choices_v01_qc.md",
        """
# Shared Choices QC

Status: pending image generation.

Pass criteria:
- Board includes character canon, hero props, palette, environment, blocking/floor plan, storyboard panels, lighting/style rules, and production notes.
- Text is non-readable or minimal placeholder only.
- No real logos, exact likenesses, or official court marks.
""",
    )
    write(
        "README.md",
        """
# RUN 034 MASTER PACKAGE - Moral High Ground Cake Receipt Court

This run package was created after fresh research intake and candidate scoring.

Selected premise:
A fictional tech-billionaire founder archetype stands at a courthouse bakery counter, trying to return a halo-shaped charity cake while receipts, capped-profit slices, and old tweets are bagged as evidence.

No video footage was generated or requested.
""",
    )
    write_json(
        "RUN_034_MASTER_PACKAGE.json",
        {
            "run_id": RUN_ID,
            "slug": SLUG,
            "title": "Moral High Ground Cake Receipt Court",
            "created_at": NOW,
            "selected_candidate": winner,
            "source_count": len(SOURCES),
            "status": "awaiting_image_generation",
            "video_generation": "not_requested_not_performed",
        },
    )
    write(
        "FACTORY_RUN_STATUS.md",
        """
# Factory Run Status - Run 034

Status: awaiting generated still images.

Completed:
- Research intake.
- Source log.
- Candidate board and scoring.
- Winner decision.
- Strategy audits.
- CHAI and scene JSON handoff package.
- Caption/distribution/skool drafts.
- Image prompts.

Remaining:
- Generate first frame image.
- Generate Shared Choices director-bible board.
- Update QC and manifest with saved image paths.
""",
    )

    prev_report = ROOT / "sgflix_runs" / "run_033_luxury_nda_invoice_shredder" / "NEXT_STEP_REPORT_2026-05-02.md"
    if not prev_report.exists():
        prev_report.write_text(
            f"""# Next Step Report - Run 033

Created: {NOW}

Run 033 appears incomplete from package scan: it has a master package and blocked image-generation note, but the expected full artifact set and generated still paths were not present at the scanned depth.

Next human/automation action:
Resume Run 033 only after Run 034 is complete. Start by opening `RUN_033_MASTER_PACKAGE/BLOCKED_IMAGE_GENERATION.md`, then either generate the required first-frame and Shared Choices stills or leave the run explicitly blocked with a complete missing-artifact list.
""",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
