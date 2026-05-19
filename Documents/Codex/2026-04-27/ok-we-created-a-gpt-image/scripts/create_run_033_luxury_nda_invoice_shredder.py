from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "033"
SLUG = "luxury_nda_invoice_shredder"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


SOURCES = [
    {
        "title": "Kim Kardashian & Kris Jenner's $7 Million Demand to Ray J Revealed in Court",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/17/kim-kardashian-kris-jenner-millions-demand-revealed-in-court/",
        "published": "2026-04-17",
        "used_for": "Primary public report that Kim Kardashian and Kris Jenner demanded $7 million from Ray J over alleged breach of a 2023 deal.",
        "verification": "Entertainment legal report based on court materials TMZ says it obtained. Treat detailed claims as allegations or reported court-document contents unless separately docket-verified.",
    },
    {
        "title": "Justin Baldoni's Lawyers Say Blake Lively's Businesses Failed Because She's Unlikable",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/28/justin-baldonis-lawyers-says-blake-lively-has-track-record-of-businesses-failing/",
        "published": "2026-04-28",
        "used_for": "Candidate contrast: fresh celebrity litigation with blunt business-failure argument.",
        "verification": "Court-hearing entertainment report; use only as reported argument.",
    },
    {
        "title": "Taylor Swift files 3 new trademark applications. One expert says it is to curb AI threats",
        "publisher": "AP News",
        "url": "https://apnews.com/article/7f56fbafb269d4959009f3ad34e28fc1",
        "published": "2026-04-28",
        "used_for": "Candidate contrast: current famous-face identity protection and AI voice filings.",
        "verification": "Primary mainstream wire context for trademark filings.",
    },
    {
        "title": "Trump's lawsuit against Wall Street Journal over Epstein story dismissed for now",
        "publisher": "Reuters via Investing.com",
        "url": "https://www.investing.com/news/stock-market-news/trumps-lawsuit-against-wall-street-journal-over-epstein-story-dismissed-for-now-4610594",
        "published": "2026-04-13",
        "used_for": "Candidate contrast: high-stakes defamation suit dismissed with leave to refile.",
        "verification": "Reuters report; political/legal risk is high and not selected.",
    },
]


CANDIDATES = [
    {
        "id": "A",
        "premise": "Luxury NDA Invoice Shredder: a $7 million demand letter becomes a Beverly Hills courtroom checkout lane, with an NDA stamp, installment receipts, and a velvet rope around a document shredder.",
        "source_basis": ["TMZ 2026-04-17"],
        "famous_face": 9,
        "public_conflict": 8,
        "ego_humiliation": 8,
        "absurd_quote_or_defense": 8,
        "brand_location_contrast": 9,
        "first_frame_contradiction": 10,
        "taste_risk_inverse": 6,
        "freshness": 8,
        "total": 66,
        "notes": "Best prop engine: money, NDA, payment schedule, luxury family-office courtroom. Keep sex-tape subject offscreen and non-explicit.",
    },
    {
        "id": "B",
        "premise": "Brand Busts Court Shelf: Blake Lively's alleged business damages are staged as a courtroom taste-test shelf where every bottle has a subpoena tag.",
        "source_basis": ["TMZ 2026-04-28"],
        "famous_face": 8,
        "public_conflict": 9,
        "ego_humiliation": 9,
        "absurd_quote_or_defense": 8,
        "brand_location_contrast": 8,
        "first_frame_contradiction": 9,
        "taste_risk_inverse": 5,
        "freshness": 10,
        "total": 66,
        "notes": "Strong humiliation quote, but franchise has recently used this litigation lane multiple times.",
    },
    {
        "id": "C",
        "premise": "Voice Vault Customs Desk: a pop superstar tries to pass through airport security with a pink guitar and two trademarked greeting phrases in evidence bags.",
        "source_basis": ["AP 2026-04-28"],
        "famous_face": 10,
        "public_conflict": 6,
        "ego_humiliation": 5,
        "absurd_quote_or_defense": 8,
        "brand_location_contrast": 8,
        "first_frame_contradiction": 9,
        "taste_risk_inverse": 8,
        "freshness": 9,
        "total": 63,
        "notes": "Clean and current, but prior SGFLIX run already explored voice/likeness vault territory.",
    },
    {
        "id": "D",
        "premise": "Powerhouse Refile Copy Room: a political defamation complaint is sent back through a courthouse copier labeled actual malice.",
        "source_basis": ["Reuters 2026-04-13"],
        "famous_face": 10,
        "public_conflict": 10,
        "ego_humiliation": 8,
        "absurd_quote_or_defense": 9,
        "brand_location_contrast": 7,
        "first_frame_contradiction": 8,
        "taste_risk_inverse": 3,
        "freshness": 7,
        "total": 62,
        "notes": "Very strong quote but too close to recent political/legal run and higher defamation/sexual-abuse adjacency risk.",
    },
]


FIRST_FRAME_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 9:16 vertical cinematic satire still.
subject: fictional luxury reality-TV matriarch archetype and fictional celebrity daughter archetype, both clearly non-identical to any real person, standing beside a giant sealed demand letter marked only with abstract redaction bars and a large non-readable dollar-symbol invoice stamp.
scene: Beverly Hills family-office courtroom hybrid, marble floor, velvet rope around an evidence cart, chrome document shredder, blank payment schedule cards, anonymous opposing musician silhouette at the far end holding a guitar case like a briefcase.
composition: first frame freeze before chaos; the demand letter is centered like a sacred artifact, the shredder glows behind it, two lawyers in sunglasses guard it like nightclub security.
camera: 28mm lens, low angle, vertical frame, slight handheld reality-TV energy, premium legal-drama lighting.
style: absurdist SGFLIX editorial satire, realistic cinematic photo, high-end tabloid courtroom, clean prop comedy, natural faces, no caricature ugliness.
photo quality and vibe: focused cinematic shot, natural light, highly aesthetic scene, movie-still composition, raw quality, warm rim light, subtle film grain, clean composition, cool ambient shadows, colors with a slight gray tone, make sure the lighting is natural and matches the background, no oversaturation, no oversharpening, a lively vibe as if the frame was taken while the characters were doing something, strong vignette, raw quality.
avoid next: exact celebrity likeness, real Kardashian/Jenner/Ray J faces, real brand logos, readable legal text, explicit sexual content, tabloid mastheads, fake news graphics, official court seals, distorted hands, watermarks, captions, excessive yellow in the photo."""


SHARED_CHOICES_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 16:9 director's-bible storyboard board.
purpose: SGFLIX Run 033 Shared Choices board for "Luxury NDA Invoice Shredder".
include: fictional character canon for a luxury reality-TV matriarch archetype, fictional celebrity daughter archetype, anonymous opposing musician archetype, two sunglass lawyers, hero props of redacted demand letter, NDA stamp, velvet rope, chrome shredder, installment receipt cards, guitar-case briefcase.
include: color palette swatches of marble white, legal red, chrome silver, velvet black, paparazzi flash, champagne beige.
include: environment/set design for Beverly Hills family-office courtroom, floor plan/blocking, six storyboard panels with camera/lens/movement notes, lighting and mood notes, visual rules, production notes.
style: premium production reference board, clean cinematic sketches mixed with realistic prop photos, minimal non-readable placeholder text only.
avoid next: exact public-figure likenesses, real brand logos, readable names, explicit sexual material, defamatory claims, official court insignia, messy typography, watermarks, excessive yellow in the photo."""


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


def write(rel: str, content: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_json(rel: str, data) -> None:
    write(rel, json.dumps(data, indent=2, ensure_ascii=True))


def generate_image(prompt: str, rel: str, size: str) -> str:
    path = PKG / rel
    try:
        from openai import OpenAI

        client = OpenAI()
        result = client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
        path.write_bytes(base64.b64decode(result.data[0].b64_json))
        return "openai_gpt_image_api"
    except Exception as exc:
        path.with_suffix(".generation_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return "blocked_openai_image_api"


def main() -> None:
    mkdirs()
    winner = CANDIDATES[0]

    write(
        "research/last30days_report.md",
        f"""
# Step 1: Research Intake - Run {RUN_ID}

Created: {NOW}

Research query/topic: current celebrity/legal conflicts from April 2026 with famous faces, ego, humiliation, absurd defenses, brand/location contrast, and strong prop-first visual contradictions.

Intake order followed:
1. Fresh web/source scan first.
2. Candidate board built from current reports.
3. Candidates scored before selection.
4. Run package created only after selecting the winner.
5. Still-image prompts and generation run after selection.

Fresh source context:
- TMZ reported on April 17, 2026 that Kim Kardashian and Kris Jenner demanded $7 million from Ray J over an alleged breach of a 2023 agreement tied to public discussion of the sex tape. The reported letter and payment schedule become the prop engine. Detailed claims are treated as reported allegations.
- TMZ reported on April 28, 2026 that Justin Baldoni's lawyers argued Blake Lively's brands were not damaged by him and had independent business issues. Strong quote/humiliation energy, but this litigation lane has been used recently by SGFLIX.
- AP reported on April 28, 2026 that Taylor Swift filed trademark applications for two voice phrases and a specific visual image, reportedly to guard against AI misuse. Clean but too close to a prior voice-vault run.
- Reuters reported on April 13, 2026 that a judge dismissed President Trump's WSJ defamation suit with leave to refile. High quote energy, but political/legal risk is high and recent run coverage overlaps.

Selected winner: {winner["premise"]}

Verification posture:
- Do not state that any party breached a contract as fact.
- Do not visualize or reference explicit sexual material.
- Use fictional archetypes for all stills, not exact public-figure likenesses.
- The comedy target is the luxury legal invoice spectacle, not private sexual history.
""",
    )
    write_json("research/sources.json", SOURCES)
    write_json("strategy/candidate_board.json", {"created_at": NOW, "candidates": CANDIDATES, "winner_id": winner["id"]})
    write(
        "strategy/winner_decision.md",
        f"""
# Winner Decision - Run {RUN_ID}

Winner: {winner["premise"]}

Score summary:
- A Luxury NDA Invoice Shredder: 66, selected because its prop engine is clearest and it moves away from the recently repeated Lively/Baldoni courtroom lane.
- B Brand Busts Court Shelf: 66, tied on score but rejected for franchise repetition.
- C Voice Vault Customs Desk: 63, rejected for prior SGFLIX voice/likeness overlap.
- D Powerhouse Refile Copy Room: 62, rejected for high political/legal risk and overlap with recent defamation-court packaging.

Creative thesis: turn a reported $7 million demand letter into a luxury checkout/evidence ritual. The first frame should read instantly without needing explicit details.
""",
    )
    write_json(
        "strategy/phase_minus_one_worthiness_audit.json",
        {
            "premise": winner["premise"],
            "worthiness_pass": True,
            "why_now": "Published April 17, 2026 and still adjacent to current celebrity legal chatter.",
            "humor_gates": {
                "famous_face": "strong",
                "public_conflict": "strong",
                "ego": "strong",
                "humiliation": "moderate",
                "absurd_prop": "very strong",
                "first_frame": "very strong",
            },
            "red_lines": ["No explicit sex-tape depiction", "No exact likeness cloning", "No statement that allegations are proven"],
        },
    )
    write_json(
        "strategy/source_entropy_audit.json",
        {
            "source_count": len(SOURCES),
            "source_mix": ["entertainment legal report", "mainstream wire", "Reuters political legal context"],
            "entropy_grade": "B",
            "risk": "Winner relies primarily on TMZ for the specific Kardashian/Jenner-Ray J details; mark as reported allegation.",
        },
    )
    write_json(
        "strategy/humor_logic_bridge.json",
        {
            "real_context": "A reported demand letter and payment schedule from a celebrity legal dispute.",
            "satire_transform": "The letter becomes a luxury invoice/evidence cart with nightclub security and a document shredder.",
            "first_frame_logic": "The seriousness of a courtroom collides with the absurdity of velvet-rope invoice handling.",
            "no_go": ["explicit sexual imagery", "readable real legal claims", "real logos"],
        },
    )
    write_json(
        "strategy/tribe_meta_score.json",
        {
            "TRiBE": {"topicality": 8, "recognizability": 9, "irony": 9, "bite": 7, "ease": 9, "total": 42},
            "meta": {"shareability": 8, "remixability": 8, "first_frame_stop": 9, "caption_fuel": 8, "total": 33},
        },
    )
    write_json(
        "strategy/risk_taste_score.json",
        {
            "overall_risk": "medium-high",
            "taste_score": 7,
            "risk_notes": [
                "Underlying topic touches private sexual history; keep visual and copy non-explicit.",
                "Do not imply court findings beyond reported allegations.",
                "Use fictional luxury reality-TV archetypes, not exact likenesses.",
            ],
        },
    )
    write(
        "strategy/franchise_decision.md",
        """
# Franchise Decision

Decision: one-off with optional "luxury legal checkout" franchise pattern.

Repeatable engine: famous-person legal documents are treated like retail checkout artifacts, coat-check tickets, receipts, or velvet-rope guest lists.

Do not repeat the sex-tape context. Future variants should use cleaner prop disputes.
""",
    )
    write(
        "frames/gpt_image_2/first_frame_v01_prompt.md",
        "# First Frame GPT Image Prompt\n\n" + FIRST_FRAME_PROMPT,
    )
    write(
        "storyboards/shared_choices/shared_choices_v01_prompt.md",
        "# Shared Choices GPT Image Prompt\n\n" + SHARED_CHOICES_PROMPT,
    )

    first_mode = generate_image(FIRST_FRAME_PROMPT, "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board_mode = generate_image(SHARED_CHOICES_PROMPT, "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")
    blocked = first_mode.startswith("blocked") or board_mode.startswith("blocked")

    write_json(
        "chai/chai_shot_specs.json",
        {
            "run_id": RUN_ID,
            "title": "Luxury NDA Invoice Shredder",
            "source_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shots": [
                {
                    "shot": "001",
                    "duration_sec": 6,
                    "subject": "redacted demand letter on velvet-roped evidence cart",
                    "scene": "Beverly Hills family-office courtroom",
                    "motion": "slow push-in, sunglass lawyer hand blocks the shredder button",
                    "camera": "28mm low vertical push",
                    "critique": "Must read as legal-invoice absurdity, not explicit scandal reenactment.",
                    "revision": "If faces drift toward exact celebrities, crop tighter on props.",
                }
            ],
        },
    )
    scene = {
        "run_id": RUN_ID,
        "shot_id": "001",
        "title": "Luxury NDA Invoice Shredder first frame",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "duration_seconds": 6,
        "no_video_generation": True,
        "workflow_spec": {"source_image": "frames/gpt_image_2/first_frame_v01.png", "render_lane": "manual_closed_tool_only_after_human_approval"},
    }
    write_json("scene_json/shot_0001.json", scene)
    write_json("scene_json/shot_001.json", scene)
    write_json(
        "handoffs/closed_tool_handoff.json",
        {
            "run_id": RUN_ID,
            "title": "Luxury NDA Invoice Shredder",
            "approved_source_image": "frames/gpt_image_2/first_frame_v01.png",
            "storyboard_board": "storyboards/shared_choices/shared_choices_v01.png",
            "hard_stop": "Do not generate video until human review and approval.",
            "prompt_summary": "Courtroom-family-office satire: redacted demand letter as luxury invoice, velvet rope, NDA stamp, chrome shredder.",
        },
    )
    write(
        "handoffs/grok_agent_prompt.md",
        """
# Grok Agent Prompt - Manual Reference Only

Use the approved first frame and Shared Choices board as visual anchors. Do not create video automatically.

Task after human approval: develop optional closed-tool reference prompts for a 6 second vertical clip where a redacted demand letter is treated like a luxury invoice at a courtroom checkout lane. Keep all public figures fictionalized.
""",
    )
    write(
        "captions/instagram_caption.md",
        """
The most expensive receipt in Beverly Hills just asked for its own velvet rope.

Reported legal-dispute satire. Allegations and filings are not findings.

#sgflix #satire #celebritylaw #popculture #courtroomcomedy
""",
    )
    write(
        "distribution/post_plan.md",
        """
# Distribution Post Plan

Primary surface: Instagram Reels.

Hook: "When the NDA gets bottle service."

Post only after human review confirms no exact likeness, no real logos, no readable defamatory text, and no explicit sexual reference.
""",
    )
    write(
        "skool/case_study.md",
        """
# Skool Case Study - Run 033

Lesson: turn risky celebrity scandal context into safer prop satire by moving the joke to legal paperwork, luxury process, and institutional absurdity.

The selected premise wins because the first frame can communicate the joke through a demand letter, invoice stamp, velvet rope, and shredder without repeating explicit source details.
""",
    )
    assets = [
        {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "generation_mode": first_mode},
        {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt"},
        {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "generation_mode": board_mode},
        {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt"},
    ]
    write_json("manifests/asset_manifest.json", {"created_at": NOW, "assets": assets, "blocked": blocked})
    write_json(
        f"RUN_{RUN_ID}_MASTER_PACKAGE.json",
        {
            "run_id": RUN_ID,
            "slug": SLUG,
            "title": "Luxury NDA Invoice Shredder",
            "status": "BLOCKED_IMAGE_GENERATION" if blocked else "COMPLETE_PACKAGE_NO_VIDEO",
            "selected_premise": winner["premise"],
            "research_topic": "current celebrity/legal conflicts with absurd prop-first visual contradictions",
            "generated_stills": assets,
            "no_video_generation": True,
        },
    )
    write(
        "qc/first_frame_v01_qc.md",
        f"""
# First Frame QC

Asset: `frames/gpt_image_2/first_frame_v01.png`
Generation mode: {first_mode}

Pass criteria:
- Prop comedy reads as demand-letter/invoice spectacle.
- No exact public-figure likeness.
- No real logos or official court seals.
- No explicit sexual content.
- No readable defamatory claims.

QC result: {"BLOCKED - image generation failed; see generation_error.txt" if first_mode.startswith("blocked") else "NEEDS HUMAN VISUAL REVIEW - generated still exists and should be checked for likeness/text drift."}
""",
    )
    write(
        "qc/shared_choices_v01_qc.md",
        f"""
# Shared Choices QC

Asset: `storyboards/shared_choices/shared_choices_v01.png`
Generation mode: {board_mode}

Pass criteria:
- Includes character/prop/environment/blocking/storyboard/lighting/visual-rule sections.
- Text is minimal or non-readable enough for private production use.
- No exact public-figure likenesses or real logos.

QC result: {"BLOCKED - image generation failed; see generation_error.txt" if board_mode.startswith("blocked") else "NEEDS HUMAN VISUAL REVIEW - generated board exists; verify text cleanliness and likeness safety."}
""",
    )
    readme_status = "blocked at image generation" if blocked else "complete for factory cycle, no video generated"
    write(
        "README.md",
        f"""
# RUN {RUN_ID} MASTER PACKAGE - Luxury NDA Invoice Shredder

Status: {readme_status}

Selected premise: {winner["premise"]}

Research query/topic: current celebrity/legal conflicts from April 2026 with famous faces, public conflict, ego, humiliation, absurd defenses, and strong first-frame visual contradiction.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

No video footage was generated. No social posting was attempted.

Next human action: review the generated first frame and Shared Choices board for likeness, logo, readable-text, and taste risk before any manual closed-tool video workflow.
""",
    )
    write(
        "FACTORY_RUN_STATUS.md",
        f"""
# Factory Run Status - Run {RUN_ID}

Status: {"BLOCKED_IMAGE_GENERATION" if blocked else "COMPLETE_PACKAGE_NO_VIDEO"}
Created at: {NOW}

Research-first order followed:
1. Current source scan completed.
2. Candidate board scored.
3. Winner selected.
4. Run package created after selection.
5. Still-image generation attempted after selection.
6. No video generation tools called.

Generated stills:
- `RUN_{RUN_ID}_MASTER_PACKAGE/frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `RUN_{RUN_ID}_MASTER_PACKAGE/storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

High-risk issues:
- Underlying dispute references private sexual history; keep all outputs non-explicit.
- TMZ-sourced dollar-demand details must remain framed as reported/alleged.
- Avoid exact public-figure likenesses and real logos.

Missing files: {"image PNGs are blocked; see generation_error.txt files" if blocked else "none from the required package list"}

Exact next human action: review stills for likeness/logo/readable-text drift and taste risk before any manual video work.
""",
    )
    if blocked:
        write(
            "BLOCKED_IMAGE_GENERATION.md",
            "Image generation did not complete for one or more required stills. Do not mark this run post-ready until prompts are rerun through GPT Image 2 and QC is updated.",
        )


if __name__ == "__main__":
    main()
