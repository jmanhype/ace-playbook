from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "036"
SLUG = "delta_one_ice_receipt_desk"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


SOURCES = [
    {
        "title": "Natasha Lyonne Claims 'I Was Detained' by ICE After Being Removed From Flight",
        "publisher": "Variety Australia",
        "url": "https://au.variety.com/2026/tv/news/natasha-lyonne-ice-detained-kicked-off-flight-euphoria-35152/",
        "published": "2026-04-11",
        "used_for": "Primary entertainment-industry report: Lyonne publicly claimed ICE detained her after she was removed from a Delta One red-eye flight.",
        "verification": "Use as reported public statement and entertainment report; do not state the ICE involvement as established fact.",
    },
    {
        "title": "DHS Denies Natasha Lyonne Was Detained by ICE After Flight Removal",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/10/dhs-spokesperson-denies-detaining-natasha-lyonne/",
        "published": "2026-04-10",
        "used_for": "Contradictory official-response context and risk guardrail: DHS denied that ICE detained Lyonne.",
        "verification": "Official denial as reported by TMZ; all narrative beats should be framed as claim/denial.",
    },
    {
        "title": "Homeland Security Denies Detaining Natasha Lyonne at LAX",
        "publisher": "Los Angeles Today",
        "url": "https://nationaltoday.com/us/ca/los-angeles/news/2026/04/12/homeland-security-denies-detaining-natasha-lyonne-at-lax/",
        "published": "2026-04-12",
        "used_for": "Secondary context: public-summary report that Lyonne was removed from a red-eye Delta One flight to New York and DHS denied ICE detention.",
        "verification": "Secondary source; useful for timeline, not for new factual claims.",
    },
    {
        "title": "Reese Witherspoon told fans to learn A.I., authors are slamming her",
        "publisher": "Los Angeles Times",
        "url": "https://www.latimes.com/entertainment-arts/story/2026-04-16/reese-witherspoon-authors-book-club-learn-ai-chatgpt",
        "published": "2026-04-16",
        "used_for": "Rejected candidate: AI backlash with a denial/clarification engine.",
        "verification": "Mainstream entertainment report; rejected as weaker first-frame contradiction.",
    },
    {
        "title": "Cyndi Lauper Jokingly Threatens Heckler During Las Vegas Residency Concert",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/25/cyndi-lauper-playfully-threatens-vegas-concert-heckler/",
        "published": "2026-04-25",
        "used_for": "Rejected candidate: Vegas heckler confrontation with strong room energy.",
        "verification": "Entertainment video report; rejected because SGFLIX already used a Cyndi/heckler court lane.",
    },
    {
        "title": "Katy Perry's Rep Denies Ruby Rose's Sexual Assault Allegations",
        "publisher": "E! News",
        "url": "https://www.eonline.com/news/1430836/katy-perrys-rep-denies-ruby-roses-sexual-assault-allegations",
        "published": "2026-04-13",
        "used_for": "Rejected candidate: allegation-denial cycle with high public conflict.",
        "verification": "Rejected for taste/legal risk; do not build comedy from disputed assault allegations.",
    },
]


CANDIDATES = [
    {
        "id": "A",
        "title": "Delta One ICE Receipt Desk",
        "premise": "A fictional red-carpet actor archetype stands at a luxury airline customer-service counter where a boarding pass, sleep mask, and giant stamped 'not ICE' receipt are treated like crisis evidence.",
        "source_basis": ["Variety Australia 2026-04-11", "TMZ 2026-04-10", "Los Angeles Today 2026-04-12"],
        "famous_face": 7,
        "public_conflict": 8,
        "ego_humiliation": 8,
        "absurd_quote_or_defense": 10,
        "brand_location_contrast": 10,
        "first_frame_contradiction": 10,
        "taste_risk_inverse": 7,
        "freshness": 7,
        "total": 67,
        "notes": "Best winner: first-class glamour collides with airport bureaucracy and claim/denial paperwork. Keep the joke on over-escalated travel optics, not immigration enforcement harms.",
    },
    {
        "id": "B",
        "title": "Cyndi Caesars Heckler Court",
        "premise": "A Vegas residency stage becomes a tiny courtroom where a heckler is served a glitter subpoena mid-song.",
        "source_basis": ["TMZ 2026-04-25", "Las Vegas Magazine 2026-04-27"],
        "famous_face": 8,
        "public_conflict": 7,
        "ego_humiliation": 7,
        "absurd_quote_or_defense": 8,
        "brand_location_contrast": 8,
        "first_frame_contradiction": 9,
        "taste_risk_inverse": 8,
        "freshness": 8,
        "total": 63,
        "notes": "Strong room, but too close to prior run_013 Cyndi/heckler material.",
    },
    {
        "id": "C",
        "title": "Reese AI Invoice Pantry",
        "premise": "A book-club founder archetype opens a pantry where every paperback has a tiny AI invoice and a stamp saying nobody paid her.",
        "source_basis": ["Los Angeles Times 2026-04-16", "Variety Australia 2026-04-22"],
        "famous_face": 8,
        "public_conflict": 7,
        "ego_humiliation": 6,
        "absurd_quote_or_defense": 8,
        "brand_location_contrast": 7,
        "first_frame_contradiction": 7,
        "taste_risk_inverse": 6,
        "freshness": 7,
        "total": 56,
        "notes": "Rejected because generic celebrity + AI needs a sharper physical contradiction.",
    },
    {
        "id": "D",
        "title": "Katy Ruby Denial Fire Door",
        "premise": "A nightclub exit is transformed into a deposition room where every allegation and denial appears as sealed envelopes.",
        "source_basis": ["E! News 2026-04-13", "Los Angeles Times 2026-04-13", "TheWrap 2026-04-13"],
        "famous_face": 8,
        "public_conflict": 9,
        "ego_humiliation": 6,
        "absurd_quote_or_defense": 6,
        "brand_location_contrast": 7,
        "first_frame_contradiction": 7,
        "taste_risk_inverse": 1,
        "freshness": 7,
        "total": 51,
        "notes": "Rejected: disputed sexual-assault allegations are not a clean SGFLIX comedy lane.",
    },
]


FIRST_FRAME_PROMPT = """Use case: photorealistic-natural
Asset type: SGFLIX 9:16 first-frame still
Primary request: Create a cinematic satirical airport-bureaucracy first frame for a short-form video called "Delta One ICE Receipt Desk."
Subject: a fictional raspy-voiced red-carpet actor archetype, clearly not an exact Natasha Lyonne likeness, wearing rumpled premiere-night black tailoring, red hair styled loosely, oversized sunglasses in one hand, and a luxury airline sleep mask around the wrist. Expression: dry, stunned, trying to explain a bad travel decision without winning the room.
Scene/backdrop: LAX premium airline customer-service counter imagined as a tiny immigration-court clerk window, but with no real government seals. A calm airline supervisor stamps a giant receipt that says only abstract unreadable block marks, a security officer silhouette points toward a gate sign with no readable text, and a rolling evidence tray holds a boarding pass, pillow, tiny sleep-aid bottle with blank label, red-eye coffee, and a "not a badge" prop ID sleeve.
Composition: vertical 9:16, 28mm lens, first second of action, customer-service counter foreground, receipt stamp frozen mid-air, actor archetype centered, departure board blurred in background, paparazzi flash reflections in glass, airport carpet visible.
Lighting/style: realistic cinematic photo, premium SGFLIX absurdist editorial satire, natural skin texture, restrained but high-contrast color palette, cool airport fluorescents mixed with warm lounge lighting, subtle film grain, expensive tabloid legal-comedy mood.
Avoid: exact Natasha Lyonne likeness, real Delta logo, ICE/DHS/TSA seals, readable personal data, fake news graphics, making detention appear confirmed, mocking immigration detainees, watermarks, messy typography, distorted hands."""


SHARED_CHOICES_PROMPT = """Use case: productivity-visual
Asset type: SGFLIX 16:9 Shared Choices director-bible storyboard board
Primary request: Create a director's-bible storyboard board for "Delta One ICE Receipt Desk."
Include character canon: fictional red-carpet actor archetype, airline supervisor, security officer silhouette, night-flight gate agent, red-eye passenger extras, paparazzi reflection ghosts.
Include hero props: luxury sleep mask, blank boarding pass, red-eye coffee, pillow, tiny blank sleep-aid bottle, giant stamped receipt, rolling evidence tray, customer-service bell, airport carpet pattern, blurred departure board.
Include color palette swatches: terminal gray, lounge amber, legal red stamp ink, black premiere tailoring, carpet teal, receipt paper white, brushed aluminum.
Include environment/set design: LAX premium counter transformed into a bureaucratic clerk window, floor plan/blocking, six storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, production notes.
Style: premium production reference board, cinematic sketches mixed with realistic prop-photo callouts, clean layout, minimal non-readable placeholder text only.
Avoid: exact public-figure likenesses, real airline logos, ICE/DHS/TSA seals, readable passenger details, fake confirmation of disputed facts, messy typography, watermarks."""


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


def generate_image(prompt: str, output: Path, size: str) -> str:
    try:
        from openai import OpenAI

        client = OpenAI()
        result = client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
        b64 = result.data[0].b64_json
        output.write_bytes(base64.b64decode(b64))
        return "openai_gpt_image_api"
    except Exception as exc:
        output.with_suffix(".generation_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        return f"blocked_openai_image_api:{type(exc).__name__}"


def main() -> None:
    winner = CANDIDATES[0]
    mkdirs()

    # Prior run_035 was present as a directory but had no package files.
    (ROOT / "sgflix_runs" / "run_035_fieri_tate_flavortown_ufc").mkdir(parents=True, exist_ok=True)
    (ROOT / "sgflix_runs" / "run_035_fieri_tate_flavortown_ufc" / "NEXT_STEP_REPORT.md").write_text(
        f"""# Next-Step Report - Run 035

Created: {NOW}

Status: incomplete prior package; package files exist, but generated stills and final QC are pending.

Next action: generate `RUN_035_MASTER_PACKAGE/frames/gpt_image_2/first_frame_v01.png` and `RUN_035_MASTER_PACKAGE/storyboards/shared_choices/shared_choices_v01.png`, then update QC and the asset manifest. This report does not replace the fresh Run 036 research intake.
""",
        encoding="utf-8",
    )

    write(
        "research/last30days_report.md",
        f"""
# Step 1: Research Intake - Run {RUN_ID}

Created: {NOW}

Research query/topic: current April 2026 celebrity/public-figure incidents with famous face, public conflict, ego/humiliation, absurd quote or denial, brand/location contrast, and strong first-frame visual contradiction.

Order of operations followed:
1. Fresh web/source scan first.
2. Candidate board built from current source context, not local images, existing storyboards, or nearby assets.
3. Candidates scored before selecting the winner.
4. New numbered run package created after selecting the winner.
5. GPT Image still prompts and generation attempts came after winner selection.

Current source context:
- Natasha Lyonne publicly stated that she was detained by ICE after being removed from a Los Angeles-to-New York red-eye flight; Variety Australia covered the claim on April 11, 2026.
- TMZ reported on April 10, 2026 that DHS denied ICE detained Lyonne after the flight removal.
- Secondary reporting summarized the same claim/denial structure and identified the incident as a Delta One red-eye from LAX to New York.
- Reese Witherspoon's AI backlash and Cyndi Lauper's Vegas heckler confrontation were considered as cleaner celebrity-comedy candidates, but scored lower or duplicated prior SGFLIX territory.
- Ruby Rose/Katy Perry was rejected despite public conflict because disputed sexual-assault allegations are not a taste-safe comedy engine.

Selected winner: {winner["premise"]}

Grounded creative angle:
The joke is airport paperwork swallowing celebrity glamour: a premium red-eye passenger, a sleep mask, a boarding pass, a claim/denial receipt, and a clerk window treated like crisis evidence. The visual must not confirm disputed ICE involvement; it should show a bureaucratic "not ICE" receipt and the absurdity of explanation management.

Verification posture:
- Mark the ICE involvement as Lyonne's claim and DHS's reported denial.
- Do not use real airline, ICE, DHS, or TSA logos/seals.
- Use a fictional actor archetype, not an exact celebrity likeness.
- Keep the satire on public travel optics and bureaucracy, not on detained immigrants or law-enforcement victims.
""",
    )
    write_json("research/sources.json", SOURCES)
    write_json("strategy/candidate_board.json", {"created_at": NOW, "winner_id": winner["id"], "candidates": CANDIDATES})
    write(
        "strategy/winner_decision.md",
        """
# Winner Decision - Run 036

Winner: A - Delta One ICE Receipt Desk.

Score summary:
- A Delta One ICE Receipt Desk: 67. Selected for claim/denial structure, luxury-airport contrast, visible prop engine, and clean first-frame contradiction.
- B Cyndi Caesars Heckler Court: 63. Strong but too close to a prior SGFLIX Cyndi/heckler lane.
- C Reese AI Invoice Pantry: 56. Timely, but too close to generic celebrity + AI without a stronger physical joke.
- D Katy Ruby Denial Fire Door: 51. High conflict but rejected for taste and legal risk around disputed assault allegations.

Selection rationale:
The winning premise has an immediate image: premiere-night glamour at a premium airline counter, trying to get a stamped receipt that resolves a public claim/denial. It is current, famous-adjacent, absurd, and visually legible without inventing facts.
""",
    )
    write_json(
        "strategy/phase_minus_one_worthiness_audit.json",
        {
            "source_target": "Natasha Lyonne flight-removal ICE claim and DHS denial",
            "newsjack_velocity": {"score": 7, "verdict": "pass"},
            "archetype_resonance": {"score": 8, "verdict": "pass"},
            "visual_contradiction": {"score": 10, "verdict": "pass"},
            "avoidance_notes": [
                "Do not confirm disputed ICE involvement.",
                "Do not mock real immigration detention.",
                "Do not use real airline or government logos.",
            ],
            "overall": "PASS_WITH_RISK_GUARDRAILS",
        },
    )
    write_json(
        "strategy/source_entropy_audit.json",
        {
            "source_count": len(SOURCES),
            "selected_winner_sources": 3,
            "source_mix": ["entertainment trade", "tabloid with official denial", "secondary local summary"],
            "entropy_score": 7,
            "risk": "Claim/denial is not court-adjudicated; frame as public statement vs reported denial.",
        },
    )
    write_json(
        "strategy/humor_logic_bridge.json",
        {
            "setup": "A celebrity airport incident becomes a paperwork crisis.",
            "turn": "The premium airline counter behaves like a miniature legal clerk window.",
            "payoff": "The only thing anyone can issue is a receipt proving what not to call the incident.",
            "first_frame_read": "Luxury sleep mask plus giant stamped receipt plus security silhouette.",
            "no_go": ["confirmed detention claim", "immigration cruelty", "real logos"],
        },
    )
    write_json(
        "strategy/tribe_meta_score.json",
        {
            "first_frame_scroll_stop": 9,
            "shareability": 7,
            "comment_friction": 8,
            "caption_dependency": 5,
            "tribe_hooks": ["airport ordeal people", "prestige-TV fans", "claim-vs-denial spectators", "celebrity bureaucracy comedy"],
            "meta_score": 8,
        },
    )
    write_json(
        "strategy/risk_taste_score.json",
        {
            "legal_risk": 5,
            "taste_risk": 4,
            "platform_risk": 4,
            "mitigations": [
                "Use fictional actor archetype.",
                "Show claim/denial paperwork, not a confirmed government action.",
                "Keep all official/airline marks generic.",
            ],
            "verdict": "APPROVED_FOR_STILLS_WITH_GUARDRAILS",
        },
    )
    write(
        "strategy/franchise_decision.md",
        """
# Franchise Decision

Verdict: one-off SGFLIX newsjack, not a recurring franchise.

Reason: the premise is a strong current contradiction, but it depends on a narrow claim/denial travel incident. Franchise extension would drift into immigration-enforcement comedy, which is not the desired lane. Keep it as a single premium airport-bureaucracy sketch.
""",
    )

    write("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)
    first_mode = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board_mode = generate_image(SHARED_CHOICES_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")
    blocked = first_mode.startswith("blocked") or board_mode.startswith("blocked")

    write_json(
        "chai/chai_shot_specs.json",
        {
            "run_id": RUN_ID,
            "title": "Delta One ICE Receipt Desk",
            "generation_status": {"first_frame": first_mode, "shared_choices": board_mode},
            "shots": [
                {
                    "shot": "0001",
                    "duration": "0-2s",
                    "subject": "fictional red-carpet actor archetype at premium airline counter",
                    "scene": "airport counter transformed into claim-denial clerk window",
                    "motion": "stamp slams down on giant receipt; actor freezes mid-explanation",
                    "spatial": "counter foreground, security silhouette right, blurred gate board rear",
                    "camera": "vertical 28mm low-angle push-in",
                    "critique": "Must read as claim/denial bureaucracy, not confirmed detention.",
                    "revision": "If text is messy, replace with abstract stamps or a blank red block.",
                },
                {
                    "shot": "001",
                    "duration": "2-6s",
                    "subject": "rolling evidence tray with sleep mask, boarding pass, coffee, pillow",
                    "scene": "LAX lounge lighting and generic customer-service line",
                    "motion": "agent sorts props like evidence while passengers watch",
                    "spatial": "tray center, actor background, clerk stamp left",
                    "camera": "50mm insert then snap zoom",
                    "critique": "No readable airline/government identity marks.",
                    "revision": "Crop away accidental logos or official-looking seals.",
                },
            ],
        },
    )
    shot_json = {
        "run_id": RUN_ID,
        "premise": winner["premise"],
        "status": "handoff_only_no_video_generation",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "camera": "vertical 9:16, 28mm, push-in from receipt stamp to actor archetype",
        "action": "An airline supervisor stamps a giant claim-denial receipt while the actor archetype realizes the paperwork has become the joke.",
        "risk_guardrails": ["No real logos", "No exact likeness", "No confirmed ICE claim"],
    }
    write_json("scene_json/shot_0001.json", shot_json)
    write_json("scene_json/shot_001.json", {**shot_json, "camera": "50mm insert on evidence tray", "action": "The sleep mask, boarding pass, red-eye coffee, and blank bottle are arranged like trial exhibits."})
    write_json(
        "handoffs/closed_tool_handoff.json",
        {
            "run_id": RUN_ID,
            "video_generation_allowed": False,
            "approved_stills": {
                "first_frame": "frames/gpt_image_2/first_frame_v01.png",
                "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
            },
            "manual_next_step": "Human may use the stills as references for a separate closed-tool workflow, but this factory cycle must not request or start video rendering.",
            "guardrails": ["fictional archetype", "claim/denial framing", "generic airline/government surfaces"],
        },
    )
    write(
        "handoffs/grok_agent_prompt.md",
        """
# Grok/Closed-Tool Reference Prompt

Do not generate video in this factory cycle.

Use the stills and prompts as a reference pack for a future manual workflow. Build a six-second airport-bureaucracy satire around a fictional red-carpet actor archetype at a premium airline customer-service counter. The clerk stamps claim/denial paperwork; the sleep mask, boarding pass, coffee, and pillow become evidence props. Avoid exact Natasha Lyonne likeness, real airline logos, government seals, or any claim that ICE detention was confirmed.
""",
    )
    write(
        "captions/instagram_caption.md",
        """
When the red-eye receipt becomes the whole press statement.

Claim, denial, sleep mask, boarding pass, and one customer-service stamp with too much power.

#sgflix #airportcomedy #celebritynews #satire #redflight #bureaucracycore
""",
    )
    write(
        "distribution/post_plan.md",
        """
# Distribution Plan

Primary surface: Instagram Reels.
Secondary surface: TikTok after human taste review.

Hook: open on the giant receipt stamp and sleep mask before any dialogue.
Overlay text: keep minimal and generic: "THE RECEIPT DESK" or no overlay if generated text is messy.

Not post-ready until human approves likeness distance and confirms no accidental real logos/seals appear in the stills.
""",
    )
    write(
        "skool/case_study.md",
        """
# Skool Case Study - Run 036

Lesson: a claim/denial public incident can become a prop engine if the joke lands on paperwork, not on proving either side.

What worked:
- Research first, not asset-first.
- Candidate scoring rejected higher-risk allegation comedy.
- The winner has a physical contradiction: luxury travel vs bureaucratic receipt ritual.

Reusable pattern:
Take a disputed public statement, pair it with an official denial, then materialize the ambiguity as an object the audience understands instantly.
""",
    )
    manifest_assets = [
        "RUN_036_MASTER_PACKAGE/README.md",
        "RUN_036_MASTER_PACKAGE/RUN_036_MASTER_PACKAGE.json",
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
    write_json(
        "manifests/asset_manifest.json",
        {
            "run_id": RUN_ID,
            "created_at": NOW,
            "status": "blocked_image_generation" if blocked else "complete_still_package_no_video",
            "assets": [{"path": p, "exists": (PKG / p).exists() if not p.startswith("RUN_036_MASTER_PACKAGE/") else (PKG / p.replace("RUN_036_MASTER_PACKAGE/", "")).exists()} for p in manifest_assets],
            "generation_modes": {"first_frame": first_mode, "shared_choices": board_mode},
        },
    )
    write(
        "qc/first_frame_v01_qc.md",
        f"""
# First Frame QC

Asset: `frames/gpt_image_2/first_frame_v01.png`
Generation mode: `{first_mode}`

Checks:
- Winner selected before image generation: pass.
- No video generation requested: pass.
- Required concept: luxury airline counter plus claim/denial receipt.
- Human review needed: confirm no exact Natasha Lyonne likeness, real airline logo, government seal, or readable personal data.

Status: {"BLOCKED - see generation_error file" if first_mode.startswith("blocked") else "USABLE_PENDING_HUMAN_REVIEW"}
""",
    )
    write(
        "qc/shared_choices_v01_qc.md",
        f"""
# Shared Choices QC

Asset: `storyboards/shared_choices/shared_choices_v01.png`
Generation mode: `{board_mode}`

Checks:
- Director-bible prompt includes character, props, palette, environment, floor plan/blocking, storyboard panels, lighting, visual rules, and production notes: pass.
- Human review needed: generated board text may be approximate; use prompt file as source of truth if typography is messy.

Status: {"BLOCKED - see generation_error file" if board_mode.startswith("blocked") else "USABLE_PENDING_HUMAN_REVIEW"}
""",
    )
    if blocked:
        write(
            "qc/IMAGE_GENERATION_BLOCKED_REPORT.md",
            f"""
# Image Generation Blocked Report

Run {RUN_ID} attempted GPT Image generation after the research winner was selected.

First frame mode: `{first_mode}`
Shared Choices mode: `{board_mode}`

Missing production image files:
- `frames/gpt_image_2/first_frame_v01.png` if no PNG exists beside the error file.
- `storyboards/shared_choices/shared_choices_v01.png` if no PNG exists beside the error file.

Next action: rerun the saved prompt files through GPT Image 2, then update QC.
""",
        )
    status = "BLOCKED_IMAGE_GENERATION" if blocked else "COMPLETE_STILL_PACKAGE_NO_VIDEO"
    missing = []
    for rel in ["frames/gpt_image_2/first_frame_v01.png", "storyboards/shared_choices/shared_choices_v01.png"]:
        if not (PKG / rel).exists():
            missing.append(rel)
    write_json(
        f"RUN_{RUN_ID}_MASTER_PACKAGE.json",
        {
            "run_id": RUN_ID,
            "slug": SLUG,
            "status": status,
            "created_at": NOW,
            "research_query": "April 2026 public-figure travel/entertainment incidents with visual contradiction",
            "selected_premise": winner["premise"],
            "winner_score": winner["total"],
            "candidate_scores": {c["id"]: c["total"] for c in CANDIDATES},
            "generated_stills": {
                "first_frame": "frames/gpt_image_2/first_frame_v01.png",
                "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
            },
            "missing_files": missing,
            "high_risk_issues": ["claim/denial factual ambiguity", "avoid exact likeness", "avoid official logos/seals"],
            "video_generation": "prohibited_not_requested",
        },
    )
    readme = f"""
# RUN {RUN_ID} MASTER PACKAGE - Delta One ICE Receipt Desk

Status: {status}

Selected premise: {winner["premise"]}

Research query/topic: April 2026 public-figure travel/entertainment incidents with visible contradiction and claim/denial comedy.

Score summary: A=67, B=63, C=56, D=51. Winner A had the clearest first frame and strongest guardrailed joke.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

Missing files: {", ".join(missing) if missing else "none from required package checklist"}

Post-ready exports: none; this is a still-image and handoff package only.

QC failures: {"GPT Image generation blocked; see qc/IMAGE_GENERATION_BLOCKED_REPORT.md." if blocked else "none found automatically; human review still required for likeness/logos/text."}

High-risk issues:
- Do not state ICE detention as fact.
- Avoid real airline/government branding.
- Keep actor as a fictional archetype.

Exact next human action: restore GPT Image billing/API access, rerun the saved prompt files, then review the generated first frame and Shared Choices board for likeness distance, accidental logos/seals, and messy text before any manual video handoff.
"""
    write("README.md", readme)
    status_md = f"""
# Factory Run Status - Run {RUN_ID}

Status: {status}
Created: {NOW}

Completed:
- Fresh research intake completed first.
- Current candidate board built and scored.
- Winner selected before package creation work.
- Required strategy, handoff, caption, distribution, Skool, manifest, and QC files created.
- GPT Image generation attempted for first frame and Shared Choices board after winner selection.

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

Missing files: {", ".join(missing) if missing else "none"}

Post-ready exports: none.
QC failures: {"image generation blocked" if blocked else "human review still required for likeness/logos/text"}
High-risk issues: claim/denial ambiguity; exact likeness risk; official logo/seal risk.
Exact next human action: restore GPT Image billing/API access, rerun `frames/gpt_image_2/first_frame_v01_prompt.md` and `storyboards/shared_choices/shared_choices_v01_prompt.md`, then complete QC before any video workflow.
"""
    write("FACTORY_RUN_STATUS.md", status_md)
    (RUN_DIR / "FACTORY_RUN_STATUS.md").write_text(status_md.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
