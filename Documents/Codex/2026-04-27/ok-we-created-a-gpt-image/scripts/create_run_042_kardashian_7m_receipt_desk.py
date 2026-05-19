import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN = "042"
SLUG = "kardashian_7m_receipt_desk"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

SOURCES = [
    {
        "id": "tmz_kim_kris_rayj_7m_demand",
        "title": "Kim Kardashian & Kris Jenner's $7 Million Demand to Ray J Revealed in Court",
        "url": "https://www.tmz.com/2026/04/17/kim-kardashian-kris-jenner-millions-demand-revealed-in-court/",
        "publisher": "TMZ",
        "published": "2026-04-17",
        "verification_status": "reported_by_entertainment_legal_outlet",
        "used_for": "winner source context"
    },
    {
        "id": "tmz_sabrina_coachella_yodel",
        "title": "Sabrina Carpenter Confuses Cultural Cry For Yodeling During Coachella Set",
        "url": "https://www.tmz.com/2026/04/11/sabrina-carpenter-slammed-for-disliking-cultural-cry-at-coachella/",
        "publisher": "TMZ",
        "published": "2026-04-11",
        "verification_status": "reported_by_entertainment_outlet",
        "used_for": "candidate board alternative"
    },
    {
        "id": "tmz_cyndi_lauper_heckler",
        "title": "Cyndi Lauper Jokingly Threatens Heckler During Las Vegas Residency Concert",
        "url": "https://www.tmz.com/2026/04/25/cyndi-lauper-playfully-threatens-vegas-concert-heckler/",
        "publisher": "TMZ",
        "published": "2026-04-25",
        "verification_status": "reported_by_entertainment_outlet",
        "used_for": "candidate board alternative"
    },
    {
        "id": "ars_ticketmaster_monopoly",
        "title": "Jury finds Live Nation/Ticketmaster is illegal monopoly that overcharged fans",
        "url": "https://arstechnica.com/tech-policy/2026/04/jury-finds-live-nation-ticketmaster-is-illegal-monopoly-that-overcharged-fans/",
        "publisher": "Ars Technica",
        "published": "2026-04-16",
        "verification_status": "reported_by_technology_policy_outlet",
        "used_for": "candidate board alternative"
    },
    {
        "id": "independent_jorginho_chappell_bodyguard",
        "title": "Jorginho expresses regret over hotel security incident involving Chappell Roan",
        "url": "https://www.independent.co.uk/arts-entertainment/music/news/chappell-roan-jorginho-security-guard-incident-statement-b2956887.html",
        "publisher": "The Independent",
        "published": "2026-04-13",
        "verification_status": "reported_by_news_outlet",
        "used_for": "candidate board alternative"
    }
]

CANDIDATES = [
    {
        "rank": 1,
        "slug": SLUG,
        "premise": "A fictional reality-TV legal desk where a seven-million-dollar demand letter prints as a luxury CVS receipt so long it becomes the furniture.",
        "source_basis": ["tmz_kim_kris_rayj_7m_demand"],
        "scores": {
            "famous_face": 10,
            "public_conflict": 9,
            "ego_or_status_pressure": 9,
            "humiliation_or_absurd_defense": 8,
            "brand_location_contrast": 8,
            "first_frame_contradiction": 10,
            "taste_safety": 7,
            "freshness": 8
        },
        "total": 69,
        "risk_notes": "Use fictionalized silhouettes and document satire. Do not depict intimate material, repeat allegations as fact, or show real legal docs."
    },
    {
        "rank": 2,
        "slug": "cyndi_vegas_heckler_receipt",
        "premise": "A Vegas residency stage manager issues a heckler a glittery cease-and-desist ticket from a microphone stand.",
        "source_basis": ["tmz_cyndi_lauper_heckler"],
        "scores": {
            "famous_face": 7,
            "public_conflict": 6,
            "ego_or_status_pressure": 6,
            "humiliation_or_absurd_defense": 8,
            "brand_location_contrast": 8,
            "first_frame_contradiction": 8,
            "taste_safety": 8,
            "freshness": 8
        },
        "total": 59,
        "risk_notes": "Safe, but smaller stakes and less recognizable first-frame object."
    },
    {
        "rank": 3,
        "slug": "ticketmaster_fee_confessional_repair",
        "premise": "A concert fee receipt is cross-examined under oath until the service charges confess.",
        "source_basis": ["ars_ticketmaster_monopoly"],
        "scores": {
            "famous_face": 4,
            "public_conflict": 9,
            "ego_or_status_pressure": 6,
            "humiliation_or_absurd_defense": 8,
            "brand_location_contrast": 8,
            "first_frame_contradiction": 8,
            "taste_safety": 9,
            "freshness": 7
        },
        "total": 59,
        "risk_notes": "Strong system satire but weaker SGFLIX celebrity-face hook; also overlaps prior run_033 topic."
    },
    {
        "rank": 4,
        "slug": "chappell_bodyguard_breakfast_map",
        "premise": "A hotel breakfast table becomes a security org chart proving the bodyguard belonged to nobody.",
        "source_basis": ["independent_jorginho_chappell_bodyguard"],
        "scores": {
            "famous_face": 8,
            "public_conflict": 7,
            "ego_or_status_pressure": 7,
            "humiliation_or_absurd_defense": 7,
            "brand_location_contrast": 7,
            "first_frame_contradiction": 8,
            "taste_safety": 6,
            "freshness": 6
        },
        "total": 56,
        "risk_notes": "Risk around a child and online harassment; use only with heavy fictionalization."
    },
    {
        "rank": 5,
        "slug": "sabrina_yodel_translation_booth",
        "premise": "A pop star's Coachella piano sprouts an emergency translation booth for every audience sound.",
        "source_basis": ["tmz_sabrina_coachella_yodel"],
        "scores": {
            "famous_face": 9,
            "public_conflict": 8,
            "ego_or_status_pressure": 8,
            "humiliation_or_absurd_defense": 8,
            "brand_location_contrast": 8,
            "first_frame_contradiction": 9,
            "taste_safety": 4,
            "freshness": 6
        },
        "total": 60,
        "risk_notes": "Rejected despite score because run_016 already covered this territory and cultural mockery risk is high."
    }
]

FIRST_PROMPT = """Vertical 9:16 satirical cinematic first frame. A fictional high-glam reality-TV momager archetype and a fictional influencer-law-student archetype sit at a chrome legal desk in a white marble office. Between them, a seven-million-dollar demand letter has printed as an absurdly long luxury store receipt that curls across the desk, spills onto the floor, and forms a paper runway. The visible paper text is generic: PARODY DEMAND, SEVEN MILLION, RECEIPT DESK. No real faces, no photorealistic celebrity likeness, no real legal document, no intimate imagery, no logos. Visual contradiction: ultra-luxury legal war room versus mundane cash-register receipt. Glossy reality-TV lighting, 35mm lens, shallow depth, chrome, white marble, black tailoring, blush-pink legal tabs, comedic forensic seriousness."""

BOARD_PROMPT = """Create a single 16:9 private director's-bible storyboard board for SGFLIX Run 042, 'Seven Million Receipt Desk'. Include fictional character canon for a glam momager archetype and influencer-law-student archetype, hero props of an endless demand-letter receipt, chrome legal desk, pink legal tabs, receipt printer, white marble office set, color palette swatches, floor plan and blocking, six storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes. Keep generated text minimal and generic. Do not use real celebrity faces, real names, real logos, real court documents, or intimate imagery."""

def w(rel, text):
    p = PKG / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n", encoding="utf-8")

def j(rel, data):
    w(rel, json.dumps(data, indent=2))

def generate_image(prompt, rel, size):
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from openai import OpenAI
        client = OpenAI()
        result = client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
        path.write_bytes(base64.b64decode(result.data[0].b64_json))
        return {"path": rel, "mode": "openai_gpt_image_api", "ok": True}
    except Exception as exc:
        err = path.with_suffix(".generation_error.txt")
        err.write_text(str(exc) + "\n", encoding="utf-8")
        return {"path": rel, "mode": "blocked_openai_gpt_image_api", "ok": False, "error_path": str(err.relative_to(PKG))}

def main():
    for d in ["research", "strategy", "frames/gpt_image_2", "storyboards/shared_choices", "qc", "chai", "scene_json", "handoffs", "captions", "distribution", "skool", "manifests"]:
        (PKG / d).mkdir(parents=True, exist_ok=True)

    (ROOT / "sgflix_runs" / "run_041_qatar_air_force_one_gift_shop" / "NEXT_STEP_REPORT_2026-05-02.md").write_text(
        "# Next-Step Report - Run 041\n\nRun 041 is marked `BLOCKED_IMAGE_GENERATION`. Next step: rerun its saved first-frame and Shared Choices prompts through GPT Image 2 after image generation/disk capacity is available, then update QC before any manual video workflow.\n",
        encoding="utf-8",
    )

    w("research/last30days_report.md", f"""# Step 1: Research Intake - Run {RUN}

Created: {NOW}

Research query/topic: current late-April/early-May 2026 pop-culture conflict with strong SGFLIX visual contradiction: celebrity legal demand, Coachella apologies, Vegas heckler, Ticketmaster monopoly, and bodyguard misunderstanding.

Fresh source context:
- TMZ reported on April 17, 2026 that Kim Kardashian and Kris Jenner demanded Ray J pay $7 million over an alleged breach of a secret 2023 deal. This package treats the claim as reported legal context only.
- TMZ reported current Coachella and Vegas-stage alternatives; Ars Technica reported the Live Nation/Ticketmaster verdict; The Independent reported the Jorginho/Chappell Roan bodyguard clarification.

Selection logic:
The seven-million-dollar demand-letter premise won because it has famous-face recognition, public conflict, ego/status pressure, and a clear first-frame contradiction: luxury legal seriousness versus a mundane receipt that physically overwhelms the room.

Verification notes:
Facts are not independently asserted beyond source reporting. The creative package uses fictional archetypes, generic paperwork, and no real legal documents or intimate imagery.
""")
    j("research/sources.json", {"created": NOW, "sources": SOURCES})
    j("strategy/candidate_board.json", {"run": RUN, "created": NOW, "scoring_scale": "0-10", "candidates": CANDIDATES, "winner": SLUG})
    w("strategy/winner_decision.md", f"""# Winner Decision - Run {RUN}

Winner: `{SLUG}`

Selected premise: A fictional reality-TV legal desk where a seven-million-dollar demand letter prints as a luxury receipt so long it becomes the furniture.

Score summary:
- Kardashian Seven Million Receipt Desk: 69
- Sabrina Yodel Translation Booth: 60, rejected for overlap and cultural-risk concerns
- Cyndi Vegas Heckler Receipt: 59
- Ticketmaster Fee Confessional Repair: 59, rejected for overlap with prior run_033
- Chappell Bodyguard Breakfast Map: 56

Decision: proceed with a still-image package only. Keep the joke on document absurdity and status theater.
""")
    j("strategy/phase_minus_one_worthiness_audit.json", {"final_decision": "PROCEED", "primary_vector": "TRACK_A_NEWSJACK", "reason": "fresh legal-pop-culture story with high-recognition names and a physical receipt gag", "score": 69})
    j("strategy/source_entropy_audit.json", {"native_entropy_score": 7, "entropy": ["secret deal", "demand letter", "seven-million figure", "legal filing made private paperwork public"], "recommended_strategy": "micro_spotlight"})
    j("strategy/humor_logic_bridge.json", {"straight_reality": "A reported legal demand over a private settlement dispute.", "comic_inversion": "The demand letter behaves like a retail receipt and overtakes the luxury office.", "first_frame_joke": "Seven-million-dollar paperwork is treated like a checkout slip.", "do_not_do": ["no real legal document", "no explicit intimate reference", "no exact likeness"]})
    j("strategy/tribe_meta_score.json", {"shareability": 9, "comment_prompt": 9, "remixability": 8, "first_frame_scroll_stop": 10, "total": 36, "notes": "Reality TV, celebrity law, and money absurdity all create comment lanes."})
    j("strategy/risk_taste_score.json", {"legal_risk": 6, "likeness_risk": 6, "sexual_content_risk": 7, "taste_verdict": "Proceed only with fictionalized archetypes and generic document visuals."})
    w("strategy/franchise_decision.md", "# Franchise Decision\n\nVerdict: `SERIES_FORMAT_CANDIDATE`\n\nA recurring celebrity-legal-receipt desk could work when the object is paperwork, invoices, or settlement math. Do not overuse one family or one lawsuit.")
    w("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT)
    w("storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)
    first = generate_image(FIRST_PROMPT, "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board = generate_image(BOARD_PROMPT, "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")
    image_ok = first["ok"] and board["ok"]
    if not image_ok:
        w("qc/IMAGE_GENERATION_BLOCKED_REPORT.md", f"""# Image Generation Blocked Report - Run {RUN}

At least one GPT Image generation call failed during this factory cycle.

First frame result: `{first['mode']}`
Shared Choices result: `{board['mode']}`

Known environment pressure: filesystem had about 231 MiB free before package creation.

Next action: free disk space and rerun the saved prompts, then update QC before video handoff.
""")
    j("chai/chai_shot_specs.json", {"run": RUN, "shots": [{"id": "shot_0001", "duration_sec": 6, "subject": "fictional reality-TV legal desk", "scene": "white marble office with endless receipt demand letter", "motion": "slow push-in", "camera": "vertical 9:16, 35mm", "critique": "Avoid real faces, real docs, and explicit source subject matter.", "revision": "Use generic archetypes and generic visible text only."}]})
    shot = {"run": RUN, "shot_id": "shot_0001", "source_frame": "frames/gpt_image_2/first_frame_v01.png", "duration_sec": 6, "camera": {"format": "9:16", "move": "slow push-in"}, "overlay": "SEVEN MILLION RECEIPT", "video_generation": "prohibited"}
    j("scene_json/shot_0001.json", shot)
    j("scene_json/shot_001.json", {**shot, "shot_id": "shot_001", "duration_sec": 10, "overlay": "THE DEMAND LETTER NEEDED A BAG"})
    j("handoffs/closed_tool_handoff.json", {"run": RUN, "video_generation_allowed": False, "input_assets": {"first_frame": first["path"], "shared_choices": board["path"]}, "manual_note": "Human review required before any later closed-tool workflow."})
    w("handoffs/grok_agent_prompt.md", "# Grok Agent Prompt - Run 042\n\nDo not generate video in this automation. Use the saved first-frame and Shared Choices prompts/assets as reference for a fictionalized celebrity-legal receipt gag. No real faces, names, logos, court documents, or intimate imagery.")
    w("captions/instagram_caption.md", "When the demand letter prints like a luxury receipt.\n\nReported context only; parody uses fictional characters and generic documents.\n\n#SGFLIX #PopCultureLaw #RealityTVParody #Satire #ReceiptDesk")
    w("distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Reels/TikTok after human visual approval.\n\nHook overlay: `SEVEN MILLION RECEIPT`\n\nDo not auto-post. Do not route to video until still QC passes.")
    w("skool/case_study.md", "# Skool Case Study\n\nThis run turns abstract celebrity legal math into a physical prop. The joke is not the underlying intimate dispute; it is the absurd scale of status paperwork becoming a receipt that overtakes the room.")
    j("manifests/asset_manifest.json", {"run": RUN, "created": NOW, "assets": [first, board, {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt"}, {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt"}], "video_generation": "not_performed"})
    w("qc/first_frame_v01_qc.md", f"# First Frame QC - Run {RUN}\n\nAsset: `frames/gpt_image_2/first_frame_v01.png`\n\nGeneration mode: `{first['mode']}`\n\nVerdict: {'usable for internal review pending human likeness/text check' if first['ok'] else 'blocked; rerun saved prompt after resolving generation issue'}.\n\nCautions: no real likeness, no real legal document, no explicit intimate references, no logos.")
    w("qc/shared_choices_v01_qc.md", f"# Shared Choices QC - Run {RUN}\n\nAsset: `storyboards/shared_choices/shared_choices_v01.png`\n\nGeneration mode: `{board['mode']}`\n\nVerdict: {'usable as private director-bible reference pending human text check' if board['ok'] else 'blocked; rerun saved prompt after resolving generation issue'}.\n\nCautions: generated microtext is not final copy.")
    status = "complete_for_factory_cycle" if image_ok else "BLOCKED_IMAGE_GENERATION"
    j(f"RUN_{RUN}_MASTER_PACKAGE.json", {"run": RUN, "slug": SLUG, "title": "Seven Million Receipt Desk", "status": status, "created": NOW, "selected_after_research": True, "premise": CANDIDATES[0]["premise"], "generated_stills": {"first_frame": first, "shared_choices": board}, "hard_stop_compliance": {"called_video_generation_tool": False, "requested_video_render": False, "auto_posted": False, "overwrote_approved_assets": False}, "next_human_action": "Review generated stills if present; if blocked, free disk space and rerun saved prompts."})
    w("README.md", f"# RUN {RUN} MASTER PACKAGE - Seven Million Receipt Desk\n\nStatus: `{status}`; no video footage generated.\n\nResearch topic: late-April/early-May 2026 celebrity legal and pop-culture conflict scan.\n\nSelected premise: fictional reality-TV legal desk where a seven-million-dollar demand letter becomes an endless luxury receipt.\n\nNext human action: review the stills if generated; if blocked, free disk space and rerun the saved GPT Image prompts.")
    w("FACTORY_RUN_STATUS.md", f"""# Factory Run Status

Run: {RUN}
Slug: `{SLUG}`
Status: `{status}`
Created: {NOW}

Order of operations:
1. Research intake completed first.
2. Fresh candidate board built from current web/source context.
3. Candidates scored before selection.
4. Winner selected before creating `run_{RUN}_{SLUG}`.
5. Still image prompts saved and GPT Image API attempted.

Generated still paths:
- `{first['path']}`: `{first['mode']}`
- `{board['path']}`: `{board['mode']}`

Missing files:
None from the metadata package checklist. Image binaries are missing if their generation mode is blocked.

Post-ready exports:
Captions and post plan only; no rendered video export.

QC failures:
{'None known before human review.' if image_ok else 'Image generation blocked; see qc/IMAGE_GENERATION_BLOCKED_REPORT.md.'}

High-risk issues:
- Likeness risk around real public figures.
- Legal-risk if real documents or exact claims are recreated.
- Taste risk around the underlying intimate-material dispute; keep the joke on paperwork.

Exact next human action:
Review generated stills if present; if blocked, free disk space and rerun `frames/gpt_image_2/first_frame_v01_prompt.md` and `storyboards/shared_choices/shared_choices_v01_prompt.md`.
""")

if __name__ == "__main__":
    main()
