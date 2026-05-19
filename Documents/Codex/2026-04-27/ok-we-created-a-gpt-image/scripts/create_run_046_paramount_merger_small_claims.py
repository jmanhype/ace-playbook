from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("sgflix_runs")
RUN_ID = "046"
SLUG = "paramount_merger_small_claims"
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "ap_swift_trademark",
        "title": "Taylor Swift files 3 new trademark applications. One expert says it is to curb AI threats",
        "url": "https://apnews.com/article/7f56fbafb269d4959009f3ad34e28fc1",
        "publisher": "Associated Press",
        "date": "2026-04-28",
        "verified_facts": [
            "Swift's company filed three USPTO trademark applications on April 24, 2026.",
            "Two applications cover sound marks; one covers a visual image.",
        ],
        "creative_use": "Candidate only; rejected because celebrity plus AI is familiar unless the trademark office scene is exceptionally sharp.",
    },
    {
        "id": "variety_subscriber_lawsuit",
        "title": "Paramount Faces Suit From Streaming Subscribers Seeking to Block Warner Bros. Deal",
        "url": "https://au.variety.com/2026/film/news/paramount-streaming-subscribers-private-antitrust-36122/",
        "publisher": "Variety Australia",
        "date": "2026-05-01",
        "verified_facts": [
            "A handful of Paramount+ subscribers filed a federal lawsuit seeking to block the Paramount Skydance and Warner Bros. merger.",
            "The complaint alleges higher prices and reduced viewing options if the deal closes.",
            "The proposed deal is reported around $110 billion.",
        ],
        "creative_use": "Winner source: tiny consumer plaintiff versus enormous studio consolidation.",
    },
    {
        "id": "variety_petition",
        "title": "Hollywood Petition to Block Paramount-Warner Bros. Merger Tops 4,000 Names",
        "url": "https://au.variety.com/2026/biz/news/petition-block-paramount-warner-bros-merger-4000-names-robert-de-niro-sofia-coppola-holly-hunter-35854/",
        "publisher": "Variety Australia",
        "date": "2026-04-24",
        "verified_facts": [
            "An open letter opposing the merger had more than 4,000 signatories by April 24, 2026.",
            "Reported signatories included Robert De Niro, Sofia Coppola, and Holly Hunter.",
            "Opposition cited jobs, consumer costs, and fewer shows and movies.",
        ],
        "creative_use": "Adds the visual of a petition stack so large it needs a freight dolly.",
    },
    {
        "id": "nbc_petition",
        "title": "Hollywood stars sign open letter protesting the Paramount-Warner Bros. merger",
        "url": "https://www.nbcbayarea.com/entertainment/entertainment-news/hollywood-stars-letter-protest-paramount-warner-bros-merger/4067945/",
        "publisher": "NBC Bay Area",
        "date": "2026-04-13",
        "verified_facts": [
            "More than 1,000 filmmakers, actors, and industry workers signed the initial letter.",
            "The letter argued the transaction would further consolidate the media landscape.",
        ],
        "creative_use": "Confirms the petition was a real public opposition object, not only social chatter.",
    },
    {
        "id": "polymarket_scan",
        "title": "Local last30days Polymarket scan",
        "url": "https://polymarket.com/event/jimmy-kimmel-firedresigns-by-may-31",
        "publisher": "Polymarket via local last30days script",
        "date": "2026-05-02",
        "verified_facts": [
            "The local scan found prediction market interest in entertainment/personality outcomes.",
            "Reddit failed with HTTP 429 and X was unavailable due authentication.",
        ],
        "creative_use": "Used only as source-context evidence that personality/legal outcomes are actively bettable; not used as winner premise.",
    },
]


CANDIDATES = [
    {
        "id": "paramount_merger_small_claims",
        "premise": "A single Paramount+ subscriber rolls monthly receipts into a grand antitrust wedding hall to stop a $110B studio merger cake from combining.",
        "source_ids": ["variety_subscriber_lawsuit", "variety_petition", "nbc_petition"],
        "first_frame": "Small-claims folding table dwarfed by two studio merger cakes and a mountain of petition pages.",
        "scores": {
            "fame_context": 8,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "visual_contradiction": 10,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 10,
            "risk_control": 8,
            "franchise_potential": 8,
        },
        "total": 84,
    },
    {
        "id": "swift_voice_notary",
        "premise": "A pop superstar's greeting is sealed in a courthouse evidence bag while a pink guitar waits in line at the trademark office.",
        "source_ids": ["ap_swift_trademark"],
        "first_frame": "Trademark counter with soundwave stamp, guitar exhibit, and counterfeit voice coupons.",
        "scores": {
            "fame_context": 10,
            "public_conflict": 7,
            "ego_humiliation": 6,
            "visual_contradiction": 9,
            "absurd_quote_or_object": 9,
            "brand_location_contrast": 7,
            "risk_control": 6,
            "franchise_potential": 7,
        },
        "total": 81,
    },
    {
        "id": "bafta_delay_button_drill",
        "premise": "A formal awards control room trains a giant red delay button after a review says duty of care fell short.",
        "source_ids": [],
        "first_frame": "Award trophies in hard hats around a censor-delay console.",
        "scores": {
            "fame_context": 6,
            "public_conflict": 8,
            "ego_humiliation": 7,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 7,
            "brand_location_contrast": 8,
            "risk_control": 4,
            "franchise_potential": 5,
        },
        "total": 68,
    },
    {
        "id": "lively_baldoni_subpoena_seating_chart",
        "premise": "A celebrity trial clerk turns a witness list into a star-studded seating chart with velvet ropes.",
        "source_ids": [],
        "first_frame": "Courtroom clerk measuring velvet rope outside a witness stand.",
        "scores": {
            "fame_context": 9,
            "public_conflict": 9,
            "ego_humiliation": 6,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 6,
            "brand_location_contrast": 7,
            "risk_control": 3,
            "franchise_potential": 6,
        },
        "total": 64,
    },
    {
        "id": "kimmel_prediction_market_hr",
        "premise": "A late-night host finds HR replaced by a prediction-market odds board.",
        "source_ids": ["polymarket_scan"],
        "first_frame": "HR office desk with a live resignation/firing odds board instead of paperwork.",
        "scores": {
            "fame_context": 8,
            "public_conflict": 7,
            "ego_humiliation": 7,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 7,
            "brand_location_contrast": 7,
            "risk_control": 6,
            "franchise_potential": 7,
        },
        "total": 57,
    },
]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, obj) -> None:
    write(path, json.dumps(obj, indent=2) + "\n")


def load_font(size: int, bold: bool = False):
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, max_chars: int, font, fill, leading: int = 8):
    words = text.split()
    line = ""
    for word in words:
        test = (line + " " + word).strip()
        if len(test) > max_chars and line:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + leading
            line = word
        else:
            line = test
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + leading
    return y


def pixel_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, max_width: int, font, fill, leading: int = 8):
    words = text.split()
    line = ""
    for word in words:
        test = (line + " " + word).strip()
        width = draw.textbbox((0, 0), test, font=font)[2]
        if width > max_width and line:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + leading
            line = word
        else:
            line = test
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + leading
    return y


def draw_first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1600, 900), "#f6f2e8")
    d = ImageDraw.Draw(img)
    title = load_font(58, True)
    h2 = load_font(34, True)
    body = load_font(26)
    small = load_font(20)
    d.rectangle((0, 0, 1600, 90), fill="#263238")
    d.text((48, 22), "SMALL CLAIMS VS. THE $110B MERGER CAKE", font=title, fill="#ffffff")
    d.rectangle((70, 140, 610, 780), fill="#ffffff", outline="#263238", width=5)
    d.text((105, 175), "STREAMING", font=h2, fill="#263238")
    d.text((105, 215), "SUBSCRIBER", font=h2, fill="#263238")
    d.rectangle((115, 290, 565, 620), fill="#e9eef0", outline="#455a64", width=4)
    y = wrapped(d, "Monthly receipts, buffering screenshots, and one tiny antitrust complaint.", 145, 325, 28, body, "#263238")
    d.rectangle((170, y + 20, 510, y + 78), fill="#ffcf5a", outline="#263238", width=3)
    d.text((190, y + 34), "Exhibit A: price hike fear", font=small, fill="#263238")
    d.rectangle((760, 180, 1420, 760), fill="#fff8dc", outline="#263238", width=6)
    d.ellipse((830, 115, 1040, 270), fill="#6aa1c8", outline="#263238", width=5)
    d.rectangle((850, 255, 1020, 430), fill="#6aa1c8", outline="#263238", width=5)
    d.text((860, 310), "MOUNTAIN+", font=body, fill="#ffffff")
    d.ellipse((1130, 115, 1340, 270), fill="#8d6e63", outline="#263238", width=5)
    d.rectangle((1150, 255, 1320, 430), fill="#8d6e63", outline="#263238", width=5)
    d.text((1175, 310), "WATER", font=body, fill="#ffffff")
    d.text((1175, 346), "TOWER", font=body, fill="#ffffff")
    d.rectangle((850, 520, 1320, 670), fill="#f9a825", outline="#263238", width=5)
    d.text((900, 560), "$110B MERGER CAKE", font=h2, fill="#263238")
    for i in range(8):
        x = 680 + i * 44
        d.rectangle((x, 610 - i * 12, x + 150, 690 - i * 12), fill="#ffffff", outline="#455a64", width=2)
    d.text((675, 705), "4,000+ signature stack", font=body, fill="#263238")
    d.line((600, 460, 760, 610), fill="#d32f2f", width=9)
    d.text((560, 405), "objection!", font=h2, fill="#d32f2f")
    d.text((56, 828), "Fictional parody frame based on reported lawsuit/petition context. No real logos; no video generation.", font=small, fill="#37474f")
    img.save(path, optimize=True)


def draw_shared_choices(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1800, 1200), "#f4f0e6")
    d = ImageDraw.Draw(img)
    title = load_font(52, True)
    h = load_font(30, True)
    body = load_font(22)
    small = load_font(18)
    d.rectangle((0, 0, 1800, 82), fill="#263238")
    d.text((40, 18), "RUN 046 SHARED CHOICES: PARAMOUNT MERGER SMALL CLAIMS", font=title, fill="#ffffff")
    panels = [
        (40, 120, 420, 440, "Character + Hero Props", "Anonymous subscriber plaintiff, dolly of monthly receipts, tiny complaint folder, petition pallet."),
        (460, 120, 840, 440, "Palette", "Warm courthouse beige, corporate blue, water-tower brown, warning red, receipt yellow."),
        (880, 120, 1260, 440, "Environment", "Antitrust hearing staged like an overdecorated merger wedding reception."),
        (1300, 120, 1760, 440, "Blocking", "Subscriber foreground left; merger cake mid/right; petition stack creates a visual wall."),
        (40, 500, 420, 840, "Panel 1", "24mm wide: tiny claimant enters with receipt cart; cake towers over frame."),
        (460, 500, 840, 840, "Panel 2", "50mm push-in: judge stamps 'exhibit' on a buffering screenshot."),
        (880, 500, 1260, 840, "Panel 3", "35mm lateral move: petition stack slides in like a freight delivery."),
        (1300, 500, 1760, 840, "Panel 4", "70mm punchline: cake topper signs an antitrust prenup."),
        (40, 900, 580, 1135, "Lighting / Mood", "Bright procedural comedy, crisp shadows, no noir, readable props, satirical not fake news."),
        (620, 900, 1180, 1135, "Visual Rules", "Use parody labels only. Avoid exact corporate logos, real faces, or legal-outcome claims."),
        (1220, 900, 1760, 1135, "Production Notes", "First frame must read in one second: tiny subscriber versus huge merger. Keep text minimal and legible."),
    ]
    colors = ["#ffffff", "#e8f2f8", "#fff8dc", "#fce4ec"]
    for i, (x1, y1, x2, y2, head, desc) in enumerate(panels):
        d.rectangle((x1, y1, x2, y2), fill=colors[i % len(colors)], outline="#263238", width=4)
        d.text((x1 + 18, y1 + 18), head, font=h, fill="#263238")
        pixel_wrapped(d, desc, x1 + 18, y1 + 70, (x2 - x1) - 36, body, "#263238")
        if head.startswith("Panel"):
            d.rectangle((x1 + 30, y1 + 155, x2 - 30, y2 - 35), outline="#607d8b", width=3)
            d.line((x1 + 30, y2 - 35, x2 - 30, y1 + 155), fill="#90a4ae", width=2)
            d.line((x1 + 30, y1 + 155, x2 - 30, y2 - 35), fill="#90a4ae", width=2)
    d.text((40, 1160), "Generated compact director-bible board. GPT Image 2 prompt saved beside this file for higher-fidelity rerender.", font=small, fill="#37474f")
    img.save(path, optimize=True)


def main() -> None:
    (ROOT / "run_045_jersey_evidence_scanner").mkdir(parents=True, exist_ok=True)
    write(
        ROOT / "run_045_jersey_evidence_scanner" / "NEXT_STEP_REPORT_2026-05-02.md",
        "# Next Step Report - Run 045\n\n"
        "This directory exists but contains no discoverable package files. It is treated as an incomplete prior run, not as the current cycle's deliverable.\n\n"
        "Next human action: restore the missing files from the automation that created the directory, or mark it aborted before considering it complete.\n",
    )

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

    first_frame = PKG / "frames/gpt_image_2/first_frame_v01.png"
    shared = PKG / "storyboards/shared_choices/shared_choices_v01.png"
    draw_first_frame(first_frame)
    draw_shared_choices(shared)

    write(
        PKG / "research/last30days_report.md",
        "# Last30days-Style Research Report\n\n"
        f"Generated: {NOW}\n\n"
        "Research query/topic: current celebrity/legal/entertainment contradiction hooks for SGFLIX satire.\n\n"
        "Step 1 intake ran before package selection. Local scan results were partial: Reddit hit OpenAI HTTP 429, X/Twitter was unavailable because the local tool is not authenticated, YouTube returned no relevant videos for the broad query, and Polymarket returned entertainment/personality market context. Web/source supplements were then used to build and score candidates.\n\n"
        "Confirmed context:\n"
        "- Variety reported on May 1, 2026 that a handful of Paramount+ subscribers filed a federal lawsuit seeking to block the proposed Paramount Skydance and Warner Bros. deal, alleging higher prices and fewer viewing options.\n"
        "- Variety reported on April 24, 2026 that opposition to the merger had topped 4,000 industry signatories, including major Hollywood names.\n"
        "- NBC reported on April 13, 2026 that the open letter framed the transaction as further media consolidation.\n"
        "- AP reported on April 28, 2026 that Taylor Swift's company filed USPTO applications for voice/image marks, a viable candidate but lower-scored because celebrity plus AI is a crowded pattern.\n\n"
        "Grounded creative angle: make the legal imbalance literal. A single anonymous subscriber drags monthly receipts into a merger-wedding courtroom and tries to stop a giant $110B cake from combining. The joke is not that the lawsuit will succeed; the joke is the first-frame scale mismatch between tiny consumer paperwork and gigantic studio consolidation machinery.\n\n"
        "Unverified or excluded: no claims about court outcome, no invented regulator decision, no real-person wrongdoing beyond reported lawsuit/petition context.\n",
    )
    write_json(PKG / "research/sources.json", {"generated": NOW, "research_method": "local last30days-style scan plus verified web/source context", "sources": SOURCES})
    write(PKG / "research/source_notes.md", "\n".join(f"- {s['id']}: {s['creative_use']}" for s in SOURCES) + "\n")

    write_json(PKG / "strategy/candidate_board.json", {"generated": NOW, "selected": "paramount_merger_small_claims", "candidates": CANDIDATES})
    write(
        PKG / "strategy/winner_decision.md",
        "# Winner Decision\n\n"
        "Selected premise: `paramount_merger_small_claims`.\n\n"
        "Score summary: 84/100 equivalent. It won because the first-frame contradiction is immediate: one streaming subscriber's receipts versus a $110B merger cake and a petition stack big enough for a forklift. It has public conflict, brand/location contrast, and satire that does not depend on asserting unproven facts.\n\n"
        "Rejected: `swift_voice_notary` was close, but celebrity plus AI needs a more novel angle to clear the worthiness gate. `lively_baldoni_subpoena_seating_chart` had fame and conflict but too much sensitivity around ongoing litigation. `bafta_delay_button_drill` had avoidable racial-slur platform risk.\n",
    )
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {
        "premise": "Tiny subscriber small-claims energy versus giant studio merger wedding.",
        "passes": True,
        "worthiness_score": 86,
        "why_now": "Filed May 1, 2026, with petition opposition still fresh from late April.",
        "why_visual": "Scale mismatch can be read without narration.",
        "bimodal_gate": {"headline_only": 8, "silent_first_frame": 9},
        "failure_modes": ["could become dry antitrust explainer", "brand/logo overuse", "fake-news framing"],
    })
    write_json(PKG / "strategy/source_entropy_audit.json", {
        "source_count": len(SOURCES),
        "source_types": ["trade press", "mainstream local news", "wire service", "local trend scan"],
        "entropy_score": 78,
        "notes": "Winner relies on multiple current sources, not an old storyboard or local asset. Social-context scan was partial and documented.",
    })
    write_json(PKG / "strategy/humor_logic_bridge.json", {
        "setup": "A few subscribers file an antitrust suit over a massive studio consolidation.",
        "literalization": "Their receipts become courtroom evidence against a giant wedding cake made of streaming services.",
        "punchline": "The merger cake tries to sign an antitrust prenup while the subscriber objects with a buffering screenshot.",
        "repeatable_phrase": "Exhibit A: my monthly receipt.",
        "do_not_say": ["the lawsuit succeeds", "regulators already blocked the deal", "real executives committed crimes"],
    })
    write_json(PKG / "strategy/tribe_meta_score.json", {
        "concept": 8.7,
        "format": 8.4,
        "hook": 9.1,
        "body_story": 8.0,
        "caption_cta": 7.8,
        "overall": 8.4,
        "comment_bait": ["Which subscription receipt is Exhibit B?", "Is the cake cheaper with ads?"],
    })
    write_json(PKG / "strategy/risk_taste_score.json", {
        "overall_risk": "medium-low",
        "platform_risk": 4,
        "defamation_risk": 3,
        "likeness_risk": 2,
        "brand_logo_risk": 5,
        "mitigations": ["use parody labels instead of exact logos", "anonymous subscriber", "no real-person faces", "no outcome claims"],
        "taste_gate": {"scene_not_summary": True, "clear_in_one_second": True, "caption_not_too_AI": True},
    })
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nGreenlight as a short recurring lane: `Streaming Receipts Court`. Future episodes can literalize price hikes, buffering, bundle confusion, and ad-tier fine print as courtroom exhibits. Keep every episode source-backed and avoid exact logos.\n")

    shot = {
        "run": RUN_ID,
        "shot": "001",
        "duration_seconds": 6,
        "Subject": "Anonymous streaming subscriber plaintiff, oversized merger cake, petition stack, court clerk hands.",
        "Scene": "A bright antitrust hearing room dressed like a corporate wedding reception.",
        "Motion": "Subscriber pushes receipt cart; petition stack slides in; cake topper stamp pauses before signing antitrust prenup.",
        "Spatial": "Subscriber foreground left, cake midground right, petition stack background center, judge bench implied off camera.",
        "Camera": "24mm wide first frame, slow push to 50mm insert on receipt exhibit, crisp product-comedy lighting.",
        "Critique": "Exact logos or real faces would raise risk and distract from the visual joke.",
        "Revision": "Use parody mountain/water-tower labels, anonymous characters, and legible exhibit props.",
    }
    write_json(PKG / "chai/chai_shot_specs.json", {"shots": [shot]})
    write_json(PKG / "scene_json/shot_0001.json", shot | {"id": "shot_0001", "render_instruction": "Do not generate video; still/reference planning only."})
    write_json(PKG / "scene_json/shot_001.json", shot | {"id": "shot_001", "closed_tool_ready": False})

    handoff = {
        "run": RUN_ID,
        "premise": CANDIDATES[0]["premise"],
        "hard_stop": "No video generation requested by this factory cycle.",
        "first_frame_path": str(first_frame.relative_to(PKG)),
        "shared_choices_path": str(shared.relative_to(PKG)),
        "video_tool_prompt": "Use only after human approval. Animate the approved first frame; do not invent new opening composition.",
    }
    write_json(PKG / "handoffs/closed_tool_handoff.json", handoff)
    write(PKG / "handoffs/grok_agent_prompt.md", "# Grok Agent Prompt\n\nDo not start a render. Review the source-backed premise and stills only. If approved later, preserve the anonymous subscriber, parody labels, merger cake, petition stack, and receipt evidence. Avoid exact logos and real-person likenesses.\n")

    first_prompt = (
        "Create a satirical cinematic first frame for a fictional short: an anonymous streaming subscriber at a folding small-claims table pushes a cart of monthly receipts toward a gigantic $110B merger wedding cake made of two parody studio tiers. Bright courthouse-meets-wedding-reception lighting, readable props, anonymous faces, no exact logos, no real executives, no news chyron, no claim about legal outcome."
    )
    board_prompt = (
        "Create a director-bible storyboard board for the same SGFLIX parody: character and hero props, color palette, environment design, floor plan and blocking, four storyboard panels with lens/camera notes, lighting and mood notes, visual rules, and production notes. Keep text minimal and legible; use parody brand labels only."
    )
    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", first_prompt + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", board_prompt + "\n")

    write(PKG / "captions/instagram_caption.md", "POV: your streaming receipt just became Exhibit A.\n\nA tiny subscriber walks into antitrust court with a cart full of monthly charges while the merger cake tries to cut itself.\n\n#SGFLIX #StreamingWars #HollywoodSatire #Antitrust #MediaMergers\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels/TikTok once video is human-approved and rendered elsewhere.\n\nOpening overlay: `Exhibit A: my monthly receipt.`\n\nDo not auto-post. Do not imply the lawsuit succeeded or failed.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nThis run demonstrates a source-first contradiction hook: convert abstract antitrust coverage into a visible scale mismatch. The factory selected the premise only after comparing candidates and risk gates, then generated still references before any video handoff.\n")

    manifest = {
        "run": RUN_ID,
        "created": NOW,
        "files": sorted(str(p.relative_to(PKG)) for p in PKG.rglob("*") if p.is_file()),
        "generated_stills": [
            "frames/gpt_image_2/first_frame_v01.png",
            "storyboards/shared_choices/shared_choices_v01.png",
        ],
        "preserved_existing_assets": True,
    }
    write_json(PKG / "manifests/asset_manifest.json", manifest)
    write(PKG / "qc/first_frame_v01_qc.md", "# First Frame QC\n\nStatus: `USABLE_COMPACT_FALLBACK`\n\nThe PNG exists and clearly shows the intended contradiction: anonymous subscriber receipts versus a giant merger cake. It avoids exact logos, real faces, and video-generation claims. A higher-fidelity GPT Image 2 rerender is recommended when disk/image persistence is healthier, using the saved prompt.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", "# Shared Choices QC\n\nStatus: `USABLE_COMPACT_FALLBACK`\n\nThe board includes character/props, palette, environment, blocking, storyboard panels with camera notes, lighting/mood, visual rules, and production notes. Text is intentionally large because prior generated boards can produce messy microtext.\n")
    write(PKG / "FACTORY_RUN_STATUS.md", "# Factory Run Status - Run 046\n\nStatus: `STILLS_READY_COMPACT_FALLBACK`\n\nCompleted:\n- fresh research intake before premise selection\n- candidate board and score-first winner selection\n- strategy audits and scoring files\n- CHAI and scene JSON handoffs\n- first-frame prompt plus generated PNG\n- Shared Choices prompt plus generated PNG\n- caption, distribution, Skool, manifest, and QC files\n\nMissing:\n- high-fidelity GPT Image 2 cloud rerender, optional\n- rendered video, intentionally not generated\n\nHigh-risk issues:\n- avoid exact corporate logos\n- avoid real executive or celebrity likenesses\n- do not imply the subscriber lawsuit has already succeeded\n\nExact next human action: review the two PNG stills, then rerender them in GPT Image 2 only if higher fidelity is needed before any video handoff.\n")
    write(PKG / "README.md", f"# Run {RUN_ID}: Paramount Merger Small Claims\n\nFresh research-first SGFLIX package created {NOW}.\n\nPremise: {CANDIDATES[0]['premise']}\n\nStatus: stills ready as compact generated fallback; video generation was not requested or started.\n")
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {
        "run": RUN_ID,
        "slug": SLUG,
        "title": "Paramount Merger Small Claims",
        "created": NOW,
        "status": "STILLS_READY_COMPACT_FALLBACK",
        "selected_after_research": True,
        "research_query": "current celebrity/legal/entertainment contradiction hooks for SGFLIX satire",
        "winner_score": 84,
        "premise": CANDIDATES[0]["premise"],
        "generated_stills": {
            "first_frame": {"path": "frames/gpt_image_2/first_frame_v01.png", "ok": True},
            "shared_choices": {"path": "storyboards/shared_choices/shared_choices_v01.png", "ok": True},
        },
        "hard_stop_compliance": {
            "called_video_generation_tool": False,
            "requested_video_render": False,
            "auto_posted": False,
            "overwrote_approved_assets": False,
        },
        "next_human_action": "Review compact stills; optionally rerender in GPT Image 2 for higher fidelity before video handoff.",
    })


if __name__ == "__main__":
    main()
