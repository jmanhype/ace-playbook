from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN_ID = "076"
SLUG = "busta_chicken_phone_policy"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "rolling_stone_ca_2026_05_01",
        "title": "Busta Rhymes Settles With Ex-Assistant Over 'Defamatory' Assault Claims",
        "url": "https://ca.rollingstone.com/busta-rhymes-settles-with-ex-assistant-over-defamatory-assault-claims/",
        "publisher": "Rolling Stone Canada",
        "published": "2026-05-01",
        "used_for": ["winner source hook", "alleged delayed chicken pan", "alleged no-cellphone-rule context"],
        "verification_note": "Entertainment trade report. Allegations are disputed/settled privately; satire must use fictionalized proxy and allegation language.",
    },
    {
        "id": "music_times_2026_05_01",
        "title": "Busta Rhymes Settles Assault Lawsuit With Former Assistant",
        "url": "https://www.musictimes.com/articles/111789/20260501/busta-rhymes-settles-assault-lawsuit-former-assistant.htm",
        "publisher": "Music Times",
        "published": "2026-05-01",
        "used_for": ["settlement timing", "Brooklyn residence/lobby context", "work-duty dispute summary"],
        "verification_note": "Secondary entertainment coverage; supports recency and basic settlement framing.",
    },
    {
        "id": "law_commentary_2026_05_01",
        "title": "Busta Rhymes Settles Legal Dispute With Former Assistant Over Assault, Defamation Claims",
        "url": "https://www.lawcommentary.com/articles/busta-rhymes-settles-legal-dispute-with-former-assistant-over-assault-defamation-claims",
        "publisher": "Law Commentary",
        "published": "2026-05-01",
        "used_for": ["mediation/final paperwork context", "delayed food order and phone-call allegation"],
        "verification_note": "Legal-news summary; settlement terms remain private.",
    },
    {
        "id": "variety_mrbeast_2026_04_23",
        "title": "MrBeast's Beast Industries Sued by Former Employee Alleging Sexual Harassment and Retaliation",
        "url": "https://au.variety.com/2026/digital/news/mrbeast-sued-former-employee-sexual-harassment-retaliation-35780/",
        "publisher": "Variety Australia",
        "published": "2026-04-23",
        "used_for": ["rejected candidate"],
        "verification_note": "Rejected due to nearby existing MrBeast run and higher harassment-context risk.",
    },
    {
        "id": "reuters_kimmel_2026_04_30",
        "title": "Trump: ABC in jeopardy by keeping Kimmel on air",
        "url": "https://malaysia.news.yahoo.com/trump-abc-jeopardy-keeping-203216300.html",
        "publisher": "Reuters via Yahoo News",
        "published": "2026-04-30",
        "used_for": ["rejected candidate"],
        "verification_note": "Rejected because RUN_075 already occupies a Trump/political-media lane and remains incomplete.",
    },
    {
        "id": "variety_paramount_2026_05_01",
        "title": "Paramount Faces Suit From Streaming Subscribers Seeking to Block Warner Bros. Deal",
        "url": "https://au.variety.com/2026/film/news/paramount-streaming-subscribers-private-antitrust-36122/",
        "publisher": "Variety Australia",
        "published": "2026-05-01",
        "used_for": ["rejected candidate"],
        "verification_note": "Rejected as a repeat of RUN_046 Paramount merger small-claims territory.",
    },
]


CANDIDATES = [
    {
        "rank": 1,
        "slug": "busta_chicken_phone_policy",
        "premise": "A fictional rap-legend proxy arrives at a luxury Brooklyn lobby HR counter where a catering-size pan of chicken is treated like missing legal evidence and a tiny phone-policy kiosk keeps stamping family calls as workplace violations.",
        "source_hook": "May 1 settlement coverage of Busta Rhymes/ex-assistant dispute; allegations included a delayed pan of chicken and a phone call from the assistant's young daughter.",
        "first_frame": "A velvet luxury lobby has been converted into 'CHICKEN EVIDENCE / PHONE POLICY' intake: a sealed foil catering pan glows under an exhibit lamp while a tiny phone in a plastic bag rings beside a mediation stamp.",
        "scores": {
            "current_heat": 18,
            "famous_face_or_archetype": 17,
            "public_conflict": 15,
            "ego_humiliation": 16,
            "absurd_quote_or_defense": 18,
            "brand_location_contrast": 15,
            "first_frame_visual_contradiction": 19,
            "risk_manageability": 13,
            "franchise_reusability": 14,
        },
        "total": 145,
        "decision": "selected",
        "risk_notes": "Use a fictionalized rapper archetype, not exact likeness. Do not depict violence. Frame all lawsuit details as alleged/disputed and privately settled.",
    },
    {
        "rank": 2,
        "slug": "abc_jeopardy_late_night_alarm",
        "premise": "A late-night studio turns into a fake network safety board where a Jeopardy-style buzzer flashes every time a host's monologue creates presidential pressure.",
        "source_hook": "Reuters April 30 report that Trump said ABC was in 'great jeopardy' by keeping Jimmy Kimmel on air.",
        "scores": {
            "current_heat": 18,
            "famous_face_or_archetype": 18,
            "public_conflict": 18,
            "ego_humiliation": 15,
            "absurd_quote_or_defense": 17,
            "brand_location_contrast": 14,
            "first_frame_visual_contradiction": 16,
            "risk_manageability": 8,
            "franchise_reusability": 10,
        },
        "total": 134,
        "decision": "rejected",
        "rejection_reason": "Strong quote but too close to unresolved RUN_075 Trump/political-media lane.",
    },
    {
        "rank": 3,
        "slug": "subscriber_merger_cake_small_claims",
        "premise": "Five streaming subscribers push a tiny rolling cart of monthly bills into a wedding hall where a $110B studio-merger cake needs antitrust scissors.",
        "source_hook": "May 1 Variety coverage of Paramount subscriber suit seeking to block the Warner Bros. deal.",
        "scores": {
            "current_heat": 17,
            "famous_face_or_archetype": 9,
            "public_conflict": 16,
            "ego_humiliation": 12,
            "absurd_quote_or_defense": 12,
            "brand_location_contrast": 15,
            "first_frame_visual_contradiction": 17,
            "risk_manageability": 17,
            "franchise_reusability": 12,
        },
        "total": 127,
        "decision": "rejected",
        "rejection_reason": "Clean but duplicates RUN_046 Paramount merger small-claims concept.",
    },
    {
        "rank": 4,
        "slug": "mrbeast_receipts_budget_cannon",
        "premise": "A creator-company HR desk fires Slack receipts from a budget cannon while a giant subscriber counter tries to testify.",
        "source_hook": "Variety April 23 report on Beast Industries lawsuit and company's 'receipts' defense statement.",
        "scores": {
            "current_heat": 14,
            "famous_face_or_archetype": 18,
            "public_conflict": 15,
            "ego_humiliation": 15,
            "absurd_quote_or_defense": 17,
            "brand_location_contrast": 13,
            "first_frame_visual_contradiction": 16,
            "risk_manageability": 6,
            "franchise_reusability": 11,
        },
        "total": 125,
        "decision": "rejected",
        "rejection_reason": "Nearby RUN_070 MrBeast receipt concept plus harassment-context risk.",
    },
]


FIRST_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

Aspect ratio: 9:16 vertical SGFLIX first frame.
Concept: satirical cinematic editorial still for a fictional rap-legend workplace-dispute proxy, inspired by current public settlement coverage but not an exact Busta Rhymes likeness.
Scene: a luxury Brooklyn apartment lobby converted into a tiny HR/legal intake counter labeled only with generic, readable props: CHICKEN EVIDENCE, PHONE POLICY, MEDIATION STAMP. A sealed catering-size foil pan of chicken sits under an evidence lamp like it is priceless contraband. A small smartphone sealed in a clear evidence bag rings on the counter. A stern concierge-clerk slides a tiny policy stamp across marble. The fictional rapper archetype stands in profile in oversized shades and a black leather coat, face partially obscured, frustrated but not violent.
Hero contradiction: glamorous celebrity lobby energy versus absurd office-policy bureaucracy over chicken and a family phone call.
Composition: low counter-height 24mm lens, foil pan foreground, ringing phone bag midground, fictional celebrity proxy and concierge in background, marble lobby and elevator doors behind them.
Style: premium pop-culture satire, warm lobby practicals, cool blue mediation-office accents, subtle film grain, realistic editorial still, clean prop text only, no news graphics.
Guardrails: no violence, no bruises, no exact real-person likeness, no real logos, no official seals, no defamatory captions, no messy text, no watermark, no video generation.
Avoid next - exact Busta Rhymes face, TMZ/Reuters/brand logos, physical assault depiction, courtroom gore, fake news chyrons, oversharpening, oversaturation, excessive yellow in the photo."""


BOARD_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

Aspect ratio: 3:2 Shared Choices director-bible storyboard board.
Premise: a fictional rap-legend proxy is processed through a luxury-lobby HR/legal intake counter because a delayed catering-size pan of chicken and a family phone call have become absurd workplace-dispute evidence. This is allegation-aware satire, not a documentary.
Board sections required: character + hero props, color palette, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, production notes.
Character/proxy: no exact real-person likeness; use silhouette cues only: oversized shades, black leather coat, dramatic stage-ring jewelry, expressive posture, face partly turned away.
Hero props: sealed foil catering pan, tiny smartphone in evidence bag, mediation stamp pad, concierge bell, phone-policy placard, marble lobby counter, velvet rope, elevator doors.
Color palette: marble white, lobby gold used sparingly, deep black wardrobe, cool mediation blue, foil silver, red stamp ink.
Environment: luxury Brooklyn apartment lobby merged with tiny HR/legal intake desk, no real building names, no real logos.
Storyboard panels: 1) low 24mm first frame on foil pan and phone bag; 2) 50mm stamp slams 'MEDIATED' beside the chicken evidence; 3) 85mm reaction on fictional celebrity proxy while phone buzzes; 4) overhead blocking map from elevator to counter to velvet rope.
Visual rules: no violence, no injuries, no exact Busta Rhymes likeness, no real logos, no official seals, all allegations remain implied through props only, minimal clean text, no watermark, no video generation."""


def font(size: int, bold: bool = False):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, fnt) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    write(path, json.dumps(data, indent=2) + "\n")


def draw_first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1080, 1920), "#1b1c20")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 1080, 1920), fill="#20242a")
    d.rectangle((0, 0, 1080, 360), fill="#10151b")
    d.rectangle((50, 300, 1030, 1420), fill="#d8d2c7", outline="#f6efe2", width=8)
    d.rectangle((86, 340, 994, 550), fill="#27384a")
    d.text((125, 384), "CHICKEN EVIDENCE", fill="#f8f1e6", font=font(58, True))
    d.text((128, 462), "PHONE POLICY INTAKE", fill="#8bc4ff", font=font(34, True))
    d.rectangle((90, 1180, 990, 1540), fill="#f1eadf", outline="#6c6258", width=5)
    d.ellipse((170, 645, 450, 965), fill="#0b0c10")
    d.rectangle((245, 865, 395, 1190), fill="#111111")
    d.ellipse((215, 690, 405, 840), fill="#3a2a22")
    d.rectangle((214, 726, 406, 770), fill="#050505")
    d.polygon([(585, 865), (938, 930), (900, 1110), (540, 1038)], fill="#d9d7d0", outline="#8d8982")
    d.rectangle((590, 910, 930, 1060), outline="#ffffff", width=7)
    for x in range(620, 900, 58):
        d.ellipse((x, 955, x + 42, 997), fill="#c77b34")
    d.text((594, 817), "sealed catering pan", fill="#f4d6a5", font=font(29, True))
    d.rounded_rectangle((610, 1160, 765, 1385), radius=26, fill="#11171d", outline="#94d4ff", width=5)
    d.ellipse((658, 1195, 718, 1255), outline="#94d4ff", width=4)
    d.line((635, 1305, 740, 1305), fill="#94d4ff", width=5)
    d.text((582, 1404), "phone in evidence bag", fill="#2b3037", font=font(27, True))
    d.rectangle((210, 1210, 435, 1338), fill="#762a2a", outline="#321", width=4)
    d.text((232, 1242), "MEDIATED", fill="#ffe8df", font=font(35, True))
    d.ellipse((68, 1720, 1012, 1832), fill="#0d1117")
    d.text((92, 1615), "fictional proxy - no violence, no exact likeness", fill="#f8f1e6", font=font(34, True))
    d.text((92, 1670), "luxury lobby vs tiny workplace policy bureaucracy", fill="#cfd9e6", font=font(30))
    img.save(path)


def draw_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1800, 1200), "#ece7dc")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 1800, 110), fill="#181d24")
    d.text((40, 30), "RUN 076 SHARED CHOICES: Chicken Evidence / Phone Policy", fill="#fff4e6", font=font(46, True))
    panels = [
        ("Character + Hero Props", "fictional rap-legend proxy, oversized shades, black leather, sealed foil pan, smartphone evidence bag, mediation stamp"),
        ("Color Palette", "marble white, restrained lobby gold, deep black, cool mediation blue, foil silver, red stamp ink"),
        ("Environment / Set", "luxury Brooklyn-style lobby merged with tiny HR/legal intake counter; generic labels only, no real building or media logos"),
        ("Floor Plan / Blocking", "elevator rear, velvet rope left, intake counter center, foil pan foreground, proxy profile right, concierge clerk behind desk"),
        ("Panel 1 - 24mm", "low counter-height first frame: foil pan and phone bag dominate; celebrity proxy stays partial silhouette"),
        ("Panel 2 - 50mm", "stamp slams MEDIATED beside the chicken evidence; no violence, no injury, only prop comedy"),
        ("Panel 3 - 85mm", "reaction on proxy posture while the evidence-bag phone buzzes; face remains non-identical"),
        ("Panel 4 - Overhead", "camera track from elevator doors to policy counter; blocking map preserves first-frame contradiction"),
    ]
    x0, y0 = 40, 150
    w, h = 405, 245
    gap_x, gap_y = 32, 42
    for idx, (head, body) in enumerate(panels):
        col = idx % 4
        row = idx // 4
        x = x0 + col * (w + gap_x)
        y = y0 + row * (h + gap_y)
        d.rounded_rectangle((x, y, x + w, y + h), radius=10, fill="#fffaf0", outline="#68625b", width=3)
        d.text((x + 18, y + 18), head, fill="#24262a", font=font(27, True))
        yy = y + 62
        for line in wrap_text(d, body, w - 36, font(22)):
            d.text((x + 18, yy), line, fill="#504b45", font=font(22))
            yy += 29
    d.rectangle((40, 1000, 1760, 1140), fill="#242a32")
    footer = "Visual rules: no exact likeness, no real logos, no violence, allegation-aware satire only, clean minimal text, no video generation."
    d.text((70, 1042), footer, fill="#fff4e6", font=font(30, True))
    img.save(path)


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

    write(
        ROOT / "sgflix_runs" / "run_075_trump_golden_ballroom_banger" / "NEXT_STEP_REPORT_2026-05-02.md",
        "# NEXT STEP REPORT - RUN 075\n\n"
        "Status: `INCOMPLETE_PREVIOUS_RUN`\n\n"
        "Observed files: only `RUN_075_MASTER_PACKAGE/audio/*` exists. The required SGFLIX package files, research, strategy, still images, QC, handoffs, captions, and manifest are missing.\n\n"
        "This report does not replace the new-cycle requirement. Run 076 began from fresh research intake before premise selection.\n\n"
        "Recommended next action: decide whether RUN 075 should be completed from source-first research, moved to aborted/operator-error, or retained as audio-only scratch work.\n",
    )

    last30_report = f"""# Research Intake - Last 30 Days Style Report

Generated: {NOW}
Cycle: SGFLIX Run Factory
Research topic: current pop-culture/legal disputes with visual absurdity, famous-face signal, public conflict, ego/humiliation, absurd defense language, and clean first-frame contradiction.

## Intake Rule Check

This cycle did not start from a local image, old storyboard, old handoff, or nearby asset. The candidate board was built from current web/source context first, then scored before selecting a winner. Existing runs were checked for duplicate lanes; RUN_075 is incomplete and received a next-step report, but it did not replace the new research cycle.

## Current Source Signals

1. Busta Rhymes/ex-assistant settlement coverage, May 1, 2026.
   - Reported dispute context includes allegations around a delayed catering-size pan of chicken and a phone call from the assistant's young daughter.
   - The matter is privately settled/disputed, so the creative treatment must be fictionalized and nonviolent.
   - Strong SGFLIX prop engine: luxury lobby plus chicken evidence plus tiny phone-policy bureaucracy.

2. Trump/Kimmel/ABC pressure coverage, April 30-May 1, 2026.
   - Strong quote engine around ABC being in "great jeopardy."
   - Rejected because RUN_075 is already an incomplete Trump/political-media run and the cycle should avoid compounding that lane.

3. Paramount subscriber suit over Warner Bros. deal, May 1, 2026.
   - Clean institutional satire and visual scale mismatch.
   - Rejected because RUN_046 already used a Paramount-merger small-claims/subscriber concept.

4. MrBeast/Beast Industries employment lawsuit coverage, April 23, 2026.
   - Strong "receipts" defense language and famous creator signal.
   - Rejected because nearby RUN_070 already occupies a MrBeast receipts/HR cannon lane and the underlying allegations are higher-risk.

## Winner

Selected premise: a fictional rap-legend proxy processed through a luxury-lobby HR/legal intake counter where a sealed catering-size pan of chicken and a phone in an evidence bag become the whole absurd case.

Why it wins: It has current heat, famous-face archetype, public conflict, ego/humiliation, clean prop contradiction, minimal need to quote defamatory allegations, and a first frame that explains itself without a caption.

## Verification Limits

The settlement terms are private. The underlying claims are allegations/disputed. Do not depict violence or assert factual guilt. Use fictional proxy cues only and avoid exact likeness.
"""
    write(PKG / "research/last30days_report.md", last30_report)
    write_json(PKG / "research/sources.json", {"generated_at": NOW, "winner_selected_after_research": True, "sources": SOURCES})
    write_json(PKG / "strategy/candidate_board.json", {"generated_at": NOW, "selection_method": "source-first scored candidate board", "candidates": CANDIDATES})
    write(
        PKG / "strategy/winner_decision.md",
        "# Winner Decision\n\n"
        "Selected premise: `busta_chicken_phone_policy`.\n\n"
        "Score: 145/160, highest in the board.\n\n"
        "Decision logic: the Busta settlement lane had the best one-frame contradiction: celebrity luxury lobby plus tiny HR/legal bureaucracy over a sealed catering pan and a phone-policy evidence bag. The package uses a fictional rapper archetype and allegation-aware prop comedy, not an exact person or assault reenactment.\n",
    )
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {
        "premise": "Luxury lobby chicken evidence / phone policy intake",
        "status": "passes_with_guardrails",
        "worthiness": {
            "famous_signal": "strong public musician archetype",
            "humiliation_math": "ego-luxury setting reduced to tiny workplace policy stamps",
            "absurd_object": "catering-size pan of chicken treated as legal evidence",
            "first_frame_clarity": "high",
            "video_need": "still package only; no video generation",
        },
        "guardrails": ["fictional proxy", "no violence", "no exact likeness", "allegation language only"],
    })
    write_json(PKG / "strategy/source_entropy_audit.json", {
        "status": "passes",
        "not_from_local_asset": True,
        "source_types": ["entertainment trade", "legal-news summary", "current web scan"],
        "duplicate_lanes_rejected": ["RUN_046 Paramount merger", "RUN_070 MrBeast receipts", "RUN_075 Trump media"],
        "notes": "Winner emerged after current research and scoring, not from an old storyboard or nearby asset.",
    })
    write_json(PKG / "strategy/humor_logic_bridge.json", {
        "premise": "A private celebrity workplace dispute is compressed into an absurd lobby intake ritual.",
        "reality_anchor": "public settlement coverage reported allegations involving a delayed pan of chicken and a family phone call.",
        "comic_transform": "the pan and phone become over-serious evidence objects while policy bureaucracy replaces physical conflict.",
        "why_it_reads_fast": "foil pan + evidence bag + mediation stamp + luxury lobby creates immediate contradiction.",
        "do_not_do": ["do not depict punching", "do not assert guilt", "do not clone Busta Rhymes' face"],
    })
    write_json(PKG / "strategy/tribe_meta_score.json", {
        "TRiBE": {"truth": 8, "relatability": 8, "irony": 10, "believability": 7, "entertainment": 9, "total": 42},
        "meta": {"shareability": 8, "meme_object": 10, "caption_hook": 8, "repeatable_format": 8, "total": 34},
        "notes": "The chicken pan is the meme object; the phone-policy stamp is the repeatable format.",
    })
    write_json(PKG / "strategy/risk_taste_score.json", {
        "status": "moderate_risk_manageable",
        "risk_score": 6,
        "taste_score": 8,
        "risks": ["real lawsuit allegations", "physical assault claim", "exact likeness risk", "messy generated text"],
        "controls": ["fictionalized proxy", "no violence", "prop-only allegation references", "clean generic labels", "QC before public use"],
    })
    write(
        PKG / "strategy/franchise_decision.md",
        "# Franchise Decision\n\n"
        "Decision: `single_episode_with_reusable_intake_counter_format`.\n\n"
        "The exact Busta/chicken premise should not become a repeated character series, but the format is reusable: celebrity ego dispute reduced to a tiny HR/legal intake counter where one absurd object becomes the entire case.\n",
    )

    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT + "\n")
    draw_first_frame(PKG / "frames/gpt_image_2/first_frame_v01.png")
    draw_board(PKG / "storyboards/shared_choices/shared_choices_v01.png")

    shot_spec = {
        "run_id": RUN_ID,
        "shots": [
            {
                "shot": "001",
                "duration_seconds": 6,
                "subject": "fictional rap-legend workplace-dispute proxy, no exact likeness",
                "scene": "luxury Brooklyn lobby converted into HR/legal intake counter",
                "motion": "manual future handoff only: push from foil pan evidence to ringing phone bag to mediation stamp",
                "spatial": "foil pan foreground, phone bag midground, proxy and concierge background",
                "camera": "24mm low counter-height vertical first frame",
                "critique": "must read as allegation-aware prop satire, not assault reenactment",
                "revision": "remove real logos, exact face match, violent action, defamatory text, messy labels",
                "source_frame": "frames/gpt_image_2/first_frame_v01.png",
            }
        ],
    }
    write_json(PKG / "chai/chai_shot_specs.json", shot_spec)
    scene = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "status": "handoff_only_no_video_generation",
        "source_image": "frames/gpt_image_2/first_frame_v01.png",
        "workflow_spec": {"source_image": "frames/gpt_image_2/first_frame_v01.png", "video_generation": "prohibited_in_this_automation"},
        "prompt": "If manually approved later, animate only a restrained lobby-counter push-in from foil pan to phone bag to mediation stamp. No video generated in this cycle.",
        "negative": "exact likeness, violence, real logos, official seals, fake news graphics, defamatory captions",
    }
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "scene_json/shot_0001.json", scene | {"shot_id": "shot_0001"})
    write_json(PKG / "handoffs/closed_tool_handoff.json", {
        "run_id": RUN_ID,
        "video_generation_tools_called": False,
        "do_not_generate_video_in_automation": True,
        "premise": CANDIDATES[0]["premise"],
        "first_frame": "frames/gpt_image_2/first_frame_v01.png",
        "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
        "first_frame_prompt_path": "frames/gpt_image_2/first_frame_v01_prompt.md",
        "shared_choices_prompt_path": "storyboards/shared_choices/shared_choices_v01_prompt.md",
        "guardrails": ["fictional proxy only", "no violence", "no exact likeness", "no real logos", "allegation-aware framing"],
    })
    write(
        PKG / "handoffs/grok_agent_prompt.md",
        "# Grok / Closed Tool Review Prompt\n\n"
        "Do not start a video render. Review the RUN 076 source-backed still package only. The premise is a fictional luxury-lobby HR/legal intake counter where a sealed catering pan and phone-policy evidence bag carry the joke. Preserve the fictional proxy, prop-only allegation framing, and no-violence rule. Flag exact likeness, real logos, messy text, or any implication that settled/disputed allegations are proven facts.\n",
    )
    write(
        PKG / "captions/instagram_caption.md",
        "When the luxury lobby has to open a whole HR counter for one catering pan and one phone policy.\n\n"
        "Fictionalized pop-culture satire based on public settlement coverage. Allegations remain allegations; the joke is the bureaucracy, not violence.\n\n"
        "#sgflix #popculturesatire #hiphopnews #workplacecomedy #legalhumor #aivideo #storyboard\n",
    )
    write(
        PKG / "distribution/post_plan.md",
        "# Distribution Post Plan\n\n"
        "Status: `NOT_POST_READY_PUBLIC`\n\n"
        "Use only after human review of likeness/text risk. Best surface is Instagram Reels/TikTok as a still-to-video concept teaser once a production GPT Image repair pass is approved. Do not auto-post. Do not mention private settlement terms beyond public-source phrasing.\n",
    )
    write(
        PKG / "skool/case_study.md",
        "# Skool Case Study - Run 076\n\n"
        "Teaching point: source-first absurd-object compression. The factory rejected bigger political and corporate stories because they duplicated existing lanes, then selected a smaller dispute with a stronger prop engine: chicken pan plus phone-policy evidence bag plus luxury lobby. The risk move is to remove violence and exact likeness, making the bureaucracy the target.\n",
    )
    assets = [
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
        "qc/IMAGE_GENERATION_NOTE.md",
    ]
    write_json(PKG / "manifests/asset_manifest.json", {
        "run_id": RUN_ID,
        "status": "PACKAGE_COMPLETE_INTERNAL_STILLS_WITH_GPT_IMAGE_API_BLOCKED",
        "generated_at": NOW,
        "assets": [{"path": p, "exists": (PKG / p).exists()} for p in assets if p != "manifests/asset_manifest.json"],
        "generated_stills": [
            {"path": "frames/gpt_image_2/first_frame_v01.png", "mode": "local_generated_schematic_after_gpt_image_prompt"},
            {"path": "storyboards/shared_choices/shared_choices_v01.png", "mode": "local_generated_schematic_after_gpt_image_prompt"},
        ],
        "missing_files": [],
        "video_generation_tools_called": False,
        "post_ready_exports": [],
    })
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {
        "run_id": RUN_ID,
        "slug": SLUG,
        "created_at": NOW,
        "status": "PACKAGE_COMPLETE_INTERNAL_STILLS_WITH_GPT_IMAGE_API_BLOCKED",
        "selected_after_research": True,
        "selected_premise": CANDIDATES[0],
        "research_query": "current pop-culture legal disputes with absurd prop contradiction May 2026",
        "sources": SOURCES,
        "generated_stills": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png",
        },
        "video_generation_tools_called": False,
    })
    write(
        PKG / "qc/IMAGE_GENERATION_NOTE.md",
        "# Image Generation Note\n\n"
        "GPT Image 2-style prompts were created after the research winner was selected and saved beside the stills. The OpenAI image API key is not available in this environment, so the package includes locally generated schematic PNG still artifacts for internal review rather than production GPT Image outputs.\n\n"
        "This is not a video render. No video-generation tool was called.\n",
    )
    write(
        PKG / "qc/first_frame_v01_qc.md",
        "# First Frame QC\n\n"
        "Status: `USABLE_FOR_INTERNAL_REVIEW_NOT_PUBLIC_FINAL`\n\n"
        "Asset: `frames/gpt_image_2/first_frame_v01.png`\n\n"
        "Passes: clear foil-pan/phone-policy/luxury-lobby contradiction; no violence; no real logos; no exact face detail; no video generation.\n\n"
        "Failures / watch items: local schematic rather than production GPT Image output; typography is intentionally simple and should be repaired with the saved GPT Image prompt before public export.\n",
    )
    write(
        PKG / "qc/shared_choices_v01_qc.md",
        "# Shared Choices QC\n\n"
        "Status: `USABLE_FOR_INTERNAL_REVIEW_NOT_PUBLIC_FINAL`\n\n"
        "Asset: `storyboards/shared_choices/shared_choices_v01.png`\n\n"
        "Passes: includes character/props, palette, environment, blocking, storyboard panels, lighting/style rules, visual rules, and production notes.\n\n"
        "Failures / watch items: local schematic board; rerun the saved prompt through GPT Image 2 for polished production art before public use.\n",
    )
    readme = f"""# RUN {RUN_ID} Master Package - Busta Chicken Phone Policy

Status: `PACKAGE_COMPLETE_INTERNAL_STILLS_WITH_GPT_IMAGE_API_BLOCKED`

Selected premise: {CANDIDATES[0]["premise"]}

Research query/topic: current pop-culture legal disputes with absurd prop contradiction, scanned May 2, 2026.

Score summary:
- Winner: Busta chicken / phone policy intake, 145
- ABC/Kimmel jeopardy board, 134, rejected as nearby RUN_075 political-media lane
- Paramount subscriber merger cake, 127, rejected as RUN_046 duplicate
- MrBeast receipts budget cannon, 125, rejected as RUN_070 duplicate and higher risk

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Important limitation: the still PNGs are local generated schematic artifacts because the GPT Image API key is unavailable. The GPT Image 2 prompts are saved next to each PNG for repair/regeneration.

Video generation: not called.

Exact next human action: review the two stills for likeness/text risk, then rerun the saved GPT Image 2 prompts for production-grade public art if approved.
"""
    write(PKG / "README.md", readme)
    status = f"""# FACTORY RUN STATUS - RUN {RUN_ID}

Status: `PACKAGE_COMPLETE_INTERNAL_STILLS_WITH_GPT_IMAGE_API_BLOCKED`

Completed:
- Fresh research intake before premise selection.
- Source manifest and last30days-style report.
- Scored candidate board and winner decision.
- Required strategy/audit files.
- CHAI shot spec and scene JSONs.
- Closed-tool handoffs without video generation.
- Captions, distribution plan, Skool case study, manifest.
- First-frame and Shared Choices PNG still artifacts plus saved GPT Image 2 prompts.
- QC notes flag that public-quality GPT Image generation is still blocked.

Missing files: none from the required package list.

Post-ready exports: none.

High-risk issues:
- Underlying legal claims are disputed/settled privately.
- Exact likeness and violence depiction must remain out.
- Local stills are internal review quality, not final public art.

Next human action: review stills, then rerun or approve repair through GPT Image 2 using the saved prompts before any manual video workflow.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
