from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "086"
SLUG = "pcd_peace_turnstile"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
MASTER = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


def write(rel: str, text: str) -> None:
    path = RUN_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(rel: str, data: object) -> None:
    path = RUN_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_master(rel: str, text: str) -> None:
    path = MASTER / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_master_json(rel: str, data: object) -> None:
    path = MASTER / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, size: int, fill: str, bold: bool = False, width: int | None = None) -> int:
    fnt = font(size, bold)
    lines = wrap(draw, text, fnt, width) if width else text.splitlines()
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(size * 1.23)
    return y


SOURCES = [
    {
        "id": "variety_pcd_maga_2026_03_25",
        "title": "Ex-Pussycat Doll Thinks She Was Not Asked to Return for 2026 Tour Because She's MAGA",
        "publisher": "Variety",
        "url": "https://au.variety.com/2026/music/global/pussycat-doll-maga-reunion-tour-bobby-kennedy-support-34586/",
        "published": "2026-03-25",
        "used_for": "Member allegation, tour context, and public quote framing.",
        "fact_status": "reported allegation by Jessica Sutta; do not state as proven motive",
    },
    {
        "id": "nme_protect_peace_2026_03_24",
        "title": "The Pussycat Dolls say not inviting former members on reunion tour was to protect our peace",
        "publisher": "NME",
        "url": "https://www.nme.com/news/music/the-pussycat-dolls-say-not-inviting-former-members-on-their-reunion-tour-was-to-protect-our-peace-3935843",
        "published": "2026-03-24",
        "used_for": "Public response phrase and Today Show awkward-interview framing.",
        "fact_status": "reported public interview response",
    },
    {
        "id": "us_weekly_pcd_reunion_2026_03_25",
        "title": "Ex-Pussycat Dolls Member Shares Why She's Not in Reunion",
        "publisher": "Us Weekly",
        "url": "https://www.usmagazine.com/entertainment/news/ex-pussycat-dolls-member-shares-why-shes-not-in-reunion/",
        "published": "2026-03-25",
        "used_for": "Second entertainment-source corroboration of Sutta's theory and trio lineup.",
        "fact_status": "reported theory and tour lineup",
    },
    {
        "id": "aftenposten_pcd_april_context_2026_04_05",
        "title": "Politikk splitter The Pussycat Dolls: Er Maga",
        "publisher": "Aftenposten",
        "url": "https://www.aftenposten.no/kultur/i/3ppazA/politikk-splitter-the-pussycat-dolls-er-maga",
        "published": "2026-04-05",
        "used_for": "Last-30-days-style scan anchor showing the story remained live in April.",
        "fact_status": "secondary coverage; do not rely on for new claims",
    },
    {
        "id": "variety_bieber_youtube_2026_04_12",
        "title": "Justin Bieber Favors Swag Songs in a Minimalist Coachella Set, but Also Revisits His Baby-Hood, Bingeing on Old YouTube Clips",
        "publisher": "Variety",
        "url": "https://au.variety.com/2026/music/news/justin-bieber-coachella-set-swag-youtube-videos-35180/",
        "published": "2026-04-12",
        "used_for": "Rejected candidate; strong visual but already covered by an official SGFLIX run.",
        "fact_status": "reported performance review/context",
    },
    {
        "id": "variety_reese_ai_2026_04_22",
        "title": "Reese Witherspoon Confronts Backlash Over AI Support",
        "publisher": "Variety",
        "url": "https://au.variety.com/2026/tv/news/reese-witherspoon-confronts-ai-backlash-35730/",
        "published": "2026-04-22",
        "used_for": "Rejected candidate; high quote clarity but duplicate official SGFLIX lane.",
        "fact_status": "reported social backlash and public response",
    },
]


CANDIDATES = [
    {
        "id": "pcd_peace_turnstile",
        "premise": "A reunion-tour security turnstile labels three open lanes as 'PCD Forever' while three glitter ID badges bounce back with 'protect our peace' stamps.",
        "source_ids": ["variety_pcd_maga_2026_03_25", "nme_protect_peace_2026_03_24", "us_weekly_pcd_reunion_2026_03_25", "aftenposten_pcd_april_context_2026_04_05"],
        "scores": {
            "famous_face_or_group": 8,
            "public_conflict": 9,
            "absurd_quote_or_defense": 9,
            "first_frame_visual_contradiction": 10,
            "audio_hook": 8,
            "freshness": 7,
            "duplicate_penalty": 0,
            "risk_penalty": -2,
        },
        "total": 49,
        "risk": "Moderate. Keep motive as alleged/theorized, target reunion PR choreography rather than personal politics or protected traits.",
    },
    {
        "id": "bieber_youtube_premium_receipt",
        "premise": "A Coachella headline desk prints a giant YouTube Premium receipt while a pop-star proxy performs a nostalgic laptop karaoke audit.",
        "source_ids": ["variety_bieber_youtube_2026_04_12"],
        "scores": {
            "famous_face_or_group": 10,
            "public_conflict": 8,
            "absurd_quote_or_defense": 8,
            "first_frame_visual_contradiction": 9,
            "audio_hook": 9,
            "freshness": 8,
            "duplicate_penalty": -8,
            "risk_penalty": -1,
        },
        "total": 43,
        "risk": "Low-moderate, but official run_011 already covered the same Coachella laptop lane.",
    },
    {
        "id": "reese_ai_invoice_notary",
        "premise": "An AI-literacy seminar becomes a notary counter where every inspirational post prints a data-center invoice.",
        "source_ids": ["variety_reese_ai_2026_04_22"],
        "scores": {
            "famous_face_or_group": 8,
            "public_conflict": 7,
            "absurd_quote_or_defense": 7,
            "first_frame_visual_contradiction": 8,
            "audio_hook": 6,
            "freshness": 9,
            "duplicate_penalty": -8,
            "risk_penalty": -1,
        },
        "total": 36,
        "risk": "Low, but official run_053 already owns the Reese AI invoice audit lane.",
    },
    {
        "id": "ice_spice_wendys_witness",
        "premise": "A fast-food witness stand tries to transfer a Hollywood booth incident from McDonald's to a Wendy's brand-safety counter.",
        "source_ids": ["variety_ice_spice_wendys_2026_04_18"],
        "scores": {
            "famous_face_or_group": 8,
            "public_conflict": 8,
            "absurd_quote_or_defense": 10,
            "first_frame_visual_contradiction": 9,
            "audio_hook": 8,
            "freshness": 9,
            "duplicate_penalty": -9,
            "risk_penalty": -2,
        },
        "total": 41,
        "risk": "Already covered by multiple SGFLIX lanes; avoid reuse.",
    },
]


WINNER = CANDIDATES[0]


FIRST_FRAME_PROMPT = """Use case: stylized-concept
Asset type: SGFLIX first frame, 16:9.
Primary request: Create a satirical, editorial-pop first frame for a fictional reunion-tour security turnstile. Three open lanes glow with clean signage reading PCD FOREVER, while three glittering old backstage ID badges bounce back at the scanner with big stamps reading PROTECT OUR PEACE and NOT ON LIST. The setting is half arena entrance, half backstage vanity desk: velvet ropes, chrome turnstiles, headset clipboard, stage lights, pink receipt printer, and a tiny calm PR podium.
Subject: fictional girl-group reunion bureaucracy, no exact likenesses, no real logos, no real faces.
Composition: wide symmetrical turnstile desk; open lanes on the left, rejected glitter badges on the right, stamp hand in foreground, concert lights behind.
Style: glossy tabloid-satire production design, crisp readable labels, high contrast black/chrome/hot pink/teal/gold palette, cinematic 28mm lens, sharp prop details, no clutter.
Guardrails: do not depict real Pussycat Dolls likenesses, do not use real tour branding, do not state a proven political motive, avoid hateful political symbols, no defamatory text, no watermark."""


BOARD_PROMPT = """Use case: infographic-diagram
Asset type: SGFLIX Shared Choices director's-bible board.
Primary request: Build a single polished storyboard/design board for the fictional PCD Peace Turnstile satire. Include labeled zones for character archetypes, hero props, color palette swatches, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, audio hook notes, and production notes.
Required visual elements: chrome arena turnstile, glitter backstage ID badges, PCD FOREVER open lanes, PROTECT OUR PEACE stamp, NOT ON LIST receipt printer, velvet rope, headset clipboard, vanity bulbs, stage haze, PR podium.
Style: clean director-bible board, editorial pop satire, readable block labels, mixed storyboard thumbnails and prop callouts, no exact public-figure likenesses, no real logos, no dense illegible microtext.
Guardrails: source-frame the premise as reported/claimed; do not imply proven discrimination; target PR choreography and reunion bureaucracy."""


ACE_PAYLOAD = {
    "run_id": f"run_{RUN_ID}_{SLUG}",
    "title": "Turnstile Peace Stamp",
    "mode": "text_to_music",
    "customMode": True,
    "lyrics": "[Intro]\nScanner beep, velvet rope, lights up\n\n[Verse]\nThree lanes open, three badges spin\nClipboard says who gets in\nGlitter on the scanner, receipt prints twice\nProtect our peace at the turnstile price\n\n[Hook]\nNot on list, stamp-stamp, keep it moving\nPeace gate clicks while the bassline's grooving\nNot on list, stamp-stamp, doors are shining\nReunion clock says perfect timing",
    "style": "Original short Y2K pop-club satire hook, 124 BPM, glossy dance-pop drums, rubbery bass, chrome synth stabs, chantable anonymous ensemble vocals, paparazzi camera risers, playful backstage arena energy. No artist imitation, no copyrighted melody, no Pussycat Dolls vocal clone.",
    "trackName": "RUN 086 - Turnstile Peace Stamp",
    "instrumental": False,
    "vocalLanguage": "en",
    "duration": 28,
    "bpm": 124,
    "keyScale": "A minor",
    "timeSignature": "4/4",
    "batchSize": 1,
    "randomSeed": False,
    "seed": 86086,
    "thinking": False,
    "audioFormat": "mp3",
    "inferMethod": "ode",
    "taskType": "text2music",
    "instruction": "Generate an original anonymous club-pop hook for a satirical reunion-turnstile scene. Emphasize scanner beeps, stamp hits, and a clear hook by 8 seconds. Do not imitate The Pussycat Dolls or any real singer.",
    "guidanceScale": 7.0,
    "inferenceSteps": 100,
    "lmTemperature": 0.6,
    "lmCfgScale": 1.6,
    "lmTopP": 0.8,
    "lmTopK": 0,
    "lmNegativePrompt": "Pussycat Dolls clone, Nicole Scherzinger voice, real song melody, Don't Cha, Buttons, political chant, hateful slogan, distorted mid-song voice drift, slow ballad, acoustic guitar, choir, mumble vocals, off-tempo drums",
    "shift": 3.0,
    "audioCoverStrength": 1.0,
    "useCotMetas": False,
    "useCotCaption": False,
    "useCotLanguage": False,
    "allowLmBatch": True,
    "getScores": False,
    "getLrc": False,
    "scoreScale": 0.5,
    "lmBatchChunkSize": 8,
    "completeTrackClasses": ["drums", "bass", "synth", "vocals", "fx"],
}


def draw_first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 864), "#101113")
    d = ImageDraw.Draw(img)
    d.rectangle((52, 48, 1484, 816), fill="#f7f0e8", outline="#0c0c0c", width=6)
    d.rectangle((88, 82, 1448, 172), fill="#0c0c0c")
    label(d, 118, 108, "PCD PEACE TURNSTILE", 54, "#f7f0e8", True)
    label(d, 1180, 126, "RUN 086", 28, "#ff3f8e", True)
    colors = ["#17c3b2", "#ff3f8e", "#f7c948"]
    for i, color in enumerate(colors):
        x = 120 + i * 300
        d.rectangle((x, 260, x + 228, 650), fill="#20242a", outline="#0c0c0c", width=4)
        d.rectangle((x + 18, 288, x + 210, 358), fill=color, outline="#0c0c0c", width=3)
        label(d, x + 38, 305, "PCD\nFOREVER", 24, "#0c0c0c", True)
        d.line((x + 52, 420, x + 178, 420), fill="#d9dee8", width=16)
        d.arc((x + 42, 444, x + 188, 620), 180, 360, fill="#d9dee8", width=14)
    d.rectangle((1018, 250, 1388, 682), fill="#fff7f0", outline="#0c0c0c", width=5)
    label(d, 1052, 284, "SCANNER REJECTS", 36, "#0c0c0c", True)
    for j, text in enumerate(["PROTECT\nOUR PEACE", "NOT\nON LIST", "CALLBACK\nPENDING"]):
        y = 376 + j * 86
        d.rounded_rectangle((1060, y, 1320, y + 62), radius=8, fill="#ff3f8e" if j == 0 else "#f7c948", outline="#0c0c0c", width=3)
        label(d, 1080, y + 8, text, 19, "#0c0c0c", True)
    d.rectangle((780, 318, 948, 642), fill="#17c3b2", outline="#0c0c0c", width=5)
    label(d, 806, 352, "GLITTER\nID\nBADGES", 31, "#0c0c0c", True)
    d.line((932, 500, 1018, 466), fill="#ff3f8e", width=10)
    label(d, 130, 720, "Satire target: reunion PR choreography and access-control absurdity. All motive claims remain reported/claimed, not proven.", 25, "#0c0c0c", True, 1220)
    img.save(path)


def draw_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1800, 1200), "#ece5dc")
    d = ImageDraw.Draw(img)
    label(d, 54, 34, "RUN 086 SHARED CHOICES: PCD PEACE TURNSTILE", 38, "#101113", True)
    label(d, 56, 82, "Director bible: arena turnstile, glitter ID rejects, PR peace stamp, anonymous club-pop hook.", 22, "#333333")
    for i, color in enumerate(["#101113", "#f7f0e8", "#ff3f8e", "#17c3b2", "#f7c948", "#8a4fff", "#d9dee8"]):
        d.rectangle((60 + i * 78, 140, 120 + i * 78, 200), fill=color, outline="#101113")
    cards = [
        ("Character", "Anonymous reunion-pop archetypes only: silhouettes, hands, badges, headset clerk, PR podium. No real faces."),
        ("Hero Props", "Chrome turnstile, glitter ID badges, PROTECT OUR PEACE stamp, NOT ON LIST receipt, velvet rope."),
        ("Environment", "Half arena entrance, half backstage vanity station; concert haze behind a calm access-control desk."),
        ("Floor Plan", "Open lanes left, scanner desk center, rejected badges right, PR podium rear, camera starts counter-height."),
        ("Visual Rules", "No real logos or exact likeness. Use reported/claimed language. Joke is bureaucracy, not identity."),
        ("Camera", "28mm wide contradiction, 70mm stamp insert, 50mm badge tumble, top-down receipt printer button."),
        ("Audio Hook", "124 BPM Y2K club-pop: scanner beep intro, stamp hits as snare, hook lands by 8 seconds."),
        ("Production", "Readable labels only: PCD FOREVER, PROTECT OUR PEACE, NOT ON LIST, CLAIMED/REPORTED."),
    ]
    for i, (title, body) in enumerate(cards):
        x = 60 + (i % 4) * 430
        y = 230 + (i // 4) * 210
        d.rectangle((x, y, x + 392, y + 174), fill="#fffaf4", outline="#101113", width=3)
        d.rectangle((x, y, x + 392, y + 42), fill="#101113")
        label(d, x + 14, y + 10, title, 21, "#f7f0e8", True)
        label(d, x + 16, y + 58, body, 20, "#333333", False, 350)
    panels = [
        "Wide: three open lanes glow while glitter badges queue at scanner.",
        "Insert: PROTECT OUR PEACE stamp lands exactly on first snare.",
        "Motion: rejected badge flips into a pink receipt printer tray.",
        "Button: anonymous group silhouettes enter; empty badges keep spinning.",
    ]
    for i, body in enumerate(panels):
        x = 60 + i * 430
        y = 720
        d.rectangle((x, y, x + 392, y + 320), fill="#20242a", outline="#f7c948", width=4)
        label(d, x + 18, y + 18, f"Panel {i + 1}", 25, "#ff3f8e", True)
        label(d, x + 18, y + 66, body, 23, "#f7f0e8", False, 335)
    label(d, 64, 1080, "Lighting: chrome glints, vanity bulbs, teal/pink split, no dark single-hue palette. QC: inspect generated text before any public use.", 23, "#101113", True, 1560)
    img.save(path)


def main() -> None:
    for rel in [
        "research",
        "strategy",
        "audio/generated_candidates",
        "qc",
        "chai",
        "scene_json",
        "handoffs",
        "captions",
        "distribution",
        "skool",
        "manifests",
        "frames/gpt_image_2",
        "storyboards/shared_choices",
    ]:
        (RUN_DIR / rel).mkdir(parents=True, exist_ok=True)
    MASTER.mkdir(parents=True, exist_ok=True)

    write_json("research/sources.json", SOURCES)
    write(
        "research/last30days_report.md",
        f"""# Last30days Research Intake - RUN {RUN_ID}

Created: {NOW}

Order-of-operations note: this run began with fresh web/source intake and duplicate checks before selecting a premise. No local image, old storyboard, old handoff, or existing audio file was used as the premise.

## Current Scan

- The current entertainment scan surfaced several April 2026 lanes: Justin Bieber's Coachella YouTube/laptop section, Reese Witherspoon's AI backlash response, Ice Spice's McDonald's/Wendy's quote, Chappell Roan fan-boundary followups, and the Pussycat Dolls reunion lineup dispute.
- Recent official SGFLIX runs already cover Reese AI invoice, Ice Spice/Wendy's, Chappell boundary, Bieber Coachella, and Wireless cancellation lanes, so those were penalized for duplication.
- The selected lane uses the Pussycat Dolls reunion story because it has a clear public phrase, a built-in visual contradiction, a famous pop-group container, and no recent official numbered duplicate.
- The strongest source-framed joke is the access-control contradiction: a reunion sells unity while a turnstile sorts who is allowed through.

## Verification Notes

- Variety and Us Weekly report Jessica Sutta's theory/claim that politics contributed to her exclusion. This package does not present that motive as proven.
- NME reports the public response phrase "protect our peace" and the awkward trio-reunion framing. This phrase drives the stamp/turnstile visual.
- Aftenposten's April 5 coverage keeps the story inside the last-30-days-style scan window as of May 2, 2026.

## Rejected Fresh Candidates

- Bieber laptop karaoke: visually strong but already covered by run_011.
- Reese AI invoice: timely but already covered by run_053.
- Ice Spice Wendy's witness: excellent quote but already covered by run_014/run_056-style lanes.
- Chappell autograph/security boundary: culturally live but already covered by run_019/run_070-style lanes.

## Research Query/Topic

Pussycat Dolls 2026 reunion lineup dispute, "protect our peace" response, and member claim/theory around MAGA/RFK politics.
""",
    )
    write_json("strategy/candidate_board.json", {"created_at": NOW, "selection_method": "fresh source scan -> duplicate penalty -> scored winner", "candidates": CANDIDATES, "winner_id": WINNER["id"]})
    write(
        "strategy/winner_decision.md",
        f"""# Winner Decision - RUN {RUN_ID}

Selected premise: {WINNER['premise']}

Why it wins:
- The first frame is instantly legible: a reunion-tour turnstile that lets some glitter badges through and stamps others as `PROTECT OUR PEACE`.
- It uses public entertainment reporting without needing to prove the disputed motive.
- It has an original audio hook lane: scanner beeps, stamp snares, and a Y2K club-pop chant.
- It avoids duplicating recent numbered lanes on Reese, Ice Spice, Chappell, Bieber, and Wireless.

Score summary: total `{WINNER['total']}` after duplicate/risk penalties.

Risk posture: keep all motive language as `claimed`, `reported`, or `theorized`; do not depict real faces, party symbols, hateful slogans, or exact tour marks.
""",
    )
    write_json("strategy/phase_minus_one_worthiness_audit.json", {"run_id": RUN_ID, "passes": True, "scroll_stop": 9, "native_absurdity": 9, "humiliation_without_cruelty": 7, "source_grounding": 8, "reason": "Unity-branded reunion collides with access-control exclusion; phrase is public and prop-ready."})
    write_json("strategy/source_entropy_audit.json", {"run_id": RUN_ID, "source_count": len(SOURCES), "independent_source_types": ["trade", "music press", "celebrity weekly", "international newspaper"], "local_asset_contamination": False, "duplicate_lane_penalty_applied": True, "unverified_claims": ["political motive for exclusion"], "handling": "Satire the PR/access-control contradiction, not the truth of the motive."})
    write_json("strategy/humor_logic_bridge.json", {"setup": "A reunion tour promises togetherness.", "pressure": "Several former members say they were not included or not informed.", "turn": "The reunion becomes an airport-style turnstile sorting glitter IDs.", "button": "The stamp does the public-relations work: PROTECT OUR PEACE.", "do_not_do": ["do not claim discrimination was proven", "do not mock political identity", "do not use exact likenesses"]})
    write_json("strategy/tribe_meta_score.json", {"tribes": [{"name": "Y2K pop nostalgia watchers", "score": 8}, {"name": "Pop-culture litigation/PR watchers", "score": 7}, {"name": "Girl-group fandom", "score": 7}], "meta_angle": "reunion nostalgia now operates like brand-access bureaucracy", "overall": 7.4})
    write_json("strategy/risk_taste_score.json", {"legal_risk": "medium-low if source-framed", "taste_risk": "medium around politics; avoid symbols and protected-trait framing", "likeness_risk": "medium; use anonymous archetypes", "brand_risk": "medium; no real logos", "go": True})
    write("strategy/franchise_decision.md", "# Franchise Decision\n\nDecision: `one-shot with franchise option`.\n\nThe access-control desk can recur for other reunion-tour disputes, but this package should stay focused on the PCD public-response phrase and not become a general political dunk lane.")

    write("audio/audio_concept.md", "# Audio Concept\n\nTitle: `Turnstile Peace Stamp`\n\nA 28-second original Y2K club-pop hook built from scanner beeps, stamp hits, chrome synth stabs, and anonymous ensemble chants. The audio should feel like a glossy arena entrance desk turning into a dance floor.\n\nDo not imitate The Pussycat Dolls, Nicole Scherzinger, or any copyrighted PCD melody. No source audio is being cloned; this is text-to-music only.")
    write_json("audio/ace_step_payload.json", ACE_PAYLOAD)
    write_json("audio/music_handoff.json", {"selected_drop_window": {"start": 6.0, "end": 14.0}, "visual_sync": [{"time": 0.0, "event": "scanner beep; turnstile lights wake up"}, {"time": 2.0, "event": "three open PCD FOREVER lanes glow"}, {"time": 6.0, "event": "PROTECT OUR PEACE stamp lands on first hook"}, {"time": 10.0, "event": "NOT ON LIST receipt prints in pink"}, {"time": 14.0, "event": "glitter badge flips into freeze-frame caption"}], "keeper_requirements": ["clear hook by 8 seconds", "anonymous vocals only", "no recognizable PCD melody or vocal", "stable vocal timbre through hook"]})
    write("qc/audio_qc.md", "# Audio QC\n\nStatus: `PENDING_REMOTE_ACE_STEP_GENERATION`\n\n3090 preflight must run before declaring audio blocked. Required remote run label: `run_086_pcd_peace_turnstile`. After generation, copy MP3, metrics, critique, and manifest into this package and update keeper decision.")

    write("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)
    draw_first_frame(RUN_DIR / "frames/gpt_image_2/first_frame_v01.png")
    draw_board(RUN_DIR / "storyboards/shared_choices/shared_choices_v01.png")

    write_json("chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot": "001", "duration_seconds": 6, "subject": "fictional reunion-tour turnstile with glitter ID badges", "scene": "arena entrance/backstage vanity hybrid with scanner desk and PR podium", "motion": "slow push from glowing open lanes to rejected badges to stamp hand", "spatial": "open lanes left, scanner center, rejected badge desk right, PR podium rear", "camera": "28mm counter-height dolly, crisp editorial-pop lighting", "critique": "joke must read as reunion bureaucracy, not a proven political motive", "revision": "remove exact likenesses, real logos, hateful symbols, dense text, or defamation"}]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "status": "handoff_only_no_video_generation", "source_image": "frames/gpt_image_2/first_frame_v01.png", "prompt": "Manual-only 6-second clip: scanner beep, open lanes glow, rejected glitter badge gets PROTECT OUR PEACE stamp, receipt prints NOT ON LIST. No video generated by this automation.", "negative": "real PCD likeness, real logos, proven motive claim, political hate symbol, watermark"}
    write_json("scene_json/shot_001.json", scene)
    write_json("scene_json/shot_0001.json", scene)
    write_json("handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "video_generation_tools_called": False, "manual_only": True, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "audio_payload": "audio/ace_step_payload.json", "guardrails": ["no exact likenesses", "no real logos", "source-frame motive as reported/claimed", "do not auto-post"]})
    write("handoffs/grok_agent_prompt.md", f"# Grok Agent Prompt\n\nUse the saved first-frame and Shared Choices prompts to create reference still repairs only. Do not generate video. Keep the premise source-framed: a fictional reunion-tour access-control desk inspired by reported PCD reunion lineup dispute coverage. Required props: PCD FOREVER open lanes, PROTECT OUR PEACE stamp, NOT ON LIST receipt, glitter ID badges. Avoid exact real-person likenesses and real logos.")
    write("captions/instagram_caption.md", "When the reunion tour has more access control than the airport.\n\nReported/claimed context only: the joke is the PR turnstile, not a proven motive.\n\n#SGFLIX #PopCultureSatire #Y2KNostalgia #ReunionTour #EntertainmentNews")
    write("distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels/TikTok still-to-video handoff after human approval.\n\nHook text: `REUNION TOUR SECURITY JUST STAMPED THE VIBES`.\n\nDo not post until audio keeper is reviewed and still-image text is inspected.")
    write("skool/case_study.md", "# Skool Case Study\n\nThis run shows how to turn a disputed entertainment-news motive into a safer prop-comedy system. The premise avoids adjudicating the claim and instead visualizes the public contradiction: reunion branding plus backstage access control.")
    write("qc/first_frame_v01_qc.md", "# First Frame QC\n\nStatus: `USABLE_INTERNAL_SCHEMATIC_NEEDS_HUMAN_IMAGE_REVIEW`\n\nThe saved PNG is a locally generated schematic paired with a GPT Image 2-style prompt. It includes the required contradiction, avoids exact likenesses/logos, and keeps the motive source-framed. Before public use, rerun or repair with GPT Image 2 and inspect all text.")
    write("qc/shared_choices_v01_qc.md", "# Shared Choices QC\n\nStatus: `USABLE_INTERNAL_DIRECTOR_BOARD_NEEDS_HUMAN_IMAGE_REVIEW`\n\nThe board includes character/props, palette, environment, blocking, storyboard panels, lighting, visual rules, audio notes, and production notes. It is suitable for internal direction; public-facing image repair should preserve the same guardrails.")
    write("qc/IMAGE_GENERATION_NOTE.md", "# Image Generation Note\n\nA GPT Image 2-style workflow prompt was created after research, scoring, and winner selection. This runtime does not expose the built-in image generation tool, so local schematic PNGs were generated instead of production GPT Image 2 outputs. No video-generation tool was called.")

    required = [
        "RUN_086_MASTER_PACKAGE/README.md",
        "RUN_086_MASTER_PACKAGE/RUN_086_MASTER_PACKAGE.json",
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
        "audio/audio_concept.md",
        "audio/ace_step_payload.json",
        "audio/music_handoff.json",
        "qc/audio_qc.md",
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
    assets = [
        {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", "mode": "local_schematic_after_gpt_image_2_prompt", "exists": True},
        {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", "mode": "local_schematic_after_gpt_image_2_prompt", "exists": True},
        {"path": "audio/ace_step_payload.json", "type": "audio_payload", "mode": "pending_remote_ace_step", "exists": True},
    ]
    write_json("manifests/asset_manifest.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "required_files": required, "assets": assets, "post_ready_exports": [], "missing_files_before_remote_audio": ["audio/generated_candidates/*.mp3", "audio/audio_scorecard.json", "audio/keeper_manifest.json", "qc/hook_timing_qc.md"], "video_generation_tools_called": False})
    write_master_json(f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "selected_premise": WINNER, "sources": SOURCES, "candidate_board": CANDIDATES, "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}, "audio": {"payload": "audio/ace_step_payload.json", "status": "pending_remote_generation"}, "video_generation_tools_called": False})
    write_master("README.md", f"# RUN {RUN_ID} MASTER PACKAGE - PCD Peace Turnstile\n\nStatus: package created; remote audio generation pending.\n\nSelected premise: {WINNER['premise']}\n\nGenerated still-image artifacts:\n- `frames/gpt_image_2/first_frame_v01.png`\n- `storyboards/shared_choices/shared_choices_v01.png`\n\nAudio payload:\n- `audio/ace_step_payload.json`\n")
    write("FACTORY_RUN_STATUS.md", f"""# FACTORY RUN STATUS - RUN {RUN_ID}

Status: `PACKAGE_CREATED_REMOTE_AUDIO_PENDING`

Research topic: Pussycat Dolls 2026 reunion lineup dispute and `protect our peace` response.

Selected premise: {WINNER['premise']}

Created package artifacts include research, source manifest, candidate board, scoring, winner decision, strategy audits, audio concept/payload/handoff, CHAI shot spec, scene JSONs, closed-tool handoffs, captions, distribution, Skool case study, manifest, first-frame schematic, Shared Choices board, and QC.

Still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Remote audio next step:
- Copy `audio/ace_step_payload.json` to `/mnt/bulk/home/straughter/sgflix_audio_factory/payloads/run_086_pcd_peace_turnstile_payload_001.json`.
- Run ACE-Step one-iteration factory with run label `run_086_pcd_peace_turnstile`.
- Copy generated MP3, metrics, critique, and manifest back into the package.

Video generation: not called and prohibited.
""")

    print(RUN_DIR)


if __name__ == "__main__":
    main()
