from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "087"
SLUG = "spirit_final_boarding"
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
    lines: list[str] = []
    current = ""
    for word in text.split():
        test = f"{current} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, fill: str, bold: bool = False, width: int | None = None) -> int:
    x, y = xy
    fnt = font(size, bold)
    lines = wrap(draw, text, fnt, width) if width else text.splitlines()
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(size * 1.22)
    return y


SOURCES = [
    {
        "id": "spirit_prnewswire_2026_05_02",
        "title": "Spirit Airlines Begins Orderly Wind-Down of Operations",
        "publisher": "PRNewswire / Spirit Aviation Holdings",
        "url": "https://www.prnewswire.com/news-releases/spirit-airlines-begins-orderly-wind-down-of-operations-302760586.html",
        "published": "2026-05-02",
        "used_for": "Primary source for effective-immediately shutdown, canceled flights, no airport guidance.",
        "fact_status": "company statement",
    },
    {
        "id": "ap_spirit_shutdown_2026_05_02",
        "title": "Spirit Airlines goes out of business after 34 years, ending operations immediately",
        "publisher": "Associated Press",
        "url": "https://apnews.com/article/37a4818e1b71c0905d022f669d85948c",
        "published": "2026-05-02",
        "used_for": "Independent confirmation and scale/context for the shutdown.",
        "fact_status": "reported by AP",
    },
    {
        "id": "axios_spirit_shutdown_2026_05_02",
        "title": "Spirit Airlines shutting down, canceling all flights",
        "publisher": "Axios",
        "url": "https://www.axios.com/2026/05/02/spirit-airlines-shutdown",
        "published": "2026-05-02",
        "used_for": "Public impact framing and rescue-fare context.",
        "fact_status": "reported by Axios",
    },
    {
        "id": "tmz_hart_rock_2026_05_02",
        "title": "Kevin Hart Calls Dwayne 'The Rock' Johnson A Piece of S*** After Traffic Stop",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/05/02/kevin-hart-jokingly-blasts-dwayne-johnson/",
        "published": "2026-05-02",
        "used_for": "Rejected candidate: funny but duplicate of existing tint-ticket lane.",
        "fact_status": "entertainment report; gag framing",
    },
    {
        "id": "tmz_post_malone_2026_05_02",
        "title": "Post Malone Cancels First Few Weeks of Upcoming Tour",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/05/02/post-malone-cancels-beginning-of-tour/",
        "published": "2026-05-02",
        "used_for": "Rejected candidate: music hook is strong, but visual contradiction is softer.",
        "fact_status": "reported social announcement",
    },
    {
        "id": "iheart_north_ep_2026_05_01",
        "title": "Ye & Kim Kardashian's Daughter North West Releases Her First EP 'N0rth4evr'",
        "publisher": "iHeart",
        "url": "https://www.iheart.com/content/2026-05-01-ye-kim-kardashians-daughter-north-west-releases-her-first-ep-n0rth4evr/",
        "published": "2026-05-01",
        "used_for": "Rejected candidate: famous-family music event, avoided because the performer is a minor.",
        "fact_status": "music/entertainment report",
    },
]


CANDIDATES = [
    {
        "id": "spirit_final_boarding",
        "premise": "A bright yellow budget-airline gate becomes a foreclosure auction where every boarding pass prints FINAL BOARDING: NO AIRCRAFT ATTACHED.",
        "source_ids": ["spirit_prnewswire_2026_05_02", "ap_spirit_shutdown_2026_05_02", "axios_spirit_shutdown_2026_05_02"],
        "scores": {
            "famous_brand_or_face": 9,
            "public_conflict": 9,
            "absurd_quote_or_defense": 8,
            "first_frame_visual_contradiction": 10,
            "audio_hook": 9,
            "freshness": 10,
            "duplicate_penalty": -2,
            "risk_penalty": -1,
        },
        "total": 52,
        "risk": "Low-moderate. Public company shutdown affects workers/travelers; satire should target corporate budget-airline absurdity, not stranded passengers.",
    },
    {
        "id": "hart_rock_tint_snitch",
        "premise": "Kevin Hart appears as a tiny confidential informant at a giant tint-meter traffic court for The Rock's SUV.",
        "source_ids": ["tmz_hart_rock_2026_05_02"],
        "scores": {
            "famous_brand_or_face": 10,
            "public_conflict": 7,
            "absurd_quote_or_defense": 10,
            "first_frame_visual_contradiction": 9,
            "audio_hook": 7,
            "freshness": 10,
            "duplicate_penalty": -9,
            "risk_penalty": -1,
        },
        "total": 43,
        "risk": "Low, but too close to run_071_rock_tint_ticket_dispatch.",
    },
    {
        "id": "postpone_stadium_studio",
        "premise": "A stadium tour gate is converted into a recording booth with Jelly Roll seat maps waiting outside.",
        "source_ids": ["tmz_post_malone_2026_05_02"],
        "scores": {
            "famous_brand_or_face": 9,
            "public_conflict": 7,
            "absurd_quote_or_defense": 8,
            "first_frame_visual_contradiction": 8,
            "audio_hook": 9,
            "freshness": 10,
            "duplicate_penalty": -4,
            "risk_penalty": -1,
        },
        "total": 46,
        "risk": "Low, but less culturally sharp than the Spirit shutdown and close to run_077_postpone_studio_tollbooth.",
    },
    {
        "id": "north_popup_parental_merch",
        "premise": "A celebrity-family pop-up shop sells tiny executive-producer lanyards beside rage-rap merch.",
        "source_ids": ["iheart_north_ep_2026_05_01"],
        "scores": {
            "famous_brand_or_face": 9,
            "public_conflict": 5,
            "absurd_quote_or_defense": 6,
            "first_frame_visual_contradiction": 8,
            "audio_hook": 8,
            "freshness": 9,
            "duplicate_penalty": 0,
            "risk_penalty": -8,
        },
        "total": 37,
        "risk": "Avoid: central performer is a minor.",
    },
]

WINNER = CANDIDATES[0]

FIRST_FRAME_PROMPT = """Use case: stylized-concept
Asset type: SGFLIX first frame, 16:9.
Primary request: Create a satirical editorial-pop first frame for a fictional budget-airline gate after an immediate shutdown. A bright yellow check-in counter has become a foreclosure auction desk. A boarding pass printer spits receipts reading FINAL BOARDING and NO AIRCRAFT ATTACHED. A glowing departures board says ALL FLIGHTS CANCELLED. Empty yellow seat rows, velvet ropes, rescue-fare signs from rival airlines, a lonely luggage scale, and a paper airplane made from a bankruptcy filing complete the scene.
Subject: fictional budget-airline collapse scene; no real airline logo, no real passengers in distress, no exact brand marks.
Composition: wide airport gate tableau, auction gavel foreground, boarding pass stream center, empty jet bridge in background.
Style: glossy tabloid-satire production design, yellow/black/white with emergency blue accents, crisp readable prop text, cinematic 28mm lens, sharp details, not cruel.
Guardrails: target corporate absurdity and cheap-flight bureaucracy; do not mock stranded travelers or workers; no real logo reproduction, no misinformation, no watermark."""

BOARD_PROMPT = """Use case: infographic-diagram
Asset type: SGFLIX Shared Choices director's-bible board.
Primary request: Build a single polished storyboard/design board for the fictional Spirit Final Boarding satire. Include labeled zones for character archetypes, hero props, color palette, environment/set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, audio hook notes, and production notes.
Required visual elements: yellow check-in counter, boarding pass printer, FINAL BOARDING receipt, NO AIRCRAFT ATTACHED tag, ALL FLIGHTS CANCELLED departures board, auction gavel, empty jet bridge, rescue-fare sign, paper airplane made from bankruptcy filing.
Style: clean director-bible board, readable labels, mixed storyboard thumbnails and prop callouts, airport satire with humane tone, no real airline logo.
Guardrails: source-frame the shutdown as reported on May 2, 2026; do not blame front-line workers; no new video generation."""

ACE_PAYLOAD = {
    "run_id": f"run_{RUN_ID}_{SLUG}",
    "title": "Final Boarding No Plane",
    "mode": "text_to_music",
    "customMode": True,
    "lyrics": "[Intro]\nGate agent voice: final boarding, no aircraft attached\n\n[Verse]\nYellow bag on the scale, printer coughs twice\nThirty-dollar seat map, no wings on ice\nRescue fare flashing at the next-door lane\nPaper plane made from a bankruptcy claim\n\n[Hook]\nFinal boarding, no plane at the gate\nCheap seat dreams got a brand-new fate\nStamp that pass, let the receipt sing\nAll flights cancelled, but the bass still rings",
    "style": "Original short satirical airport-club hook, 122 BPM, bouncy electro-pop drums, sub bass, scanner beeps, airport PA chops, chantable anonymous vocals, bright yellow budget-airline comedy energy. No artist imitation, no copyrighted melody, no real airline jingle.",
    "trackName": "RUN 087 - Final Boarding No Plane",
    "instrumental": False,
    "vocalLanguage": "en",
    "duration": 30,
    "negative_tags": "real Spirit Airlines jingle, copyrighted melody, cruel stranded-passenger mockery, airline logo, celebrity voice clone",
}


def create_first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (1920, 1080), "#101014")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1920, 1080], fill="#101014")
    d.rectangle([0, 740, 1920, 1080], fill="#242329")
    d.rectangle([110, 270, 1810, 390], fill="#f7df1e")
    label(d, (150, 294), "ALL FLIGHTS CANCELLED", 62, "#111111", True)
    d.rectangle([250, 470, 1270, 780], fill="#f5d51b", outline="#ffffff", width=6)
    label(d, (310, 510), "FINAL BOARDING", 76, "#111111", True)
    label(d, (312, 612), "NO AIRCRAFT ATTACHED", 58, "#111111", True)
    d.rectangle([1320, 500, 1640, 725], fill="#ffffff", outline="#f7df1e", width=8)
    label(d, (1360, 540), "RESCUE\nFARES", 54, "#182b45", True)
    d.line([560, 845, 1460, 845], fill="#f7df1e", width=10)
    d.polygon([(1450, 220), (1740, 290), (1510, 360)], fill="#ffffff", outline="#f7df1e")
    label(d, (1330, 895), "auction gavel foreground / empty jet bridge behind", 34, "#f2f2f2")
    label(d, (112, 910), "SGFLIX RUN 087 FIRST FRAME PROXY", 28, "#9aa3ad")
    im.save(path)


def create_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (2200, 1400), "#f5f2e9")
    d = ImageDraw.Draw(im)
    label(d, (70, 50), "RUN 087 SHARED CHOICES - SPIRIT FINAL BOARDING", 58, "#111111", True)
    boxes = [
        (70, 150, 560, 510, "Character Archetypes", "tired gate supervisor silhouette; corporate auctioneer; helpful rival-airline rescue desk; no distressed passenger punchline"),
        (600, 150, 1090, 510, "Hero Props", "FINAL BOARDING receipt, NO AIRCRAFT ATTACHED tag, auction gavel, paper-plane filing, empty yellow seats"),
        (1130, 150, 1620, 510, "Palette", "warning yellow, printer white, matte black, airport chrome, rescue blue"),
        (1660, 150, 2130, 510, "Set Design", "airport gate becomes foreclosure desk; empty jet bridge centered; departures board overhead"),
        (70, 560, 560, 960, "Blocking", "gavel foreground -> receipt printer center -> empty jet bridge vanishing point -> rescue desk side lane"),
        (600, 560, 1090, 960, "Panel 1", "28mm wide: ALL FLIGHTS CANCELLED board snaps on as printer starts"),
        (1130, 560, 1620, 960, "Panel 2", "50mm prop insert: boarding pass says NO AIRCRAFT ATTACHED"),
        (1660, 560, 2130, 960, "Panel 3", "35mm push: rival rescue fare sign lights up beside dead gate"),
        (70, 1010, 560, 1320, "Panel 4", "slow dolly to paper airplane filing landing on empty seat"),
        (600, 1010, 1090, 1320, "Audio Hook", "scanner beep + PA chop + chant: final boarding, no plane at the gate"),
        (1130, 1010, 1620, 1320, "Visual Rules", "no real logo, no worker blame, comedy points at corporate fee machine"),
        (1660, 1010, 2130, 1320, "Production Notes", "still-image proxy created locally; use GPT Image 2 prompt for final polish"),
    ]
    for x1, y1, x2, y2, title, body in boxes:
        d.rounded_rectangle([x1, y1, x2, y2], radius=16, fill="#ffffff", outline="#111111", width=3)
        label(d, (x1 + 24, y1 + 24), title, 34, "#111111", True)
        label(d, (x1 + 24, y1 + 82), body, 28, "#222222", width=x2 - x1 - 48)
    im.save(path)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    write("research/last30days_report.md", f"""# Step 1: Research Intake - Run {RUN_ID}

Run timestamp: {NOW}

Fresh scan focus: current entertainment/culture hooks with a strong SGFLIX first-frame contradiction. Sources reviewed included primary company statement and current news/entertainment coverage from May 1-2, 2026.

Winner selected after scoring: **Spirit Final Boarding**. The company statement and multiple news reports say Spirit began an orderly wind-down on May 2, 2026, effective immediately, with flights canceled. The premise is not based on local assets, prior boards, old handoffs, or nearby media.

Rejected/held candidates:
- Kevin Hart/The Rock tint joke: very funny, but too close to existing run_071.
- Post Malone postponing tour: audio-relevant but visually softer and too close to run_077.
- North West EP/pop-up: avoided because the central performer is a minor.

Taste note: keep the joke pointed at corporate fee/boarding bureaucracy and the surreal brand collapse. Do not mock stranded travelers or front-line workers.
""")
    write_json("research/sources.json", SOURCES)
    write_json("strategy/candidate_board.json", {"created_at": NOW, "selection_order": "research_intake_then_scoring_then_package_creation", "candidates": CANDIDATES, "winner_id": WINNER["id"]})
    write("strategy/winner_decision.md", f"""# Winner Decision

Selected premise: **{WINNER['premise']}**

Score: **{WINNER['total']}**. It wins on freshness, first-frame clarity, public brand recognition, and audio-hook potential. The prior Spirit bailout run is noted as related context, but this is a new May 2 shutdown chapter with different stakes and visual logic.

Source posture: primary source plus AP/Axios corroboration. No invented facts. Use "reported" language for non-primary context.
""")
    write_json("strategy/phase_minus_one_worthiness_audit.json", {"run_id": RUN_ID, "premise": WINNER["premise"], "passes": True, "why_now": "May 2, 2026 immediate airline wind-down/cancelled flights", "first_frame": "airport gate as foreclosure auction", "humor_target": "corporate cheap-flight bureaucracy", "avoid": ["front-line worker blame", "stranded-passenger cruelty", "real logo reproduction"]})
    write_json("strategy/source_entropy_audit.json", {"source_count": 3, "primary_source_count": 1, "entropy_score": 8, "notes": "Company statement plus AP/Axios independent reports; rejected candidates from TMZ/iHeart only used for board comparison."})
    write_json("strategy/humor_logic_bridge.json", {"setup": "Budget airline abruptly cancels all flights and winds down.", "contradiction": "The gate still performs boarding rituals even though no aircraft exists.", "visual_bridge": "boarding pass printer becomes receipt/foreclosure machine", "audio_bridge": "airport PA call becomes club hook", "punchline": "Final boarding, no plane at the gate."})
    write_json("strategy/tribe_meta_score.json", {"tribe": "air travel internet, budget-flight survivors, business-collapse watchers, meme news accounts", "shareability": 9, "comment_prompts": ["fee jokes", "rescue fare jokes", "yellow plane nostalgia"], "meta_score": 8.7})
    write_json("strategy/risk_taste_score.json", {"risk": "medium-low", "taste_score": 8, "risks": ["real workers impacted", "travel disruption"], "mitigations": ["human-safe caption", "target corporate machinery", "no passenger distress as gag"]})
    write("strategy/franchise_decision.md", "# Franchise Decision\n\nGreenlight as a one-off sequel to the earlier Spirit bailout bag-counter lane. Do not extend into a long franchise unless there is a clear rescue-fare or bankruptcy-court update.")

    write("audio/audio_concept.md", """# Audio Concept

Title: Final Boarding No Plane

Audio lane: original short satirical airport-club hook. The track should start with a dry PA line, scanner beep, then bouncy electro-pop drums. The hook must be chantable and timed to a printer/receipt visual.

Key moment for visuals: "Final boarding, no plane at the gate" lands on the first receipt reveal.
""")
    write_json("audio/ace_step_payload.json", ACE_PAYLOAD)
    write_json("audio/music_handoff.json", {"run_id": RUN_ID, "hook_line": "Final boarding, no plane at the gate", "bpm": 122, "structure": [{"time": "0.0-2.0", "event": "PA voice + scanner beep"}, {"time": "2.0-10.0", "event": "verse, printer coughs twice"}, {"time": "10.0-22.0", "event": "hook, receipt/board reveals"}, {"time": "22.0-30.0", "event": "bass rings out over empty jet bridge"}], "visual_anchor": "boarding pass printer and ALL FLIGHTS CANCELLED board"})
    write("qc/audio_qc.md", "# Audio QC\n\nPending remote ACE-Step generation. Required checks after return: file presence, audible hook, no real airline jingle, no celebrity/brand voice clone, humane tone, usable 0-30s timing.")

    write_json("chai/chai_shot_specs.json", {"shot": "first_frame_to_6s", "subject": "fictional yellow budget-airline gate turned foreclosure auction desk", "scene": "airport gate with cancelled board, receipt printer, empty jet bridge", "motion": "printer spits boarding pass, gavel taps, rescue sign flickers", "camera": "28mm wide push-in to receipt insert", "guardrails": ["no real logo", "no mocked stranded travelers", "no video generation in this factory run"]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "duration_s": 6, "prompt": FIRST_FRAME_PROMPT, "audio_sync": "receipt reveal hits on hook line", "negative": ["real logos", "passenger distress", "worker blame"]}
    write_json("scene_json/shot_0001.json", scene)
    write_json("scene_json/shot_001.json", scene)
    write_json("handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "premise": WINNER["premise"], "first_frame_prompt": FIRST_FRAME_PROMPT, "storyboard_prompt": BOARD_PROMPT, "no_video_generation": True, "audio_payload": ACE_PAYLOAD})
    write("handoffs/grok_agent_prompt.md", f"""# Grok/Closed Tool Prompt

Use the prompts in this package to create final still/reference assets only. Do not render video.

First frame prompt:
{FIRST_FRAME_PROMPT}

Shared Choices board prompt:
{BOARD_PROMPT}
""")
    write("captions/instagram_caption.md", """Final boarding, no plane at the gate.

The fee printer kept printing after the aircraft left the chat.

#sgflix #airporttok #budgetairline #travelnews #satire #finalboarding""")
    write("distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels/TikTok still-led teaser once video is generated elsewhere by a human. Use humane caption; avoid making impacted travelers the punchline.")
    write("skool/case_study.md", "# Skool Case Study\n\nLesson: a corporate collapse hook works when the first frame converts an abstract business event into a physical contradiction: a boarding ritual with no aircraft attached.")
    write_json("manifests/asset_manifest.json", {"run_id": RUN_ID, "created_at": NOW, "expected_assets": ["first_frame_v01.png", "shared_choices_v01.png", "audio candidate or blocked report"], "sources": [s["id"] for s in SOURCES]})

    write("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)
    create_first_frame(RUN_DIR / "frames/gpt_image_2/first_frame_v01.png")
    create_board(RUN_DIR / "storyboards/shared_choices/shared_choices_v01.png")
    write("qc/first_frame_v01_qc.md", "# First Frame QC\n\nUsable as local proxy/reference. Text is readable, no real logo, no passenger cruelty. Final GPT Image 2 polish is recommended when the image tool is available.")
    write("qc/shared_choices_v01_qc.md", "# Shared Choices QC\n\nUsable director-bible proxy. Contains character/prop/palette/set/blocking/panel/audio/production zones. Final GPT Image 2 board polish recommended.")

    write("FACTORY_RUN_STATUS.md", f"""# Factory Run Status

Run: {RUN_ID} - {SLUG}
Status: AUDIO_PENDING_REMOTE_RETURN
Created: {NOW}

Research intake completed first. Candidate board scored before package creation. Still-image proxy artifacts and prompts created. Remote audio generation still needs to be copied back and scored.
""")
    write_master("README.md", f"# RUN {RUN_ID} MASTER PACKAGE - Spirit Final Boarding\n\nPremise: {WINNER['premise']}\n\nStatus: audio pending remote generation return; still prompts and proxy images present.")
    write_master_json(f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "premise": WINNER["premise"], "status": "AUDIO_PENDING_REMOTE_RETURN", "required_next": "Run remote ACE-Step factory and copy generated artifacts back."})


if __name__ == "__main__":
    main()
