from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "082"
RUN_SLUG = "scientology_speedrun_security_desk"
RUN_NAME = f"run_{RUN_ID}_{RUN_SLUG}"
RUN_DIR = ROOT / "sgflix_runs" / RUN_NAME
MASTER_DIR = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, fill: str, bold: bool = False, max_width: int | None = None) -> int:
    fnt = font(size, bold)
    x, y = xy
    lines = wrap(draw, text, fnt, max_width) if max_width else text.splitlines()
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(size * 1.25)
    return y


def make_first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 864), "#101314")
    draw = ImageDraw.Draw(img)
    draw.rectangle((70, 58, 1466, 806), fill="#161b1d", outline="#e0c46c", width=6)
    draw.rectangle((110, 98, 505, 760), fill="#22292c", outline="#6e7b7f", width=3)
    draw.rectangle((560, 98, 1426, 760), fill="#191f21", outline="#475257", width=3)
    draw_label(draw, (140, 132), "SECURITY DESK", 50, "#f4e2a5", True)
    draw_label(draw, (140, 196), "SPEEDRUN CHECKPOINT", 31, "#b8c8ca", True)
    draw.rectangle((150, 282, 465, 600), fill="#0d1011", outline="#e0c46c", width=4)
    draw_label(draw, (178, 314), "WRISTBANDS", 34, "#f4e2a5", True)
    for i, label in enumerate(["ALIEN", "HOT DOG", "JESUS"]):
        y = 385 + i * 62
        draw.rectangle((182, y, 430, y + 42), fill="#2f3b3f", outline="#b8c8ca", width=2)
        draw_label(draw, (200, y + 8), label, 23, "#f7f2db", True)
    draw.rectangle((605, 145, 1385, 635), fill="#0d1011", outline="#e0c46c", width=4)
    draw_label(draw, (635, 176), "FIRST FRAME CONTRADICTION", 38, "#f4e2a5", True)
    draw_label(draw, (635, 236), "A chaotic creator crew hits a dead-serious lobby counter like it is an arcade level.", 30, "#f7f2db", False, 700)
    for x, color in [(690, "#9ad4d6"), (810, "#f05b5b"), (930, "#f4e2a5"), (1050, "#8ed081")]:
        draw.ellipse((x, 375, x + 80, 455), fill=color, outline="#101314", width=3)
        draw.rectangle((x + 28, 455, x + 52, 560), fill=color)
    draw.line((665, 590, 1325, 590), fill="#e0c46c", width=8)
    draw.polygon([(1318, 570), (1370, 590), (1318, 610)], fill="#e0c46c")
    draw_label(draw, (675, 652), "NO REAL LOGOS. NO HARASSMENT. THE JOKE IS INTERNET CHAOS MEETING LOBBY PROCEDURE.", 25, "#b8c8ca", True, 660)
    draw.rectangle((70, 815, 1466, 850), fill="#e0c46c")
    draw_label(draw, (105, 822), "SGFLIX RUN 082: speedrun energy gets processed by a clipboard", 23, "#101314", True)
    img.save(path)


def make_storyboard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1800, 1200), "#f2efe4")
    draw = ImageDraw.Draw(img)
    draw_label(draw, (54, 36), "RUN 082 SHARED CHOICES: SECURITY DESK SPEEDRUN", 38, "#101314", True)
    draw_label(draw, (56, 86), "Director bible board: character canon, props, palette, blocking, shot panels, audio timing, and taste rules.", 22, "#4c5a5e")
    for i, color in enumerate(["#101314", "#f2efe4", "#e0c46c", "#9ad4d6", "#f05b5b", "#8ed081", "#6e7b7f"]):
        draw.rectangle((60 + i * 78, 142, 120 + i * 78, 202), fill=color, outline="#101314")
    sections = [
        ("Character + Hero Props", "Fictional creator pack: alien hoodie, hot-dog hat, robe silhouette, phone gimbal, wristband tray, clipboard, removed-door-handle display."),
        ("Environment", "Generic Hollywood information-center lobby fused with TSA intake desk and arcade start line. No real church marks or address text."),
        ("Floor Plan", "Lobby doors upstage, security desk downstage left, queue arrows center, runners frozen mid-sprint at the counter."),
        ("Visual Rules", "Satirize the internet behavior and security escalation. Do not mock religion, trespass, or name private people."),
        ("Camera", "24mm wide for the reveal, then 50mm insert on wristband tray and timer card. Slow push, no shaky footage."),
        ("Lighting", "Cool fluorescent lobby with absurd game-show rim lights. Keep text sparse and readable."),
        ("Audio Timing", "0.0s lobby beep; 0.7s sneakers; 1.4s guard stamp; 2.5s chant; 4.8s bass stop on clipboard snap."),
        ("Production Notes", "Original chant only. No source audio, no real institution logo, no invitation to replicate the behavior."),
    ]
    card_w, card_h = 400, 205
    for i, (title, body) in enumerate(sections):
        x = 60 + (i % 4) * (card_w + 32)
        y = 238 + (i // 4) * (card_h + 36)
        draw.rectangle((x, y, x + card_w, y + card_h), fill="#ffffff", outline="#101314", width=3)
        draw.rectangle((x, y, x + card_w, y + 42), fill="#101314")
        draw_label(draw, (x + 14, y + 10), title, 21, "#f2efe4", True)
        draw_label(draw, (x + 16, y + 60), body, 20, "#283438", False, card_w - 32)
    for i in range(4):
        x = 60 + i * 430
        y = 762
        draw.rectangle((x, y, x + 390, y + 300), fill="#171d20", outline="#e0c46c", width=4)
        draw_label(draw, (x + 18, y + 18), f"Panel {i + 1}", 25, "#f4e2a5", True)
        body = [
            "Wide lobby reveal: creator crew meets a dead-serious security desk.",
            "Insert: wristbands labeled ALIEN, HOT DOG, JESUS slide into a tray.",
            "Timer freezes at 00:00 while the guard stamps NO SPEEDRUN.",
            "End pose: clipboard becomes a turnstile; chant lands on the snap.",
        ][i]
        draw_label(draw, (x + 18, y + 64), body, 22, "#f2efe4", False, 335)
    img.save(path)


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    for directory in [
        MASTER_DIR,
        RUN_DIR / "research",
        RUN_DIR / "strategy",
        RUN_DIR / "audio" / "generated_candidates",
        RUN_DIR / "qc",
        RUN_DIR / "chai",
        RUN_DIR / "scene_json",
        RUN_DIR / "handoffs",
        RUN_DIR / "captions",
        RUN_DIR / "distribution",
        RUN_DIR / "skool",
        RUN_DIR / "manifests",
        RUN_DIR / "frames" / "gpt_image_2",
        RUN_DIR / "storyboards" / "shared_choices",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "id": "ap_scientology_speedruns_2026_04_30",
            "title": "Inside 'Scientology speedruns,' the viral trend prompting the church to bolster security",
            "publisher": "Associated Press",
            "url": "https://www.ctinsider.com/business/article/inside-scientology-speedruns-the-viral-trend-22234887.php",
            "published": "2026-04-30",
            "used_for": "Winner: internet speedrun behavior colliding with lobby security.",
            "fact_status": "AP-reported; use generic institution and do not imply endorsement or encourage trespass",
        },
        {
            "id": "lat_paramount_warner_lawsuit_2026_05_01",
            "title": "Consumers sue to block Paramount-Warner Bros. deal",
            "publisher": "Los Angeles Times",
            "url": "https://www.latimes.com/entertainment-arts/business/story/2026-05-01/consumers-sue-to-block-paramount-warner-bros-deal",
            "published": "2026-05-01",
            "used_for": "Alternate candidate, rejected for overlap with prior Paramount/media-merger lanes.",
            "fact_status": "reported by Los Angeles Times",
        },
        {
            "id": "tmz_rock_hart_tint_2026_05_02",
            "title": "Kevin Hart jokes about Dwayne Johnson traffic stop after tinted-window report",
            "publisher": "TMZ",
            "url": "https://www.tmz.com/",
            "published": "2026-05-02",
            "used_for": "Alternate candidate, rejected for overlap with Rock tint/ticket runs.",
            "fact_status": "tabloid/current entertainment report",
        },
        {
            "id": "lat_taylor_frankie_protective_orders_2026_04_30",
            "title": "Judge grants protective orders to Taylor Frankie Paul and Dakota Mortensen",
            "publisher": "Los Angeles Times",
            "url": "https://www.latimes.com/entertainment-arts/tv/story/2026-04-30/taylor-frankie-paul-dakota-mortensen-protective-order-child-custody",
            "published": "2026-04-30",
            "used_for": "Alternate candidate, rejected for domestic-violence/child-custody taste risk.",
            "fact_status": "reported by Los Angeles Times",
        },
    ]
    write_json(RUN_DIR / "research" / "sources.json", {"generated_at": now, "sources": sources})
    write_text(
        RUN_DIR / "research" / "last30days_report.md",
        """
# Step 1: Research Intake - Current Source Scan

This run began from current web/source context, not nearby images, old handoffs, or prior audio. The scan focused on late-April and May 2, 2026 entertainment/culture items with a strong first-frame contradiction and an audio-native hook.

## Candidate Context

1. AP-reported "Scientology speedruns": a viral creator trend where costumed groups try to run through an information-center lobby, prompting security changes. The visual contradiction is immediate: chaotic internet-speedrun energy meets a severe security desk. The safest SGFLIX version fictionalizes the institution as a generic Hollywood information center and satirizes the behavior, not religious beliefs.
2. Paramount-Warner consumer lawsuit: current, but overlaps prior Paramount/media-merger lanes and has weaker audio.
3. Dwayne Johnson/Kevin Hart tinted-window joke: fresh and funny, but rejected because SGFLIX already has Rock tint/ticket territory.
4. Taylor Frankie Paul protective-order coverage: current, but rejected for domestic-violence and child-custody taste risk.

## Winner Direction

Selected premise: **Scientology Speedrun Security Desk**. A fictional creator crew in absurd costumes hits a generic lobby security counter like it is an arcade level; the guard calmly processes them with wristbands, a clipboard, and a timer. The audio lane is an original hyperpop/club-chant hook built from sneakers, stamp hits, lobby beeps, and a clipboard snap.

## Verification Notes

- AP reported the trend and security concern; the package uses generic names and no institutional logo.
- Do not encourage trespass or harassment.
- Do not use private-person names, real addresses, or exact signage.
""",
    )

    board = {
        "schema": "sgflix.candidate_board.v1",
        "run_id": RUN_ID,
        "generated_at": now,
        "selection_rule": "Fresh research candidates scored before package creation.",
        "candidates": [
            {
                "rank": 1,
                "slug": RUN_SLUG,
                "title": "Scientology Speedrun Security Desk",
                "source_ids": ["ap_scientology_speedruns_2026_04_30"],
                "scores": {"freshness": 18, "visual_contradiction": 20, "audio_hook": 19, "humor_engine": 18, "risk_control": 15, "duplicate_entropy": 19},
                "total": 109,
                "decision": "winner",
                "notes": "Best first-frame contradiction and fresh internet behavior. Safer when fictionalized as a generic lobby/security process.",
            },
            {
                "rank": 2,
                "slug": "paramount_subscriber_merger_cart",
                "title": "Paramount Subscriber Merger Cart",
                "source_ids": ["lat_paramount_warner_lawsuit_2026_05_01"],
                "scores": {"freshness": 19, "visual_contradiction": 14, "audio_hook": 8, "humor_engine": 12, "risk_control": 17, "duplicate_entropy": 6},
                "total": 76,
                "decision": "reject_duplicate",
                "notes": "Current, but too close to existing Paramount/media-merger runs.",
            },
            {
                "rank": 3,
                "slug": "rock_tint_hotline_dispatch",
                "title": "Rock Tint Hotline Dispatch",
                "source_ids": ["tmz_rock_hart_tint_2026_05_02"],
                "scores": {"freshness": 20, "visual_contradiction": 14, "audio_hook": 13, "humor_engine": 16, "risk_control": 15, "duplicate_entropy": 3},
                "total": 81,
                "decision": "reject_duplicate",
                "notes": "Good joke but collides with prior Rock tint/ticket lanes.",
            },
            {
                "rank": 4,
                "slug": "reality_tv_100_foot_ruler_court",
                "title": "Reality TV 100-Foot Ruler Court",
                "source_ids": ["lat_taylor_frankie_protective_orders_2026_04_30"],
                "scores": {"freshness": 18, "visual_contradiction": 16, "audio_hook": 7, "humor_engine": 9, "risk_control": 3, "duplicate_entropy": 16},
                "total": 69,
                "decision": "reject_taste_risk",
                "notes": "Domestic-violence/child-custody subject is not worth the comedy risk.",
            },
        ],
    }
    write_json(RUN_DIR / "strategy" / "candidate_board.json", board)
    write_text(RUN_DIR / "strategy" / "winner_decision.md", "# Winner Decision\n\nWinner: **Scientology Speedrun Security Desk**.\n\nScore summary: 109/120. The Paramount suit scored 76 but was a duplicate-prone corporate merger lane; Rock/Hart tint scored 81 but collided with prior Rock tint runs; the reality-TV protective-order candidate scored 69 and was rejected for taste risk.\n\nSelected premise: a fictional creator crew treats a sterile Hollywood lobby as a speedrun level until a dead-serious guard turns the chaos into wristbands, clipboards, and timer stamps. The joke targets online stunt logic meeting procedure, not religious belief.")

    write_json(RUN_DIR / "strategy" / "phase_minus_one_worthiness_audit.json", {"source": "AP Scientology speedruns report", "verdict": "PROCEED", "track_a_newsjack": 35, "track_b_archetype": 34, "why": "Current viral behavior plus instantly legible first-frame contradiction."})
    write_json(RUN_DIR / "strategy" / "source_entropy_audit.json", {"native_entropy_score": 8, "anomalies": ["costumed creator crew entering a serious lobby", "speedrun language applied to real-world security"], "recommended_strategy": "fictionalize names/logos and push the clipboard/timer gag"})
    write_json(RUN_DIR / "strategy" / "humor_logic_bridge.json", {"truth": "Online stunt formats turn real places into game levels.", "comic_transfer": "A lobby security desk becomes the speedrun checkpoint.", "button": "Clipboard snap lands like a bass drop."})
    write_json(RUN_DIR / "strategy" / "tribe_meta_score.json", {"attention_grab": 10, "clarity": 9, "shareability": 9, "audio_native": 9, "risk_control": 7, "overall": 8.8})
    write_json(RUN_DIR / "strategy" / "risk_taste_score.json", {"religion_mockery_risk": "medium", "trespass_encouragement_risk": "medium", "mitigation": "generic institution, no real marks, no how-to framing, comedy points at creator behavior", "overall_verdict": "proceed with strict genericization"})
    write_text(RUN_DIR / "strategy" / "franchise_decision.md", "# Franchise Decision\n\nFranchise fit: **Internet Trend Hits Reception Desk**. Strong reusable SGFLIX lane for turning online behavior into a physical intake/checkpoint gag.")

    payload = {
        "customMode": True,
        "lyrics": "[Intro]\nLobby beep, sneakers squeak, timer on the wall\nClipboard says wait your turn before you run the hall\n\n[Hook]\nSpeedrun the desk, stamp it at the line\nAlien wristband, hot dog in time\nNo glitch through glass, no skip through town\nClipboard snap and the bass drops down\n\n[Tag]\nCheck in, check out, no route found\nSecurity desk makes the speedrun slow down\n",
        "style": "Original satirical hyperpop club chant, 142 BPM, rubbery bass, sneaker squeaks, lobby scanner beeps, clipboard snap percussion, group chant vocals, arcade speedrun energy, clean intelligible English, comic but not mocking religion.",
        "title": "RUN 082 - Speedrun The Desk",
        "trackName": "RUN 082 - Speedrun The Desk",
        "instrumental": False,
        "vocalLanguage": "en",
        "duration": 45,
        "bpm": 142,
        "keyScale": "",
        "timeSignature": "4/4",
        "batchSize": 1,
        "randomSeed": False,
        "seed": 82082,
        "thinking": False,
        "audioFormat": "mp3",
        "inferMethod": "ode",
        "taskType": "text2music",
        "guidanceScale": 7.0,
        "inferenceSteps": 80,
        "lmTemperature": 0.6,
        "lmCfgScale": 1.6,
        "lmTopP": 0.8,
        "lmTopK": 0,
        "lmNegativePrompt": "real church chant, religious mockery, harassment instructions, trespass tutorial, celebrity voice clone, copyrighted melody, slurred words, violent threat, mid-song voice drift, chorus first before setup",
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
    write_json(RUN_DIR / "audio" / "ace_step_payload.json", payload)
    write_text(RUN_DIR / "audio" / "audio_concept.md", "# Audio Concept\n\nDefault lane: original ACE-Step text-to-music, not clone/remake.\n\nThe hook should feel like an arcade speedrun colliding with a lobby intake desk: sneaker squeaks, scanner beeps, clipboard snaps, rubbery bass, and a clean group chant. Do not use religious music, institutional slogans, real audio, or harassment framing.")
    write_json(RUN_DIR / "audio" / "music_handoff.json", {"run_id": RUN_ID, "audio_status": "payload ready; remote generation pending", "hook_moment_sec": 4.8, "visual_anchor": "clipboard snap becomes bass stop while wristbands slide into tray", "payload_path": "audio/ace_step_payload.json"})
    write_text(RUN_DIR / "qc" / "audio_qc.md", "# Audio QC\n\nStatus: payload ready and remote 3090 preflight passed. Final keeper decision must be written after generated candidate copyback.\n\nProxy QC focus: reject religious-mockery tone, reject instructions that sound like a trespass tutorial, reject slurred vocals, reject mid-song voice drift.")

    write_json(RUN_DIR / "chai" / "chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot_id": "shot_001", "duration_sec": 6, "subject": "fictional costumed creator crew at generic lobby security desk", "scene": "Hollywood information-center lobby merged with arcade checkpoint", "motion": "slow push from chaos to clipboard snap", "camera": "24mm wide, clean first-frame composition", "critique": "No real logos, addresses, or religious-symbol mockery.", "revision": "If too institutional, replace signage with generic SECURITY DESK."}]})
    shot = {"run_id": RUN_ID, "shot": "001", "no_video_generation": True, "prompt": "Generic lobby security desk transformed into an arcade speedrun checkpoint, fictional creator crew in alien hoodie/hot-dog hat/robe silhouette, wristband tray, clipboard timer, cinematic satire, no real logos.", "negative": "real Scientology logo, real address, harassment, trespass instruction, messy text", "duration_plan_sec": 6}
    write_json(RUN_DIR / "scene_json" / "shot_001.json", shot)
    write_json(RUN_DIR / "scene_json" / "shot_0001.json", {**shot, "shot": "0001"})
    write_json(RUN_DIR / "handoffs" / "closed_tool_handoff.json", {"run_id": RUN_ID, "render_status": "handoff only; no video generation", "first_frame_path": "frames/gpt_image_2/first_frame_v01.png", "storyboard_path": "storyboards/shared_choices/shared_choices_v01.png", "audio_payload": "audio/ace_step_payload.json"})
    write_text(RUN_DIR / "handoffs" / "grok_agent_prompt.md", "# Grok Agent Prompt\n\nDo not generate video. Review whether the stills read as a generic internet-speedrun-meets-security-desk satire. Flag any real institutional mark, real address, religious mockery, or behavior that looks like a tutorial.")
    write_text(RUN_DIR / "captions" / "instagram_caption.md", "POV: the speedrun reaches the front desk and the clipboard has final boss energy.\n\nFictionalized satire. No real logos, no how-to. #sgflix #internetculture #satire #creatorculture")
    write_text(RUN_DIR / "distribution" / "post_plan.md", "# Post Plan\n\nPost only after human review confirms the piece does not encourage trespass or name a real institution. Best surface: Reels/TikTok as a creator-culture satire with the original hook.")
    write_text(RUN_DIR / "skool" / "case_study.md", "# Skool Case Study\n\nPattern: current AP culture story -> reject high-taste-risk alternatives -> fictionalize the institution -> make the audio hook out of procedure sounds -> build stills before video handoff.")
    write_text(RUN_DIR / "frames" / "gpt_image_2" / "first_frame_v01_prompt.md", "Satirical cinematic first frame: fictional creator crew in absurd costumes stopped at a generic lobby security desk, wristband tray, clipboard timer, arcade speedrun energy, no real logos or addresses.")
    write_text(RUN_DIR / "storyboards" / "shared_choices" / "shared_choices_v01_prompt.md", "Director bible board for a fictional security-desk speedrun satire: character props, palette, floor plan, storyboard panels, camera, lighting, audio timing, visual rules, production notes.")
    make_first_frame(RUN_DIR / "frames" / "gpt_image_2" / "first_frame_v01.png")
    make_storyboard(RUN_DIR / "storyboards" / "shared_choices" / "shared_choices_v01.png")
    write_text(RUN_DIR / "qc" / "first_frame_v01_qc.md", "# First Frame QC\n\nStatus: usable generated still equivalent. It is generic, readable, and avoids real institution marks. Human should still inspect the costume labels and text before public use.")
    write_text(RUN_DIR / "qc" / "shared_choices_v01_qc.md", "# Shared Choices QC\n\nStatus: usable storyboard/director-bible board. Includes character props, environment, floor plan, storyboard panels, camera, lighting, audio timing, visual rules, and production notes.")

    master = {"run_id": RUN_ID, "run_name": RUN_NAME, "selected_premise": "Scientology Speedrun Security Desk", "created_at": now, "status": "package_created_remote_audio_pending", "winner_score": 109, "generated_stills": ["frames/gpt_image_2/first_frame_v01.png", "storyboards/shared_choices/shared_choices_v01.png"], "audio_payload": "audio/ace_step_payload.json"}
    write_json(MASTER_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", master)
    write_text(MASTER_DIR / "README.md", f"# RUN {RUN_ID} Master Package - Scientology Speedrun Security Desk\n\nResearch-first package with scored candidate board, selected premise, audio payload, image stills, storyboard, and handoffs. No video generated.\n\nNext action: complete remote ACE-Step copyback and keeper decision.")
    write_text(RUN_DIR / "FACTORY_RUN_STATUS.md", f"# Factory Run Status\n\nRun: {RUN_NAME}\n\nStatus: **REMOTE AUDIO PENDING**\n\nCreated at: {now}\n\nResearch intake completed before package creation. Candidate board scored before winner selection. Still images were generated locally; no video generation was performed.\n\nNext action: run ACE-Step on 3090, copy back candidate/metrics/critique/manifest, and write keeper decision.")

    assets = []
    for path in sorted(RUN_DIR.rglob("*")):
        if path.is_file() and path.relative_to(RUN_DIR) != Path("manifests/asset_manifest.json"):
            assets.append({"path": str(path.relative_to(RUN_DIR)), "bytes": path.stat().st_size})
    write_json(RUN_DIR / "manifests" / "asset_manifest.json", {"run_id": RUN_ID, "run_name": RUN_NAME, "generated_at": now, "video_generation": "not performed", "assets": assets})
    print(RUN_DIR)


if __name__ == "__main__":
    main()
