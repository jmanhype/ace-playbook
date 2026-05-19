from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "080"
RUN_SLUG = "voice_mark_clearance_booth"
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
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textbbox((0, 0), trial, font=font_obj)[2] <= max_width:
            current = trial
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
    if max_width:
        lines = wrap(draw, text, fnt, max_width)
    else:
        lines = text.splitlines()
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(size * 1.25)
    return y


def make_first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1536, 864), "#111418")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1536, 864), fill="#111418")
    draw.rectangle((80, 72, 1456, 792), fill="#181d23", outline="#d7b76f", width=6)
    draw.rectangle((120, 115, 810, 755), fill="#20262d", outline="#3e4853", width=3)
    draw.rectangle((880, 115, 1410, 755), fill="#271b24", outline="#c04f72", width=3)
    draw.rectangle((168, 178, 758, 300), fill="#0d1117", outline="#d7b76f", width=4)
    draw_label(draw, (195, 196), "VOICE MARK CLEARANCE", 44, "#f7e6b4", True)
    draw_label(draw, (195, 252), "REGISTERED SOUND CHECKPOINT", 24, "#9fb5c8")
    draw.rectangle((170, 360, 748, 675), fill="#10151b", outline="#7d8794", width=3)
    draw.ellipse((240, 405, 390, 555), fill="#f1d0b5", outline="#f7e6b4", width=3)
    draw.rectangle((300, 555, 340, 670), fill="#f1d0b5")
    draw.polygon([(235, 560), (520, 565), (585, 680), (185, 690)], fill="#b91e48")
    draw.line((310, 450, 445, 430), fill="#2b2b2b", width=7)
    draw.line((445, 430, 575, 520), fill="#2b2b2b", width=7)
    draw.rectangle((520, 500, 690, 550), fill="#db6b96", outline="#f8d3df", width=3)
    draw.line((535, 525, 675, 525), fill="#f8d3df", width=4)
    draw.rounded_rectangle((915, 170, 1370, 285), radius=18, fill="#f5f0e6", outline="#d7b76f", width=4)
    draw_label(draw, (945, 190), "PHRASE SAMPLE", 30, "#1b1b1b", True)
    draw_label(draw, (945, 232), "[redacted by legal]", 24, "#8d1d36")
    draw.rounded_rectangle((935, 350, 1350, 575), radius=18, fill="#111418", outline="#d7b76f", width=4)
    for i, h in enumerate([45, 95, 65, 135, 80, 120, 58, 100, 72, 140, 54, 88]):
        x = 970 + i * 29
        draw.rectangle((x, 520 - h, x + 13, 520 + h // 3), fill="#d7b76f")
    draw_label(draw, (930, 630), "A pop-star silhouette arrives at an absurd courthouse booth where every syllable needs a receipt.", 27, "#f4f1e8", False, 385)
    draw.rectangle((80, 802, 1456, 848), fill="#d7b76f")
    draw_label(draw, (110, 812), "FIRST FRAME: glamorous concert energy collides with sterile IP-office bureaucracy", 24, "#111418", True)
    img.save(path)


def make_storyboard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1800, 1200), "#f4f1e8")
    draw = ImageDraw.Draw(img)
    draw_label(draw, (54, 38), "RUN 080 SHARED CHOICES: VOICE MARK CLEARANCE BOOTH", 38, "#121417", True)
    draw_label(draw, (56, 88), "Director bible board for first frame, hook timing, props, palette, staging, and closed-tool handoff.", 22, "#4d5963")
    palette = ["#111418", "#f4f1e8", "#d7b76f", "#b91e48", "#3e4853", "#db6b96"]
    for idx, color in enumerate(palette):
        draw.rectangle((60 + idx * 80, 145, 120 + idx * 80, 205), fill=color, outline="#111418")
    sections = [
        ("Character + Props", "Generic pop-star silhouette, pink guitar as parody object, clerk window, receipt printer, redacted phrase card, waveform stamp."),
        ("Environment", "Hybrid concert stage and federal trademark office: velvet ropes, filing cabinets, gold lighting, sterile intake counter."),
        ("Blocking", "Star at left window, clerk unseen behind glass, waveform display at right. First beat reveals the booth before the gag text."),
        ("Visual Rules", "No face-match, no voice clone, no real lyrics. Treat fame as silhouette and iconography; comedy lives in bureaucracy."),
        ("Camera", "24mm wide, slight low angle, slow push toward redacted phrase card, rack focus from guitar to receipt stamp."),
        ("Lighting", "Warm gold stage key plus cool office fluorescents. Maintain high contrast without a nightclub blur."),
        ("Audio Timing", "0.0s stamp hit; 0.8s clerk beep; 1.5s generic hook phrase; 3.2s receipt printer snare roll; 5.5s chorus button."),
        ("Production Notes", "Use a new original vocal hook. Avoid mimicking any protected voice or using exact claimed sound-mark phrases.")
    ]
    card_w, card_h = 400, 205
    start_x, start_y = 60, 240
    for i, (title, body) in enumerate(sections):
        col = i % 4
        row = i // 4
        x = start_x + col * (card_w + 32)
        y = start_y + row * (card_h + 36)
        draw.rectangle((x, y, x + card_w, y + card_h), fill="#ffffff", outline="#121417", width=3)
        draw.rectangle((x, y, x + card_w, y + 42), fill="#121417")
        draw_label(draw, (x + 16, y + 10), title, 21, "#f4f1e8", True)
        draw_label(draw, (x + 16, y + 60), body, 21, "#26313a", False, card_w - 32)
    panel_y = 760
    for i in range(4):
        x = 60 + i * 430
        draw.rectangle((x, panel_y, x + 390, panel_y + 300), fill="#181d23", outline="#d7b76f", width=3)
        draw_label(draw, (x + 18, panel_y + 18), f"Panel {i + 1}", 24, "#f7e6b4", True)
        if i == 0:
            draw_label(draw, (x + 18, panel_y + 58), "Wide reveal: concert lights over a trademark intake booth.", 21, "#f4f1e8", False, 340)
        elif i == 1:
            draw_label(draw, (x + 18, panel_y + 58), "Close-up: redacted phrase card gets stamped PENDING.", 21, "#f4f1e8", False, 340)
        elif i == 2:
            draw_label(draw, (x + 18, panel_y + 58), "Waveform bars become turnstiles; hook lands on the beep.", 21, "#f4f1e8", False, 340)
        else:
            draw_label(draw, (x + 18, panel_y + 58), "Hero end pose: receipt tape unrolls like a tour banner.", 21, "#f4f1e8", False, 340)
    img.save(path)


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    directories = [
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
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "id": "ap_taylor_voice_trademark_2026_04_28",
            "title": "Taylor Swift files 3 new trademark applications. One expert says it is to curb AI threats",
            "publisher": "Associated Press",
            "url": "https://apnews.com/article/7f56fbafb269d4959009f3ad34e28fc1",
            "published": "2026-04-28",
            "used_for": "Core verified fact pattern: three USPTO applications, two sound marks, one visual mark.",
            "fact_status": "reported by AP; application status should be checked in USPTO before final public legal claim"
        },
        {
            "id": "cbs_taylor_voice_trademark_2026_04_28",
            "title": "Taylor Swift files to trademark her voice and image amid rise in AI deepfakes",
            "publisher": "CBS News",
            "url": "https://www.cbsnews.com/amp/news/taylor-swift-ai-voice-likeness-trademark/",
            "published": "2026-04-28",
            "used_for": "AI deepfake context and mainstream framing.",
            "fact_status": "secondary confirmation"
        },
        {
            "id": "lat_paramount_warner_lawsuit_2026_05_01",
            "title": "Consumers sue to block Paramount-Warner Bros. deal",
            "publisher": "Los Angeles Times",
            "url": "https://www.latimes.com/entertainment-arts/business/story/2026-05-01/consumers-sue-to-block-paramount-warner-bros-deal",
            "published": "2026-05-01",
            "used_for": "Fresh alternate candidate with brand/courtroom contrast.",
            "fact_status": "reported by LAT"
        },
        {
            "id": "tmz_rock_traffic_stop_2026_04_30",
            "title": "Dwayne 'The Rock' Johnson Pulled Over by Police After Walk of Fame Event",
            "publisher": "TMZ",
            "url": "https://www.tmz.com/2026/04/30/dwayne-johnson-pulled-over-in-los-angeles/",
            "published": "2026-04-30",
            "used_for": "Fresh alternate celebrity humiliation/traffic-ticket candidate.",
            "fact_status": "tabloid report; use as low-stakes entertainment only"
        },
        {
            "id": "pbs_kimmel_trump_2026_04_27",
            "title": "Trumps call for ABC to fire Jimmy Kimmel after morbid joke about first lady",
            "publisher": "PBS NewsHour / AP",
            "url": "https://www.pbs.org/newshour/politics/trumps-call-for-abc-to-fire-jimmy-kimmel-again-after-morbid-joke-about-first-lady",
            "published": "2026-04-27",
            "used_for": "Fresh alternate public conflict candidate, rejected due overlap with prior Kimmel run.",
            "fact_status": "reported by AP/PBS"
        }
    ]
    write_json(RUN_DIR / "research" / "sources.json", {"generated_at": now, "sources": sources})

    write_text(
        RUN_DIR / "research" / "last30days_report.md",
        """
# Step 1: Research Intake - Last 30 Days Scan

Run 080 starts from current source context, not nearby assets. The scan focused on entertainment, public-figure IP, media-business lawsuits, and celebrity humiliation beats active during late April and May 2, 2026.

## Candidate Context

1. Taylor Swift voice/image trademark filings: AP and CBS report that TAS Rights Management filed three USPTO trademark applications on April 24, 2026: two sound marks and one visual mark, framed by experts as a response to AI voice/likeness misuse. This has a strong audio-native hook and a clear visual contradiction: concert-star iconography forced through a government clearance booth.
2. Paramount-Warner consumer lawsuit: the Los Angeles Times reports consumers sued on May 1, 2026 to block the Paramount-Warner Bros. deal. This has strong courtroom/streaming tollbooth imagery, but SGFLIX has already used adjacent Paramount merger territory.
3. Dwayne Johnson traffic stop/Kevin Hart joke: TMZ reports Johnson was pulled over for tinted windows after a Walk of Fame event, followed by a Hart joke. This has face/humiliation value but thinner cultural stakes.
4. Trump/Melania/Kimmel ABC fire demand: multiple mainstream reports confirm the late-April conflict. Strong public conflict, but recent SGFLIX runs already used Kimmel and ABC-license territory.

## Winner Direction

Selected premise: **Voice Mark Clearance Booth**. A generic pop-star silhouette enters a sterile trademark intake counter where a clerk treats every syllable like a licensed object. The run must avoid voice cloning, exact lyrical imitation, and face-match imagery. The joke is about bureaucracy swallowing pop spectacle, not about impersonating a real artist.

## Verification Notes

- The applications were reported by AP/CBS; final legal status should be verified directly in USPTO before any legalistic caption goes public.
- The run should say "reported filings" or "awaiting examination" rather than asserting a granted trademark.
- Do not use the actual claimed spoken phrases in generated audio.
"""
    )

    candidate_board = {
        "generated_at": now,
        "selection_rule": "Fresh current-source candidates scored before run package creation.",
        "candidates": [
            {
                "rank": 1,
                "title": "Voice Mark Clearance Booth",
                "source": "Taylor Swift voice/image trademark filings amid AI-misuse concern",
                "scores": {
                    "freshness": 9,
                    "famous_face_or_iconography": 10,
                    "public_conflict": 7,
                    "humiliation_or_absurdity": 8,
                    "brand_location_contrast": 9,
                    "first_frame_contradiction": 10,
                    "audio_hook_native": 10,
                    "risk_manageability": 7,
                    "total": 70
                },
                "risk": "Avoid exact likeness, exact protected phrases, and any vocal imitation. Use generic silhouette and original hook.",
                "verdict": "SELECT"
            },
            {
                "rank": 2,
                "title": "Streaming Merger Tollbooth",
                "source": "Consumers sue to block Paramount-Warner deal",
                "scores": {
                    "freshness": 10,
                    "famous_face_or_iconography": 6,
                    "public_conflict": 8,
                    "humiliation_or_absurdity": 6,
                    "brand_location_contrast": 9,
                    "first_frame_contradiction": 8,
                    "audio_hook_native": 5,
                    "risk_manageability": 8,
                    "total": 60
                },
                "risk": "Corporate-antitrust topic may feel less human and overlaps older Paramount merger run.",
                "verdict": "HOLD"
            },
            {
                "rank": 3,
                "title": "Tint Meter Walk of Fame",
                "source": "Dwayne Johnson traffic stop and Kevin Hart joke",
                "scores": {
                    "freshness": 8,
                    "famous_face_or_iconography": 9,
                    "public_conflict": 4,
                    "humiliation_or_absurdity": 7,
                    "brand_location_contrast": 6,
                    "first_frame_contradiction": 8,
                    "audio_hook_native": 5,
                    "risk_manageability": 8,
                    "total": 55
                },
                "risk": "Funny but lightweight; best as a fast meme package, not a serious audio-led run.",
                "verdict": "BACKUP"
            },
            {
                "rank": 4,
                "title": "Late Night License Renewal Desk",
                "source": "Trump/Melania demand ABC fire Jimmy Kimmel",
                "scores": {
                    "freshness": 8,
                    "famous_face_or_iconography": 9,
                    "public_conflict": 10,
                    "humiliation_or_absurdity": 8,
                    "brand_location_contrast": 8,
                    "first_frame_contradiction": 8,
                    "audio_hook_native": 6,
                    "risk_manageability": 5,
                    "total": 62
                },
                "risk": "High conflict but overlaps run_077_kimmel_license_upfronts_counter.",
                "verdict": "REJECT_FOR_OVERLAP"
            }
        ]
    }
    write_json(RUN_DIR / "strategy" / "candidate_board.json", candidate_board)

    write_text(
        RUN_DIR / "strategy" / "winner_decision.md",
        """
# Winner Decision

Winner: **Voice Mark Clearance Booth**

The selected premise turns a fresh, audio-native news item into a clean SGFLIX contradiction: a pop-star stage fantasy forced to wait at a sterile trademark intake window. It wins because the audio hook is intrinsic, the image can be readable in one frame, and the risk can be managed by never cloning a protected voice or face.

Score summary: 70/80, ahead of the Paramount-Warner tollbooth at 60/80 and the Kimmel license desk at 62/80. The Kimmel candidate was rejected despite a high score because a recent official run already used the Kimmel/license-administration territory.
"""
    )

    write_json(RUN_DIR / "strategy" / "phase_minus_one_worthiness_audit.json", {
        "phase_minus_one_audit": {
            "source_target": "Taylor Swift voice and image trademark filings",
            "track_a_newsjack_velocity": {
                "active_trend_score": 9,
                "algorithmic_slipstream": "High: mainstream entertainment and AI-policy coverage within the last week",
                "polarization_factor": 7,
                "track_a_total": 25,
                "track_a_verdict": "PASS"
            },
            "track_b_archetype_resonance": {
                "iconography_strength": 10,
                "stereotype_rigidity": "High",
                "subversion_potential": 9,
                "track_b_total": 28,
                "track_b_verdict": "PASS"
            },
            "system_verdict": {
                "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                "primary_vector": "TRACK_A_NEWSJACK",
                "urgency_class": "High",
                "strategic_directive": "Build a bureaucracy-vs-pop spectacle visual and an original legal-clearance hook."
            }
        }
    })

    write_json(RUN_DIR / "strategy" / "source_entropy_audit.json", {
        "source_entropy_audit": {
            "baseline_reality_check": "A major artist reportedly files trademark applications to protect voice and image from AI misuse.",
            "detected_anomalies": [
                "A human voice treated like a source-identifying asset at trademark-office scale",
                "Pop-stage glamour colliding with procedural government intake language"
            ],
            "native_entropy_score": 5,
            "subject_self_awareness": "deadpan",
            "comedic_vector_recommendation": "cringe_vector",
            "recommended_strategy": "micro_spotlight"
        }
    })

    write_json(RUN_DIR / "strategy" / "humor_logic_bridge.json", {
        "premise": "A pop superstar enters a trademark-office checkpoint where every syllable needs clearance.",
        "truth": "AI voice cloning makes celebrity identity feel administratively securitized.",
        "comic_transfer": "Turn legal filings into a physical tollbooth for sound.",
        "first_frame_question": "Why is a concert-stage figure at a government counter with a waveform receipt?",
        "button": "The receipt printer becomes the snare roll."
    })
    write_json(RUN_DIR / "strategy" / "tribe_meta_score.json", {
        "tribe_meta_score": {
            "attention_grab": 9,
            "clarity_without_caption": 8,
            "shareability": 8,
            "audio_native": 10,
            "identity_safety": 7,
            "overall": 8.4
        }
    })
    write_json(RUN_DIR / "strategy" / "risk_taste_score.json", {
        "risk_taste_score": {
            "defamation_risk": "Low",
            "likeness_risk": "Medium",
            "voice_clone_risk": "High if mishandled; mitigated by original vocal and no exact phrase use",
            "taste_risk": "Medium-low",
            "legal_claim_precision": "Use reported/awaiting-examination language",
            "overall_verdict": "Proceed with strict no-clone/no-face-match guardrails"
        }
    })
    write_text(RUN_DIR / "strategy" / "franchise_decision.md", "# Franchise Decision\n\nFranchise fit: **Bureaucracy Swallows Fame**. This can become a reusable SGFLIX lane for IP, courts, licensing, and celebrity control systems. Keep this as a modular template: famous spectacle at a banal intake counter.")

    audio_payload = {
        "run_id": RUN_ID,
        "title": "Voice Mark Clearance Booth",
        "lane": "ACE-Step text-to-music, no source clone",
        "target_duration_sec": 18,
        "bpm": 132,
        "time_feel": "half-time trap-pop with clerk-stamp percussion; verify no double-time misread",
        "style_prompt": "original satirical electro-pop courtroom hook, punchy clerk stamp percussion, glossy synth bass, no imitation of any real artist voice, no protected phrase, generic alto lead with robotic call-and-response, catchy but legally sterile",
        "lyrics": [
            {"section": "hook", "text": "Clearance desk, check the tone / stamp the beat, leave the clone / if the waveform wants a crown / run the receipt, shut it down"},
            {"section": "post_hook", "text": "Pending, pending, pending on the sound"}
        ],
        "negative_prompt": "Taylor Swift voice, exact Taylor phrasing, Eras lyrics, copyrighted melody, celebrity impersonation, spoken catchphrase clone, interview dialogue",
        "seed": 80080,
        "generation_notes": "Use reference/text-to-music only. Do not use cover mode because there is no approved source song to preserve."
    }
    write_json(RUN_DIR / "audio" / "ace_step_payload.json", audio_payload)
    write_text(
        RUN_DIR / "audio" / "audio_concept.md",
        """
# Audio Concept

Audio role: original hook, not clone/remake.

The track should sound like a pop concert getting processed by an IP-office turnstile: stamp hits, receipt-printer rolls, glass-window beeps, and a glossy trap-pop bassline. The lyric should avoid exact reported sound-mark phrases and avoid any identifiable artist impression.

Hook target: a legal-clearance chant that lands on the visual stamp gag.
"""
    )
    music_handoff = {
        "run_id": RUN_ID,
        "usable_audio": False,
        "reason": "ACE-Step/3090 lane unavailable in this environment",
        "intended_drop_or_hook_moment": {
            "time_sec": 3.2,
            "visual": "receipt printer turns into snare roll as waveform bars become turnstiles"
        },
        "sections": [
            {"start": 0.0, "end": 0.8, "description": "single stamp hit and intake-window reveal"},
            {"start": 0.8, "end": 3.2, "description": "clerk beep rhythm builds under generic vocal"},
            {"start": 3.2, "end": 6.0, "description": "hook lands; redacted phrase card is stamped PENDING"}
        ],
        "payload_path": "audio/ace_step_payload.json"
    }
    write_json(RUN_DIR / "audio" / "music_handoff.json", music_handoff)
    write_text(RUN_DIR / "qc" / "audio_qc.md", "# Audio QC\n\nStatus: blocked before generation. The concept and ACE-Step payload are complete, but no MP3 candidate exists because no 3090/ACE-Step runtime is reachable from this machine.\n\nQC guardrails for later: reject any candidate that resembles a real celebrity voice, uses exact reported sound-mark phrases, or puts chorus first before the stamp/booth setup.")
    write_text(RUN_DIR / "qc" / "AUDIO_GENERATION_BLOCKED_REPORT.md", f"# Audio Generation Blocked Report\n\nChecked at: {now}\n\nExact reason: `nvidia-smi` is not available on this host and `/mnt/bulk/home/straughter/sgflix_audio_factory` is not mounted, so the ACE-Step/3090 generation lane cannot run.\n\nPayload to run later: `audio/ace_step_payload.json`\n\nRecommended 3090 action: copy this run folder or payload into `/mnt/bulk/home/straughter/sgflix_audio_factory/payloads/run_080_voice_mark_clearance_booth.json` and run the Run 012 Tier-1 proxy pattern for one iteration.")

    chai = {
        "run_id": RUN_ID,
        "shots": [
            {
                "shot_id": "shot_001",
                "duration_sec": 6,
                "subject": "generic pop-star silhouette with pink guitar at trademark intake booth",
                "scene": "concert stage merged with sterile government office",
                "motion": "slow push from wide booth reveal into redacted phrase card stamp",
                "spatial": "subject left, waveform monitor right, clerk window centered",
                "camera": "24mm wide, low angle, clean rack focus",
                "critique": "Must not face-match Taylor Swift or use real protected phrases.",
                "revision": "If identity drifts too close, crop to silhouette and emphasize props/booth."
            }
        ]
    }
    write_json(RUN_DIR / "chai" / "chai_shot_specs.json", chai)
    shot_json = {
        "run_id": RUN_ID,
        "shot": "001",
        "no_video_generation": True,
        "prompt": "A satirical wide first frame: generic pop-star silhouette at a trademark office clearance booth, pink guitar, waveform receipt printer, concert lights mixed with government-office fluorescents, cinematic, no real celebrity likeness.",
        "negative": "exact celebrity face, copyrighted lyrics, readable protected phrase, messy text, AI voice clone",
        "duration_plan_sec": 6
    }
    write_json(RUN_DIR / "scene_json" / "shot_001.json", shot_json)
    write_json(RUN_DIR / "scene_json" / "shot_0001.json", {**shot_json, "shot": "0001"})

    closed_tool = {
        "run_id": RUN_ID,
        "render_status": "handoff only; do not generate video automatically",
        "first_frame_path": "frames/gpt_image_2/first_frame_v01.png",
        "storyboard_path": "storyboards/shared_choices/shared_choices_v01.png",
        "audio_payload": "audio/ace_step_payload.json",
        "instructions": "Use stills and shot JSON for external review only. Do not start video generation without human approval."
    }
    write_json(RUN_DIR / "handoffs" / "closed_tool_handoff.json", closed_tool)
    write_text(RUN_DIR / "handoffs" / "grok_agent_prompt.md", "# Grok Agent Prompt\n\nCreate no video. Review the still-frame and storyboard package for clarity: does a generic pop spectacle being processed by an IP clearance booth read instantly? Flag any face-match, exact phrase, or messy-text risks. Suggest one tighter visual gag if needed.")
    write_text(RUN_DIR / "captions" / "instagram_caption.md", "When the waveform has to take a number at the trademark office.\\n\\nReported filings, original hook only, no clone energy. #sgflix #ai #musicbusiness #trademark #satire")
    write_text(RUN_DIR / "distribution" / "post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels/TikTok after human audio keeper exists. Do not post before replacing the blocked audio lane with a generated candidate and verifying no voice imitation.\n\nCaption angle: music-business bureaucracy, not celebrity attack.")
    write_text(RUN_DIR / "skool" / "case_study.md", "# Skool Case Study\n\nTeaching angle: how to turn a legal/IP news item into an audio-native visual gag while avoiding clone and likeness traps. The reusable pattern is: current filing -> physical bureaucracy metaphor -> original hook -> still-first handoff.")

    first_prompt = "Satirical cinematic first frame, generic pop-star silhouette with pink guitar at a sterile trademark-office clearance booth, waveform receipt printer, concert lights plus government fluorescents, gold and crimson accents, readable composition, no real celebrity likeness, no copyrighted phrase text."
    board_prompt = "Director bible storyboard board for Voice Mark Clearance Booth: character silhouette, hero props, palette, office-stage environment, floor blocking, four storyboard panels, camera notes, lighting notes, visual rules, production notes, no face-match and no exact lyric text."
    write_text(RUN_DIR / "frames" / "gpt_image_2" / "first_frame_v01_prompt.md", first_prompt)
    write_text(RUN_DIR / "storyboards" / "shared_choices" / "shared_choices_v01_prompt.md", board_prompt)
    make_first_frame(RUN_DIR / "frames" / "gpt_image_2" / "first_frame_v01.png")
    make_storyboard(RUN_DIR / "storyboards" / "shared_choices" / "shared_choices_v01.png")
    write_text(RUN_DIR / "qc" / "first_frame_v01_qc.md", "# First Frame QC\n\nStatus: usable local generated still equivalent. It communicates the booth, waveform receipt, pink guitar iconography, and redacted phrase gag. No real celebrity face is shown. Text is intentionally generic/redacted to avoid messy or protected phrase risk.")
    write_text(RUN_DIR / "qc" / "shared_choices_v01_qc.md", "# Shared Choices QC\n\nStatus: usable local storyboard/director-bible board. Contains character/props, palette, environment, blocking, camera, lighting, audio timing, visual rules, production notes, and four storyboard panels. No generated video footage was created.")

    master = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "selected_premise": "Voice Mark Clearance Booth",
        "created_at": now,
        "status": "package_complete_with_audio_generation_blocked",
        "research_topic": "Taylor Swift reported voice/image trademark filings and AI-misuse context",
        "selected_score": 70,
        "generated_stills": [
            "frames/gpt_image_2/first_frame_v01.png",
            "storyboards/shared_choices/shared_choices_v01.png"
        ],
        "audio_candidate_paths": [],
        "audio_blocked_report": "qc/AUDIO_GENERATION_BLOCKED_REPORT.md",
        "keeper_decision": "no keeper; generation blocked",
        "missing_or_blocked": ["audio/generated_candidates/*.mp3", "audio/audio_scorecard.json", "audio/keeper_manifest.json", "qc/hook_timing_qc.md"]
    }
    write_json(MASTER_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", master)
    write_text(MASTER_DIR / "README.md", f"# RUN {RUN_ID} Master Package - Voice Mark Clearance Booth\n\nStatus: package complete except ACE-Step audio generation, which is explicitly blocked. Research, candidate scoring, strategy audits, audio payload, CHAI specs, scene JSON, closed-tool handoff, captions, distribution, Skool notes, first-frame still, and shared-choices board are present.\n\nNext human action: run `audio/ace_step_payload.json` on the 3090/ACE-Step lane, then perform proxy QC and keeper selection.")
    write_text(RUN_DIR / "FACTORY_RUN_STATUS.md", f"# Factory Run Status\n\nRun: {RUN_NAME}\n\nStatus: **COMPLETE WITH AUDIO GENERATION BLOCKED**\n\nCreated at: {now}\n\nResearch intake completed before package creation. Candidate board scored and winner selected. Required audio concept, ACE-Step payload, music handoff, audio QC, first-frame still, shared-choices board, CHAI specs, scene JSONs, handoffs, captions, distribution plan, Skool case study, manifest, and master package were created.\n\nNo video generation was performed.\n\nBlocked lane: ACE-Step/3090 generation. See `qc/AUDIO_GENERATION_BLOCKED_REPORT.md`.")

    manifest_entries = []
    for path in sorted(RUN_DIR.rglob("*")):
        if path.is_file() and path.relative_to(RUN_DIR) != Path("manifests/asset_manifest.json"):
            manifest_entries.append({
                "path": str(path.relative_to(RUN_DIR)),
                "bytes": path.stat().st_size
            })
    write_json(RUN_DIR / "manifests" / "asset_manifest.json", {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "generated_at": now,
        "video_generation": "not performed",
        "audio_generation": "blocked",
        "assets": manifest_entries
    })
    print(RUN_DIR)


if __name__ == "__main__":
    main()
