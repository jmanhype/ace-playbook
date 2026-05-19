from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = "069"
SLUG = "banksy_flag_optometry"
ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def generate_image(prompt: str, path: Path, size: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("OPENAI_API_KEY"):
        err = path.with_suffix(".blocked.txt")
        write(err, "OPENAI_API_KEY was not available, so GPT Image generation could not run.")
        return {"ok": False, "mode": "blocked_no_openai_api_key", "error_path": str(err)}
    try:
        from openai import OpenAI

        result = OpenAI().images.generate(model="gpt-image-1", prompt=prompt, size=size)
        b64 = result.data[0].b64_json
        if not b64:
            raise RuntimeError("image response did not include b64_json")
        path.write_bytes(base64.b64decode(b64))
        return {"ok": True, "mode": "openai_gpt_image_api", "size": size}
    except Exception as exc:
        err = path.with_suffix(".blocked.txt")
        write(err, f"GPT Image generation failed: {type(exc).__name__}: {exc}")
        return {"ok": False, "mode": "blocked_openai_gpt_image_api", "error_path": str(err)}


SOURCES = [
    {
        "id": "ap_banksy_flag_statue_2026_04_30",
        "publisher": "Associated Press",
        "title": "Banksy confirms a new statue in central London of a man blinded by a flag is his work",
        "url": "https://apnews.com/article/a9094fcb5836e9c066eda36f33cb7395",
        "published": "2026-04-30",
        "verified_points": [
            "Banksy confirmed a central London sculpture of a suited man whose face is covered by a billowing flag.",
            "The sculpture appeared on a plinth at Waterloo Place near Buckingham Palace.",
            "Banksy posted a humorous installation video on Instagram.",
        ],
        "unverified_or_sensitive": [
            "The artist's intended political meaning is interpretive unless stated by Banksy.",
            "Do not imply a specific party, country, or real official is the subject.",
        ],
    },
    {
        "id": "ap_britney_dui_charge_2026_04_30",
        "publisher": "Associated Press",
        "title": "Britney Spears charged with driving under the influence of alcohol and drugs",
        "url": "https://apnews.com/article/395ba1c567ec3865a80ffe57e92ad127",
        "published": "2026-04-30",
        "verified_points": [
            "Spears was charged with a misdemeanor count in Ventura County.",
            "Prosecutors said a wet reckless offer would be standard protocol under the stated conditions.",
        ],
        "unverified_or_sensitive": ["Rejected for this run because treatment/addiction context raises taste risk."],
    },
    {
        "id": "ap_ye_wireless_sponsors_2026_04_06",
        "publisher": "Associated Press",
        "title": "Wireless Festival boss stands by Ye headlining concerts as sponsors pull out",
        "url": "https://apnews.com/article/458d0e3ea9b787f80ad503a269db7ed0",
        "published": "2026-04-06",
        "verified_points": ["Festival leadership defended Ye headlining while sponsor pressure was reported."],
        "unverified_or_sensitive": ["Rejected as duplicate-adjacent and high-risk hate-speech context."],
    },
    {
        "id": "ap_kimmel_fire_calls_2026_04_27",
        "publisher": "Associated Press",
        "title": "Trumps call for ABC to fire Jimmy Kimmel again after morbid joke about first lady",
        "url": "https://www.local10.com/news/politics/2026/04/27/trumps-call-for-abc-to-fire-jimmy-kimmel-again-after-morbid-joke-about-first-lady/",
        "published": "2026-04-27",
        "verified_points": ["AP reported renewed calls for ABC to fire Kimmel."],
        "unverified_or_sensitive": ["Rejected because multiple prior SGFLIX packages already own this lane."],
    },
    {
        "id": "ap_qatar_air_force_one_2026_05_02",
        "publisher": "Associated Press",
        "title": "Air Force says former Qatari 747 will be ready to fly Trump as Air Force One this summer",
        "url": "https://apnews.com/article/32966a04767cbe9c22a53979467c7f92",
        "published": "2026-05-02",
        "verified_points": ["AP reported the former Qatari 747 was expected to be ready for temporary Air Force One use this summer."],
        "unverified_or_sensitive": ["Rejected because run_041 already covered this specific gift-shop lane."],
    },
]

CANDIDATES = [
    {
        "id": "banksy_flag_optometry",
        "title": "Banksy Flag Optometry",
        "premise": "A faceless London monument walks into a street optometry clinic because the flag that makes him heroic also blocks the eye chart.",
        "source_ids": ["ap_banksy_flag_statue_2026_04_30"],
        "scores": {
            "freshness": 18,
            "famous_signal": 13,
            "visual_contradiction": 20,
            "conflict": 15,
            "object_comedy": 18,
            "taste_safety": 16,
        },
        "total": 100,
        "risk": "Do not copy Banksy's exact sculpture; use a fictional municipal statue and generic flag.",
        "decision": "selected",
    },
    {
        "id": "britney_wet_reckless_counter",
        "title": "Wet Reckless DMV Returns Counter",
        "premise": "A pop-star silhouette tries to return a luxury car key at a courthouse DMV counter labeled wet reckless.",
        "source_ids": ["ap_britney_dui_charge_2026_04_30"],
        "scores": {"freshness": 18, "famous_signal": 18, "visual_contradiction": 14, "conflict": 16, "object_comedy": 14, "taste_safety": 6},
        "total": 86,
        "risk": "High taste risk around impairment, treatment, and public mental-health history.",
        "decision": "rejected_taste_risk",
    },
    {
        "id": "ye_sponsor_exit_luggage",
        "title": "Sponsor Exit Luggage Claim",
        "premise": "Festival sponsors retrieve their logos from a baggage carousel while a headliner-shaped spotlight argues with border paperwork.",
        "source_ids": ["ap_ye_wireless_sponsors_2026_04_06"],
        "scores": {"freshness": 12, "famous_signal": 18, "visual_contradiction": 15, "conflict": 18, "object_comedy": 15, "taste_safety": 5},
        "total": 83,
        "risk": "Prior Ye/Wireless packages and hate-speech adjacency.",
        "decision": "rejected_duplicate_and_risk",
    },
    {
        "id": "kimmel_fire_alarm_remote",
        "title": "Kimmel Fire Alarm Remote",
        "premise": "A late-night desk is fitted with a corporate fire alarm that rings every time a punchline mentions the first family.",
        "source_ids": ["ap_kimmel_fire_calls_2026_04_27"],
        "scores": {"freshness": 13, "famous_signal": 17, "visual_contradiction": 17, "conflict": 18, "object_comedy": 14, "taste_safety": 12},
        "total": 91,
        "risk": "Strong but duplicate: run_032 and adjacent packages already cover Kimmel/ABC firing mechanics.",
        "decision": "rejected_duplicate_lane",
    },
    {
        "id": "qatar_747_lease_counter",
        "title": "Qatar 747 Lease Counter",
        "premise": "A luxury jet sits at an airport rental counter while a clerk asks whether the presidential library wants damage coverage.",
        "source_ids": ["ap_qatar_air_force_one_2026_05_02"],
        "scores": {"freshness": 20, "famous_signal": 16, "visual_contradiction": 18, "conflict": 17, "object_comedy": 18, "taste_safety": 13},
        "total": 102,
        "risk": "Rejected despite score because run_041 already used the gift-shop version of this exact source.",
        "decision": "rejected_duplicate_source",
    },
]

FIRST_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 9:16 vertical.
subject: a fictional anonymous bronze suited municipal statue, not a real person, not an exact Banksy work, standing at a tiny street optometry counter in central London.
scene: the statue's generic flag has blown across its face like a blindfold while a deadpan optometrist points to an eye chart that only shows simple shapes, not readable political text. The plinth has a small brass tag reading PUBLIC OBJECT, not a logo.
visual contradiction: heroic monument posture versus mundane eye exam; civic grandeur collapsed into a practical vision appointment.
environment: rainy London traffic island near grand stone buildings, cones, clipboard, portable eye chart, tourists blurred in the background.
camera: cinematic 24mm low-angle street-photo frame, slight handheld imperfection, first-frame hook, centered on the flag-covered face and eye chart.
lighting and mood: overcast daylight, wet pavement reflections, dry British absurdity, satirical but not mean.
style: premium SGFLIX cinematic still, realistic public-art texture, subtle film grain, no fake news graphics.
avoid next - exact Banksy sculpture copy, real Union Jack design specificity, party logos, readable slogans, official insignia, real political figures, watermarks, captions, messy text, distorted faces, excessive yellow in the photo."""

BOARD_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 16:9.
subject: Shared Choices director bible board for an SGFLIX satire called Banksy Flag Optometry.
layout: one clean production-design board with six labeled visual zones, but keep text minimal and legible.
include: character canon for anonymous bronze suited statue and deadpan optometrist; hero props including generic flag blindfold, eye chart with shape symbols, plinth tag, clipboard, traffic cones; color palette of rain-gray stone, bronze, muted red, off-white paper, black umbrella accents; environment set design for central London traffic island; floor plan/blocking diagram showing statue, optometry counter, tourists, camera path; three storyboard panels with camera/lens/movement notes; lighting/mood/style notes and production guardrails.
camera/story panels: panel 1 low angle reveals flag-covered statue stepping off plinth; panel 2 optometrist holds eye chart in the rain; panel 3 close-up of flag edge, eye chart shapes, and brass PUBLIC OBJECT tag.
style: cinematic director's bible, polished pitch-board, realistic mixed-media production art, no real logos, no exact Banksy replica.
avoid next - dense unreadable text, real political slogans, real flags copied exactly, official insignia, fake news graphics, watermarks, distorted faces, excessive yellow in the photo."""


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
        ROOT / "sgflix_runs" / "run_068_kim_128k_fee_counter" / "NEXT_STEP_REPORT_2026-05-02_RUN_069_CYCLE.md",
        """# Next-Step Report - Run 064

Observed highest previous official run: `run_068_kim_128k_fee_counter`.

Status: incomplete or empty package shell. No package files were present within three directory levels during the run 069 factory check.

Required next human action: recover or regenerate the run 068 package from its original research winner, or mark it aborted if no source artifacts can be recovered. This report does not replace the new research-first run 069 cycle.""",
    )

    write_json(PKG / "research/sources.json", SOURCES)
    write(
        PKG / "research/last30days_report.md",
        f"""# Step 1: Research Intake - Run {RUN_ID}

Created: {NOW}

Research query/topic: current late-April/May 2026 entertainment, art, politics, and celebrity controversy scan with explicit duplicate filtering against prior SGFLIX runs.

Fresh source context: AP reported on April 30, 2026 that Banksy confirmed a central London sculpture of a suited man whose face is covered by a billowing flag. The sculpture appeared on a plinth at Waterloo Place, near Buckingham Palace, and drew public attention before Banksy confirmed it.

Other scanned context: AP Britney Spears DUI charge coverage, AP Ye/Wireless sponsor fallout, AP Kimmel firing-call coverage, AP Qatar 747/Air Force One update, and Bloomberg/Reuters-style Paramount-WBD merger litigation coverage. Kimmel, Paramount, Qatar, Ye, and Taylor/AI trademark territory were rejected as duplicate or near-duplicate lanes in this workspace.

Winner selected after candidate scoring: `banksy_flag_optometry`.

Verified facts only: a Banksy-confirmed statue appeared in central London; it shows a suited man with his face covered by a flag; it was placed on/near a plinth at Waterloo Place; Banksy posted installation material. All intended political meanings are treated as interpretation, not fact.
""",
    )

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "created_at": NOW, "candidates": CANDIDATES, "winner": "banksy_flag_optometry"})
    write(
        PKG / "strategy/winner_decision.md",
        """# Winner Decision

Selected premise: **Banksy Flag Optometry**.

Score summary: Banksy Flag Optometry 100, Qatar 747 Lease Counter 102 but rejected for duplicate source, Kimmel Fire Alarm Remote 91 but rejected for duplicate lane, Britney Wet Reckless Counter 86 but rejected for taste risk, Ye Sponsor Exit Luggage 83 but rejected for duplicate/risk.

Why this wins: it is fresh, visually immediate, low-defamation, and object-driven. The joke is not that the public-art message is wrong; the joke is that civic symbolism has to book an eye exam because its own heroic prop blocks its vision.

Guardrails: fictionalize the statue, do not copy the exact Banksy sculpture, avoid real flags/logos/official insignia, and do not assign a specific political meaning beyond the verified visual setup.
""",
    )

    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"run_id": RUN_ID, "verdict": "worthy", "why": ["strong first-frame contradiction", "famous artist signal without needing a face", "object comedy carries the bit", "lower personal-harm risk than celebrity legal/addiction stories"], "kill_conditions": ["exact Banksy replica", "partisan slogan text", "looks like a real government ad"]})
    write_json(PKG / "strategy/source_entropy_audit.json", {"run_id": RUN_ID, "source_entropy": "medium_high", "source_types": ["primary wire reporting", "art/public-space event", "public reaction scan", "duplicate workspace scan"], "duplicate_lanes_rejected": ["Kimmel/ABC", "Paramount-WBD", "Qatar 747 gift", "Ye/Wireless", "Taylor AI trademark"]})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"run_id": RUN_ID, "setup": "A public-art statue is already visually blinded by its own flag.", "bridge": "Treat the symbolic blindness as a literal optometry problem.", "payoff": "The monument tries to pass an eye exam while remaining heroic.", "target": "symbolic overconfidence and institutional solemnity", "not_target": ["nationality", "veterans", "specific voters", "real medical blindness"]})
    write_json(PKG / "strategy/tribe_meta_score.json", {"run_id": RUN_ID, "overall": 88, "tribe": {"art_world": 18, "politics_watchers": 17, "british_absurdism": 18, "shortform_visual_hook": 20, "repeatability": 15}, "notes": "Works for viewers who recognize Banksy, but still reads without knowing the exact source."})
    write_json(PKG / "strategy/risk_taste_score.json", {"run_id": RUN_ID, "overall_risk": "medium_low", "risk_score": 24, "risks": [{"issue": "copying exact living artist artwork", "mitigation": "fictional municipal statue, generic flag, no exact title or replica"}, {"issue": "partisan misread", "mitigation": "no slogans, no real politicians, no official insignia"}, {"issue": "messy generated text", "mitigation": "use symbol eye chart and minimal PUBLIC OBJECT tag"}]})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nDecision: `single_with_franchise_potential`.\n\nThis can become a recurring SGFLIX public-object help-desk lane: monuments at customer service counters, symbolic props treated as mundane repair tickets, and civic grandeur colliding with retail procedure.")

    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT)
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)
    first = generate_image(FIRST_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board = generate_image(BOARD_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x864")

    image_ok = first["ok"] and board["ok"]
    status = "complete_still_package_no_video_generated" if image_ok else "blocked_at_gpt_image_generation"

    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot": "001", "duration_seconds": 6, "subject": "fictional bronze suited municipal statue with generic flag blindfold", "scene": "rainy London traffic island converted into a pop-up optometry counter", "motion": "manual-only future push from plinth to eye chart to flag-covered face", "spatial": "plinth foreground, eye chart right, tourists and stone buildings behind", "camera": "24mm low-angle handheld street-cinema", "critique": "must read as fictional public-art satire, not a fake Banksy copy", "revision": "remove slogans, official insignia, exact flag design, and dense text", "source_frame": "frames/gpt_image_2/first_frame_v01.png"}]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "status": "handoff_only_no_video_generation", "source_image": "frames/gpt_image_2/first_frame_v01.png", "prompt": "Manual-only future clip: slow push from wet plinth to eye chart shapes, then rack focus to the generic flag covering the statue's face. No video generation in this automation.", "negative": "exact Banksy replica, real politicians, official insignia, party slogans, messy text"}
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "video_generation_tools_called": False, "manual_only": True, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "guardrails": ["fictional statue", "generic flag", "no exact Banksy copy", "no official insignia", "no real politicians"]})
    write(PKG / "handoffs/grok_agent_prompt.md", "# Grok/Closed-Tool Prompt\n\nUse the approved first frame as a visual anchor only after human still approval. Create no video in this automation. Preserve fictional statue, generic flag, rainy London optometry-counter absurdity, and remove any real logos or slogans.")
    write(PKG / "captions/instagram_caption.md", "A monument walked into an eye exam because the symbol was doing too much.\n\nVerified source context: AP reported Banksy confirmed a London statue of a suited man blinded by a flag. This SGFLIX package fictionalizes the setup into a public-object optometry desk.\n\n#sgflix #satire #banksy #publicart #london #visualcomedy #aivideo")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nSurface: Instagram Reels/TikTok after human still approval.\n\nHook: `When the statue needs an eye exam before the speech.`\n\nDo not post automatically. Do not claim Banksy endorsed this package. Do not use exact sculpture replication in public exports.")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nResearch-first selection avoided nearby existing boards and duplicate celebrity lanes. The winning pattern is symbolic-literal inversion: take a current public-art image already built on metaphor and force it through a mundane service counter.")

    assets = [
        {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "first_frame", **first, "exists": (PKG / "frames/gpt_image_2/first_frame_v01.png").exists()},
        {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True},
        {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "shared_choices_board", **board, "exists": (PKG / "storyboards/shared_choices/shared_choices_v01.png").exists()},
        {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True},
    ]
    required = [
        "README.md", f"RUN_{RUN_ID}_MASTER_PACKAGE.json", "research/last30days_report.md", "research/sources.json", "strategy/candidate_board.json", "strategy/winner_decision.md", "strategy/phase_minus_one_worthiness_audit.json", "strategy/source_entropy_audit.json", "strategy/humor_logic_bridge.json", "strategy/tribe_meta_score.json", "strategy/risk_taste_score.json", "strategy/franchise_decision.md", "chai/chai_shot_specs.json", "scene_json/shot_0001.json", "scene_json/shot_001.json", "handoffs/closed_tool_handoff.json", "handoffs/grok_agent_prompt.md", "captions/instagram_caption.md", "distribution/post_plan.md", "skool/case_study.md", "manifests/asset_manifest.json", "FACTORY_RUN_STATUS.md", "frames/gpt_image_2/first_frame_v01.png", "frames/gpt_image_2/first_frame_v01_prompt.md", "storyboards/shared_choices/shared_choices_v01.png", "storyboards/shared_choices/shared_choices_v01_prompt.md", "qc/first_frame_v01_qc.md", "qc/shared_choices_v01_qc.md",
    ]
    missing = [rel for rel in required if not (PKG / rel).exists() and rel not in ["README.md", f"RUN_{RUN_ID}_MASTER_PACKAGE.json", "manifests/asset_manifest.json", "FACTORY_RUN_STATUS.md", "qc/first_frame_v01_qc.md", "qc/shared_choices_v01_qc.md"]]

    if not image_ok:
        write(PKG / "qc/IMAGE_GENERATION_BLOCKED_REPORT.md", f"# Image Generation Blocked Report\n\nFirst frame mode: `{first['mode']}`\n\nShared Choices mode: `{board['mode']}`\n\nThe package is blocked until both saved prompts are run successfully through GPT Image 2 and the resulting PNGs are inspected.")

    write(PKG / "qc/first_frame_v01_qc.md", f"# First Frame QC\n\nStatus: `{'USABLE_FOR_INTERNAL_REVIEW' if first['ok'] else 'BLOCKED_IMAGE_GENERATION'}`\n\nGeneration mode: `{first['mode']}`\n\nPass checks: fictional statue required; generic flag; no real politician; no official insignia; no exact Banksy replica; no video generated.\n\nHuman watch item: inspect the PNG for accidental exact flag/slogan text and repair before public export.")
    write(PKG / "qc/shared_choices_v01_qc.md", f"# Shared Choices QC\n\nStatus: `{'USABLE_FOR_INTERNAL_REVIEW' if board['ok'] else 'BLOCKED_IMAGE_GENERATION'}`\n\nGeneration mode: `{board['mode']}`\n\nRequired board contents: character, props, palette, environment, floor plan, storyboard panels, lighting/style rules, and production notes.\n\nHuman watch item: reject if text is unreadable or if the board copies the real Banksy sculpture too closely.")
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "status": status, "assets": assets, "missing_files": missing, "post_ready_exports": [], "video_generation_tools_called": False})
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "status": status, "research_topic": "Banksy central London flag-blinded statue current-source scan", "selected_premise": CANDIDATES[0], "candidate_count": len(CANDIDATES), "sources": SOURCES, "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png" if first["ok"] else None, "shared_choices": "storyboards/shared_choices/shared_choices_v01.png" if board["ok"] else None}, "video_generation_tools_called": False})
    write(PKG / "README.md", f"# RUN {RUN_ID} Master Package - Banksy Flag Optometry\n\nStatus: `{status}`.\n\nSelected from fresh research on {NOW}.\n\nPremise: a fictional public-art statue whose generic flag blocks its face is forced into a pop-up optometry exam on a rainy London traffic island.\n\nGenerated stills:\n- `frames/gpt_image_2/first_frame_v01.png` ({first['mode']})\n- `storyboards/shared_choices/shared_choices_v01.png` ({board['mode']})\n\nNo video footage was generated.")
    write(PKG / "FACTORY_RUN_STATUS.md", f"# Factory Run Status - Run {RUN_ID}\n\nStatus: `{status}`\n\nResearch query/topic: Banksy central London flag-blinded statue current-source scan.\n\nCandidate board: 5 candidates scored before selection.\n\nSelected premise: Banksy Flag Optometry.\n\nScore summary: Banksy 100 selected; Qatar 102 duplicate rejected; Kimmel 91 duplicate rejected; Britney 86 taste-risk rejected; Ye 83 duplicate/risk rejected.\n\nGenerated still-image paths:\n- `{PKG / 'frames/gpt_image_2/first_frame_v01.png'}` ({first['mode']})\n- `{PKG / 'storyboards/shared_choices/shared_choices_v01.png'}` ({board['mode']})\n\nMissing files: {missing or 'none observed after script write'}\n\nPost-ready exports: none; stills require human review before public use.\n\nQC failures/high-risk issues: {'none from API call; human should inspect text/flag specificity/exact-artwork similarity' if image_ok else 'image generation blocked; see qc/IMAGE_GENERATION_BLOCKED_REPORT.md'}.\n\nExact next human action: open the first frame and Shared Choices board, reject if they copy the Banksy sculpture or contain slogans/messy text, otherwise approve for later manual video handoff.")
    print(json.dumps({"run_dir": str(RUN_DIR), "package": str(PKG), "status": status, "first": first, "board": board}, indent=2))


if __name__ == "__main__":
    main()
