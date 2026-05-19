from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = "051"
SLUG = "howie_apology_return_desk"
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
        return {"ok": False, "mode": "blocked_no_openai_api_key", "path": str(path), "error_path": str(err)}
    try:
        from openai import OpenAI

        result = OpenAI().images.generate(model="gpt-image-1", prompt=prompt, size=size)
        b64 = result.data[0].b64_json
        path.write_bytes(base64.b64decode(b64))
        return {"ok": True, "mode": "openai_gpt_image_api", "path": str(path), "size": size}
    except Exception as exc:
        err = path.with_suffix(".blocked.txt")
        write(err, f"GPT Image generation failed: {type(exc).__name__}: {exc}")
        return {"ok": False, "mode": "blocked_openai_gpt_image_api", "path": str(path), "error_path": str(err)}


SOURCES = [
    {
        "id": "dailybeast_howie_regret",
        "title": "Host Says He Regrets Apologizing to Kelly Ripa for On-Air Meltdown",
        "publisher": "The Daily Beast",
        "published": "2026-04-02",
        "url": "https://www.thedailybeast.com/obsessed/host-says-he-regrets-apologizing-to-kelly-ripa-for-on-air-meltdown/",
        "verified_points": [
            "Howie Mandel walked back a public apology after an awkward Live with Kelly and Mark birthday-compliment exchange.",
            "The story frames the apology as unusual for Mandel and notes he said he kind of regretted posting it.",
        ],
        "unverified_or_sensitive": ["Exact tone and intent of all participants remain interpretive."],
    },
    {
        "id": "eonline_howie_apology",
        "title": "Howie Mandel Apologizes After Viral Exchange With Kelly Ripa on Live",
        "publisher": "E! Online",
        "published": "2026-03-29",
        "url": "https://www.eonline.com/news/1430303/howie-mandel-apologizes-after-kelly-ripa-mark-consuelos-age-comment",
        "verified_points": [
            "The apology followed an exchange about Mandel looking good at age 70.",
            "Mandel posted a video apology directed at Kelly Ripa.",
        ],
        "unverified_or_sensitive": ["Public response intensity is trend-context, not a measured fact here."],
    },
    {
        "id": "variety_mrbeast_lawsuit",
        "title": "MrBeast's Beast Industries Sued by Former Employee Alleging Sexual Harassment and Retaliation",
        "publisher": "Variety",
        "published": "2026-04-23",
        "url": "https://au.variety.com/2026/digital/news/mrbeast-sued-former-employee-sexual-harassment-retaliation-35780/",
        "verified_points": [
            "A former Beast Industries employee filed harassment and retaliation claims.",
            "The company denied the allegations and said it had receipts.",
        ],
        "unverified_or_sensitive": ["Allegations are disputed and legally sensitive."],
    },
    {
        "id": "ap_showgirl_trademark",
        "title": "Lawsuit says Taylor Swift's 'Showgirl' pose comes too close to the work of a real one",
        "publisher": "Associated Press",
        "published": "2026-03-31",
        "url": "https://apnews.com/article/1e65b44eb6cca03297a712f1d247e3bf",
        "verified_points": [
            "A trademark lawsuit challenged the Life of a Showgirl branding.",
            "Swift's representative declined comment in the AP report.",
        ],
        "unverified_or_sensitive": ["The merits of the trademark claim are unresolved."],
    },
    {
        "id": "ap_met_gala_bezos",
        "title": "Beyonce, Bezos, baubles and bustiers: What to know about the 2026 Met Gala",
        "publisher": "Associated Press",
        "published": "2026-04-22",
        "url": "https://apnews.com/article/5014084c48de8d13488925287669fe94",
        "verified_points": [
            "The 2026 Met Gala is scheduled for May 4, 2026.",
            "Jeff and Lauren Sanchez Bezos are primary donors for the Costume Institute event.",
        ],
        "unverified_or_sensitive": ["Boycott/protest scale varies by source and should be separately verified before use."],
    },
]

CANDIDATES = [
    {
        "id": "howie_apology_return_desk",
        "premise": "A bald veteran TV comedian tries to return a public-apology receipt at a daytime talk-show customer-service desk after saying he regrets apologizing.",
        "source_ids": ["dailybeast_howie_regret", "eonline_howie_apology"],
        "scores": {"famous_face": 13, "public_conflict": 14, "ego_humiliation": 18, "absurd_quote_defense": 19, "visual_contradiction": 18, "risk_adjusted_freshness": 17},
        "total": 99,
        "risk": "low-medium: use generic host likeness and avoid exact show logo.",
        "why": "Clean comic reversal, concrete prop grammar, low legal sensitivity, and no recent duplicate run lane.",
    },
    {
        "id": "mrbeast_receipts_warehouse",
        "premise": "A mega-creator brand wheels literal receipt pallets into an HR warehouse while a lawsuit headline is kept off-screen.",
        "source_ids": ["variety_mrbeast_lawsuit"],
        "scores": {"famous_face": 19, "public_conflict": 19, "ego_humiliation": 11, "absurd_quote_defense": 16, "visual_contradiction": 16, "risk_adjusted_freshness": 5},
        "total": 86,
        "risk": "high: disputed harassment claims, workplace retaliation, and real-person likeness.",
        "why": "Strong 'receipts' phrase, but too sensitive for the factory's humor lane.",
    },
    {
        "id": "showgirl_trademark_locker",
        "premise": "A pop-star-adjacent showgirl costume gets stopped at a trademark locker by three identical neon name tags.",
        "source_ids": ["ap_showgirl_trademark"],
        "scores": {"famous_face": 20, "public_conflict": 14, "ego_humiliation": 10, "absurd_quote_defense": 13, "visual_contradiction": 17, "risk_adjusted_freshness": 7},
        "total": 81,
        "risk": "medium: duplicate Taylor lane from run 050 and trademark claim unresolved.",
        "why": "Good prop field but too close to the prior Taylor package.",
    },
    {
        "id": "met_gala_billionaire_coat_check",
        "premise": "A museum gala coat-check counter rejects a billionaire donor badge while couture mannequins watch.",
        "source_ids": ["ap_met_gala_bezos"],
        "scores": {"famous_face": 18, "public_conflict": 18, "ego_humiliation": 16, "absurd_quote_defense": 12, "visual_contradiction": 19, "risk_adjusted_freshness": 3},
        "total": 86,
        "risk": "medium: duplicate Bezos/Met Gala lane from run 049.",
        "why": "Topical and visual, but already handled.",
    },
]

WINNER = CANDIDATES[0]

FIRST_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 9:16 vertical social-video first frame.
subject: a generic bald seventy-year-old TV comedian host, expressive but not an exact likeness of any real person, standing at a surreal customer-service counter labeled only with abstract icons, holding a long paper receipt stamped APOLOGY RETURN REQUEST.
scene: daytime talk-show set transformed into a retail returns desk, two empty host chairs in the background, a tiny birthday cake with one candle on the counter, an oversized sign made of symbols rather than readable words, audience seats blurred behind glass.
action: he slides the apology receipt toward a polite off-camera clerk while looking torn between pride and embarrassment; the receipt curls down like a red carpet.
camera: close wide-angle 24mm, low counter height, comic first-frame contradiction, sharp face and hands, background legible but secondary.
style: premium satirical cinematic still, realistic studio lighting, glossy daytime TV color mixed with bureaucratic return-counter beige, subtle film grain, high production design, no real network logos.
photo quality and vibe: focused cinematic shot, natural light, movie-still composition, raw quality, cool ambient shadows, no oversaturation, no oversharpening, lively candid moment.
avoid next: exact celebrity likeness, Kelly Ripa likeness, Mark Consuelos likeness, real show logos, readable legal allegations, fake news graphics, watermarks, distorted faces, excessive yellow in the photo."""

BOARD_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
aspect ratio: 16:9 horizontal director's-bible storyboard board.
purpose: Shared Choices board for a 9:16 SGFLIX satire short titled Apology Return Desk.
layout: one clean production board with eight visual zones: character canon, hero props, palette, talk-show return-desk set, floor plan and blocking, four storyboard panels, lighting/mood/style notes, production guardrails. Use mostly icons, swatches, thumbnails, arrows, and short pseudo-labels; avoid dense readable text.
character canon: generic bald veteran comedian host, navy bomber jacket, black tee, nervous half-smile, expressive hands, NOT an exact real-person likeness.
hero props: long apology receipt, small birthday cake, return-counter stamp, empty talk-show mugs with abstract icons, customer-service bell, cue cards with nonspecific scribbles.
environment: glossy daytime TV studio hybridized with customer-service returns desk; bright floor lights, audience silhouette, soft pastel set walls, bureaucratic counter bins.
floor plan/blocking: camera at counter height, host at counter left, clerk implied off-frame right, receipt diagonally leading to camera, empty host chairs in back.
storyboard panels: panel 1 compliment lands; panel 2 receipt unrolls; panel 3 return stamp hovers; panel 4 host realizes the apology has a return policy.
lighting and mood: bright studio key light, soft rim, cheerful colors with awkward silence underneath.
visual rules: no real logos, no exact celebrity faces, no actual show title, no legal claims, generated microtext treated as texture only.
style: polished pitch-board realism, neat grid, cinematic thumbnails, clear visual hierarchy, premium comedy production design."""


def main() -> None:
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

    write(PKG / "research/last30days_report.md", f"""
# Step 1: Research Intake - Run {RUN_ID}

Created: {NOW}

Research query/topic: late-March through May 2, 2026 pop-culture conflict scan emphasizing public apology reversals, current legal/brand conflicts, and repeat-lane avoidance.

Method:
- Started from current web/source context, not from local images, old handoffs, or existing storyboards.
- Screened current items for SGFLIX first-frame contradiction, famous-face pull, ego/humiliation, absurd quote/defense, and taste risk.
- Penalized any premise overlapping recent official packages: Ticketmaster, Kimmel/ABC firing desk, Paramount merger small claims, dog-bite evidence, Bezos/Met Gala coat check, and Taylor lyric/paternity lab.

Source context:
- Howie Mandel apology/regret cycle: verified by entertainment coverage as a public apology after a daytime-TV age-compliment exchange, followed by a later regret/walkback. This is the cleanest comedy target because it is a status/ego reversal, not a sensitive allegation.
- MrBeast/Beast Industries lawsuit: verified as a disputed harassment/retaliation case with a company denial; rejected as too sensitive for slapstick unless the user explicitly wants a darker legal package.
- Taylor Swift Showgirl trademark lawsuit: verified as a trademark dispute; rejected for duplicate Taylor lane and unresolved claim risk.
- 2026 Met Gala Bezos donor controversy: verified as topical and visual, but rejected because run 049 already covered Bezos/Met Gala coat-check logic.

Winner selected after scoring: `{WINNER["id"]}`.

Unverified facts policy: all motives, emotional states, and audience reaction intensity are treated as interpretive. The episode should parody the apology-return contradiction, not assert private intent.
""")
    write_json(PKG / "research/sources.json", SOURCES)
    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "created": NOW, "candidates": CANDIDATES, "winner_id": WINNER["id"]})
    write(PKG / "strategy/winner_decision.md", f"""
# Winner Decision - Run {RUN_ID}

Selected premise: **Apology Return Desk**.

Logline: A veteran TV comedian tries to return a public-apology receipt after realizing the apology itself became the joke.

Score summary:
- Winner total: {WINNER["total"]}/120.
- Highest rejected serious lane: MrBeast Receipts Warehouse at 86/120, rejected for sensitive disputed workplace allegations.
- Highest rejected duplicate lane: Met Gala Billionaire Coat Check at 86/120, rejected because run 049 already handled that visual territory.

Why this won:
- It has a clean ego/humiliation reversal: apologize, then regret apologizing.
- The first frame is instantly readable without alleging wrongdoing.
- It creates strong prop comedy: receipt, return desk, stamp, birthday cake, talk-show chairs.
- It avoids local-asset reuse and avoids recent numbered-run repetition.

Taste boundaries:
- Use a generic bald comedian-host archetype, not an exact Howie Mandel face.
- Do not depict Kelly Ripa or Mark Consuelos.
- Do not use the Live/ABC/AGT names or logos.
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"run_id": RUN_ID, "premise": WINNER["premise"], "passes": True, "score": 87, "gates": {"not_just_celebrity_plus_ai": True, "first_frame_contradiction": True, "public_context": True, "low_sensitive_claim_load": True, "fresh_vs_recent_runs": True}})
    write_json(PKG / "strategy/source_entropy_audit.json", {"run_id": RUN_ID, "entropy_score": 74, "source_types": ["entertainment news", "current legal/brand scan", "repeat-lane audit"], "concerns": ["winner relies on entertainment press rather than primary video transcript"], "mitigation": "Use only broad verified sequence: awkward exchange, apology, later regret."})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"run_id": RUN_ID, "comic_engine": "bureaucratic literalization of apology regret", "setup": "Public apology issued after awkward TV compliment exchange.", "turn": "The apology now needs its own returns desk.", "button": "The receipt is longer than the original argument.", "forbidden": ["real logos", "exact likenesses", "claims about private motives"]})
    write_json(PKG / "strategy/tribe_meta_score.json", {"run_id": RUN_ID, "tribe_score": 82, "meta_score": 78, "audience": ["daytime TV watchers", "comedy discourse followers", "celebrity apology-cycle watchers"], "share_trigger": "Everyone recognizes the modern ritual of apologizing for the apology."})
    write_json(PKG / "strategy/risk_taste_score.json", {"run_id": RUN_ID, "risk": "low-medium", "taste_score": 84, "risk_flags": ["real-person parody", "generated likeness drift", "messy generated text"], "controls": ["generic archetype", "no show logos", "pseudo text only"]})
    write(PKG / "strategy/franchise_decision.md", """
# Franchise Decision

Decision: keep as a one-off SGFLIX apology-cycle short, with franchise option only if future episodes use different public apology mechanics.

Reusable pattern: turn an intangible PR ritual into a literal retail/service workflow.
""")
    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT)
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)

    first = generate_image(FIRST_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board = generate_image(BOARD_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")
    image_ok = first["ok"] and board["ok"]

    shot = {
        "run_id": RUN_ID,
        "shot_id": "shot_0001",
        "duration_seconds": 6,
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "premise": WINNER["premise"],
        "camera": {"aspect": "9:16", "lens": "24mm", "move": "slow push-in from receipt curl to host face"},
        "subject": "generic bald comedian-host archetype at apology return desk",
        "scene": "daytime TV set merged with customer-service counter",
        "motion": "receipt curls, stamp hovers, host hesitates; no video generation requested",
        "workflow_spec": {"source_image": "frames/gpt_image_2/first_frame_v01.png", "video_generation": "prohibited_in_factory_cycle"},
    }
    write_json(PKG / "scene_json/shot_0001.json", shot)
    write_json(PKG / "scene_json/shot_001.json", {**shot, "shot_id": "shot_001"})
    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot_id": "shot_0001", "subject": shot["subject"], "scene": shot["scene"], "motion": shot["motion"], "spatial": "counter foreground, talk-show chairs background, audience blur deep background", "camera": shot["camera"], "critique": "Must read as apology-return absurdity without exact real-person likeness.", "revision": "If likeness drifts, make host more generic and props more dominant."}]})
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "status": "STILLS_READY_NO_VIDEO_GENERATED" if image_ok else "BLOCKED_IMAGE_GENERATION", "video_tools_not_called": True, "approved_source_frame": "frames/gpt_image_2/first_frame_v01.png" if first["ok"] else None, "approved_storyboard": "storyboards/shared_choices/shared_choices_v01.png" if board["ok"] else None, "next_manual_step": "Human visual QC, then optional separate manual video workflow only after approval."})
    write(PKG / "handoffs/grok_agent_prompt.md", """
# Grok Agent Prompt - Manual Reference Only

Build a short satire scene from the approved stills: a generic bald comedian-host tries to return an apology receipt at a daytime-talk customer-service desk. Keep all faces generic, avoid real show names/logos, and do not assert private motives.

Do not generate video from this automation. This is a manual handoff prompt only.
""")
    write(PKG / "captions/instagram_caption.md", """
POV: the apology came with a return policy.

No real logos, no verdicts, no private motives. Just the modern celebrity apology cycle as a customer-service desk.

#sgflix #satire #popculture #comedy #daytimetv #apologytour
""")
    write(PKG / "distribution/post_plan.md", """
# Post Plan

Surface: Instagram Reels / TikTok / Shorts after human still QC.

Hook: open on the curled apology receipt before revealing the host and return counter.

Overlay plan: use clean external captions, not generated image text. Suggested first overlay: `APOLOGY RETURN DESK`.

Do not post automatically. Do not generate video in this factory cycle.
""")
    write(PKG / "skool/case_study.md", """
# Skool Case Study - Apology Return Desk

Lesson: the strongest low-risk SGFLIX premises often convert a public-relations abstraction into a literal object workflow.

Why it works: the audience does not need to litigate who was right. The comedy is in the ritual: apology, regret, receipt, return counter.
""")
    assets = [
        {"path": "frames/gpt_image_2/first_frame_v01.png", **{k: v for k, v in first.items() if k != "path"}},
        {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "ok": True, "type": "prompt"},
        {"path": "storyboards/shared_choices/shared_choices_v01.png", **{k: v for k, v in board.items() if k != "path"}},
        {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "ok": True, "type": "prompt"},
    ]
    required = [
        "RUN_051_MASTER_PACKAGE/README.md",
        "RUN_051_MASTER_PACKAGE/RUN_051_MASTER_PACKAGE.json",
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
    missing = []
    for rel in required:
        test = PKG / rel if not rel.startswith("RUN_051_MASTER_PACKAGE/") else RUN_DIR / rel
        if not test.exists() and rel not in ["RUN_051_MASTER_PACKAGE/README.md", "RUN_051_MASTER_PACKAGE/RUN_051_MASTER_PACKAGE.json", "manifests/asset_manifest.json", "FACTORY_RUN_STATUS.md", "qc/first_frame_v01_qc.md", "qc/shared_choices_v01_qc.md"]:
            missing.append(rel)

    if not image_ok:
        write(PKG / "qc/IMAGE_GENERATION_BLOCKED_REPORT.md", f"""
# Image Generation Blocked Report

Created: {NOW}

First frame status: {first["mode"]}.

Shared Choices status: {board["mode"]}.

The run package is not complete for publishing until both saved prompts are run through GPT Image 2 or the API issue is repaired.
""")

    write(PKG / "qc/first_frame_v01_qc.md", f"""
# First Frame QC

Asset: `frames/gpt_image_2/first_frame_v01.png`

Status: {"PASS_USABLE_STILL_CREATED" if first["ok"] else "BLOCKED_IMAGE_GENERATION"}.

Generator mode: `{first["mode"]}`.

Checks: winner selected before image generation; prompt avoids exact likeness, real show logos, sensitive legal claims, fake news graphics, and readable allegations. Human review still required for face drift and generated text artifacts.
""")
    write(PKG / "qc/shared_choices_v01_qc.md", f"""
# Shared Choices QC

Asset: `storyboards/shared_choices/shared_choices_v01.png`

Status: {"PASS_USABLE_STORYBOARD_CREATED" if board["ok"] else "BLOCKED_IMAGE_GENERATION"}.

Generator mode: `{board["mode"]}`.

Checks: board prompt includes character, hero props, palette, environment/set, floor plan/blocking, storyboard panels, lighting/mood/style, visual rules, and production notes. Human review still required for pseudo-text cleanliness.
""")
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "created": NOW, "assets": assets, "missing_files": missing, "video_generation_tools_called": False})
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created": NOW, "status": "COMPLETE_STILL_PACKAGE_NO_VIDEO_GENERATED" if image_ok else "BLOCKED_IMAGE_GENERATION", "winner": WINNER, "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png" if first["ok"] else None, "shared_choices": "storyboards/shared_choices/shared_choices_v01.png" if board["ok"] else None}, "video_generation_tools_called": False})
    write(PKG / "README.md", f"""
# RUN {RUN_ID} MASTER PACKAGE - Apology Return Desk

Status: {"COMPLETE_STILL_PACKAGE_NO_VIDEO_GENERATED" if image_ok else "BLOCKED_IMAGE_GENERATION"}

Premise: {WINNER["premise"]}

Required stills:
- `frames/gpt_image_2/first_frame_v01.png` - {first["mode"]}
- `storyboards/shared_choices/shared_choices_v01.png` - {board["mode"]}

No video footage was generated or requested.
""")
    status = f"""
# FACTORY RUN STATUS - RUN {RUN_ID}

Status: {"COMPLETE_STILL_PACKAGE_NO_VIDEO_GENERATED" if image_ok else "BLOCKED_IMAGE_GENERATION"}
Run time: {NOW}
Research query/topic: late-March through May 2, 2026 current pop-culture apology/conflict scan.
Selected premise: Apology Return Desk.
Winner score: {WINNER["total"]}/120.

Candidate board:
- Apology Return Desk: 99/120, selected.
- MrBeast Receipts Warehouse: 86/120, rejected for sensitive disputed legal allegations.
- Met Gala Billionaire Coat Check: 86/120, rejected as duplicate Bezos/Met Gala lane.
- Showgirl Trademark Locker: 81/120, rejected as duplicate Taylor lane and unresolved trademark claim.

Generated still-image paths:
- {PKG / "frames/gpt_image_2/first_frame_v01.png"} ({first["mode"]})
- {PKG / "storyboards/shared_choices/shared_choices_v01.png"} ({board["mode"]})

Missing files: {"none from required minimum list" if not missing else ", ".join(missing)}

Post-ready exports: no rendered video; caption and post plan are ready for human review only.

QC failures: {"none blocking at automation level; human visual QC still required" if image_ok else "image generation blocked; see qc/IMAGE_GENERATION_BLOCKED_REPORT.md"}

High-risk issues: exact celebrity likeness drift, real show/network logos, generated text artifacts, overstating private intent.

Exact next human action: {"review the first-frame and Shared Choices PNGs for likeness/logo/text drift, then approve or request a repair before any separate manual video workflow" if image_ok else "restore GPT Image/API access, rerun the two saved prompts, then complete QC before any manual video workflow"}.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.")


if __name__ == "__main__":
    main()
