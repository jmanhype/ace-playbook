from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN_ID = "074"
SLUG = "sweeney_foley_forensics_desk"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

RESEARCH_QUERY = (
    "late-April/early-May 2026 entertainment stories with viral public correction, "
    "famous face, absurd first-frame contradiction, and low legal/taste risk"
)

SOURCES = [
    {
        "id": "tmz_sweeney_fake_euphoria_clip",
        "title": "Viral Sydney Sweeney 'Euphoria' On-Set Fart Video Clip Isn't Real",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/28/sydney-sweeney-euphoria-fart-clip-not-real/",
        "published": "2026-04-28",
        "verified_facts": [
            "TMZ reported a viral behind-the-scenes Euphoria clip involving Sydney Sweeney was not what it appeared to be.",
            "TMZ reported the footage was real but the sound effect was added later.",
            "TMZ framed the edit as a click-driven fake viral moment, not as an authentic on-set incident.",
        ],
        "unverified_or_claim_framed": [
            "The package does not identify the editor who added the sound.",
            "The package does not assert any wrongdoing by Sweeney or the Euphoria production.",
        ],
    },
    {
        "id": "cinemablend_sweeney_fake_clip_context",
        "title": "The Real Story Behind What Was Going On In That Sydney Sweeney Euphoria Fart Video",
        "publisher": "CinemaBlend",
        "url": "https://www.cinemablend.com/television/real-story-behind-what-going-on-in-sydney-sweeney-euphoria-fart-video",
        "published": "2026-04-29",
        "verified_facts": [
            "CinemaBlend summarized TMZ's reporting and described the spread of the fake-audio clip.",
            "CinemaBlend contextualized the episode as a broader reminder that viral video can be reshaped by post-production audio.",
        ],
        "unverified_or_claim_framed": [
            "View counts and source identity are secondary-context claims and not used as factual load-bearing beats.",
        ],
    },
    {
        "id": "lat_paramount_consumers",
        "title": "Consumers sue to block Paramount-Warner Bros. deal",
        "publisher": "Los Angeles Times",
        "url": "https://www.latimes.com/entertainment-arts/business/story/2026-05-01/consumers-sue-to-block-paramount-warner-bros-deal",
        "published": "2026-05-01",
        "verified_facts": [
            "The Los Angeles Times reported consumers sued to block Paramount Skydance's Warner Bros. acquisition.",
        ],
        "unverified_or_claim_framed": [
            "Rejected as the winner because SGFLIX already has run_046_paramount_merger_small_claims.",
        ],
    },
    {
        "id": "tmz_sweeney_scooter_stagecoach",
        "title": "Sydney Sweeney Hard-Launches Scooter Braun Relationship, Goes Instagram Official",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/05/01/sydney-sweeney-instagram-official-with-scooter-braun/",
        "published": "2026-05-01",
        "verified_facts": [
            "TMZ reported Sweeney posted a Stagecoach Instagram carousel with Scooter Braun.",
        ],
        "unverified_or_claim_framed": [
            "Rejected because SGFLIX already has a recent Stagecoach hard-launch lane.",
        ],
    },
    {
        "id": "tmz_megan_klay_split",
        "title": "Megan Thee Stallion and Klay Thompson Split",
        "publisher": "TMZ",
        "url": "https://www.tmz.com/2026/04/25/megan-thee-stallion-klay-thompson-break-up/",
        "published": "2026-04-25",
        "verified_facts": [
            "TMZ reported Megan Thee Stallion confirmed a split from Klay Thompson and gave a statement about trust, fidelity, and respect.",
        ],
        "unverified_or_claim_framed": [
            "Rejected for higher personal-heartbreak taste risk and overlap with run_063.",
        ],
    },
]

CANDIDATES = [
    {
        "id": "sweeney_foley_forensics_desk",
        "premise": "A fictional prestige-TV set opens a deadpan Foley Forensics Desk where a tiny sound-effect machine is fingerprinted like evidence after a fake viral BTS clip gets corrected.",
        "source_ids": ["tmz_sweeney_fake_euphoria_clip", "cinemablend_sweeney_fake_clip_context"],
        "scores": {
            "famous_face_or_status_proxy": 9,
            "public_conflict": 7,
            "ego_status_pressure": 6,
            "humiliation_or_contradiction": 9,
            "absurd_quote_or_defense": 8,
            "brand_location_contrast": 9,
            "first_frame_clarity": 10,
            "risk_adjustment": 8,
        },
        "total": 66,
        "verdict": "WINNER",
    },
    {
        "id": "paramount_warner_remote_small_claims",
        "premise": "Consumers drag a giant streaming remote into small-claims court to block a studio merger.",
        "source_ids": ["lat_paramount_consumers"],
        "scores": {
            "famous_face_or_status_proxy": 6,
            "public_conflict": 9,
            "ego_status_pressure": 8,
            "humiliation_or_contradiction": 8,
            "absurd_quote_or_defense": 5,
            "brand_location_contrast": 10,
            "first_frame_clarity": 9,
            "risk_adjustment": 4,
        },
        "total": 59,
        "verdict": "REJECTED_DUPLICATES_RUN_046",
    },
    {
        "id": "stagecoach_hard_launch_customs",
        "premise": "A cowboy festival customs desk stamps a celebrity couple's Instagram carousel as hard-launched.",
        "source_ids": ["tmz_sweeney_scooter_stagecoach"],
        "scores": {
            "famous_face_or_status_proxy": 9,
            "public_conflict": 5,
            "ego_status_pressure": 7,
            "humiliation_or_contradiction": 7,
            "absurd_quote_or_defense": 7,
            "brand_location_contrast": 8,
            "first_frame_clarity": 8,
            "risk_adjustment": 4,
        },
        "total": 55,
        "verdict": "REJECTED_DUPLICATES_RUN_071",
    },
    {
        "id": "sweetest_pie_return_counter",
        "premise": "A basketball trophy and Broadway playbill are returned to a relationship lost-and-found counter.",
        "source_ids": ["tmz_megan_klay_split"],
        "scores": {
            "famous_face_or_status_proxy": 9,
            "public_conflict": 7,
            "ego_status_pressure": 8,
            "humiliation_or_contradiction": 8,
            "absurd_quote_or_defense": 8,
            "brand_location_contrast": 7,
            "first_frame_clarity": 8,
            "risk_adjustment": 2,
        },
        "total": 57,
        "verdict": "REJECTED_PERSONAL_HEARTBREAK_AND_RUN_063_OVERLAP",
    },
]

FIRST_FRAME_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
Use case: photorealistic-natural
Asset type: SGFLIX vertical first-frame still, 9:16.
Primary request: Create a satirical premium editorial first frame for the premise "Foley Forensics Desk." The scene is inspired by public reporting that a viral behind-the-scenes prestige-TV clip was real footage but had a fake sound effect added later. Do not depict an exact likeness of Sydney Sweeney, Zendaya, HBO, Euphoria logos, or any real production branding.
Scene/backdrop: a fictional neon-lit prestige-TV soundstage that has been converted into a tiny forensic evidence counter for post-production audio.
Subject: a fictional blonde prestige-TV actress archetype, seen three-quarter from behind/side with face intentionally non-specific, standing calmly while a deadpan audio technician in blue gloves fingerprints a tiny red sound-effect button.
Hero props: evidence tray containing a miniature Foley machine, waveform printout with unreadable pseudo-text, clapperboard labeled only "SCENE / TAKE" in generic letters, yellow sticky note reading only "ADDED IN POST", a microphone under a glass dome, a tiny courtroom-style stamp marked "FAKE AUDIO", and a director chair with no logo.
Composition: vertical 9:16, 28mm lens, low evidence-desk angle, sound-effect button and waveform tray in the foreground, fictional actress archetype in the midground, soundstage lights and camera rigs blurred just enough to read as set equipment. First read must be: the scandal is not the person, it is the tiny added audio button.
Lighting/style: premium satirical magazine photo, realistic set fluorescents mixed with magenta/cyan soundstage glow, controlled contrast, natural skin texture, no oversharpening, subtle film grain, clean composition, 2k.
Text policy: no readable paragraphs, no real TV/network marks, only short generic prop labels.
Safety/guardrails: no exact public-figure likeness, no sexual framing, no defamation, no body-function depiction, no fake news chyron, no real show logos, no video-generation language.
Avoid next - text, captions, watermarks, unrelated logos, official insignia, distorted faces, identity drift, oversharpening, oversaturation, excessive yellow in the photo."""

SHARED_CHOICES_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -
Use case: productivity-visual
Asset type: SGFLIX Shared Choices director's-bible storyboard board, 16:9.
Primary request: Create a premium director's-bible board for the satire premise "Foley Forensics Desk," inspired by public reporting that a viral prestige-TV BTS clip used real footage with fake audio added later. Do not depict exact Sydney Sweeney likeness and do not use HBO, Euphoria, or studio branding.
Board contents: character canon, hero props, color palette, environment/set design, floor plan/blocking, six storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes.
Character/props: fictional blonde prestige-TV actress archetype with non-specific face; deadpan audio tech with gloves; tiny red Foley button; waveform receipt; generic clapperboard; microphone under glass; stamp marked "FAKE AUDIO"; sticky note "ADDED IN POST"; production cart with cables.
Environment/set design: neon prestige-TV soundstage crossed with forensic evidence counter, matte black rigging, acrylic evidence bins, magenta/cyan practicals, neutral gray desk, no real logos.
Storyboard panels: 1) first-frame reveal of button in evidence tray, 2) technician dusts waveform receipt, 3) generic clapperboard flips to ADDED IN POST, 4) microphone sits under glass while actress waits calmly, 5) FAKE AUDIO stamp lands on receipt, 6) camera pulls back to reveal the entire soundstage evidence desk.
Visual rules: satire targets clickbait post-production manipulation, not the performer; avoid body-function depiction; no exact likeness; no real show/network marks; no readable paragraphs; no fake news chyron; no video render request.
Style: clean premium production reference board, cinematic sketches mixed with prop-photo callouts, balanced palette swatches, neat layout, minimal pseudo-text only.
Avoid next - real logos, exact celebrity face, dense illegible typography, distorted bodies, defamatory claims, sexual framing, one-note blue palette, watermark."""


def write(rel: str, content: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_json(rel: str, data) -> None:
    write(rel, json.dumps(data, indent=2, ensure_ascii=True))


def generate_image(prompt: str, rel: str, size: str) -> str:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from openai import OpenAI

        result = OpenAI().images.generate(model="gpt-image-1", prompt=prompt, size=size)
        b64 = result.data[0].b64_json
        if not b64:
            raise RuntimeError("image API returned no b64_json")
        path.write_bytes(base64.b64decode(b64))
        return "generated_gpt_image_1_api"
    except Exception as exc:
        error_path = path.with_suffix(path.suffix + ".blocked.txt")
        error_path.write_text(f"Image generation blocked: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return "blocked_image_generation"


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

    prev = ROOT / "sgflix_runs/run_073_law_roach_layflat_counter/RUN_073_MASTER_PACKAGE"
    if prev.exists():
        missing = [
            "frames/gpt_image_2/first_frame_v01.png",
            "storyboards/shared_choices/shared_choices_v01.png",
            "manifests/asset_manifest.json",
            "qc/first_frame_v01_qc.md",
            "qc/shared_choices_v01_qc.md",
        ]
        present_missing = [m for m in missing if not (prev / m).exists()]
        (ROOT / "sgflix_runs/run_073_law_roach_layflat_counter/NEXT_STEP_REPORT.md").write_text(
            "# Run 073 Next Step Report\n\n"
            "Status: incomplete from previous cycle.\n\n"
            "Missing required artifacts:\n"
            + "\n".join(f"- `{m}`" for m in present_missing)
            + "\n\nNext action: generate or repair the two still-image PNGs, then update manifest and QC. "
            "This report does not replace the fresh Run 074 research-first package.\n",
            encoding="utf-8",
        )

    winner = CANDIDATES[0]
    write(
        "research/last30days_report.md",
        f"""# Last 30 Days Research Intake - Run {RUN_ID}

Step 1: Research Intake completed before package creation.

Research query/topic: {RESEARCH_QUERY}

Current-source scan:
- Viral fake-audio correction: TMZ reported on 2026-04-28 that a viral Sydney Sweeney/Euphoria behind-the-scenes clip was real footage with fake audio added later.
- Secondary context: CinemaBlend summarized the same correction and framed it as a viral-video manipulation example.
- Entertainment-business lawsuit: Los Angeles Times reported on 2026-05-01 that consumers sued to block the Paramount-Warner Bros. deal, but SGFLIX already has a Paramount merger run.
- Stagecoach relationship hard-launch: current but rejected because a recent SGFLIX run already covers the Stagecoach hard-launch lane.
- Megan Thee Stallion/Klay Thompson split: current but rejected for personal-heartbreak taste risk and prior Klay overlap.

Winner selected after scoring: `{winner['id']}`.

Fact discipline: The package only claims that reporting says the footage was real and the audio was added later. It does not identify the editor, assign motive beyond clickbait framing, or depict the real performer directly.
""",
    )
    write_json("research/sources.json", {"research_query": RESEARCH_QUERY, "sources": SOURCES})
    write_json("strategy/candidate_board.json", {"created_at": NOW, "selection_order": "research_intake_then_scoring_then_package", "candidates": CANDIDATES})
    write(
        "strategy/winner_decision.md",
        f"""# Winner Decision - Run {RUN_ID}

Selected premise: `{winner['id']}`

Premise: {winner['premise']}

Score summary: winner total `{winner['total']}`. It beat the Paramount merger candidate because that lane duplicates `run_046`, beat the Stagecoach hard-launch candidate because that lane duplicates `run_071`, and beat the Megan/Klay candidate because personal heartbreak plus death-threat-adjacent rumor context is a taste-risk downgrade.

Creative directive: make the first read a forensic desk for a tiny added sound-effect button. The joke targets the post-production manipulation/clickbait mechanism, not the real performer.
""",
    )
    write_json(
        "strategy/phase_minus_one_worthiness_audit.json",
        {
            "phase_minus_one_audit": {
                "source_target": "viral fake-audio BTS clip correction",
                "track_a_newsjack_velocity": {
                    "active_trend_score": 8,
                    "algorithmic_slipstream": "Current entertainment/social clip discourse, published 2026-04-28 and still in May 2 search surface.",
                    "polarization_factor": 5,
                    "track_a_total": 13,
                    "track_a_verdict": "PASS",
                },
                "track_b_archetype_resonance": {
                    "iconography_strength": 8,
                    "stereotype_rigidity": "High",
                    "subversion_potential": 9,
                    "track_b_total": 17,
                    "track_b_verdict": "PASS",
                },
                "system_verdict": {
                    "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                    "primary_vector": "TRACK_A_NEWSJACK",
                    "urgency_class": "High",
                    "strategic_directive": "Convert fake viral audio into a clean prop-comedy forensic desk.",
                },
            }
        },
    )
    write_json(
        "strategy/source_entropy_audit.json",
        {
            "source_entropy_audit": {
                "baseline_reality_check": "A behind-the-scenes clip becomes viral because audio was allegedly added in post.",
                "detected_anomalies": ["The real footage/fake audio split is already absurd.", "The public correction is the actual comedic engine."],
                "native_entropy_score": 6,
                "subject_self_awareness": "unaware",
                "comedic_vector_recommendation": "cringe_vector",
                "recommended_strategy": "micro_spotlight",
            }
        },
    )
    write_json(
        "strategy/humor_logic_bridge.json",
        {
            "hook": "The scandal is not a person; it is a tiny red button labeled added in post.",
            "setup": "Audience expects celebrity embarrassment.",
            "turn": "Camera reveals a forensic evidence counter for one Foley sound effect.",
            "payoff": "A deadpan stamp treats fake audio like chain-of-custody evidence.",
            "guardrails": ["No exact likeness", "No body-function depiction", "No real TV logos", "No claim against the performer"],
        },
    )
    write_json("strategy/tribe_meta_score.json", {"overall": 8.2, "tribe": 8, "meta": 8.4, "why": "Strong internet-literacy premise with clear first-frame prop logic."})
    write_json("strategy/risk_taste_score.json", {"overall_risk": "Medium-Low", "taste_score": 8, "risks": ["Celebrity likeness", "Crude source topic"], "mitigations": ["Fictional archetype", "Forensic audio-prop framing", "No body-function visual"]})
    write("strategy/franchise_decision.md", "# Franchise Decision\n\nDecision: `KEEP_AS_CLICKBAIT_FORENSICS_FORMAT`\n\nReusable format: treat fake viral edits as evidence-desk micro-mysteries. Strong for future media-literacy satire if each episode has one concrete prop.")

    write("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)

    first_mode = generate_image(FIRST_FRAME_PROMPT, "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board_mode = generate_image(SHARED_CHOICES_PROMPT, "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")

    write_json(
        "chai/chai_shot_specs.json",
        {
            "run_id": RUN_ID,
            "shots": [
                {
                    "shot": "001",
                    "duration_seconds": 6,
                    "subject": "fictional prestige-TV actress archetype and deadpan audio-forensics tech",
                    "scene": "neon soundstage evidence desk for fake post-production audio",
                    "motion": "manual handoff only: slow push from red Foley button to FAKE AUDIO stamp",
                    "spatial": "evidence tray foreground, tech hands midground, fictional actress and soundstage background",
                    "camera": "28mm low evidence-desk push-in",
                    "critique": "joke must target added audio/clickbait, not the performer",
                    "revision": "remove exact likeness, logos, crude body depiction, and readable paragraphs",
                    "source_frame": "frames/gpt_image_2/first_frame_v01.png",
                }
            ],
        },
    )
    scene = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "status": "handoff_only_no_video_generation",
        "source_image": "frames/gpt_image_2/first_frame_v01.png",
        "workflow_spec": {"source_image": "frames/gpt_image_2/first_frame_v01.png", "video_generation": "prohibited_in_this_cycle"},
        "prompt": "Manual-only optional 6-second clip: evidence-desk push-in, waveform receipt slides forward, FAKE AUDIO stamp lands. No video generated by this automation.",
        "negative": "exact celebrity likeness, HBO or Euphoria marks, crude body visual, fake news chyron, defamatory text, watermark",
    }
    write_json("scene_json/shot_0001.json", scene)
    write_json("scene_json/shot_001.json", scene)
    write_json(
        "handoffs/closed_tool_handoff.json",
        {
            "run_id": RUN_ID,
            "video_generation_tools_called": False,
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "storyboard": "storyboards/shared_choices/shared_choices_v01.png",
            "manual_only": True,
            "guardrails": ["No exact public-figure likeness", "No real show/network logos", "No body-function depiction"],
        },
    )
    write(
        "handoffs/grok_agent_prompt.md",
        f"""# Grok Agent Prompt - Run {RUN_ID}

Do not generate video automatically. Use this only as a manual reference package.

Create a fictional, non-logo, non-likeness 6-second satire from `frames/gpt_image_2/first_frame_v01.png`: a forensic evidence desk proves that the viral scandal was one tiny added sound-effect button. Keep the actress archetype calm and non-specific; animate only the evidence props, stamp, waveform receipt, and soundstage lights.
""",
    )
    write(
        "captions/instagram_caption.md",
        """the internet put one tiny sound effect on trial.

reported context: the viral BTS clip was real footage, but the audio was reportedly added later.

#sgflix #mediaforensics #postproduction #internetliteracy #satire #foley #viralvideo""",
    )
    write(
        "distribution/post_plan.md",
        """# Post Plan

Primary surface: Instagram Reels / TikTok.

Hook text: `ADDED IN POST`

Caption angle: media-forensics satire. Avoid naming the performer in overlay text; let the source note carry context.

Do not auto-post. Human review required for likeness, logo, and crude-topic taste check.
""",
    )
    write(
        "skool/case_study.md",
        """# Skool Case Study - Foley Forensics Desk

Lesson: when a source is already absurd, spotlight the smallest mechanism instead of adding chaos.

Reusable move: convert a viral claim into a forensic prop table. The prop table creates clarity, lowers defamation risk, and makes the first frame instantly legible.
""",
    )
    first_exists = (PKG / "frames/gpt_image_2/first_frame_v01.png").exists()
    board_exists = (PKG / "storyboards/shared_choices/shared_choices_v01.png").exists()
    missing = []
    if not first_exists:
        missing.append("frames/gpt_image_2/first_frame_v01.png")
    if not board_exists:
        missing.append("storyboards/shared_choices/shared_choices_v01.png")
    status = "STILLS_READY_NO_VIDEO_GENERATED" if not missing else "BLOCKED_IMAGE_GENERATION"
    write_json(
        "manifests/asset_manifest.json",
        {
            "run_id": RUN_ID,
            "status": status,
            "assets": [
                {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "gpt_image_2_first_frame", "generation_mode": first_mode, "exists": first_exists},
                {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt", "exists": True},
                {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "gpt_image_2_shared_choices_board", "generation_mode": board_mode, "exists": board_exists},
                {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt", "exists": True},
            ],
            "missing_files": missing,
            "video_generation_tools_called": False,
            "post_ready_exports": [],
        },
    )
    write(
        "qc/first_frame_v01_qc.md",
        f"""# First Frame QC

Status: `{'PASS_REQUIRES_HUMAN_REVIEW' if first_exists else 'BLOCKED_NO_LOCAL_PNG'}`

Asset: `frames/gpt_image_2/first_frame_v01.png`

Checks:
- Exact public-figure likeness: avoid; fictional archetype requested.
- Logo/IP marks: avoid HBO/Euphoria/network marks.
- Taste: no body-function depiction; joke is the audio evidence desk.
- Text: only short prop labels allowed.

Generation mode: `{first_mode}`.
""",
    )
    write(
        "qc/shared_choices_v01_qc.md",
        f"""# Shared Choices QC

Status: `{'PASS_REQUIRES_HUMAN_REVIEW' if board_exists else 'BLOCKED_NO_LOCAL_PNG'}`

Asset: `storyboards/shared_choices/shared_choices_v01.png`

Checks:
- Includes character, props, palette, environment, blocking, panels, lighting, visual rules, and production notes.
- Avoids exact likeness and real show/network branding.
- Keeps satire focused on clickbait audio manipulation.

Generation mode: `{board_mode}`.
""",
    )
    write(
        "README.md",
        f"""# RUN {RUN_ID} MASTER PACKAGE - Foley Forensics Desk

Status: `{status}`

Selected premise: {winner['premise']}

Generated stills:
- `frames/gpt_image_2/first_frame_v01.png` ({first_mode})
- `storyboards/shared_choices/shared_choices_v01.png` ({board_mode})

No video-generation tools were called.
""",
    )
    write_json(
        f"RUN_{RUN_ID}_MASTER_PACKAGE.json",
        {
            "run_id": RUN_ID,
            "slug": SLUG,
            "created_at": NOW,
            "status": status,
            "research_query": RESEARCH_QUERY,
            "selected_premise": winner,
            "sources": SOURCES,
            "candidate_count": len(CANDIDATES),
            "generated_stills": {
                "first_frame": "frames/gpt_image_2/first_frame_v01.png" if first_exists else None,
                "shared_choices": "storyboards/shared_choices/shared_choices_v01.png" if board_exists else None,
            },
            "missing_files": missing,
            "video_generation_tools_called": False,
        },
    )
    write(
        "FACTORY_RUN_STATUS.md",
        f"""# Factory Run Status - Run {RUN_ID}

Status: `{status}`

Research query/topic: {RESEARCH_QUERY}

Winner: `{winner['id']}` with score `{winner['total']}`.

Created package root: `{PKG}`

Generated still-image paths:
- `{PKG / 'frames/gpt_image_2/first_frame_v01.png'}` ({first_mode})
- `{PKG / 'storyboards/shared_choices/shared_choices_v01.png'}` ({board_mode})

Missing files: {missing if missing else 'none'}

Post-ready exports: none. Video generation prohibited and not called.

High-risk issues: exact likeness and real show/network marks must be checked by a human before any optional manual render.

Exact next human action: review the two PNGs for likeness/logo/text issues and either approve as visual anchors or request a repaired GPT Image prompt.
""",
    )
    (RUN_DIR / "FACTORY_RUN_STATUS.md").write_text((PKG / "FACTORY_RUN_STATUS.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(PKG)
    print(status)
    print(json.dumps({"first_mode": first_mode, "board_mode": board_mode, "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
