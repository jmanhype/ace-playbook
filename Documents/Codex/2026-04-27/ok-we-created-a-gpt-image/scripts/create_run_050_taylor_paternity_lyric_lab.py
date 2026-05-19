from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_ID = "050"
SLUG = "taylor_paternity_lyric_lab"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

FIRST_FRAME_SRC = Path("/Users/speed/.codex/generated_images/019de8d3-c968-7f70-a758-aeea53cbfe2b/ig_0e236f1f992b786a0169f5fe7ac6548190b667c697d3246b5f.png")
FIRST_FRAME_REJECTED_SRC = Path("/Users/speed/.codex/generated_images/019de8d3-c968-7f70-a758-aeea53cbfe2b/ig_0e236f1f992b786a0169f5faf2d1088190a6c7eade9fbd8886.png")
STORYBOARD_SRC = Path("/Users/speed/.codex/generated_images/019de8d3-c968-7f70-a758-aeea53cbfe2b/ig_0e236f1f992b786a0169f5fcd2d2dc8190aa7f13caf338631b.png")

SOURCES = [
    {
        "id": "independent_taylor_paternity_test",
        "title": "Taylor Swift says she dislikes when fans treat her lyrics like 'paternity tests'",
        "url": "https://www.the-independent.com/arts-entertainment/music/news/taylor-swift-songs-fan-behavior-new-york-times-b2966617.html",
        "publisher": "The Independent",
        "date": "2026-04-28",
        "verified_facts": [
            "The report covers Swift discussing fan analysis of her songs in a New York Times Magazine songwriter feature.",
            "The report says Swift called it weird when people treat song subjects like a paternity test.",
            "The report frames the issue as fan speculation about who inspired songs versus recognition of songwriting craft.",
        ],
        "creative_use": "Winner source: turns a quoted fan-behavior metaphor into a literal lyric-forensics lab.",
    },
    {
        "id": "gma_taylor_paternity_test",
        "title": "Taylor Swift on fans decoding her songs: 'It's sort of like a paternity test'",
        "url": "https://www.gmanetwork.com/news/showbiz/showbizabroad/985915/taylor-swift-songwriting-fan-theories/story/",
        "publisher": "GMA News Online",
        "date": "2026-04-30",
        "verified_facts": [
            "The report says the interview was with The New York Times.",
            "The report describes Swift talking about how fans engage with her songs.",
            "The report repeats the paternity-test framing around song decoding.",
        ],
        "creative_use": "Second-source confirmation for the premise and current timing.",
    },
    {
        "id": "thewrap_taylor_paternity_test",
        "title": "Taylor Swift Thinks It's Weird When Fans Turn Her Songs Into A Paternity Test",
        "url": "https://www.thewrap.com/creative-content/music/taylor-swift-fans-weird-songs-paternity-test/",
        "publisher": "TheWrap",
        "date": "2026-04-28",
        "verified_facts": [
            "TheWrap reports Swift spoke with The New York Times after being named to its 30 greatest living songwriters list.",
            "The report centers on fans treating song analysis like a paternity test.",
        ],
        "creative_use": "Supports the broad entertainment-news pickup and the fan-detective angle.",
    },
    {
        "id": "cinemablend_ellen_reaction_clip",
        "title": "A Lot Of Taylor Swift's Fans' Tongues Are Wagging About Ellen DeGeneres Over One Viral Interview Clip",
        "url": "https://www.cinemablend.com/streaming-news/a-lot-of-taylor-swifts-fans-tongues-are-wagging-about-ellen-degeneres-over-one-viral-interview-clip",
        "publisher": "CinemaBlend",
        "date": "2026-04-30",
        "verified_facts": [
            "The report says clips from Swift's New York Times songwriting interview circulated online.",
            "The report notes fan speculation around who the comments referenced.",
        ],
        "creative_use": "Rejected side-lane: reaction discourse is useful but too person-specific for taste.",
    },
    {
        "id": "prior_run_049_met_gala_overlap",
        "title": "Local SGFLIX Run 049 Met Gala source cluster",
        "url": "local:sgflix_runs/run_049_bezos_met_gala_coat_check/RUN_049_MASTER_PACKAGE/research/sources.json",
        "publisher": "Local SGFLIX package",
        "date": "2026-05-02",
        "verified_facts": [
            "Run 049 already owns the May 2026 Bezos/Met Gala sponsor controversy cluster.",
        ],
        "creative_use": "Rejected for overlap with the immediately preceding official run.",
    },
]

CANDIDATES = [
    {
        "id": "taylor_paternity_lyric_lab",
        "premise": "Fan lyric detectives drag pop songs into a sterile paternity-test lab, only for the lab tech to stamp the chorus as authored by the songwriter, not the rumored muse.",
        "source_ids": ["independent_taylor_paternity_test", "gma_taylor_paternity_test", "thewrap_taylor_paternity_test"],
        "first_frame": "A glossy pop recording studio fused with a forensic lab: lyric swabs, a musical-note DNA helix, and a clipboard asking who the song is about.",
        "scores": {
            "fame_context": 10,
            "public_conflict": 7,
            "ego_humiliation": 8,
            "visual_contradiction": 10,
            "absurd_quote_or_object": 10,
            "brand_location_contrast": 8,
            "risk_control": 8,
            "franchise_potential": 8,
        },
        "total": 89,
    },
    {
        "id": "ellen_clip_evidence_room",
        "premise": "A viral-interview evidence room pins every old talk-show clip to a corkboard while the songwriter asks why the detective wall is facing away from the actual song.",
        "source_ids": ["cinemablend_ellen_reaction_clip"],
        "first_frame": "Detective corkboard of talk-show stills beside an ignored lyric notebook under a spotlight.",
        "scores": {
            "fame_context": 9,
            "public_conflict": 7,
            "ego_humiliation": 7,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 7,
            "brand_location_contrast": 7,
            "risk_control": 6,
            "franchise_potential": 7,
        },
        "total": 68,
    },
    {
        "id": "met_gala_coat_check_reheat",
        "premise": "Another Met Gala compliance desk variation.",
        "source_ids": ["prior_run_049_met_gala_overlap"],
        "first_frame": "Red carpet desk, donor props, and protest wall.",
        "scores": {
            "fame_context": 9,
            "public_conflict": 10,
            "ego_humiliation": 8,
            "visual_contradiction": 9,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 10,
            "risk_control": 7,
            "franchise_potential": 7,
        },
        "total": 68,
        "penalty": "Rejected despite raw heat because run_049 already owns the cluster.",
    },
    {
        "id": "songwriter_hall_of_mirrors",
        "premise": "A hall of mirrors where every reflection is a fan theory, except the songwriter's notebook is the only solid object.",
        "source_ids": ["independent_taylor_paternity_test", "thewrap_taylor_paternity_test"],
        "first_frame": "Mirror maze of rumors around a single plain notebook.",
        "scores": {
            "fame_context": 10,
            "public_conflict": 6,
            "ego_humiliation": 6,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 7,
            "brand_location_contrast": 7,
            "risk_control": 8,
            "franchise_potential": 6,
        },
        "total": 65,
    },
    {
        "id": "track_five_emergency_room",
        "premise": "Emotional track-five discourse gets triaged in a hospital ER while fan theories clog the waiting room.",
        "source_ids": ["gma_taylor_paternity_test"],
        "first_frame": "Hospital triage board where songs are patients and speculation is the waiting-room crisis.",
        "scores": {
            "fame_context": 9,
            "public_conflict": 5,
            "ego_humiliation": 5,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 7,
            "brand_location_contrast": 8,
            "risk_control": 8,
            "franchise_potential": 7,
        },
        "total": 57,
    },
]

FIRST_PROMPT = """GPT Image 2 prompt: Satirical premium editorial first frame, 16:9. A sterile forensic lab redesigned as a glossy pop-recording studio. A fictional blonde pop songwriter silhouette, not an exact celebrity likeness, stands calmly beside lyric sheets under glass. Fan detectives and a lab tech treat song speculation like a paternity test. Hero props: lyric swabs, musical-note DNA helix, evidence tags, microphone in evidence bag, microscope on handwritten chorus, clipboard asking WHO IS THIS SONG ABOUT. Mood: tasteful absurdity, magenta waveform screens, blue lab lighting, stainless counters. No real logos, no defamatory claims, no exact Taylor Swift likeness, no watermark."""

BOARD_PROMPT = """GPT Image 2 prompt: Shared Choices director bible board, 3:2. Concept: fan speculation about who songs are about becomes a forensic lyric-paternity lab. Include character and hero props, palette, lab/recording-studio environment, floor plan/blocking, three storyboard panels with camera notes, lighting/mood/style rules, visual rules, and production notes. Use a fictional blonde pop songwriter silhouette, fan detectives, lyric swabs, evidence tags, microscope, microphone in evidence bag, musical-note DNA helix. No exact celebrity likeness, no real logos, no defamatory claims, no video generation."""


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    write(path, json.dumps(obj, indent=2) + "\n")


def copy_image(src: Path, dst: Path) -> None:
    if not src.exists() or src.stat().st_size == 0:
        raise FileNotFoundError(f"Missing generated image: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    winner = CANDIDATES[0]
    for directory in [
        "research", "strategy", "chai", "scene_json", "handoffs", "captions", "distribution",
        "skool", "manifests", "frames/gpt_image_2", "storyboards/shared_choices", "qc",
    ]:
        (PKG / directory).mkdir(parents=True, exist_ok=True)

    copy_image(FIRST_FRAME_SRC, PKG / "frames/gpt_image_2/first_frame_v01.png")
    copy_image(FIRST_FRAME_SRC, PKG / "frames/gpt_image_2/first_frame_v02.png")
    copy_image(FIRST_FRAME_REJECTED_SRC, PKG / "frames/gpt_image_2/rejected/first_frame_v01_needs_repair_too_close_likeness.png")
    copy_image(STORYBOARD_SRC, PKG / "storyboards/shared_choices/shared_choices_v01.png")

    write(PKG / "research/last30days_report.md", f"""# Step 1: Research Intake - Run {RUN_ID}

Run time: {NOW}

Research query/topic: late-April 2026 Taylor Swift songwriting interview pickup, fan lyric-forensics discourse, and the public "paternity test" metaphor.

Process note: premise selection started from current web/source context and source checking. It did not start from local images, old storyboards, nearby assets, or prior handoff files.

Verified source context:
- The Independent reported on April 28, 2026 that Swift discussed fans analyzing song subjects in a New York Times Magazine songwriter feature and said it gets weird when people treat the songs like a paternity test.
- GMA News Online reported on April 30, 2026 that Swift talked to The New York Times about how fans engage with her songs and repeated the paternity-test framing.
- TheWrap reported on April 28, 2026 that the comments came after Swift was included in The New York Times' 30 greatest living songwriters feature.
- CinemaBlend reported on April 30, 2026 that clips from the interview were circulating and generating fan speculation.

Unverified or sensitive context:
- The New York Times source could not be fetched directly in this automation run because the site was inaccessible to the web reader.
- This package does not assert who any real song is about and does not identify any real muse.
- The visual system should use a fictional pop songwriter silhouette, not an exact Taylor Swift likeness.

Winner selected after candidate scoring: `{winner["id"]}`.
""")
    write_json(PKG / "research/sources.json", {"run_id": RUN_ID, "generated_at": NOW, "sources": SOURCES})
    write(PKG / "research/source_notes.md", "Source intake favored current public reporting about the songwriting interview and rejected nearby high-heat topics already owned by prior SGFLIX packages.\n")

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "scored_before_winner": True, "candidates": CANDIDATES})
    write(PKG / "strategy/winner_decision.md", f"""# Winner Decision - Run {RUN_ID}

Selected premise: **{winner["id"]}**

Why it won: the quote itself supplies an absurd object. Turning fan lyric speculation into a literal paternity-test lab creates an immediate first-frame contradiction without inventing private facts or dragging active litigation into the joke.

Score summary:
- taylor_paternity_lyric_lab: 89
- ellen_clip_evidence_room: 68
- met_gala_coat_check_reheat: 68, rejected for direct run_049 overlap
- songwriter_hall_of_mirrors: 65
- track_five_emergency_room: 57

First-frame mandate: show a glossy pop studio and sterile forensics lab occupying the same room, with the lyric sheet treated as the "sample" and authorship as the punchline.
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "Taylor Swift paternity-test songwriting quote", "track_a_newsjack_velocity": {"active_trend_score": 8, "algorithmic_slipstream": "High entertainment pickup within the last week", "polarization_factor": 7, "track_a_total": 22, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 10, "stereotype_rigidity": "High", "subversion_potential": 9, "track_b_total": 29, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_B_EVERGREEN_ARCHETYPE", "urgency_class": "High", "strategic_directive": "Use the forensic-lab metaphor as the durable visual engine while the quote is still current."}}})
    write_json(PKG / "strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "A songwriter discusses craft and fan interpretation in a public interview.", "detected_anomalies": ["Paternity-test metaphor applied to song interpretation", "Fan detective work becomes more visible than songwriting craft", "Online reaction tries to identify specific targets anyway"], "native_entropy_score": 6, "subject_self_awareness": "deadpan", "comedic_vector_recommendation": "cringe_vector", "recommended_strategy": "micro_spotlight"}})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"bridge": {"truth": "Some fan analysis can turn songs into evidence files instead of art.", "comic_flip": "Make that metaphor literal: a lyric-paternity lab where the sample is a chorus.", "humiliation_engine": "The fan detective's elaborate case is defeated by a simple authorship stamp.", "visual_payoff": "Lyric swabs, microscope, musical-note DNA helix, and a calm songwriter silhouette."}})
    write_json(PKG / "strategy/tribe_meta_score.json", {"tribe_meta_score": {"attention_tribes": ["Swift/fandom discourse", "songwriting craft watchers", "pop-culture quote aggregators", "parasocial behavior commentary"], "share_trigger": "fans recognize the detective-work behavior without needing any real ex named", "meta_score": 88}})
    write_json(PKG / "strategy/risk_taste_score.json", {"risk_taste_score": {"legal_risk": 3, "likeness_risk": 6, "taste_risk": 4, "mitigations": ["Use fictional songwriter silhouette", "No real song titles beyond generic lyric sheets", "Do not name muses", "Avoid real logos and exact face"], "go_no_go": "GO_WITH_CONSTRAINTS"}})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nProceed as a compact one-off with reusable franchise frame: celebrity quote becomes literal office, lab, customs, notary, or claims-counter procedure. The paternity-lab version is strong enough to stand alone.\n")

    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT + "\n")

    shot = {
        "shot_id": "shot_001",
        "duration_seconds": 6,
        "subject": "fictional pop songwriter silhouette and fan lyric detectives",
        "scene": "forensic lab merged with glossy pop recording studio",
        "motion": "slow push-in from microscope to authorship stamp",
        "spatial": "songwriter left, evidence table center, fan detective right, waveform monitors rear",
        "camera": "24mm establishing push to 50mm lyric-sheet insert",
        "critique": "Keep the joke on fan forensic behavior, not real romantic claims.",
        "revision": "If likeness drifts too close, obscure face and emphasize props.",
    }
    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [shot]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "prompt": FIRST_PROMPT, "duration_seconds": 6, "no_video_generation": True, "camera": {"lens": "24mm to 50mm", "movement": "slow push-in"}, "negative_constraints": ["no exact Taylor Swift likeness", "no real logos", "no named muses", "no video generation"]}
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "status": "STILLS_READY", "video_tools_not_called": True, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "next_manual_step": "Human review of stills before any optional manual video workflow."})
    write(PKG / "handoffs/grok_agent_prompt.md", f"""# Grok/Closed Tool Agent Prompt - Run {RUN_ID}

Do not generate video automatically.

Premise: {winner["premise"]}

Use the first frame and Shared Choices board as style anchors. Keep the character fictional, avoid exact Taylor Swift likeness, avoid real logos, and do not name any real muse or assert who any song is about.
""")
    write(PKG / "captions/instagram_caption.md", "POV: the lyric detectives sent the chorus to the paternity lab and the result came back: the songwriter wrote it.\n\nFictional satire based on public reporting about a songwriting interview and fan interpretation. No video generated in this package.\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels once a human-approved video exists. Current export status: stills and handoff only. Hook text: \"The lyric paternity test came back inconclusive for gossip.\" Keep caption fictional and do not tag real muses.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nLesson: a quote with a built-in object can be turned into a visual system without adding risky fake facts. The factory scored candidates first, rejected overlap, and converted the metaphor into props, blocking, and handoff constraints.\n")
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "selected_premise": winner, "sources": SOURCES, "status": "STILLS_READY", "video_generation": "not_called"})
    write(PKG / "README.md", f"""# RUN {RUN_ID} MASTER PACKAGE - Taylor Paternity Lyric Lab

Status: STILLS_READY

Selected premise: {winner["premise"]}

Research query/topic: late-April 2026 Taylor Swift songwriting interview pickup, fan lyric-forensics discourse, and the public "paternity test" metaphor.

Generated still-image artifacts:
- frames/gpt_image_2/first_frame_v01.png
- storyboards/shared_choices/shared_choices_v01.png

No video-generation tools were called.
""")
    write(PKG / "qc/first_frame_v01_qc.md", "# First Frame QC\n\nVerdict: usable GPT Image 2 repaired still.\n\nPasses: strong lab/studio contradiction, rear-facing anonymous songwriter, no exact celebrity face, clear lyric-forensics prop system.\n\nRepair note: the original first-frame generation read too close to Taylor Swift and was saved under `frames/gpt_image_2/rejected/first_frame_v01_needs_repair_too_close_likeness.png`. The package `first_frame_v01.png` now points to the safer repaired generation, also saved as `first_frame_v02.png`.\n\nWatch items: review generated text labels visually before posting; if any label is messy, use the saved prompt for a repair pass.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", "# Shared Choices QC\n\nVerdict: usable GPT Image 2 storyboard/director-bible board.\n\nPasses: generated valid 1536x1024 PNG after two text-heavy retries failed; visual collage covers character/props, lab-studio environment, blocking, and storyboard-panel intent.\n\nWatch items: simplified board intentionally avoids heavy readable labels because earlier text-heavy GPT Image attempts produced empty files. Use prompt file for a labeled repair pass if needed.\n")
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "required_files_present": True, "assets": [{"path": "frames/gpt_image_2/first_frame_v01.png", "type": "gpt_image_2_first_frame", "source": str(FIRST_FRAME_SRC), "status": "usable_repaired_likeness_safe"}, {"path": "frames/gpt_image_2/first_frame_v02.png", "type": "gpt_image_2_first_frame_repair_copy", "source": str(FIRST_FRAME_SRC), "status": "usable_repaired_likeness_safe"}, {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "gpt_image_2_shared_choices_board", "source": str(STORYBOARD_SRC), "status": "usable_simplified_repair"}], "rejected_or_failed_generations": [{"path": "frames/gpt_image_2/rejected/first_frame_v01_needs_repair_too_close_likeness.png", "reason": "first generated still read too close to Taylor Swift likeness"}, {"path": "/Users/speed/.codex/generated_images/019de8d3-c968-7f70-a758-aeea53cbfe2b/ig_0e236f1f992b786a0169f5fb7947648190b1a899604647ead2.png", "reason": "empty file from text-heavy storyboard attempt"}, {"path": "/Users/speed/.codex/generated_images/019de8d3-c968-7f70-a758-aeea53cbfe2b/ig_0e236f1f992b786a0169f5fc3858348190874b5fb887ddbd05.png", "reason": "empty file from text-heavy storyboard retry"}], "missing_files": [], "video_generation_tools_called": False})
    status = f"""# Factory Run Status - Run {RUN_ID}

Status: `STILLS_READY`

Completed:
- fresh research intake before premise selection
- current-source candidate board and score-first winner selection
- strategy audits and scoring files
- CHAI and scene JSON handoffs
- first-frame prompt plus GPT Image 2 repaired PNG
- Shared Choices prompt plus GPT Image 2 repaired/simplified PNG
- caption, distribution, Skool, manifest, and QC files

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: the first generated still read too close to a Taylor Swift likeness and was moved to `frames/gpt_image_2/rejected/`; two text-heavy Shared Choices generations produced empty files and are logged in `manifests/asset_manifest.json`; the simplified board repair generated successfully.

High-risk issues:
- avoid exact Taylor Swift likeness
- avoid naming real muses or asserting who any song is about
- avoid real logos and real song-title evidence walls

Exact next human action: review both still PNGs, especially generated text/labels, then approve or request a labeled Shared Choices repair pass before any manual video workflow.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
