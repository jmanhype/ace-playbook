from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_ID = "052"
SLUG = "lion_king_translation_court"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

FIRST_FRAME_SRC = Path("/Users/speed/.codex/generated_images/019de8ea-38db-7c83-955f-c4efc070c814/ig_084a704596dd92160169f5ff40a3fc8195bd2f7bcb3ba3a2c9.png")
STORYBOARD_SRC = Path("/Users/speed/.codex/generated_images/019de8ea-38db-7c83-955f-c4efc070c814/ig_084a704596dd92160169f5ffcf0c388195a3fdc94e196e703b.png")

FIRST_PROMPT = """Use case: stylized-concept
Asset type: SGFLIX first-frame still, 16:9 cinematic editorial satire
Primary request: Create a premium satirical first-frame image for a fictional SGFLIX short about a public report that a comedian was sued over a viral mistranslation joke about an iconic movie-opening chant. Do not use Disney logos, copyrighted character likenesses, real faces, real names, or exact song text.
Scene/backdrop: A grand Los Angeles courtroom absurdly merged with a golden savanna soundstage and a sterile translation lab. One side has court benches and a judge's desk; the other side has a phonetics lab table with headphones, waveform monitors, sealed lyric evidence bags, and a giant receipt-like damages calculator reading only "$27M" in clean generic typography.
Subject: A fictional stand-up comedian silhouette at a tiny witness microphone, a fictional composer silhouette holding sheet music, and neutral court clerks treating a joke translation like forensic evidence. No real people.
Composition: Wide 16:9, strong first-frame contradiction, centered evidence table, dramatic shafts of warm stage light, blue lab monitor highlights, crisp readable props but minimal text.
Mood/style: Premium magazine editorial, tasteful absurdity, cinematic, high production design, sharp details, no messy labels.
Avoid: real logos, copyrighted characters, lions as recognizable movie characters, exact lyrics, defamatory framing, watermark, gibberish text, photoreal likeness of any real person."""

BOARD_PROMPT = """Use case: infographic-diagram
Asset type: SGFLIX Shared Choices director-bible storyboard board, 3:2
Primary request: Create a clean director-bible storyboard board for the same fictional satire: a comedy translation lawsuit becomes a courtroom plus savanna soundstage plus phonetics lab. This is a visual planning board, not a video render.
Board layout: 6 organized zones with very minimal readable labels: CHARACTER + PROPS, PALETTE, SET DESIGN, BLOCKING, PANELS, RULES. Keep labels large and simple only; avoid paragraphs and tiny text.
Include: fictional comedian silhouette, fictional composer silhouette, neutral judge/clerks, hero props (sealed lyric sheet, headphones, waveform monitor, giant "$27M" damages receipt, translation notes, evidence bags), warm gold savanna set, cool blue lab, courtroom wood.
Storyboard panels: three small panels showing 1. evidence-table reveal, 2. translation lab close-up, 3. damages receipt punchline. Add simple camera glyphs and arrows, not text-heavy notes.
Style: premium cinematic production board, polished collage grid, clear floor plan/blocking diagram, color swatches, lighting/mood strip, sharp composition, no nested cards.
Avoid: real logos, Disney marks, copyrighted characters, exact lyrics, real people, defamatory framing, gibberish text, watermark, video generation references."""

SOURCES = [
    {
        "id": "ap_lion_king_lawsuit",
        "title": "Composer of iconic Lion King chant sues comedian over Circle of Life translation",
        "url": "https://apnews.com/article/2b653d3ae9337f231e6a74201f4011e8",
        "publisher": "Associated Press",
        "date": "2026-03-24",
        "verified_facts": [
            "AP reported that Lebohang Morake, known as Lebo M, sued Zimbabwean comedian Learnmore Mwanyenyeka, known as Learnmore Jonasi, in federal court in Los Angeles.",
            "AP reported the complaint seeks more than $20 million in actual damages and $7 million in punitive damages.",
            "AP reported the complaint argues Jonasi presented the translation as authoritative fact, not comedy.",
            "AP reported Jonasi said he is a fan of Morake's work and wanted the situation to educate people.",
        ],
        "creative_use": "Primary verified source for the $27M joke-translation courtroom premise.",
    },
    {
        "id": "nbc_la_lion_king_lawsuit",
        "title": "Comedian sued for $27M over Circle of Life joke",
        "url": "https://www.nbclosangeles.com/news/national-international/lion-king-song-composer-sues-comedian-circle-of-life-joke/3867591/",
        "publisher": "NBC Los Angeles",
        "date": "2026-03-25",
        "verified_facts": [
            "NBC Los Angeles reported the California federal complaint and $27M damages number.",
            "The report framed the dispute around whether the viral translation joke damaged the composer's work and reputation.",
        ],
        "creative_use": "Local court-jurisdiction support for a Los Angeles courtroom set.",
    },
    {
        "id": "lat_lion_king_lawsuit",
        "title": "Lion King composer sues comedian for viral botched translation",
        "url": "https://www.latimes.com/entertainment-arts/story/2026-03-27/lion-king-composer-sues-comedian-for-viral-botched-translation",
        "publisher": "Los Angeles Times",
        "date": "2026-03-27",
        "verified_facts": [
            "The Los Angeles Times reported on the lawsuit and the allegation that the translation joke misrepresented the song's meaning.",
            "The report connects the case to the Los Angeles performance context.",
        ],
        "creative_use": "Entertainment-local confirmation and flavor for the LA courtroom frame.",
    },
    {
        "id": "parade_lion_king_case_escalates",
        "title": "Lion King lawsuit: comedian makes move after getting sued for $27 million",
        "url": "https://parade.com/celebrities/lion-king-lawsuit-comedian-learnmore-jonasi-sued/",
        "publisher": "Parade",
        "date": "2026-03-27",
        "verified_facts": [
            "Parade reported the case as a viral joke turned high-stakes legal battle.",
            "The report cited the complaint's argument that the joke should not be protected as comedy.",
        ],
        "creative_use": "Supports the absurd legal-procedure angle and comedy-versus-fact visual bridge.",
    },
    {
        "id": "vice_served_mid_show",
        "title": "The on-stage moment a comedian found out he was being sued for $27 million over a Lion King joke",
        "url": "https://www.vice.com/en/article/the-on-stage-moment-a-comedian-found-out-he-was-being-sued-27-million-for-his-r-rated-translation-of-a-lion-king-song/",
        "publisher": "Vice",
        "date": "2026-03-27",
        "verified_facts": [
            "Vice reported the story as an on-stage lawsuit shock and viral comedy controversy.",
            "The report is used only as cultural pickup, not as the primary factual authority.",
        ],
        "creative_use": "Supports the stand-up microphone plus courtroom contradiction.",
    },
    {
        "id": "ap_afroman_speech_lawsuit",
        "title": "Rapper Afroman wins lawsuit against police over mocking their 2022 raid in viral music videos",
        "url": "https://apnews.com/article/309accc1ce068620e19cfd7d0f70dae1",
        "publisher": "Associated Press",
        "date": "2026-03-19",
        "verified_facts": [
            "AP reported Afroman won a lawsuit brought by Ohio sheriff's deputies over mockery in videos.",
            "The case also centered on satire, criticism, and public speech.",
        ],
        "creative_use": "Rejected candidate source: strong free-speech lane, but less fresh and lower first-frame novelty for this cycle.",
    },
    {
        "id": "cnn_maher_loomer_dismissal",
        "title": "Judge tosses Laura Loomer's lawsuit, says Bill Maher joke wasn't defamation",
        "url": "https://kvia.com/news/business-technology/cnn-business-consumer/2026/04/22/judge-tosses-laura-loomers-lawsuit-says-bill-maher-joke-wasnt-defamation/",
        "publisher": "CNN via KVIA",
        "date": "2026-04-22",
        "verified_facts": [
            "CNN reported a federal judge dismissed Laura Loomer's defamation suit over a Bill Maher joke.",
            "The report is politically sensitive and sexually framed, making it a poor fit for a tasteful SGFLIX package.",
        ],
        "creative_use": "Rejected for taste and platform risk despite high fame context.",
    },
    {
        "id": "ap_maher_mark_twain",
        "title": "Bill Maher will win the Kennedy Center's Mark Twain humor prize following White House denial",
        "url": "https://apnews.com/article/0c41af4f1460a1b52cd234c6ce5d2c02",
        "publisher": "Associated Press",
        "date": "2026-03-26",
        "verified_facts": [
            "AP reported Bill Maher would receive the Mark Twain Prize after White House pushback calling the story fake news.",
        ],
        "creative_use": "Rejected candidate source: funny institutional contradiction but less visual than the $27M translation courtroom.",
    },
]

CANDIDATES = [
    {
        "id": "lion_king_translation_court",
        "premise": "A joke translation gets treated like courtroom forensics: a savanna soundstage, phonetics lab, and $27M damages receipt all arguing whether comedy is allowed to be comedy.",
        "source_ids": ["ap_lion_king_lawsuit", "nbc_la_lion_king_lawsuit", "lat_lion_king_lawsuit", "parade_lion_king_case_escalates", "vice_served_mid_show"],
        "first_frame": "Los Angeles courtroom fused with golden savanna soundstage and blue phonetics lab; a giant $27M receipt towers over sealed lyric evidence.",
        "scores": {
            "fame_context": 9,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "visual_contradiction": 10,
            "absurd_quote_or_object": 10,
            "brand_location_contrast": 10,
            "risk_control": 7,
            "franchise_potential": 8,
        },
        "total": 71,
    },
    {
        "id": "afroman_raid_music_video_verdict",
        "premise": "A courthouse evidence cart plays home-security footage as a rap video while a verdict stamp tries to dance.",
        "source_ids": ["ap_afroman_speech_lawsuit"],
        "first_frame": "Courthouse hallway, raid footage monitor, lemon-pound-cake exhibit, freedom-of-speech stamp.",
        "scores": {
            "fame_context": 7,
            "public_conflict": 8,
            "ego_humiliation": 8,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 7,
            "risk_control": 7,
            "franchise_potential": 6,
        },
        "total": 59,
    },
    {
        "id": "maher_fake_news_trophy_claim_check",
        "premise": "The Mark Twain Prize arrives at a claims counter after being stamped FAKE NEWS by the wrong office.",
        "source_ids": ["ap_maher_mark_twain"],
        "first_frame": "Kennedy Center trophy in a bureaucratic lost-and-found window labeled humor prize verification.",
        "scores": {
            "fame_context": 8,
            "public_conflict": 7,
            "ego_humiliation": 7,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 7,
            "risk_control": 6,
            "franchise_potential": 6,
        },
        "total": 57,
    },
    {
        "id": "loomer_maher_joke_defamation_shredder",
        "premise": "A defamation shredder rejects a joke lawsuit while cable-news chyron foam floods the room.",
        "source_ids": ["cnn_maher_loomer_dismissal"],
        "first_frame": "Late-night desk meets courthouse shredder with redacted complaint pages.",
        "scores": {
            "fame_context": 8,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "visual_contradiction": 7,
            "absurd_quote_or_object": 7,
            "brand_location_contrast": 7,
            "risk_control": 3,
            "franchise_potential": 6,
        },
        "total": 55,
        "penalty": "Rejected for sexual/political taste risk.",
    },
    {
        "id": "kimmel_fire_button_tour",
        "premise": "A late-night joke is led through an HR firing simulator with an oversized network button.",
        "source_ids": [],
        "first_frame": "Network control room with joke teleprompter, HR badge printer, and fire button.",
        "scores": {
            "fame_context": 8,
            "public_conflict": 8,
            "ego_humiliation": 7,
            "visual_contradiction": 7,
            "absurd_quote_or_object": 7,
            "brand_location_contrast": 7,
            "risk_control": 4,
            "franchise_potential": 6,
        },
        "total": 54,
        "penalty": "Rejected because a recent SGFLIX Kimmel/ABC firing-desk package already owns the lane.",
    },
]


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
    copy_image(STORYBOARD_SRC, PKG / "storyboards/shared_choices/shared_choices_v01.png")

    write(PKG / "research/last30days_report.md", f"""# Step 1: Research Intake - Run {RUN_ID}

Run time: {NOW}

Research query/topic: late-March to late-April 2026 entertainment-law stories where public jokes, parody, or satire were pulled into formal legal/institutional machinery.

Process note: premise selection started from fresh current web/source context and not from a nearby image, old storyboard, previous handoff, or existing local asset. Candidate scoring happened before the run package was created.

Verified source context:
- Associated Press reported on March 24, 2026 that composer Lebohang Morake, known as Lebo M, sued comedian Learnmore Mwanyenyeka, known as Learnmore Jonasi, in federal court in Los Angeles over an alleged mistranslation joke involving an iconic movie-opening chant.
- AP reported the complaint seeks more than $20 million in actual damages and $7 million in punitive damages.
- AP reported the complaint argues the translation was presented as authoritative fact, not comedy.
- NBC Los Angeles, the Los Angeles Times, Parade, and Vice all picked up the $27M lawsuit story, confirming broader entertainment-law attention.
- AP also reported Afroman winning a lawsuit over mocking raid footage, and CNN reported a judge dismissed a defamation lawsuit over a Bill Maher joke. Those were scored but not selected.

Unverified or sensitive context:
- This package does not adjudicate the lawsuit and does not claim either side is legally right.
- This package does not reproduce exact song lyrics, official translations, Disney marks, copyrighted characters, or photoreal likenesses of real people.
- All visuals use fictional silhouettes and generic props.

Winner selected after candidate scoring: `{winner["id"]}`.
""")
    write_json(PKG / "research/sources.json", {"run_id": RUN_ID, "generated_at": NOW, "sources": SOURCES})

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "scored_before_winner": True, "candidates": CANDIDATES})
    write(PKG / "strategy/winner_decision.md", f"""# Winner Decision - Run {RUN_ID}

Selected premise: **{winner["id"]}**

Selected premise: {winner["premise"]}

Why it won: the source supplies an instant absurd object: a giant damages number attached to a joke translation. The courtroom plus savanna soundstage plus phonetics-lab collision creates a strong first-frame contradiction while the package can avoid exact lyrics, logos, real faces, and legal conclusions.

Score summary:
- lion_king_translation_court: 71
- afroman_raid_music_video_verdict: 59
- maher_fake_news_trophy_claim_check: 57
- loomer_maher_joke_defamation_shredder: 55, rejected for sexual/political taste risk
- kimmel_fire_button_tour: 54, rejected for recent SGFLIX overlap

First-frame mandate: show formal legal machinery treating a joke translation like forensic evidence. Keep the joke on the institutional overreaction and the collision of settings, not on any private claim about the real parties.
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "public $27M joke-translation lawsuit", "track_a_newsjack_velocity": {"active_trend_score": 7, "algorithmic_slipstream": "Medium-high entertainment and legal pickup inside the last 30 days", "polarization_factor": 7, "track_a_total": 21, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 10, "stereotype_rigidity": "High", "subversion_potential": 10, "track_b_total": 30, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_B_EVERGREEN_ARCHETYPE", "urgency_class": "High", "strategic_directive": "Use famous cultural iconography only as generic silhouette and set logic; make the court/translation machinery the comedy engine."}}})
    write_json(PKG / "strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "A real legal dispute over whether a viral translation joke damaged a composer's reputation and business interests.", "detected_anomalies": ["A joke translation becomes a multimillion-dollar damages claim", "Comedy is argued as authoritative fact", "A movie-song cultural dispute lands in a Los Angeles federal court frame"], "native_entropy_score": 7, "subject_self_awareness": "trying_to_look_serious", "comedic_vector_recommendation": "native_absurdity", "recommended_strategy": "straight_man_framing"}})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"bridge": {"truth": "Legal systems can turn a joke into exhibits, damages math, and procedural gravity.", "comic_flip": "Render the translation joke as a phonetics lab and courtroom treating sound waves like felony evidence.", "humiliation_engine": "The giant $27M receipt looks more serious than the tiny witness microphone.", "visual_payoff": "Courtroom wood, savanna stage light, lab-blue waveform monitors, sealed lyric evidence, and a damages receipt."}})
    write_json(PKG / "strategy/tribe_meta_score.json", {"tribe_meta_score": {"attention_tribes": ["stand-up comedy fans", "free-speech and parody watchers", "Disney-adjacent nostalgia discourse", "entertainment-law spectators", "African diaspora media conversation"], "share_trigger": "viewers recognize the absurdity of legal machinery processing a joke without needing exact lyrics or real faces", "meta_score": 86}})
    write_json(PKG / "strategy/risk_taste_score.json", {"risk_taste_score": {"legal_risk": 5, "likeness_risk": 3, "ip_risk": 7, "taste_risk": 4, "mitigations": ["No Disney logos or copyrighted characters", "No exact lyrics or official translation text", "Fictional silhouettes only", "Mark lawsuit claims as allegations", "Avoid declaring legal outcome"], "go_no_go": "GO_WITH_CONSTRAINTS"}})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nProceed as a serious one-off with reusable franchise shape: public joke enters absurd legal/institutional apparatus. The package can seed future \"joke on trial\" episodes, but this source should stand alone because of IP risk around the famous movie-song context.\n")

    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT + "\n")

    shot = {
        "shot_id": "shot_001",
        "duration_seconds": 6,
        "subject": "fictional comedian silhouette, fictional composer silhouette, neutral judge, court clerks, and phonetics technicians",
        "scene": "Los Angeles courtroom merged with golden savanna soundstage and cool-blue phonetics lab",
        "motion": "slow push-in from sealed lyric evidence to giant $27M damages receipt",
        "spatial": "courtroom left, savanna stage center, phonetics lab right, evidence table foreground",
        "camera": "24mm wide first-frame, 50mm evidence insert, 70mm damages receipt punchline",
        "critique": "Do not reproduce exact lyrics, official translations, real faces, logos, or copyrighted character likenesses.",
        "revision": "If IP markers appear, repair by removing recognizable movie elements and emphasizing generic court/lab props.",
    }
    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [shot]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "prompt": FIRST_PROMPT, "duration_seconds": 6, "no_video_generation": True, "camera": {"lens": "24mm to 70mm", "movement": "slow push-in"}, "negative_constraints": ["no video generation", "no Disney marks", "no exact song lyrics", "no real faces", "no legal conclusion"] }
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)

    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "status": "STILLS_READY", "video_tools_not_called": True, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "next_manual_step": "Human review of stills for IP/text issues before any optional manual video workflow.", "render_constraints": ["Do not call Grok Video, Kling, Runway, Sora, Luma, Seedance, or any video tool from this package.", "Keep all subjects fictional silhouettes.", "Avoid exact lyrics and Disney marks."]})
    write(PKG / "handoffs/grok_agent_prompt.md", f"""# Grok/Closed Tool Agent Prompt - Run {RUN_ID}

Do not generate video automatically.

Premise: {winner["premise"]}

Use the first frame and Shared Choices board as visual anchors only. Keep the story fictionalized, use silhouettes instead of real faces, avoid Disney logos, avoid copyrighted character likenesses, avoid exact lyrics, and do not state who is legally correct. The joke is the legal/translation machinery treating comedy like forensic evidence.
""")

    write(PKG / "captions/instagram_caption.md", "POV: the translation joke got subpoenaed and the damages calculator came dressed for opening night.\n\nFictional satire based on public reporting about a lawsuit over a viral comedy translation joke. No legal conclusion, no exact lyrics, no video generated in this package.\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels after human-approved video exists. Current export status: stills and handoff only.\n\nHook text options:\n- \"When the joke enters evidence.\"\n- \"The damages receipt heard the punchline first.\"\n- \"Courtroom, savanna, phonetics lab: normal lawsuit stuff.\"\n\nDo not tag real parties or Disney. Keep caption framed as fictional satire based on public reporting.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nLesson: when a current source already contains a legal/comedy contradiction, the strongest move is straight-man production design. This run scored multiple joke-lawsuit candidates, selected the one with the clearest visual machine, and reduced risk by fictionalizing faces, removing exact lyrics, and treating claims as allegations.\n")

    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "selected_premise": winner, "sources": SOURCES, "status": "STILLS_READY_NO_VIDEO_GENERATED", "video_generation": "not_called", "generated_stills": {"first_frame": "frames/gpt_image_2/first_frame_v01.png", "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"}})
    write(PKG / "README.md", f"""# RUN {RUN_ID} MASTER PACKAGE - Lion King Translation Court

Status: STILLS_READY_NO_VIDEO_GENERATED

Selected premise: {winner["premise"]}

Research query/topic: late-March to late-April 2026 entertainment-law stories where public jokes, parody, or satire were pulled into formal legal/institutional machinery.

Generated still-image artifacts:
- frames/gpt_image_2/first_frame_v01.png
- storyboards/shared_choices/shared_choices_v01.png

No video-generation tools were called.
""")

    write(PKG / "qc/first_frame_v01_qc.md", "# First Frame QC\n\nVerdict: usable GPT Image 2 first-frame still.\n\nPasses: 1672x941 PNG, strong courtroom/savanna/lab contradiction, readable $27M receipt, fictional silhouettes instead of real faces, clear evidence-table prop system.\n\nWatch items: image includes generic lion silhouettes and courtroom insignia-like graphics. Human should confirm there are no Disney-specific marks before posting. Some paper text is intentionally non-substantive and should not be treated as readable claims.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", "# Shared Choices QC\n\nVerdict: usable GPT Image 2 director-bible board.\n\nPasses: 1536x1024 PNG, includes character and hero props, color palette, environment/set design, floor plan/blocking, three storyboard panels with camera arrows, lighting/mood/rules strip, and production-board organization.\n\nWatch items: labels are readable but generated; human should verify no accidental brand marks or exact lyrics. Board is approved for internal direction, not for public posting as-is.\n")
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "required_files_present": True, "assets": [{"path": "frames/gpt_image_2/first_frame_v01.png", "type": "gpt_image_2_first_frame", "source": str(FIRST_FRAME_SRC), "status": "usable"}, {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "gpt_image_2_shared_choices_board", "source": str(STORYBOARD_SRC), "status": "usable"}], "missing_files": [], "video_generation_tools_called": False, "post_ready_exports": []})

    status = f"""# Factory Run Status - Run {RUN_ID}

Status: `STILLS_READY_NO_VIDEO_GENERATED`

Completed:
- fresh research intake before premise selection
- current-source candidate board and score-first winner selection
- strategy audits and scoring files
- CHAI and scene JSON handoffs
- first-frame prompt plus GPT Image 2 PNG
- Shared Choices prompt plus GPT Image 2 PNG
- caption, distribution, Skool, manifest, and QC files

Research query/topic: late-March to late-April 2026 entertainment-law stories where public jokes, parody, or satire were pulled into formal legal/institutional machinery.

Selected premise: {winner["premise"]}

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: none blocking. Human review should verify no accidental IP marks or exact lyric-like text in generated images.

High-risk issues:
- famous movie-song IP context: avoid logos, copyrighted characters, and exact lyrics
- lawsuit facts are allegations: do not state a legal conclusion
- real-party likeness: keep all humans as fictional silhouettes

Exact next human action: review both still PNGs for accidental IP/text issues, then approve or request an IP-clean repair pass before any manual video workflow.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
