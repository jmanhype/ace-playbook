from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = "054"
SLUG = "kash_breaching_receipt_desk"
RUN_DIR = Path("sgflix_runs") / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

FIRST_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

Aspect ratio: 9:16 vertical first frame for SGFLIX.

Concept: Satirical premium editorial first-frame image based on a public defamation lawsuit news story, using a fictional public-official silhouette rather than an exact likeness. A federal media-litigation evidence room has been converted into an absurd $250 million complaint intake desk. Hero visual contradiction: a giant legal complaint stamped "$250M" feeds into an oversized evidence scanner while a SWAT-style breaching ram rests on a velvet exhibit stand labeled only "LOCKED DOOR KIT". Nearby: an unbranded magazine cover silhouette, a toy Gulfstream receipt clipped to a cork board, redacted source notes, a sober-looking courthouse clock, and a tiny press-room microphone forest behind glass.

Subject: fictional male public official archetype in a navy suit, recognizable only as a general political/legal authority figure, no exact real-person likeness, no badge, no official seal, no real agency logo.

Scene and camera: low wide 24mm lens from desk height, foreground dominated by the $250M complaint and scanner glow, background stacked with legal boxes and press lights. Cinematic documentary realism, crisp production design, sharp satirical object comedy, premium news-magazine color grade.

Style: tasteful absurdity, high-end editorial satire, concrete prop logic, clean readable shapes, neutral steel/red/white palette with small warning-yellow accents, natural overhead office lighting plus scanner glow, subtle film grain.

Text rules: only minimal large prop text "$250M" and "LOCKED DOOR KIT" may appear; no captions, no fake news chyron, no watermark, no real logos, no official insignia.

Avoid next - exact Kash Patel likeness, real FBI/Atlantic logos, defamatory depiction of intoxication as fact, alcohol props, messy text, distorted faces, identity drift, oversharpening, oversaturation, excessive yellow in the photo.
"""

BOARD_PROMPT = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

Aspect ratio: 3:2 Shared Choices director-bible storyboard board for SGFLIX.

Concept: A polished production design board for a satirical short about a public-official defamation suit turning into a surreal federal media-litigation intake room. The board must feel like a director's bible, not a meme, and must avoid exact real-person likenesses or real logos.

Board zones to include visually: character canon for a fictional navy-suited public official silhouette; hero props including a $250M legal complaint, evidence scanner, SWAT-style breaching ram on velvet stand, unbranded magazine silhouette, toy Gulfstream receipt, redacted source notes, courthouse clock, and press microphones behind glass; color palette swatches in steel gray, courtroom red, paper white, scanner blue, restrained warning yellow; environment/set design for a federal records room crossed with a media press pen; simple floor plan/blocking with scanner desk foreground, locked-door-kit pedestal right, corkboard left, press lights background; three storyboard panels with camera/lens/movement notes showing first frame, scanner push-in, and press-room reveal; lighting/mood/style notes; visual rules; production notes.

Style: premium editorial satire, cinematic documentary realism, clean prop logic, legible layout, practical production-board design, neat annotations, high-end commercial storyboard polish.

Text rules: concise readable labels only, no fake article text, no real agency logos, no real magazine mastheads, no watermarks.

Avoid next - exact Kash Patel likeness, real FBI/Atlantic logos, defamatory depiction of intoxication as fact, alcohol props, messy generated paragraphs, distorted faces, one-note blue palette, oversharpening, oversaturation, excessive yellow in the photo.
"""

SOURCES = [
    {
        "id": "washpost_patel_sues_atlantic",
        "title": "FBI chief Kash Patel sues the Atlantic, alleging defamation",
        "url": "https://www.washingtonpost.com/business/2026/04/20/fbi-patel-lawsuit-atlantic-defamation/",
        "publisher": "The Washington Post",
        "date": "2026-04-20",
        "verified_facts": ["Patel sued The Atlantic and Sarah Fitzpatrick.", "The complaint seeks $250 million.", "The Atlantic said it stands by its reporting."],
    },
    {
        "id": "axios_patel_defamation",
        "title": "Kash Patel files $250M defamation lawsuit against The Atlantic",
        "url": "https://www.axios.com/2026/04/20/kash-patel-lawsuit-the-atlantic",
        "publisher": "Axios",
        "date": "2026-04-20",
        "verified_facts": ["Reported the $250 million defamation filing.", "Noted the high actual-malice bar for public-figure defamation claims."],
    },
    {
        "id": "cbs_patel_17_allegations",
        "title": "FBI Director Kash Patel sues The Atlantic for $250 million over story on alleged drinking, absences",
        "url": "https://www.cbsnews.com/news/kash-patel-lawsuit-the-atlantic-250-million/",
        "publisher": "CBS News",
        "date": "2026-04-20",
        "verified_facts": ["The lawsuit named The Atlantic and Sarah Fitzpatrick.", "The complaint listed article statements Patel's team alleges are false and defamatory."],
    },
    {
        "id": "atlantic_podcast_breaching_equipment",
        "title": "The Kash Patel Fallout",
        "url": "https://www.theatlantic.com/podcasts/2026/04/kash-patel-fallout/686907/",
        "publisher": "The Atlantic",
        "date": "2026-04-23",
        "verified_facts": ["The Atlantic described its reporting as alleging erratic behavior and excessive drinking.", "The podcast page says Fitzpatrick reported a breaching-equipment request because Patel was allegedly unreachable behind locked doors.", "Patel called the story a lie and sued."],
    },
    {
        "id": "spokesman_gulfstream_context",
        "title": "FBI Director Kash Patel sues the Atlantic for $250M, alleging defamation",
        "url": "https://www.spokesman.com/stories/2026/apr/20/fbi-director-kash-patel-sues-the-atlantic-for-250m/",
        "publisher": "The Spokesman-Review",
        "date": "2026-04-20",
        "verified_facts": ["Republished Washington Post reporting on the lawsuit.", "Mentioned public criticism of Patel's agency Gulfstream use."],
    },
    {
        "id": "local_last30days_attempt",
        "title": "Local last30days scan attempt",
        "url": "local:/Users/speed/.codex/skills/last30days-skill/scripts/last30days.py",
        "publisher": "Local SGFLIX research workflow",
        "date": "2026-05-02",
        "verified_facts": ["Reddit enrichment failed on API 429 rate limits.", "The script failed to persist output because the local volume was effectively full.", "The failed scan is recorded as a limitation, not as factual support."],
    },
]

CANDIDATES = [
    {"id": "kash_breaching_receipt_desk", "premise": "A $250M defamation complaint becomes a federal media-litigation intake desk where the absurd object evidence is a locked-door kit, redacted source notes, and a toy Gulfstream receipt.", "source_ids": ["washpost_patel_sues_atlantic", "axios_patel_defamation", "cbs_patel_17_allegations", "atlantic_podcast_breaching_equipment", "spokesman_gulfstream_context"], "scores": {"famous_face_or_power_archetype": 8, "public_conflict": 10, "ego_humiliation": 8, "absurd_quote_or_object": 9, "brand_location_contrast": 8, "first_frame_visual_contradiction": 10, "freshness": 9, "risk_control": 7, "franchise_potential": 8}, "total": 77},
    {"id": "comey_seashell_forensics_redux", "premise": "A beach-shell Instagram photo is processed like federal threat forensics, with shells in evidence bags and a literal ambiguity meter.", "source_ids": ["web_search_nbc_ap_comey"], "scores": {"famous_face_or_power_archetype": 8, "public_conflict": 9, "ego_humiliation": 7, "absurd_quote_or_object": 10, "brand_location_contrast": 8, "first_frame_visual_contradiction": 9, "freshness": 10, "risk_control": 6, "franchise_potential": 7}, "total": 58, "penalty": "Rejected for overlap with existing run_027_comey_seashell_evidence."},
    {"id": "ye_chateau_lobby_claims_desk", "premise": "A luxury hotel lobby turns into a civil-complaint lost-and-found counter for a celebrity altercation claim.", "source_ids": ["web_search_latimes_ye_chateau"], "scores": {"famous_face_or_power_archetype": 9, "public_conflict": 8, "ego_humiliation": 8, "absurd_quote_or_object": 6, "brand_location_contrast": 9, "first_frame_visual_contradiction": 8, "freshness": 8, "risk_control": 4, "franchise_potential": 6}, "total": 56, "penalty": "Rejected for higher violence/defamation risk and less clean object comedy."},
    {"id": "blake_brand_damage_checkout", "premise": "A celebrity brand-damage claim becomes a beauty/alcohol checkout lane where every receipt argues with the lawsuit.", "source_ids": ["web_search_lively_baldoni_reddit_discourse"], "scores": {"famous_face_or_power_archetype": 9, "public_conflict": 9, "ego_humiliation": 8, "absurd_quote_or_object": 7, "brand_location_contrast": 10, "first_frame_visual_contradiction": 8, "freshness": 7, "risk_control": 5, "franchise_potential": 8}, "total": 55, "penalty": "Rejected for litigated interpersonal allegations and messy source quality."},
    {"id": "generic_public_figure_sues_media", "premise": "A public figure sues a magazine and a courtroom fills with microphones.", "source_ids": ["washpost_patel_sues_atlantic"], "scores": {"famous_face_or_power_archetype": 7, "public_conflict": 8, "ego_humiliation": 5, "absurd_quote_or_object": 3, "brand_location_contrast": 4, "first_frame_visual_contradiction": 4, "freshness": 8, "risk_control": 8, "franchise_potential": 4}, "total": 51, "penalty": "Fails SGFLIX specificity gate."},
]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    write(path, json.dumps(obj, indent=2) + "\n")


def main() -> None:
    winner = CANDIDATES[0]
    for directory in ["research", "strategy", "chai", "scene_json", "handoffs", "captions", "distribution", "skool", "manifests", "frames/gpt_image_2", "storyboards/shared_choices", "qc"]:
        (PKG / directory).mkdir(parents=True, exist_ok=True)

    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT)
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT)
    write(PKG / "research/last30days_report.md", f"""# Step 1: Research Intake - Run {RUN_ID}

Run time: {NOW}

Research query/topic: current late-April to May 2026 public-figure/media-law conflicts with absurd props, power humiliation, and strong first-frame visual contradiction.

Process note: this cycle began with current research intake. No local image, old storyboard, or existing handoff was used to choose the premise.

Local last30days-style scan:
- Command attempted with the current-culture SGFLIX premise query.
- Result: partial intake only. Reddit enrichment hit API 429 rate limits; the script then failed writing its report cache because the local volume was effectively full.
- Salvageable signal was not used as factual support because no durable report was emitted.

Verified web/source context:
- On April 20, 2026, multiple outlets reported that Kash Patel sued The Atlantic and staff writer Sarah Fitzpatrick for defamation, seeking $250 million.
- The reporting centers on The Atlantic article's allegations about drinking, absences, erratic behavior, and a reported locked-door/breaching-equipment incident; Patel denies the reporting and calls it false.
- Some coverage also notes public criticism of Patel's agency Gulfstream use. This package uses that only as public-criticism context.
- All disputed allegations are treated as allegations. The creative does not assert intoxication, misconduct, or legal truth.

Winner selected after candidate scoring: `{winner["id"]}`.
""")
    write_json(PKG / "research/sources.json", {"run_id": RUN_ID, "generated_at": NOW, "sources": SOURCES})
    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "scored_before_winner": True, "candidates": CANDIDATES})
    write(PKG / "strategy/winner_decision.md", f"""# Winner Decision - Run {RUN_ID}

Selected premise: **{winner["id"]}**

Premise: {winner["premise"]}

Why it won: the story is current, public, and specific. The $250M claim gives the scale, the locked-door-kit allegation gives a concrete absurd prop, and the media-litigation intake room gives a controlled satire frame.

Score summary:
- kash_breaching_receipt_desk: 77
- comey_seashell_forensics_redux: 58 after overlap penalty
- ye_chateau_lobby_claims_desk: 56 after risk penalty
- blake_brand_damage_checkout: 55 after source/risk penalty
- generic_public_figure_sues_media: 51
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "Kash Patel Atlantic defamation suit", "track_a_newsjack_velocity": {"active_trend_score": 9, "algorithmic_slipstream": "high current media-law and politics attention", "polarization_factor": 9, "track_a_total": 27, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 8, "stereotype_rigidity": "High", "subversion_potential": 8, "track_b_total": 24, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_A_NEWSJACK", "urgency_class": "High", "strategic_directive": "Use legal-media object comedy while avoiding disputed factual assertions."}}})
    write_json(PKG / "strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "A public official filed a large defamation lawsuit over disputed magazine reporting.", "detected_anomalies": ["$250M demand", "reported breaching-equipment allegation", "public jet-use criticism"], "native_entropy_score": 7, "subject_self_awareness": "trying_to_look_serious", "comedic_vector_recommendation": "native_absurdity", "recommended_strategy": "straight_man_framing"}})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"setup": "A defamation complaint tries to turn a magazine article into a giant damages machine.", "logic_bridge": "If the public fight is about whether allegations can survive scrutiny, the room itself becomes an evidence scanner for the weirdest objects in the complaint discourse.", "payoff": "The scanner is not scanning truth; it is scanning optics, dollar signs, and redactions.", "do_not_do": ["Do not show alcohol", "Do not use real agency logos", "Do not state allegations as fact"]})
    write_json(PKG / "strategy/tribe_meta_score.json", {"tribe_meta_score": {"clarity": 8, "shareability": 8, "visual_specificity": 10, "caption_friction": 7, "remix_potential": 8, "overall": 8.2}})
    write_json(PKG / "strategy/risk_taste_score.json", {"risk_taste_score": {"defamation_risk": "medium-high controlled by allegation framing", "political_sensitivity": "high", "likeness_risk": "medium controlled by fictional silhouette", "taste_risk": "medium", "required_guardrails": ["No exact Kash Patel likeness", "No real FBI or Atlantic marks", "No alcohol props", "All disputed story details labeled as allegations"], "go_no_go": "GO_WITH_GUARDRAILS_BUT_IMAGE_LOCALIZATION_BLOCKED"}})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nDecision: `ONE-OFF_WITH_TEMPLATE_POTENTIAL`\n\nThis can become a repeatable Complaint Intake Desk format for public-figure defamation stories where the claimed damages number and disputed props are more visually interesting than a courtroom. Do not franchise around this specific official or disputed allegations.\n")
    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot": "001", "duration_seconds": 6, "subject": "fictional navy-suited public official archetype, no exact likeness", "scene": "federal media-litigation complaint intake desk", "motion": "slow scanner push-in across the $250M complaint toward the locked-door-kit pedestal", "spatial": "complaint scanner foreground, corkboard left, locked-door-kit right, press lights background", "camera": "24mm low desk-height dolly", "critique": "must read as legal/media satire, not as a fake documentary still", "revision": "remove real logos, alcohol props, or exact face match", "source_frame": "frames/gpt_image_2/first_frame_v01.png"}]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "status": "handoff_only_no_video_generation", "source_image": "frames/gpt_image_2/first_frame_v01.png", "prompt": "Animate a controlled scanner push-in in a federal records room, with paper flutter, scanner glow, and press lights pulsing behind glass. No new video generation in this automation.", "negative": "real logos, exact likeness, alcohol props, fake news chyron, messy text"}
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "do_not_generate_video_in_automation": True, "premise": winner["premise"], "first_frame_prompt_path": "frames/gpt_image_2/first_frame_v01_prompt.md", "storyboard_prompt_path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "required_guardrails": ["Use fictional silhouette", "No real agency/magazine logos", "No alcohol props", "Frame all disputed claims as allegations"]})
    write(PKG / "handoffs/grok_agent_prompt.md", f"# Grok Agent Prompt - Run {RUN_ID}\n\nDo not generate video. Review the satire logic and improve closed-tool handoff language.\n\nPremise: {winner['premise']}\n\nCheck allegation framing, logo avoidance, and one-second first-frame readability.\n")
    write(PKG / "captions/instagram_caption.md", "When the damages number is bigger than the room, the evidence scanner starts asking follow-up questions.\n\nPublic reporting says the lawsuit seeks $250M. The props are satire; the disputed allegations stay disputed.\n\n#sgflix #satire #medialaw #publicfigures #defamation #shortformvideo\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels/TikTok after image QC and manual video render.\n\nHook: \"The $250M complaint intake desk is open.\"\n\nDo not post until a human confirms the first frame contains no real logos, no exact likeness, no alcohol props, and no messy text.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nLesson: turn a risky political/legal newsjack into prop comedy by moving from accusation to paperwork.\n\nThe key move is object translation: $250M becomes a scanner prop, \"breaching equipment\" becomes a labeled kit, source disputes become redacted notes. This keeps the joke visual while reducing factual overclaim risk.\n")
    manifest = {"run_id": RUN_ID, "status": "BLOCKED_IMAGE_LOCALIZATION", "present_assets": ["frames/gpt_image_2/first_frame_v01_prompt.md", "storyboards/shared_choices/shared_choices_v01_prompt.md"], "missing_assets": ["frames/gpt_image_2/first_frame_v01.png", "storyboards/shared_choices/shared_choices_v01.png"]}
    write_json(PKG / "manifests/asset_manifest.json", manifest)
    write(PKG / "qc/IMAGE_GENERATION_BLOCKED_REPORT.md", "# Image Generation Blocked Report\n\nGPT Image 2 prompts were sent after winner selection, but no new accessible PNG appeared under `/Users/speed/.codex/generated_images` during verification.\n\nThe local filesystem was also effectively full during this run, which caused the last30days cache write failure. This run is not marked stills-ready.\n")
    write(PKG / "qc/first_frame_v01_qc.md", "# First Frame QC\n\nStatus: `BLOCKED_NO_LOCAL_PNG`\n\nPrompt exists. Required PNG is missing. Rerun the prompt, save as `frames/gpt_image_2/first_frame_v01.png`, then inspect for exact likeness, logos, alcohol props, and messy text.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", "# Shared Choices QC\n\nStatus: `BLOCKED_NO_LOCAL_PNG`\n\nPrompt exists. Required PNG is missing. Rerun the prompt, save as `storyboards/shared_choices/shared_choices_v01.png`, then inspect labels and layout.\n")
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "status": "BLOCKED_IMAGE_LOCALIZATION", "research_query": "current public-figure media-law conflicts with absurd prop logic", "selected_premise": winner["premise"], "score_summary": {c["id"]: c["total"] for c in CANDIDATES}, "missing_files": manifest["missing_assets"], "post_ready_exports": [], "high_risk_issues": ["political/legal allegations", "defamation-risk framing", "exact public-figure likeness", "real agency/magazine logos"]})
    write(PKG / "README.md", f"# RUN {RUN_ID} Master Package - Kash Breaching Receipt Desk\n\nStatus: `BLOCKED_IMAGE_LOCALIZATION`\n\nSelected premise: {winner['premise']}\n\nThe package was built after research intake and scored candidate selection. Video generation was not called. GPT Image 2 prompts were sent after winner selection, but no accessible new local PNG was found, so still-image work is blocked rather than marked complete.\n\nNext human action: free disk space or locate the generated GPT Image outputs, rerun the two saved prompt files if needed, then place PNGs at the two required paths and perform QC.\n")
    status = f"""# Factory Run Status - Run {RUN_ID}

Status: `BLOCKED_IMAGE_LOCALIZATION`

Completed:
- fresh research intake before premise selection
- current-source candidate board and score-first winner selection
- strategy audits and scoring files
- CHAI and scene JSON handoffs
- first-frame and Shared Choices GPT Image 2 prompt files
- caption, distribution, Skool, manifest, and QC blocked reports

Research query/topic: current late-April to May 2026 public-figure/media-law conflicts with absurd props, power humiliation, and strong first-frame visual contradiction.

Selected premise: {winner['premise']}

Score summary:
- kash_breaching_receipt_desk: 77
- comey_seashell_forensics_redux: 58 after overlap penalty
- ye_chateau_lobby_claims_desk: 56 after risk penalty
- blake_brand_damage_checkout: 55 after source/risk penalty
- generic_public_figure_sues_media: 51

Generated still-image paths: none accessible. GPT Image 2 calls returned no local PNG path under `/Users/speed/.codex/generated_images`.

Missing files:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Post-ready exports: none; no video was generated.

QC failures:
- first-frame PNG missing
- Shared Choices board PNG missing
- local disk nearly full, causing last30days cache write failure

High-risk issues:
- disputed legal allegations must remain allegations
- avoid exact Kash Patel likeness
- avoid real FBI/Atlantic logos or official seals
- avoid alcohol props or defamatory visual assertions

Exact next human action: free disk space, rerun or locate the two saved GPT Image prompt outputs, copy them into the required PNG paths, then review for logos, likeness, alcohol props, and messy text.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
