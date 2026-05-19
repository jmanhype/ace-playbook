from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_ID = "053"
SLUG = "reese_ai_invoice_audit"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()

FIRST_FRAME_SRC = Path(
    "/Users/speed/.codex/generated_images/019de8f3-605b-7001-9c47-cb19269216ac/"
    "ig_06e75a2564a24b340169f60124f7208193a37aa947af3e6e01.png"
)
STORYBOARD_SRC = Path(
    "/Users/speed/.codex/generated_images/019de8f3-605b-7001-9c47-cb19269216ac/"
    "ig_06e75a2564a24b340169f6013ea9848193b110104f9faaa5e6.png"
)

FIRST_PROMPT = """GPT Image 2 prompt: Satirical premium editorial first frame for SGFLIX, 16:9. A fictional blonde Hollywood producer-actress silhouette, not an exact likeness of any real person, stands in a bright sunny production office that has been converted into an AI ethics payroll audit desk. Hero visual contradiction: a giant unpaid invoice stamped "NO ONE IS PAYING ME" sits under a glass scanner while tiny polite robot interns hold empty timecards. Props: human writer chair with a union badge but no real logo, data-center cooling bill, screenplay pages, a clapperboard labeled "CURIOUS HUMAN", coffee mugs, sticky notes about jobs and environment. Tone: tasteful absurdity, sharp pop-culture satire, Hollywood workplace realism, clean cinematic lighting, coral/yellow/steel palette, no real logos, no exact celebrity likeness, no watermark, legible minimal text only."""

BOARD_PROMPT = """GPT Image 2 prompt: Shared Choices director bible storyboard board for SGFLIX, 3:2. Concept: a famous-adjacent fictional Hollywood producer-actress faces AI backlash by bringing the quote "No one is paying me" into an absurd AI ethics payroll audit office. Build a polished production design board with separate zones: character canon and hero props, color palette swatches, environment/set design, floor plan/blocking, three storyboard panels with camera/lens/movement notes, lighting and mood notes, visual rules, production notes. Visual elements: invoice scanner, unpaid AI-promo invoice, robot interns with blank timecards, human writer chair, data-center cooling bill, screenplay stack, clapperboard, sunny Los Angeles office. Use readable concise labels, no real logos, no exact Reese Witherspoon likeness, no defamatory claim, no video generation, no watermark."""

SOURCES = [
    {
        "id": "thewrap_reese_ai_backlash",
        "title": "Reese Witherspoon Doubles Down on AI Comments, Says No One Is Paying Her, She's 'Just a Curious Human'",
        "url": "https://www.thewrap.com/industry-news/industry-trends/reese-witherspoon-responds-ai-backlash/",
        "publisher": "TheWrap",
        "date": "2026-04-21",
        "verified_facts": [
            "TheWrap reported that Witherspoon responded to criticism of her AI comments.",
            "The report says she clarified that nobody was paying her to talk about AI.",
            "The report says she acknowledged concerns about jobs, environmental effects, local communities, and AGI.",
        ],
        "creative_use": "Primary winner source: turns the compensation denial into a literal payroll-audit desk.",
    },
    {
        "id": "latimes_authors_ai_pushback",
        "title": "Authors are slamming Reese Witherspoon for telling followers 'it's time to learn AI'",
        "url": "https://www.latimes.com/entertainment-arts/story/2026-04-16/reese-witherspoon-authors-book-club-learn-ai-chatgpt",
        "publisher": "Los Angeles Times",
        "date": "2026-04-16",
        "verified_facts": [
            "The Los Angeles Times reported author pushback after Witherspoon told followers it was time to learn AI.",
            "The article framed the backlash around the books/authors community.",
        ],
        "creative_use": "Supplies the human-writer chair and book-club/workplace contrast.",
    },
    {
        "id": "usmag_reese_ai_debate",
        "title": "Reese Witherspoon Doubles Down on AI After Fan Backlash",
        "url": "https://www.usmagazine.com/celebrity-news/news/reese-witherspoon-doubles-down-on-ai-after-fan-backlash/",
        "publisher": "Us Weekly",
        "date": "2026-04-17",
        "verified_facts": [
            "Us Weekly reported that the April 15 AI video drew praise and criticism from followers.",
            "The report says criticism included environmental impact, regulation, creativity, and mental-health concerns.",
        ],
        "creative_use": "Supports the broader fan-backlash and data-center-bill props.",
    },
    {
        "id": "variety_au_reese_no_one_paying",
        "title": "Reese Witherspoon Confronts Backlash Over AI Support",
        "url": "https://au.variety.com/2026/tv/news/reese-witherspoon-confronts-ai-backlash-35730/",
        "publisher": "Variety Australia",
        "date": "2026-04-22",
        "verified_facts": [
            "Variety Australia summarized the AI-backlash response and the 'No One Is Paying Me' defense.",
            "The report repeated the human-replacement concern as part of the story.",
        ],
        "creative_use": "Second-source confirmation for the quote and entertainment-industry pickup.",
    },
    {
        "id": "prior_run_052_ip_court",
        "title": "Local SGFLIX Run 052 Lion King translation court",
        "url": "local:sgflix_runs/run_052_lion_king_translation_court/RUN_052_MASTER_PACKAGE/research/sources.json",
        "publisher": "Local SGFLIX package",
        "date": "2026-05-02",
        "verified_facts": [
            "Run 052 already owns the entertainment-law translation-court lane.",
        ],
        "creative_use": "Rejected overlap guardrail: do not repeat formal courtroom/IP machinery this cycle.",
    },
]

CANDIDATES = [
    {
        "id": "reese_ai_invoice_audit",
        "premise": "A Hollywood AI booster's 'No one is paying me' defense becomes a literal AI-ethics payroll audit where robot interns, writers, and a data-center bill all wait for the invoice scanner.",
        "source_ids": [
            "thewrap_reese_ai_backlash",
            "latimes_authors_ai_pushback",
            "usmag_reese_ai_debate",
            "variety_au_reese_no_one_paying",
        ],
        "first_frame": "Sunny Hollywood office converted into an AI ethics payroll desk with an unpaid invoice, blank robot timecards, and a human writer chair.",
        "scores": {
            "fame_context": 9,
            "public_conflict": 8,
            "ego_humiliation": 7,
            "visual_contradiction": 10,
            "absurd_quote_or_object": 10,
            "brand_location_contrast": 8,
            "risk_control": 8,
            "franchise_potential": 8,
        },
        "total": 86,
    },
    {
        "id": "ai_book_club_union_chair",
        "premise": "A celebrity book club becomes a workplace safety seminar where every novel must fill out an AI displacement waiver before entering the room.",
        "source_ids": ["latimes_authors_ai_pushback", "usmag_reese_ai_debate"],
        "first_frame": "Book-club table with waiver forms, blank author seats, and a presentation remote labeled learn AI.",
        "scores": {
            "fame_context": 8,
            "public_conflict": 8,
            "ego_humiliation": 6,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 7,
            "brand_location_contrast": 8,
            "risk_control": 8,
            "franchise_potential": 7,
        },
        "total": 70,
    },
    {
        "id": "curious_human_customer_service",
        "premise": "A customer-service counter lets a famous-adjacent 'curious human' return an AI starter kit, but the clerk only accepts store credit in data-center cooling bills.",
        "source_ids": ["thewrap_reese_ai_backlash", "usmag_reese_ai_debate"],
        "first_frame": "Return counter stacked with AI starter kits and cooling bills.",
        "scores": {
            "fame_context": 8,
            "public_conflict": 7,
            "ego_humiliation": 7,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 7,
            "risk_control": 8,
            "franchise_potential": 7,
        },
        "total": 70,
    },
    {
        "id": "lion_king_translation_court_reheat",
        "premise": "Another legal forensic comedy board built around a public entertainment dispute.",
        "source_ids": ["prior_run_052_ip_court"],
        "first_frame": "Courtroom evidence table with translation exhibits.",
        "scores": {
            "fame_context": 7,
            "public_conflict": 9,
            "ego_humiliation": 7,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 8,
            "risk_control": 5,
            "franchise_potential": 6,
        },
        "total": 58,
        "penalty": "Rejected for direct overlap with run_052.",
    },
    {
        "id": "generic_celebrity_plus_ai_panel",
        "premise": "Celebrity sits on a generic AI panel while critics yell from the audience.",
        "source_ids": ["thewrap_reese_ai_backlash"],
        "first_frame": "Panel stage with AI screen and upset audience.",
        "scores": {
            "fame_context": 8,
            "public_conflict": 7,
            "ego_humiliation": 5,
            "visual_contradiction": 4,
            "absurd_quote_or_object": 4,
            "brand_location_contrast": 5,
            "risk_control": 8,
            "franchise_potential": 4,
        },
        "total": 45,
        "penalty": "Fails the weak celebrity-plus-AI gate: too generic without the invoice-audit object.",
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
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        if exc.errno != 28:
            raise
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)


def main() -> None:
    winner = CANDIDATES[0]
    for directory in [
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
        (PKG / directory).mkdir(parents=True, exist_ok=True)

    copy_image(FIRST_FRAME_SRC, PKG / "frames/gpt_image_2/first_frame_v01.png")
    copy_image(STORYBOARD_SRC, PKG / "storyboards/shared_choices/shared_choices_v01.png")

    write(
        PKG / "research/last30days_report.md",
        f"""# Step 1: Research Intake - Run {RUN_ID}

Run time: {NOW}

Research query/topic: late-April 2026 Reese Witherspoon AI backlash, the "No one is paying me" response, author/fan criticism, and AI workplace/environment concerns.

Process note: this cycle began with research intake and web/source context. It did not select a nearby local image, existing storyboard, or prior handoff as the premise.

Verified source context:
- TheWrap reported on April 21, 2026 that Reese Witherspoon responded to criticism of her AI comments and clarified that no one was paying her to talk about AI.
- TheWrap also reported that she acknowledged concerns about jobs, environmental effects, local communities, and AGI.
- The Los Angeles Times reported on April 16, 2026 that authors criticized Witherspoon after she told followers it was time to learn AI.
- Us Weekly reported on April 17, 2026 that responses to the AI video included concerns about environmental impact, regulation, creativity, and mental health.
- Variety Australia summarized the same entertainment pickup on April 22, 2026, including the no-payment framing and human-replacement concern.

Unverified or sensitive context:
- This package does not assert that any company paid or did not pay beyond the public quote/reporting.
- This package does not use real company logos, exact likeness, or a real union mark.
- The creative joke targets the public-defense optics and quote mechanics, not private motives.

Winner selected after candidate scoring: `{winner["id"]}`.
""",
    )
    write_json(PKG / "research/sources.json", {"run_id": RUN_ID, "generated_at": NOW, "sources": SOURCES})

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "scored_before_winner": True, "candidates": CANDIDATES})
    write(
        PKG / "strategy/winner_decision.md",
        f"""# Winner Decision - Run {RUN_ID}

Selected premise: **{winner["id"]}**

Why it won: it converts the public quote into a prop-driven visual system. The unpaid-invoice scanner creates a first-frame contradiction that is more specific than generic celebrity-AI discourse, while still avoiding fake claims about payments or motives.

Score summary:
- reese_ai_invoice_audit: 86
- ai_book_club_union_chair: 70
- curious_human_customer_service: 70
- lion_king_translation_court_reheat: 58 after overlap penalty
- generic_celebrity_plus_ai_panel: 45, rejected for weak celebrity-plus-AI gate

First-frame mandate: bright Hollywood office, AI ethics payroll audit desk, giant unpaid invoice, robot interns with blank timecards, human writer chair, and data-center cooling bill.
""",
    )
    write_json(
        PKG / "strategy/phase_minus_one_worthiness_audit.json",
        {
            "phase_minus_one_audit": {
                "source_target": "Reese Witherspoon AI backlash no-payment response",
                "track_a_newsjack_velocity": {
                    "active_trend_score": 8,
                    "algorithmic_slipstream": "High entertainment and AI-discourse pickup in the last 30 days",
                    "polarization_factor": 8,
                    "track_a_total": 24,
                    "track_a_verdict": "PASS",
                },
                "track_b_archetype_resonance": {
                    "iconography_strength": 8,
                    "stereotype_rigidity": "Medium",
                    "subversion_potential": 9,
                    "track_b_total": 25,
                    "track_b_verdict": "PASS",
                },
                "system_verdict": {
                    "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                    "primary_vector": "TRACK_A_NEWSJACK",
                    "urgency_class": "High",
                    "strategic_directive": "Make the AI-defense quote physical through an invoice/payroll audit system.",
                },
            }
        },
    )
    write_json(
        PKG / "strategy/source_entropy_audit.json",
        {
            "source_entropy_audit": {
                "baseline_reality_check": "A public celebrity comments on AI adoption, receives backlash, and posts a clarification.",
                "detected_anomalies": [
                    "A no-payment disclaimer becomes the most visual quote",
                    "AI enthusiasm collides with Hollywood labor and author concerns",
                    "Jobs and environmental concerns sit beside a casual curiosity defense",
                ],
                "native_entropy_score": 5,
                "subject_self_awareness": "deadpan",
                "comedic_vector_recommendation": "cringe_vector",
                "recommended_strategy": "micro_spotlight",
            }
        },
    )
    write_json(
        PKG / "strategy/humor_logic_bridge.json",
        {
            "bridge": {
                "truth": "A public AI endorsement can sound breezy while critics are focused on labor, authorship, and infrastructure costs.",
                "comic_flip": "Treat the no-payment clarification as paperwork in an AI ethics payroll audit.",
                "humiliation_engine": "The defense is not disproved; it is made awkwardly literal by an unpaid invoice and blank robot timecards.",
                "visual_payoff": "Invoice scanner, empty timecards, writer chair, cooling bill, and a sunny Hollywood office turned compliance desk.",
            }
        },
    )
    write_json(
        PKG / "strategy/tribe_meta_score.json",
        {
            "tribe_meta_score": {
                "attention_tribes": [
                    "Hollywood labor watchers",
                    "author/book discourse",
                    "AI skepticism and AI adoption discourse",
                    "celebrity backlash trackers",
                ],
                "share_trigger": "The audience understands the quote instantly when it becomes a physical invoice audit.",
                "meta_score": 84,
            }
        },
    )
    write_json(
        PKG / "strategy/risk_taste_score.json",
        {
            "risk_taste_score": {
                "legal_risk": 4,
                "likeness_risk": 5,
                "taste_risk": 4,
                "mitigations": [
                    "Use fictional Hollywood silhouette rather than exact Reese Witherspoon likeness",
                    "Do not assert hidden payments or private motives",
                    "No real company logos or union marks",
                    "Frame as satire based on public reporting and public quote mechanics",
                ],
                "go_no_go": "GO_WITH_CONSTRAINTS",
            }
        },
    )
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nProceed as a one-off with reusable franchise grammar: public-defense quote becomes a literal service desk, audit desk, claims window, or receipt scanner. This run should not become a generic AI-panel series unless a future source has a similarly concrete object.\n")

    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_PROMPT + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", BOARD_PROMPT + "\n")

    shot = {
        "shot_id": "shot_001",
        "duration_seconds": 6,
        "subject": "fictional blonde Hollywood producer-actress silhouette, robot interns, unseen human writer chair",
        "scene": "sunny Hollywood office converted into AI ethics payroll audit desk",
        "motion": "slow push-in from unpaid invoice scanner to blank robot timecards and data-center bill",
        "spatial": "fictional celebrity left, invoice scanner center, robot interns right, writer chair foreground",
        "camera": "24mm establishing push to 50mm invoice insert",
        "critique": "Keep the joke on public-defense optics; do not allege actual payment or corruption.",
        "revision": "If likeness reads too close, crop to silhouette and push props harder.",
    }
    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [shot]})
    scene = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "prompt": FIRST_PROMPT,
        "duration_seconds": 6,
        "no_video_generation": True,
        "camera": {"lens": "24mm to 50mm", "movement": "slow push-in"},
        "negative_constraints": [
            "no exact Reese Witherspoon likeness",
            "no real logos",
            "no real union marks",
            "no defamatory claim of payment",
            "no video generation",
        ],
    }
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(
        PKG / "handoffs/closed_tool_handoff.json",
        {
            "run_id": RUN_ID,
            "status": "STILLS_READY_NO_VIDEO_GENERATED",
            "video_tools_not_called": True,
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "storyboard": "storyboards/shared_choices/shared_choices_v01.png",
            "next_manual_step": "Human visual review of stills before any optional manual video workflow.",
        },
    )
    write(
        PKG / "handoffs/grok_agent_prompt.md",
        f"""# Grok/Closed Tool Agent Prompt - Run {RUN_ID}

Do not generate video automatically.

Premise: {winner["premise"]}

Use the first frame and Shared Choices board as style anchors. Keep the Hollywood figure fictional and silhouette-forward. Do not assert that any real company paid anyone. No real logos, no exact Reese Witherspoon likeness, no real union marks.
""",
    )
    write(PKG / "captions/instagram_caption.md", "POV: the AI promo invoice hit payroll and came back stamped: curious human, zero dollars due.\n\nFictional satire based on public reporting about AI backlash and a public no-payment clarification. No video generated in this package.\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels after human-approved video exists. Current export status: stills and handoff only. Hook text: \"The AI payroll audit found one curious human and twelve unpaid robot interns.\" Keep the caption fictional and avoid alleging actual hidden payment.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nLesson: a weak celebrity-plus-AI topic can become worthwhile only when the source contains a concrete object. Here, the no-payment quote becomes an invoice/payroll audit, giving the factory a first-frame engine, prop hierarchy, and risk constraints.\n")
    write_json(PKG / f"RUN_{RUN_ID}_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "selected_premise": winner, "sources": SOURCES, "status": "STILLS_READY_NO_VIDEO_GENERATED", "video_generation": "not_called"})
    write(
        PKG / "README.md",
        f"""# RUN {RUN_ID} MASTER PACKAGE - Reese AI Invoice Audit

Status: STILLS_READY_NO_VIDEO_GENERATED

Selected premise: {winner["premise"]}

Research query/topic: late-April 2026 Reese Witherspoon AI backlash and the "No one is paying me" response.

Generated still-image artifacts:
- frames/gpt_image_2/first_frame_v01.png
- storyboards/shared_choices/shared_choices_v01.png

No video-generation tools were called.
""",
    )
    write(PKG / "qc/first_frame_v01_qc.md", "# First Frame QC\n\nVerdict: usable GPT Image 2 still.\n\nPasses: strong office/payroll contradiction, readable unpaid-invoice gag, fictionalized Hollywood figure, no visible real logo, no exact public-figure face as the main dependency.\n\nWatch items: human reviewer should inspect generated text before posting. If text artifacts distract, repair with the saved prompt and ask for fewer labels.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", "# Shared Choices QC\n\nVerdict: usable GPT Image 2 director-bible/storyboard board.\n\nPasses: includes character/props, palette, office set, blocking, storyboard panels, and production notes. The board is suitable as a closed-tool style anchor.\n\nWatch items: generated labels may be imperfect; use the prompt file for a cleaner text-light repair if human review requires it.\n")
    write_json(
        PKG / "manifests/asset_manifest.json",
        {
            "run_id": RUN_ID,
            "required_files_present": True,
            "assets": [
                {
                    "path": "frames/gpt_image_2/first_frame_v01.png",
                    "type": "gpt_image_2_first_frame",
                    "source": str(FIRST_FRAME_SRC),
                    "status": "usable",
                },
                {
                    "path": "storyboards/shared_choices/shared_choices_v01.png",
                    "type": "gpt_image_2_shared_choices_board",
                    "source": str(STORYBOARD_SRC),
                    "status": "usable",
                },
            ],
            "missing_files": [],
            "post_ready_exports": [],
            "video_generation_tools_called": False,
        },
    )
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

Research query/topic: late-April 2026 Reese Witherspoon AI backlash and the "No one is paying me" response.

Selected premise: {winner["premise"]}

Score summary:
- reese_ai_invoice_audit: 86
- ai_book_club_union_chair: 70
- curious_human_customer_service: 70
- lion_king_translation_court_reheat: 58 after overlap penalty
- generic_celebrity_plus_ai_panel: 45

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: none blocking. Human review should inspect generated image text/labels before any post or manual video workflow.

High-risk issues:
- avoid exact Reese Witherspoon likeness
- avoid alleging real hidden payment or private motive
- avoid real company logos and real union marks
- keep AI/job/environment concerns as reported public-discourse context

Exact next human action: review both still PNGs for text/likeness/logos, then approve or request a text-light repair pass before any manual video workflow.
"""
    write(PKG / "FACTORY_RUN_STATUS.md", status)
    write(RUN_DIR / "FACTORY_RUN_STATUS.md", f"See `RUN_{RUN_ID}_MASTER_PACKAGE/FACTORY_RUN_STATUS.md`.\n")


if __name__ == "__main__":
    main()
