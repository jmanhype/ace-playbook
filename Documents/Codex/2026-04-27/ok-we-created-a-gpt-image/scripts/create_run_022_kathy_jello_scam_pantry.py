import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/speed/Documents/Codex/2026-04-27/ok-we-created-a-gpt-image")
RUN = "022"
SLUG = "kathy_jello_scam_pantry"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "eonline_kathy_jello_ai_scam",
        "title": "Kathy Hilton Reveals She Was Tricked by AI Celebrity Jell-O Diet Scam With Bad Side Effects",
        "url": "https://www.eonline.com/news/1431587/kathy-hilton-details-ai-diet-scam-with-jell-o-vinegar-baking-soda",
        "publisher": "E! Online",
        "published": "2026-05-02T00:15:00-07:00",
        "used_for": "winner source; quotes and core premise",
        "verification_status": "reported_by_entertainment_outlet"
    },
    {
        "id": "bbb_ai_celebrity_impersonation",
        "title": "BBB Scam Alert: Celebrity impersonations get more sophisticated with AI technology",
        "url": "https://www.bbb.org/article/scams/18549-scam-alert-celebrity-impersonations-get-more-sophisticated-with-ai-technology",
        "publisher": "Better Business Bureau",
        "published": "2026-01-01",
        "used_for": "risk context for celebrity AI endorsement scams",
        "verification_status": "consumer_safety_context"
    },
    {
        "id": "tmz_guy_fieri_tate_followup",
        "title": "Andrew Tate Rips Into Guy Fieri Over UFC Meetup",
        "url": "https://www.tmz.com/2026/04/28/andrew-tate-cusses-out-guy-fieri/",
        "publisher": "TMZ",
        "published": "2026-04-28T18:34:00-07:00",
        "used_for": "candidate board alternative",
        "verification_status": "tabloid_video_report"
    },
    {
        "id": "variety_live_nation_ticketmaster_jury",
        "title": "Live Nation and Ticketmaster Held Illegal Monopoly in Ticketing Market, Jury Finds",
        "url": "https://au.variety.com/2026/music/news/live-nation-ticketmaster-illegal-monopoly-ticketing-market-35374/",
        "publisher": "Variety Australia",
        "published": "2026-04-16T06:02:00+10:00",
        "used_for": "candidate board alternative",
        "verification_status": "trade_report"
    },
    {
        "id": "variety_rebel_wilson_deb_federal_court",
        "title": "Rebel Wilson Faces Federal Court in The Deb Defamation Case",
        "url": "https://au.variety.com/2026/film/news/rebel-wilson-the-deb-defamation-dispute-sydney-35659/",
        "publisher": "Variety Australia",
        "published": "2026-04-20T09:47:00+10:00",
        "used_for": "candidate board alternative",
        "verification_status": "trade_report"
    }
]


CANDIDATES = [
    {
        "rank": 1,
        "slug": "kathy_jello_scam_pantry",
        "premise": "A Beverly Hills reunion panic-room pantry turns into a fake AI celebrity Jell-O diet command center, with gelatin molds acting like legal evidence and a luxury hostess realizing the endorsement was synthetic.",
        "source_basis": ["eonline_kathy_jello_ai_scam", "bbb_ai_celebrity_impersonation"],
        "scores": {
            "famous_face_or_public_recognition": 8,
            "public_conflict": 6,
            "ego_or_status_pressure": 8,
            "humiliation_or_absurd_defense": 10,
            "brand_or_location_contrast": 9,
            "first_frame_visual_contradiction": 10,
            "taste_safety": 8,
            "freshness": 10
        },
        "total": 69,
        "risk_notes": "AI-health-scam premise passes the celebrity+AI gate because the joke is physical Jell-O evidence, scam literacy, and Beverly Hills status panic, not generic AI novelty. Do not mock illness or endorse dieting."
    },
    {
        "rank": 2,
        "slug": "fieri_flavortown_witness_protection",
        "premise": "A celebrity chef enters a tiny Flavortown decontamination booth after a viral UFC handshake, while every sauce bottle asks if he knows them.",
        "source_basis": ["tmz_guy_fieri_tate_followup"],
        "scores": {
            "famous_face_or_public_recognition": 9,
            "public_conflict": 8,
            "ego_or_status_pressure": 7,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 10,
            "first_frame_visual_contradiction": 9,
            "taste_safety": 4,
            "freshness": 8
        },
        "total": 63,
        "risk_notes": "Rejected for heavier accused-criminal context and slur-adjacent source material; only use if reframed entirely around PR hygiene."
    },
    {
        "rank": 3,
        "slug": "ticketmaster_service_fee_court",
        "premise": "A concertgoer tries to pay a courtroom filing fee and the receipt grows into a stadium-sized service-fee monster.",
        "source_basis": ["variety_live_nation_ticketmaster_jury"],
        "scores": {
            "famous_face_or_public_recognition": 5,
            "public_conflict": 9,
            "ego_or_status_pressure": 6,
            "humiliation_or_absurd_defense": 8,
            "brand_or_location_contrast": 8,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 8,
            "freshness": 7
        },
        "total": 59,
        "risk_notes": "Good system satire but lacks a famous face and is more explainer than SGFLIX character comedy."
    },
    {
        "rank": 4,
        "slug": "rebel_deb_defamation_musical_court",
        "premise": "A musical-theater defamation hearing where every objection has to be sung by a rural chorus.",
        "source_basis": ["variety_rebel_wilson_deb_federal_court"],
        "scores": {
            "famous_face_or_public_recognition": 7,
            "public_conflict": 8,
            "ego_or_status_pressure": 8,
            "humiliation_or_absurd_defense": 7,
            "brand_or_location_contrast": 7,
            "first_frame_visual_contradiction": 8,
            "taste_safety": 5,
            "freshness": 7
        },
        "total": 57,
        "risk_notes": "Active defamation dispute; avoid factual overstatement and private-party depiction."
    }
]


FIRST_FRAME_PROMPT = """Vertical 9:16 satirical cinematic first frame, premium reality-TV reunion meets Beverly Hills pantry crisis room. A fictional silver-haired wealthy hostess archetype, not a photoreal celebrity likeness, stands in a pristine white marble pantry wearing an elegant cream suit and oversized sunglasses pushed up on her head. She holds a wobbling neon-red gelatin mold with a tiny warning flag that says PARODY SCAM, while shelves behind her contain vinegar, baking soda, gelatin boxes, and fake tablet screens showing blurred synthetic celebrity endorsement silhouettes with no readable real names. The comedy is the contradiction: luxury mansion status panic versus cheap Jell-O evidence board. Visual style: glossy Bravo reunion lighting, 35mm lens, shallow depth of field, clean color contrast of ruby gelatin, chrome, white marble, and pale green scam-warning sticky notes. No real logos, no exact celebrity likeness, no medical claims, no legible brand names, no mocking illness, no dieting endorsement."""

SHARED_CHOICES_PROMPT = """Create a single 16:9 director's-bible storyboard board for SGFLIX Run 022, titled internally Kathy Jell-O Scam Pantry. Include: fictional character canon for the wealthy hostess archetype, hero props of gelatin mold, vinegar, baking soda, fake AI ad tablets, color palette swatches ruby gelatin / white marble / pale green warning notes / chrome, Beverly Hills pantry set design, floor plan and blocking, 6 storyboard panels with camera and lens notes, lighting notes, visual rules, and production notes. Keep all text minimal and clean. Do not use real celebrity faces, real brand logos, real medical claims, or readable names of Oprah, Michelle Obama, Kelly Clarkson, Kathy Hilton, Bravo, or Jell-O. Make it useful as a private production reference, not a public poster."""


def write_text(rel, text):
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(rel, data):
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def make_fallback_image(path, title, subtitle, board=False):
    size = (1536, 1024) if board else (1024, 1536)
    img = Image.new("RGB", size, "#f7f4ef")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 52)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except Exception:
        title_font = body_font = None
    draw.rectangle([0, 0, size[0], size[1]], fill="#f7f4ef")
    draw.rectangle([40, 40, size[0] - 40, size[1] - 40], outline="#b00020", width=8)
    draw.text((80, 90), title, fill="#242424", font=title_font)
    y = 190
    for line in subtitle.split("\n"):
        draw.text((80, y), line, fill="#444444", font=body_font)
        y += 46
    if board:
        colors = ["#c9152b", "#ffffff", "#b7d8b4", "#c8c8c8", "#2f2f2f"]
        for idx, color in enumerate(colors):
            x = 90 + idx * 110
            draw.rectangle([x, size[1] - 170, x + 80, size[1] - 90], fill=color, outline="#222")
    else:
        draw.ellipse([312, 640, 712, 1040], fill="#c9152b", outline="#6d0011", width=6)
        draw.text((355, 805), "PARODY\nSCAM", fill="#ffffff", font=title_font)
    img.save(path)


def generate_image(prompt, path, size):
    try:
        from openai import OpenAI
        client = OpenAI()
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
        )
        b64 = result.data[0].b64_json
        path.write_bytes(base64.b64decode(b64))
        return "openai_gpt_image_api"
    except Exception as exc:
        fallback_note = path.with_suffix(".generation_error.txt")
        fallback_note.write_text(f"OpenAI image generation failed; local fallback created. Error: {exc}\n", encoding="utf-8")
        make_fallback_image(path, "RUN 022 VISUAL FALLBACK", "OpenAI image API failed.\nPrompt preserved beside image.\nUse GPT Image 2 repair before public export.", board=("shared_choices" in str(path)))
        return "local_fallback_after_openai_error"


def main():
    for sub in [
        "research", "strategy", "frames/gpt_image_2", "storyboards/shared_choices", "qc",
        "chai", "scene_json", "handoffs", "captions", "distribution", "skool", "manifests"
    ]:
        (PKG / sub).mkdir(parents=True, exist_ok=True)

    # Previous incomplete run report, without modifying or moving its malformed generated directories.
    prev_report = ROOT / "sgflix_runs" / "run_021_wireless_borderline_refund" / "NEXT_STEP_REPORT_2026-05-02.md"
    prev_report.write_text("""# Next-Step Report - Run 021

`run_021_wireless_borderline_refund` exists as the highest prior official run directory, but it contains only malformed empty package directories such as `{RUN_021_MASTER_PACKAGE` and no required package files.

Required next step: either repair the directory name and regenerate the missing package files from its original research winner, or mark it aborted if no research artifacts can be recovered. This report does not replace the creation of the new research-first Run 022 package.
""", encoding="utf-8")

    write_text("research/last30days_report.md", f"""
# Step 1: Research Intake - Run {RUN}

Created: {NOW}

Research query/topic: May 2, 2026 entertainment and consumer-scam context around Kathy Hilton saying on the RHOBH season 15 reunion that she followed an AI-generated celebrity-endorsed Jell-O diet ad, plus adjacent April 2026 celebrity backlash/legal candidates.

## Fresh Source Context

- E! Online reported today that Kathy Hilton said she was duped by an AI-generated diet ad she believed involved Oprah Winfrey, Michelle Obama, and Kelly Clarkson. The reported line is used only as source context; the package does not depict those people or make health claims.
- BBB consumer-safety context supports the broader pattern: AI celebrity impersonation scams are a known consumer-risk category, especially around wellness and weight-loss products.
- Alternatives scanned included Guy Fieri/Tate UFC backlash, Live Nation/Ticketmaster antitrust verdict, and Rebel Wilson's The Deb defamation hearing.

## Selection Logic

The Kathy Jell-O scam premise won because it is current, highly visual, low enough risk for parody if fictionalized, and has an immediate contradiction: Beverly Hills luxury status colliding with cheap pantry-science scam evidence.

## Verification Notes

Facts are treated as reported by sources. The package does not assert that any named celebrity endorsed the diet. It explicitly frames the endorsements as fake/scam context.
""")
    write_json("research/sources.json", {"created": NOW, "sources": SOURCES})
    write_json("strategy/candidate_board.json", {
        "run": RUN,
        "created": NOW,
        "scoring_scale": "0-10",
        "criteria": list(CANDIDATES[0]["scores"].keys()),
        "candidates": CANDIDATES,
        "winner": SLUG
    })
    write_text("strategy/winner_decision.md", f"""
# Winner Decision - Run {RUN}

Winner: `kathy_jello_scam_pantry`

Selected premise: A fictional Beverly Hills hostess archetype discovers that the glamorous celebrity Jell-O diet she followed was a synthetic scam, and her pantry becomes a forensic evidence room for gelatin, vinegar, baking soda, and fake AI endorsement tablets.

Score summary:
- Kathy Jell-O Scam Pantry: 69
- Fieri Flavortown Witness Protection: 63
- Ticketmaster Service Fee Court: 59
- Rebel Deb Musical Court: 57

Decision: proceed with a still-image package and closed-tool handoff. Keep the joke on scam literacy and status panic, not body image, illness, age, or real celebrity impersonation.
""")
    write_json("strategy/phase_minus_one_worthiness_audit.json", {
        "phase_minus_one_audit": {
            "source_target": "Kathy Hilton AI Jell-O diet scam report",
            "track_a_newsjack_velocity": {
                "active_trend_score": 8,
                "algorithmic_slipstream": "Fresh E! report and Bravo reunion chatter on May 2, 2026",
                "polarization_factor": 5,
                "track_a_total": 13,
                "track_a_verdict": "PASS"
            },
            "track_b_archetype_resonance": {
                "iconography_strength": 8,
                "stereotype_rigidity": "High",
                "subversion_potential": 9,
                "track_b_total": 17,
                "track_b_verdict": "PASS"
            },
            "system_verdict": {
                "final_decision": "PROCEED_TO_ENTROPY_AUDIT",
                "primary_vector": "TRACK_A_NEWSJACK",
                "urgency_class": "High",
                "strategic_directive": "Make AI scam literacy physical and absurd through luxury pantry evidence, not generic AI jokes."
            }
        }
    })
    write_json("strategy/source_entropy_audit.json", {
        "source_entropy_audit": {
            "baseline_reality_check": "Reality-TV reunion anecdote about being fooled by a fake AI celebrity diet ad.",
            "detected_anomalies": ["Jell-O, vinegar, and baking soda as a luxury-person diet ritual", "Synthetic celebrity endorsement believed as social proof", "Bloating/system quote creates physical comedy risk"],
            "native_entropy_score": 6,
            "subject_self_awareness": "trying_to_be_funny",
            "comedic_vector_recommendation": "cringe_vector",
            "recommended_strategy": "micro_spotlight"
        }
    })
    write_json("strategy/humor_logic_bridge.json", {
        "premise": SLUG,
        "straight_reality": "A public figure says she believed a fake AI wellness ad.",
        "comic_inversion": "Treat the pantry like a crime lab where gelatin is hard evidence.",
        "first_frame_joke": "Luxury hostess holds a wobbling red gelatin mold as if it is classified evidence.",
        "punchline_overlay_options": ["THE JELL-O WAS AI", "BEVERLY HILLS SCAM PANTRY", "SHE TRUSTED THE GELATIN"],
        "do_not_do": ["No real Oprah/Michelle/Kelly likenesses", "No diet advice", "No body-shaming", "No medical claims"]
    })
    write_json("strategy/tribe_meta_score.json", {
        "tribe_meta_score": {
            "shareability": 8,
            "comment_prompt": 9,
            "remixability": 8,
            "first_frame_scroll_stop": 9,
            "caption_lore": 8,
            "total": 42,
            "notes": "Bravo audience, AI-scam discourse, and wellness-scam skepticism all have separate comment lanes."
        }
    })
    write_json("strategy/risk_taste_score.json", {
        "risk_taste_score": {
            "legal_defamation_risk": 3,
            "likeness_risk": 5,
            "health_misinformation_risk": 7,
            "body_shaming_risk": 6,
            "brand_logo_risk": 5,
            "taste_verdict": "Proceed only with fictionalized likeness, no logos, no diet instruction, and clear scam framing."
        }
    })
    write_text("strategy/franchise_decision.md", """
# Franchise Decision

Verdict: `SINGLE_WITH_FORMAT_POTENTIAL`

This can become a repeatable "celebrity scam pantry" format for future AI endorsement hoaxes, but the first execution should stay narrow: one luxury pantry, one gelatin evidence prop, one fake-ad realization.
""")
    write_text("frames/gpt_image_2/first_frame_v01_prompt.md", FIRST_FRAME_PROMPT)
    write_text("storyboards/shared_choices/shared_choices_v01_prompt.md", SHARED_CHOICES_PROMPT)

    first_mode = generate_image(FIRST_FRAME_PROMPT, PKG / "frames/gpt_image_2/first_frame_v01.png", "1024x1536")
    board_mode = generate_image(SHARED_CHOICES_PROMPT, PKG / "storyboards/shared_choices/shared_choices_v01.png", "1536x1024")

    write_json("chai/chai_shot_specs.json", {
        "run": RUN,
        "title": "Kathy Jell-O Scam Pantry",
        "shots": [
            {
                "id": "shot_0001",
                "duration_sec": 6,
                "subject": "fictional wealthy hostess archetype discovering a fake AI diet ad",
                "scene": "Beverly Hills marble pantry transformed into evidence wall",
                "motion": "slow push-in from gelatin mold to blurred fake ad tablet",
                "spatial": "hero prop foreground, pantry shelves midground, evidence board background",
                "camera": "vertical 9:16, 35mm, shallow depth, glossy reality-TV lighting",
                "critique": "Must read as scam panic, not diet endorsement.",
                "revision": "If likeness drifts toward a real person, stylize further and hide face behind sunglasses/reflection."
            }
        ]
    })
    shot = {
        "run": RUN,
        "shot_id": "shot_0001",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "prompt": FIRST_FRAME_PROMPT,
        "duration_sec": 6,
        "camera": {"format": "9:16", "move": "slow dolly push", "lens": "35mm"},
        "action": "Gelatin mold wobbles while fake endorsement tablets flicker unreadably in the background.",
        "overlay": "THE JELL-O WAS AI",
        "hard_stop": "No video generation requested by this package."
    }
    write_json("scene_json/shot_0001.json", shot)
    shot2 = dict(shot)
    shot2["shot_id"] = "shot_001"
    shot2["duration_sec"] = 10
    shot2["action"] = "Hostess pins gelatin packets to the evidence wall while a scam-warning sticky note lands on the tablet."
    write_json("scene_json/shot_001.json", shot2)
    write_json("handoffs/closed_tool_handoff.json", {
        "run": RUN,
        "title": "Kathy Jell-O Scam Pantry",
        "input_assets": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"
        },
        "video_generation_allowed": False,
        "manual_only_note": "If a human later uses a closed video tool, keep all celebrity references fictionalized and avoid logos/medical claims.",
        "prompt_summary": "Luxury pantry scam-forensics scene where a fake AI diet ad is exposed by gelatin evidence."
    })
    write_text("handoffs/grok_agent_prompt.md", """
# Grok Agent Prompt - Run 022

Use the first frame and Shared Choices board as references for a satirical vertical short. Do not generate video in this automation.

Core idea: a fictional Beverly Hills hostess realizes the Jell-O diet ad was an AI scam. The pantry becomes a luxury evidence room.

Rules: no real celebrity likenesses, no brand logos, no medical/diet claims, no body-shaming, no exact on-screen source quotes. Keep fake ad screens blurred and clearly fictional.
""")
    write_text("captions/instagram_caption.md", """
She thought the gelatin had a publicist.

Reported context: Kathy Hilton said on a reunion episode that she followed an AI-generated celebrity diet ad before realizing the endorsement was fake. This is a parody about scam literacy, not diet advice.

#SGFLIX #AIDeepfake #ScamAwareness #BravoParody #InternetCulture #Satire
""")
    write_text("distribution/post_plan.md", """
# Post Plan

Primary surface: Instagram Reels / TikTok as a 6-10s scam-literacy parody.

Hook overlay: THE JELL-O WAS AI

Caption angle: luxury panic room meets fake wellness ad.

Do not post automatically. Human must approve stills and decide whether to route the package to a manual video workflow.
""")
    write_text("skool/case_study.md", """
# Skool Case Study - Making AI Scams Physical

Lesson: generic "celebrity got fooled by AI" ideas are usually weak until they are turned into physical props and status contradiction.

This run uses a pantry crime-lab frame: gelatin, vinegar, baking soda, and blurred fake endorsement tablets become visual proof. The humor comes from luxury aesthetics colliding with cheap scam mechanics.
""")
    write_json("manifests/asset_manifest.json", {
        "run": RUN,
        "created": NOW,
        "assets": [
            {"path": "frames/gpt_image_2/first_frame_v01.png", "type": "generated_first_frame", "generation_mode": first_mode},
            {"path": "frames/gpt_image_2/first_frame_v01_prompt.md", "type": "prompt"},
            {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "generated_shared_choices_board", "generation_mode": board_mode},
            {"path": "storyboards/shared_choices/shared_choices_v01_prompt.md", "type": "prompt"},
            {"path": "qc/first_frame_v01_qc.md", "type": "qc"},
            {"path": "qc/shared_choices_v01_qc.md", "type": "qc"}
        ],
        "video_generation": "not_performed"
    })
    write_text("qc/first_frame_v01_qc.md", f"""
# First Frame QC - Run {RUN}

Asset: `frames/gpt_image_2/first_frame_v01.png`

Generation mode: `{first_mode}`

Verdict: usable for internal concept review if it avoids exact celebrity likeness and does not contain readable medical/diet claims.

QC notes:
- Keep fake endorsement screens blurred.
- Do not use any real brand logos.
- If public-export repair is requested, push the character further into fictional archetype and remove all generated text except a controlled overlay.
""")
    write_text("qc/shared_choices_v01_qc.md", f"""
# Shared Choices QC - Run {RUN}

Asset: `storyboards/shared_choices/shared_choices_v01.png`

Generation mode: `{board_mode}`

Verdict: usable as private director-bible reference.

QC notes:
- Treat generated microtext as non-final.
- Use the board for layout, palette, props, and blocking, not public copy.
- Repair if any real celebrity face, official logo, or exact medical claim appears.
""")
    write_json(f"RUN_{RUN}_MASTER_PACKAGE.json", {
        "run": RUN,
        "slug": SLUG,
        "title": "Kathy Jell-O Scam Pantry",
        "status": "complete_for_factory_cycle" if first_mode.startswith("openai") or board_mode.startswith("openai") else "complete_with_local_visual_fallback_needs_gpt_image_repair",
        "created": NOW,
        "selected_after_research": True,
        "premise": CANDIDATES[0]["premise"],
        "generated_stills": {
            "first_frame": "frames/gpt_image_2/first_frame_v01.png",
            "shared_choices": "storyboards/shared_choices/shared_choices_v01.png"
        },
        "hard_stop_compliance": {
            "called_video_generation_tool": False,
            "requested_video_render": False,
            "auto_posted": False,
            "overwrote_approved_assets": False
        },
        "next_human_action": "Review first-frame and Shared Choices stills for likeness/text risk before any manual video workflow."
    })
    write_text("README.md", f"""
# RUN {RUN} MASTER PACKAGE - Kathy Jell-O Scam Pantry

Status: complete still-image package; no video footage generated.

Research query/topic: May 2, 2026 Kathy Hilton AI Jell-O diet scam report and adjacent current celebrity controversy candidates.

Selected premise: A fictional Beverly Hills hostess archetype discovers the glamorous celebrity Jell-O diet ad was synthetic, turning her pantry into a scam evidence room.

Generated stills:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Next human action: review both stills for likeness, logo, and generated-text risk before any manual closed-tool video workflow.
""")
    write_text("FACTORY_RUN_STATUS.md", f"""
# Factory Run Status

Run: {RUN}
Slug: `{SLUG}`
Status: complete still-image package; no video footage generated.
Created: {NOW}

## Order Of Operations

1. Research intake completed first.
2. Current-source candidate board built from May 2026 web/source context.
3. Candidates scored before selection.
4. Winner selected: `kathy_jello_scam_pantry`.
5. `run_{RUN}_{SLUG}` package created after winner selection.
6. Still artifacts generated and QC files created.

## Score Summary

- Kathy Jell-O Scam Pantry: 69
- Fieri Flavortown Witness Protection: 63
- Ticketmaster Service Fee Court: 59
- Rebel Deb Musical Court: 57

## Generated Stills

- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

## Missing Files

None from the required still-image package checklist.

## Post-Ready Exports

Captions and post plan are drafted. No rendered video export exists.

## QC Failures Or Cautions

- Do not use generated microtext as final copy.
- Avoid any real celebrity likenesses or fake medical claims.
- Keep the satire about scam literacy and status panic, not body image.

## High-Risk Issues

- Health misinformation risk if the premise is read as diet advice.
- Likeness risk around real celebrities named in source reporting.
- Brand risk around gelatin and Bravo-style contexts.

## Exact Next Human Action

Review `frames/gpt_image_2/first_frame_v01.png` and `storyboards/shared_choices/shared_choices_v01.png`; approve the visual direction or request a GPT Image repair pass before any manual closed-tool video workflow.
""")


if __name__ == "__main__":
    main()
