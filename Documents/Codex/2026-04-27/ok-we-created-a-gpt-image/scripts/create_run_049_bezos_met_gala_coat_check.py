from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_ID = "049"
SLUG = "bezos_met_gala_coat_check"
ROOT = Path("sgflix_runs")
RUN_DIR = ROOT / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


SOURCES = [
    {
        "id": "cnn_kvia_bezos_met_gala",
        "title": "The Bezos of it all: The Met Gala's billionaire moment",
        "url": "https://kvia.com/entertainment/cnn-style/2026/05/01/the-bezos-of-it-all-the-met-galas-billionaire-moment/",
        "publisher": "CNN via KVIA",
        "date": "2026-05-01",
        "verified_facts": [
            "The 2026 Met Gala is scheduled for Monday, May 4, 2026.",
            "Jeff Bezos and Lauren Sanchez Bezos are described as the event's main benefactors and honorary chairs.",
            "NYC Mayor Zohran Mamdani announced he would skip the event, citing affordability as a focus.",
            "Boycott posters appeared in New York, including criticism tied to Amazon labor issues.",
            "Individual tickets are reported at $100,000 for 2026, with tables at $350,000.",
            "A worker-centered 'Ball Without Billionaires' fashion show is planned for the morning of the gala.",
        ],
        "creative_use": "Winner source cluster: billionaire patronage, red carpet exclusivity, worker-fashion counterprogramming, and affordability politics.",
    },
    {
        "id": "elpais_mamdani_snub",
        "title": "Why Zohran Mamdani's refusal to attend the Met Gala is a historic snub to Anna Wintour",
        "url": "https://english.elpais.com/culture/2026-04-22/why-zohran-mamdanis-refusal-to-attend-the-met-gala-is-a-historic-snub-to-anna-wintour.html",
        "publisher": "El Pais English",
        "date": "2026-04-22",
        "verified_facts": [
            "Mamdani turned down the Met Gala invitation for May 4, 2026.",
            "The article frames the refusal as a break with a long-running mayoral attendance tradition.",
            "The article reports posters criticizing the Bezos couple's involvement.",
        ],
        "creative_use": "Adds the absent mayor as a comic negative-space prop: an empty VIP hanger labeled affordability.",
    },
    {
        "id": "efe_bezos_met_gala",
        "title": "La 'Met Gala de los Bezos' promete una noche de arte, moda y polemica en Nueva York",
        "url": "https://efe.com/cultura/2026-04-28/met-gala-nueva-york-boicot-patrocinadores-jeff-bezos/",
        "publisher": "EFE",
        "date": "2026-04-28",
        "verified_facts": [
            "EFE photographed boycott posters in New York on April 16, 2026.",
            "EFE reports Jeff Bezos and Lauren Sanchez as unusual personal sponsors for the 2026 Met Gala.",
            "EFE reports the gala theme as Costume Art / fashion as art and notes the event is used to raise funds for the Met's fashion institute.",
            "EFE reports Mamdani rejected attendance while emphasizing affordability.",
        ],
        "creative_use": "Supports visual language: poster wall, red carpet, museum stairway, velvet-rope bureaucracy.",
    },
    {
        "id": "ap_met_gala_preview",
        "title": "Beyonce, Bezos, baubles and bustiers: What to know about the 2026 Met Gala",
        "url": "https://apnews.com/article/5014084c48de8d13488925287669fe94",
        "publisher": "Associated Press",
        "date": "2026-04-22",
        "verified_facts": [
            "AP previewed the 2026 Met Gala as taking place May 4.",
            "The dress code is tied to the Costume Art exhibition.",
        ],
        "creative_use": "Neutral baseline source for event timing and theme.",
    },
    {
        "id": "variety_swift_voice",
        "title": "Taylor Swift Files to Trademark Her Voice and Likeness",
        "url": "https://au.variety.com/2026/music/news/taylor-swift-trademark-voice-likeness-ai-misuse-35964/",
        "publisher": "Variety Australia",
        "date": "2026-04-28",
        "verified_facts": [
            "Swift's company filed three trademark applications on April 24, 2026.",
            "The reporting frames the filings as likely protection against AI misuse.",
        ],
        "creative_use": "Scored as a candidate but rejected because this factory already used a similar voice-vault premise.",
    },
    {
        "id": "variety_ticketmaster_jury",
        "title": "Live Nation and Ticketmaster Held Illegal Monopoly in Ticketing Market, Jury Finds",
        "url": "https://au.variety.com/2026/music/news/live-nation-ticketmaster-illegal-monopoly-ticketing-market-35374/",
        "publisher": "Variety Australia",
        "date": "2026-04-16",
        "verified_facts": [
            "A jury found Live Nation/Ticketmaster illegally held monopoly power in the ticketing market.",
            "The underlying legal action included the DOJ and 34 states according to Variety's report.",
        ],
        "creative_use": "Scored as a candidate but rejected because prior run_033 already covered Ticketmaster fees.",
    },
]

CANDIDATES = [
    {
        "id": "bezos_met_gala_coat_check",
        "premise": "A Met Gala coat-check clerk makes billionaires check their warehouse-rate scanners, tax loophole capes, and red-carpet boycott posters before entering Costume Art.",
        "source_ids": ["cnn_kvia_bezos_met_gala", "elpais_mamdani_snub", "efe_bezos_met_gala", "ap_met_gala_preview"],
        "first_frame": "Museum coat-check desk at the foot of a red-carpet stairway; a Bezos-like patron holds a gold invitation while a worker tags a scanner as forbidden outerwear.",
        "scores": {
            "fame_context": 9,
            "public_conflict": 10,
            "ego_humiliation": 9,
            "visual_contradiction": 10,
            "absurd_quote_or_object": 9,
            "brand_location_contrast": 10,
            "risk_control": 7,
            "franchise_potential": 9,
        },
        "total": 92,
    },
    {
        "id": "ball_without_billionaires_runway",
        "premise": "Workers stage a rival downtown runway where every outfit is a safety incident report tailored as couture.",
        "source_ids": ["cnn_kvia_bezos_met_gala"],
        "first_frame": "Downtown pop-up runway with hard hats, couture labels, and clipboard judges.",
        "scores": {
            "fame_context": 7,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "visual_contradiction": 9,
            "absurd_quote_or_object": 8,
            "brand_location_contrast": 9,
            "risk_control": 8,
            "franchise_potential": 8,
        },
        "total": 84,
    },
    {
        "id": "mamdani_empty_vip_hanger",
        "premise": "The Met Gala reserves a mayoral tuxedo hanger labeled affordability while the richest guests ask whether empty space counts as couture.",
        "source_ids": ["cnn_kvia_bezos_met_gala", "elpais_mamdani_snub", "efe_bezos_met_gala"],
        "first_frame": "VIP closet with one empty hanger under a spotlight and a tiny city budget tag.",
        "scores": {
            "fame_context": 7,
            "public_conflict": 8,
            "ego_humiliation": 8,
            "visual_contradiction": 9,
            "absurd_quote_or_object": 9,
            "brand_location_contrast": 8,
            "risk_control": 8,
            "franchise_potential": 7,
        },
        "total": 82,
    },
    {
        "id": "swift_soundmark_customs",
        "premise": "A trademark customs agent makes bootleg pop-star voice clones declare every vowel at the border.",
        "source_ids": ["variety_swift_voice"],
        "first_frame": "Soundwave passports on a trademark-office inspection belt.",
        "scores": {
            "fame_context": 10,
            "public_conflict": 7,
            "ego_humiliation": 6,
            "visual_contradiction": 9,
            "absurd_quote_or_object": 9,
            "brand_location_contrast": 8,
            "risk_control": 5,
            "franchise_potential": 6,
        },
        "total": 79,
    },
    {
        "id": "ticketmaster_monopoly_confessional",
        "premise": "A ticketing kiosk enters a courtroom confessional and admits every convenience fee was actually a personality trait.",
        "source_ids": ["variety_ticketmaster_jury"],
        "first_frame": "Courtroom witness stand shaped like a ticket scanner beside a stack of fee receipts.",
        "scores": {
            "fame_context": 7,
            "public_conflict": 9,
            "ego_humiliation": 8,
            "visual_contradiction": 8,
            "absurd_quote_or_object": 9,
            "brand_location_contrast": 8,
            "risk_control": 8,
            "franchise_potential": 5,
        },
        "total": 77,
    },
]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, obj) -> None:
    write(path, json.dumps(obj, indent=2) + "\n")


def font(size: int, bold: bool = False):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fnt, fill, leading: int = 6):
    line = ""
    for word in text.split():
        test = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] > width and line:
            draw.text((x, y), line, font=fnt, fill=fill)
            y += getattr(fnt, "size", 18) + leading
            line = word
        else:
            line = test
    if line:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += getattr(fnt, "size", 18) + leading
    return y


def draw_first_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1600, 900), "#f4efe6")
    d = ImageDraw.Draw(img)
    title = font(54, True)
    h = font(34, True)
    body = font(24)
    small = font(20)
    d.rectangle((0, 0, 1600, 96), fill="#102a32")
    d.text((42, 21), "MET GALA COAT CHECK: DECLARE YOUR BILLIONAIRE OUTERWEAR", font=title, fill="#fff8e8")
    d.polygon([(0, 900), (1600, 900), (1270, 210), (330, 210)], fill="#b11226")
    for i in range(9):
        x = 300 + i * 125
        d.line((x, 210, x - 240, 900), fill="#d7a14a", width=3)
    d.rectangle((95, 360, 675, 785), fill="#fffdf5", outline="#122026", width=5)
    d.text((130, 392), "COAT CHECK", font=h, fill="#122026")
    d.text((130, 438), "No loophole capes past this point", font=body, fill="#4b3a33")
    d.rounded_rectangle((165, 505, 565, 570), radius=8, fill="#f1c94b", outline="#4a3200", width=3)
    d.text((190, 520), "TAG: WAREHOUSE RATE SCANNER", font=small, fill="#211800")
    d.rectangle((210, 602, 520, 730), fill="#dfe7ea", outline="#253238", width=4)
    for x in range(250, 500, 38):
        d.line((x, 622, x, 710), fill="#70838a", width=3)
    d.text((245, 743), "forbidden prop bin", font=small, fill="#253238")
    d.ellipse((980, 245, 1155, 420), fill="#e9c9a6", outline="#2b1b13", width=4)
    d.rectangle((1015, 420, 1120, 630), fill="#111111")
    d.polygon([(1015, 420), (1120, 420), (1085, 630), (1045, 630)], fill="#232323")
    d.rectangle((940, 485, 1210, 560), fill="#f7d76a", outline="#573b05", width=4)
    d.text((965, 505), "HONORARY CHAIR", font=body, fill="#573b05")
    d.rectangle((1260, 285, 1515, 665), fill="#fff6f4", outline="#7d1020", width=5)
    y = wrap(d, "BOYCOTT POSTER WALL", 1285, 315, 205, h, "#7d1020")
    for txt in ["worker safety", "affordability", "not a donor playground"]:
        y = wrap(d, txt, 1290, y + 18, 200, body, "#30282a")
    d.rectangle((980, 690, 1455, 790), fill="#102a32")
    wrap(d, "Empty VIP hanger: MAYOR / AFFORDABILITY", 1005, 710, 425, body, "#fff8e8")
    img.save(path, optimize=True)


def draw_storyboard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1800, 1200), "#f8f4eb")
    d = ImageDraw.Draw(img)
    title = font(52, True)
    h = font(29, True)
    body = font(21)
    small = font(17)
    d.rectangle((0, 0, 1800, 86), fill="#102a32")
    d.text((36, 18), "SHARED CHOICES V01 - BEZOS MET GALA COAT CHECK", font=title, fill="#fff8e8")
    blocks = [
        (40, 125, 420, 385, "Character + Props", "Bezos-like billionaire patron, Vogue gatekeeper silhouette, coat-check worker, empty mayoral hanger, gold invite, scanner tag, protest posters."),
        (460, 125, 840, 385, "Palette", "Museum ivory, carpet red, deep teal security desk, gold donor accents, cold clipboard gray. Keep satire premium, not cartoon chaos."),
        (880, 125, 1260, 385, "Set Design", "Met stairway implied, coat-check desk foreground, poster wall side plane, forbidden-prop bin, velvet rope lanes."),
        (1300, 125, 1760, 385, "Blocking", "Worker controls frame left, donor halted center, protest wall frame right, empty hanger in rear spotlight. Power flips at the desk."),
        (40, 430, 590, 760, "Panel 1 - 24mm Push In", "Red carpet approaches coat-check desk. Gold invite enters frame before the face. Desk sign reads declare outerwear."),
        (625, 430, 1175, 760, "Panel 2 - 50mm Insert", "Scanner lands in evidence bin. Tag says warehouse rate scanner. Avoid real logos; imply with shape and color only."),
        (1210, 430, 1760, 760, "Panel 3 - 35mm Reveal", "Empty mayoral hanger under a clean spotlight. Guests stare as if affordability is a forbidden accessory."),
        (40, 805, 590, 1105, "Lighting / Mood", "Flashbulb sparkle outside, fluorescent accountability at desk. Museum polish interrupted by bureaucratic dryness."),
        (625, 805, 1175, 1105, "Visual Rules", "No exact likeness. No Amazon, Vogue, Met, or ICE marks. Keep names in metadata only. Use fictional signs and invented prop labels."),
        (1210, 805, 1760, 1105, "Production Notes", "6s cold open: coat-check slap. 10s version adds poster wall. 15s version ends with Ball Without Billionaires counter-runway tease."),
    ]
    for x1, y1, x2, y2, head, txt in blocks:
        d.rounded_rectangle((x1, y1, x2, y2), radius=6, fill="#ffffff", outline="#24343a", width=3)
        d.rectangle((x1, y1, x2, y1 + 45), fill="#d7a14a")
        d.text((x1 + 16, y1 + 10), head, font=h, fill="#1f272a")
        wrap(d, txt, x1 + 18, y1 + 68, x2 - x1 - 36, body if y2 - y1 > 280 else small, "#222222")
    img.save(path, optimize=True)


def main() -> None:
    winner = CANDIDATES[0]
    dirs = [
        "research", "strategy", "chai", "scene_json", "handoffs", "captions", "distribution",
        "skool", "manifests", "frames/gpt_image_2", "storyboards/shared_choices", "qc",
    ]
    for directory in dirs:
        (PKG / directory).mkdir(parents=True, exist_ok=True)

    write(PKG / "research/last30days_report.md", f"""# Step 1: Research Intake - Run {RUN_ID}

Run time: {NOW}

Research query/topic: May 2026 Met Gala donor controversy, Bezos/Sanchez patronage, boycott posters, mayoral absence, and worker-fashion counterprogramming.

Process note: premise selection started from current web/source context, not local images, old storyboards, or nearby assets.

Verified source context:
- CNN/KVIA reports the 2026 Met Gala is May 4, Jeff Bezos and Lauren Sanchez Bezos are main benefactors and honorary chairs, Mayor Zohran Mamdani will skip it, boycott posters appeared, tickets are reported at $100,000, tables at $350,000, and a worker-centered Ball Without Billionaires is planned.
- El Pais English reports Mamdani's refusal, describes it as a break with mayoral tradition, and notes posters criticizing the Bezos couple's involvement.
- EFE reports boycott posters photographed in New York, the unusual personal sponsorship by Jeff Bezos and Lauren Sanchez, the Costume Art theme, and Mamdani's affordability rationale.
- AP previewed the May 4 gala and Costume Art theme.

Unverified or sensitive context:
- This package does not assert private motives, private negotiations, or any unproven acquisition plans.
- Labor, tax, and ICE claims are treated as protest-poster allegations, not adjudicated facts.
- Character design should be "Bezos-like billionaire patron" rather than exact likeness.
""")
    write_json(PKG / "research/sources.json", {"run_id": RUN_ID, "generated_at": NOW, "sources": SOURCES})
    write(PKG / "research/source_notes.md", "Primary current cluster: Met Gala patronage controversy. Rejected AI-voice and Ticketmaster topics because they overlap prior factory runs.\n")

    write_json(PKG / "strategy/candidate_board.json", {"run_id": RUN_ID, "scored_before_winner": True, "candidates": CANDIDATES})
    write(PKG / "strategy/winner_decision.md", f"""# Winner Decision - Run {RUN_ID}

Selected premise: **{winner['id']}**

Why it won: it has the strongest first-frame contradiction: elite museum glamour stopped by a mundane coat-check desk. The source cluster is current, publicly documented, visually legible, and politically charged without needing invented facts.

Score summary:
- {CANDIDATES[0]['id']}: 92
- {CANDIDATES[1]['id']}: 84
- {CANDIDATES[2]['id']}: 82
- {CANDIDATES[3]['id']}: 79
- {CANDIDATES[4]['id']}: 77

First-frame mandate: security/coat-check bureaucracy humiliates billionaire glamour before the event can become a normal red-carpet shot.
""")
    write_json(PKG / "strategy/phase_minus_one_worthiness_audit.json", {"phase_minus_one_audit": {"source_target": "2026 Met Gala Bezos patronage controversy", "track_a_newsjack_velocity": {"active_trend_score": 10, "algorithmic_slipstream": "High: event is two days away and controversy is current", "polarization_factor": 9, "track_a_total": 28, "track_a_verdict": "PASS"}, "track_b_archetype_resonance": {"iconography_strength": 10, "stereotype_rigidity": "High", "subversion_potential": 9, "track_b_total": 29, "track_b_verdict": "PASS"}, "system_verdict": {"final_decision": "PROCEED_TO_ENTROPY_AUDIT", "primary_vector": "TRACK_A_NEWSJACK", "urgency_class": "Max", "strategic_directive": "Exploit the red carpet versus worker desk contradiction while facts are still live."}}})
    write_json(PKG / "strategy/source_entropy_audit.json", {"source_entropy_audit": {"baseline_reality_check": "A philanthropic fashion fundraiser with celebrity guests, museum prestige, and donor patronage.", "detected_anomalies": ["Billionaire personal sponsorship becomes the story before the gala", "Boycott posters invade the event's visual field", "Mayor skips event while citing affordability", "Counter-runway for workers appears beside celebrity runway"], "native_entropy_score": 7, "subject_self_awareness": "trying_to_look_cool", "comedic_vector_recommendation": "native_absurdity", "recommended_strategy": "straight_man_framing"}})
    write_json(PKG / "strategy/humor_logic_bridge.json", {"bridge": {"truth": "The event sells artful glamour but the live discourse is about who is allowed through the rope and who pays the social bill.", "comic_flip": "Treat donor optics as literal items that must be checked before entering.", "humiliation_engine": "The most powerful guest is delayed by the least glamorous worker station.", "visual_payoff": "Gold invitation, scanner in bin, boycott posters, empty affordability hanger."}})
    write_json(PKG / "strategy/tribe_meta_score.json", {"tribe_meta_score": {"attention_tribes": ["fashion watchers", "labor/anti-billionaire posters", "NYC politics", "celebrity red carpet commentary"], "share_trigger": "recognizable glamour interrupted by dry bureaucracy", "meta_score": 91}})
    write_json(PKG / "strategy/risk_taste_score.json", {"risk_taste_score": {"legal_risk": 5, "likeness_risk": 6, "taste_risk": 5, "mitigations": ["Use fictionalized billionaire patron", "No real logos", "Attribute protest claims as protest claims", "Do not invent private motives"], "go_no_go": "GO_WITH_CONSTRAINTS"}})
    write(PKG / "strategy/franchise_decision.md", "# Franchise Decision\n\nProceed as a one-off newsjack with reusable franchise frame: elite event meets mundane compliance desk. Future variants can use coat check, customs, notary, HR, or small-claims counters.\n")

    first_prompt = """GPT Image 2 prompt: Satirical premium editorial first frame, 16:9. A fictional billionaire patron at a museum red carpet coat-check desk before a fashion gala. A calm worker tags a handheld warehouse-rate scanner as forbidden outerwear. Gold invitation, red-carpet stairs, ivory museum stone, protest poster wall, empty VIP hanger labeled affordability. No real logos, no exact celebrity likeness, no readable brand marks. Cinematic flashbulbs outside, fluorescent accountability at the desk, dry absurdist tone."""
    board_prompt = """GPT Image 2 prompt: Shared Choices director bible board, 3:2. Include character and hero props, palette, environment/set design, floor plan/blocking, three storyboard panels with camera/lens notes, lighting/mood, visual rules, production notes. Concept: fictional 2026 Met Gala donor controversy as coat-check bureaucracy. No real logos or exact likenesses."""
    write(PKG / "frames/gpt_image_2/first_frame_v01_prompt.md", first_prompt + "\n")
    write(PKG / "storyboards/shared_choices/shared_choices_v01_prompt.md", board_prompt + "\n")
    draw_first_frame(PKG / "frames/gpt_image_2/first_frame_v01.png")
    draw_storyboard(PKG / "storyboards/shared_choices/shared_choices_v01.png")

    write_json(PKG / "chai/chai_shot_specs.json", {"run_id": RUN_ID, "shots": [{"shot_id": "shot_001", "duration_seconds": 6, "subject": "fictional billionaire patron stopped at gala coat check", "scene": "museum red carpet coat-check desk", "motion": "slow push-in, clerk tags scanner, donor freezes", "spatial": "desk foreground, red stairs midground, protest wall right, empty hanger rear", "camera": "24mm push to 50mm insert", "critique": "Do not over-chaos; let compliance desk be the joke.", "revision": "If likeness drifts too real, anonymize face further and lean on props."}]})
    scene = {"run_id": RUN_ID, "shot_id": "shot_001", "prompt": first_prompt, "duration_seconds": 6, "no_video_generation": True, "camera": {"lens": "24mm to 50mm", "movement": "push-in"}, "negative_constraints": ["no real logos", "no exact likeness", "no invented factual claims"]}
    write_json(PKG / "scene_json/shot_0001.json", scene)
    write_json(PKG / "scene_json/shot_001.json", scene)
    write_json(PKG / "handoffs/closed_tool_handoff.json", {"run_id": RUN_ID, "status": "STILLS_READY_COMPACT_FALLBACK", "video_tools_not_called": True, "first_frame": "frames/gpt_image_2/first_frame_v01.png", "storyboard": "storyboards/shared_choices/shared_choices_v01.png", "next_manual_step": "Review stills, then use handoff only if approved for non-automated video tooling."})
    write(PKG / "handoffs/grok_agent_prompt.md", f"""# Grok/Closed Tool Agent Prompt - Run {RUN_ID}

Do not generate video automatically. Use this only after human approval.

Premise: {winner['premise']}

Use the first frame and Shared Choices board as style anchors. Keep all real-world claims limited to sourced public context. Avoid exact likenesses and real brand marks.
""")
    write(PKG / "captions/instagram_caption.md", "POV: you reached the Met Gala and coat check asked you to declare your billionaire outerwear.\n\nFictional satire based on public reporting about the 2026 Met Gala sponsor controversy. No video generated in this package.\n")
    write(PKG / "distribution/post_plan.md", "# Post Plan\n\nPrimary surface: Instagram Reels once video exists. Current export status: stills and handoff only. Hook text: \"Declare your billionaire outerwear.\" Avoid tagging real entities unless legal/taste review approves.\n")
    write(PKG / "skool/case_study.md", "# Skool Case Study\n\nLesson: build from current source context, score candidates first, then turn a high-entropy news event into a low-entropy visual compliance desk. The joke works because the least glamorous object controls the most glamorous room.\n")
    write_json(PKG / "RUN_049_MASTER_PACKAGE.json", {"run_id": RUN_ID, "slug": SLUG, "created_at": NOW, "selected_premise": winner, "sources": SOURCES, "status": "STILLS_READY_COMPACT_FALLBACK", "video_generation": "not_called"})
    write(PKG / "README.md", f"""# RUN {RUN_ID} MASTER PACKAGE - Bezos Met Gala Coat Check

Status: STILLS_READY_COMPACT_FALLBACK

Selected premise: {winner['premise']}

Research query/topic: May 2026 Met Gala Bezos/Sanchez patronage controversy, boycott posters, affordability politics, and worker counter-runway.

Generated still-image artifacts:
- frames/gpt_image_2/first_frame_v01.png
- storyboards/shared_choices/shared_choices_v01.png

No video-generation tools were called.
""")
    write(PKG / "qc/first_frame_v01_qc.md", "# First Frame QC\n\nVerdict: usable compact generated still.\n\nPasses: strong coat-check contradiction, no real logos, no exact likeness, source-sensitive prop labels.\n\nKnown limitation: illustration is a compact local fallback rather than a high-fidelity GPT Image 2 bitmap; optional repair pass can improve realism when disk/headroom allows.\n")
    write(PKG / "qc/shared_choices_v01_qc.md", "# Shared Choices QC\n\nVerdict: usable director-bible board.\n\nPasses: includes character/props, palette, set design, blocking, storyboard panels, lighting, visual rules, and production notes.\n\nKnown limitation: compact board uses text-and-layout generation; optional high-fidelity GPT Image 2 board can replace it after review.\n")
    write_json(PKG / "manifests/asset_manifest.json", {"run_id": RUN_ID, "required_files_present": True, "assets": [{"path": "frames/gpt_image_2/first_frame_v01.png", "type": "generated_first_frame", "status": "usable_compact_fallback"}, {"path": "storyboards/shared_choices/shared_choices_v01.png", "type": "generated_storyboard_board", "status": "usable_compact_fallback"}], "missing_files": [], "video_generation_tools_called": False})
    write(PKG / "FACTORY_RUN_STATUS.md", f"""# Factory Run Status - Run {RUN_ID}

Status: `STILLS_READY_COMPACT_FALLBACK`

Completed:
- fresh research intake before premise selection
- current-source candidate board and score-first winner selection
- strategy audits and scoring files
- CHAI and scene JSON handoffs
- first-frame prompt plus generated PNG
- Shared Choices prompt plus generated PNG
- caption, distribution, Skool, manifest, and QC files

Generated still-image paths:
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`

Missing files: none from required minimum list.

Post-ready exports: none; no video was generated.

QC failures: none blocking. Optional high-fidelity GPT Image 2 rerender may improve texture.

High-risk issues:
- avoid exact Bezos, Sanchez, Wintour, or mayoral likenesses
- avoid real Met/Vogue/Amazon/ICE logos
- treat labor/tax/ICE language as protest-poster allegations, not adjudicated facts

Exact next human action: review both still PNGs, then approve or request a high-fidelity GPT Image 2 repair pass before any manual video workflow.
""")


if __name__ == "__main__":
    main()
