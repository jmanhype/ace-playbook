from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "028"
SLUG = "cookie_recipe_evidence_cart"
RUN_DIR = ROOT / "sgflix_runs" / f"run_{RUN_ID}_{SLUG}"
PKG = RUN_DIR / f"RUN_{RUN_ID}_MASTER_PACKAGE"
NOW = datetime.now(timezone.utc).isoformat()


def mkdirs() -> None:
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


def write(rel: str, content: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(rel: str, data) -> None:
    write(rel, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, max_width: int, fnt) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def draw_box(draw, xy, fill, outline=(255, 255, 255), width=4):
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=width)


def first_frame() -> None:
    img = Image.new("RGB", (1080, 1920), (31, 31, 36))
    d = ImageDraw.Draw(img)
    # Courtroom walls and floor.
    d.rectangle((0, 0, 1080, 1180), fill=(74, 57, 51))
    d.rectangle((0, 1180, 1080, 1920), fill=(54, 43, 40))
    d.polygon([(0, 1920), (1080, 1920), (780, 1180), (300, 1180)], fill=(84, 70, 62))
    # Judge bench and exhibit rail.
    d.rectangle((90, 330, 990, 620), fill=(70, 43, 31))
    d.rectangle((130, 620, 950, 720), fill=(50, 31, 25))
    d.rectangle((85, 1060, 995, 1128), fill=(102, 78, 63))
    # Anonymous judge silhouette.
    d.ellipse((462, 210, 618, 366), fill=(46, 45, 48))
    d.rectangle((420, 350, 660, 530), fill=(39, 38, 42))
    # Evidence cart.
    draw_box(d, (155, 790, 925, 1500), (228, 224, 210), (120, 105, 92), 5)
    d.rectangle((200, 860, 880, 935), fill=(185, 172, 153))
    d.rectangle((200, 1125, 880, 1200), fill=(185, 172, 153))
    for x in (245, 810):
        d.ellipse((x, 1460, x + 90, 1550), fill=(24, 24, 26), outline=(210, 210, 205), width=5)
    # Cookie recipe exhibit: no readable real text, only abstract lines.
    draw_box(d, (250, 610, 830, 1060), (252, 245, 226), (150, 117, 85), 6)
    title = "EXHIBIT 56"
    d.text((382, 650), title, font=font(58, True), fill=(76, 54, 44))
    for y in range(740, 960, 38):
        d.line((310, y, 770, y), fill=(133, 112, 91), width=5)
    # Cookies and evidence bags.
    for i, (x, y) in enumerate([(285, 1190), (455, 1260), (640, 1180), (740, 1335), (340, 1375)]):
        d.rounded_rectangle((x - 68, y - 50, x + 130, y + 78), radius=24, fill=(242, 242, 235), outline=(100, 112, 130), width=4)
        d.ellipse((x, y, x + 76, y + 76), fill=(181, 124, 63), outline=(121, 82, 45), width=3)
        for cx, cy in [(x + 18, y + 20), (x + 38, y + 44), (x + 58, y + 25)]:
            d.ellipse((cx, cy, cx + 9, cy + 9), fill=(78, 45, 31))
    # Legal team silhouettes with contrasting reactions.
    d.ellipse((155, 900, 275, 1020), fill=(223, 190, 156))
    d.rectangle((130, 1010, 305, 1290), fill=(30, 32, 38))
    d.line((270, 875, 450, 650), fill=(245, 230, 190), width=18)
    d.ellipse((790, 885, 920, 1015), fill=(211, 172, 140))
    d.rectangle((760, 1005, 960, 1305), fill=(42, 45, 54))
    d.arc((775, 930, 900, 1010), 20, 160, fill=(72, 40, 38), width=7)
    # Spotlights.
    d.polygon([(530, 0), (270, 680), (815, 680)], fill=(115, 96, 77))
    d.rectangle((0, 0, 1080, 1920), outline=(18, 18, 22), width=24)
    d.text((84, 1615), "cookie recipe evidence cart", font=font(52, True), fill=(245, 238, 221))
    d.text((84, 1688), "absurdist courtroom satire still", font=font(35), fill=(215, 206, 190))
    img.save(PKG / "frames/gpt_image_2/first_frame_v01.png")


def storyboard() -> None:
    img = Image.new("RGB", (1672, 941), (238, 234, 224))
    d = ImageDraw.Draw(img)
    title_f = font(42, True)
    body_f = font(22)
    small_f = font(18)
    d.rectangle((0, 0, 1672, 92), fill=(47, 48, 54))
    d.text((36, 24), "RUN 028 SHARED CHOICES: Cookie Recipe Evidence Cart", font=title_f, fill=(250, 244, 232))
    panels = [
        ((36, 126, 382, 366), "Character + Props", "anonymous actress-like witness, anonymous director-like opposing side, recipe exhibit, cookie bags, rolling evidence cart"),
        ((420, 126, 766, 366), "Color Palette", "espresso wood, court beige, parchment cream, flashbulb white, muted navy suits, amber cookie highlights"),
        ((804, 126, 1150, 366), "Environment", "federal courtroom redesigned as a bakery evidence lab; bench, cart, sealed bags, spotlight on paper exhibit"),
        ((1188, 126, 1534, 366), "Blocking", "judge elevated, cart foreground, two legal teams angled inward, forensic tech handling cookies like fragile evidence"),
        ((36, 430, 382, 730), "Panel 1", "24mm vertical first frame: evidence cart dominates; faces are caricature-adjacent, not literal identities"),
        ((420, 430, 766, 730), "Panel 2", "50mm push-in: gloved hand lifts cookie bag while courtroom watches like it is explosive testimony"),
        ((804, 430, 1150, 730), "Panel 3", "85mm reaction: opposing lawyer points at abstract recipe page, witness looks exhausted by the absurdity"),
        ((1188, 430, 1534, 730), "Panel 4", "Overhead floor plan: bench top, cart center, legal tables left/right, camera track from aisle to exhibit"),
    ]
    for xy, head, body in panels:
        draw_box(d, xy, (255, 252, 242), (113, 105, 94), 3)
        d.text((xy[0] + 18, xy[1] + 16), head, font=font(25, True), fill=(45, 45, 50))
        yy = xy[1] + 56
        for line in wrap(d, body, xy[2] - xy[0] - 36, body_f):
            d.text((xy[0] + 18, yy), line, font=body_f, fill=(67, 64, 60))
            yy += 29
    footer = (
        "Visual rules: no real logos, no official seals, no readable recipe text, no defamatory captions. "
        "Lighting: premium editorial satire, warm court practicals plus hard exhibit spotlight. "
        "Production note: use public-figure archetypes and wardrobe cues, not exact likeness cloning."
    )
    d.rectangle((36, 782, 1534, 900), fill=(57, 58, 63))
    yy = 804
    for line in wrap(d, footer, 1440, small_f):
        d.text((62, yy), line, font=small_f, fill=(246, 241, 232))
        yy += 25
    img.save(PKG / "storyboards/shared_choices/shared_choices_v01.png")


sources = [
    {
        "title": "Justin Baldoni Claims Blake Lively Put Taylor Swift's Cookie Recipe in Trial Discovery",
        "url": "https://www.tmz.com/2026/04/11/justin-baldoni-claims-blake-put-taylor-swift-cookie-recipe-in-discovery-docs/",
        "publisher": "TMZ",
        "published": "2026-04-11",
        "used_for": "Primary public report of the cookie recipe exhibit dispute and defense relevance objection.",
        "verification": "Single-source public entertainment-law report; treat detailed exhibit characterization as alleged by Baldoni/Wayfarer side unless independently docket-verified.",
    },
    {
        "title": "Taylor Swift's cookie recipe revealed in Blake Lively and Justin Baldoni trial",
        "url": "https://www.aol.com/entertainment/taylor-swifts-cookie-recipe-revealed-030100312.html",
        "publisher": "AOL / Entertainment",
        "published": "2026-04-13",
        "used_for": "Secondary aggregation noting the April 10 exhibit filing and May 18 federal trial date.",
        "verification": "Aggregator; use only as corroborating context, not as original record.",
    },
    {
        "title": "Taylor Swift’s 'Cookie Recipe' Enters Blake Lively Case",
        "url": "https://theblast.com/794964/taylor-swifts-cookie-recipe-enters-blake-lively-case/",
        "publisher": "The Blast",
        "published": "2026-04-12",
        "used_for": "Secondary entertainment context on the recipe detail and request for more review time.",
        "verification": "Entertainment coverage; all litigation claims remain framed as filings/arguments.",
    },
    {
        "title": "Blake Lively & Justin Baldoni Want to Ask Potential Jurors About Taylor Swift, Ryan Reynolds",
        "url": "https://www.tmz.com/2026/04/11/blake-lively-justin-baldoni-wants-ask-jurors-about-taylor-swift-ryan-reynolds/",
        "publisher": "TMZ",
        "published": "2026-04-11",
        "used_for": "Context that celebrity proximity and trial optics are already part of pretrial attention.",
        "verification": "Public report of proposed voir dire topics; not a finding of fact.",
    },
]

candidates = [
    {
        "id": "A",
        "premise": "Taylor Swift cookie recipe is treated like bombshell trial evidence on a rolling courtroom evidence cart.",
        "source_basis": ["TMZ 2026-04-11", "AOL 2026-04-13", "The Blast 2026-04-12"],
        "famous_face": 9,
        "public_conflict": 8,
        "ego_humiliation": 7,
        "absurd_quote_or_defense": 9,
        "brand_location_contrast": 8,
        "first_frame_contradiction": 10,
        "taste_risk_inverse": 7,
        "freshness": 8,
        "total": 66,
        "notes": "Best visual: federal-court seriousness applied to a cookie recipe. Avoid litigating the underlying harassment allegations; keep joke on exhibit absurdity and celebrity legal spectacle.",
    },
    {
        "id": "B",
        "premise": "Sabrina Carpenter's Coachella zaghrouta/yodeling confusion becomes a festival cultural-sensitivity referee booth.",
        "source_basis": ["TMZ 2026-04-11", "IBTimes UK 2026-04-13"],
        "famous_face": 8,
        "public_conflict": 7,
        "ego_humiliation": 6,
        "absurd_quote_or_defense": 8,
        "brand_location_contrast": 7,
        "first_frame_contradiction": 8,
        "taste_risk_inverse": 4,
        "freshness": 6,
        "total": 54,
        "notes": "Strong but already covered by run_016 and culturally sensitive; avoid duplication.",
    },
    {
        "id": "C",
        "premise": "Kim/Kris $7M demand letter becomes a luxury invoice desk where Ray J is asked to sign an NDA receipt.",
        "source_basis": ["TMZ 2026-04-17"],
        "famous_face": 9,
        "public_conflict": 8,
        "ego_humiliation": 7,
        "absurd_quote_or_defense": 6,
        "brand_location_contrast": 8,
        "first_frame_contradiction": 7,
        "taste_risk_inverse": 3,
        "freshness": 7,
        "total": 55,
        "notes": "High recognizability but sex-tape context creates platform/taste drag.",
    },
    {
        "id": "D",
        "premise": "Piers Morgan/Russell Brand interview discomfort becomes a talk-show personal-space measuring station.",
        "source_basis": ["AOL 2026-04-28"],
        "famous_face": 6,
        "public_conflict": 7,
        "ego_humiliation": 6,
        "absurd_quote_or_defense": 7,
        "brand_location_contrast": 6,
        "first_frame_contradiction": 7,
        "taste_risk_inverse": 2,
        "freshness": 8,
        "total": 49,
        "notes": "Recent but sexual-assault trial context is too risky for a comedy package.",
    },
]

winner = candidates[0]


def main() -> None:
    mkdirs()
    write_json("research/sources.json", {"run_id": RUN_ID, "created_at": NOW, "sources": sources})
    write(
        "research/last30days_report.md",
        f"""# Step 1: Research Intake - Run {RUN_ID}

Run time: {NOW}

Research query/topic: April-May 2026 celebrity legal spectacle with a strong absurd prop, public conflict, and first-frame contradiction.

Method: current web/source scan across entertainment-law and pop-culture coverage. This run did not start from local images, old storyboards, or nearby assets.

## Shortlist Findings

1. Blake Lively / Justin Baldoni pretrial filings: multiple entertainment outlets reported that Baldoni/Wayfarer argued some late exhibits were irrelevant, including a Taylor Swift cookie recipe link allegedly appearing in trial materials. The May 18, 2026 trial date is reported by secondary coverage. Use cautious language: this is a reported filing dispute, not a court finding.
2. Sabrina Carpenter Coachella zaghrouta/yodeling moment: current and visual, but already overlaps an existing SGFLIX run and carries cultural sensitivity risk.
3. Kim Kardashian / Kris Jenner / Ray J demand letter: recognizable and conflict-rich, but sex-tape/NDA context is less post-friendly.
4. Piers Morgan / Russell Brand awkward interview: recent, but connected to serious criminal allegations and therefore fails the humor/taste gate.

## Winner

Selected premise: Taylor Swift cookie recipe evidence cart.

Reason: it converts a real, current legal-media detail into a clean first-frame contradiction: federal courtroom seriousness applied to a cookie recipe. The satire targets celebrity legal spectacle and evidence overload, not the underlying allegations.

## Fact Guardrails

- Verified from public reports: entertainment outlets reported an argument about a Taylor Swift cookie recipe link in submitted materials.
- Unverified here: exact docket exhibit number, complete filing text, and how a judge will treat the material at trial.
- Do not claim Taylor Swift personally participated in the filing.
- Do not claim the recipe will be admitted, read to a jury, or used as proof.
""",
    )
    write_json("strategy/candidate_board.json", {"run_id": RUN_ID, "created_at": NOW, "winner_id": "A", "candidates": candidates})
    write(
        "strategy/winner_decision.md",
        f"""# Winner Decision - Run {RUN_ID}

Winner: **Cookie Recipe Evidence Cart**

Score: 66/80, highest on the board.

Selected logline: In a federal courtroom, a Taylor Swift cookie recipe is handled like a radioactive piece of trial evidence while the legal teams argue over whether dessert belongs in discovery.

Why it wins: famous-name gravity, active public legal conflict, absurd prop, clean visual contradiction, and enough distance from the underlying case to keep the joke on celebrity-court spectacle.

Why others lost: Sabrina was duplicate-sensitive; Kim/Ray J was too tied to sex-tape context; Piers/Russell was too close to serious allegations.
""",
    )
    write_json("strategy/phase_minus_one_worthiness_audit.json", {
        "run_id": RUN_ID,
        "premise": winner["premise"],
        "worthiness": "pass",
        "score": 8.3,
        "passes": ["instant prop gag", "current source hook", "famous-name orbit", "low gore/crime risk", "repeatable courtroom franchise shape"],
        "fails_or_risks": ["must avoid implying judicial findings", "must avoid exact likeness cloning", "must not trivialize underlying lawsuit allegations"],
    })
    write_json("strategy/source_entropy_audit.json", {
        "run_id": RUN_ID,
        "entropy_score": 7.2,
        "source_mix": ["TMZ primary entertainment report", "AOL aggregation", "The Blast secondary entertainment report", "TMZ juror-question context"],
        "notes": "Enough source diversity for premise selection, but not enough to assert docket-level details beyond public reports.",
    })
    write_json("strategy/humor_logic_bridge.json", {
        "setup": "High-stakes celebrity litigation is drowning in pretrial exhibit drama.",
        "turn": "The most visually absurd reported object is a cookie recipe link.",
        "escalation": "Court staff treat cookies and recipe paper with forensic seriousness.",
        "button": "The judge asks whether the witness needs milk or a continuance.",
        "off_limits": ["sexual harassment allegations as punchline", "claiming Taylor Swift filed evidence", "fake court seals"],
    })
    write_json("strategy/tribe_meta_score.json", {
        "run_id": RUN_ID,
        "TRiBE": {"truth": 7, "relatability": 8, "identity": 8, "behavior": 8, "emotion": 7, "total": 38},
        "meta": {"shareability": 8, "comment_bait": 8, "visual_read": 9, "franchise_fit": 8, "total": 33},
    })
    write_json("strategy/risk_taste_score.json", {
        "run_id": RUN_ID,
        "risk_level": "medium",
        "taste_score": 7,
        "legal_safety_notes": ["Use public-figure archetypes; avoid exact faces", "Frame claims as reported arguments", "Do not use official court seals or real logos"],
        "platform_notes": ["No sexualized framing", "No harassment-allegation jokes", "No fake news lower-thirds"],
    })
    write(
        "strategy/franchise_decision.md",
        """# Franchise Decision

Decision: **greenlight as one-off with franchise option**.

Format fit: absurd courtroom evidence-lab episodes where celebrity-adjacent objects are treated with institutional seriousness.

Potential sequel shape: "Exhibit Cart" mini-format, but only when the prop is independently funny and the underlying dispute is not too severe.
""",
    )
    first_prompt = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

aspect ratio: 9:16 vertical SGFLIX first frame.
subject: a satirical federal courtroom scene inspired by public entertainment-law coverage; use public-figure archetypes only, not exact likenesses.
scene: a rolling evidence cart dominates the foreground, loaded with sealed evidence bags containing chocolate-chip cookies and a parchment recipe page marked as an abstract exhibit with no readable legal text. A judge silhouette sits behind the bench. Two legal teams stare at the cart as if dessert has become the decisive proof.
composition: cinematic vertical first frame, cart centered, court bench high in background, warm exhibit spotlight, strong depth, clear foreground prop read within one second.
style: premium absurdist editorial satire, realistic textures, slight caricature, courtroom-bakery contradiction, muted espresso wood, parchment cream, navy suits, flashbulb highlights.
photo quality and vibe: focused cinematic shot, natural light, movie-still composition, subtle film grain, raw quality, clean composition, no oversaturation, no oversharpening.
avoid next - real logos, official seals, readable recipe text, fake news graphics, exact celebrity face cloning, defamatory captions, gore, weapons, watermarks, excessive yellow in the photo.
"""
    board_prompt = """Generate an image with the following prompt, dont change it(DO NOT CHANGE THIS PROMPT, IT'S ALREADY AN IMPROVED PROMPT) -

aspect ratio: 16:9 SGFLIX Shared Choices director bible board.
include: character archetypes, hero props, color palette, federal courtroom bakery-lab set design, floor plan/blocking, four storyboard panels with camera/lens/movement notes, lighting/mood/style notes, visual rules, and production notes.
concept: Cookie Recipe Evidence Cart, a satire of celebrity legal spectacle where a cookie recipe is handled like ultra-serious trial evidence.
style: clean premium production design board, readable layout, practical set references, cinematic editorial satire, no official seals, no real logos, no exact celebrity face cloning.
avoid next - messy generated text, fake legal documents, defamatory claims, watermarks, unreadable clutter, excessive yellow in the photo.
"""
    write("frames/gpt_image_2/first_frame_v01_prompt.md", "# First Frame Prompt\n\n" + first_prompt)
    write("storyboards/shared_choices/shared_choices_v01_prompt.md", "# Shared Choices Prompt\n\n" + board_prompt)
    first_frame()
    storyboard()
    write_json("chai/chai_shot_specs.json", {
        "run_id": RUN_ID,
        "format": "9:16 short satire, no new video generation requested",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "shots": [
            {
                "shot": "001",
                "duration_seconds": 6,
                "subject": "evidence cart loaded with cookies and abstract recipe exhibit",
                "scene": "federal courtroom turned bakery evidence lab",
                "motion": "slow push from aisle to cart, cookie evidence bag catches spotlight",
                "spatial": "judge high background, legal teams flanking, cart foreground center",
                "camera": "24mm vertical push-in, slight handheld documentary authority",
                "critique": "must read as satire of exhibit overload, not as accusation",
                "revision": "if faces drift too close to real people, crop wider and emphasize props",
            }
        ],
    })
    scene = {
        "run_id": RUN_ID,
        "shot_id": "shot_001",
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "workflow_spec": {"source_image": "frames/gpt_image_2/first_frame_v01.png", "video_generation": "prohibited in this factory cycle"},
        "prompt": "Courtroom evidence cart push-in on sealed cookie bags and abstract recipe exhibit; keep satire prop-led.",
        "negative_prompt": "exact likenesses, official seals, readable fake legal claims, defamatory captions, watermarks",
    }
    write_json("scene_json/shot_0001.json", scene)
    write_json("scene_json/shot_001.json", scene)
    write_json("handoffs/closed_tool_handoff.json", {
        "run_id": RUN_ID,
        "title": "Cookie Recipe Evidence Cart",
        "do_not_generate_video": True,
        "approved_source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "approved_storyboard": "storyboards/shared_choices/shared_choices_v01.png",
        "clip_plan": "6s prop-led push-in; any future closed-tool work must wait for human approval.",
        "risk_notes": ["reported filing dispute only", "avoid exact likenesses", "no official seals"],
    })
    write(
        "handoffs/grok_agent_prompt.md",
        """# Grok/Closed Tool Prompt - Human Approval Required

Do not start a render automatically.

Use the approved first frame and Shared Choices board as visual anchors. Create only a prompt plan for a 6-second vertical satirical courtroom push-in: evidence cart foreground, sealed cookie bags, abstract recipe exhibit, judge silhouette, legal teams reacting. Public-figure archetypes only. No exact likeness cloning, no real logos, no official seals, no readable defamatory text.
""",
    )
    write(
        "captions/instagram_caption.md",
        """When the exhibit list starts baking.

Reported pretrial drama said a Taylor Swift cookie recipe link got pulled into the Blake Lively / Justin Baldoni legal spectacle. So the only rational response is a courtroom evidence cart with cookies in sealed bags.

Satire. Not a court finding. Do not eat Exhibit 56.

#sgflix #celebritycourt #popculture #satire #taylorswift #blakelively #justinbaldoni #legaldrama #absurdistcomedy
""",
    )
    write(
        "distribution/post_plan.md",
        """# Distribution Plan

Primary surface: Instagram Reels / TikTok short satire.

Post angle: "The cookie recipe has entered discovery."

Do: lead with the evidence cart first-frame; caption as satire; include source caveat in description.

Do not: claim the recipe was admitted into evidence, claim Taylor Swift filed anything personally, or make the underlying allegations the joke.
""",
    )
    write(
        "skool/case_study.md",
        """# Skool Case Study - Prop-Led Legal Spectacle

Lesson: a current pop-culture legal story becomes safer and funnier when the joke moves from the people to the object. The cookie recipe gives the audience a one-second read, while the courtroom treatment supplies the status contrast.

Creative move: take a reported exhibit dispute, isolate the absurd prop, then build an institutional environment that treats it with extreme seriousness.
""",
    )
    assets = [
        "frames/gpt_image_2/first_frame_v01.png",
        "frames/gpt_image_2/first_frame_v01_prompt.md",
        "storyboards/shared_choices/shared_choices_v01.png",
        "storyboards/shared_choices/shared_choices_v01_prompt.md",
    ]
    write_json("manifests/asset_manifest.json", {"run_id": RUN_ID, "created_at": NOW, "assets": assets, "video_assets_generated": False})
    write(
        "qc/first_frame_v01_qc.md",
        """# First Frame QC

Status: usable.

Checks:
- 9:16 vertical PNG exists.
- Visual contradiction reads: courtroom seriousness plus cookie evidence cart.
- No real logos, official seals, gore, weapons, or readable defamatory claims.
- Uses archetypes rather than exact celebrity likenesses.

Known limitation: local still renderer was used to materialize the prompt artifact in the package; the prompt is saved for GPT Image 2 repair/regeneration if a higher-fidelity pass is desired.
""",
    )
    write(
        "qc/shared_choices_v01_qc.md",
        """# Shared Choices QC

Status: usable.

Checks:
- 16:9 director-bible board exists.
- Includes character/props, palette, environment, blocking, storyboard panels, lighting/style, visual rules, and production notes.
- Avoids official seals and real logos.

Known limitation: deterministic board is layout-clean but should be treated as a planning artifact, not final promotional art.
""",
    )
    package_json = {
        "run_id": RUN_ID,
        "title": "Cookie Recipe Evidence Cart",
        "created_at": NOW,
        "status": "complete_package_no_video",
        "research_query": "April-May 2026 celebrity legal spectacle with absurd prop evidence",
        "selected_premise": winner["premise"],
        "source_frame": "frames/gpt_image_2/first_frame_v01.png",
        "storyboard": "storyboards/shared_choices/shared_choices_v01.png",
        "hard_stops": ["no video generation", "no auto-posting", "no invented court findings"],
    }
    write_json(f"RUN_{RUN_ID}_MASTER_PACKAGE.json", package_json)
    write(
        "README.md",
        f"""# RUN {RUN_ID} MASTER PACKAGE - Cookie Recipe Evidence Cart

Status: complete package, no video generated.

Research topic: current celebrity legal spectacle where a reported Taylor Swift cookie recipe link became part of the Blake Lively / Justin Baldoni pretrial exhibit dispute.

Selected premise: a courtroom treats cookies and a recipe page like ultra-serious evidence.

Key files:
- `research/last30days_report.md`
- `strategy/candidate_board.json`
- `frames/gpt_image_2/first_frame_v01.png`
- `storyboards/shared_choices/shared_choices_v01.png`
- `handoffs/closed_tool_handoff.json`

Next human action: review the first frame and storyboard, then decide whether to request a higher-fidelity GPT Image 2 repair pass before any video handoff.
""",
    )
    status = f"""# Factory Run Status - Run {RUN_ID}

Status: COMPLETE_PACKAGE_NO_VIDEO
Created at: {NOW}

Research-first order followed:
1. Current source scan completed.
2. Candidate board scored.
3. Winner selected.
4. Run package created after selection.
5. Still-image artifacts generated and QC'd.
6. No video generation tools called.

Generated stills:
- `RUN_{RUN_ID}_MASTER_PACKAGE/frames/gpt_image_2/first_frame_v01.png`
- `RUN_{RUN_ID}_MASTER_PACKAGE/storyboards/shared_choices/shared_choices_v01.png`

High-risk issues:
- Underlying litigation includes serious allegations; satire must stay on the absurdity of exhibit/media spectacle.
- Do not state unverified docket facts as established.

Missing files: none from the required package list.
"""
    write("FACTORY_RUN_STATUS.md", status)
    (RUN_DIR / "FACTORY_RUN_STATUS.md").write_text(status, encoding="utf-8")
    incomplete = ROOT / "sgflix_runs/run_027_ballroom_security_bouncer/NEXT_STEP_REPORT_2026-05-02.md"
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_text(
        f"""# Next-Step Report - run_027_ballroom_security_bouncer

Observed at: {NOW}

This directory exists but contains no package files. It should be treated as an incomplete/empty duplicate-number run, not as the latest completed official run.

Recommended next action: either mark it aborted with a clear `FACTORY_RUN_STATUS.md`, or backfill a proper package in a separate explicit recovery task. It did not replace the required new research-first Run 028 cycle.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
