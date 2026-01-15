#!/usr/bin/env python3
"""
Prompt Generator for Letta MAS Autonomous Video Production

Generates prompts based on:
- User style preferences (dark moody, purple/blue, moonlit)
- Proven success patterns (ultra-short, stationary subjects)
- Theme rotation (fantasy creatures, landscapes, atmospheric)
"""

import random
import json
import requests
from datetime import datetime

LETTA_URL = "http://192.168.1.143:8283"
DIRECTOR_ID = "agent-22069f59-7a79-4890-bf4f-1f2a69696267"

# === USER STYLE (from user_style block) ===
STYLE = {
    "colors": ["dark", "shadowy", "moonlit", "twilight", "dusk"],
    "moods": ["moody", "atmospheric", "mysterious", "ethereal", "haunting"],
    "lighting": ["moonlit", "torchlit", "starlit", "firelit", "candlelit"],
    "avoid": ["bright", "cheerful", "vibrant", "sunny", "colorful"]
}

# === PROVEN SUCCESS PATTERNS ===
SUBJECTS = {
    "creatures": [
        "wolf", "raven", "owl", "panther", "fox", "stag", "eagle", "hawk",
        "phoenix", "dragon", "griffin", "serpent", "lion", "tiger", "bear",
        "crow", "bat", "moth", "spider", "scorpion"
    ],
    "fantasy": [
        "wizard", "witch", "sorcerer", "knight", "samurai", "warrior",
        "assassin", "ranger", "druid", "necromancer", "paladin"
    ],
    "elements": [
        "fire elemental", "water spirit", "shadow wraith", "frost giant",
        "storm herald", "earth golem", "wind dancer", "void walker"
    ]
}

POSES = {
    "stationary": [
        "standing in", "perched in", "resting in", "watching from",
        "guarding", "lurking in", "emerging from", "silhouetted against"
    ],
    "subtle_motion": [
        "prowling through", "stalking through", "gliding through",
        "drifting through", "hovering above", "circling"
    ]
}

ENVIRONMENTS = {
    "moonlit": [
        "moonlit forest", "moonlit ruins", "moonlit cliffs", "moonlit lake",
        "moonlit graveyard", "moonlit castle", "moonlit mountains"
    ],
    "dark": [
        "dark cavern", "shadowy temple", "ancient crypt", "misty swamp",
        "haunted manor", "forgotten shrine", "abandoned cathedral"
    ],
    "atmospheric": [
        "stormy peaks", "frozen tundra", "volcanic plains", "crystal cave",
        "ancient library", "mystic grove", "sacred ruins"
    ]
}

def generate_prompt() -> str:
    """Generate a single prompt matching user style."""
    # Pick category
    category = random.choice(list(SUBJECTS.keys()))
    subject = random.choice(SUBJECTS[category])

    # Add dark/moody modifier
    modifier = random.choice(STYLE["colors"])

    # Pick pose type (favor stationary)
    pose_type = random.choices(
        ["stationary", "subtle_motion"],
        weights=[0.7, 0.3]
    )[0]
    pose = random.choice(POSES[pose_type])

    # Pick environment (favor moonlit)
    env_type = random.choices(
        ["moonlit", "dark", "atmospheric"],
        weights=[0.5, 0.3, 0.2]
    )[0]
    environment = random.choice(ENVIRONMENTS[env_type])

    # Build prompt
    prompt = f"{modifier} {subject} {pose} {environment}"

    # Ensure under 60 chars (optimal)
    if len(prompt) > 60:
        # Simplify
        prompt = f"{modifier} {subject} in {environment}"

    return prompt

def generate_batch(count: int) -> list:
    """Generate a batch of unique prompts."""
    prompts = set()
    attempts = 0
    max_attempts = count * 3

    while len(prompts) < count and attempts < max_attempts:
        prompt = generate_prompt()
        prompts.add(prompt)
        attempts += 1

    return list(prompts)

def populate_queue(prompts: list) -> dict:
    """Send prompts to Director to populate production_queue."""
    prompt_list = "\n".join([f"- {p}" for p in prompts])

    message = f"""AUTONOMOUS BATCH PRODUCTION - LOAD QUEUE

Add these {len(prompts)} prompts to your production_queue.PENDING_VIDEOS:

{prompt_list}

Instructions:
1. Update production_queue block with all prompts
2. Set QUEUE_STATUS: active
3. Set BATCH_TOTAL: {len(prompts)}
4. Set BATCH_COMPLETE: 0
5. Begin processing - generate one video at a time
6. Apply user_style preferences to each
7. Store all results in archival memory
8. Update BATCH_COMPLETE after each video

This is autonomous mode - process continuously until queue is empty."""

    response = requests.post(
        f"{LETTA_URL}/v1/agents/{DIRECTOR_ID}/messages/",
        headers={"Content-Type": "application/json"},
        json={"messages": [{"role": "user", "content": message}]},
        timeout=300
    )
    return response.json()

def get_queue_status() -> dict:
    """Get current production queue status."""
    response = requests.get(
        f"{LETTA_URL}/v1/blocks/block-3adb1fce-3a68-4dde-b95d-f1e5f5369364/"
    )
    return response.json()

def refill_if_low(threshold: int = 10, refill_count: int = 20):
    """Auto-refill queue when it gets low."""
    print(f"[{datetime.now()}] Checking queue level...")

    status = get_queue_status()
    value = status.get("value", "")

    # Parse queue status
    if "PENDING_VIDEOS: []" in value or "QUEUE_STATUS: complete" in value:
        print("Queue empty, generating new prompts...")
        prompts = generate_batch(refill_count)
        populate_queue(prompts)
        print(f"Added {len(prompts)} new prompts to queue")
        return True

    # Try to find remaining count
    import re
    match = re.search(r'(\d+)\s*remaining', value)
    if match:
        remaining = int(match.group(1))
        if remaining < threshold:
            print(f"Queue low ({remaining}), adding more prompts...")
            prompts = generate_batch(refill_count)
            populate_queue(prompts)
            print(f"Added {len(prompts)} new prompts to queue")
            return True
        else:
            print(f"Queue has {remaining} pending, no refill needed")
    else:
        print("Could not parse queue status, skipping refill")

    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prompt Generator")
    parser.add_argument("--count", type=int, default=20, help="Number of prompts")
    parser.add_argument("--preview", action="store_true", help="Preview only, don't send")
    parser.add_argument("--status", action="store_true", help="Check queue status")
    parser.add_argument("--refill", action="store_true", help="Auto-refill if queue is low")
    parser.add_argument("--threshold", type=int, default=10, help="Refill threshold")

    args = parser.parse_args()

    if args.status:
        status = get_queue_status()
        print(json.dumps(status, indent=2))
        return

    if args.refill:
        refill_if_low(threshold=args.threshold, refill_count=args.count)
        return

    prompts = generate_batch(args.count)

    print(f"Generated {len(prompts)} prompts:")
    for i, p in enumerate(prompts, 1):
        print(f"  {i:2}. {p} ({len(p)} chars)")

    if not args.preview:
        print(f"\nSending to Director...")
        result = populate_queue(prompts)
        print("Queue populated!")
    else:
        print("\n[Preview mode - not sent]")

if __name__ == "__main__":
    main()
