#!/usr/bin/env python3
"""
Simple Cross-Model Image Tool
==============================

Grok 4.3 + GPT-5.4 working together for image generation
"""

import subprocess
import sys
from pathlib import Path

def generate_image(prompt):
    """Generate image with GPT-5.4"""
    print(f"🎨 Generating with GPT-5.4...")
    print(f"   Prompt: {prompt}")

    cmd = [
        "hermes", "chat",
        "--provider", "openai-codex",
        "-m", "gpt-5.4",
        "-q", prompt
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Save output for user to see
    print(output)

    return output

def analyze_image(image_path):
    """Analyze image with Grok 4.3"""
    print(f"\n👁️  Analyzing with Grok 4.3...")

    cmd = [
        "hermes", "chat",
        "--provider", "xai-oauth",
        "-m", "grok-4.3",
        "--image", image_path,
        "-q", "Describe this image in detail. What are the strengths and what could be improved?"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    print(output)

    return output

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 simple_cross_model.py 'your prompt here'")
        print("\nExample:")
        print('  python3 simple_cross_model.py "A cyberpunk city at sunset"')
        sys.exit(1)

    prompt = sys.argv[1]

    print("="*60)
    print("Cross-Model Image Generation")
    print("="*60)

    # Step 1: Generate
    print("\n--- STEP 1: Generate with GPT-5.4 ---")
    gen_output = generate_image(prompt)

    print("\n" + "="*60)
    print("⚠️  NOTE: Check the GPT-5.4 output above for the image path")
    print("="*60)
    print("\nOnce you have the image path, run:")
    print(f"  python3 simple_cross_model.py analyze /path/to/image.png")

    if len(sys.argv) >= 3 and sys.argv[1] == "analyze":
        image_path = sys.argv[2]
        print("\n--- STEP 2: Analyze with Grok 4.3 ---")
        analyze_image(image_path)

if __name__ == "__main__":
    main()
