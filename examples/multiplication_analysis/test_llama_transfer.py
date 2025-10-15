#!/usr/bin/env python3
"""
Quick test: Transfer to Llama 3.1 8B Instruct via OpenRouter
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from multiplication_analysis import setup_database
from model_transfer import run_model_transfer_test

def main():
    """Test Llama 3.1 8B transfer."""
    print("\n🚀 ACE Framework - Llama 3.1 8B Transfer Test\n")

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("❌ OPENROUTER_API_KEY required")
        sys.exit(1)

    transfer_model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    print(f"Testing transfer to: {transfer_model}\n")

    try:
        session = setup_database()

        # Run transfer test
        run_model_transfer_test(
            session=session,
            transfer_model_name=transfer_model,
            num_problems=20,
            test_seed=888
        )

        session.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
