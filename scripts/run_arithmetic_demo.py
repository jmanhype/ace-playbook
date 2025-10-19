"""Convenience entrypoint to run the arithmetic ACE demo."""

import os
from pathlib import Path


def main() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "arithmetic_learning_dspy.py"
    num_problems = os.getenv("NUM_PROBLEMS", "20")
    num_epochs = os.getenv("NUM_EPOCHS", "3")

    print("Running ACE arithmetic demo…")
    print(f"NUM_PROBLEMS={num_problems} NUM_EPOCHS={num_epochs}")
    os.execvp("python", ["python", str(example_path)])


if __name__ == "__main__":
    main()

