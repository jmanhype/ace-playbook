"""LLM Prompts for BLACKICE 3.0 Flywheel.

System prompts for each phase of the flywheel:
- Planner: Decomposes vision into actionable tasks
- Coder: Generates implementation code
- Tester: Creates test suites
- Verifier: Validates implementation against vision
"""

from blackice.prompts.coder import CODER_SYSTEM_PROMPT, build_coder_prompt
from blackice.prompts.planner import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from blackice.prompts.tester import TESTER_SYSTEM_PROMPT, build_tester_prompt
from blackice.prompts.verifier import VERIFIER_SYSTEM_PROMPT, build_verifier_prompt

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "build_planner_prompt",
    "CODER_SYSTEM_PROMPT",
    "build_coder_prompt",
    "TESTER_SYSTEM_PROMPT",
    "build_tester_prompt",
    "VERIFIER_SYSTEM_PROMPT",
    "build_verifier_prompt",
]
