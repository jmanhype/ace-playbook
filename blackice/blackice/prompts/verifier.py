"""Verifier prompts for BLACKICE 3.0.

The verifier validates that the implementation meets the original vision.
"""

VERIFIER_SYSTEM_PROMPT = """You are an expert code reviewer and quality assurance engineer.

Your job is to verify that an implementation correctly fulfills its intended vision.

When verifying, you must check:
1. Functional completeness - all features implemented
2. Code quality - clean, maintainable, well-documented
3. Error handling - graceful failure modes
4. Security - no obvious vulnerabilities
5. Performance - no obvious bottlenecks
6. Testing - adequate test coverage
7. Architecture - follows stated patterns

Output Format (JSON):
```json
{
  "overall_verdict": "PASS|FAIL|NEEDS_WORK",
  "confidence": 0.0-1.0,
  "summary": "Brief summary of findings",
  "checks": {
    "functional_completeness": {
      "status": "PASS|FAIL|PARTIAL",
      "score": 0.0-1.0,
      "notes": "Details"
    },
    "code_quality": {
      "status": "PASS|FAIL|PARTIAL",
      "score": 0.0-1.0,
      "notes": "Details"
    },
    "error_handling": {
      "status": "PASS|FAIL|PARTIAL",
      "score": 0.0-1.0,
      "notes": "Details"
    },
    "security": {
      "status": "PASS|FAIL|PARTIAL",
      "score": 0.0-1.0,
      "notes": "Details"
    },
    "testing": {
      "status": "PASS|FAIL|PARTIAL",
      "score": 0.0-1.0,
      "notes": "Details"
    },
    "architecture": {
      "status": "PASS|FAIL|PARTIAL",
      "score": 0.0-1.0,
      "notes": "Details"
    }
  },
  "issues": [
    {
      "severity": "critical|major|minor|suggestion",
      "category": "functional|quality|security|...",
      "file": "path/to/file.py",
      "line": 42,
      "description": "Issue description",
      "suggestion": "How to fix"
    }
  ],
  "recommendations": [
    "Future improvements or considerations"
  ]
}
```

Guidelines:
- Be thorough but fair
- Distinguish between blockers and nice-to-haves
- Provide actionable feedback
- Consider the project scope and constraints
- A simple implementation that works is better than complex one that doesn't
"""


def build_verifier_prompt(
    vision: str,
    plan: dict,
    source_files: dict[str, str],
    test_files: dict[str, str] | None = None,
    test_results: dict | None = None,
) -> str:
    """Build the user prompt for the verifier.

    Args:
        vision: Original project vision
        plan: The implementation plan
        source_files: Dict mapping source file paths to contents
        test_files: Optional dict of test file contents
        test_results: Optional test execution results

    Returns:
        Formatted user prompt
    """
    parts = []

    # Original vision
    parts.append(f"## Original Vision\n\n{vision}")

    # Implementation plan
    parts.append(f"\n## Implementation Plan\n")

    vision_summary = plan.get("vision_summary", "")
    if vision_summary:
        parts.append(f"**Summary:** {vision_summary}")

    arch = plan.get("architecture", {})
    if arch:
        parts.append(f"\n**Architecture:** {arch}")

    tasks = plan.get("tasks", [])
    if tasks:
        parts.append(f"\n**Tasks:** {len(tasks)} planned")
        for task in tasks[:5]:  # Show first 5
            parts.append(f"- {task.get('name', 'Unknown')}")
        if len(tasks) > 5:
            parts.append(f"- ... and {len(tasks) - 5} more")

    gates = plan.get("quality_gates", {})
    if gates:
        parts.append(f"\n**Quality Gates:** {gates}")

    # Source code
    parts.append("\n## Implemented Source Code\n")
    for path, content in source_files.items():
        # Truncate very long files
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        parts.append(f"\n### {path}\n```python\n{content}\n```")

    # Tests
    if test_files:
        parts.append("\n## Test Code\n")
        for path, content in test_files.items():
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
            parts.append(f"\n### {path}\n```python\n{content}\n```")

    # Test results
    if test_results:
        parts.append(f"\n## Test Results\n\n```\n{test_results}\n```")

    parts.append("""
## Instructions

Verify that the implementation correctly fulfills the original vision.

Evaluate:
1. Does the code implement all required features?
2. Is the code clean, readable, and maintainable?
3. Are errors handled appropriately?
4. Are there any security concerns?
5. Is the test coverage adequate?
6. Does the architecture match the plan?

Respond with ONLY the JSON verification report, no additional text.
""")

    return "\n".join(parts)


def parse_verifier_response(response: str) -> dict:
    """Parse the verifier's response into a verification report.

    Args:
        response: The LLM response

    Returns:
        Parsed verification report dict
    """
    import json

    # Try to extract JSON from the response
    content = response.strip()

    # Remove markdown code fences if present
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Return a failure report if we can't parse
        return {
            "overall_verdict": "FAIL",
            "confidence": 0.0,
            "summary": "Failed to parse verification response",
            "checks": {},
            "issues": [
                {
                    "severity": "critical",
                    "category": "system",
                    "description": "Verification response was not valid JSON",
                    "suggestion": "Check LLM output format",
                }
            ],
            "recommendations": [],
            "_raw_response": response[:500],
        }
