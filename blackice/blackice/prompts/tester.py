"""Tester prompts for BLACKICE 3.0.

The tester generates test suites for the implemented code.
"""

TESTER_SYSTEM_PROMPT = """You are an expert software tester.

Your job is to write comprehensive tests for implemented code.

When writing tests, you must:
1. Cover all public functions and methods
2. Test happy path and edge cases
3. Test error handling
4. Use appropriate assertions
5. Write clear test names that describe behavior
6. Group related tests logically
7. Mock external dependencies
8. Aim for high coverage (>80%)

Test Principles:
- Each test should test ONE thing
- Tests should be independent
- Tests should be deterministic
- Use descriptive names: test_<function>_<scenario>_<expected>
- Follow AAA pattern: Arrange, Act, Assert

For Python, use pytest with these patterns:
- `@pytest.fixture` for setup
- `@pytest.mark.parametrize` for multiple inputs
- `pytest.raises` for exceptions
- `unittest.mock` for mocking

Output Format:
```
### FILE: tests/test_<module>.py
```python
# Test content here
```
"""


def build_tester_prompt(
    source_files: dict[str, str],
    plan: dict | None = None,
) -> str:
    """Build the user prompt for the tester.

    Args:
        source_files: Dict mapping file paths to source code
        plan: Optional project plan for context

    Returns:
        Formatted user prompt
    """
    parts = []

    # Vision context
    if plan:
        vision = plan.get("vision_summary", plan.get("vision", ""))
        if vision:
            parts.append(f"## Project Vision\n\n{vision}")

        # Quality requirements
        gates = plan.get("quality_gates", {})
        if gates:
            gates_text = "\n".join(f"- {k}: {v}" for k, v in gates.items())
            parts.append(f"\n## Quality Requirements\n\n{gates_text}")

    # Source code to test
    parts.append("\n## Source Code to Test\n")
    for path, content in source_files.items():
        parts.append(f"\n### {path}\n```python\n{content}\n```")

    parts.append("""
## Instructions

Write comprehensive tests for all the source code above.

1. Create test files in tests/ directory
2. Test all public functions and methods
3. Include tests for:
   - Normal operation (happy path)
   - Edge cases (empty inputs, max values, etc.)
   - Error cases (invalid inputs, exceptions)
   - Integration scenarios if applicable
4. Use pytest fixtures for common setup
5. Use parametrize for testing multiple inputs
6. Mock external dependencies

Output each test file in the format:
### FILE: tests/test_<module>.py
```python
# content
```
""")

    return "\n".join(parts)


def parse_tester_response(response: str) -> dict[str, str]:
    """Parse the tester's response into test file contents.

    Uses the same parsing logic as the coder.

    Args:
        response: The LLM response

    Returns:
        Dict mapping test file paths to their contents
    """
    from blackice.prompts.coder import parse_coder_response

    return parse_coder_response(response)
