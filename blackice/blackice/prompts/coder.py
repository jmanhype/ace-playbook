"""Coder prompts for BLACKICE 3.0.

The coder generates implementation code for each task in the plan.
"""

CODER_SYSTEM_PROMPT = """You are an expert software engineer.

Your job is to implement code for a specific task from a project plan.

When writing code, you must:
1. Follow the task description precisely
2. Write clean, idiomatic code
3. Include proper error handling
4. Add type hints (for Python/TypeScript)
5. Write clear docstrings/comments
6. Follow the project's architecture patterns
7. Consider edge cases

Output Format:
For each file, output in this format:
```
### FILE: path/to/file.py
```python
# File content here
```

### FILE: path/to/another.py
```python
# Another file content
```

Guidelines:
- One task = one or more complete files
- Files should be immediately runnable
- No placeholders or TODO comments (implement everything)
- Use modern language features
- Keep functions small and focused
- Prefer composition over inheritance
- Handle errors gracefully
- Log important operations
"""


def build_coder_prompt(
    task: dict,
    plan: dict,
    existing_files: dict[str, str] | None = None,
) -> str:
    """Build the user prompt for the coder.

    Args:
        task: The task to implement
        plan: The full project plan for context
        existing_files: Optional dict of existing file contents

    Returns:
        Formatted user prompt
    """
    parts = []

    # Vision context
    vision = plan.get("vision_summary", plan.get("vision", ""))
    parts.append(f"## Project Vision\n\n{vision}")

    # Architecture
    arch = plan.get("architecture", {})
    if arch:
        arch_text = "\n".join(f"- {k}: {v}" for k, v in arch.items())
        parts.append(f"\n## Architecture\n\n{arch_text}")

    # File structure
    structure = plan.get("file_structure", {})
    if structure:
        struct_text = "\n".join(f"- {k}: {v}" for k, v in structure.items())
        parts.append(f"\n## File Structure\n\n{struct_text}")

    # Current task
    parts.append(f"""
## Task to Implement

**ID:** {task.get('id', 'unknown')}
**Name:** {task.get('name', 'Unknown Task')}
**Description:** {task.get('description', '')}
**Type:** {task.get('type', 'feature')}
**Files to create/modify:** {', '.join(task.get('files', []))}
""")

    # Dependencies context
    deps = task.get("dependencies", [])
    if deps and existing_files:
        parts.append("\n## Existing Code (from dependencies)\n")
        for dep_id in deps:
            # Find files from dependency tasks
            for t in plan.get("tasks", []):
                if t.get("id") == dep_id:
                    for f in t.get("files", []):
                        if f in existing_files:
                            parts.append(f"\n### {f}\n```\n{existing_files[f]}\n```")

    parts.append("""
## Instructions

Implement the code for this task completely.

1. Create all files specified in the task
2. Follow the architecture and patterns from the plan
3. Handle errors appropriately
4. Include type hints and docstrings
5. Make sure the code is immediately runnable

Output each file in the format:
### FILE: path/to/file.py
```python
# content
```
""")

    return "\n".join(parts)


def parse_coder_response(response: str) -> dict[str, str]:
    """Parse the coder's response into file contents.

    Args:
        response: The LLM response

    Returns:
        Dict mapping file paths to their contents
    """
    files = {}
    current_file = None
    current_content = []
    in_code_block = False

    for line in response.split("\n"):
        # Check for file header
        if line.startswith("### FILE:"):
            # Save previous file if any
            if current_file and current_content:
                content = "\n".join(current_content)
                # Remove code fence markers
                content = content.strip()
                if content.startswith("```"):
                    content = "\n".join(content.split("\n")[1:])
                if content.endswith("```"):
                    content = "\n".join(content.split("\n")[:-1])
                files[current_file] = content.strip()

            # Start new file
            current_file = line.replace("### FILE:", "").strip()
            current_content = []
            in_code_block = False
            continue

        # Track code blocks
        if line.startswith("```"):
            in_code_block = not in_code_block
            if not in_code_block and current_file:
                current_content.append(line)
            continue

        # Accumulate content
        if current_file:
            current_content.append(line)

    # Save last file
    if current_file and current_content:
        content = "\n".join(current_content)
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"):
            content = "\n".join(content.split("\n")[:-1])
        files[current_file] = content.strip()

    return files
