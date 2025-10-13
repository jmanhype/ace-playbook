# Playbook Storage

This directory contains domain-specific playbooks following the ACE (Agentic Context Engineering) framework.

## Structure

Playbooks are organized by domain:

- `code-quality.json` - Coding strategies, anti-patterns, best practices
- `testing.json` - Test strategies, edge cases, integration patterns
- `ux.json` - User experience patterns, interaction guidelines
- `performance.json` - Optimization techniques, bottleneck solutions
- `architecture.json` - Interface design patterns, module organization

## Playbook Format

Each playbook is a JSON array of bullets:

```json
[
  {
    "content": "Always validate input parameters before processing",
    "section": "Strategies",
    "helpful_count": 5,
    "harmful_count": 0,
    "tags": ["validation", "security"]
  },
  {
    "content": "Avoid nested loops without considering algorithmic complexity",
    "section": "Pitfalls",
    "helpful_count": 0,
    "harmful_count": 3,
    "tags": ["performance", "algorithms"]
  }
]
```

## Usage

1. **Reading**: Load relevant playbook before starting feature work
2. **Updating**: After reflection, use Curator logic to merge new bullets
3. **Committing**: Commit playbook changes with descriptive messages like:
   - `playbook: add validation strategy for API inputs`
   - `playbook: record bottleneck solution for database queries`

## Maintenance

- **Deduplication**: Run periodically to merge semantically similar bullets (threshold ≥ 0.8)
- **Pruning**: Remove bullets with (helpful_count = 0 and harmful_count = 0) after 10+ iterations
- **Review**: Monthly audit to assess bullet effectiveness

## Constitution Reference

See `.specify/memory/constitution.md` for complete playbook governance rules.
