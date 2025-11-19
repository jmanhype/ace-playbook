# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements a **Spec Kit Workflow System** - a comprehensive specification-driven development framework that guides features from conception through implementation. The system enforces a structured workflow: constitution → specify → clarify → plan → checklist → tasks → analyze → implement.

## Key Architecture Principles

### Spec-Driven Development
- Features progress through distinct phases with quality gates at each step
- All features are organized in `specs/[###-feature-name]/` directories with numbered prefixes (e.g., `001-user-auth/`)
- Each feature directory contains: `spec.md` (WHAT/WHY), `plan.md` (HOW), `tasks.md` (executable steps), and optional artifacts like `data-model.md`, `contracts/`, `research.md`
- Specifications are technology-agnostic; technical decisions live in planning documents

### Branch and Directory Naming
- Feature branches follow the pattern `[###-short-name]` (e.g., `004-oauth-integration`)
- The numeric prefix (`###`) determines the feature directory: all branches with the same prefix (e.g., `004-fix-bug`, `004-add-feature`) work on the same spec in `specs/004-*/`
- When creating new features, check remote branches, local branches, AND spec directories to find the highest number for a given short-name

### Quality Gates
- **Specification Quality**: Validates that specs are testable, unambiguous, and free of implementation details
- **Checklist Validation**: Domain-specific checklists (security, accessibility, performance, etc.) must pass before implementation
- **Consistency Analysis**: Cross-artifact analysis ensures spec.md, plan.md, and tasks.md are aligned
- **Implementation Gates**: Tasks are checked off in tasks.md as `[X]` to track progress

### Task Organization
- Tasks are organized by **user story priority** (P1, P2, P3) to enable independent implementation and testing
- Each task follows the format: `- [ ] [TaskID] [P] [StoryLabel] Description with file path`
  - Example: `- [ ] T012 [P] [US1] Create User model in src/models/user.py`
  - `[P]` indicates parallelizable tasks
  - `[US1]`, `[US2]` etc. map to user stories from spec.md
- Task phases: Setup → Foundational → User Stories (in priority order) → Polish

## Common Commands

### Core Workflow Commands

All workflow commands are executed via slash commands that invoke bash scripts in `.specify/scripts/bash/`:

```bash
# Start a new feature (creates branch, spec directory, and initial spec.md)
/speckit.specify <feature description>

# Clarify underspecified requirements (max 3 targeted questions)
/speckit.clarify

# Create technical implementation plan
/speckit.plan

# Generate domain-specific quality checklists
/speckit.checklist

# Generate dependency-ordered executable tasks
/speckit.tasks

# Analyze cross-artifact consistency
/speckit.analyze

# Execute implementation following tasks.md
/speckit.implement

# Convert tasks to GitHub issues
/speckit.taskstoissues
```

### Full Workflow Orchestration

```bash
# Run complete workflow with all gates
/speckit-workflow-v2 <feature brief> [--domains 'security,accessibility'] [--strict] [--auto] [--parallel]

# Quick workflow without all quality gates
/speckit-orchestrate <feature brief> [--meta] [--parallel]
```

### Utility Scripts

```bash
# Check prerequisites and get feature paths
./.specify/scripts/bash/check-prerequisites.sh --json

# Check prerequisites for implementation (requires tasks.md)
./.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks

# Get just the paths without validation
./.specify/scripts/bash/check-prerequisites.sh --paths-only

# Create new feature manually (usually invoked by /speckit.specify)
./.specify/scripts/bash/create-new-feature.sh --json --number 5 --short-name "user-auth" "Add user authentication"
```

### Working with Features

```bash
# View current feature context
./.specify/scripts/bash/check-prerequisites.sh --paths-only

# Update agent context after making changes
./.specify/scripts/bash/update-agent-context.sh

# Initialize planning phase artifacts
./.specify/scripts/bash/setup-plan.sh
```

## Directory Structure

```
.
├── .claude/
│   ├── commands/           # Slash command definitions
│   │   ├── speckit.specify.md
│   │   ├── speckit.plan.md
│   │   ├── speckit.tasks.md
│   │   ├── speckit.implement.md
│   │   └── speckit-workflow-v2.md
│   └── hooks/              # Workflow hooks and logs
├── .specify/
│   ├── memory/
│   │   └── constitution.md # Project principles and constraints
│   ├── scripts/bash/       # Workflow automation scripts
│   │   ├── common.sh
│   │   ├── check-prerequisites.sh
│   │   ├── create-new-feature.sh
│   │   ├── setup-plan.sh
│   │   └── update-agent-context.sh
│   └── templates/          # Document templates
│       ├── spec-template.md
│       ├── plan-template.md
│       ├── tasks-template.md
│       ├── checklist-template.md
│       └── agent-file-template.md
└── specs/                  # Feature specifications (created per feature)
    └── [###-feature-name]/
        ├── spec.md         # User-focused specification (WHAT/WHY)
        ├── plan.md         # Technical plan (HOW)
        ├── tasks.md        # Executable implementation tasks
        ├── data-model.md   # Optional: Entity definitions
        ├── research.md     # Optional: Technical research and decisions
        ├── quickstart.md   # Optional: Integration scenarios
        ├── contracts/      # Optional: API contracts and interfaces
        └── checklists/     # Optional: Domain-specific quality checklists
```

## Development Workflow

### Creating a New Feature

1. Use `/speckit.specify <description>` - this will:
   - Analyze the description and generate a short-name (2-4 words)
   - Check all sources (remote branches, local branches, spec directories) for the highest number with that short-name
   - Create a new branch `[N+1]-short-name`
   - Initialize `specs/[N+1]-short-name/spec.md` with user scenarios, requirements, and success criteria
   - Run quality validation (max 3 clarification questions if needed)

2. If clarifications needed: `/speckit.clarify`

3. Create technical plan: `/speckit.plan` - generates architecture, tech stack, file structure

4. Generate quality checklists: `/speckit.checklist` - validates against security, accessibility, performance, etc.

5. Generate tasks: `/speckit.tasks` - creates dependency-ordered, executable task list organized by user story priority

6. Analyze consistency: `/speckit.analyze` - validates alignment across all artifacts

7. Execute implementation: `/speckit.implement` - follows tasks.md, marking tasks as `[X]` when complete

### Working with Existing Features

```bash
# Get current feature paths
eval $(./.specify/scripts/bash/check-prerequisites.sh --paths-only)

# The output includes:
# REPO_ROOT, BRANCH, FEATURE_DIR, FEATURE_SPEC, IMPL_PLAN, TASKS
```

### Key Conventions

- **Specs are non-technical**: No mention of frameworks, languages, or implementation details in spec.md
- **Plans are technical**: plan.md contains tech stack, architecture, libraries, and structure
- **Tasks are executable**: Each task in tasks.md has clear file paths and can be completed independently
- **User stories are prioritized**: P1 is MVP, P2+ are incremental enhancements
- **Each story is independently testable**: You should be able to implement and test just P1 for a viable MVP

## Environment Variables

```bash
# Override feature detection (useful for non-git repos)
export SPECIFY_FEATURE="005-my-feature"
```

## Important Notes

- All bash scripts handle both git and non-git repositories gracefully
- For arguments with single quotes (e.g., "I'm Groot"), use escape syntax: `'I'\''m Groot'` or use double quotes
- Quality gates (`--strict` flag) will halt execution if checklists or analysis fails
- Parallel execution (`--parallel` flag) runs independent tasks concurrently
- The constitution file (`.specify/memory/constitution.md`) defines project-wide principles that supersede all other practices

## Troubleshooting

### "Not on a feature branch" Error
Ensure your branch name follows the `###-feature-name` pattern, or set `SPECIFY_FEATURE` environment variable.

### Missing Prerequisites
Run `.specify/scripts/bash/check-prerequisites.sh --json` to see what's available. Different phases require different artifacts (plan.md required for tasks, tasks.md required for implementation).

### Multiple Spec Directories with Same Prefix
Only one spec directory should exist per numeric prefix. If you see this error, rename or consolidate directories.

### Tasks Not Executing in Correct Order
Check tasks.md for proper dependency markers and task IDs. Sequential tasks must have no `[P]` marker; parallelizable tasks should have `[P]`.
