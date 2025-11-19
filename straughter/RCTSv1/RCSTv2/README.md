# Spec Kit Workflow System

A comprehensive specification-driven development framework for Claude Code that guides features from conception through implementation with quality gates at every step.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Workflow Orchestrators](#workflow-orchestrators)
- [Core Commands](#core-commands)
- [Quality Gate Commands](#quality-gate-commands)
- [Workflow Phases Explained](#workflow-phases-explained)
- [Usage Examples](#usage-examples)
- [Directory Structure](#directory-structure)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

The Spec Kit Workflow System implements a structured approach to feature development:

1. **Constitution** - Establish project principles and standards
2. **Specify** - Define WHAT and WHY (user-focused, technology-agnostic)
3. **Clarify** - Resolve ambiguities with targeted questions
4. **Plan** - Define HOW (technical architecture, stack, structure)
5. **Checklist** - Validate requirement quality across domains
6. **Tasks** - Generate executable, dependency-ordered task list
7. **Analyze** - Ensure consistency across all artifacts
8. **Implement** - Execute the plan with progress tracking

### Key Benefits

- **Quality Gates**: Catch issues before implementation
- **Technology Separation**: Business requirements stay separate from technical decisions
- **Independent Testing**: Each user story can be tested independently
- **Parallel Execution**: Tasks can run concurrently where dependencies allow
- **Requirement Validation**: Checklists act as "unit tests for English"

## Quick Start

```bash
# 1. Install (one-time setup)
./scripts/setup-spec-kit.sh

# 2. Run a complete feature workflow with quality gates
/speckit-workflow-v2 "Add user authentication with OAuth2" \
  --domains "security,accessibility,performance" \
  --strict --auto --parallel

# 3. Or use the streamlined workflow (no quality gates)
/speckit-orchestrate "Add user dashboard" --parallel
```

## Installation

### Prerequisites

- **uv** (Python package installer): https://github.com/astral-sh/uv
- **Git** (optional, but recommended)
- **Claude Code** with slash command support

### Automatic Installation

Run the setup script from the project root:

```bash
./scripts/setup-spec-kit.sh
```

This will:
1. Install the Spec Kit CLI via `uv`
2. Initialize the project with `.specify/` directory structure
3. Set up templates and scripts
4. Verify the installation

### Manual Installation

If you prefer manual setup:

```bash
# Install Spec Kit CLI
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git

# Initialize in your project
cd /path/to/your/project
specify init --here --ai claude

# Verify installation
specify check
```

### Optional: TCHES Integration

To enable meta-prompting capabilities:

```bash
# Clone TCHES resources
git clone https://github.com/glittercowboy/taches-cc-resources.git /tmp/taches

# Copy meta-prompting commands
cp /tmp/taches/prompts/meta-prompting/*.md ~/.claude/commands/

# Copy skills
cp -r /tmp/taches/skills/* ~/.claude/skills/
```

## Workflow Orchestrators

### `/speckit-workflow-v2` - Full Workflow with Quality Gates

**Best for**: Production features, compliance-critical work, high-quality requirements

```bash
/speckit-workflow-v2 <feature-brief> [flags]
```

**Flags**:
- `--domains "security,privacy,accessibility,performance,observability,compliance,UX"` - Quality domains to validate
- `--strict` - Fail the run if any quality gate doesn't pass
- `--auto` - Proceed automatically between phases (unless blocking ambiguity detected)
- `--parallel` - Enable parallel task execution during implementation

**Phases**: Constitution → Specify → Clarify → Plan → **Checklist** → Tasks → **Analyze** → Implement

**Example**:
```bash
/speckit-workflow-v2 "Build a payment processing system with Stripe integration" \
  --domains "security,compliance,performance,accessibility" \
  --strict --auto --parallel
```

### `/speckit-orchestrate` - Streamlined Workflow

**Best for**: Prototypes, exploratory work, quick iterations

```bash
/speckit-orchestrate <feature-brief> [flags]
```

**Flags**:
- `--meta` - Use TCHES meta-prompting to refine the brief first
- `--parallel` - Enable parallel task execution during implementation

**Phases**: (Optional: Meta-Prompt) → Specify → Plan → Tasks → Implement

**Example**:
```bash
/speckit-orchestrate "Create admin dashboard for analytics" --meta --parallel
```

## Core Commands

### `/speckit.constitution`

Establish or update project principles and standards.

```bash
/speckit.constitution Create principles for code quality, testing standards, security, accessibility
```

**Creates**: `.specify/memory/constitution.md`

**When to use**:
- First time setting up a project
- When project standards need updating
- Before starting any features

---

### `/speckit.specify`

Create a feature specification (WHAT and WHY).

```bash
/speckit.specify Add user authentication with email/password and OAuth2
```

**Creates**:
- New feature branch `###-feature-name`
- `specs/###-feature-name/spec.md`
- Quality validation checklist

**Output includes**:
- User scenarios and journeys (prioritized P1, P2, P3...)
- Functional requirements
- Success criteria (measurable, technology-agnostic)
- Key entities and edge cases

**Important**: Specs are **technology-agnostic** - no mention of frameworks, languages, or implementation details.

---

### `/speckit.plan`

Create technical implementation plan (HOW).

```bash
/speckit.plan
```

**Creates**:
- `specs/###-feature-name/plan.md`
- `specs/###-feature-name/research.md` (Phase 0)
- `specs/###-feature-name/data-model.md` (Phase 1)
- `specs/###-feature-name/quickstart.md` (Phase 1)
- `specs/###-feature-name/contracts/` (Phase 1)

**Output includes**:
- Tech stack and dependencies
- Architecture and project structure
- Data models and API contracts
- Testing and observability strategy
- Performance budgets and constraints

---

### `/speckit.tasks`

Generate executable, dependency-ordered tasks.

```bash
/speckit.tasks
```

**Creates**: `specs/###-feature-name/tasks.md`

**Task format**:
```markdown
- [ ] T001 [P] [US1] Create User model in src/models/user.py
- [ ] T002 [US1] Implement UserService in src/services/user_service.py
- [ ] T003 Setup authentication middleware in src/middleware/auth.py
```

**Task organization**:
- **Phase 1**: Setup (project initialization)
- **Phase 2**: Foundational (blocking prerequisites)
- **Phase 3+**: User Stories in priority order (P1, P2, P3...)
- **Final Phase**: Polish & cross-cutting concerns

**Markers**:
- `[P]` = Parallelizable task (can run concurrently)
- `[US1]`, `[US2]` = User story label

---

### `/speckit.implement`

Execute the implementation following tasks.md.

```bash
/speckit.implement
```

**Process**:
1. Check checklist status (if checklists exist)
2. Load implementation context (tasks, plan, spec, contracts, etc.)
3. Verify/create ignore files (.gitignore, .dockerignore, etc.)
4. Execute tasks phase-by-phase
5. Mark completed tasks as `[X]` in tasks.md
6. Report progress and final status

---

### `/speckit.taskstoissues`

Convert tasks.md to GitHub issues.

```bash
/speckit.taskstoissues
```

**Creates**: GitHub issues with:
- Task dependencies
- User story labels
- Parallel execution markers
- File path references

## Quality Gate Commands

### `/speckit.clarify`

Resolve specification ambiguities with targeted questions.

```bash
/speckit.clarify
```

**Process**:
1. Scan spec for ambiguities across 10+ categories
2. Ask up to 5 highly targeted questions
3. Present recommended answers with options
4. Update spec.md with clarifications
5. Validate completeness

**Question categories**:
- Functional scope & behavior
- Domain & data model
- Interaction & UX flow
- Non-functional quality attributes
- Security & privacy
- Edge cases & failure handling

**Example interaction**:
```
Q1: Authentication Method
**Recommended:** Option B - OAuth2 + Email/Password (most flexible)

| Option | Description |
|--------|-------------|
| A | Email/Password only |
| B | OAuth2 + Email/Password |
| C | OAuth2 only |

Your choice: _
```

---

### `/speckit.checklist`

Generate domain-specific quality checklists ("unit tests for English").

```bash
/speckit.checklist
```

**Creates**: `specs/###-feature-name/checklists/[domain].md`

**Checklist domains**:
- Security
- Privacy
- Accessibility
- Performance
- Observability
- Compliance
- UX

**Important concept**: Checklists validate **requirement quality**, NOT implementation:

❌ **WRONG** (testing implementation):
```markdown
- [ ] CHK001 Verify login button works
- [ ] CHK002 Test password validation
```

✅ **CORRECT** (testing requirements):
```markdown
- [ ] CHK001 Are password requirements specified with concrete criteria? [Clarity, Spec §FR-2]
- [ ] CHK002 Are error messages defined for all authentication failure scenarios? [Completeness, Gap]
- [ ] CHK003 Is session timeout duration explicitly stated? [Gap]
```

---

### `/speckit.analyze`

Cross-artifact consistency validation.

```bash
/speckit.analyze
```

**Validates**:
- Spec.md ↔ Plan.md alignment
- Plan.md ↔ Tasks.md alignment
- Spec.md ↔ Tasks.md traceability
- Requirement coverage
- Inconsistencies and conflicts

**Output**:
- Consistency report
- List of gaps or conflicts
- Recommended fixes

## Workflow Phases Explained

### Phase 0: Constitution (One-time)

**Purpose**: Establish project-wide principles

**Outputs**: `.specify/memory/constitution.md`

**Principles typically include**:
- Code quality standards
- Testing requirements (TDD, coverage, etc.)
- Performance/SLO targets
- Security requirements
- Accessibility standards
- Deployment policies

---

### Phase 1: Specify (WHAT/WHY)

**Purpose**: Define user-facing requirements without technical details

**Key principle**: Technology-agnostic - focus on user value and business needs

**Outputs**:
- `spec.md` with user stories (P1, P2, P3...)
- Functional requirements
- Success criteria (measurable)

**Example user story**:
```markdown
### User Story 1 - Quick Login (Priority: P1)

Users can log in with email/password in under 10 seconds.

**Why this priority**: Core feature blocking all other functionality

**Independent Test**: Can fully test authentication flow without any other features

**Acceptance Scenarios**:
1. **Given** valid credentials, **When** user submits login, **Then** user is authenticated within 3 seconds
2. **Given** invalid password, **When** user submits login, **Then** clear error message shown
```

---

### Phase 2: Clarify (Optional Quality Gate)

**Purpose**: Remove ambiguities before planning

**Process**:
- AI scans spec across 10+ categories
- Identifies critical gaps
- Asks up to 5 targeted questions
- Updates spec with answers

**When to use**:
- Spec has `[NEEDS CLARIFICATION]` markers
- High-stakes features (security, compliance)
- Complex domain with many edge cases

---

### Phase 3: Plan (HOW)

**Purpose**: Define technical approach

**Key principle**: Technology-specific - frameworks, languages, architecture

**Outputs**:
- `plan.md` - Tech stack, architecture, structure
- `research.md` - Technical decisions and alternatives
- `data-model.md` - Entities and relationships
- `contracts/` - API specifications
- `quickstart.md` - Integration scenarios

**Example structure**:
```markdown
## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic
**Storage**: PostgreSQL 15
**Testing**: pytest, pytest-asyncio
**Performance Goals**: <200ms p95 latency, 1000 req/s
```

---

### Phase 4: Checklist (Quality Gate)

**Purpose**: Validate requirement quality across domains

**Process**:
1. Generate domain-specific checklists
2. Check each requirement for completeness, clarity, consistency
3. Identify gaps, ambiguities, conflicts
4. Loop back to clarify/update if needed

**In strict mode**: Blocks progression until all items pass

**Example items**:
```markdown
## Security

- [ ] CHK001 Are authentication requirements specified for all protected endpoints? [Coverage, Spec §FR-10]
- [ ] CHK002 Is password hashing algorithm documented? [Gap]
- [ ] CHK003 Are session management requirements defined? [Completeness, Spec §FR-12]
```

---

### Phase 5: Tasks (Executable Plan)

**Purpose**: Break down work into concrete, ordered tasks

**Key principle**: Organized by user story priority for independent implementation

**Task format**:
```markdown
## Phase 3: User Story 1 - Quick Login (P1)

**Goal**: Users can authenticate via email/password

**Independent Test**: Complete authentication flow works end-to-end

### Implementation

- [ ] T010 [P] [US1] Create User model in src/models/user.py
- [ ] T011 [P] [US1] Create AuthService in src/services/auth.py
- [ ] T012 [US1] Implement login endpoint in src/api/auth.py
- [ ] T013 [US1] Add authentication middleware in src/middleware/auth.py
```

---

### Phase 6: Analyze (Quality Gate)

**Purpose**: Ensure consistency across all artifacts

**Validates**:
- All user stories have corresponding tasks
- All requirements are addressed
- No conflicts between spec and plan
- Technical plan supports all requirements

**In strict mode**: Blocks implementation until clean

---

### Phase 7: Implement (Execution)

**Purpose**: Execute the plan

**Process**:
1. Verify prerequisites (checklists complete)
2. Setup project structure and ignore files
3. Execute tasks phase-by-phase
4. Mark tasks as `[X]` when complete
5. Report progress

**Execution rules**:
- Sequential tasks run in order
- Parallel `[P]` tasks can run concurrently
- Tasks affecting same files run sequentially
- Validation checkpoints after each phase

## Usage Examples

### Example 1: Production Feature with Full Quality Gates

```bash
/speckit-workflow-v2 "Build payment processing with Stripe: \
  - Support credit cards and ACH \
  - Handle webhooks for async events \
  - Store transaction history \
  - Admin dashboard for refunds" \
  --domains "security,compliance,performance,accessibility,observability" \
  --strict --auto --parallel
```

**What happens**:
1. Creates spec with user stories prioritized
2. Asks clarifying questions about PCI compliance, data retention, etc.
3. Generates technical plan with Stripe SDK integration
4. Runs security, compliance, performance, accessibility, observability checklists
5. If any checklist fails in strict mode → stops with actionable report
6. Generates tasks organized by user story (P1: basic payments, P2: webhooks, P3: admin dashboard)
7. Analyzes consistency across spec/plan/tasks
8. If analysis passes → implements with parallel execution where possible

---

### Example 2: Quick Prototype

```bash
/speckit-orchestrate "Create a simple blog with posts and comments" --parallel
```

**What happens**:
1. Creates spec with basic requirements
2. Generates technical plan
3. Generates tasks
4. Implements immediately (no quality gates)

---

### Example 3: Enhanced with Meta-Prompting

```bash
/speckit-orchestrate "I want users to be able to share their progress" --meta --parallel
```

**What happens**:
1. TCHES meta-prompt asks clarifying questions to sharpen the brief
2. Refined brief flows into spec creation
3. Rest of workflow continues normally

---

### Example 4: Manual Step-by-Step (Maximum Control)

```bash
# 1. Create specification
/speckit.specify Build a RESTful API for task management with user accounts, projects, and tasks

# 2. Review spec, then clarify if needed
/speckit.clarify

# 3. Create technical plan
/speckit.plan

# 4. Generate security checklist
/speckit.checklist Generate security checklist

# 5. Generate accessibility checklist
/speckit.checklist Generate accessibility checklist

# 6. Generate tasks
/speckit.tasks

# 7. Analyze consistency
/speckit.analyze

# 8. Implement
/speckit.implement
```

## Directory Structure

```
your-project/
├── .claude/
│   ├── commands/              # Slash commands
│   │   ├── speckit.constitution.md
│   │   ├── speckit.specify.md
│   │   ├── speckit.clarify.md
│   │   ├── speckit.plan.md
│   │   ├── speckit.checklist.md
│   │   ├── speckit.tasks.md
│   │   ├── speckit.analyze.md
│   │   ├── speckit.implement.md
│   │   ├── speckit-orchestrate.md
│   │   └── speckit-workflow-v2.md
│   └── hooks/                 # Workflow hooks and logs
│
├── .specify/
│   ├── memory/
│   │   └── constitution.md    # Project principles
│   ├── scripts/bash/          # Automation scripts
│   │   ├── common.sh
│   │   ├── check-prerequisites.sh
│   │   ├── create-new-feature.sh
│   │   ├── setup-plan.sh
│   │   └── update-agent-context.sh
│   └── templates/             # Document templates
│       ├── spec-template.md
│       ├── plan-template.md
│       ├── tasks-template.md
│       ├── checklist-template.md
│       └── agent-file-template.md
│
├── specs/                     # Feature specifications (created per feature)
│   └── 001-user-auth/
│       ├── spec.md            # User-focused specification (WHAT/WHY)
│       ├── plan.md            # Technical plan (HOW)
│       ├── tasks.md           # Executable tasks
│       ├── research.md        # Technical research
│       ├── data-model.md      # Entity definitions
│       ├── quickstart.md      # Integration scenarios
│       ├── contracts/         # API contracts
│       │   ├── auth.yaml
│       │   └── users.yaml
│       └── checklists/        # Quality checklists
│           ├── security.md
│           ├── accessibility.md
│           └── performance.md
│
├── src/                       # Your actual code (created during implementation)
├── tests/                     # Your tests
├── CLAUDE.md                  # Claude Code guidance
└── README.md                  # This file
```

## Best Practices

### 1. Spec Quality

**DO**:
- ✅ Write specs from user perspective
- ✅ Use measurable success criteria ("under 3 seconds", "90% task completion")
- ✅ Prioritize user stories (P1 = MVP, P2+ = enhancements)
- ✅ Make each user story independently testable
- ✅ Document assumptions explicitly

**DON'T**:
- ❌ Include implementation details in specs (no "React", "PostgreSQL", "REST API")
- ❌ Use vague language ("fast", "intuitive", "robust") without quantification
- ❌ Skip edge cases and error scenarios
- ❌ Create interdependent user stories

---

### 2. Planning

**DO**:
- ✅ Research alternatives before choosing tech stack
- ✅ Document complexity justifications (if violating constitution)
- ✅ Define clear performance budgets
- ✅ Plan for observability from the start
- ✅ Include testing strategy

**DON'T**:
- ❌ Over-engineer the architecture
- ❌ Skip the research phase
- ❌ Ignore constitution constraints without justification
- ❌ Forget about deployment and operations

---

### 3. Task Organization

**DO**:
- ✅ Organize by user story priority
- ✅ Mark parallelizable tasks with `[P]`
- ✅ Include file paths in every task
- ✅ Make tasks small and verifiable
- ✅ Define clear dependencies

**DON'T**:
- ❌ Mix setup, implementation, and polish in same phase
- ❌ Create tasks without file paths
- ❌ Make tasks too large (split if >1 hour)
- ❌ Forget to mark independent tasks as parallel

---

### 4. Checklists

**DO**:
- ✅ Validate requirement quality, not implementation
- ✅ Reference specific spec sections `[Spec §FR-5]`
- ✅ Mark gaps with `[Gap]` marker
- ✅ Focus on completeness, clarity, consistency
- ✅ Run domain-specific checklists (security, accessibility, etc.)

**DON'T**:
- ❌ Write verification tests ("Verify button works")
- ❌ Check implementation behavior
- ❌ Skip traceability references
- ❌ Use vague checklist items

---

### 5. Branch and Directory Naming

**DO**:
- ✅ Use `###-short-name` format (e.g., `004-oauth-integration`)
- ✅ Keep short-name to 2-4 words
- ✅ Check all sources (remote branches, local branches, spec directories) for highest number
- ✅ Use action-noun format when possible

**DON'T**:
- ❌ Reuse numeric prefixes for different features
- ❌ Create long, verbose branch names
- ❌ Skip the numeric prefix
- ❌ Use special characters or spaces

## Troubleshooting

### "Not on a feature branch" Error

**Problem**: Current branch doesn't follow `###-feature-name` pattern

**Solution**:
```bash
# Option 1: Create new feature via /speckit.specify
/speckit.specify Your feature description

# Option 2: Set environment variable
export SPECIFY_FEATURE="005-my-feature"

# Option 3: Rename branch
git branch -m 005-my-feature
```

---

### Missing Prerequisites

**Problem**: Commands fail because required files don't exist

**Solution**:
```bash
# Check what's available
./.specify/scripts/bash/check-prerequisites.sh --json

# For tasks.md error: run /speckit.tasks first
/speckit.tasks

# For plan.md error: run /speckit.plan first
/speckit.plan

# For spec.md error: run /speckit.specify first
/speckit.specify Your feature description
```

---

### Multiple Spec Directories with Same Prefix

**Problem**: Error about multiple directories like `004-feature-a` and `004-feature-b`

**Solution**: Only one spec directory per numeric prefix. Consolidate or rename:
```bash
# Rename the older one
mv specs/004-old-feature specs/005-old-feature

# Or delete if obsolete
rm -rf specs/004-old-feature
```

---

### Checklist Won't Pass

**Problem**: Checklist keeps failing even after updates

**Common issues**:
1. **Missing requirements**: Add missing requirements to spec.md
2. **Vague language**: Quantify vague terms ("fast" → "under 200ms")
3. **No edge cases**: Document error handling, empty states, failures
4. **Inconsistencies**: Align conflicting requirements

**Debug process**:
```bash
# 1. Review failing checklist items
cat specs/###-feature/checklists/security.md

# 2. Update spec.md to address gaps
# (edit spec.md)

# 3. Re-run checklist
/speckit.checklist

# 4. If still failing, use clarify
/speckit.clarify
```

---

### Tasks Not Executing in Order

**Problem**: Tasks run out of order or have dependency issues

**Solution**: Check task markers
```markdown
✅ CORRECT:
- [ ] T001 Setup project structure
- [ ] T002 [P] Create User model (depends on T001 complete)
- [ ] T003 [P] Create Auth model (depends on T001 complete)
- [ ] T004 Implement UserService (depends on T002 complete)

❌ WRONG:
- [ ] T001 [P] Setup project structure (shouldn't be parallel - it's a dependency)
- [ ] T002 Implement UserService (missing dependency on model)
```

---

### Spec Has Too Many Clarification Markers

**Problem**: Spec ends up with >3 `[NEEDS CLARIFICATION]` markers

**Solution**: `/speckit.specify` enforces max 3 clarifications. If you see more:
1. Run `/speckit.clarify` to resolve them
2. Or make informed guesses and document as assumptions
3. Re-run `/speckit.specify` if needed

---

### Implementation Ignores Parallel Markers

**Problem**: Tasks marked `[P]` still run sequentially

**Solution**:
- Ensure you used `--parallel` flag
- Check that tasks don't affect the same files
- Verify no hidden dependencies exist

```bash
# Use parallel flag
/speckit.implement --parallel

# Or in orchestrator
/speckit-workflow-v2 "..." --parallel
```

## Advanced Topics

### Custom Constitution Principles

Edit `.specify/memory/constitution.md` to add project-specific principles:

```markdown
### VIII. API Versioning

All public APIs MUST:
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Support previous MINOR version for 6 months
- Document breaking changes in CHANGELOG.md
- Provide migration guides for MAJOR bumps
```

---

### Environment Variables

```bash
# Override feature detection (useful for non-git repos)
export SPECIFY_FEATURE="005-my-feature"

# Useful for CI/CD or non-git workflows
```

---

### Non-Git Repositories

The workflow system supports non-git repositories:

```bash
# Scripts automatically detect and adapt
./.specify/scripts/bash/check-prerequisites.sh --paths-only

# You'll see: "Git repository not detected; skipped branch validation"
# Feature detection falls back to spec directories
```

---

### Extending with Custom Commands

Create custom slash commands in `.claude/commands/`:

```markdown
<!-- .claude/commands/my-custom-command.md -->
---
description: My custom workflow step
---

## Objective
Do something custom after planning

## Flow
1. Load plan.md
2. Do custom processing
3. Update artifacts
```

---

### Integration with CI/CD

```yaml
# .github/workflows/spec-validation.yml
name: Spec Validation
on: [pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install Spec Kit
        run: uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
      - name: Validate spec
        run: specify check
      - name: Check checklists
        run: |
          # Count incomplete checklist items
          if grep -r "^- \[ \]" specs/*/checklists/ ; then
            echo "❌ Incomplete checklist items found"
            exit 1
          fi
```

---

### Combining with TCHES Resources

TCHES provides additional capabilities:

1. **Meta-Prompting**: `/create-meta-prompt` - Refine vague briefs
2. **Todo Management**: `/check-todos`, `/add-to-todos` - Track work across sessions
3. **Context Handoff**: `/whats-next` - Create handoff docs for fresh sessions
4. **Skill Building**: Custom skill creation for repetitive patterns

**Setup**:
```bash
# Install TCHES commands
git clone https://github.com/glittercowboy/taches-cc-resources.git /tmp/taches
cp /tmp/taches/prompts/meta-prompting/*.md ~/.claude/commands/
cp /tmp/taches/prompts/todo-management/*.md ~/.claude/commands/
cp /tmp/taches/prompts/context-handoff/*.md ~/.claude/commands/
```

**Usage**:
```bash
# Use meta-prompting with orchestrator
/speckit-orchestrate "something vague" --meta

# Or use standalone
/create-meta-prompt I want to improve user engagement
# Then use refined output with Spec Kit
```

## Contributing

Contributions welcome! Please follow these guidelines:

1. **Spec changes**: Run `/speckit.specify` for new features
2. **Quality gates**: Always include checklist validation
3. **Testing**: Verify changes don't break existing workflows
4. **Documentation**: Update README.md and CLAUDE.md

## License

This project follows the Spec Kit licensing. See individual components for details.

## Resources

- **Spec Kit**: https://github.com/github/spec-kit
- **TCHES Resources**: https://github.com/glittercowboy/taches-cc-resources
- **Claude Code**: https://claude.ai/code
- **uv installer**: https://github.com/astral-sh/uv

## Support

- **Issues**: File issues in the respective repositories
- **Questions**: Check CLAUDE.md for Claude Code specific guidance
- **Troubleshooting**: See [Troubleshooting](#troubleshooting) section above

---

**Built with ❤️ for specification-driven development**
