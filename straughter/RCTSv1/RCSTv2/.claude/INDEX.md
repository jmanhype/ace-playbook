# Claude Code Directory Index

Complete index of all files and their purposes in the `.claude/` directory structure.

## Quick Navigation

- [Agents](#agents) - 5 specialized agents
- [Commands](#commands) - 11 slash commands
- [Workflows](#workflows) - 2 workflow definitions
- [Skills](#skills) - 1 skill definition
- [Configuration](#configuration) - Project settings
- [Specialized Agents](#specialized-agents) - Extended agent system

---

## Agents

Location: `.claude/agents/`

| File | Agent | Purpose |
|------|-------|---------|
| `code-reviewer.md` | Code Reviewer | Quality, security, performance review |
| `frontend-engineer.md` | Frontend Engineer | React/Vue/Svelte development, UX |
| `tech-lead-architect.md` | Tech Lead & Architect | System design, architecture decisions |
| `project-manager.md` | Project Manager | Planning, coordination, delivery |
| `ux-designer.md` | UX/UI Designer | User research, design, accessibility |

**Usage**:
```bash
# Agents are invoked automatically based on context
# Or explicitly with Task tool
```

---

## Commands

Location: `.claude/commands/`

### Core Workflow Commands

| Command | Purpose | Phase |
|---------|---------|-------|
| `/speckit.constitution` | Create project principles | Setup |
| `/speckit.specify` | Create feature spec (WHAT/WHY) | Specification |
| `/speckit.clarify` | Resolve specification ambiguities | Clarification |
| `/speckit.plan` | Create technical plan (HOW) | Planning |
| `/speckit.checklist` | Generate quality checklists | Validation |
| `/speckit.tasks` | Generate executable tasks | Task Breakdown |
| `/speckit.analyze` | Cross-artifact consistency check | Analysis |
| `/speckit.implement` | Execute implementation | Implementation |
| `/speckit.taskstoissues` | Convert tasks to GitHub issues | Integration |

### Orchestrator Commands

| Command | Purpose | Use Case |
|---------|---------|----------|
| `/speckit-orchestrate` | Basic workflow (specify→plan→tasks→implement) | Quick iterations |
| `/speckit-workflow-v2` | Full workflow with quality gates | Production features |

**Orchestrator Comparison**:

| Feature | `/speckit-orchestrate` | `/speckit-workflow-v2` |
|---------|----------------------|----------------------|
| Meta-prompting | ✅ Optional (--meta) | ❌ |
| Clarify phase | ❌ | ✅ |
| Checklist gate | ❌ | ✅ |
| Analyze gate | ❌ | ✅ |
| Strict mode | ❌ | ✅ |
| Auto mode | ❌ | ✅ |
| Best for | Prototypes, exploration | Production, compliance |

---

## Workflows

Location: `.claude/workflows/`

| File | Workflow | Phases | Use Case |
|------|----------|--------|----------|
| `feature-development.md` | Feature Development | Planning → Implementation → Review → Deployment | New features |
| `bug-fix.md` | Bug Fix | Investigation → Fix → Review → Deploy → Post-mortem | Bug fixes (P0-P3) |

**Workflow Usage**:
```bash
# Workflows are referenced by orchestrators or invoked directly
# They coordinate multiple agents and phases
```

---

## Skills

Location: `.claude/skills/`

| File | Skill | Category | Difficulty |
|------|-------|----------|------------|
| `code-review.md` | Code Review | Quality Assurance | Intermediate |

**Skill Usage**:
```bash
# Skills are reusable techniques applied across contexts
# Used by agents or invoked directly
```

---

## Configuration

Location: `.claude/`

| File | Purpose | Format |
|------|---------|--------|
| `settings.json` | Project-wide Claude Code configuration | JSON |

**Key Settings**:
- Project metadata
- Agent configuration
- Tool permissions
- Quality gates
- Testing requirements
- Git configuration
- Security settings

---

## Specialized Agents

Extended agent system for detailed definitions.

### Descriptions

Location: `specialized-agents/Descriptions/`

| File | Agent | Format |
|------|-------|--------|
| `code-reviewer.txt` | Code Reviewer | Plain text |

**Purpose**: Quick reference, documentation, onboarding

### System Prompts

Location: `specialized-agents/system-prompts/`

| File | Agent | Format |
|------|-------|--------|
| `code-reviewer-system-prompt.md` | Code Reviewer | Markdown |

**Purpose**: Comprehensive behavior specification, decision-making guidelines

---

## Context Storage

Location: `.claude/context/`

**Auto-managed by Claude Code**

Stores:
- Session context
- Project decisions
- User preferences
- Conversation history

**Max Size**: 10 MB (configurable in settings.json)
**Auto-cleanup**: 30 days (configurable)

---

## Templates

Location: `.claude/templates/`

**Currently Empty** - Add custom file templates as needed

Example templates:
- Component templates
- API route templates
- Test file templates
- Documentation templates

---

## Hooks & Logs

Location: `.claude/hooks/logs/`

| File | Event | Purpose |
|------|-------|---------|
| `chat.json` | Chat messages | Message logging |
| `pre_tool_use.json` | Before tool execution | Pre-execution logging |
| `post_tool_use.json` | After tool execution | Post-execution logging |
| `stop.json` | Session end | Session termination logging |

**Format**: JSON with timestamp, session_id, metadata

---

## File Count Summary

| Directory | Files | Purpose |
|-----------|-------|---------|
| `.claude/agents/` | 5 | Specialized agent definitions |
| `.claude/commands/` | 11 | Slash commands |
| `.claude/workflows/` | 2 | Multi-phase workflows |
| `.claude/skills/` | 1 | Reusable skills |
| `.claude/context/` | Auto-managed | Session context |
| `.claude/templates/` | 0 | File templates |
| `.claude/hooks/logs/` | 4 | Event logs |
| `specialized-agents/Descriptions/` | 1 | Agent descriptions |
| `specialized-agents/system-prompts/` | 1 | Agent system prompts |
| **Total** | **25+** | Complete Claude Code setup |

---

## Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Comprehensive project documentation |
| `CLAUDE.md` | Claude Code-specific guidance |
| `CLAUDE_FOLDER_STRUCTURE.md` | Complete folder structure guide |
| `.claude/INDEX.md` | This file - quick reference index |

---

## Quick Reference

### Common Tasks

**Start a new feature**:
```bash
/speckit-workflow-v2 "Your feature description" --strict --parallel
```

**Quick prototype**:
```bash
/speckit-orchestrate "Your idea" --parallel
```

**Code review**:
```bash
# Use code-reviewer agent via Task tool
# Or let it be auto-invoked during implementation
```

**Bug fix**:
```bash
# Follow bug-fix workflow
# P0: Immediate hotfix
# P1+: Standard process
```

### Finding Files

**Agent definition**: `.claude/agents/{agent-name}.md`
**Command definition**: `.claude/commands/{command-name}.md`
**Workflow**: `.claude/workflows/{workflow-name}.md`
**Skill**: `.claude/skills/{skill-name}.md`
**Settings**: `.claude/settings.json`

---

## Maintenance

### Adding New Agents

1. Create `.claude/agents/{name}.md`
2. Create `specialized-agents/Descriptions/{name}.txt`
3. Create `specialized-agents/system-prompts/{name}-system-prompt.md`
4. Add to `settings.json` → `agents.available`

### Adding New Commands

1. Create `.claude/commands/{name}.md`
2. Add frontmatter with description
3. Document in README.md

### Adding New Workflows

1. Create `.claude/workflows/{name}.md`
2. Define phases and agents
3. Add to `settings.json` → `workflows.available`

### Adding New Skills

1. Create `.claude/skills/{name}.md`
2. Document purpose and process
3. Tag with category and difficulty

---

**Last Updated**: 2025-11-18
**Version**: 1.0.0
