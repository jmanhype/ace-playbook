# Claude Code Folder Structure Documentation

Complete guide to the `.claude/` directory structure and specialized agent system.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Core Directories](#core-directories)
- [Specialized Agents](#specialized-agents)
- [Configuration Files](#configuration-files)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

The `.claude/` folder contains all Claude Code-specific configuration, commands, agents, workflows, and skills that customize how Claude Code operates in your project.

## Directory Structure

```
project-root/
├── .claude/
│   ├── agents/               # Specialized agent definitions
│   │   ├── code-reviewer.md
│   │   ├── frontend-engineer.md
│   │   ├── project-manager.md
│   │   ├── tech-lead-architect.md
│   │   └── ux-designer.md
│   │
│   ├── commands/             # Slash command definitions
│   │   ├── speckit-orchestrate.md
│   │   ├── speckit-workflow-v2.md
│   │   ├── speckit.analyze.md
│   │   ├── speckit.checklist.md
│   │   ├── speckit.clarify.md
│   │   ├── speckit.constitution.md
│   │   ├── speckit.implement.md
│   │   ├── speckit.plan.md
│   │   ├── speckit.specify.md
│   │   ├── speckit.tasks.md
│   │   └── speckit.taskstoissues.md
│   │
│   ├── workflows/            # Multi-step workflow definitions
│   │   ├── feature-development.md
│   │   └── bug-fix.md
│   │
│   ├── skills/               # Reusable skill definitions
│   │   └── code-review.md
│   │
│   ├── context/              # Persistent context storage
│   │   └── (session-specific context files)
│   │
│   ├── templates/            # File templates
│   │   └── (custom templates)
│   │
│   ├── hooks/                # Event hooks
│   │   └── logs/            # Hook execution logs
│   │       ├── chat.json
│   │       ├── pre_tool_use.json
│   │       ├── post_tool_use.json
│   │       └── stop.json
│   │
│   └── settings.json         # Project-wide Claude Code settings
│
├── specialized-agents/       # Extended agent system
│   ├── Descriptions/        # Agent role descriptions
│   │   └── code-reviewer.txt
│   └── system-prompts/      # Agent system prompts
│       └── code-reviewer-system-prompt.md
│
└── [your project files]
```

## Core Directories

### `.claude/agents/`

**Purpose**: Define specialized agents with specific roles and expertise.

**When to use**:
- You need domain-specific expertise (frontend, backend, DevOps)
- You want consistent behavior for specific tasks
- You need role-based code review or implementation
- You want to enforce specific patterns or standards

**Agent File Structure**:
```markdown
---
name: agent-name
description: Brief description
role: Role Title
expertise:
  - Area 1
  - Area 2
tools:
  - Tool1
  - Tool2
---

# Agent Name

## Role
[Detailed role description]

## Responsibilities
[What this agent is responsible for]

## Process
[How this agent approaches tasks]
```

**Examples**:
- `code-reviewer.md` - Code quality, security, performance review
- `frontend-engineer.md` - React/Vue/UI development
- `tech-lead-architect.md` - System design, architecture decisions
- `project-manager.md` - Planning, coordination, delivery
- `ux-designer.md` - User research, design, accessibility

---

### `.claude/commands/`

**Purpose**: Define custom slash commands that can be invoked in chat.

**When to use**:
- You have repetitive workflows to automate
- You want to standardize common operations
- You need multi-step processes with specific ordering
- You want to enforce project conventions

**Command File Structure**:
```markdown
---
description: "Brief description for /help"
allowed-tools:
  - Tool1
  - Tool2
---

# Command Name

## Objective
[What this command accomplishes]

## Flow
[Step-by-step execution]

## Success Criteria
[How to know when complete]
```

**Examples**:
- `/speckit.specify` - Create feature specification
- `/speckit.implement` - Execute implementation plan
- `/speckit-workflow-v2` - Full workflow with quality gates

**Usage**:
```bash
# In Claude Code chat
/speckit.specify Add user authentication
/speckit-workflow-v2 "Build payment system" --strict --parallel
```

---

### `.claude/workflows/`

**Purpose**: Define multi-phase workflows that coordinate multiple agents or steps.

**When to use**:
- You have complex multi-step processes
- Different agents need to work sequentially or in parallel
- You want to standardize team processes (feature dev, bug fix, release)
- You need handoff points between phases

**Workflow File Structure**:
```markdown
---
name: workflow-name
description: Workflow purpose
agents:
  - agent1
  - agent2
phases:
  - phase1
  - phase2
---

# Workflow Name

## Overview
[Workflow description]

## Phases

### Phase 1: [Name]
**Agent**: [agent-name]
**Activities**: [What happens]
**Outputs**: [What gets created]
**Handoff Criteria**: [When complete]

### Phase 2: [Name]
...
```

**Examples**:
- `feature-development.md` - Spec → Plan → Implement → Review → Deploy
- `bug-fix.md` - Investigate → Fix → Review → Deploy → Post-mortem

---

### `.claude/skills/`

**Purpose**: Define reusable skills that can be applied across different contexts.

**When to use**:
- You have techniques used across multiple agents
- You want to standardize specific approaches (code review, testing)
- You need training materials for new team members
- You want to share best practices

**Skill File Structure**:
```markdown
---
skill: skill-name
category: category
difficulty: beginner|intermediate|advanced
tags:
  - tag1
  - tag2
---

# Skill Name

## Purpose
[What this skill enables]

## When to Use
[Applicable scenarios]

## Process
[Step-by-step approach]

## Examples
[Concrete examples]
```

**Examples**:
- `code-review.md` - Systematic code review process
- `debugging.md` - Structured debugging approach
- `refactoring.md` - Safe refactoring patterns

---

### `.claude/context/`

**Purpose**: Store persistent context across sessions.

**When to use**:
- You want to maintain state between sessions
- You need to remember project-specific decisions
- You want to cache frequently used information
- You need session history for handoffs

**Structure**:
```
context/
├── project-context.json      # Project-wide context
├── session-YYYY-MM-DD.json   # Daily session context
└── decisions/                # Architectural decisions
    └── decision-001.md
```

**Auto-managed**: Claude Code automatically stores and retrieves context.

---

### `.claude/templates/`

**Purpose**: File templates for commonly created files.

**When to use**:
- You want consistent file structure
- You have boilerplate that repeats
- You want to enforce conventions
- You need starter code for new files

**Examples**:
```
templates/
├── component.tsx.template
├── api-route.ts.template
├── test.spec.ts.template
└── readme.template.md
```

---

### `.claude/hooks/`

**Purpose**: Execute scripts or log events at specific lifecycle points.

**Hook Types**:
- `pre_tool_use` - Before any tool executes
- `post_tool_use` - After tool execution
- `chat` - On chat messages
- `stop` - On session end

**Logs Structure**:
```json
{
  "timestamp": "2025-11-18T17:21:00.480870",
  "session_id": "unique-id",
  "conversation": {},
  "message_count": 42,
  "transcript_path": ".claude/hooks/logs/chat.json"
}
```

**Use cases**:
- Audit logging
- Metrics collection
- Integration triggers
- Custom validation

---

## Specialized Agents Directory

### `specialized-agents/Descriptions/`

**Purpose**: Plain-text agent descriptions for quick reference and documentation.

**Format**: Simple text file (`.txt`) with structured information.

**Content**:
```text
AGENT: [Name]

PRIMARY ROLE:
[One-line description]

EXPERTISE AREAS:
- [Area 1]
- [Area 2]

KEY RESPONSIBILITIES:
1. [Responsibility 1]
2. [Responsibility 2]

WHEN TO USE:
- [Scenario 1]
- [Scenario 2]

SUCCESS CRITERIA:
- [Criteria 1]
- [Criteria 2]
```

**Usage**: Documentation, onboarding, quick reference

---

### `specialized-agents/system-prompts/`

**Purpose**: Detailed system prompts that define agent behavior and decision-making.

**Format**: Markdown files (`.md`) with comprehensive instructions.

**Structure**:
```markdown
# [Agent Name] System Prompt

You are [role description with expertise].

## Your Mission
[Primary objective]

## Core Principles
[Guiding principles for decisions]

## Process
[Step-by-step approach]

## Output Format
[Expected deliverables]

## Communication Guidelines
[How to communicate with users]
```

**Usage**: Agent initialization, behavior specification, training

---

## Configuration Files

### `.claude/settings.json`

Comprehensive project configuration for Claude Code behavior.

**Key Sections**:

#### 1. Project Metadata
```json
{
  "project": {
    "name": "Project Name",
    "version": "1.0.0",
    "description": "...",
    "type": "web-app|library|framework"
  }
}
```

#### 2. Agent Configuration
```json
{
  "agents": {
    "default": "general-purpose",
    "available": ["code-reviewer", "frontend-engineer"],
    "auto_select": true
  }
}
```

#### 3. Tool Permissions
```json
{
  "tools": {
    "permissions": {
      "Read": {
        "enabled": true,
        "paths": ["**/*"],
        "excluded_paths": ["node_modules/**", ".env*"]
      },
      "Write": {
        "enabled": true,
        "require_confirmation": ["package.json"]
      }
    }
  }
}
```

#### 4. Quality Gates
```json
{
  "quality_gates": {
    "enabled": true,
    "strict_mode": false,
    "domains": ["security", "accessibility", "performance"]
  }
}
```

#### 5. Testing Requirements
```json
{
  "testing": {
    "required": true,
    "min_coverage": 80,
    "frameworks": {
      "javascript": ["jest", "vitest"]
    }
  }
}
```

#### 6. Git Configuration
```json
{
  "git": {
    "auto_commit": false,
    "commit_message_format": {
      "type": "conventional",
      "include_co_author": true
    }
  }
}
```

---

## Best Practices

### 1. Agent Organization

**DO**:
- ✅ Create agents for distinct roles (frontend, backend, review)
- ✅ Keep agent files focused and single-purpose
- ✅ Document agent expertise clearly
- ✅ Include concrete examples in agent definitions

**DON'T**:
- ❌ Create overlapping agents with duplicate responsibilities
- ❌ Make agents too generic (defeats the purpose)
- ❌ Skip documentation - agents should be self-explanatory

### 2. Command Design

**DO**:
- ✅ Name commands clearly (`/speckit.specify` not `/ss`)
- ✅ Include usage examples in command description
- ✅ Define clear success criteria
- ✅ Document expected inputs and outputs

**DON'T**:
- ❌ Create commands for one-off tasks (use direct instructions)
- ❌ Make commands too complex (break into workflows)
- ❌ Forget to handle error cases

### 3. Workflow Structure

**DO**:
- ✅ Define clear phase transitions
- ✅ Specify handoff criteria between phases
- ✅ Include quality gates at appropriate points
- ✅ Allow for parallel work where possible

**DON'T**:
- ❌ Create linear workflows when parallelism possible
- ❌ Skip validation between phases
- ❌ Forget to document agent coordination

### 4. Settings Management

**DO**:
- ✅ Use least-privilege tool permissions
- ✅ Enable quality gates for important projects
- ✅ Configure appropriate test requirements
- ✅ Document custom settings

**DON'T**:
- ❌ Give unrestricted tool access
- ❌ Disable security checks without reason
- ❌ Forget to version control settings.json

---

## Examples

### Example 1: Creating a New Agent

```bash
# 1. Create agent definition
cat > .claude/agents/backend-engineer.md <<'EOF'
---
name: backend-engineer
description: Expert backend developer for APIs and data systems
role: Backend Engineering Specialist
expertise:
  - RESTful API design
  - Database optimization
  - Authentication/Authorization
  - Microservices
tools:
  - Read
  - Write
  - Edit
  - Bash
---

# Backend Engineer Agent

## Role
Expert backend developer specializing in scalable APIs...

[Continue with full agent definition]
EOF

# 2. Create description
cat > specialized-agents/Descriptions/backend-engineer.txt <<'EOF'
AGENT: Backend Engineer

PRIMARY ROLE:
Expert backend developer for APIs and data systems
...
EOF

# 3. Create system prompt
cat > specialized-agents/system-prompts/backend-engineer-system-prompt.md <<'EOF'
# Backend Engineer System Prompt

You are an expert backend engineer...
...
EOF

# 4. Update settings.json
# Add "backend-engineer" to agents.available array
```

### Example 2: Creating a Custom Workflow

```markdown
<!-- .claude/workflows/code-review-workflow.md -->
---
name: code-review-workflow
description: Structured code review process
agents:
  - code-reviewer
phases:
  - automated-checks
  - manual-review
  - feedback
---

# Code Review Workflow

## Phase 1: Automated Checks
**Activities**:
1. Run linter
2. Run tests
3. Check coverage
4. Security scan

**Handoff**: All automated checks pass

## Phase 2: Manual Review
**Agent**: code-reviewer
**Activities**:
1. Review code quality
2. Check for security issues
3. Assess performance
4. Verify tests

**Handoff**: Review complete with feedback

## Phase 3: Feedback
**Activities**:
1. Provide structured feedback
2. Request changes or approve
3. Document lessons learned
```

### Example 3: Using Context Storage

```javascript
// Claude Code automatically manages context
// Access via .claude/context/

// Example context file:
{
  "project": "MyApp",
  "lastSession": "2025-11-18",
  "decisions": [
    {
      "date": "2025-11-15",
      "decision": "Use PostgreSQL for primary database",
      "rationale": "Team expertise, ACID compliance needed"
    }
  ],
  "preferences": {
    "testFramework": "vitest",
    "linter": "eslint",
    "formatter": "prettier"
  }
}
```

---

## Directory Size Guidelines

| Directory | Typical Size | Max Recommended |
|-----------|--------------|-----------------|
| `.claude/agents/` | 5-10 files | 20 files |
| `.claude/commands/` | 10-20 files | 50 files |
| `.claude/workflows/` | 2-5 files | 10 files |
| `.claude/skills/` | 5-15 files | 30 files |
| `.claude/context/` | Auto-managed | 10 MB |
| `.claude/templates/` | 5-20 files | 50 files |

---

## Troubleshooting

### Agents Not Available

**Problem**: Custom agents don't appear in agent selection

**Solutions**:
1. Check agent file has proper frontmatter (YAML between `---`)
2. Verify agent name matches filename (e.g., `code-reviewer.md` → `name: code-reviewer`)
3. Add agent to `settings.json` under `agents.available`
4. Restart Claude Code session

---

### Commands Not Found

**Problem**: Slash commands don't work

**Solutions**:
1. Verify command files are in `.claude/commands/`
2. Check frontmatter has `description` field
3. Ensure filename matches command (e.g., `/mycommand` → `mycommand.md`)
4. Check `.claude/commands/` is in project root

---

### Tool Permission Errors

**Problem**: "Tool X is not permitted" errors

**Solutions**:
1. Check `settings.json` → `tools.permissions.[ToolName].enabled`
2. Verify paths are in `paths` array and not in `excluded_paths`
3. Add specific tool permissions if too restrictive
4. Check for `require_confirmation` settings

---

## Migration from Older Structures

If you have an older Claude Code setup:

```bash
# Backup existing structure
cp -r .claude .claude.backup

# Create new directories
mkdir -p .claude/{agents,workflows,skills,context,templates}
mkdir -p specialized-agents/{Descriptions,system-prompts}

# Move existing files to appropriate locations
# (manual process based on file types)

# Create settings.json from defaults
# Copy from this documentation

# Test new structure
# Verify agents load, commands work, workflows execute
```

---

## Additional Resources

- **Claude Code Documentation**: https://docs.claude.com/en/docs/claude-code
- **Spec Kit**: https://github.com/github/spec-kit
- **TCHES Resources**: https://github.com/glittercowboy/taches-cc-resources
- **Project README.md**: Comprehensive usage guide
- **Project CLAUDE.md**: Claude Code-specific guidance

---

**Last Updated**: 2025-11-18
**Version**: 1.0.0
