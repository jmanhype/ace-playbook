---
description: "Optionally run TCHES metaprompt, then chain Spec Kit passes (specify→plan→tasks→implement). Usage: /speckit-orchestrate <brief> [--meta] [--parallel]"
allowed-tools:
  - Bash(specify:*)
  - SlashCommand
---

# Speckit Orchestrate (Basic)

## Purpose

Guide the agent to (optionally) refine the brief via TCHES' metaprompt, then run Spec Kit's passes end-to-end via SlashCommand.

## Objective

Turn "$ARGUMENTS" into an executed, spec-driven change:
1. (if `--meta` present and available) run TCHES' `/create-meta-prompt`
2. run the Spec Kit passes: specify → plan → tasks → implement
3. summarize artifacts, diffs, and verification status

## Inputs

- `$ARGUMENTS` = brief + optional flags:
  - `--meta`        Use TCHES metaprompt to refine the brief before starting
  - `--parallel`    Prefer parallel execution during implement where deps allow

## Flow

### 1. Optional Meta-Prompting (if --meta flag present)

- Check if `/create-meta-prompt` command exists
- If exists and `--meta` flag present:
  - Run: `/create-meta-prompt $ARGUMENTS`
  - Use its outputs (users, outcomes, constraints) to enrich the spec phase
- If not exists, skip and proceed to Specify

### 2. Specify (WHAT/WHY)

- Run:
  - `/speckit.specify $ARGUMENTS`
  - (fallback if not found) `/specify $ARGUMENTS`
- Keep tech decisions out of the spec

### 3. Plan (HOW)

- Run:
  - `/speckit.plan` Draft a technical plan aligned to the spec: architecture, data model, interfaces, testing, observability, performance budgets, risks.
  - (fallback) `/plan`

### 4. Tasks (make it executable)

- Run:
  - `/speckit.tasks`
  - (fallback) `/tasks`
- Require small, verifiable tasks with dependencies, Definition of Done, and verification steps

### 5. Implement (do the work)

- Run:
  - `/speckit.implement`
  - (fallback) `/implement`
- If `--parallel` present, request parallelization where dependencies allow

### 6. Report

- Emit a concise summary:
  - Phases executed
  - Artifacts created/changed
  - Outstanding TODOs
  - Verification status

## Notes

- Use SlashCommand to **INVOKE** commands; don't just print them
- Spec Kit passes may appear as `/speckit.specify` etc. **or** `/specify` etc., depending on integration. Use whichever exists.
- Keep tech choices out of "specify"; put them in "plan"
- This is a **streamlined workflow** - for quality gates (clarify, checklist, analyze), use `/speckit-workflow-v2` instead

## Success Criteria

- Spec captures users, journeys, outcomes, and success criteria
- Plan covers architecture, data model, testing, observability
- Tasks are verifiable and dependency-ordered
- Implementation completes or yields actionable failures with paths to fix
