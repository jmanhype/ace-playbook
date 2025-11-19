---
description: "Full Spec Kit flow with quality gates: constitution→specify→clarify→plan→checklist→tasks→analyze→implement. Usage: /speckit-workflow-v2 <brief> [--domains 'security,accessibility,performance'] [--strict] [--auto] [--parallel]"
# Purpose: Orchestrate ALL major Spec Kit features—including domain checklists—via SlashCommand without manual chaining.
# Tools: Bash only for sanity checks; the heavy lifting is done by slash commands the agent invokes autonomously.
allowed-tools:
  - Bash(specify:*)
---

# Speckit Workflow v2 (Feature-complete)

## Objective

Convert "$ARGUMENTS" into a validated spec-driven change using **every relevant Spec Kit phase**:
constitution → specify → (clarify) → plan → **checklist** → tasks → analyze → implement.

## Prechecks
- If /speckit.* commands aren't visible in /help:
  !uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
  !specify init --here --ai claude
  !specify check

## Inputs
- `$ARGUMENTS` = brief + optional flags:
  - `--domains "<csv>"` e.g., "security,privacy,accessibility,performance,observability"
  - `--strict`          fail the run if checklist/analyze gates don't pass
  - `--auto`            proceed without asking between gates (unless a blocking ambiguity is detected)
  - `--parallel`        prefer parallel execution during implement where deps allow

## Flow
1) Constitution (first run only)
   - If no principles exist, run:
     /speckit.constitution Create principles for code quality, testing standards, UX consistency, performance/SLOs, security, accessibility.

2) Specify (WHAT/WHY)
   - Run:
     /speckit.specify $ARGUMENTS
   - Keep tech decisions out of the spec.

3) Clarify (targeted; only if needed)
   - If the spec is underspecified, run:
     /speckit.clarify Ask only focused questions to resolve ambiguities blocking the plan.

4) Plan (HOW)
   - Run:
     /speckit.plan Draft the technical plan aligned to the spec: architecture, data model, interfaces, testing/observability, performance budgets, risks.

5) **Checklist (quality gate)**
   - Build the domain list:
     - If `--domains` present, use it; otherwise default to: security, privacy, accessibility, performance, observability, compliance, UX.
   - Run:
     /speckit.checklist Generate a checklist for the chosen domains. Identify missing requirements, non-goals, acceptance criteria, and risky gaps.
   - If items remain unclear or unchecked:
     - Loop: `/speckit.clarify` to resolve; update spec/plan; rerun `/speckit.checklist`.
   - If `--strict` and checklist is not green → **stop with actionable summary**.

6) Tasks (make it executable)
   - Run:
     /speckit.tasks
   - Require small, verifiable tasks with dependencies, Definition of Done, and verification steps.

7) Analyze (consistency gate)
   - Run:
     /speckit.analyze
   - If inconsistencies between spec/plan/tasks remain:
     - Repair artifacts; rerun `/speckit.tasks`; rerun `/speckit.analyze`.
   - If `--strict` and analysis isn't clean → **stop with actionable summary**.

8) Implement (do the work)
   - Run:
     /speckit.implement
   - If `--parallel` present, request parallelization where dependencies allow.

9) Report
   - Emit a concise summary: phases executed, artifacts changed, outstanding TODOs, and what to do next if gates failed.

## Notes
- Use SlashCommand to **invoke** each command; do not merely print them.
- Ask ≤5 questions only when ambiguity blocks a gate and `--auto` is not present.

## Success
- Spec articulates users, journeys, outcomes, and success criteria.
- Plan covers stack, architecture, data model, testing, observability, and perf budgets.
- Checklist is green across requested domains.
- Analyze finds no cross-artifact contradictions.
- Implementation completes or yields actionable failures with verification steps.
