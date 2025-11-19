---
name: feature-development
description: Complete feature development workflow from spec to deployment
agents:
  - tech-lead-architect
  - frontend-engineer
  - code-reviewer
phases:
  - planning
  - implementation
  - review
  - deployment
---

# Feature Development Workflow

## Overview

This workflow guides a complete feature from specification through deployment, coordinating multiple specialized agents.

## Phases

### Phase 1: Planning & Architecture

**Agent**: tech-lead-architect

**Activities**:
1. Review feature specification (spec.md)
2. Design technical architecture
3. Create implementation plan (plan.md)
4. Define API contracts
5. Identify risks and dependencies

**Outputs**:
- `specs/###-feature/plan.md`
- `specs/###-feature/data-model.md`
- `specs/###-feature/contracts/`
- Architecture diagrams (if complex)

**Handoff Criteria**:
- [ ] Technical approach documented
- [ ] Dependencies identified
- [ ] API contracts defined
- [ ] Risks assessed and mitigated
- [ ] Team has reviewed and approved

---

### Phase 2: Implementation

**Agents**: frontend-engineer (UI), backend-engineer (API), database-engineer (Data)

**Activities**:
1. Set up project structure
2. Implement features per tasks.md
3. Write tests (unit, integration)
4. Document code and APIs
5. Handle edge cases and errors

**Outputs**:
- Source code in `src/`
- Tests in `tests/`
- Updated documentation
- Completed tasks marked in tasks.md

**Handoff Criteria**:
- [ ] All tasks completed
- [ ] Tests passing
- [ ] Code documented
- [ ] No linting errors
- [ ] Ready for code review

---

### Phase 3: Code Review

**Agent**: code-reviewer

**Activities**:
1. Review code quality and style
2. Check security vulnerabilities
3. Verify test coverage
4. Assess performance implications
5. Provide constructive feedback

**Outputs**:
- Code review report
- List of required changes
- List of suggested improvements

**Handoff Criteria**:
- [ ] No critical issues found
- [ ] Major concerns addressed
- [ ] Code approved by reviewer
- [ ] Tests comprehensive
- [ ] Documentation adequate

---

### Phase 4: Deployment

**Agent**: tech-lead-architect (or devops-engineer)

**Activities**:
1. Create deployment plan
2. Run final integration tests
3. Deploy to staging
4. Smoke test in staging
5. Deploy to production
6. Monitor for issues

**Outputs**:
- Deployment logs
- Monitoring dashboards
- Rollback plan (if needed)

**Completion Criteria**:
- [ ] Successfully deployed to production
- [ ] No errors in production logs
- [ ] Key metrics stable
- [ ] Feature flag enabled (if applicable)
- [ ] Stakeholders notified

## Workflow Execution

```bash
# 1. Start with specification
/speckit.specify <feature description>

# 2. Create technical plan
/speckit.plan

# 3. Generate tasks
/speckit.tasks

# 4. Implement with appropriate agent
# For frontend: Use frontend-engineer agent
# For backend: Use standard implementation
# For full-stack: Coordinate both

# 5. Review code
# Invoke code-reviewer agent

# 6. Deploy
# Follow deployment runbook
```

## Agent Coordination

### Sequential Handoffs

```
Tech Lead → Implementation → Code Review → Deployment
   ↓            ↓                ↓             ↓
 Plan         Code           Review        Deploy
```

### Parallel Work

- Frontend and Backend can work in parallel after API contracts defined
- Different user stories can be implemented in parallel
- Tests can be written in parallel with implementation (TDD)

## Communication Protocol

### Handoff Document Template

```markdown
## Handoff: [From Agent] → [To Agent]

**Date**: YYYY-MM-DD
**Feature**: ###-feature-name

### Completed
- [What was completed]
- [Key decisions made]
- [Artifacts created]

### Context for Next Phase
- [Important information to know]
- [Assumptions made]
- [Known limitations]

### Open Questions
- [Question 1]
- [Question 2]

### Next Steps
- [ ] [Action 1]
- [ ] [Action 2]
```

## Quality Gates

### After Planning
- [ ] Architecture reviewed by tech lead
- [ ] API contracts validated
- [ ] Dependencies confirmed available
- [ ] Timeline approved by PM

### After Implementation
- [ ] All tests passing
- [ ] Code coverage >80%
- [ ] No critical security issues
- [ ] Performance benchmarks met

### After Review
- [ ] All critical issues resolved
- [ ] Code approved by reviewer
- [ ] Documentation complete
- [ ] Ready for deployment

### After Deployment
- [ ] Health checks passing
- [ ] No error spikes
- [ ] Key metrics stable
- [ ] Rollback plan ready
