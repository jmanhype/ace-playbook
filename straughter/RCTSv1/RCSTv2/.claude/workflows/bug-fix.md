---
name: bug-fix
description: Rapid bug investigation, fix, and deployment workflow
priority: high
agents:
  - code-reviewer
estimated_time: 1-4 hours
---

# Bug Fix Workflow

## Overview

Streamlined workflow for investigating, fixing, and deploying bug fixes quickly while maintaining quality.

## Severity Levels

### P0 - Critical (Production Down)
- **Response Time**: Immediate
- **Fix Timeline**: < 2 hours
- **Process**: Expedited, minimal review
- **Deployment**: Hotfix directly to production

### P1 - High (Major Feature Broken)
- **Response Time**: < 1 hour
- **Fix Timeline**: < 8 hours (same day)
- **Process**: Standard review, thorough testing
- **Deployment**: Via fast-track release

### P2 - Medium (Minor Feature Broken)
- **Response Time**: < 4 hours
- **Fix Timeline**: < 3 days
- **Process**: Full review and testing
- **Deployment**: Next regular release

### P3 - Low (Cosmetic/Minor Issue)
- **Response Time**: < 1 day
- **Fix Timeline**: < 1 week
- **Process**: Full review, batched with other fixes
- **Deployment**: Planned release

## Workflow Phases

### Phase 1: Investigation (15-30 min)

**Activities**:
1. Reproduce the bug
2. Identify root cause
3. Assess impact and severity
4. Document findings

**Checklist**:
- [ ] Bug reproduced locally or in logs
- [ ] Root cause identified
- [ ] Severity level assigned
- [ ] Affected users/features documented

**Template**:
```markdown
## Bug Investigation: [Bug Title]

**Severity**: [P0/P1/P2/P3]
**Affected Users**: [Number or percentage]
**Affected Feature**: [Feature name]

### Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Root Cause
[Technical explanation of the issue]

### Proposed Fix
[High-level approach to fix]

### Risk Assessment
**Risk of Fix**: [LOW/MEDIUM/HIGH]
**Risk of Not Fixing**: [LOW/MEDIUM/HIGH]
```

---

### Phase 2: Fix Implementation (30 min - 4 hours)

**Activities**:
1. Write failing test that reproduces bug
2. Implement minimal fix
3. Verify test passes
4. Check for regressions
5. Update documentation if needed

**Checklist**:
- [ ] Test written that reproduces bug
- [ ] Fix implemented
- [ ] Test passes
- [ ] No new test failures
- [ ] Regression tests run

**Code Changes**:
```markdown
## Fix Summary

**Files Changed**:
- `src/module/file.js` - [Description of change]
- `tests/module/file.test.js` - [Added test case]

**Approach**:
[Brief explanation of the fix]

**Alternatives Considered**:
[Why this approach over others]
```

---

### Phase 3: Review (15-60 min)

**Activities**:
1. Self-review the changes
2. Peer review (if P1+)
3. Verify fix doesn't introduce new issues
4. Approve for deployment

**For P0 (Critical)**:
- Quick self-review
- Deploy immediately
- Post-deployment review

**For P1+ (High/Medium/Low)**:
- Full peer code review
- Review by code-reviewer agent if available
- Approval before deployment

**Review Checklist**:
- [ ] Fix addresses root cause
- [ ] No unintended side effects
- [ ] Test coverage adequate
- [ ] Code quality maintained
- [ ] Documentation updated

---

### Phase 4: Deployment (15-30 min)

**Activities**:
1. Deploy to staging
2. Smoke test in staging
3. Deploy to production
4. Monitor for issues
5. Communicate fix to stakeholders

**Deployment Checklist**:
- [ ] Deployed to staging
- [ ] Smoke tests pass
- [ ] Deployed to production
- [ ] Monitoring in place
- [ ] Rollback plan ready
- [ ] Stakeholders notified

---

### Phase 5: Post-Mortem (P0/P1 only)

**Activities**:
1. Document timeline
2. Identify what went well
3. Identify areas for improvement
4. Create action items to prevent recurrence

**Post-Mortem Template**:
```markdown
## Post-Mortem: [Bug Title]

**Date**: YYYY-MM-DD
**Severity**: [P0/P1]
**Duration**: [Detection to resolution time]

### Timeline
- 10:00 AM - Bug reported by user
- 10:15 AM - Investigation started
- 10:45 AM - Root cause identified
- 11:30 AM - Fix implemented and tested
- 12:00 PM - Deployed to production
- 12:15 PM - Confirmed resolved

### Root Cause
[Detailed technical explanation]

### Impact
- **Users Affected**: [Number/percentage]
- **Duration**: [Time]
- **Data Loss**: [Yes/No - details]

### What Went Well
- Quick detection via monitoring
- Clear logs helped identify issue
- Team coordinated effectively

### What Could Be Improved
- Better test coverage would have caught this
- Monitoring could alert earlier
- Documentation was unclear

### Action Items
- [ ] Add integration test for this scenario
- [ ] Improve monitoring for [metric]
- [ ] Update documentation in [location]
- [ ] Review similar code for same issue

### Lessons Learned
[Key takeaways for the team]
```

## Quick Reference

### P0 Critical Bug (Production Down)

```bash
# 1. Investigate immediately
# Document in bug-investigation.md

# 2. Implement fix with test
# Focus on minimal, safe fix

# 3. Quick self-review
# Verify no obvious issues

# 4. Deploy hotfix
git checkout -b hotfix/critical-bug-fix
# Make changes
git commit -m "hotfix: Fix critical bug [ISSUE-123]"
git push
# Deploy to production immediately

# 5. Monitor closely
# Watch logs, metrics, alerts

# 6. Post-mortem within 24 hours
# Document learnings
```

### P1 High Priority Bug

```bash
# 1. Investigate within 1 hour
# Document root cause

# 2. Implement fix same day
# Write test, implement fix, verify

# 3. Get peer review
# Code review before deployment

# 4. Deploy via fast-track
# Staging → Production

# 5. Post-mortem within 48 hours
```

### P2/P3 Medium/Low Priority

```bash
# 1. Investigate within SLA
# Add to bug backlog

# 2. Implement with next sprint
# Follow normal development flow

# 3. Full review process
# Standard code review

# 4. Deploy in planned release
# No rush, thorough testing
```

## Monitoring After Fix

```markdown
## Monitoring Checklist

### Immediately After Deployment (0-15 min)
- [ ] Error rate in logs
- [ ] Response time metrics
- [ ] Specific feature usage
- [ ] User reports/support tickets

### Short-term (1-24 hours)
- [ ] No regression in related features
- [ ] Metrics trending positively
- [ ] No new bugs introduced
- [ ] User feedback positive

### Long-term (1-7 days)
- [ ] Issue fully resolved
- [ ] No recurrence
- [ ] Preventative measures working
```

## Communication Template

```markdown
## Bug Fix Notification

**To**: [Stakeholders]
**Subject**: [P0/P1/P2/P3] Bug Fixed - [Brief Description]

### Summary
[One paragraph explaining what was broken and what was fixed]

### Impact
- **Affected Users**: [Who was impacted]
- **Duration**: [How long the issue existed]
- **Severity**: [How serious it was]

### Resolution
[What was done to fix it]

### Timeline
- **Reported**: [Date/Time]
- **Fixed**: [Date/Time]
- **Deployed**: [Date/Time]

### Prevention
[What we're doing to prevent this in the future]

### Next Steps
[Any follow-up actions or monitoring]
```
