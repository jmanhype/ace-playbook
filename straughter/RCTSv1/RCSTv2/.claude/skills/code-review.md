---
skill: code-review
category: quality-assurance
difficulty: intermediate
tags:
  - code-quality
  - security
  - best-practices
---

# Code Review Skill

## Purpose

Systematically review code for quality, security, performance, and maintainability using industry best practices.

## When to Use

- Before merging pull requests
- During pair programming sessions
- When onboarding new team members
- For security-sensitive code
- When reviewing external contributions

## Review Checklist

### 1. Correctness

```markdown
- [ ] Code does what it's supposed to do
- [ ] Logic is sound and handles edge cases
- [ ] No off-by-one errors
- [ ] Null/undefined checks where needed
- [ ] Error handling is comprehensive
```

### 2. Code Quality

```markdown
- [ ] Functions are small and focused (< 50 lines)
- [ ] Variable/function names are descriptive
- [ ] No magic numbers or strings
- [ ] DRY principle followed (no duplication)
- [ ] SOLID principles applied
- [ ] Complexity is manageable (cyclomatic complexity < 10)
```

### 3. Security

```markdown
- [ ] Input validation and sanitization
- [ ] No SQL injection vulnerabilities
- [ ] No XSS vulnerabilities
- [ ] No command injection vulnerabilities
- [ ] Secrets not hardcoded
- [ ] Authentication/authorization checks present
- [ ] Sensitive data properly encrypted
- [ ] Dependencies are up-to-date and secure
```

### 4. Performance

```markdown
- [ ] No N+1 query problems
- [ ] Appropriate data structures used
- [ ] Database queries optimized
- [ ] Caching used where appropriate
- [ ] No memory leaks
- [ ] Async operations handled correctly
- [ ] Resource cleanup (close files, connections)
```

### 5. Testing

```markdown
- [ ] Unit tests cover new functionality
- [ ] Edge cases are tested
- [ ] Integration tests for API changes
- [ ] Test names are descriptive
- [ ] Tests are maintainable
- [ ] No test dependencies or flaky tests
- [ ] Code coverage meets threshold (>80%)
```

### 6. Documentation

```markdown
- [ ] Public APIs documented
- [ ] Complex logic has comments explaining why
- [ ] README updated if needed
- [ ] CHANGELOG updated
- [ ] API documentation updated
- [ ] Migration guides if breaking changes
```

### 7. Style & Conventions

```markdown
- [ ] Follows project coding style
- [ ] Linter passes
- [ ] Consistent formatting
- [ ] No commented-out code
- [ ] No console.log/debug statements
- [ ] Imports organized properly
```

## Review Process

### Step 1: Understand Context

```markdown
## Before Reviewing

1. Read the PR description
2. Understand the feature/bug being addressed
3. Review related spec.md or issue
4. Check related files for context
5. Understand the acceptance criteria
```

### Step 2: High-Level Review

```markdown
## Architecture Check

- Does the approach make sense?
- Are there better patterns to use?
- Is it consistent with existing code?
- Are dependencies appropriate?
- Is it in the right place in the codebase?
```

### Step 3: Line-by-Line Review

```markdown
## Detailed Review

Go through each file:
1. Check for logic errors
2. Verify error handling
3. Look for security issues
4. Check performance implications
5. Assess code quality
6. Verify tests
```

### Step 4: Testing

```markdown
## Verify Tests

1. Run the test suite
2. Check coverage report
3. Review test quality
4. Test edge cases manually if needed
5. Verify no regressions
```

### Step 5: Provide Feedback

```markdown
## Feedback Guidelines

✅ **DO**:
- Be specific with file:line references
- Explain the "why" behind suggestions
- Offer alternatives
- Highlight good practices
- Be constructive and respectful

❌ **DON'T**:
- Be vague ("this looks wrong")
- Just say "change this" without explaining
- Be condescending or dismissive
- Focus only on negatives
- Nitpick style if linter handles it
```

## Comment Templates

### Security Issue (Critical)

```markdown
🔴 **Security Risk** [file.js:42]

**Issue**: User input is directly used in SQL query without sanitization.

**Impact**: SQL injection vulnerability could expose or modify database.

**Suggested Fix**:
```js
// Instead of:
const query = `SELECT * FROM users WHERE id = ${userId}`;

// Use parameterized query:
const query = 'SELECT * FROM users WHERE id = ?';
db.query(query, [userId]);
```

**References**: [OWASP SQL Injection](https://owasp.org/...)
```

### Performance Issue (Major)

```markdown
🟡 **Performance Concern** [file.js:123]

**Issue**: N+1 query problem in loop - querying database for each item.

**Impact**: 1000 items = 1000 database queries = slow response time.

**Suggested Fix**:
```js
// Instead of:
for (const item of items) {
  const details = await db.getDetails(item.id);
}

// Fetch all at once:
const ids = items.map(i => i.id);
const details = await db.getDetailsBatch(ids);
```

**Estimated Impact**: Reduces API response time from 5s to 50ms
```

### Code Quality (Minor)

```markdown
💡 **Suggestion** [file.js:67]

**Issue**: Function is doing too many things (checking auth, validating input, updating DB, sending email).

**Suggested Refactoring**:
```js
async function updateUserProfile(userId, data) {
  await validateAuth(userId);
  const validData = validateProfileData(data);
  const updated = await db.updateUser(userId, validData);
  await emailService.sendUpdateNotification(userId);
  return updated;
}
```

This improves:
- Testability (each function can be tested independently)
- Reusability (auth/validation can be used elsewhere)
- Readability (clear separation of concerns)
```

### Positive Feedback

```markdown
✅ **Well Done** [file.js:200-250]

Great use of the builder pattern here! This makes the complex object construction much more readable and maintainable. The fluent interface is intuitive.

Also appreciate the comprehensive error handling with specific error messages.
```

## Red Flags

Watch out for these common issues:

### Critical Red Flags 🔴

```markdown
- Hardcoded credentials or secrets
- SQL injection vulnerabilities
- XSS vulnerabilities
- Missing authentication checks
- Unencrypted sensitive data
- Infinite loops or recursion without base case
- Memory leaks (unclosed resources)
```

### Major Concerns 🟡

```markdown
- N+1 query problems
- Missing error handling
- No input validation
- Overly complex logic (high cyclomatic complexity)
- Large functions (>100 lines)
- No tests for critical logic
- Breaking changes without migration
```

### Minor Issues 💡

```markdown
- Magic numbers without constants
- Poor variable naming
- Code duplication
- Missing comments for complex logic
- Inconsistent formatting (if linter doesn't catch)
- TODO comments without tickets
```

## Output Format

```markdown
## Code Review: [PR Title]

**Reviewer**: [Your Name]
**Date**: YYYY-MM-DD
**Status**: ✅ APPROVED | 🟡 APPROVED WITH COMMENTS | 🔴 CHANGES REQUESTED

### Summary
[Brief overview of the changes and overall assessment]

### Critical Issues 🔴
[Must be fixed before merging]

1. **[Issue Title]** [file:line]
   - **Problem**: [What's wrong]
   - **Impact**: [Why it matters]
   - **Fix**: [How to address it]

### Major Concerns 🟡
[Should be addressed]

1. **[Concern Title]** [file:line]
   - **Issue**: [What could be better]
   - **Suggestion**: [Recommended improvement]

### Minor Suggestions 💡
[Nice to have]

1. **[Suggestion Title]** [file:line]
   - **Opportunity**: [Potential improvement]

### Positive Highlights ✅
[What was done well]

- [Good practice 1]
- [Good practice 2]

### Testing Notes
- [ ] All tests pass
- [ ] Coverage: [X]%
- [ ] Manual testing: [Results]

### Next Steps
[What needs to happen before this can merge]
```

## Examples

### Example 1: Approving Good Code

```markdown
## Code Review: Add User Authentication

**Status**: ✅ APPROVED

### Summary
Excellent implementation of OAuth2 authentication. Code is clean, well-tested, and follows security best practices.

### Positive Highlights ✅

- Proper use of bcrypt for password hashing (file.js:45)
- Comprehensive input validation (file.js:67)
- Great test coverage (95%)
- Clear error messages for users
- Good separation of concerns (auth logic in dedicated service)

### Minor Suggestions 💡

1. **Consider rate limiting** [auth.js:100]
   - Add rate limiting to prevent brute force attacks
   - Suggestion: 5 attempts per 15 minutes per IP

### Next Steps
No blocking issues. Ready to merge! 🚀
```

### Example 2: Requesting Changes

```markdown
## Code Review: Update Payment Processing

**Status**: 🔴 CHANGES REQUESTED

### Summary
The payment processing logic has some security concerns and performance issues that need to be addressed before merging.

### Critical Issues 🔴

1. **Credit card numbers logged in plaintext** [payment.js:123]
   - **Problem**: `console.log(cardNumber)` exposes sensitive data
   - **Impact**: PCI compliance violation, security risk
   - **Fix**: Remove logging or log only last 4 digits: `***-${cardNumber.slice(-4)}`

2. **Missing error handling for payment gateway** [payment.js:200]
   - **Problem**: Network errors will crash the application
   - **Impact**: Poor user experience, potential data loss
   - **Fix**: Wrap in try/catch and handle failures gracefully

### Major Concerns 🟡

1. **Synchronous payment processing** [payment.js:150-200]
   - **Issue**: Blocks the event loop for up to 30 seconds
   - **Impact**: Other requests will queue up and timeout
   - **Suggestion**: Move to async queue (e.g., Bull, RabbitMQ)

### Next Steps
1. Address critical security issues
2. Add error handling
3. Consider async processing (can be follow-up PR)
```

## Tips for Effective Reviews

1. **Review in small batches**: Effectiveness drops after 60 minutes
2. **Review < 400 lines at a time**: Catch 70-90% of defects
3. **Use a checklist**: Ensures consistency
4. **Balance speed and thoroughness**: Based on change criticality
5. **Automate what you can**: Use linters, static analysis, automated tests
6. **Focus on important things**: Don't nitpick style if linter handles it
7. **Be constructive**: Frame as learning opportunities
8. **Ask questions**: "Could you explain why..." vs "This is wrong"
