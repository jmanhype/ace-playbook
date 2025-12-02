# UMES Beads Quick Start Guide

**Status**: ✅ Phase 1 (12 tasks) imported into Beads
**Remaining**: 276 tasks across Phases 2-11

---

## Current State

### Phase 1 Tasks (Complete in Beads)

| Task ID | Beads ID | Title |
|---------|----------|-------|
| T001 | hextropian-0u7 | Initialize backend project structure |
| T002 | hextropian-9d6 | Create pyproject.toml with all dependencies |
| T003 | hextropian-j4j | Configure pytest with asyncio support |
| T004 | hextropian-w2k | Set up Testcontainers fixtures |
| T005 | hextropian-hni | Create Docker Compose for local development |
| T006 | hextropian-406 | Configure SQLAlchemy async engine |
| T007 | hextropian-foz | Implement RLS context manager |
| T008 | hextropian-ht1 | Create base SQLAlchemy models |
| T009 | hextropian-54i | Configure two-level caching |
| T010 | hextropian-ezf | Implement circuit breaker factory |
| T011 | hextropian-8fw | Create configuration management |
| T012 | hextropian-jwp | Write smoke test for project setup |

---

## Quick Start Workflow

### 1. View Ready Tasks

```bash
cd /Users/speed/code/hextropian
bd ready
```

You should see 10-12 tasks from Phase 1 ready to work.

### 2. Start Working on a Task

```bash
# Pick the first task (T001)
bd show hextropian-0u7

# Claim it
bd update hextropian-0u7 --status=in_progress

# Do the work...
# (Create directory structure per tasks.md)

# Mark complete
bd close hextropian-0u7
```

### 3. Update tasks.md

After completing a task, check it off in tasks.md:

```markdown
- [x] (hextropian-0u7) [T001] [P1] [SETUP] Initialize backend project structure...
```

**Note**: You'll need to manually add the Beads ID `(hextropian-0u7)` to tasks.md for now.

---

## Importing Remaining Tasks

### Option A: Batch Import by Phase (Recommended)

Create tasks phase-by-phase as you complete earlier phases:

```bash
# After completing Phase 1, create Phase 2 tasks (T013-T060)
cd /Users/speed/code/hextropian

# Phase 2: Foundation (48 tasks)
for i in {13..60}; do
  LINE=$(grep "^\- \[ \] \[T$(printf "%03d" $i)\]" specs/001-umes-identity-subsystem/tasks.md)
  TITLE=$(echo "$LINE" | sed -E 's/.*\] //' | cut -d'-' -f1 | head -c 120)
  echo "Creating T$(printf "%03d" $i)..."
  bd create --title="[T$(printf "%03d" $i)] $TITLE" --type=task --labels="umes,phase-2,foundation,p1" --priority=1
  sleep 0.5
done
```

### Option B: Create Tasks On-Demand

Create tasks as you need them:

```bash
# When ready to work on T013, create it first
bd create --title="[T013] Write unit tests for User model" --type=task --labels="umes,phase-2,p1,test" --priority=1

# Then start working
bd update <beads-id> --status=in_progress
```

### Option C: Python Script for Bulk Import

Create `import-all-tasks.py`:

```python
#!/usr/bin/env python3
import re
import subprocess
import time

# Read tasks.md
with open('specs/001-umes-identity-subsystem/tasks.md', 'r') as f:
    lines = f.readlines()

# Extract tasks
tasks = []
for line in lines:
    match = re.match(r'^- \[ \] \[([T][0-9]+[a-z]?)\] \[([P][0-9])\] \[([A-Z0-9]+)\] (\[P\] )?(.*)', line)
    if match:
        task_id, priority, story, _, desc = match.groups()
        desc = desc.split(' - ')[0].strip()[:120]  # Truncate description
        tasks.append((task_id, priority, story, desc))

# Create Beads issues
for task_id, priority, story, desc in tasks[12:]:  # Skip Phase 1 (already done)
    phase_map = {
        'FOUNDATION': 'phase-2',
        'US1': 'phase-3',
        'US2': 'phase-4',
        'US3': 'phase-5',
        'US4': 'phase-6',
        'US5': 'phase-7',
        'US6': 'phase-8',
        'US7': 'phase-9',
        'POLISH': 'phase-10'
    }
    phase = phase_map.get(story, 'phase-11')

    pri_map = {'P1': '1', 'P2': '2', 'P3': '3'}
    pri_num = pri_map.get(priority, '2')

    labels = f"umes,{phase},{story.lower()},{priority.lower()}"
    title = f"[{task_id}] {desc}"

    cmd = f'bd create --title="{title}" --type=task --labels="{labels}" --priority={pri_num}'
    print(f"Creating {task_id}...")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✓ {task_id}")
        else:
            print(f"  ✗ {task_id}: {result.stderr}")
    except Exception as e:
        print(f"  ✗ {task_id}: {e}")

    time.sleep(0.5)  # Rate limit

print(f"\n✓ Complete! Created {len(tasks) - 12} tasks")
```

Run it:
```bash
chmod +x import-all-tasks.py
./import-all-tasks.py
```

---

## Setting Up Dependencies

After importing all tasks, create blocking dependencies:

```bash
# Phase 1 blocks Phase 2
bd dep hextropian-jwp <first-phase-2-task-id>

# Within phases: Tests block implementation
# Example: T013 (User model tests) blocks T014 (User model impl)
bd dep <t013-beads-id> <t014-beads-id>
```

**Note**: Setting up all 288 dependencies manually is tedious. Consider doing it incrementally as you work through phases.

---

## Daily Workflow

### Morning Routine

```bash
# Sync from main (if on ephemeral branch)
bd sync --from-main

# Check what's ready
bd ready

# Review in-progress work
bd list --status=in_progress
```

### Working on a Task

```bash
# 1. Claim task
bd update <beads-id> --status=in_progress

# 2. Work following TDD
#    - Write tests first (red)
#    - Implement minimum code (green)
#    - Refactor (clean)

# 3. Run tests
pytest backend/tests/unit/models/test_user.py

# 4. Mark complete when tests pass
bd close <beads-id>

# 5. Update tasks.md
# Mark checkbox: [x]
```

### End of Session

```bash
# Sync Beads changes
bd sync --from-main

# Commit work
git add .
git commit -m "feat(umes): Complete T001-T003 (project setup)"

# Note progress in tasks.md
# Update "Completed Tasks" count
```

---

## Task Mapping Reference

| Phase | Tasks | Story | Count | Priority |
|-------|-------|-------|-------|----------|
| 1 | T001-T012 | Setup | 12 | ✅ In Beads |
| 2 | T013-T060 | Foundation | 48 | ⏳ Import next |
| 3 | T061-T088 | US1 (Cloud) | 28 | ⏳ Import after Phase 2 |
| 4 | T089-T120 | US2 (Auth) | 32 | ⏳ Import after Phase 3 |
| 5 | T121-T150 | US3 (Authz) | 30 | ⏳ Import after Phase 4 |
| 6 | T151-T172 | US4 (SDK) | 22 | ⏳ Import after Phase 5 |
| 7 | T173-T196 | US5 (API Keys) | 24 | ⏳ Import after Phase 6 |
| 8 | T197-T216 | US6 (Audit) | 20 | ⏳ Import after Phase 7 |
| 9 | T217-T233 | US7 (Multi-tenant) | 17 | ⏳ Import after Phase 8 |
| 10 | T234-T260 | Polish | 27 | ⏳ Import after Phase 9 |
| 11 | T261-T288 | Deployment | 28 | ⏳ Import after Phase 10 |
| **Total** | **288 tasks** | **7 user stories** | **288** | **12 imported** |

---

## Useful Beads Commands

```bash
# List tasks by phase
bd list --label=phase-1 --status=open

# Show task details
bd show hextropian-0u7

# Find tasks ready to work
bd ready

# Update task status
bd update hextropian-0u7 --status=in_progress

# Add notes/discoveries
bd update hextropian-0u7 --notes="Found edge case: need to handle empty config"

# Close completed task
bd close hextropian-0u7

# View project statistics
bd stats

# Check for blocked tasks
bd list --status=blocked
```

---

## Tips

1. **Import incrementally** - Don't create all 288 tasks at once. Create them phase-by-phase as you complete work.

2. **Keep tasks.md and Beads in sync** - Update both after completing tasks:
   - Beads: `bd close <id>`
   - tasks.md: Change `- [ ]` to `- [x]`

3. **Use labels for filtering**:
   - `--label=phase-2` - Show Phase 2 tasks
   - `--label=test` - Show only test tasks
   - `--label=p1` - Show only P1 priority

4. **Track discoveries** - When you find new work during implementation:
   ```bash
   bd create --title="Fix edge case in token validation" --type=bug --labels="umes,us2" --priority=1
   ```

5. **Use Beads for session memory** - Add notes with `--notes` to preserve context across sessions

---

## Next Steps

1. ✅ **Start Phase 1** - Run `bd ready` and begin with T001
2. ⏳ **Import Phase 2** - After completing Phase 1, create Phase 2 tasks (T013-T060)
3. ⏳ **Set up dependencies** - As you import phases, add blocking relationships
4. 🚀 **Implement with TDD** - Follow constitution principle II (Test-First Development)

---

**Last Updated**: 2025-12-02
**Phase 1 Status**: ✅ 12/12 tasks in Beads
**Overall Status**: 12/288 tasks imported (4.2%)
