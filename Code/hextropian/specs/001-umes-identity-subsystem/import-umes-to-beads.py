#!/usr/bin/env python3
"""
import-umes-to-beads.py
Bulk import UMES tasks from tasks.md into Beads persistent task memory

Usage:
    ./import-umes-to-beads.py [--dry-run] [--start T013] [--end T288] [--resume]

Options:
    --dry-run: Preview commands without executing
    --start: Start from task ID (default: T013, Phase 1 already imported)
    --end: End at task ID (default: T288)
    --resume: Resume from last successful import (reads beads-mapping.txt)
    --phase N: Import only Phase N (1-11)

Features:
    - Intelligent task parsing with regex
    - Rate limiting (10 tasks/batch, 2s pause)
    - Progress tracking with ETA
    - Mapping file generation (TaskID -> Beads ID)
    - Resume capability for interrupted imports
    - Dry run mode for validation
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Configuration
TASKS_FILE = Path("tasks.md")
MAPPING_FILE = Path("beads-mapping.txt")
RATE_LIMIT_BATCH = 10
RATE_LIMIT_DELAY = 2.0

# Phase mappings
PHASE_MAP = {
    'SETUP': ('phase-1', 'T001', 'T012'),
    'FOUNDATION': ('phase-2', 'T013', 'T060'),
    'US1': ('phase-3', 'T061', 'T088'),
    'US2': ('phase-4', 'T089', 'T120'),
    'US3': ('phase-5', 'T121', 'T150'),
    'US4': ('phase-6', 'T151', 'T172'),
    'US5': ('phase-7', 'T173', 'T196'),
    'US6': ('phase-8', 'T197', 'T216'),
    'US7': ('phase-9', 'T217', 'T233'),
    'POLISH': ('phase-10', 'T234', 'T260'),
}

PRIORITY_MAP = {'P1': '1', 'P2': '2', 'P3': '3'}


class TaskImporter:
    """Imports UMES tasks from tasks.md to Beads."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.mapping = {}
        self.imported_count = 0
        self.failed_count = 0
        self.skipped_count = 0

        # Load existing mapping for resume capability
        if MAPPING_FILE.exists():
            self._load_existing_mapping()

    def _load_existing_mapping(self):
        """Load existing TaskID -> Beads ID mapping."""
        print(f"📂 Loading existing mapping from {MAPPING_FILE}...")
        with open(MAPPING_FILE, 'r') as f:
            for line in f:
                if ' -> ' in line:
                    task_id, beads_id = line.strip().split(' -> ')
                    if beads_id != 'FAILED':
                        self.mapping[task_id] = beads_id
        print(f"   Found {len(self.mapping)} existing tasks\n")

    def parse_tasks(self) -> List[Tuple[str, str, str, str]]:
        """
        Parse all tasks from tasks.md.

        Returns:
            List of (task_id, priority, story, description) tuples
        """
        if not TASKS_FILE.exists():
            print(f"❌ Error: {TASKS_FILE} not found")
            sys.exit(1)

        tasks = []
        # Pattern: - [ ] [T###] [P#] [STORY] [P]? Description - more details
        pattern = r'^- \[ \] \[([T][0-9]+[a-z]?)\] \[([P][0-9])\] \[([A-Z0-9]+)\] (\[P\] )?(.*?)(?:\s-\s|$)'

        with open(TASKS_FILE, 'r') as f:
            for line in f:
                match = re.match(pattern, line)
                if match:
                    task_id = match.group(1)
                    priority = match.group(2)
                    story = match.group(3)
                    description = match.group(5).strip()

                    # Truncate description to reasonable length
                    if len(description) > 120:
                        description = description[:117] + "..."

                    tasks.append((task_id, priority, story, description))

        print(f"📋 Parsed {len(tasks)} tasks from {TASKS_FILE}")
        return tasks

    def extract_task_number(self, task_id: str) -> int:
        """Extract numeric part from task ID (e.g., T001 -> 1, T181b -> 181)."""
        return int(re.search(r'\d+', task_id).group())

    def get_phase_label(self, story: str) -> str:
        """Map story to phase label."""
        phase_info = PHASE_MAP.get(story)
        if phase_info:
            return phase_info[0]
        # Fallback for deployment tasks
        return 'phase-11'

    def should_import(self, task_id: str, start: Optional[str], end: Optional[str]) -> bool:
        """Check if task should be imported based on range."""
        # Skip if already imported
        if task_id in self.mapping:
            return False

        if start is None and end is None:
            return True

        task_num = self.extract_task_number(task_id)
        start_num = self.extract_task_number(start) if start else 1
        end_num = self.extract_task_number(end) if end else 999

        return start_num <= task_num <= end_num

    def create_beads_issue(self, task_id: str, priority: str, story: str, description: str) -> Optional[str]:
        """
        Create a Beads issue.

        Returns:
            Beads ID if successful, None if failed
        """
        # Build labels
        phase = self.get_phase_label(story)
        labels = f"umes,{phase},{story.lower()},{priority.lower()}"

        # Add type-specific label
        if 'write' in description.lower() and 'test' in description.lower():
            labels += ",test"
        else:
            labels += ",implementation"

        # Build title
        title = f"[{task_id}] {description}"

        # Get priority number
        priority_num = PRIORITY_MAP.get(priority, '2')

        # Build command
        cmd = [
            'bd', 'create',
            '--title', title,
            '--type', 'task',
            '--labels', labels,
            '--priority', priority_num
        ]

        if self.dry_run:
            print(f"  [DRY-RUN] bd create --title=\"{title}\" --labels=\"{labels}\" --priority={priority_num}")
            return f"bd-dry-{task_id}"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Extract Beads ID from output
                match = re.search(r'hextropian-[0-9a-z]+', result.stdout)
                if match:
                    beads_id = match.group(0)
                    return beads_id
                else:
                    print(f"  ⚠️  Warning: Created but couldn't extract Beads ID")
                    return None
            else:
                print(f"  ❌ Error: {result.stderr.strip()}")
                return None

        except subprocess.TimeoutExpired:
            print(f"  ⏱️  Timeout creating task")
            return None
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return None

    def save_mapping(self, task_id: str, beads_id: Optional[str]):
        """Append mapping to file."""
        with open(MAPPING_FILE, 'a') as f:
            if beads_id:
                f.write(f"{task_id} -> {beads_id}\n")
            else:
                f.write(f"{task_id} -> FAILED\n")

    def import_tasks(self, tasks: List[Tuple[str, str, str, str]], start: Optional[str], end: Optional[str]):
        """Import tasks to Beads with progress tracking."""
        total_tasks = len([t for t in tasks if self.should_import(t[0], start, end)])

        if total_tasks == 0:
            print("✅ No tasks to import (all already exist or outside range)")
            return

        print(f"\n🚀 Starting import of {total_tasks} tasks...")
        print(f"   Rate limit: {RATE_LIMIT_BATCH} tasks per batch, {RATE_LIMIT_DELAY}s pause\n")

        start_time = time.time()
        current_batch = 0

        for i, (task_id, priority, story, description) in enumerate(tasks, 1):
            # Skip if outside range or already imported
            if not self.should_import(task_id, start, end):
                if task_id in self.mapping:
                    self.skipped_count += 1
                continue

            # Progress indicator
            progress = (self.imported_count + 1) / total_tasks * 100
            elapsed = time.time() - start_time
            eta = (elapsed / max(self.imported_count, 1)) * (total_tasks - self.imported_count)

            print(f"[{self.imported_count + 1}/{total_tasks}] ({progress:.1f}%) {task_id}: {description[:60]}...")

            # Create Beads issue
            beads_id = self.create_beads_issue(task_id, priority, story, description)

            if beads_id:
                print(f"  ✓ {beads_id}")
                self.mapping[task_id] = beads_id
                self.imported_count += 1
            else:
                print(f"  ✗ Failed")
                self.failed_count += 1

            # Save mapping immediately (for resume capability)
            self.save_mapping(task_id, beads_id)

            # Rate limiting
            current_batch += 1
            if current_batch >= RATE_LIMIT_BATCH:
                print(f"  ⏸️  Pausing {RATE_LIMIT_DELAY}s (rate limit)... ETA: {int(eta)}s")
                time.sleep(RATE_LIMIT_DELAY)
                current_batch = 0

    def print_summary(self):
        """Print import summary statistics."""
        print("\n" + "=" * 60)
        print("📊 Import Summary")
        print("=" * 60)
        print(f"✅ Successfully imported: {self.imported_count}")
        print(f"⏭️  Skipped (already exist): {self.skipped_count}")
        print(f"❌ Failed: {self.failed_count}")
        print(f"📁 Mapping file: {MAPPING_FILE}")
        print(f"📋 Total in mapping: {len(self.mapping)}")
        print("=" * 60)

        if self.failed_count > 0:
            print("\n⚠️  Some tasks failed to import. Check beads-mapping.txt for FAILED entries.")
            print("   Re-run with --resume to retry failed tasks.")

        if not self.dry_run and self.imported_count > 0:
            print("\n✨ Next steps:")
            print("   1. Verify import: bd list --label=umes")
            print("   2. Check ready tasks: bd ready")
            print("   3. Update tasks.md with Beads IDs")
            print("   4. Start working: bd update <id> --status=in_progress")


def main():
    parser = argparse.ArgumentParser(
        description='Import UMES tasks from tasks.md to Beads',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview
  ./import-umes-to-beads.py --dry-run

  # Import Phase 2 tasks (T013-T060)
  ./import-umes-to-beads.py --start T013 --end T060

  # Import all remaining tasks
  ./import-umes-to-beads.py --start T013

  # Resume after interruption
  ./import-umes-to-beads.py --resume

  # Import specific phase
  ./import-umes-to-beads.py --phase 2
        """
    )

    parser.add_argument('--dry-run', action='store_true', help='Preview without executing')
    parser.add_argument('--start', type=str, help='Start from task ID (e.g., T013)')
    parser.add_argument('--end', type=str, help='End at task ID (e.g., T060)')
    parser.add_argument('--resume', action='store_true', help='Resume from last import')
    parser.add_argument('--phase', type=int, choices=range(1, 12), help='Import only Phase N')

    args = parser.parse_args()

    # Handle --phase flag
    if args.phase:
        # Find phase range from PHASE_MAP
        phase_ranges = {
            1: ('T001', 'T012'),
            2: ('T013', 'T060'),
            3: ('T061', 'T088'),
            4: ('T089', 'T120'),
            5: ('T121', 'T150'),
            6: ('T151', 'T172'),
            7: ('T173', 'T196'),
            8: ('T197', 'T216'),
            9: ('T217', 'T233'),
            10: ('T234', 'T260'),
            11: ('T261', 'T288'),
        }
        args.start, args.end = phase_ranges[args.phase]
        print(f"📌 Phase {args.phase} selected: {args.start} to {args.end}\n")

    # Initialize importer
    importer = TaskImporter(dry_run=args.dry_run)

    # Set defaults
    if args.resume:
        # Resume from last successful import
        if importer.mapping:
            last_imported = max(importer.mapping.keys(), key=lambda x: importer.extract_task_number(x))
            last_num = importer.extract_task_number(last_imported)
            args.start = f"T{last_num + 1:03d}"
            print(f"🔄 Resuming from {args.start} (after {last_imported})\n")
        else:
            args.start = 'T013'
            print(f"🔄 No previous imports found, starting from {args.start}\n")
    elif args.start is None:
        args.start = 'T013'  # Default: Skip Phase 1 (already imported)

    if args.end is None:
        args.end = 'T288'

    # Parse tasks
    tasks = importer.parse_tasks()

    # Import tasks
    try:
        importer.import_tasks(tasks, args.start, args.end)
    except KeyboardInterrupt:
        print("\n\n⚠️  Import interrupted by user")
        print("   Progress saved to beads-mapping.txt")
        print("   Run with --resume to continue")

    # Print summary
    importer.print_summary()

    # Exit code
    sys.exit(1 if importer.failed_count > 0 else 0)


if __name__ == '__main__':
    main()
