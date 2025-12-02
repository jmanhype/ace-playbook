#!/usr/bin/env bash
#
# update-tasks-with-beads-ids.sh
# Update tasks.md with Beads issue IDs from the mapping file
#
# Usage:
#   ./update-tasks-with-beads-ids.sh [--dry-run]
#
# Reads: beads-mapping.txt (created by import-tasks-to-beads.sh)
# Updates: tasks.md with Beads IDs in format: - [ ] (bd-xxx) [T001] ...
#

set -euo pipefail

# Configuration
FEATURE_DIR="/Users/speed/code/hextropian/specs/001-umes-identity-subsystem"
TASKS_FILE="${FEATURE_DIR}/tasks.md"
MAPPING_FILE="${FEATURE_DIR}/beads-mapping.txt"
BACKUP_FILE="${TASKS_FILE}.backup-$(date +%Y%m%d-%H%M%S)"
DRY_RUN=false

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Validate files exist
if [[ ! -f "$TASKS_FILE" ]]; then
    echo "ERROR: tasks.md not found at $TASKS_FILE"
    exit 1
fi

if [[ ! -f "$MAPPING_FILE" ]]; then
    echo "ERROR: Mapping file not found at $MAPPING_FILE"
    echo "Run ./import-tasks-to-beads.sh first to generate the mapping."
    exit 1
fi

echo "=== Update tasks.md with Beads IDs ==="
echo "Tasks file: $TASKS_FILE"
echo "Mapping file: $MAPPING_FILE"
echo "Dry run: $DRY_RUN"
echo ""

# Create backup
if [[ "$DRY_RUN" == "false" ]]; then
    cp "$TASKS_FILE" "$BACKUP_FILE"
    echo "Backup created: $BACKUP_FILE"
fi

# Read mapping into associative array
declare -A task_to_beads
FAILED_COUNT=0

while IFS=' -> ' read -r task_id beads_id; do
    if [[ "$beads_id" == "FAILED" ]]; then
        ((FAILED_COUNT++))
        echo "WARNING: Skipping failed mapping for $task_id"
    else
        task_to_beads["$task_id"]="$beads_id"
    fi
done < "$MAPPING_FILE"

echo "Loaded ${#task_to_beads[@]} task-to-beads mappings"
echo "Failed mappings: $FAILED_COUNT"
echo ""

# Update tasks.md
UPDATED_COUNT=0
temp_file=$(mktemp)

while IFS= read -r line; do
    # Match task lines: - [ ] [T###] [P#] [STORY] Description...
    if [[ "$line" =~ ^(-[[:space:]]\[[[:space:]]\])[[:space:]](\[([T][0-9]+[a-z]?)\][[:space:]]\[([P][0-9])\][[:space:]]\[([A-Z0-9]+)\][[:space:]].+)$ ]]; then
        checkbox="${BASH_REMATCH[1]}"
        rest="${BASH_REMATCH[2]}"
        task_id="${BASH_REMATCH[3]}"

        # Look up Beads ID
        if [[ -n "${task_to_beads[$task_id]:-}" ]]; then
            beads_id="${task_to_beads[$task_id]}"
            # Format: - [ ] (bd-xxx) [T001] [P1] [STORY] Description...
            updated_line="$checkbox ($beads_id) $rest"
            echo "$updated_line" >> "$temp_file"
            ((UPDATED_COUNT++))

            if [[ "$DRY_RUN" == "true" ]]; then
                echo "[DRY-RUN] Would update $task_id -> $beads_id"
            fi
        else
            # No mapping found, keep original
            echo "$line" >> "$temp_file"
            echo "WARNING: No Beads ID found for $task_id, keeping original format"
        fi
    else
        # Not a task line, keep as-is
        echo "$line" >> "$temp_file"
    fi
done < "$TASKS_FILE"

# Replace original file
if [[ "$DRY_RUN" == "false" ]]; then
    mv "$temp_file" "$TASKS_FILE"
    echo ""
    echo "=== Update Complete ==="
    echo "Updated $UPDATED_COUNT tasks with Beads IDs"
    echo "Backup: $BACKUP_FILE"
    echo "Original tasks.md has been updated."
else
    rm "$temp_file"
    echo ""
    echo "=== Dry Run Complete ==="
    echo "Would update $UPDATED_COUNT tasks"
    echo "Re-run without --dry-run to apply changes."
fi

echo ""
echo "Next steps:"
echo "1. Review updated tasks.md"
echo "2. Commit changes: git add tasks.md && git commit -m 'feat: Add Beads issue IDs to tasks'"
echo "3. Set up dependencies: ./create-beads-dependencies.sh"
