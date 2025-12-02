#!/usr/bin/env bash
#
# import-tasks-to-beads.sh
# Bulk import UMES tasks from tasks.md into Beads persistent task memory
#
# Usage:
#   ./import-tasks-to-beads.sh [--dry-run] [--start TASK_ID] [--end TASK_ID]
#
# Options:
#   --dry-run: Print commands without executing
#   --start: Start importing from this task ID (e.g., T001)
#   --end: Stop importing at this task ID (e.g., T050)
#
# Output:
#   Creates Beads issues for all tasks in tasks.md
#   Prints mapping of TaskID -> Beads ID for updating tasks.md
#

set -euo pipefail

# Configuration
FEATURE_DIR="/Users/speed/code/hextropian/specs/001-umes-identity-subsystem"
TASKS_FILE="${FEATURE_DIR}/tasks.md"
DRY_RUN=false
START_TASK=""
END_TASK=""
OUTPUT_MAP="${FEATURE_DIR}/beads-mapping.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --start)
            START_TASK="$2"
            shift 2
            ;;
        --end)
            END_TASK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--start TASK_ID] [--end TASK_ID]"
            exit 1
            ;;
    esac
done

# Validate tasks.md exists
if [[ ! -f "$TASKS_FILE" ]]; then
    echo "ERROR: tasks.md not found at $TASKS_FILE"
    exit 1
fi

# Initialize output map
> "$OUTPUT_MAP"

echo "=== UMES Beads Import Script ==="
echo "Tasks file: $TASKS_FILE"
echo "Dry run: $DRY_RUN"
echo "Output map: $OUTPUT_MAP"
echo ""

# Function to create Beads issue
create_beads_issue() {
    local task_id="$1"
    local priority="$2"
    local story="$3"
    local description="$4"

    # Map priority to Beads labels
    local labels="umes"
    case "$priority" in
        P1) labels="$labels,mvp,priority-high" ;;
        P2) labels="$labels,important,priority-medium" ;;
        P3) labels="$labels,nice-to-have,priority-low" ;;
    esac

    # Add story label
    case "$story" in
        SETUP) labels="$labels,phase-1-setup" ;;
        FOUNDATION) labels="$labels,phase-2-foundation" ;;
        US1) labels="$labels,phase-3-us1,cloud-agnostic" ;;
        US2) labels="$labels,phase-4-us2,authentication" ;;
        US3) labels="$labels,phase-5-us3,authorization" ;;
        US4) labels="$labels,phase-6-us4,sdk" ;;
        US5) labels="$labels,phase-7-us5,api-keys" ;;
        US6) labels="$labels,phase-8-us6,audit" ;;
        US7) labels="$labels,phase-9-us7,multi-tenant" ;;
        POLISH) labels="$labels,phase-10-polish" ;;
    esac

    # Determine task type (test vs implementation)
    local task_type="task"
    if [[ "$description" =~ "Write unit tests" ]] || \
       [[ "$description" =~ "Write integration tests" ]] || \
       [[ "$description" =~ "Write smoke test" ]] || \
       [[ "$description" =~ "Write contract tests" ]] || \
       [[ "$description" =~ "Write security test" ]] || \
       [[ "$description" =~ "Write performance test" ]]; then
        task_type="test"
        labels="$labels,test"
    else
        labels="$labels,implementation"
    fi

    # Create Beads issue
    local title="[$task_id] $description"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] bd create --title=\"$title\" --type=$task_type --labels=\"$labels\" --status=todo"
        echo "$task_id -> bd-dry-run-id" >> "$OUTPUT_MAP"
    else
        # Execute bd create and capture output
        local bd_output
        bd_output=$(bd create --title="$title" --type="$task_type" --labels="$labels" --status=todo 2>&1)

        # Extract Beads ID from output (format: "Created issue beads-xxx")
        local beads_id
        if beads_id=$(echo "$bd_output" | grep -oE 'beads-[0-9a-f]+' | head -1); then
            echo "$task_id -> $beads_id" | tee -a "$OUTPUT_MAP"
        else
            echo "WARNING: Could not extract Beads ID for $task_id"
            echo "$task_id -> FAILED" >> "$OUTPUT_MAP"
        fi
    fi
}

# Parse tasks.md and create Beads issues
echo "Parsing tasks.md..."
echo ""

TASK_COUNT=0
IMPORTING=false

# Skip to start task if specified
if [[ -n "$START_TASK" ]]; then
    IMPORTING=false
else
    IMPORTING=true
fi

while IFS= read -r line; do
    # Match task lines: - [ ] [T###] [P#] [STORY] Description...
    if [[ "$line" =~ ^-[[:space:]]\[[[:space:]]\][[:space:]]\[([T][0-9]+[a-z]?)\][[:space:]]\[([P][0-9])\][[:space:]]\[([A-Z0-9]+)\][[:space:]](.+)$ ]]; then
        TASK_ID="${BASH_REMATCH[1]}"
        PRIORITY="${BASH_REMATCH[2]}"
        STORY="${BASH_REMATCH[3]}"
        DESCRIPTION="${BASH_REMATCH[4]}"

        # Check start/end bounds
        if [[ -n "$START_TASK" && "$TASK_ID" == "$START_TASK" ]]; then
            IMPORTING=true
        fi

        if [[ -n "$END_TASK" && "$TASK_ID" == "$END_TASK" ]]; then
            IMPORTING=false
        fi

        # Import if within bounds
        if [[ "$IMPORTING" == "true" ]]; then
            ((TASK_COUNT++))
            echo "[$TASK_COUNT] Importing $TASK_ID ($PRIORITY, $STORY)..."
            create_beads_issue "$TASK_ID" "$PRIORITY" "$STORY" "$DESCRIPTION"

            # Rate limit: pause every 10 tasks to avoid overwhelming Beads
            if (( TASK_COUNT % 10 == 0 )); then
                echo "  (Pausing 2s to avoid rate limits...)"
                sleep 2
            fi
        fi

        # Stop if we've reached end task
        if [[ -n "$END_TASK" && "$TASK_ID" == "$END_TASK" ]]; then
            break
        fi
    fi
done < "$TASKS_FILE"

echo ""
echo "=== Import Complete ==="
echo "Total tasks imported: $TASK_COUNT"
echo "Mapping saved to: $OUTPUT_MAP"
echo ""

if [[ "$DRY_RUN" == "false" ]]; then
    echo "Next steps:"
    echo "1. Review mapping in $OUTPUT_MAP"
    echo "2. Run: ./update-tasks-with-beads-ids.sh (to update tasks.md with Beads IDs)"
    echo "3. Set up dependencies: ./create-beads-dependencies.sh"
else
    echo "This was a dry run. Re-run without --dry-run to create actual Beads issues."
fi
