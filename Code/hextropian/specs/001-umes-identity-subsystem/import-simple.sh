#!/usr/bin/env bash
set -euo pipefail

# Simple Beads import for UMES tasks
# Extracts [T###] [P#] [STORY] and creates Beads issue

DRY_RUN=${1:-false}
TASKS_FILE="tasks.md"
OUTPUT_MAP="beads-mapping.txt"

> "$OUTPUT_MAP"

echo "=== UMES Beads Import ==="
echo "Dry run: $DRY_RUN"
echo ""

TASK_COUNT=0

# Read each task line
while IFS= read -r line; do
    if [[ "$line" =~ ^-[[:space:]]\[[[:space:]]\][[:space:]]\[([T][0-9]+[a-z]?)\][[:space:]]\[([P][0-9])\][[:space:]]\[([A-Z0-9]+)\] ]]; then
        TASK_ID="${BASH_REMATCH[1]}"
        PRIORITY="${BASH_REMATCH[2]}"
        STORY="${BASH_REMATCH[3]}"
        
        # Extract description (everything after the story marker)
        DESC=$(echo "$line" | sed -E 's/^- \[ \] \[[T][0-9]+[a-z]?\] \[[P][0-9]\] \[[A-Z0-9]+\] (\[P\] )?//' | cut -d'-' -f1 | xargs)
        
        ((TASK_COUNT++))
        
        # Map priority to labels
        LABELS="umes,priority-${PRIORITY,,}"
        
        # Add story label
        case "$STORY" in
            SETUP) LABELS="$LABELS,phase-1,setup" ;;
            FOUNDATION) LABELS="$LABELS,phase-2,foundation" ;;
            US1) LABELS="$LABELS,phase-3,us1" ;;
            US2) LABELS="$LABELS,phase-4,us2" ;;
            US3) LABELS="$LABELS,phase-5,us3" ;;
            US4) LABELS="$LABELS,phase-6,us4" ;;
            US5) LABELS="$LABELS,phase-7,us5" ;;
            US6) LABELS="$LABELS,phase-8,us6" ;;
            US7) LABELS="$LABELS,phase-9,us7" ;;
            POLISH) LABELS="$LABELS,phase-10,polish" ;;
        esac
        
        # Determine task type
        if [[ "$DESC" =~ ^Write.*test ]]; then
            TYPE="test"
            LABELS="$LABELS,test"
        else
            TYPE="task"
            LABELS="$LABELS,implementation"
        fi
        
        TITLE="[$TASK_ID] $DESC"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[$TASK_COUNT] $TASK_ID: bd create --title=\"$TITLE\" --type=$TYPE --labels=\"$LABELS\""
            echo "$TASK_ID -> bd-dry-run-$TASK_COUNT" >> "$OUTPUT_MAP"
        else
            echo "[$TASK_COUNT] Creating $TASK_ID..."
            BD_OUTPUT=$(bd create --title="$TITLE" --type="$TYPE" --labels="$LABELS" --status=todo 2>&1 || echo "FAILED")
            
            if BEADS_ID=$(echo "$BD_OUTPUT" | grep -oE 'beads-[0-9a-zA-Z]+' | head -1); then
                echo "  ✓ $TASK_ID -> $BEADS_ID" | tee -a "$OUTPUT_MAP"
            else
                echo "  ✗ $TASK_ID -> FAILED" | tee -a "$OUTPUT_MAP"
            fi
        fi
        
        # Rate limit
        if (( TASK_COUNT % 10 == 0 )); then
            echo "  (Pausing 2s...)"
            sleep 2
        fi
    fi
done < "$TASKS_FILE"

echo ""
echo "=== Complete ==="
echo "Total: $TASK_COUNT tasks"
