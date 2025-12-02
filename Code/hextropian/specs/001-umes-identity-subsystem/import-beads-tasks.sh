#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${1:-false}"
START_NUM="${2:-1}"
END_NUM="${3:-288}"

OUTPUT_MAP="beads-mapping.txt"
: > "$OUTPUT_MAP"

echo "=== UMES Beads Import ==="
echo "Dry run: $DRY_RUN"
echo "Range: T$(printf "%03d" $START_NUM) to T$(printf "%03d" $END_NUM)"
echo ""

TASK_COUNT=0
CURRENT_NUM=0

# Extract all task lines
grep -E "^- \[ \] \[T[0-9]" tasks.md | while IFS= read -r line; do
    # Parse task ID
    TASK_ID=$(echo "$line" | grep -oE '\[T[0-9]+[a-z]?\]' | tr -d '[]')
    
    # Extract number from task ID for range check
    TASK_NUM=$(echo "$TASK_ID" | grep -oE '[0-9]+')
    
    # Skip if outside range
    if (( TASK_NUM < START_NUM || TASK_NUM > END_NUM )); then
        continue
    fi
    
    # Extract priority
    PRIORITY=$(echo "$line" | grep -oE '\[P[0-9]\]' | tr -d '[]')
    
    # Extract story
    STORY=$(echo "$line" | grep -oE '\](SETUP|FOUNDATION|US[0-9]|POLISH)\]' | tr -d '][')
    
    # Extract description (first part before dash separator)
    DESC=$(echo "$line" | sed -E 's/^- \[ \] \[T[0-9]+[a-z]?\] \[P[0-9]\] \[[A-Z0-9]+\] (\[P\] )?//' | cut -d'-' -f1 | sed 's/[[:space:]]*$//')
    
    ((TASK_COUNT++))
    
    # Build labels
    LABELS="umes"
    case "$PRIORITY" in
        P1) LABELS="$LABELS,mvp,priority-high" ;;
        P2) LABELS="$LABELS,important,priority-medium" ;;
        P3) LABELS="$LABELS,nice-to-have,priority-low" ;;
    esac
    
    case "$STORY" in
        SETUP) LABELS="$LABELS,phase-1" ;;
        FOUNDATION) LABELS="$LABELS,phase-2" ;;
        US1) LABELS="$LABELS,phase-3" ;;
        US2) LABELS="$LABELS,phase-4" ;;
        US3) LABELS="$LABELS,phase-5" ;;
        US4) LABELS="$LABELS,phase-6" ;;
        US5) LABELS="$LABELS,phase-7" ;;
        US6) LABELS="$LABELS,phase-8" ;;
        US7) LABELS="$LABELS,phase-9" ;;
        POLISH) LABELS="$LABELS,phase-10" ;;
    esac
    
    # Determine type
    if [[ "$DESC" =~ [Ww]rite.*test ]]; then
        TYPE="test"
        LABELS="$LABELS,test"
    else
        TYPE="task"
        LABELS="$LABELS,implementation"
    fi
    
    TITLE="[$TASK_ID] $DESC"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[$TASK_COUNT] $TASK_ID: $DESC"
        echo "$TASK_ID -> bd-dry-$TASK_COUNT" >> "$OUTPUT_MAP"
    else
        echo "[$TASK_COUNT/$((END_NUM - START_NUM + 1))] Creating $TASK_ID..."
        
        BD_OUTPUT=$(bd create --title="$TITLE" --type="$TYPE" --labels="$LABELS" --status=todo 2>&1 || echo "FAILED")
        
        if BEADS_ID=$(echo "$BD_OUTPUT" | grep -oE 'beads-[0-9a-zA-Z]+' | head -1); then
            echo "  ✓ $BEADS_ID" | tee -a "$OUTPUT_MAP"
        else
            echo "  ✗ FAILED" | tee -a "$OUTPUT_MAP"
        fi
        
        # Rate limit every 10
        if (( TASK_COUNT % 10 == 0 )); then
            sleep 1
        fi
    fi
done

echo ""
echo "=== Complete ==="
echo "Processed: $TASK_COUNT tasks"
echo "Mapping: $OUTPUT_MAP"
