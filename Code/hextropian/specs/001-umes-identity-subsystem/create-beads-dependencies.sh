#!/usr/bin/env bash
#
# create-beads-dependencies.sh
# Create Beads dependencies between tasks based on phase ordering
#
# Usage:
#   ./create-beads-dependencies.sh [--dry-run]
#
# Dependencies Created:
#   - Phase 1 (Setup) tasks block Phase 2 (Foundation)
#   - Phase 2 (Foundation) tasks block Phase 3-9 (User Stories)
#   - US1 (Phase 3) blocks US2-US7 (foundational cloud deployment)
#   - US2 (Phase 4) + US3 (Phase 5) block US4, US5, US7
#   - All user stories block Phase 10 (Polish)
#   - Phase 10 blocks Phase 11 (Deployment)
#   - Within each phase: Test tasks block implementation tasks
#

set -euo pipefail

# Configuration
FEATURE_DIR="/Users/speed/code/hextropian/specs/001-umes-identity-subsystem"
MAPPING_FILE="${FEATURE_DIR}/beads-mapping.txt"
DRY_RUN=false

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Validate mapping file exists
if [[ ! -f "$MAPPING_FILE" ]]; then
    echo "ERROR: Mapping file not found at $MAPPING_FILE"
    echo "Run ./import-tasks-to-beads.sh first to generate the mapping."
    exit 1
fi

echo "=== Create Beads Dependencies ==="
echo "Mapping file: $MAPPING_FILE"
echo "Dry run: $DRY_RUN"
echo ""

# Load task-to-beads mapping
declare -A task_to_beads
while IFS=' -> ' read -r task_id beads_id; do
    if [[ "$beads_id" != "FAILED" ]]; then
        task_to_beads["$task_id"]="$beads_id"
    fi
done < "$MAPPING_FILE"

echo "Loaded ${#task_to_beads[@]} task mappings"
echo ""

# Function to create dependency (from blocks to)
create_dependency() {
    local from_task="$1"
    local to_task="$2"
    local reason="$3"

    local from_beads="${task_to_beads[$from_task]:-}"
    local to_beads="${task_to_beads[$to_task]:-}"

    if [[ -z "$from_beads" || -z "$to_beads" ]]; then
        echo "  SKIP: Missing Beads ID for $from_task -> $to_task"
        return
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] bd dep $from_beads $to_beads  # $from_task blocks $to_task ($reason)"
    else
        if bd dep "$from_beads" "$to_beads" 2>&1 | grep -q "Dependency created"; then
            echo "  ✓ $from_task blocks $to_task ($reason)"
        else
            echo "  ✗ Failed: $from_task -> $to_task"
        fi
    fi
}

# Phase blocking dependencies
echo "Creating phase-level dependencies..."
echo ""

# Phase 1 (last task) blocks Phase 2 (first task)
create_dependency "T012" "T013" "Phase 1 setup blocks Phase 2 foundation"

# Phase 2 (last task) blocks Phase 3 (first task)
create_dependency "T060" "T061" "Phase 2 foundation blocks Phase 3 US1"

# US1 (Phase 3, last task) blocks all other user stories
create_dependency "T088" "T089" "US1 (cloud deployment) blocks US2 (auth)"
create_dependency "T088" "T121" "US1 (cloud deployment) blocks US3 (authz)"
create_dependency "T088" "T151" "US1 (cloud deployment) blocks US4 (SDK)"
create_dependency "T088" "T173" "US1 (cloud deployment) blocks US5 (API keys)"
create_dependency "T088" "T197" "US1 (cloud deployment) blocks US6 (audit)"
create_dependency "T088" "T217" "US1 (cloud deployment) blocks US7 (multi-tenant)"

# US2 (Phase 4, last task) + US3 (Phase 5, last task) block dependent stories
create_dependency "T120" "T151" "US2 (auth) blocks US4 (SDK needs auth)"
create_dependency "T150" "T151" "US3 (authz) blocks US4 (SDK needs authz)"
create_dependency "T120" "T173" "US2 (auth) blocks US5 (API keys need auth)"
create_dependency "T120" "T217" "US2 (auth) blocks US7 (multi-tenant needs auth)"
create_dependency "T150" "T217" "US3 (authz) blocks US7 (multi-tenant needs authz)"

# All user stories (last tasks) block Phase 10 (Polish)
create_dependency "T120" "T234" "US2 complete blocks Phase 10 polish"
create_dependency "T150" "T234" "US3 complete blocks Phase 10 polish"
create_dependency "T172" "T234" "US4 complete blocks Phase 10 polish"
create_dependency "T196" "T234" "US5 complete blocks Phase 10 polish"
create_dependency "T216" "T234" "US6 complete blocks Phase 10 polish"
create_dependency "T233" "T234" "US7 complete blocks Phase 10 polish"

# Phase 10 (last task) blocks Phase 11 (Deployment)
create_dependency "T260" "T261" "Phase 10 polish blocks Phase 11 deployment"

echo ""
echo "Creating test-before-implementation dependencies..."
echo ""

# Within-phase dependencies: Test tasks block implementation tasks
# Phase 2 examples (demonstrate pattern)
create_dependency "T013" "T014" "User model tests block implementation"
create_dependency "T015" "T016" "Tenant model tests block implementation"
create_dependency "T017" "T018" "Membership model tests block implementation"
create_dependency "T019" "T020" "Authorization model tests block implementation"
create_dependency "T021" "T022" "Token model tests block implementation"
create_dependency "T032" "T033" "KMS adapter contract tests block protocol definition"
create_dependency "T034" "T035" "LocalKMS tests block implementation"
create_dependency "T036" "T037" "AWS KMS tests block implementation"
create_dependency "T054" "T055" "JWT service tests block implementation"
create_dependency "T056" "T057" "Token validator tests block implementation"

# Phase 4 (US2) examples
create_dependency "T089" "T090" "Login endpoint tests block implementation"
create_dependency "T091" "T092" "Logout endpoint tests block implementation"
create_dependency "T093" "T094" "Refresh endpoint tests block implementation"
create_dependency "T101" "T102" "Token validate tests block implementation"

# Phase 5 (US3) examples
create_dependency "T121" "T122" "Entitlement evaluator tests block implementation"
create_dependency "T123" "T124" "Authz checker tests block implementation"
create_dependency "T125" "T126" "Authz check endpoint tests block implementation"

echo ""
echo "Creating critical path dependencies..."
echo ""

# KMS adapters must be complete before JWT service
create_dependency "T044" "T054" "KMS adapter factory blocks JWT service tests"

# IdP adapters must be complete before auth endpoints
create_dependency "T053" "T089" "IdP adapter factory blocks auth login tests"

# Models must exist before services
create_dependency "T022" "T054" "Token models block JWT service tests"
create_dependency "T020" "T121" "Authorization models block entitlement evaluator tests"

# Audit logging must be ready for all endpoints
create_dependency "T028" "T089" "Audit verification blocks auth endpoints"

echo ""
echo "=== Dependency Creation Complete ==="
echo ""

if [[ "$DRY_RUN" == "false" ]]; then
    echo "Dependencies created successfully!"
    echo "Verify with: bd list --status=blocked"
else
    echo "This was a dry run. Re-run without --dry-run to create actual dependencies."
fi

echo ""
echo "Next steps:"
echo "1. Verify dependencies: bd list --status=blocked"
echo "2. Check ready tasks: bd ready"
echo "3. Start implementation: bd update <id> --status=in_progress"
