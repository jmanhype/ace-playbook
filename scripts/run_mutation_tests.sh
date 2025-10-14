#!/bin/bash
# Mutation Testing Script for ACE Playbook
# Uses mutmut to verify test suite quality by introducing mutations

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ACE Playbook Mutation Testing${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if mutmut is installed
if ! command -v mutmut &> /dev/null; then
    echo -e "${RED}Error: mutmut is not installed${NC}"
    echo "Install with: pip install -e \".[dev]\""
    exit 1
fi

# Clean previous mutation results
if [ -f ".mutmut-cache" ]; then
    echo -e "${YELLOW}Cleaning previous mutation cache...${NC}"
    rm -f .mutmut-cache
    rm -rf .mutmut-cache/
fi

# Run mutation tests
echo -e "${YELLOW}Running mutation tests on ace/ directory...${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"
echo ""

# Note: mutmut 3.x scans the entire project by default
# It will automatically discover Python files to mutate
mutmut run || true  # Don't exit on failure, we want to see results

# Show results
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Mutation Test Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

mutmut results || echo -e "${RED}No mutation results available${NC}"

# Generate HTML report if mutations were created
if [ -f ".mutmut-cache" ]; then
    echo ""
    echo -e "${YELLOW}Generating HTML report...${NC}"
    mutmut html || echo -e "${YELLOW}HTML report generation skipped${NC}"

    echo ""
    echo -e "${GREEN}Mutation testing complete!${NC}"

    if [ -d "html" ]; then
        echo "View detailed report at: html/index.html"
    fi

    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  mutmut show <id>     - Show specific mutation"
    echo "  mutmut results       - Show summary of results"
    echo "  mutmut apply <id>    - Apply a mutation to see its effect"
    echo ""
    echo -e "${BLUE}Mutation Score Guide:${NC}"
    echo "  100% killed          - Excellent test suite"
    echo "  90-99% killed        - Good test suite"
    echo "  80-89% killed        - Acceptable test suite"
    echo "  <80% killed          - Needs improvement"
else
    echo -e "${YELLOW}No mutations were generated. Check mutmut configuration.${NC}"
fi
