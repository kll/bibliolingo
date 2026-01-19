#!/bin/sh

# Ensure script is run from repository root
if [ ! -d "scripts" ] || [ ! -f "CLAUDE.md" ]; then
    echo "Error: This script must be run from the repository root directory."
    echo "Usage: ./scripts/export-confluence.sh"
    exit 1
fi

npx -y @aashari/nodejs-confluence-export export --space ADRs -o ./data/confluence/ADRs
npx -y @aashari/nodejs-confluence-export export --space DEV -o ./data/confluence/DEV
