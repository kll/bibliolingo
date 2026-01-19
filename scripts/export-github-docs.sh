#!/bin/sh

# Ensure script is run from repository root
if [ ! -d "scripts" ] || [ ! -f "CLAUDE.md" ]; then
    echo "Error: This script must be run from the repository root directory."
    echo "Usage: ./scripts/export-github-docs.sh"
    exit 1
fi

# Configuration
GITHUB_REPO="boostlingo/bl-platform"
GITHUB_BRANCH="release/jan-ai"
DOCS_PATH="docs"
OUTPUT_DIR="./data/github/bl-platform/docs"

# Check GitHub CLI authentication
if ! gh auth status >/dev/null 2>&1; then
    echo "Error: GitHub CLI is not authenticated."
    echo "Please run: gh auth login"
    exit 1
fi

echo "Exporting docs from ${GITHUB_REPO} (${GITHUB_BRANCH})..."

# Create temporary directory for sparse checkout
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Clone with sparse checkout
echo "Cloning repository with sparse checkout..."
git clone --filter=blob:none --sparse \
    "https://github.com/${GITHUB_REPO}.git" \
    "$TEMP_DIR" \
    --branch "$GITHUB_BRANCH" \
    --single-branch \
    --quiet 2>&1 | grep -v "Cloning into"

# Configure sparse checkout for docs only
cd "$TEMP_DIR"
git sparse-checkout set "$DOCS_PATH" 2>&1 | grep -v "^$"

# Get commit SHA for metadata
COMMIT_SHA=$(git rev-parse HEAD)

# Return to original directory
cd - >/dev/null

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy docs with structure preserved
echo "Copying documentation files..."
cp -r "$TEMP_DIR/$DOCS_PATH/"* "$OUTPUT_DIR/"

# Add metadata file with export information
cat > "$OUTPUT_DIR/.export-metadata.json" <<EOF
{
  "repository": "$GITHUB_REPO",
  "branch": "$GITHUB_BRANCH",
  "commit": "$COMMIT_SHA",
  "exported_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "docs_path": "$DOCS_PATH"
}
EOF

echo "Successfully exported docs from ${GITHUB_REPO}/${DOCS_PATH}"
echo "Output location: $OUTPUT_DIR"
echo "Commit SHA: $COMMIT_SHA"
