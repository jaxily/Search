#!/bin/bash

# Git Auto-Sync Script
# This script automatically adds, commits, and pushes all changes to git

# Set the script to exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔄 Starting Git Auto-Sync...${NC}"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Not in a git repository${NC}"
    exit 1
fi

# Check if there are any changes to commit
if git diff-index --quiet HEAD --; then
    echo -e "${GREEN}✅ No changes to commit${NC}"
    exit 0
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}📋 Current branch: ${CURRENT_BRANCH}${NC}"

# Add all changes automatically
echo -e "${YELLOW}📁 Adding all changes automatically...${NC}"
git add -A

# Get list of changed files (including untracked files)
CHANGED_FILES=$(git status --porcelain | wc -l | tr -d ' ')
echo -e "${YELLOW}📝 Total changes: ${CHANGED_FILES} files${NC}"

# Create commit message with timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Ask for optional comment
echo -e "${YELLOW}💬 Add a comment (optional, press Enter to skip):${NC}"
read -r USER_COMMENT

# Build commit message
if [ -n "$USER_COMMENT" ]; then
    COMMIT_MSG="Auto-sync: $(date '+%Y-%m-%d %H:%M:%S') - $CHANGED_FILES files updated - $USER_COMMENT"
else
    COMMIT_MSG="Auto-sync: $(date '+%Y-%m-%d %H:%M:%S') - $CHANGED_FILES files updated"
fi

echo -e "${YELLOW}💾 Committing changes...${NC}"
git commit -m "$COMMIT_MSG"

# Check if remote exists and push
if git remote -v | grep -q origin; then
    echo -e "${YELLOW}🚀 Pushing to remote...${NC}"
    git push origin "$CURRENT_BRANCH"
    echo -e "${GREEN}✅ Successfully synced to remote!${NC}"
else
    echo -e "${YELLOW}⚠️  No remote origin found, skipping push${NC}"
fi

echo -e "${GREEN}🎉 Git sync completed successfully!${NC}"
echo -e "${GREEN}📅 Last sync: ${TIMESTAMP}${NC}"
