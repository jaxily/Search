#!/bin/bash

# Git Auto-Sync Script
# This script automatically adds, commits, and pushes all changes to git
# Updated to be fully automatic with no user interaction required
# Enhanced with automatic descriptive commit messages

# Set the script to exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Starting Git Auto-Sync (Fully Automatic)...${NC}"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Not in a git repository${NC}"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}📋 Current branch: ${CURRENT_BRANCH}${NC}"

# Check if there are any changes to commit (including untracked files)
if git diff-index --quiet HEAD -- && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo -e "${GREEN}✅ No changes to commit${NC}"
    exit 0
fi

# Show status before adding
echo -e "${YELLOW}📊 Current status:${NC}"
git status --short

# Add all changes automatically (including untracked files)
echo -e "${YELLOW}📁 Adding all changes automatically...${NC}"
git add -A

# Get list of changed files (including untracked files)
CHANGED_FILES=$(git status --porcelain | wc -l | tr -d ' ')
echo -e "${YELLOW}📝 Total changes: ${CHANGED_FILES} files${NC}"

# Show what will be committed
echo -e "${YELLOW}📋 Files to be committed:${NC}"
git status --short

# Analyze changes for descriptive commit message
echo -e "${YELLOW}🔍 Analyzing changes for descriptive commit message...${NC}"

# Get detailed status for analysis
STATUS_OUTPUT=$(git status --porcelain)

# Initialize counters
MODIFIED_COUNT=0
ADDED_COUNT=0
DELETED_COUNT=0
RENAMED_COUNT=0
FILE_EXTENSIONS=""

# Analyze each line of status
while IFS= read -r line; do
    if [ -n "$line" ]; then
        STATUS="${line:0:2}"
        FILENAME="${line:3}"
        
        # Count by status
        case $STATUS in
            "M "|" M") ((MODIFIED_COUNT++)) ;;
            "A "|" A"|"??") ((ADDED_COUNT++)) ;;
            "D "|" D") ((DELETED_COUNT++)) ;;
            "R "|" R") ((RENAMED_COUNT++)) ;;
        esac
        
        # Extract file extension
        if [[ $FILENAME == *.* ]]; then
            EXT="${FILENAME##*.}"
            if [[ ! $FILE_EXTENSIONS =~ $EXT ]]; then
                if [ -n "$FILE_EXTENSIONS" ]; then
                    FILE_EXTENSIONS="$FILE_EXTENSIONS,$EXT"
                else
                    FILE_EXTENSIONS="$EXT"
                fi
            fi
        fi
    fi
done <<< "$STATUS_OUTPUT"

# Build descriptive message parts
MESSAGE_PARTS=""

if [ $ADDED_COUNT -gt 0 ]; then
    MESSAGE_PARTS="${MESSAGE_PARTS}➕ Added $ADDED_COUNT file(s)"
fi

if [ $MODIFIED_COUNT -gt 0 ]; then
    if [ -n "$MESSAGE_PARTS" ]; then
        MESSAGE_PARTS="${MESSAGE_PARTS} | ✏️ Modified $MODIFIED_COUNT file(s)"
    else
        MESSAGE_PARTS="✏️ Modified $MODIFIED_COUNT file(s)"
    fi
fi

if [ $DELETED_COUNT -gt 0 ]; then
    if [ -n "$MESSAGE_PARTS" ]; then
        MESSAGE_PARTS="${MESSAGE_PARTS} | 🗑️ Deleted $DELETED_COUNT file(s)"
    else
        MESSAGE_PARTS="🗑️ Deleted $DELETED_COUNT file(s)"
    fi
fi

if [ $RENAMED_COUNT -gt 0 ]; then
    if [ -n "$MESSAGE_PARTS" ]; then
        MESSAGE_PARTS="${MESSAGE_PARTS} | 🔄 Renamed $RENAMED_COUNT file(s)"
    else
        MESSAGE_PARTS="🔄 Renamed $RENAMED_COUNT file(s)"
    fi
fi

# Add file type information
if [ -n "$FILE_EXTENSIONS" ]; then
    if [ -n "$MESSAGE_PARTS" ]; then
        MESSAGE_PARTS="${MESSAGE_PARTS} | 📁 Types: $FILE_EXTENSIONS"
    else
        MESSAGE_PARTS="📁 Types: $FILE_EXTENSIONS"
    fi
fi

# Create automatic commit message with timestamp and description
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
COMMIT_MSG="Auto-sync: $TIMESTAMP - $MESSAGE_PARTS"

echo -e "${YELLOW}💾 Committing changes automatically...${NC}"
echo -e "${BLUE}📝 Commit message: ${COMMIT_MSG}${NC}"
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
echo -e "${GREEN}📊 Files synced: ${CHANGED_FILES}${NC}"
