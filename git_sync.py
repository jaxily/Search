#!/usr/bin/env python3
"""
Git Auto-Sync Script (Python Version)
Automatically adds, commits, and pushes all changes to git
Fully automatic with no user interaction required
Enhanced with automatic descriptive commit messages
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import logging
import re

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def run_git_command(command, check=True, capture_output=True):
    """Run a git command and return the result"""
    try:
        result = subprocess.run(
            ['git'] + command,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}❌ Git command failed: {' '.join(['git'] + command)}{Colors.NC}")
        if capture_output and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"stderr: {e.stderr}")
        raise

def is_git_repository():
    """Check if current directory is a git repository"""
    try:
        run_git_command(['rev-parse', '--git-dir'], check=False)
        return True
    except subprocess.CalledProcessError:
        return False

def get_current_branch():
    """Get the current git branch"""
    result = run_git_command(['branch', '--show-current'])
    return result.stdout.strip()

def has_changes():
    """Check if there are any changes to commit"""
    # Check for modified files
    try:
        run_git_command(['diff-index', '--quiet', 'HEAD', '--'], check=False)
        has_modified = False
    except subprocess.CalledProcessError:
        has_modified = True
    
    # Check for untracked files
    result = run_git_command(['ls-files', '--others', '--exclude-standard'])
    has_untracked = bool(result.stdout.strip())
    
    return has_modified or has_untracked

def get_status():
    """Get git status"""
    result = run_git_command(['status', '--porcelain'])
    return result.stdout.strip()

def add_all_changes():
    """Add all changes to staging area"""
    run_git_command(['add', '-A'])
    print(f"{Colors.YELLOW}📁 Added all changes automatically{Colors.NC}")

def get_changed_files_count():
    """Get the number of changed files"""
    result = run_git_command(['status', '--porcelain'])
    lines = [line for line in result.stdout.strip().split('\n') if line]
    return len(lines)

def analyze_changes():
    """Analyze changes to generate descriptive commit message"""
    result = run_git_command(['status', '--porcelain'])
    lines = [line for line in result.stdout.strip().split('\n') if line]
    
    # Categorize changes
    categories = {
        'modified': [],
        'added': [],
        'deleted': [],
        'renamed': []
    }
    
    file_extensions = set()
    
    for line in lines:
        status = line[:2].strip()
        filename = line[3:]
        
        # Extract file extension
        if '.' in filename:
            ext = filename.split('.')[-1].lower()
            file_extensions.add(ext)
        
        if status == 'M':
            categories['modified'].append(filename)
        elif status == 'A':
            categories['added'].append(filename)
        elif status == 'D':
            categories['deleted'].append(filename)
        elif status == 'R':
            categories['renamed'].append(filename)
        elif status == '??':
            categories['added'].append(filename)
    
    # Generate descriptive message
    parts = []
    
    if categories['added']:
        parts.append(f"➕ Added {len(categories['added'])} file(s)")
    if categories['modified']:
        parts.append(f"✏️ Modified {len(categories['modified'])} file(s)")
    if categories['deleted']:
        parts.append(f"🗑️ Deleted {len(categories['deleted'])} file(s)")
    if categories['renamed']:
        parts.append(f"🔄 Renamed {len(categories['renamed'])} file(s)")
    
    # Add file type information
    if file_extensions:
        ext_list = sorted(list(file_extensions))
        if len(ext_list) <= 3:
            parts.append(f"📁 Types: {', '.join(ext_list)}")
        else:
            parts.append(f"📁 Types: {', '.join(ext_list[:3])}...")
    
    # Add specific file mentions for important files
    important_files = []
    for filename in categories['modified'] + categories['added']:
        if any(keyword in filename.lower() for keyword in ['main.py', 'config.py', 'requirements.txt', 'readme', 'git_sync']):
            important_files.append(filename)
    
    if important_files:
        parts.append(f"🎯 Key files: {', '.join(important_files[:3])}")
    
    return ' | '.join(parts)

def commit_changes(file_count):
    """Commit changes with automatic descriptive message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate descriptive message
    change_description = analyze_changes()
    commit_msg = f"Auto-sync: {timestamp} - {change_description}"
    
    print(f"{Colors.YELLOW}💾 Committing changes automatically...{Colors.NC}")
    print(f"{Colors.BLUE}📝 Commit message: {commit_msg}{Colors.NC}")
    
    run_git_command(['commit', '-m', commit_msg])
    return timestamp

def push_to_remote(branch):
    """Push changes to remote repository"""
    try:
        # Check if remote exists
        result = run_git_command(['remote', '-v'])
        if 'origin' in result.stdout:
            print(f"{Colors.YELLOW}🚀 Pushing to remote...{Colors.NC}")
            run_git_command(['push', 'origin', branch])
            print(f"{Colors.GREEN}✅ Successfully synced to remote!{Colors.NC}")
            return True
        else:
            print(f"{Colors.YELLOW}⚠️  No remote origin found, skipping push{Colors.NC}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}❌ Failed to push to remote: {e}{Colors.NC}")
        return False

def main():
    """Main git sync function"""
    print(f"{Colors.BLUE}🔄 Starting Git Auto-Sync (Fully Automatic)...{Colors.NC}")
    
    # Check if we're in a git repository
    if not is_git_repository():
        print(f"{Colors.RED}❌ Error: Not in a git repository{Colors.NC}")
        sys.exit(1)
    
    # Get current branch
    current_branch = get_current_branch()
    print(f"{Colors.YELLOW}📋 Current branch: {current_branch}{Colors.NC}")
    
    # Check if there are any changes to commit
    if not has_changes():
        print(f"{Colors.GREEN}✅ No changes to commit{Colors.NC}")
        return
    
    # Show status before adding
    print(f"{Colors.YELLOW}📊 Current status:{Colors.NC}")
    status = get_status()
    if status:
        print(status)
    
    # Add all changes automatically
    add_all_changes()
    
    # Get number of changed files
    changed_files = get_changed_files_count()
    print(f"{Colors.YELLOW}📝 Total changes: {changed_files} files{Colors.NC}")
    
    # Show what will be committed
    print(f"{Colors.YELLOW}📋 Files to be committed:{Colors.NC}")
    status_after_add = get_status()
    if status_after_add:
        print(status_after_add)
    
    # Commit changes
    timestamp = commit_changes(changed_files)
    
    # Push to remote
    push_to_remote(current_branch)
    
    # Final summary
    print(f"{Colors.GREEN}🎉 Git sync completed successfully!{Colors.NC}")
    print(f"{Colors.GREEN}📅 Last sync: {timestamp}{Colors.NC}")
    print(f"{Colors.GREEN}📊 Files synced: {changed_files}{Colors.NC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Git sync interrupted by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.NC}")
        sys.exit(1)
