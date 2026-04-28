#!/usr/bin/env python
"""Push mathstack repository to GitHub using GitPython."""

import os
from pathlib import Path
from git import Repo
from git.exc import InvalidGitRepositoryError

# Configuration
REPO_PATH = Path.cwd()
GITHUB_URL = "https://github.com/imattas/mathstack.git"
GITHUB_BRANCH = "main"

print(f"📁 Working directory: {REPO_PATH}")
print(f"🔗 GitHub URL: {GITHUB_URL}")

try:
    # Try to open existing repo
    repo = Repo(REPO_PATH)
    print("✓ Existing git repository found")
except InvalidGitRepositoryError:
    print("⚙️  Initializing new git repository...")
    repo = Repo.init(REPO_PATH)
    print("✓ Repository initialized")

# Configure git user (if not already configured)
if not repo.config_reader().has_option('user', 'name'):
    with repo.config_writer() as git_config:
        git_config.set_value('user', 'name', 'mathstack-bot')
        git_config.set_value('user', 'email', 'bot@mathstack.local')
    print("✓ Git user configured")

# Add all files
print("📦 Adding files...")
repo.git.add(A=True)
print(f"✓ Added all files")

# Check if there are changes to commit
if repo.is_dirty(index=True):
    # Commit changes
    commit_msg = "chore: publish mathstack v3.0.0"
    repo.index.commit(commit_msg)
    print(f"✓ Committed: {commit_msg}")
else:
    print("ℹ️  No changes to commit (working tree clean)")

# Check if remote already exists
try:
    origin = repo.remote('origin')
    print(f"✓ Remote 'origin' already exists: {origin.url}")
except ValueError:
    print("⚙️  Adding remote 'origin'...")
    origin = repo.create_remote('origin', GITHUB_URL)
    print(f"✓ Remote created: {GITHUB_URL}")

# Determine current branch
current_branch = repo.active_branch.name
print(f"📍 Current branch: {current_branch}")

# If we need to push to a different branch, rename or create it
if current_branch != GITHUB_BRANCH:
    print(f"🔄 Renaming branch from '{current_branch}' to '{GITHUB_BRANCH}'...")
    repo.active_branch.rename(GITHUB_BRANCH)
    print(f"✓ Branch renamed to '{GITHUB_BRANCH}'")

# Push to GitHub
print(f"🚀 Pushing to {GITHUB_URL}#{GITHUB_BRANCH}...")
try:
    origin.push(GITHUB_BRANCH, force=True)
    print(f"✓ Successfully pushed to origin/{GITHUB_BRANCH}")
except Exception as e:
    print(f"⚠️  Push attempt: {str(e)}")
    print("💡 If authentication is required, you may need to:")
    print("   1. Set GITHUB_TOKEN environment variable")
    print("   2. Or configure SSH keys for GitHub")
    print("   3. Or use 'gh auth login' after installing GitHub CLI")
    raise

print("\n✅ All done!")
print(f"📍 Repository: https://github.com/imattas/mathstack/tree/{GITHUB_BRANCH}")
