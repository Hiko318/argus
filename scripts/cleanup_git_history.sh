#!/bin/bash

# Git History Cleanup Script for Foresight SAR System
# This script helps remove sensitive files and large binaries from git history

set -e

echo "🔒 Foresight Git History Cleanup"
echo "================================"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Backup current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📋 Current branch: $CURRENT_BRANCH"

# Warning message
echo "⚠️  WARNING: This will rewrite git history!"
echo "   - All commit hashes will change"
echo "   - Force push will be required"
echo "   - Collaborators will need to re-clone"
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted"
    exit 1
fi

# Files and patterns to remove from history
SECRET_PATTERNS=(
    ".env"
    ".env.local"
    ".env.production"
    ".env.staging"
    "*.key"
    "*.pem"
    "*.p12"
    "*.pfx"
    "config/secrets.yaml"
    "secrets/"
    "private_keys/"
)

LARGE_FILE_PATTERNS=(
    "*.mp4"
    "*.avi"
    "*.mov"
    "*.mkv"
    "*.bin"
    "*.so"
    "*.dll"
    "*.dylib"
    "*.tar.gz"
    "*.zip"
    "*.rar"
    "*.7z"
    "models/*.pt"
    "models/*.onnx"
    "models/*.trt"
    "data/samples/"
    "data/recordings/"
    "*.weights"
    "*.model"
)

echo "🔍 Scanning for sensitive files in history..."

# Function to remove files from git history
remove_from_history() {
    local pattern="$1"
    local description="$2"
    
    echo "🗑️  Removing $description: $pattern"
    
    # Use git filter-branch to remove files
    git filter-branch --force --index-filter \
        "git rm --cached --ignore-unmatch '$pattern'" \
        --prune-empty --tag-name-filter cat -- --all
}

# Remove secret files
echo "🔒 Removing secret files from history..."
for pattern in "${SECRET_PATTERNS[@]}"; do
    if git log --all --full-history -- "$pattern" | grep -q commit; then
        remove_from_history "$pattern" "secret files"
    fi
done

# Remove large files
echo "📦 Removing large files from history..."
for pattern in "${LARGE_FILE_PATTERNS[@]}"; do
    if git log --all --full-history -- "$pattern" | grep -q commit; then
        remove_from_history "$pattern" "large files"
    fi
done

# Clean up refs
echo "🧹 Cleaning up references..."
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Show repository size
echo "📊 Repository size after cleanup:"
du -sh .git

echo "✅ Git history cleanup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Review the changes: git log --oneline"
echo "2. Test your application thoroughly"
echo "3. Force push to remote: git push --force-with-lease origin --all"
echo "4. Force push tags: git push --force-with-lease origin --tags"
echo "5. Notify collaborators to re-clone the repository"
echo ""
echo "⚠️  Remember to:"
echo "- Set up Git LFS for large files: git lfs install"
echo "- Add .env.example but never commit .env files"
echo "- Use environment variables for all secrets"