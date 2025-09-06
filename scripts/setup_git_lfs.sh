#!/bin/bash

# Git LFS Setup Script for Foresight SAR System
# This script initializes Git LFS and migrates existing large files

set -e

echo "📦 Foresight Git LFS Setup"
echo "========================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Error: Git LFS is not installed"
    echo "📋 Please install Git LFS:"
    echo "   - macOS: brew install git-lfs"
    echo "   - Ubuntu/Debian: sudo apt install git-lfs"
    echo "   - Windows: Download from https://git-lfs.github.io/"
    exit 1
fi

echo "✅ Git LFS is installed"

# Initialize Git LFS
echo "🔧 Initializing Git LFS..."
git lfs install

# Check if .gitattributes exists
if [ ! -f ".gitattributes" ]; then
    echo "❌ Error: .gitattributes file not found"
    echo "📋 Please ensure .gitattributes is present with LFS configuration"
    exit 1
fi

echo "✅ .gitattributes file found"

# Show current LFS tracking
echo "📋 Current LFS tracking patterns:"
git lfs track

# Find large files that should be in LFS
echo "🔍 Scanning for large files (>10MB)..."
LARGE_FILES=$(find . -type f -size +10M -not -path './.git/*' -not -path './node_modules/*' 2>/dev/null || true)

if [ -n "$LARGE_FILES" ]; then
    echo "⚠️  Found large files that should be managed by LFS:"
    echo "$LARGE_FILES" | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "   $file ($size)"
    done
    echo ""
    echo "📋 Consider adding these file types to .gitattributes"
else
    echo "✅ No large files found outside of LFS patterns"
fi

# Check for files already tracked by LFS
echo "📦 Files currently in LFS:"
git lfs ls-files

# Migrate existing files to LFS if requested
read -p "🔄 Migrate existing large files to LFS? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Migrating files to LFS..."
    
    # Migrate common large file types
    MIGRATE_PATTERNS=(
        "*.pt"
        "*.pth"
        "*.onnx"
        "*.weights"
        "*.model"
        "*.mp4"
        "*.avi"
        "*.mov"
        "*.zip"
        "*.tar.gz"
    )
    
    for pattern in "${MIGRATE_PATTERNS[@]}"; do
        if git log --all --full-history -- "$pattern" | grep -q commit; then
            echo "🔄 Migrating $pattern to LFS..."
            git lfs migrate import --include="$pattern" --everything
        fi
    done
    
    echo "✅ Migration completed"
fi

# Show LFS status
echo "📊 Git LFS Status:"
git lfs env

# Show repository size
echo "📊 Repository size:"
du -sh .git

echo "✅ Git LFS setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Review LFS files: git lfs ls-files"
echo "2. Check LFS status: git lfs status"
echo "3. Commit changes: git add . && git commit -m 'Setup Git LFS'"
echo "4. Push to remote: git push origin main"
echo ""
echo "💡 Tips:"
echo "- Use 'git lfs pull' to download LFS files"
echo "- Use 'git lfs push origin main' to upload LFS files"
echo "- Monitor LFS usage: git lfs env"
echo "- Large files are now stored efficiently!"