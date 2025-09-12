#!/bin/bash
# GitHub Release Creation Script for Foresight SAR v0.9

set -e

echo "🚀 Creating GitHub release for Foresight SAR v0.9..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed"
    echo "Install from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub"
    echo "Run: gh auth login"
    exit 1
fi

echo "📝 Creating release..."
gh release create v0.9 \
    --title "Foresight SAR v0.9 - Field-Ready Prototype" \
    --notes-file release_notes.md \
    --draft

echo "📦 Uploading assets..."
gh release upload v0.9 "C:\Users\Admin\Desktop\foresight\models\yolov8n.pt"
gh release upload v0.9 "C:\Users\Admin\Desktop\foresight\models\yolov8n.onnx"
gh release upload v0.9 "C:\Users\Admin\Desktop\foresight\foresight-electron\dist\Foresight SAR-win32-x64"

echo "✅ Release created successfully!"
echo "📋 Next steps:"
echo "   1. Review the draft release at: https://github.com/Hiko318/foresight/releases"
echo "   2. Upload large model files manually or to external storage"
echo "   3. Update download_models.py with new URLs if needed"
echo "   4. Publish the release when ready"

echo "🔗 Release URL: https://github.com/Hiko318/foresight/releases/tag/v0.9"
