# Git History Cleanup Script for Foresight SAR System (PowerShell)
# This script helps remove sensitive files and large binaries from git history

Write-Host "🔒 Foresight Git History Cleanup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "❌ Error: Not in a git repository" -ForegroundColor Red
    exit 1
}

# Get current branch
$currentBranch = git branch --show-current
Write-Host "📋 Current branch: $currentBranch" -ForegroundColor Yellow

# Warning message
Write-Host "⚠️  WARNING: This will rewrite git history!" -ForegroundColor Yellow
Write-Host "   - All commit hashes will change" -ForegroundColor Yellow
Write-Host "   - Force push will be required" -ForegroundColor Yellow
Write-Host "   - Collaborators will need to re-clone" -ForegroundColor Yellow

$confirmation = Read-Host "Continue? (y/N)"
if ($confirmation -ne "y" -and $confirmation -ne "Y") {
    Write-Host "❌ Aborted" -ForegroundColor Red
    exit 1
}

# Files and patterns to remove from history
$secretPatterns = @(
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "config/secrets.yaml",
    "secrets/",
    "private_keys/"
)

$largeFilePatterns = @(
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.mkv",
    "*.bin",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.tar.gz",
    "*.zip",
    "*.rar",
    "*.7z",
    "models/*.pt",
    "models/*.onnx",
    "models/*.trt",
    "data/samples/",
    "data/recordings/",
    "*.weights",
    "*.model"
)

Write-Host "🔍 Scanning for sensitive files in history..." -ForegroundColor Blue

# Function to remove files from git history
function Remove-FromHistory {
    param(
        [string]$Pattern,
        [string]$Description
    )
    
    Write-Host "🗑️  Removing $Description`: $Pattern" -ForegroundColor Magenta
    
    # Use git filter-branch to remove files
    $filterCommand = "git rm --cached --ignore-unmatch '$Pattern'"
    git filter-branch --force --index-filter $filterCommand --prune-empty --tag-name-filter cat -- --all
}

# Check if git filter-branch is available
try {
    git filter-branch --help | Out-Null
} catch {
    Write-Host "❌ Error: git filter-branch not available. Please install Git for Windows with full tools." -ForegroundColor Red
    exit 1
}

# Remove secret files
Write-Host "🔒 Removing secret files from history..." -ForegroundColor Green
foreach ($pattern in $secretPatterns) {
    $logOutput = git log --all --full-history -- $pattern 2>$null
    if ($logOutput -and $logOutput.Contains("commit")) {
        Remove-FromHistory -Pattern $pattern -Description "secret files"
    }
}

# Remove large files
Write-Host "📦 Removing large files from history..." -ForegroundColor Green
foreach ($pattern in $largeFilePatterns) {
    $logOutput = git log --all --full-history -- $pattern 2>$null
    if ($logOutput -and $logOutput.Contains("commit")) {
        Remove-FromHistory -Pattern $pattern -Description "large files"
    }
}

# Clean up refs
Write-Host "🧹 Cleaning up references..." -ForegroundColor Blue
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Show repository size
Write-Host "📊 Repository size after cleanup:" -ForegroundColor Cyan
$gitSize = (Get-ChildItem .git -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "$([math]::Round($gitSize, 2)) MB" -ForegroundColor White

Write-Host "✅ Git history cleanup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Review the changes: git log --oneline" -ForegroundColor White
Write-Host "2. Test your application thoroughly" -ForegroundColor White
Write-Host "3. Force push to remote: git push --force-with-lease origin --all" -ForegroundColor White
Write-Host "4. Force push tags: git push --force-with-lease origin --tags" -ForegroundColor White
Write-Host "5. Notify collaborators to re-clone the repository" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  Remember to:" -ForegroundColor Yellow
Write-Host "- Set up Git LFS for large files: git lfs install" -ForegroundColor White
Write-Host "- Add .env.example but never commit .env files" -ForegroundColor White
Write-Host "- Use environment variables for all secrets" -ForegroundColor White