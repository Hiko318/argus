# Git LFS Setup Script for Foresight SAR System (PowerShell)
# This script initializes Git LFS and migrates existing large files

Write-Host "📦 Foresight Git LFS Setup" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "❌ Error: Not in a git repository" -ForegroundColor Red
    exit 1
}

# Check if Git LFS is installed
try {
    $null = Get-Command git-lfs -ErrorAction Stop
    Write-Host "✅ Git LFS is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Git LFS is not installed" -ForegroundColor Red
    Write-Host "📋 Please install Git LFS:" -ForegroundColor Yellow
    Write-Host "   - Download from https://git-lfs.github.io/" -ForegroundColor Yellow
    Write-Host "   - Or use: winget install Git.Git.LFS" -ForegroundColor Yellow
    exit 1
}

# Initialize Git LFS
Write-Host "🔧 Initializing Git LFS..." -ForegroundColor Yellow
git lfs install

# Check if .gitattributes exists
if (-not (Test-Path ".gitattributes")) {
    Write-Host "❌ Error: .gitattributes file not found" -ForegroundColor Red
    Write-Host "📋 Please ensure .gitattributes is present with LFS configuration" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ .gitattributes file found" -ForegroundColor Green

# Show current LFS tracking
Write-Host "📋 Current LFS tracking patterns:" -ForegroundColor Cyan
git lfs track

# Find large files that should be in LFS
Write-Host "🔍 Scanning for large files (>10MB)..." -ForegroundColor Yellow
$largeFiles = Get-ChildItem -Recurse -File | Where-Object { 
    $_.Length -gt 10MB -and 
    $_.FullName -notlike "*\.git\*" -and 
    $_.FullName -notlike "*\node_modules\*" 
}

if ($largeFiles) {
    Write-Host "⚠️  Found large files that should be managed by LFS:" -ForegroundColor Yellow
    foreach ($file in $largeFiles) {
        $size = [math]::Round($file.Length / 1MB, 2)
        $relativePath = $file.FullName.Replace((Get-Location).Path, ".")
        Write-Host "   $relativePath ($size MB)" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "📋 Consider adding these file types to .gitattributes" -ForegroundColor Cyan
} else {
    Write-Host "✅ No large files found outside of LFS patterns" -ForegroundColor Green
}

# Check for files already tracked by LFS
Write-Host "📦 Files currently in LFS:" -ForegroundColor Cyan
git lfs ls-files

# Migrate existing files to LFS if requested
$migrate = Read-Host "🔄 Migrate existing large files to LFS? (y/N)"
if ($migrate -eq 'y' -or $migrate -eq 'Y') {
    Write-Host "🔄 Migrating files to LFS..." -ForegroundColor Yellow
    
    # Migrate common large file types
    $migratePatterns = @(
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.weights",
        "*.model",
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.zip",
        "*.tar.gz"
    )
    
    foreach ($pattern in $migratePatterns) {
        $hasFiles = git log --all --full-history -- $pattern 2>$null
        if ($hasFiles) {
            Write-Host "🔄 Migrating $pattern to LFS..." -ForegroundColor Yellow
            git lfs migrate import --include="$pattern" --everything
        }
    }
    
    Write-Host "✅ Migration completed" -ForegroundColor Green
}

# Show LFS status
Write-Host "📊 Git LFS Status:" -ForegroundColor Cyan
git lfs env

# Show repository size
Write-Host "📊 Repository size:" -ForegroundColor Cyan
$gitSize = (Get-ChildItem .git -Recurse | Measure-Object -Property Length -Sum).Sum
$gitSizeMB = [math]::Round($gitSize / 1MB, 2)
Write-Host ".git folder: $gitSizeMB MB"

Write-Host "✅ Git LFS setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Review LFS files: git lfs ls-files"
Write-Host "2. Check LFS status: git lfs status"
Write-Host "3. Commit changes: git add . && git commit -m 'Setup Git LFS'"
Write-Host "4. Push to remote: git push origin main"
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Cyan
Write-Host "- Use 'git lfs pull' to download LFS files"
Write-Host "- Use 'git lfs push origin main' to upload LFS files"
Write-Host "- Monitor LFS usage: git lfs env"
Write-Host "- Large files are now stored efficiently!"