#!/usr/bin/env pwsh
# Security Cleanup Script for Foresight SAR
# Addresses immediate safety concerns

param(
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "Security Cleanup for Foresight SAR" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "DRY RUN MODE - No changes will be made" -ForegroundColor Yellow
}

# Function to check if we're in a git repository
function Test-GitRepo {
    try {
        git rev-parse --git-dir 2>$null
        return $true
    } catch {
        return $false
    }
}

# Function to remove sensitive files from working directory
function Remove-SensitiveFiles {
    Write-Host "`nRemoving sensitive files from working directory..." -ForegroundColor Yellow
    
    $sensitiveFiles = @(
        ".env", ".env.local", ".env.production", ".env.staging",
        "secrets.json", "credentials.json"
    )
    
    $foundFiles = @()
    foreach ($file in $sensitiveFiles) {
        if (Test-Path $file) {
            $foundFiles += $file
            Write-Host "  WARNING: Found sensitive file: $file" -ForegroundColor Red
            if (-not $DryRun) {
                Remove-Item $file -Force
                Write-Host "    REMOVED: $file" -ForegroundColor Green
            }
        }
    }
    
    if ($foundFiles.Count -eq 0) {
        Write-Host "  OK: No sensitive files found in working directory" -ForegroundColor Green
    }
}

# Function to check for large files
function Check-LargeFiles {
    Write-Host "`nChecking for large files..." -ForegroundColor Yellow
    
    $largeFiles = @()
    $modelExtensions = @("*.pt", "*.pth", "*.onnx", "*.bin", "*.weights")
    
    foreach ($ext in $modelExtensions) {
        $files = Get-ChildItem -Path . -Name $ext -Recurse -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            if (Test-Path $file) {
                $fileInfo = Get-Item $file
                $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
                if ($fileInfo.Length -gt 1MB) {
                    $largeFiles += $file
                    Write-Host "  WARNING: Large file found: $file ($sizeMB MB)" -ForegroundColor Yellow
                }
            }
        }
    }
    
    if ($largeFiles.Count -eq 0) {
        Write-Host "  OK: No large files found" -ForegroundColor Green
    } else {
        Write-Host "  INFO: Found $($largeFiles.Count) large files - ensure they are in .gitignore" -ForegroundColor Gray
    }
}

# Function to update .gitignore with security entries
function Update-GitIgnore {
    Write-Host "`nUpdating .gitignore..." -ForegroundColor Yellow
    
    $securityEntries = @(
        "# Security files",
        ".env",
        ".env.local",
        ".env.production",
        ".env.staging",
        "*.key",
        "*.pem",
        "secrets.json",
        "credentials.json",
        "",
        "# Large model files",
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.bin",
        "*.weights",
        "models/*.pt",
        "models/*.onnx",
        "models/*.pth"
    )
    
    if (Test-Path ".gitignore") {
        $existingContent = Get-Content ".gitignore"
        $newEntries = @()
        
        foreach ($entry in $securityEntries) {
            if ($entry -and $existingContent -notcontains $entry) {
                $newEntries += $entry
            }
        }
        
        if ($newEntries.Count -gt 0 -and -not $DryRun) {
            "" | Out-File ".gitignore" -Append
            "# Added by security cleanup" | Out-File ".gitignore" -Append
            $newEntries | Out-File ".gitignore" -Append
            Write-Host "  OK: Added $($newEntries.Count) new entries to .gitignore" -ForegroundColor Green
        } elseif ($newEntries.Count -eq 0) {
            Write-Host "  OK: .gitignore is already up to date" -ForegroundColor Green
        } else {
            Write-Host "  INFO: Would add $($newEntries.Count) entries to .gitignore" -ForegroundColor Gray
        }
    } else {
        Write-Host "  WARNING: .gitignore not found" -ForegroundColor Yellow
    }
}

# Function to generate security recommendations
function Generate-SecurityRecommendations {
    Write-Host "`nSecurity Recommendations:" -ForegroundColor Cyan
    
    Write-Host "  1. Rotate all API keys and secrets in production" -ForegroundColor White
    Write-Host "  2. Use environment variables for all sensitive configuration" -ForegroundColor White
    Write-Host "  3. Enable branch protection rules on main branch" -ForegroundColor White
    Write-Host "  4. Set up secret scanning in CI/CD pipeline" -ForegroundColor White
    Write-Host "  5. Use Git LFS for large model files" -ForegroundColor White
    Write-Host "  6. Implement proper access controls for production systems" -ForegroundColor White
}

# Main execution
try {
    if (-not (Test-GitRepo)) {
        Write-Host "ERROR: Not in a git repository. Please run this script from the project root." -ForegroundColor Red
        exit 1
    }
    
    if (-not $Force -and -not $DryRun) {
        $confirm = Read-Host "This will scan and potentially modify files. Continue? (y/N)"
        if ($confirm -ne "y" -and $confirm -ne "Y") {
            Write-Host "Operation cancelled" -ForegroundColor Yellow
            exit 0
        }
    }
    
    # Execute security checks
    Remove-SensitiveFiles
    Check-LargeFiles
    Update-GitIgnore
    Generate-SecurityRecommendations
    
    Write-Host "`nSecurity cleanup completed successfully!" -ForegroundColor Green
    
    if ($DryRun) {
        Write-Host "`nRun without -DryRun to apply changes" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}