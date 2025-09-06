#!/usr/bin/env pwsh
# Repository Hygiene Script for Foresight SAR
# Consolidates requirements files and cleans up repository structure

param(
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "Repository Hygiene for Foresight SAR" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "DRY RUN MODE - No changes will be made" -ForegroundColor Yellow
}

# Function to consolidate requirements files
function Consolidate-Requirements {
    Write-Host "`nConsolidating requirements files..." -ForegroundColor Yellow
    
    $requirementFiles = @(
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "dev-requirements.txt",
        "test-requirements.txt"
    )
    
    $existingFiles = @()
    foreach ($file in $requirementFiles) {
        if (Test-Path $file) {
            $existingFiles += $file
        }
    }
    
    if ($existingFiles.Count -gt 1) {
        Write-Host "  Found multiple requirements files: $($existingFiles -join ', ')" -ForegroundColor Yellow
        
        if (-not $DryRun) {
            # Read all requirements and consolidate
            $allRequirements = @()
            $devRequirements = @()
            
            foreach ($file in $existingFiles) {
                $content = Get-Content $file -ErrorAction SilentlyContinue
                if ($file -match "dev|test") {
                    $devRequirements += $content
                } else {
                    $allRequirements += $content
                }
            }
            
            # Write consolidated requirements.txt
            if ($allRequirements.Count -gt 0) {
                $allRequirements | Sort-Object -Unique | Out-File "requirements.txt" -Encoding UTF8
                Write-Host "  OK: Created consolidated requirements.txt" -ForegroundColor Green
            }
            
            # Write consolidated requirements-dev.txt
            if ($devRequirements.Count -gt 0) {
                $devRequirements | Sort-Object -Unique | Out-File "requirements-dev.txt" -Encoding UTF8
                Write-Host "  OK: Created consolidated requirements-dev.txt" -ForegroundColor Green
            }
            
            # Remove old files (except the consolidated ones)
            foreach ($file in $existingFiles) {
                if ($file -notin @("requirements.txt", "requirements-dev.txt")) {
                    Remove-Item $file -Force
                    Write-Host "  REMOVED: $file" -ForegroundColor Gray
                }
            }
        } else {
            Write-Host "  INFO: Would consolidate $($existingFiles.Count) requirements files" -ForegroundColor Gray
        }
    } else {
        Write-Host "  OK: Requirements files are already consolidated" -ForegroundColor Green
    }
}

# Function to ensure proper documentation structure
function Ensure-Documentation {
    Write-Host "`nEnsuring proper documentation structure..." -ForegroundColor Yellow
    
    $requiredDocs = @(
         "README.md",
         "CONTRIBUTING.md",
         "LICENSE"
     )
    
    foreach ($doc in $requiredDocs) {
         if (-not (Test-Path $doc)) {
             Write-Host "  WARNING: Missing $doc" -ForegroundColor Yellow
             if (-not $DryRun) {
                 # Create basic template for each document
                 switch ($doc) {
                     "README.md" {
                         "# Foresight SAR`n`nSuspect Acquisition and Recognition system for aerial surveillance.`n`n## Quick Start`n`nSee CONTRIBUTING.md for development setup." | Out-File $doc -Encoding UTF8
                     }
                     "CONTRIBUTING.md" {
                         "# Contributing to Foresight SAR`n`n## Development Setup`n`n1. Clone the repository`n2. Install dependencies: pip install -r requirements.txt`n3. Run tests: python -m pytest" | Out-File $doc -Encoding UTF8
                     }
                     "LICENSE" {
                         "MIT License`n`nCopyright (c) 2024 Foresight SAR`n`nSee https://opensource.org/licenses/MIT for full license text." | Out-File $doc -Encoding UTF8
                     }
                 }
                 Write-Host "  CREATED: $doc" -ForegroundColor Green
             }
         } else {
             Write-Host "  OK: $doc exists" -ForegroundColor Green
         }
     }
}

# Function to clean up duplicate files
function Clean-DuplicateFiles {
    Write-Host "`nCleaning up duplicate files..." -ForegroundColor Yellow
    
    # Look for common duplicate patterns
    $duplicatePatterns = @(
        "*.copy",
        "*.backup",
        "*_old",
        "*_backup",
        "*.bak"
    )
    
    $foundDuplicates = @()
    foreach ($pattern in $duplicatePatterns) {
        $files = Get-ChildItem -Path . -Name $pattern -Recurse -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            if (Test-Path $file) {
                $foundDuplicates += $file
                Write-Host "  WARNING: Found potential duplicate: $file" -ForegroundColor Yellow
                if (-not $DryRun) {
                    Remove-Item $file -Force
                    Write-Host "    REMOVED: $file" -ForegroundColor Gray
                }
            }
        }
    }
    
    if ($foundDuplicates.Count -eq 0) {
        Write-Host "  OK: No duplicate files found" -ForegroundColor Green
    }
}

# Function to create sample data structure
function Create-SampleDataStructure {
    Write-Host "`nCreating sample data structure..." -ForegroundColor Yellow
    
    $sampleDirs = @(
        "data/samples",
        "data/test_images",
        "data/test_videos"
    )
    
    foreach ($dir in $sampleDirs) {
        if (-not (Test-Path $dir)) {
            Write-Host "  WARNING: Missing directory: $dir" -ForegroundColor Yellow
            if (-not $DryRun) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
                Write-Host "  CREATED: $dir" -ForegroundColor Green
                
                # Create a README in each sample directory
                $readmeContent = "# Sample Data\n\nThis directory contains sample data for testing and development.\n\nDo not commit large files to this directory.\nUse small, representative samples only.\n"
                $readmeContent | Out-File "$dir/README.md" -Encoding UTF8
            }
        } else {
            Write-Host "  OK: Directory exists: $dir" -ForegroundColor Green
        }
    }
}

# Main execution
try {
    if (-not $Force -and -not $DryRun) {
        $confirm = Read-Host "This will modify repository structure. Continue? (y/N)"
        if ($confirm -ne "y" -and $confirm -ne "Y") {
            Write-Host "Operation cancelled" -ForegroundColor Yellow
            exit 0
        }
    }
    
    # Execute hygiene tasks
    Consolidate-Requirements
    Ensure-Documentation
    Clean-DuplicateFiles
    Create-SampleDataStructure
    
    Write-Host "`nRepository hygiene completed successfully!" -ForegroundColor Green
    
    if ($DryRun) {
        Write-Host "`nRun without -DryRun to apply changes" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}