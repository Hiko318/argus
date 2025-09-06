# Foresight SAR System - Windows Build Script
# Creates standalone Windows executable and installer

param(
    [switch]$Clean,
    [switch]$Portable,
    [switch]$Installer,
    [switch]$All,
    [string]$Version = "1.0.0"
)

# Set error handling
$ErrorActionPreference = "Stop"

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$DeployDir = Join-Path $ProjectRoot "deploy\windows"
$DistDir = Join-Path $DeployDir "dist"
$BuildDir = Join-Path $DeployDir "build"

Write-Host "=== Foresight SAR Windows Build ===" -ForegroundColor Green
Write-Host "Project Root: $ProjectRoot"
Write-Host "Deploy Directory: $DeployDir"
Write-Host "Version: $Version"
Write-Host ""

# Clean previous builds
if ($Clean -or $All) {
    Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
    if (Test-Path $DistDir) { Remove-Item $DistDir -Recurse -Force }
    if (Test-Path $BuildDir) { Remove-Item $BuildDir -Recurse -Force }
    if (Test-Path "$DeployDir\*.exe") { Remove-Item "$DeployDir\*.exe" -Force }
}

# Check Python environment
Write-Host "Checking Python environment..." -ForegroundColor Yellow
try {
    $PythonVersion = python --version 2>&1
    Write-Host "Python: $PythonVersion"
} catch {
    Write-Error "Python not found. Please install Python 3.8+ and add to PATH."
    exit 1
}

# Check if in virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "Not in a virtual environment. Consider using venv for isolation."
}

# Install/upgrade build dependencies
Write-Host "Installing build dependencies..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel
pip install pyinstaller[encryption]
pip install auto-py-to-exe  # GUI alternative

# Install project dependencies
Write-Host "Installing project dependencies..." -ForegroundColor Yellow
Set-Location $ProjectRoot
pip install -r requirements.txt

# Download required models
Write-Host "Downloading required models..." -ForegroundColor Yellow
if (Test-Path "models\download_models.py") {
    python models\download_models.py
}

# Create version info file
Write-Host "Creating version info..." -ForegroundColor Yellow
$VersionInfo = @"
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=($($Version.Replace('.', ',')),0),
    prodvers=($($Version.Replace('.', ',')),0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'Foresight SAR Systems'),
          StringStruct(u'FileDescription', u'Foresight Search and Rescue System'),
          StringStruct(u'FileVersion', u'$Version'),
          StringStruct(u'InternalName', u'foresight-sar'),
          StringStruct(u'LegalCopyright', u'Copyright (c) 2024 Foresight SAR'),
          StringStruct(u'OriginalFilename', u'Foresight-SAR.exe'),
          StringStruct(u'ProductName', u'Foresight SAR System'),
          StringStruct(u'ProductVersion', u'$Version')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"@

$VersionInfo | Out-File -FilePath "$DeployDir\version_info.txt" -Encoding UTF8

# Build with PyInstaller
Write-Host "Building executable with PyInstaller..." -ForegroundColor Yellow
Set-Location $DeployDir

$PyInstallerArgs = @(
    "--clean",
    "--noconfirm",
    "--log-level=INFO",
    "--version-file=version_info.txt",
    "foresight.spec"
)

try {
    pyinstaller @PyInstallerArgs
    Write-Host "Build completed successfully!" -ForegroundColor Green
} catch {
    Write-Error "PyInstaller build failed: $_"
    exit 1
}

# Create portable version
if ($Portable -or $All) {
    Write-Host "Creating portable version..." -ForegroundColor Yellow
    $PortableDir = Join-Path $DistDir "Foresight-SAR-Portable"
    $MainDistDir = Join-Path $DistDir "Foresight-SAR"
    
    if (Test-Path $MainDistDir) {
        Copy-Item $MainDistDir $PortableDir -Recurse -Force
        
        # Create portable launcher
        $PortableLauncher = @'
@echo off
setlocal
set "APP_DIR=%~dp0"
set "DATA_DIR=%APP_DIR%data"
set "LOGS_DIR=%APP_DIR%logs"

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

echo Starting Foresight SAR System (Portable Mode)...
start "" "%APP_DIR%\Foresight-SAR.exe" --portable --data-dir="%DATA_DIR%" --logs-dir="%LOGS_DIR%"
'@
        
        $PortableLauncher | Out-File -FilePath "$PortableDir\Start-Portable.bat" -Encoding ASCII
        
        # Create portable archive
        $ArchiveName = "Foresight-SAR-v$Version-Portable.zip"
        Compress-Archive -Path $PortableDir -DestinationPath "$DistDir\$ArchiveName" -Force
        Write-Host "Portable version created: $ArchiveName" -ForegroundColor Green
    }
}

# Create installer
if ($Installer -or $All) {
    Write-Host "Creating installer..." -ForegroundColor Yellow
    
    # Check for NSIS
    $NSISPath = Get-Command "makensis.exe" -ErrorAction SilentlyContinue
    if (-not $NSISPath) {
        Write-Warning "NSIS not found. Skipping installer creation."
        Write-Host "Download NSIS from: https://nsis.sourceforge.io/"
    } else {
        # Create NSIS script
        $NSISScript = @"
!define APP_NAME "Foresight SAR System"
!define APP_VERSION "$Version"
!define APP_PUBLISHER "Foresight SAR Systems"
!define APP_URL "https://github.com/foresight-sar"
!define APP_EXECUTABLE "Foresight-SAR.exe"

!include "MUI2.nsh"

Name "`${APP_NAME}"
OutFile "Foresight-SAR-v$Version-Setup.exe"
InstallDir "`$PROGRAMFILES\Foresight SAR"
InstallDirRegKey HKCU "Software\Foresight SAR" ""
RequestExecutionLevel admin

!define MUI_ABORTWARNING
!define MUI_ICON "..\..\assets\icon.ico"
!define MUI_UNICON "..\..\assets\icon.ico"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Main Application" SecMain
    SetOutPath "`$INSTDIR"
    File /r "dist\Foresight-SAR\*"
    
    WriteRegStr HKCU "Software\Foresight SAR" "" "`$INSTDIR"
    WriteUninstaller "`$INSTDIR\Uninstall.exe"
    
    CreateDirectory "`$SMPROGRAMS\Foresight SAR"
    CreateShortCut "`$SMPROGRAMS\Foresight SAR\Foresight SAR.lnk" "`$INSTDIR\`${APP_EXECUTABLE}"
    CreateShortCut "`$SMPROGRAMS\Foresight SAR\Uninstall.lnk" "`$INSTDIR\Uninstall.exe"
    CreateShortCut "`$DESKTOP\Foresight SAR.lnk" "`$INSTDIR\`${APP_EXECUTABLE}"
SectionEnd

Section "Uninstall"
    Delete "`$INSTDIR\Uninstall.exe"
    RMDir /r "`$INSTDIR"
    
    Delete "`$SMPROGRAMS\Foresight SAR\*"
    RMDir "`$SMPROGRAMS\Foresight SAR"
    Delete "`$DESKTOP\Foresight SAR.lnk"
    
    DeleteRegKey /ifempty HKCU "Software\Foresight SAR"
SectionEnd
"@
        
        $NSISScript | Out-File -FilePath "$DeployDir\installer.nsi" -Encoding UTF8
        
        try {
            & makensis.exe "$DeployDir\installer.nsi"
            Write-Host "Installer created successfully!" -ForegroundColor Green
        } catch {
            Write-Warning "Failed to create installer: $_"
        }
    }
}

# Create deployment package
Write-Host "Creating deployment package..." -ForegroundColor Yellow
$PackageDir = Join-Path $DistDir "Foresight-SAR-v$Version-Windows"
if (Test-Path $PackageDir) { Remove-Item $PackageDir -Recurse -Force }
New-Item $PackageDir -ItemType Directory | Out-Null

# Copy main distribution
Copy-Item "$DistDir\Foresight-SAR" "$PackageDir\Application" -Recurse -Force

# Copy documentation
Copy-Item "$ProjectRoot\README.md" $PackageDir -Force
Copy-Item "$ProjectRoot\LICENSE" $PackageDir -Force
Copy-Item "$ProjectRoot\docs" "$PackageDir\Documentation" -Recurse -Force

# Create deployment info
$DeploymentInfo = @"
Foresight SAR System - Windows Deployment
Version: $Version
Build Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Platform: Windows x64

Contents:
- Application/     : Main application files
- Documentation/   : System documentation
- README.md        : Quick start guide
- LICENSE          : Software license

Installation:
1. Extract all files to desired location
2. Run Application/Foresight-SAR.exe
3. Follow setup wizard for initial configuration

System Requirements:
- Windows 10/11 (64-bit)
- 8GB RAM minimum, 16GB recommended
- NVIDIA GPU with CUDA support (optional but recommended)
- 10GB free disk space
- Internet connection for map tiles and updates

Support:
- Documentation: See Documentation/ folder
- Issues: https://github.com/foresight-sar/issues
- Email: support@foresight-sar.com
"@

$DeploymentInfo | Out-File -FilePath "$PackageDir\DEPLOYMENT.txt" -Encoding UTF8

# Create final archive
$FinalArchive = "Foresight-SAR-v$Version-Windows.zip"
Compress-Archive -Path $PackageDir -DestinationPath "$DistDir\$FinalArchive" -Force

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Green
Write-Host "Distribution directory: $DistDir"
Write-Host "Main executable: $DistDir\Foresight-SAR\Foresight-SAR.exe"
Write-Host "Deployment package: $DistDir\$FinalArchive"

if ($Portable -or $All) {
    Write-Host "Portable version: $DistDir\Foresight-SAR-v$Version-Portable.zip"
}

if ($Installer -or $All) {
    $InstallerPath = "$DeployDir\Foresight-SAR-v$Version-Setup.exe"
    if (Test-Path $InstallerPath) {
        Write-Host "Installer: $InstallerPath"
    }
}

Write-Host ""
Write-Host "Build completed successfully!" -ForegroundColor Green