# Foresight SAR System - Windows Setup Script
# Configures Windows environment for SAR operations

param(
    [switch]$Development,
    [switch]$Production,
    [switch]$Minimal,
    [string]$InstallPath = "C:\Program Files\Foresight SAR",
    [string]$DataPath = "C:\ProgramData\Foresight SAR"
)

# Require Administrator privileges
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Error "This script requires Administrator privileges. Please run as Administrator."
    exit 1
}

$ErrorActionPreference = "Stop"

Write-Host "=== Foresight SAR Windows Setup ===" -ForegroundColor Green
Write-Host "Install Path: $InstallPath"
Write-Host "Data Path: $DataPath"
Write-Host ""

# Create directories
Write-Host "Creating application directories..." -ForegroundColor Yellow
$Directories = @(
    $InstallPath,
    $DataPath,
    "$DataPath\data",
    "$DataPath\models",
    "$DataPath\evidence",
    "$DataPath\logs",
    "$DataPath\cache",
    "$DataPath\config",
    "$DataPath\backups"
)

foreach ($Dir in $Directories) {
    if (-not (Test-Path $Dir)) {
        New-Item -Path $Dir -ItemType Directory -Force | Out-Null
        Write-Host "Created: $Dir"
    }
}

# Set directory permissions
Write-Host "Setting directory permissions..." -ForegroundColor Yellow
try {
    $Acl = Get-Acl $DataPath
    $AccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Users", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
    $Acl.SetAccessRule($AccessRule)
    Set-Acl -Path $DataPath -AclObject $Acl
} catch {
    Write-Warning "Failed to set permissions: $_"
}

# Install Chocolatey if not present
if (-not (Get-Command "choco" -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    refreshenv
}

# Install system dependencies
Write-Host "Installing system dependencies..." -ForegroundColor Yellow
$SystemPackages = @(
    "python",
    "git",
    "nodejs",
    "7zip",
    "curl",
    "wget",
    "ffmpeg",
    "vcredist140",
    "dotnet-runtime"
)

if ($Development) {
    $SystemPackages += @(
        "vscode",
        "github-desktop",
        "postman",
        "wireshark"
    )
}

foreach ($Package in $SystemPackages) {
    try {
        choco install $Package -y --no-progress
    } catch {
        Write-Warning "Failed to install $Package : $_"
    }
}

# Install Python packages
Write-Host "Setting up Python environment..." -ForegroundColor Yellow

# Create virtual environment
$VenvPath = "$DataPath\venv"
if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
}

# Activate virtual environment
& "$VenvPath\Scripts\Activate.ps1"

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install requirements
$RequirementsFile = Join-Path $PSScriptRoot "requirements-windows.txt"
if (Test-Path $RequirementsFile) {
    pip install -r $RequirementsFile
} else {
    Write-Warning "Requirements file not found: $RequirementsFile"
}

# Install CUDA if NVIDIA GPU detected
Write-Host "Checking for NVIDIA GPU..." -ForegroundColor Yellow
try {
    $NvidiaGPU = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($NvidiaGPU) {
        Write-Host "NVIDIA GPU detected: $($NvidiaGPU.Name)"
        Write-Host "Installing CUDA toolkit..." -ForegroundColor Yellow
        choco install cuda -y --no-progress
    } else {
        Write-Host "No NVIDIA GPU detected. Skipping CUDA installation."
    }
} catch {
    Write-Warning "Failed to detect GPU: $_"
}

# Configure Windows Firewall
Write-Host "Configuring Windows Firewall..." -ForegroundColor Yellow
$FirewallRules = @(
    @{ Name = "Foresight SAR - HTTP"; Port = 8000; Protocol = "TCP" },
    @{ Name = "Foresight SAR - WebSocket"; Port = 5000; Protocol = "TCP" },
    @{ Name = "Foresight SAR - Web UI"; Port = 8080; Protocol = "TCP" }
)

foreach ($Rule in $FirewallRules) {
    try {
        New-NetFirewallRule -DisplayName $Rule.Name -Direction Inbound -Protocol $Rule.Protocol -LocalPort $Rule.Port -Action Allow -ErrorAction SilentlyContinue
        Write-Host "Firewall rule created: $($Rule.Name)"
    } catch {
        Write-Warning "Failed to create firewall rule: $($Rule.Name)"
    }
}

# Install Windows services
Write-Host "Installing Windows services..." -ForegroundColor Yellow

# Create service wrapper script
$ServiceScript = @'
# Foresight SAR Service Wrapper
$VenvPath = "C:\ProgramData\Foresight SAR\venv"
$AppPath = "C:\Program Files\Foresight SAR"

# Activate virtual environment
& "$VenvPath\Scripts\Activate.ps1"

# Set working directory
Set-Location $AppPath

# Start application
python main.py --service
'@

$ServiceScript | Out-File -FilePath "$InstallPath\service.ps1" -Encoding UTF8

# Create NSSM service (if NSSM is available)
if (Get-Command "nssm" -ErrorAction SilentlyContinue) {
    try {
        nssm install "ForesightSAR" powershell.exe
        nssm set "ForesightSAR" Arguments "-ExecutionPolicy Bypass -File `"$InstallPath\service.ps1`""
        nssm set "ForesightSAR" DisplayName "Foresight SAR System"
        nssm set "ForesightSAR" Description "Foresight Search and Rescue System Service"
        nssm set "ForesightSAR" Start SERVICE_AUTO_START
        Write-Host "Windows service installed: ForesightSAR"
    } catch {
        Write-Warning "Failed to install Windows service: $_"
    }
} else {
    Write-Host "Installing NSSM for service management..." -ForegroundColor Yellow
    choco install nssm -y --no-progress
}

# Create desktop shortcuts
Write-Host "Creating desktop shortcuts..." -ForegroundColor Yellow
$WshShell = New-Object -comObject WScript.Shell

# Main application shortcut
$Shortcut = $WshShell.CreateShortcut("$env:PUBLIC\Desktop\Foresight SAR.lnk")
$Shortcut.TargetPath = "$InstallPath\Foresight-SAR.exe"
$Shortcut.WorkingDirectory = $InstallPath
$Shortcut.IconLocation = "$InstallPath\assets\icon.ico"
$Shortcut.Description = "Foresight Search and Rescue System"
$Shortcut.Save()

# Configuration shortcut
$ConfigShortcut = $WshShell.CreateShortcut("$env:PUBLIC\Desktop\Foresight SAR Config.lnk")
$ConfigShortcut.TargetPath = "notepad.exe"
$ConfigShortcut.Arguments = "$DataPath\config\settings.yaml"
$ConfigShortcut.Description = "Foresight SAR Configuration"
$ConfigShortcut.Save()

# Create Start Menu entries
Write-Host "Creating Start Menu entries..." -ForegroundColor Yellow
$StartMenuPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\Foresight SAR"
if (-not (Test-Path $StartMenuPath)) {
    New-Item -Path $StartMenuPath -ItemType Directory -Force | Out-Null
}

# Copy shortcuts to Start Menu
Copy-Item "$env:PUBLIC\Desktop\Foresight SAR.lnk" "$StartMenuPath\"
Copy-Item "$env:PUBLIC\Desktop\Foresight SAR Config.lnk" "$StartMenuPath\"

# Create uninstaller
Write-Host "Creating uninstaller..." -ForegroundColor Yellow
$UninstallScript = @"
# Foresight SAR Uninstaller
param([switch]`$Force)

if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Error "This script requires Administrator privileges. Please run as Administrator."
    exit 1
}

Write-Host "Uninstalling Foresight SAR System..." -ForegroundColor Yellow

# Stop service
try {
    Stop-Service "ForesightSAR" -Force -ErrorAction SilentlyContinue
    nssm remove "ForesightSAR" confirm
} catch {}

# Remove firewall rules
`$FirewallRules = @("Foresight SAR - HTTP", "Foresight SAR - WebSocket", "Foresight SAR - Web UI")
foreach (`$Rule in `$FirewallRules) {
    Remove-NetFirewallRule -DisplayName `$Rule -ErrorAction SilentlyContinue
}

# Remove shortcuts
Remove-Item "$env:PUBLIC\Desktop\Foresight SAR*.lnk" -Force -ErrorAction SilentlyContinue
Remove-Item "$StartMenuPath" -Recurse -Force -ErrorAction SilentlyContinue

# Remove application files
if (`$Force -or (Read-Host "Remove application files? (y/N)") -eq "y") {
    Remove-Item "$InstallPath" -Recurse -Force -ErrorAction SilentlyContinue
}

# Remove data files
if (`$Force -or (Read-Host "Remove data files? (y/N)") -eq "y") {
    Remove-Item "$DataPath" -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Uninstallation complete." -ForegroundColor Green
"@

$UninstallScript | Out-File -FilePath "$InstallPath\Uninstall.ps1" -Encoding UTF8

# Create configuration file
Write-Host "Creating default configuration..." -ForegroundColor Yellow
$DefaultConfig = @"
# Foresight SAR System Configuration
app:
  name: "Foresight SAR System"
  version: "1.0.0"
  debug: false
  
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  
ui:
  host: "0.0.0.0"
  port: 8080
  
websocket:
  host: "0.0.0.0"
  port: 5000
  
database:
  type: "sqlite"
  path: "$($DataPath.Replace('\', '//'))/data/foresight.db"
  
logging:
  level: "INFO"
  file: "$($DataPath.Replace('\', '//'))/logs/foresight.log"
  max_size: "100MB"
  backup_count: 5
  
models:
  path: "$($DataPath.Replace('\', '//'))/models"
  detection_model: "yolo11n.pt"
  
evidence:
  path: "$($DataPath.Replace('\', '//'))/evidence"
  retention_days: 30
  
cache:
  path: "$($DataPath.Replace('\', '//'))/cache"
  max_size: "1GB"
  
security:
  enable_encryption: true
  key_file: "$($DataPath.Replace('\', '//'))/config/encryption.key"
"@

$DefaultConfig | Out-File -FilePath "$DataPath\config\settings.yaml" -Encoding UTF8

# Create batch files for easy management
Write-Host "Creating management scripts..." -ForegroundColor Yellow

# Start script
$StartScript = @"
@echo off
echo Starting Foresight SAR System...
net start ForesightSAR
echo Service started. Opening web interface...
timeout /t 3 /nobreak >nul
start http://localhost:8080
"@
$StartScript | Out-File -FilePath "$InstallPath\Start.bat" -Encoding ASCII

# Stop script
$StopScript = @"
@echo off
echo Stopping Foresight SAR System...
net stop ForesightSAR
echo Service stopped.
pause
"@
$StopScript | Out-File -FilePath "$InstallPath\Stop.bat" -Encoding ASCII

# Status script
$StatusScript = @"
@echo off
echo Foresight SAR System Status:
sc query ForesightSAR
echo.
echo Recent logs:
powershell "Get-Content '$DataPath\logs\foresight.log' -Tail 10"
pause
"@
$StatusScript | Out-File -FilePath "$InstallPath\Status.bat" -Encoding ASCII

# Register with Windows Programs and Features
Write-Host "Registering with Windows..." -ForegroundColor Yellow
try {
    $RegPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ForesightSAR"
    New-Item -Path $RegPath -Force | Out-Null
    Set-ItemProperty -Path $RegPath -Name "DisplayName" -Value "Foresight SAR System"
    Set-ItemProperty -Path $RegPath -Name "DisplayVersion" -Value "1.0.0"
    Set-ItemProperty -Path $RegPath -Name "Publisher" -Value "Foresight SAR Systems"
    Set-ItemProperty -Path $RegPath -Name "InstallLocation" -Value $InstallPath
    Set-ItemProperty -Path $RegPath -Name "UninstallString" -Value "powershell.exe -ExecutionPolicy Bypass -File `"$InstallPath\Uninstall.ps1`""
    Set-ItemProperty -Path $RegPath -Name "NoModify" -Value 1
    Set-ItemProperty -Path $RegPath -Name "NoRepair" -Value 1
} catch {
    Write-Warning "Failed to register with Windows: $_"
}

# Final system optimization
Write-Host "Applying system optimizations..." -ForegroundColor Yellow

# Disable Windows Defender real-time scanning for application directories (optional)
if ((Read-Host "Disable Windows Defender scanning for Foresight directories? (y/N)") -eq "y") {
    try {
        Add-MpPreference -ExclusionPath $InstallPath
        Add-MpPreference -ExclusionPath $DataPath
        Write-Host "Windows Defender exclusions added."
    } catch {
        Write-Warning "Failed to add Windows Defender exclusions: $_"
    }
}

# Set high performance power plan
try {
    powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
    Write-Host "High performance power plan activated."
} catch {
    Write-Warning "Failed to set power plan: $_"
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host "Installation Path: $InstallPath"
Write-Host "Data Path: $DataPath"
Write-Host "Configuration: $DataPath\config\settings.yaml"
Write-Host ""
Write-Host "Management Commands:"
Write-Host "- Start Service: $InstallPath\Start.bat"
Write-Host "- Stop Service: $InstallPath\Stop.bat"
Write-Host "- Check Status: $InstallPath\Status.bat"
Write-Host "- Uninstall: $InstallPath\Uninstall.ps1"
Write-Host ""
Write-Host "Next Steps:"
Write-Host "1. Copy application files to: $InstallPath"
Write-Host "2. Configure settings in: $DataPath\config\settings.yaml"
Write-Host "3. Start the service: $InstallPath\Start.bat"
Write-Host "4. Access web interface: http://localhost:8080"
Write-Host ""
Write-Host "Setup completed successfully!" -ForegroundColor Green