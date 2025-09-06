# YOLOv8 Training Script for SAR Operations (PowerShell)
# This script provides easy training automation with various configurations for Windows

param(
    [string]$Config = "train_config.yaml",
    [string]$Dataset = "datasets/sar_dataset",
    [string]$Model = "n",
    [int]$Epochs = 100,
    [int]$BatchSize = 16,
    [string]$Device = "0",
    [string]$Resume = "",
    [switch]$Export,
    [switch]$Validate,
    [switch]$Benchmark,
    [switch]$Help
)

# Error handling
$ErrorActionPreference = "Stop"

# Function to write colored output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to show usage
function Show-Usage {
    @"
Usage: .\train.ps1 [OPTIONS]

YOLOv8 Training Script for SAR Operations

Options:
    -Config FILE            Training configuration file (default: train_config.yaml)
    -Dataset PATH           Dataset path (default: datasets/sar_dataset)
    -Model SIZE             Model size: n, s, m, l, x (default: n)
    -Epochs NUM             Number of epochs (default: 100)
    -BatchSize NUM          Batch size (default: 16)
    -Device DEVICE          Training device (default: 0)
    -Resume PATH            Resume from checkpoint
    -Export                 Export model after training
    -Validate               Validate dataset before training
    -Benchmark              Run performance benchmark after training
    -Help                   Show this help message

Examples:
    # Basic training
    .\train.ps1
    
    # Custom configuration
    .\train.ps1 -Config custom_config.yaml -Epochs 200 -BatchSize 32
    
    # Resume training
    .\train.ps1 -Resume runs/train/sar_yolov8/weights/last.pt
    
    # Full pipeline with validation and export
    .\train.ps1 -Validate -Export -Benchmark
    
    # High-performance training
    .\train.ps1 -Model l -BatchSize 64 -Epochs 300 -Device 0

"@
}

# Function to check dependencies
function Test-Dependencies {
    Write-Info "Checking dependencies..."
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Info "Found Python: $pythonVersion"
    }
    catch {
        Write-Error "Python not found. Please install Python 3.8+"
        exit 1
    }
    
    # Check pip packages
    try {
        python -c "import ultralytics" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Ultralytics not found"
        }
    }
    catch {
        Write-Error "Ultralytics not installed. Run: pip install ultralytics"
        exit 1
    }
    
    try {
        python -c "import torch" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "PyTorch not found"
        }
    }
    catch {
        Write-Error "PyTorch not installed. Run: pip install torch torchvision"
        exit 1
    }
    
    # Check CUDA if using GPU
    if ($Device -ne "cpu") {
        try {
            python -c "import torch; assert torch.cuda.is_available()" 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "CUDA not available. Falling back to CPU training"
                $script:Device = "cpu"
            }
        }
        catch {
            Write-Warning "CUDA check failed. Falling back to CPU training"
            $script:Device = "cpu"
        }
    }
    
    Write-Success "Dependencies check passed"
}

# Function to validate dataset
function Test-Dataset {
    Write-Info "Validating dataset structure..."
    
    if (-not (Test-Path $Dataset)) {
        Write-Error "Dataset directory not found: $Dataset"
        exit 1
    }
    
    # Check required directories
    $requiredDirs = @(
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    )
    
    foreach ($dir in $requiredDirs) {
        $fullPath = Join-Path $Dataset $dir
        if (-not (Test-Path $fullPath)) {
            Write-Error "Missing directory: $fullPath"
            exit 1
        }
    }
    
    # Check dataset.yaml
    $yamlPath = Join-Path $Dataset "dataset.yaml"
    if (-not (Test-Path $yamlPath)) {
        Write-Error "Missing dataset.yaml file in $Dataset"
        exit 1
    }
    
    # Count files
    $trainImages = (Get-ChildItem -Path (Join-Path $Dataset "images/train") -File).Count
    $valImages = (Get-ChildItem -Path (Join-Path $Dataset "images/val") -File).Count
    
    if ($trainImages -eq 0) {
        Write-Error "No training images found"
        exit 1
    }
    
    if ($valImages -eq 0) {
        Write-Warning "No validation images found"
    }
    
    Write-Success "Dataset validation passed ($trainImages train, $valImages val images)"
}

# Function to setup training environment
function Initialize-Environment {
    Write-Info "Setting up training environment..."
    
    # Create output directories
    New-Item -ItemType Directory -Force -Path "runs/train" | Out-Null
    New-Item -ItemType Directory -Force -Path "logs/training" | Out-Null
    
    # Set environment variables
    $env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"
    if ($Device -ne "cpu") {
        $env:CUDA_VISIBLE_DEVICES = $Device
    }
    
    # Log system info
    Write-Info "System Information:"
    $pythonVersion = python --version 2>&1
    Write-Host "  Python: $pythonVersion"
    
    try {
        $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
        Write-Host "  PyTorch: $torchVersion"
    }
    catch {
        Write-Host "  PyTorch: Version check failed"
    }
    
    try {
        $ultraVersion = python -c "import ultralytics; print(ultralytics.__version__)" 2>&1
        Write-Host "  Ultralytics: $ultraVersion"
    }
    catch {
        Write-Host "  Ultralytics: Version check failed"
    }
    
    if ($Device -ne "cpu") {
        try {
            $cudaVersion = python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'Not available')" 2>&1
            Write-Host "  CUDA: $cudaVersion"
            
            $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')" 2>&1
            Write-Host "  GPU: $gpuName"
        }
        catch {
            Write-Host "  CUDA: Check failed"
        }
    }
    
    Write-Success "Environment setup completed"
}

# Function to run training
function Start-Training {
    Write-Info "Starting YOLOv8 training..."
    
    # Build training command
    $trainArgs = @(
        "train.py",
        "--config", $Config,
        "--data", $Dataset,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--device", $Device
    )
    
    # Add optional parameters
    if ($Resume) {
        $trainArgs += @("--resume", $Resume)
    }
    
    if ($Export) {
        $trainArgs += "--export"
    }
    
    Write-Info "Training command: python $($trainArgs -join ' ')"
    
    # Run training with logging
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = "logs/training/train_$timestamp.log"
    Write-Info "Logging to: $logFile"
    
    try {
        # Start training process
        $process = Start-Process -FilePath "python" -ArgumentList $trainArgs -NoNewWindow -PassThru -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
        
        # Monitor the process
        Write-Info "Training started (PID: $($process.Id)). Monitoring progress..."
        
        # Show progress by tailing the log file
        $lastSize = 0
        while (-not $process.HasExited) {
            Start-Sleep -Seconds 5
            
            if (Test-Path $logFile) {
                $currentSize = (Get-Item $logFile).Length
                if ($currentSize -gt $lastSize) {
                    $newContent = Get-Content $logFile -Tail 10
                    $newContent | ForEach-Object {
                        if ($_ -match "Epoch|mAP|Loss") {
                            Write-Host $_ -ForegroundColor Cyan
                        }
                    }
                    $lastSize = $currentSize
                }
            }
        }
        
        # Check exit code
        if ($process.ExitCode -eq 0) {
            Write-Success "Training completed successfully"
        }
        else {
            Write-Error "Training failed with exit code $($process.ExitCode). Check log file: $logFile"
            exit 1
        }
    }
    catch {
        Write-Error "Training failed: $($_.Exception.Message)"
        exit 1
    }
}

# Function to run benchmark
function Start-Benchmark {
    Write-Info "Running performance benchmark..."
    
    # Find best model
    $bestModel = Get-ChildItem -Path "runs/train" -Recurse -Name "best.pt" | Select-Object -First 1
    
    if (-not $bestModel) {
        Write-Warning "No trained model found for benchmarking"
        return
    }
    
    $bestModelPath = Join-Path "runs/train" $bestModel
    Write-Info "Using model: $bestModelPath"
    
    # Run evaluation
    try {
        python evaluate.py --model $bestModelPath --data $Dataset --benchmark --export-results
        Write-Success "Benchmark completed"
    }
    catch {
        Write-Warning "Benchmark failed: $($_.Exception.Message)"
    }
}

# Function to cleanup
function Clear-TempFiles {
    Write-Info "Cleaning up temporary files..."
    # Add cleanup commands here if needed
}

# Main execution
function Main {
    # Show help if requested
    if ($Help) {
        Show-Usage
        exit 0
    }
    
    Write-Info "Starting SAR YOLOv8 Training Pipeline"
    Write-Info "Configuration: $Config"
    Write-Info "Dataset: $Dataset"
    Write-Info "Model: YOLOv8$Model"
    Write-Info "Epochs: $Epochs"
    Write-Info "Batch Size: $BatchSize"
    Write-Info "Device: $Device"
    
    try {
        # Check dependencies
        Test-Dependencies
        
        # Validate dataset if requested
        if ($Validate) {
            Test-Dataset
        }
        
        # Setup environment
        Initialize-Environment
        
        # Update config file with model size
        $tempConfig = $null
        if (Test-Path $Config) {
            $timestamp = [DateTimeOffset]::Now.ToUnixTimeSeconds()
            $tempConfig = "temp_config_$timestamp.yaml"
            
            # Read and modify config
            $configContent = Get-Content $Config
            $configContent = $configContent -replace "model: yolov8[nslmx].pt", "model: yolov8$Model.pt"
            $configContent | Set-Content $tempConfig
            
            $script:Config = $tempConfig
        }
        
        # Run training
        Start-Training
        
        # Run benchmark if requested
        if ($Benchmark) {
            Start-Benchmark
        }
        
        # Cleanup temporary config
        if ($tempConfig -and (Test-Path $tempConfig)) {
            Remove-Item $tempConfig
        }
        
        Write-Success "SAR YOLOv8 training pipeline completed successfully!"
        
        # Show results summary
        Write-Info "Results Summary:"
        $latestRun = Get-ChildItem -Path "runs/train" -Directory -Name "sar_yolov8*" | Sort-Object | Select-Object -Last 1
        
        if ($latestRun) {
            $runPath = Join-Path "runs/train" $latestRun
            Write-Host "  Training results: $runPath"
            
            $bestWeights = Join-Path $runPath "weights/best.pt"
            if (Test-Path $bestWeights) {
                Write-Host "  Best model: $bestWeights"
            }
            
            $resultsPlot = Join-Path $runPath "results.png"
            if (Test-Path $resultsPlot) {
                Write-Host "  Training plots: $resultsPlot"
            }
        }
    }
    catch {
        Write-Error "Pipeline failed: $($_.Exception.Message)"
        exit 1
    }
    finally {
        Clear-TempFiles
    }
}

# Run main function
Main