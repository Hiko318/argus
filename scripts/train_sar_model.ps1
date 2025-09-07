#!/usr/bin/env pwsh
<#
.SYNOPSIS
    SAR YOLOv8 Model Training Script
    
.DESCRIPTION
    Comprehensive training script for YOLOv8 models optimized for Search and Rescue operations.
    Includes dataset preparation, training execution, metrics tracking, and model export.
    
.PARAMETER Model
    YOLOv8 model variant to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
    
.PARAMETER Epochs
    Number of training epochs
    
.PARAMETER BatchSize
    Training batch size
    
.PARAMETER ImageSize
    Input image size for training
    
.PARAMETER UseWandB
    Enable Weights & Biases logging
    
.PARAMETER PrepareDataset
    Run dataset preparation before training
    
.PARAMETER ExportModel
    Export model to ONNX/TensorRT after training
    
.EXAMPLE
    .\train_sar_model.ps1 -Model "yolov8n.pt" -Epochs 100 -BatchSize 16
    
.EXAMPLE
    .\train_sar_model.ps1 -Model "yolov8s.pt" -Epochs 150 -UseWandB -PrepareDataset -ExportModel
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt")]
    [string]$Model = "yolov8n.pt",
    
    [Parameter(Mandatory=$false)]
    [ValidateRange(1, 1000)]
    [int]$Epochs = 100,
    
    [Parameter(Mandatory=$false)]
    [ValidateRange(1, 128)]
    [int]$BatchSize = 16,
    
    [Parameter(Mandatory=$false)]
    [ValidateRange(320, 2048)]
    [int]$ImageSize = 1280,
    
    [Parameter(Mandatory=$false)]
    [switch]$UseWandB,
    
    [Parameter(Mandatory=$false)]
    [switch]$PrepareDataset,
    
    [Parameter(Mandatory=$false)]
    [switch]$ExportModel,
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"

# Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DataDir = Join-Path $ProjectRoot "data"
$TrainingDir = Join-Path $DataDir "training"
$ConfigsDir = Join-Path $ProjectRoot "configs"
$LogsDir = Join-Path $ProjectRoot "logs"
$RunsDir = Join-Path $ProjectRoot "runs"

# Ensure directories exist
$RequiredDirs = @($DataDir, $TrainingDir, $ConfigsDir, $LogsDir, $RunsDir)
foreach ($dir in $RequiredDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# Configuration files
$DatasetConfig = Join-Path $TrainingDir "sar_dataset.yaml"
$TrainConfig = Join-Path $ConfigsDir "train_config.yaml"
$DatasetPrepScript = Join-Path $ProjectRoot "training" "dataset_prep.py"
$TrainScript = Join-Path $ProjectRoot "training" "train.py"

function Write-Banner {
    param([string]$Text)
    
    $Border = "=" * 60
    Write-Host $Border -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Yellow
    Write-Host $Border -ForegroundColor Cyan
}

function Test-PythonEnvironment {
    Write-Host "Checking Python environment..." -ForegroundColor Blue
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Error "Python not found. Please install Python 3.8+"
        return $false
    }
    
    # Check required packages
    $RequiredPackages = @(
        "ultralytics",
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "albumentations",
        "matplotlib",
        "pyyaml",
        "tqdm"
    )
    
    if ($UseWandB) {
        $RequiredPackages += "wandb"
    }
    
    foreach ($package in $RequiredPackages) {
        try {
            $result = python -c "import $($package.Replace('-', '_')); print('✓')" 2>&1
            if ($result -eq "✓") {
                Write-Host "✓ $package" -ForegroundColor Green
            } else {
                Write-Host "✗ $package (not found)" -ForegroundColor Red
                Write-Host "Install with: pip install $package" -ForegroundColor Yellow
                return $false
            }
        } catch {
            Write-Host "✗ $package (not found)" -ForegroundColor Red
            Write-Host "Install with: pip install $package" -ForegroundColor Yellow
            return $false
        }
    }
    
    return $true
}

function Test-RequiredFiles {
    Write-Host "Checking required files..." -ForegroundColor Blue
    
    $RequiredFiles = @(
        @{Path = $DatasetConfig; Description = "Dataset configuration"},
        @{Path = $TrainConfig; Description = "Training configuration"},
        @{Path = $TrainScript; Description = "Training script"}
    )
    
    if ($PrepareDataset) {
        $RequiredFiles += @{Path = $DatasetPrepScript; Description = "Dataset preparation script"}
    }
    
    $allExist = $true
    foreach ($file in $RequiredFiles) {
        if (Test-Path $file.Path) {
            Write-Host "✓ $($file.Description): $($file.Path)" -ForegroundColor Green
        } else {
            Write-Host "✗ $($file.Description): $($file.Path)" -ForegroundColor Red
            $allExist = $false
        }
    }
    
    return $allExist
}

function Invoke-DatasetPreparation {
    Write-Banner "Dataset Preparation"
    
    if (-not (Test-Path $DatasetPrepScript)) {
        Write-Error "Dataset preparation script not found: $DatasetPrepScript"
        return $false
    }
    
    Write-Host "Running dataset preparation..." -ForegroundColor Blue
    
    $prepArgs = @(
        "--output_dir", $TrainingDir,
        "--target_size", $ImageSize,
        "--train_split", "0.7",
        "--val_split", "0.2",
        "--test_split", "0.1",
        "--augment_factor", "3",
        "--validate_dataset"
    )
    
    if ($DryRun) {
        Write-Host "DRY RUN: Would execute: python $DatasetPrepScript $($prepArgs -join ' ')" -ForegroundColor Yellow
        return $true
    }
    
    try {
        $result = python $DatasetPrepScript @prepArgs
        Write-Host "Dataset preparation completed successfully" -ForegroundColor Green
        return $true
    } catch {
        Write-Error "Dataset preparation failed: $_"
        return $false
    }
}

function Invoke-ModelTraining {
    Write-Banner "Model Training"
    
    Write-Host "Starting YOLOv8 training with the following parameters:" -ForegroundColor Blue
    Write-Host "  Model: $Model" -ForegroundColor White
    Write-Host "  Epochs: $Epochs" -ForegroundColor White
    Write-Host "  Batch Size: $BatchSize" -ForegroundColor White
    Write-Host "  Image Size: $ImageSize" -ForegroundColor White
    Write-Host "  Dataset Config: $DatasetConfig" -ForegroundColor White
    Write-Host "  Training Config: $TrainConfig" -ForegroundColor White
    
    # Method 1: Using ultralytics CLI (as specified in requirements)
    Write-Host "\nMethod 1: Using ultralytics CLI command" -ForegroundColor Magenta
    
    $cliArgs = @(
        "task=detect",
        "mode=train",
        "model=$Model",
        "data=$DatasetConfig",
        "epochs=$Epochs",
        "imgsz=$ImageSize",
        "batch=$BatchSize",
        "project=runs/train",
        "name=sar_yolov8_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
        "save=true",
        "plots=true",
        "val=true",
        "verbose=true"
    )
    
    $cliCommand = "yolo $($cliArgs -join ' ')"
    Write-Host "Command: $cliCommand" -ForegroundColor Cyan
    
    if ($DryRun) {
        Write-Host "DRY RUN: Would execute CLI training command" -ForegroundColor Yellow
    } else {
        Write-Host "Executing CLI training..." -ForegroundColor Blue
        try {
            Invoke-Expression $cliCommand
            Write-Host "CLI training completed successfully" -ForegroundColor Green
        } catch {
            Write-Warning "CLI training failed: $_"
        }
    }
    
    # Method 2: Using our enhanced training script
    Write-Host "\nMethod 2: Using enhanced training script" -ForegroundColor Magenta
    
    $scriptArgs = @(
        "--config", $TrainConfig,
        "--model", $Model,
        "--data", $DatasetConfig,
        "--epochs", $Epochs,
        "--batch", $BatchSize,
        "--imgsz", $ImageSize
    )
    
    if ($UseWandB) {
        $scriptArgs += "--wandb"
    }
    
    $scriptCommand = "python $TrainScript $($scriptArgs -join ' ')"
    Write-Host "Command: $scriptCommand" -ForegroundColor Cyan
    
    if ($DryRun) {
        Write-Host "DRY RUN: Would execute enhanced training script" -ForegroundColor Yellow
        return $true
    }
    
    Write-Host "Executing enhanced training..." -ForegroundColor Blue
    try {
        Invoke-Expression $scriptCommand
        Write-Host "Enhanced training completed successfully" -ForegroundColor Green
        return $true
    } catch {
        Write-Error "Enhanced training failed: $_"
        return $false
    }
}

function Export-TrainedModel {
    param([string]$ModelPath)
    
    Write-Banner "Model Export"
    
    if (-not (Test-Path $ModelPath)) {
        Write-Error "Trained model not found: $ModelPath"
        return $false
    }
    
    Write-Host "Exporting model to deployment formats..." -ForegroundColor Blue
    
    # Export to ONNX
    $onnxCommand = "yolo export model=$ModelPath format=onnx opset=11 simplify=true"
    Write-Host "ONNX Export: $onnxCommand" -ForegroundColor Cyan
    
    if ($DryRun) {
        Write-Host "DRY RUN: Would export to ONNX" -ForegroundColor Yellow
    } else {
        try {
            Invoke-Expression $onnxCommand
            Write-Host "✓ ONNX export completed" -ForegroundColor Green
        } catch {
            Write-Warning "ONNX export failed: $_"
        }
    }
    
    # Export to TensorRT (if available)
    $tensorrtCommand = "trtexec --onnx=$($ModelPath.Replace('.pt', '.onnx')) --saveEngine=$($ModelPath.Replace('.pt', '.trt')) --fp16"
    Write-Host "TensorRT Export: $tensorrtCommand" -ForegroundColor Cyan
    
    if ($DryRun) {
        Write-Host "DRY RUN: Would export to TensorRT" -ForegroundColor Yellow
    } else {
        try {
            # Check if trtexec is available
            $trtexecAvailable = Get-Command trtexec -ErrorAction SilentlyContinue
            if ($trtexecAvailable) {
                Invoke-Expression $tensorrtCommand
                Write-Host "✓ TensorRT export completed" -ForegroundColor Green
            } else {
                Write-Warning "trtexec not found. Install TensorRT for TensorRT export."
            }
        } catch {
            Write-Warning "TensorRT export failed: $_"
        }
    }
    
    return $true
}

function Show-TrainingResults {
    param([string]$RunsPath)
    
    Write-Banner "Training Results"
    
    if (Test-Path $RunsPath) {
        $latestRun = Get-ChildItem $RunsPath | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        
        if ($latestRun) {
            $resultsPath = Join-Path $latestRun.FullName "results.png"
            $weightsPath = Join-Path $latestRun.FullName "weights" "best.pt"
            $metricsPath = Join-Path $latestRun.FullName "sar_metrics.json"
            
            Write-Host "Latest training run: $($latestRun.Name)" -ForegroundColor Green
            
            if (Test-Path $weightsPath) {
                Write-Host "✓ Best model weights: $weightsPath" -ForegroundColor Green
            }
            
            if (Test-Path $resultsPath) {
                Write-Host "✓ Training plots: $resultsPath" -ForegroundColor Green
            }
            
            if (Test-Path $metricsPath) {
                Write-Host "✓ SAR metrics: $metricsPath" -ForegroundColor Green
                
                try {
                    $metrics = Get-Content $metricsPath | ConvertFrom-Json
                    if ($metrics.best_metrics) {
                        Write-Host "\nBest Metrics:" -ForegroundColor Yellow
                        Write-Host "  mAP@0.5: $($metrics.best_metrics.val_map50)" -ForegroundColor White
                        Write-Host "  mAP@0.5:0.95: $($metrics.best_metrics.val_map)" -ForegroundColor White
                        Write-Host "  Precision: $($metrics.best_metrics.val_precision)" -ForegroundColor White
                        Write-Host "  Recall: $($metrics.best_metrics.val_recall)" -ForegroundColor White
                        if ($metrics.best_metrics.fps) {
                            Write-Host "  FPS: $($metrics.best_metrics.fps)" -ForegroundColor White
                        }
                    }
                } catch {
                    Write-Warning "Could not parse metrics file"
                }
            }
            
            return $weightsPath
        }
    }
    
    return $null
}

# Main execution
try {
    Write-Banner "SAR YOLOv8 Training Pipeline"
    
    # Environment checks
    if (-not (Test-PythonEnvironment)) {
        Write-Error "Python environment check failed"
        exit 1
    }
    
    if (-not (Test-RequiredFiles)) {
        Write-Error "Required files check failed"
        exit 1
    }
    
    # Dataset preparation
    if ($PrepareDataset) {
        if (-not (Invoke-DatasetPreparation)) {
            Write-Error "Dataset preparation failed"
            exit 1
        }
    }
    
    # Model training
    if (-not (Invoke-ModelTraining)) {
        Write-Error "Model training failed"
        exit 1
    }
    
    # Show results
    $bestModelPath = Show-TrainingResults (Join-Path $ProjectRoot "runs" "train")
    
    # Model export
    if ($ExportModel -and $bestModelPath) {
        Export-TrainedModel $bestModelPath
    }
    
    Write-Banner "Training Pipeline Completed Successfully!"
    
    if ($bestModelPath) {
        Write-Host "Best model saved to: $bestModelPath" -ForegroundColor Green
    }
    
} catch {
    Write-Error "Training pipeline failed: $_"
    exit 1
}