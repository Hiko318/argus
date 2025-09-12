#!/bin/bash

# YOLOv8 Training Script for SAR Operations
# This script provides easy training automation with various configurations

set -e  # Exit on any error

# Default configuration
CONFIG_FILE="train_config.yaml"
DATASET_PATH="datasets/sar_dataset"
MODEL_SIZE="n"  # n, s, m, l, x
EPOCHS=100
BATCH_SIZE=16
DEVICE="0"
RESUME=""
EXPORT=false
VALIDATE=false
BENCHMARK=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

YOLOv8 Training Script for SAR Operations

Options:
    -c, --config FILE       Training configuration file (default: train_config.yaml)
    -d, --dataset PATH      Dataset path (default: datasets/sar_dataset)
    -m, --model SIZE        Model size: n, s, m, l, x (default: n)
    -e, --epochs NUM        Number of epochs (default: 100)
    -b, --batch-size NUM    Batch size (default: 16)
    --device DEVICE         Training device (default: 0)
    -r, --resume PATH       Resume from checkpoint
    --export                Export model after training
    --validate              Validate dataset before training
    --benchmark             Run performance benchmark after training
    -h, --help              Show this help message

Examples:
    # Basic training
    $0
    
    # Custom configuration
    $0 --config custom_config.yaml --epochs 200 --batch-size 32
    
    # Resume training
    $0 --resume runs/train/sar_yolov8/weights/last.pt
    
    # Full pipeline with validation and export
    $0 --validate --export --benchmark
    
    # High-performance training
    $0 --model l --batch-size 64 --epochs 300 --device 0

EOF
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check pip packages
    python3 -c "import ultralytics" 2>/dev/null || {
        print_error "Ultralytics not installed. Run: pip install ultralytics"
        exit 1
    }
    
    python3 -c "import torch" 2>/dev/null || {
        print_error "PyTorch not installed. Run: pip install torch torchvision"
        exit 1
    }
    
    # Check CUDA if using GPU
    if [[ "$DEVICE" != "cpu" ]]; then
        python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
            print_warning "CUDA not available. Falling back to CPU training"
            DEVICE="cpu"
        }
    fi
    
    print_success "Dependencies check passed"
}

# Function to validate dataset
validate_dataset() {
    print_info "Validating dataset structure..."
    
    if [[ ! -d "$DATASET_PATH" ]]; then
        print_error "Dataset directory not found: $DATASET_PATH"
        exit 1
    fi
    
    # Check required directories
    for split in train val test; do
        if [[ ! -d "$DATASET_PATH/images/$split" ]]; then
            print_error "Missing directory: $DATASET_PATH/images/$split"
            exit 1
        fi
        if [[ ! -d "$DATASET_PATH/labels/$split" ]]; then
            print_error "Missing directory: $DATASET_PATH/labels/$split"
            exit 1
        fi
    done
    
    # Check dataset.yaml
    if [[ ! -f "$DATASET_PATH/dataset.yaml" ]]; then
        print_error "Missing dataset.yaml file in $DATASET_PATH"
        exit 1
    fi
    
    # Count files
    train_images=$(find "$DATASET_PATH/images/train" -type f | wc -l)
    val_images=$(find "$DATASET_PATH/images/val" -type f | wc -l)
    
    if [[ $train_images -eq 0 ]]; then
        print_error "No training images found"
        exit 1
    fi
    
    if [[ $val_images -eq 0 ]]; then
        print_warning "No validation images found"
    fi
    
    print_success "Dataset validation passed ($train_images train, $val_images val images)"
}

# Function to setup training environment
setup_environment() {
    print_info "Setting up training environment..."
    
    # Create output directories
    mkdir -p runs/train
    mkdir -p logs/training
    
    # Set environment variables
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    
    # Log system info
    print_info "System Information:"
    echo "  Python: $(python3 --version)"
    echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
    echo "  Ultralytics: $(python3 -c 'import ultralytics; print(ultralytics.__version__)')"
    
    if [[ "$DEVICE" != "cpu" ]]; then
        echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "Not available")')"
        echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")')"
    fi
    
    print_success "Environment setup completed"
}

# Function to run training
run_training() {
    print_info "Starting YOLOv8 training..."
    
    # Build training command
    TRAIN_CMD="python3 train.py --config $CONFIG_FILE"
    
    # Add optional parameters
    if [[ -n "$RESUME" ]]; then
        TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
    fi
    
    if [[ "$EXPORT" == "true" ]]; then
        TRAIN_CMD="$TRAIN_CMD --export"
    fi
    
    # Override config parameters
    TRAIN_CMD="$TRAIN_CMD --data $DATASET_PATH --epochs $EPOCHS --batch-size $BATCH_SIZE --device $DEVICE"
    
    print_info "Training command: $TRAIN_CMD"
    
    # Run training with logging
    LOG_FILE="logs/training/train_$(date +%Y%m%d_%H%M%S).log"
    print_info "Logging to: $LOG_FILE"
    
    if ! $TRAIN_CMD 2>&1 | tee "$LOG_FILE"; then
        print_error "Training failed. Check log file: $LOG_FILE"
        exit 1
    fi
    
    print_success "Training completed successfully"
}

# Function to run benchmark
run_benchmark() {
    print_info "Running performance benchmark..."
    
    # Find best model
    BEST_MODEL=$(find runs/train -name "best.pt" | head -1)
    
    if [[ -z "$BEST_MODEL" ]]; then
        print_warning "No trained model found for benchmarking"
        return
    fi
    
    # Run evaluation
    python3 evaluate.py --model "$BEST_MODEL" --data "$DATASET_PATH" --benchmark --export-results
    
    print_success "Benchmark completed"
}

# Function to cleanup
cleanup() {
    print_info "Cleaning up temporary files..."
    # Add cleanup commands here if needed
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME="$2"
            shift 2
            ;;
        --export)
            EXPORT=true
            shift
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "Starting SAR YOLOv8 Training Pipeline"
    print_info "Configuration: $CONFIG_FILE"
    print_info "Dataset: $DATASET_PATH"
    print_info "Model: YOLO11$MODEL_SIZE"
    print_info "Epochs: $EPOCHS"
    print_info "Batch Size: $BATCH_SIZE"
    print_info "Device: $DEVICE"
    
    # Check dependencies
    check_dependencies
    
    # Validate dataset if requested
    if [[ "$VALIDATE" == "true" ]]; then
        validate_dataset
    fi
    
    # Setup environment
    setup_environment
    
    # Update config file with model size
    if [[ -f "$CONFIG_FILE" ]]; then
        # Create temporary config with updated model
        TEMP_CONFIG="temp_config_$(date +%s).yaml"
        sed "s/model: yolo11[nslmx].pt/model: yolo11${MODEL_SIZE}.pt/g" "$CONFIG_FILE" > "$TEMP_CONFIG"
        CONFIG_FILE="$TEMP_CONFIG"
    fi
    
    # Run training
    run_training
    
    # Run benchmark if requested
    if [[ "$BENCHMARK" == "true" ]]; then
        run_benchmark
    fi
    
    # Cleanup temporary config
    if [[ -f "$TEMP_CONFIG" ]]; then
        rm "$TEMP_CONFIG"
    fi
    
    print_success "SAR YOLOv8 training pipeline completed successfully!"
    
    # Show results summary
    print_info "Results Summary:"
    LATEST_RUN=$(find runs/train -maxdepth 1 -type d -name "sar_yolov8*" | sort | tail -1)
    if [[ -n "$LATEST_RUN" ]]; then
        echo "  Training results: $LATEST_RUN"
        if [[ -f "$LATEST_RUN/weights/best.pt" ]]; then
            echo "  Best model: $LATEST_RUN/weights/best.pt"
        fi
        if [[ -f "$LATEST_RUN/results.png" ]]; then
            echo "  Training plots: $LATEST_RUN/results.png"
        fi
    fi
}

# Run main function
main