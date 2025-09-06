#!/bin/bash
# Training wrapper script for Foresight SAR models
# This script handles dataset augmentation and YOLOv8 finetuning for aerial people detection

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
MODELS_DIR="$PROJECT_ROOT/models"
TRAINING_DIR="$MODELS_DIR/training"
OUTPUT_DIR="$TRAINING_DIR/runs"
CONFIG_DIR="$PROJECT_ROOT/config"

# Default parameters
DATASET_NAME="aerial_people"
MODEL_SIZE="yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
EPOCHS=100
BATCH_SIZE=16
IMAGE_SIZE=640
WORKERS=4
DEVICE="auto"  # auto, cpu, 0, 1, etc.
PRETRAINED=true
AUGMENT=true
VALIDATION_SPLIT=0.2
TEST_SPLIT=0.1
SEED=42
RESUME=""
PROJECT_NAME="foresight_sar"
EXPERIMENT_NAME=""
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Foresight SAR Training Script

Usage: $0 [OPTIONS]

Options:
    -d, --dataset NAME          Dataset name (default: $DATASET_NAME)
    -m, --model SIZE           Model size: yolov8n|s|m|l|x (default: $MODEL_SIZE)
    -e, --epochs NUM           Number of training epochs (default: $EPOCHS)
    -b, --batch-size NUM       Batch size (default: $BATCH_SIZE)
    -i, --image-size NUM       Image size (default: $IMAGE_SIZE)
    -w, --workers NUM          Number of data loading workers (default: $WORKERS)
    --device DEVICE            Training device: auto|cpu|0|1|... (default: $DEVICE)
    --no-pretrained           Don't use pretrained weights
    --no-augment              Disable data augmentation
    --val-split FLOAT         Validation split ratio (default: $VALIDATION_SPLIT)
    --test-split FLOAT        Test split ratio (default: $TEST_SPLIT)
    --seed NUM                Random seed (default: $SEED)
    --resume PATH             Resume training from checkpoint
    --project NAME            Project name (default: $PROJECT_NAME)
    --experiment NAME         Experiment name (auto-generated if not provided)
    --verbose                 Verbose output
    -h, --help                Show this help message

Examples:
    # Basic training with default parameters
    $0
    
    # Train with custom dataset and larger model
    $0 --dataset my_aerial_data --model yolov8m --epochs 200
    
    # Resume training from checkpoint
    $0 --resume runs/train/exp1/weights/last.pt
    
    # Train on specific GPU with custom batch size
    $0 --device 0 --batch-size 32 --image-size 1024

Dataset Structure:
    data/
    └── {dataset_name}/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── dataset.yaml

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_NAME="$2"
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
        -i|--image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-pretrained)
            PRETRAINED=false
            shift
            ;;
        --no-augment)
            AUGMENT=false
            shift
            ;;
        --val-split)
            VALIDATION_SPLIT="$2"
            shift 2
            ;;
        --test-split)
            TEST_SPLIT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate model size
case $MODEL_SIZE in
    yolov8n|yolov8s|yolov8m|yolov8l|yolov8x)
        ;;
    *)
        log_error "Invalid model size: $MODEL_SIZE. Must be one of: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x"
        exit 1
        ;;
esac

# Generate experiment name if not provided
if [[ -z "$EXPERIMENT_NAME" ]]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXPERIMENT_NAME="${MODEL_SIZE}_${DATASET_NAME}_${TIMESTAMP}"
fi

# Set paths
DATASET_PATH="$DATA_DIR/$DATASET_NAME"
DATASET_YAML="$DATASET_PATH/dataset.yaml"
EXPERIMENT_DIR="$OUTPUT_DIR/train/$EXPERIMENT_NAME"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$EXPERIMENT_DIR"

# Logging setup
LOG_FILE="$EXPERIMENT_DIR/training.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log_info "Starting Foresight SAR training pipeline"
log_info "Experiment: $EXPERIMENT_NAME"
log_info "Dataset: $DATASET_NAME"
log_info "Model: $MODEL_SIZE"
log_info "Output directory: $EXPERIMENT_DIR"

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is required but not installed"
        exit 1
    fi
    
    # Check pip packages
    local required_packages=("ultralytics" "torch" "torchvision" "opencv-python" "pillow" "numpy" "matplotlib" "pyyaml")
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log_warning "$package not found, installing..."
            pip3 install "$package"
        fi
    done
    
    log_success "Dependencies check completed"
}

# Validate dataset
validate_dataset() {
    log_info "Validating dataset: $DATASET_NAME"
    
    if [[ ! -d "$DATASET_PATH" ]]; then
        log_error "Dataset directory not found: $DATASET_PATH"
        log_info "Please run dataset_prep.py first to prepare your dataset"
        exit 1
    fi
    
    if [[ ! -f "$DATASET_YAML" ]]; then
        log_error "Dataset configuration not found: $DATASET_YAML"
        log_info "Please ensure dataset.yaml exists in the dataset directory"
        exit 1
    fi
    
    # Check required directories
    local required_dirs=("images/train" "labels/train")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$DATASET_PATH/$dir" ]]; then
            log_error "Required directory not found: $DATASET_PATH/$dir"
            exit 1
        fi
    done
    
    # Count training samples
    local train_images=$(find "$DATASET_PATH/images/train" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
    local train_labels=$(find "$DATASET_PATH/labels/train" -name "*.txt" | wc -l)
    
    log_info "Training images: $train_images"
    log_info "Training labels: $train_labels"
    
    if [[ $train_images -eq 0 ]]; then
        log_error "No training images found"
        exit 1
    fi
    
    if [[ $train_labels -eq 0 ]]; then
        log_warning "No training labels found - this will be unsupervised training"
    fi
    
    log_success "Dataset validation completed"
}

# Setup training environment
setup_environment() {
    log_info "Setting up training environment..."
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    
    # Create training configuration
    local config_file="$EXPERIMENT_DIR/training_config.yaml"
    
    cat > "$config_file" << EOF
# Foresight SAR Training Configuration
# Generated on $(date)

project: $PROJECT_NAME
experiment: $EXPERIMENT_NAME
dataset: $DATASET_NAME
model: $MODEL_SIZE

# Training parameters
epochs: $EPOCHS
batch_size: $BATCH_SIZE
image_size: $IMAGE_SIZE
workers: $WORKERS
device: $DEVICE
pretrained: $PRETRAINED
augment: $AUGMENT
seed: $SEED

# Data splits
validation_split: $VALIDATION_SPLIT
test_split: $TEST_SPLIT

# Paths
dataset_path: $DATASET_PATH
dataset_yaml: $DATASET_YAML
output_dir: $EXPERIMENT_DIR

EOF
    
    log_info "Training configuration saved to: $config_file"
    log_success "Environment setup completed"
}

# Run data augmentation
run_augmentation() {
    if [[ "$AUGMENT" == "true" ]]; then
        log_info "Running data augmentation..."
        
        local augment_script="$TRAINING_DIR/augment_data.py"
        
        if [[ -f "$augment_script" ]]; then
            python3 "$augment_script" \
                --dataset "$DATASET_PATH" \
                --output "$DATASET_PATH/augmented" \
                --seed "$SEED" \
                $([ "$VERBOSE" == "true" ] && echo "--verbose")
            
            log_success "Data augmentation completed"
        else
            log_warning "Augmentation script not found: $augment_script"
            log_warning "Skipping data augmentation"
        fi
    else
        log_info "Data augmentation disabled"
    fi
}

# Train YOLOv8 model
train_model() {
    log_info "Starting YOLOv8 training..."
    
    # Build training command
    local train_cmd="python3 -c \"
import os
os.chdir('$EXPERIMENT_DIR')
from ultralytics import YOLO

# Load model
model = YOLO('$MODEL_SIZE.pt' if $PRETRAINED else '$MODEL_SIZE.yaml')

# Train model
results = model.train(
    data='$DATASET_YAML',
    epochs=$EPOCHS,
    imgsz=$IMAGE_SIZE,
    batch=$BATCH_SIZE,
    workers=$WORKERS,
    device='$DEVICE',
    project='$OUTPUT_DIR',
    name='$EXPERIMENT_NAME',
    exist_ok=True,
    pretrained=$PRETRAINED,
    augment=$AUGMENT,
    seed=$SEED,
    $([ -n "$RESUME" ] && echo "resume='$RESUME',")
    verbose=$([ "$VERBOSE" == "true" ] && echo "True" || echo "False")
)

print('Training completed successfully!')
print(f'Best model saved to: {results.save_dir}/weights/best.pt')
print(f'Last model saved to: {results.save_dir}/weights/last.pt')
\""
    
    # Execute training
    if eval "$train_cmd"; then
        log_success "Model training completed successfully"
    else
        log_error "Model training failed"
        exit 1
    fi
}

# Validate trained model
validate_model() {
    log_info "Validating trained model..."
    
    local best_model="$EXPERIMENT_DIR/weights/best.pt"
    
    if [[ -f "$best_model" ]]; then
        python3 -c "
from ultralytics import YOLO

# Load trained model
model = YOLO('$best_model')

# Validate model
results = model.val(
    data='$DATASET_YAML',
    imgsz=$IMAGE_SIZE,
    batch=$BATCH_SIZE,
    device='$DEVICE',
    project='$OUTPUT_DIR',
    name='${EXPERIMENT_NAME}_val',
    exist_ok=True
)

print('Validation completed!')
print(f'mAP50: {results.box.map50:.4f}')
print(f'mAP50-95: {results.box.map:.4f}')
"
        
        log_success "Model validation completed"
    else
        log_warning "Best model not found, skipping validation"
    fi
}

# Generate training report
generate_report() {
    log_info "Generating training report..."
    
    local report_file="$EXPERIMENT_DIR/training_report.md"
    
    cat > "$report_file" << EOF
# Foresight SAR Training Report

**Experiment:** $EXPERIMENT_NAME  
**Date:** $(date)  
**Dataset:** $DATASET_NAME  
**Model:** $MODEL_SIZE  

## Configuration

- **Epochs:** $EPOCHS
- **Batch Size:** $BATCH_SIZE
- **Image Size:** $IMAGE_SIZE
- **Device:** $DEVICE
- **Pretrained:** $PRETRAINED
- **Augmentation:** $AUGMENT
- **Seed:** $SEED

## Dataset

- **Path:** $DATASET_PATH
- **Validation Split:** $VALIDATION_SPLIT
- **Test Split:** $TEST_SPLIT

## Results

$([ -f "$EXPERIMENT_DIR/results.csv" ] && echo "Training metrics saved to: results.csv" || echo "Training metrics not available")

## Model Files

- **Best Model:** weights/best.pt
- **Last Model:** weights/last.pt
- **Training Log:** training.log

## Usage

\`\`\`python
from ultralytics import YOLO

# Load trained model
model = YOLO('$EXPERIMENT_DIR/weights/best.pt')

# Run inference
results = model('path/to/image.jpg')
\`\`\`

EOF
    
    log_success "Training report generated: $report_file"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup operations here
    log_success "Cleanup completed"
}

# Main execution
main() {
    # Set trap for cleanup on exit
    trap cleanup EXIT
    
    # Run pipeline steps
    check_dependencies
    validate_dataset
    setup_environment
    run_augmentation
    train_model
    validate_model
    generate_report
    
    log_success "Training pipeline completed successfully!"
    log_info "Experiment directory: $EXPERIMENT_DIR"
    log_info "Best model: $EXPERIMENT_DIR/weights/best.pt"
    log_info "Training log: $EXPERIMENT_DIR/training.log"
    log_info "Report: $EXPERIMENT_DIR/training_report.md"
}

# Execute main function
main "$@"