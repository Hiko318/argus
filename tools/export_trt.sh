#!/bin/bash

# Export & TensorRT Pipeline Script
# This script exports YOLO models to ONNX format and converts them to TensorRT engines

set -e  # Exit on any error

# Default values
MODEL_PATH=""
OUTPUT_DIR="./exports"
ONNX_OPSET=12
TRT_FP16=true
TRT_WORKSPACE=4096  # MB
VERBOSE=false

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

Export YOLO models to ONNX and convert to TensorRT engines

OPTIONS:
    -m, --model PATH        Path to the YOLO model (.pt file) [REQUIRED]
    -o, --output DIR        Output directory (default: ./exports)
    --opset VERSION         ONNX opset version (default: 12)
    --no-fp16              Disable FP16 precision for TensorRT
    --workspace SIZE        TensorRT workspace size in MB (default: 4096)
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message

EXAMPLES:
    # Basic export
    $0 -m runs/detect/train/weights/best.pt
    
    # Custom output directory and settings
    $0 -m best.pt -o ./models --opset 11 --workspace 8192
    
    # Disable FP16 for compatibility
    $0 -m best.pt --no-fp16

REQUIREMENTS:
    - ultralytics (pip install ultralytics)
    - TensorRT and trtexec in PATH
    - CUDA toolkit

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --opset)
            ONNX_OPSET="$2"
            shift 2
            ;;
        --no-fp16)
            TRT_FP16=false
            shift
            ;;
        --workspace)
            TRT_WORKSPACE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
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

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    print_error "Model path is required. Use -m or --model to specify."
    show_usage
    exit 1
fi

# Check if model file exists
if [[ ! -f "$MODEL_PATH" ]]; then
    print_error "Model file not found: $MODEL_PATH"
    exit 1
fi

# Check dependencies
print_info "Checking dependencies..."

# Check if yolo command is available
if ! command -v yolo &> /dev/null; then
    print_error "YOLO command not found. Please install ultralytics: pip install ultralytics"
    exit 1
fi

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    print_warning "trtexec not found in PATH. TensorRT conversion will be skipped."
    print_warning "Please install TensorRT and ensure trtexec is in your PATH."
    TRT_AVAILABLE=false
else
    TRT_AVAILABLE=true
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get model basename for naming
MODEL_BASENAME=$(basename "$MODEL_PATH" .pt)
ONNX_PATH="$OUTPUT_DIR/${MODEL_BASENAME}.onnx"
TRT_PATH="$OUTPUT_DIR/${MODEL_BASENAME}.trt"

print_info "Starting export pipeline..."
print_info "Model: $MODEL_PATH"
print_info "Output directory: $OUTPUT_DIR"
print_info "ONNX opset: $ONNX_OPSET"
print_info "TensorRT FP16: $TRT_FP16"
print_info "TensorRT workspace: ${TRT_WORKSPACE}MB"

# Step 1: Export to ONNX
print_info "Step 1: Exporting YOLO model to ONNX..."

ONNX_CMD="yolo export model='$MODEL_PATH' format=onnx opset=$ONNX_OPSET"

if [[ "$VERBOSE" == "true" ]]; then
    print_info "Running: $ONNX_CMD"
fi

if eval "$ONNX_CMD"; then
    print_success "ONNX export completed successfully"
    
    # Find the generated ONNX file (YOLO exports to same directory as model)
    MODEL_DIR=$(dirname "$MODEL_PATH")
    GENERATED_ONNX="$MODEL_DIR/${MODEL_BASENAME}.onnx"
    
    if [[ -f "$GENERATED_ONNX" ]]; then
        # Move to output directory if different
        if [[ "$GENERATED_ONNX" != "$ONNX_PATH" ]]; then
            mv "$GENERATED_ONNX" "$ONNX_PATH"
            print_info "Moved ONNX file to: $ONNX_PATH"
        fi
    else
        print_error "ONNX file not found after export: $GENERATED_ONNX"
        exit 1
    fi
else
    print_error "ONNX export failed"
    exit 1
fi

# Step 2: Convert to TensorRT (if available)
if [[ "$TRT_AVAILABLE" == "true" ]]; then
    print_info "Step 2: Converting ONNX to TensorRT engine..."
    
    TRT_CMD="trtexec --onnx='$ONNX_PATH' --saveEngine='$TRT_PATH' --workspace=${TRT_WORKSPACE}"
    
    if [[ "$TRT_FP16" == "true" ]]; then
        TRT_CMD="$TRT_CMD --fp16"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        TRT_CMD="$TRT_CMD --verbose"
        print_info "Running: $TRT_CMD"
    fi
    
    if eval "$TRT_CMD"; then
        print_success "TensorRT conversion completed successfully"
        print_success "TensorRT engine saved to: $TRT_PATH"
    else
        print_error "TensorRT conversion failed"
        print_warning "ONNX file is still available at: $ONNX_PATH"
        exit 1
    fi
else
    print_warning "Skipping TensorRT conversion (trtexec not available)"
fi

# Step 3: Generate summary
print_info "Export pipeline completed!"
echo
print_success "=== EXPORT SUMMARY ==="
print_info "Original model: $MODEL_PATH"
print_info "ONNX model: $ONNX_PATH"

if [[ "$TRT_AVAILABLE" == "true" && -f "$TRT_PATH" ]]; then
    print_info "TensorRT engine: $TRT_PATH"
fi

echo
print_info "File sizes:"
if [[ -f "$MODEL_PATH" ]]; then
    echo "  Original (.pt): $(du -h '$MODEL_PATH' | cut -f1)"
fi
if [[ -f "$ONNX_PATH" ]]; then
    echo "  ONNX (.onnx): $(du -h '$ONNX_PATH' | cut -f1)"
fi
if [[ -f "$TRT_PATH" ]]; then
    echo "  TensorRT (.trt): $(du -h '$TRT_PATH' | cut -f1)"
fi

echo
print_success "Export pipeline finished successfully!"

# Step 4: Generate usage examples
cat << EOF

=== USAGE EXAMPLES ===

# Python inference with ONNX:
from ultralytics import YOLO
model = YOLO('$ONNX_PATH')
results = model('image.jpg')

# Python inference with TensorRT (if available):
from ultralytics import YOLO
model = YOLO('$TRT_PATH')
results = model('image.jpg')

# Command line inference:
yolo predict model='$ONNX_PATH' source='image.jpg'

EOF

print_info "For more information, see: https://docs.ultralytics.com/modes/export/"