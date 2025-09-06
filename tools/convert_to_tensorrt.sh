#!/bin/bash

# TensorRT Model Conversion Script for Foresight SAR System
# This script converts ONNX models to TensorRT format for optimized inference on Jetson devices

set -e  # Exit on any error

# Configuration
DEFAULT_WORKSPACE_SIZE="1GB"
DEFAULT_PRECISION="fp16"
DEFAULT_BATCH_SIZE=1
DEFAULT_MAX_BATCH_SIZE=8

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
Usage: $0 [OPTIONS] <input_onnx_model> <output_tensorrt_engine>

Convert ONNX models to TensorRT engines for optimized inference on Jetson devices.

Arguments:
  input_onnx_model      Path to input ONNX model file
  output_tensorrt_engine Path to output TensorRT engine file

Options:
  -h, --help           Show this help message
  -p, --precision      Precision mode: fp32, fp16, int8 (default: ${DEFAULT_PRECISION})
  -w, --workspace      Workspace size for TensorRT (default: ${DEFAULT_WORKSPACE_SIZE})
  -b, --batch-size     Fixed batch size (default: ${DEFAULT_BATCH_SIZE})
  -m, --max-batch      Maximum batch size for dynamic batching (default: ${DEFAULT_MAX_BATCH_SIZE})
  -s, --input-shape    Input shape override (e.g., "1,3,640,640")
  -v, --verbose        Enable verbose output
  -d, --device         Target device: jetson_nano, jetson_xavier, jetson_orin (auto-detect if not specified)
  --dynamic-shapes     Enable dynamic input shapes
  --calibration-data   Path to calibration dataset for INT8 quantization
  --profile            Create multiple optimization profiles
  --validate           Validate converted model against original

Examples:
  # Basic conversion with FP16 precision
  $0 yolov8n.onnx yolov8n.trt
  
  # Convert with INT8 quantization
  $0 -p int8 --calibration-data ./calibration_images yolov8n.onnx yolov8n_int8.trt
  
  # Convert with dynamic batch sizes
  $0 --dynamic-shapes -m 8 yolov8n.onnx yolov8n_dynamic.trt
  
  # Convert for specific Jetson device
  $0 -d jetson_orin -p fp16 yolov8n.onnx yolov8n_orin.trt

EOF
}

# Function to detect Jetson device
detect_jetson_device() {
    if [ -f "/etc/nv_tegra_release" ]; then
        local tegra_info=$(cat /etc/nv_tegra_release)
        if echo "$tegra_info" | grep -q "R32"; then
            echo "jetson_nano"
        elif echo "$tegra_info" | grep -q "R34"; then
            echo "jetson_xavier"
        elif echo "$tegra_info" | grep -q "R35"; then
            echo "jetson_orin"
        else
            echo "unknown"
        fi
    else
        echo "not_jetson"
    fi
}

# Function to check TensorRT installation
check_tensorrt() {
    if ! command -v trtexec &> /dev/null; then
        print_error "TensorRT (trtexec) not found. Please install TensorRT."
        print_info "For Jetson devices, TensorRT is included in JetPack."
        print_info "For x86 systems, install TensorRT from NVIDIA Developer website."
        exit 1
    fi
    
    local trt_version=$(trtexec --help 2>&1 | grep -o "TensorRT [0-9]\+\.[0-9]\+\.[0-9]\+" | head -1)
    print_info "Found $trt_version"
}

# Function to validate ONNX model
validate_onnx() {
    local onnx_file="$1"
    
    if [ ! -f "$onnx_file" ]; then
        print_error "ONNX model file not found: $onnx_file"
        exit 1
    fi
    
    print_info "Validating ONNX model: $onnx_file"
    
    # Check if onnx python package is available
    if command -v python3 &> /dev/null; then
        python3 -c "
import onnx
try:
    model = onnx.load('$onnx_file')
    onnx.checker.check_model(model)
    print('ONNX model is valid')
except Exception as e:
    print(f'ONNX validation failed: {e}')
    exit(1)
" 2>/dev/null || print_warning "Could not validate ONNX model (onnx package not available)"
    fi
}

# Function to get model info
get_model_info() {
    local onnx_file="$1"
    
    print_info "Analyzing model: $onnx_file"
    
    if command -v python3 &> /dev/null; then
        python3 -c "
import onnx
try:
    model = onnx.load('$onnx_file')
    print(f'Model IR version: {model.ir_version}')
    print(f'Producer: {model.producer_name} {model.producer_version}')
    
    # Print input/output info
    print('\nInputs:')
    for input_tensor in model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_tensor.type.tensor_type.shape.dim]
        print(f'  {input_tensor.name}: {input_tensor.type.tensor_type.elem_type} {shape}')
    
    print('\nOutputs:')
    for output_tensor in model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_tensor.type.tensor_type.shape.dim]
        print(f'  {output_tensor.name}: {output_tensor.type.tensor_type.elem_type} {shape}')
except Exception as e:
    print(f'Could not analyze model: {e}')
" 2>/dev/null || print_warning "Could not analyze model (onnx package not available)"
    fi
}

# Function to convert workspace size to bytes
parse_workspace_size() {
    local size_str="$1"
    local size_num=$(echo "$size_str" | grep -o '[0-9]\+')
    local size_unit=$(echo "$size_str" | grep -o '[A-Za-z]\+' | tr '[:upper:]' '[:lower:]')
    
    case "$size_unit" in
        "gb"|"g")
            echo $((size_num * 1024 * 1024 * 1024))
            ;;
        "mb"|"m")
            echo $((size_num * 1024 * 1024))
            ;;
        "kb"|"k")
            echo $((size_num * 1024))
            ;;
        "b"|"")
            echo "$size_num"
            ;;
        *)
            print_error "Invalid workspace size unit: $size_unit"
            exit 1
            ;;
    esac
}

# Function to build TensorRT command
build_trt_command() {
    local onnx_file="$1"
    local output_file="$2"
    local precision="$3"
    local workspace_bytes="$4"
    local batch_size="$5"
    local max_batch_size="$6"
    local input_shape="$7"
    local dynamic_shapes="$8"
    local calibration_data="$9"
    local verbose="${10}"
    local device="${11}"
    local profile="${12}"
    
    local cmd="trtexec --onnx='$onnx_file' --saveEngine='$output_file'"
    
    # Precision settings
    case "$precision" in
        "fp16")
            cmd="$cmd --fp16"
            ;;
        "int8")
            cmd="$cmd --int8"
            if [ -n "$calibration_data" ]; then
                cmd="$cmd --calib='$calibration_data'"
            else
                print_warning "INT8 precision specified but no calibration data provided"
                print_warning "Using random calibration (may result in poor accuracy)"
            fi
            ;;
        "fp32")
            # Default precision, no flag needed
            ;;
        *)
            print_error "Invalid precision: $precision"
            exit 1
            ;;
    esac
    
    # Workspace size
    cmd="$cmd --workspace=$workspace_bytes"
    
    # Batch size settings
    if [ "$dynamic_shapes" = "true" ]; then
        cmd="$cmd --minShapes=input:1,3,640,640 --optShapes=input:$batch_size,3,640,640 --maxShapes=input:$max_batch_size,3,640,640"
    else
        if [ -n "$input_shape" ]; then
            cmd="$cmd --shapes=input:$input_shape"
        else
            cmd="$cmd --shapes=input:$batch_size,3,640,640"
        fi
    fi
    
    # Device-specific optimizations
    case "$device" in
        "jetson_nano")
            cmd="$cmd --useDLACore=0 --allowGPUFallback"
            ;;
        "jetson_xavier")
            cmd="$cmd --useDLACore=0 --allowGPUFallback"
            ;;
        "jetson_orin")
            # Orin has more powerful GPU, less need for DLA
            ;;
    esac
    
    # Verbose output
    if [ "$verbose" = "true" ]; then
        cmd="$cmd --verbose"
    fi
    
    # Profiling
    if [ "$profile" = "true" ]; then
        cmd="$cmd --profilingVerbosity=detailed"
    fi
    
    echo "$cmd"
}

# Function to validate converted engine
validate_engine() {
    local engine_file="$1"
    local onnx_file="$2"
    
    if [ ! -f "$engine_file" ]; then
        print_error "TensorRT engine was not created: $engine_file"
        return 1
    fi
    
    print_info "Validating TensorRT engine: $engine_file"
    
    # Basic validation - check if engine can be loaded
    local validation_cmd="trtexec --loadEngine='$engine_file' --warmUp=0 --iterations=1"
    
    if $validation_cmd > /dev/null 2>&1; then
        print_success "TensorRT engine validation passed"
        
        # Get engine info
        local engine_size=$(du -h "$engine_file" | cut -f1)
        local onnx_size=$(du -h "$onnx_file" | cut -f1)
        
        print_info "Original ONNX size: $onnx_size"
        print_info "TensorRT engine size: $engine_size"
        
        return 0
    else
        print_error "TensorRT engine validation failed"
        return 1
    fi
}

# Function to benchmark engine
benchmark_engine() {
    local engine_file="$1"
    local iterations="${2:-100}"
    
    print_info "Benchmarking TensorRT engine (${iterations} iterations)..."
    
    local benchmark_cmd="trtexec --loadEngine='$engine_file' --warmUp=10 --iterations=$iterations"
    
    if $benchmark_cmd; then
        print_success "Benchmark completed"
    else
        print_warning "Benchmark failed"
    fi
}

# Main conversion function
convert_model() {
    local onnx_file="$1"
    local output_file="$2"
    local precision="$3"
    local workspace="$4"
    local batch_size="$5"
    local max_batch_size="$6"
    local input_shape="$7"
    local dynamic_shapes="$8"
    local calibration_data="$9"
    local verbose="${10}"
    local device="${11}"
    local profile="${12}"
    local validate="${13}"
    
    # Validate inputs
    validate_onnx "$onnx_file"
    
    # Get model info
    get_model_info "$onnx_file"
    
    # Parse workspace size
    local workspace_bytes=$(parse_workspace_size "$workspace")
    
    # Build TensorRT command
    local trt_cmd=$(build_trt_command "$onnx_file" "$output_file" "$precision" "$workspace_bytes" "$batch_size" "$max_batch_size" "$input_shape" "$dynamic_shapes" "$calibration_data" "$verbose" "$device" "$profile")
    
    print_info "Starting TensorRT conversion..."
    print_info "Command: $trt_cmd"
    
    # Execute conversion
    if eval "$trt_cmd"; then
        print_success "TensorRT conversion completed: $output_file"
        
        # Validate if requested
        if [ "$validate" = "true" ]; then
            validate_engine "$output_file" "$onnx_file"
        fi
        
        # Benchmark the engine
        benchmark_engine "$output_file"
        
        return 0
    else
        print_error "TensorRT conversion failed"
        return 1
    fi
}

# Parse command line arguments
PRECISION="$DEFAULT_PRECISION"
WORKSPACE="$DEFAULT_WORKSPACE_SIZE"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
MAX_BATCH_SIZE="$DEFAULT_MAX_BATCH_SIZE"
INPUT_SHAPE=""
DYNAMIC_SHAPES="false"
CALIBRATION_DATA=""
VERBOSE="false"
DEVICE=""
PROFILE="false"
VALIDATE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -p|--precision)
            PRECISION="$2"
            shift 2
            ;;
        -w|--workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -m|--max-batch)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        -s|--input-shape)
            INPUT_SHAPE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        --dynamic-shapes)
            DYNAMIC_SHAPES="true"
            shift
            ;;
        --calibration-data)
            CALIBRATION_DATA="$2"
            shift 2
            ;;
        --profile)
            PROFILE="true"
            shift
            ;;
        --validate)
            VALIDATE="true"
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Check required arguments
if [ $# -lt 2 ]; then
    print_error "Missing required arguments"
    show_usage
    exit 1
fi

ONNX_FILE="$1"
OUTPUT_FILE="$2"

# Auto-detect device if not specified
if [ -z "$DEVICE" ]; then
    DEVICE=$(detect_jetson_device)
    if [ "$DEVICE" != "not_jetson" ] && [ "$DEVICE" != "unknown" ]; then
        print_info "Detected Jetson device: $DEVICE"
    else
        print_info "Running on non-Jetson device"
        DEVICE="generic"
    fi
fi

# Check TensorRT installation
check_tensorrt

# Create output directory if it doesn't exist
output_dir=$(dirname "$OUTPUT_FILE")
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

# Start conversion
print_info "Converting ONNX model to TensorRT engine"
print_info "Input: $ONNX_FILE"
print_info "Output: $OUTPUT_FILE"
print_info "Precision: $PRECISION"
print_info "Workspace: $WORKSPACE"
print_info "Batch size: $BATCH_SIZE"
print_info "Device: $DEVICE"

# Perform conversion
if convert_model "$ONNX_FILE" "$OUTPUT_FILE" "$PRECISION" "$WORKSPACE" "$BATCH_SIZE" "$MAX_BATCH_SIZE" "$INPUT_SHAPE" "$DYNAMIC_SHAPES" "$CALIBRATION_DATA" "$VERBOSE" "$DEVICE" "$PROFILE" "$VALIDATE"; then
    print_success "Model conversion completed successfully!"
    print_info "TensorRT engine saved to: $OUTPUT_FILE"
    exit 0
else
    print_error "Model conversion failed!"
    exit 1
fi