#!/bin/bash

# Foresight Data Ingestion Runner
# This script manages the data ingestion pipeline

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$PROJECT_ROOT/data"
TEMP_DIR="$PROJECT_ROOT/temp"

# Create directories if they don't exist
mkdir -p "$LOG_DIR" "$DATA_DIR" "$TEMP_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/ingest.log"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if virtual environment exists and activate it
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    log "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
else
    log "Warning: Virtual environment not found. Using system Python."
fi

# Function to process phone stream data
process_phone_stream() {
    local input_file="$1"
    local output_dir="$2"
    
    log "Processing phone stream: $input_file"
    
    if [ ! -f "$input_file" ]; then
        error_exit "Input file not found: $input_file"
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run the processing pipeline
    python3 "$PROJECT_ROOT/src/backend/services/pipeline.py" \
        --input "$input_file" \
        --output "$output_dir" \
        --log-level INFO
    
    log "Processing completed for: $input_file"
}

# Function to run SAR analysis
run_sar_analysis() {
    local data_dir="$1"
    local output_file="$2"
    
    log "Running SAR analysis on: $data_dir"
    
    python3 "$PROJECT_ROOT/src/backend/routers/sar.py" \
        --data-dir "$data_dir" \
        --output "$output_file" \
        --format json
    
    log "SAR analysis completed: $output_file"
}

# Function to run geolocation analysis
run_geolocation() {
    local input_data="$1"
    local output_file="$2"
    
    log "Running geolocation analysis..."
    
    python3 "$PROJECT_ROOT/tools/geolocate.py" \
        --input "$input_data" \
        --output "$output_file"
    
    log "Geolocation analysis completed: $output_file"
}

# Main ingestion pipeline
run_pipeline() {
    local input_source="$1"
    local pipeline_type="${2:-full}"
    
    log "Starting ingestion pipeline: $pipeline_type"
    log "Input source: $input_source"
    
    # Create timestamped output directory
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local output_dir="$DATA_DIR/processed_$timestamp"
    mkdir -p "$output_dir"
    
    case "$pipeline_type" in
        "phone")
            process_phone_stream "$input_source" "$output_dir"
            ;;
        "sar")
            run_sar_analysis "$input_source" "$output_dir/sar_results.json"
            ;;
        "geo")
            run_geolocation "$input_source" "$output_dir/geolocation.json"
            ;;
        "full")
            process_phone_stream "$input_source" "$output_dir"
            run_sar_analysis "$output_dir" "$output_dir/sar_results.json"
            run_geolocation "$output_dir" "$output_dir/geolocation.json"
            ;;
        *)
            error_exit "Unknown pipeline type: $pipeline_type"
            ;;
    esac
    
    log "Pipeline completed successfully. Output: $output_dir"
    echo "$output_dir"
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] <input_source>"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE    Pipeline type: phone, sar, geo, full (default: full)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/phone_stream.mp4"
    echo "  $0 -t phone /path/to/stream.mp4"
    echo "  $0 -t sar /path/to/data_directory"
    echo "  $0 -t geo /path/to/input_data.json"
}

# Parse command line arguments
PIPELINE_TYPE="full"
INPUT_SOURCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            PIPELINE_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            error_exit "Unknown option: $1"
            ;;
        *)
            if [ -z "$INPUT_SOURCE" ]; then
                INPUT_SOURCE="$1"
            else
                error_exit "Multiple input sources specified"
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$INPUT_SOURCE" ]; then
    error_exit "Input source is required"
fi

# Run the pipeline
log "Foresight Ingestion Runner started"
run_pipeline "$INPUT_SOURCE" "$PIPELINE_TYPE"
log "Foresight Ingestion Runner completed"