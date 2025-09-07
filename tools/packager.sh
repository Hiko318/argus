#!/bin/bash

# Evidence Packager - Creates tamper-evident evidence packages
# Usage: ./packager.sh <video_file> <metadata_json> [output_dir]

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGER_SCRIPT="$SCRIPT_DIR/evidence_packager.py"
VERIFIER_SCRIPT="$SCRIPT_DIR/verify_evidence.py"

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
Evidence Packager - Creates tamper-evident evidence packages

USAGE:
    $0 <video_file> <metadata_json> [output_dir]
    $0 --verify <evidence_package>
    $0 --help

ARGUMENTS:
    video_file      Path to MP4 video file to package
    metadata_json   Path to metadata JSON file
    output_dir      Output directory (default: ./evidence_packages)

OPTIONS:
    --verify        Verify an existing evidence package
    --help          Show this help message

EXAMPLES:
    # Create evidence package
    $0 mission_001.mp4 metadata.json
    
    # Create evidence package in specific directory
    $0 mission_001.mp4 metadata.json /path/to/output
    
    # Verify evidence package
    $0 --verify evidence_20240115_143022.zip

OUTPUT:
    Creates evidence.zip containing:
    - MP4 video file
    - metadata.json (evidence metadata)
    - metadata.sig (digital signature)
    - manifest.sha256 (file hashes)
    - metadata.ots (OpenTimestamps proof)

EOF
}

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check evidence packager script
    if [[ ! -f "$PACKAGER_SCRIPT" ]]; then
        missing_deps+=("evidence_packager.py")
    fi
    
    # Check if we can import required Python modules
    if ! python3 -c "import cryptography, hashlib, json" &> /dev/null; then
        missing_deps+=("python3-cryptography")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            case $dep in
                "python3")
                    echo "  - Install Python 3: https://python.org/downloads/"
                    ;;
                "evidence_packager.py")
                    echo "  - Ensure evidence_packager.py is in the tools directory"
                    ;;
                "python3-cryptography")
                    echo "  - Install cryptography: pip install cryptography"
                    ;;
            esac
        done
        exit 1
    fi
}

# Validate input files
validate_inputs() {
    local video_file="$1"
    local metadata_file="$2"
    
    # Check video file
    if [[ ! -f "$video_file" ]]; then
        log_error "Video file not found: $video_file"
        exit 1
    fi
    
    # Check file extension
    if [[ "${video_file,,}" != *.mp4 ]]; then
        log_warning "Video file is not MP4 format: $video_file"
    fi
    
    # Check metadata file
    if [[ ! -f "$metadata_file" ]]; then
        log_error "Metadata file not found: $metadata_file"
        exit 1
    fi
    
    # Validate JSON format
    if ! python3 -c "import json; json.load(open('$metadata_file'))" &> /dev/null; then
        log_error "Invalid JSON format in metadata file: $metadata_file"
        exit 1
    fi
    
    log_success "Input validation passed"
}

# Create evidence package
create_package() {
    local video_file="$1"
    local metadata_file="$2"
    local output_dir="$3"
    
    log_info "Creating evidence package..."
    log_info "Video: $video_file"
    log_info "Metadata: $metadata_file"
    log_info "Output: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Generate timestamp for package name
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local package_name="evidence_${timestamp}.zip"
    local package_path="$output_dir/$package_name"
    
    # Run Python packager
    log_info "Running evidence packager..."
    if python3 "$PACKAGER_SCRIPT" "$video_file" "$metadata_file" "$package_path"; then
        log_success "Evidence package created: $package_path"
        
        # Show package info
        local package_size=$(du -h "$package_path" | cut -f1)
        log_info "Package size: $package_size"
        
        # Verify package immediately
        log_info "Verifying package integrity..."
        if [[ -f "$VERIFIER_SCRIPT" ]]; then
            if python3 "$VERIFIER_SCRIPT" "$package_path" --quiet; then
                log_success "Package verification passed"
            else
                log_warning "Package verification failed"
            fi
        else
            log_warning "Verifier script not found, skipping verification"
        fi
        
        echo
        log_success "Evidence package ready: $package_path"
        return 0
    else
        log_error "Failed to create evidence package"
        return 1
    fi
}

# Verify evidence package
verify_package() {
    local package_path="$1"
    
    log_info "Verifying evidence package: $package_path"
    
    if [[ ! -f "$package_path" ]]; then
        log_error "Package file not found: $package_path"
        exit 1
    fi
    
    if [[ ! -f "$VERIFIER_SCRIPT" ]]; then
        log_error "Verifier script not found: $VERIFIER_SCRIPT"
        exit 1
    fi
    
    # Run verification
    if python3 "$VERIFIER_SCRIPT" "$package_path"; then
        log_success "Package verification completed"
        return 0
    else
        log_error "Package verification failed"
        return 1
    fi
}

# Main function
main() {
    # Parse arguments
    case "${1:-}" in
        "--help"|-h)
            show_help
            exit 0
            ;;
        "--verify")
            if [[ -z "${2:-}" ]]; then
                log_error "Missing package path for verification"
                show_help
                exit 1
            fi
            check_dependencies
            verify_package "$2"
            exit $?
            ;;
        "")
            log_error "Missing arguments"
            show_help
            exit 1
            ;;
        *)
            # Create package mode
            if [[ $# -lt 2 ]]; then
                log_error "Missing required arguments"
                show_help
                exit 1
            fi
            
            local video_file="$1"
            local metadata_file="$2"
            local output_dir="${3:-./evidence_packages}"
            
            check_dependencies
            validate_inputs "$video_file" "$metadata_file"
            create_package "$video_file" "$metadata_file" "$output_dir"
            exit $?
            ;;
    esac
}

# Run main function with all arguments
main "$@"