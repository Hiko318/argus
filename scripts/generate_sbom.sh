#!/bin/bash

# Software Bill of Materials (SBOM) Generator for Foresight
# This script generates a comprehensive SBOM for the project

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/sbom"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Generate Python dependencies SBOM
generate_python_sbom() {
    log "Generating Python dependencies SBOM..."
    
    local output_file="$OUTPUT_DIR/python_dependencies_$TIMESTAMP.json"
    
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        # If pip-audit is available, use it for security info
        if command_exists pip-audit; then
            log "Using pip-audit for enhanced Python SBOM..."
            pip-audit --format=json --output="$output_file" -r "$PROJECT_ROOT/requirements.txt"
        else
            # Fallback to pip freeze
            log "Using pip freeze for Python dependencies..."
            {
                echo "{"
                echo "  \"metadata\": {"
                echo "    \"timestamp\": \"$(date -Iseconds)\","
                echo "    \"generator\": \"foresight-sbom-generator\","
                echo "    \"project\": \"foresight\""
                echo "  },"
                echo "  \"dependencies\": ["
                
                # Activate virtual environment if it exists
                if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
                    source "$PROJECT_ROOT/venv/bin/activate"
                fi
                
                pip freeze | while IFS= read -r line; do
                    if [ -n "$line" ]; then
                        package=$(echo "$line" | cut -d'=' -f1)
                        version=$(echo "$line" | cut -d'=' -f3)
                        echo "    {"
                        echo "      \"name\": \"$package\","
                        echo "      \"version\": \"$version\","
                        echo "      \"type\": \"python-package\""
                        echo "    },"
                    fi
                done | sed '$ s/,$//'  # Remove trailing comma
                
                echo "  ]"
                echo "}"
            } > "$output_file"
        fi
    else
        log "Warning: requirements.txt not found"
    fi
    
    log "Python SBOM generated: $output_file"
}

# Generate Node.js dependencies SBOM
generate_nodejs_sbom() {
    log "Generating Node.js dependencies SBOM..."
    
    local frontend_dir="$PROJECT_ROOT/src/frontend"
    local output_file="$OUTPUT_DIR/nodejs_dependencies_$TIMESTAMP.json"
    
    if [ -f "$frontend_dir/package.json" ]; then
        cd "$frontend_dir"
        
        if command_exists npm; then
            # Use npm audit for security information
            if npm audit --json > "$output_file" 2>/dev/null; then
                log "Node.js SBOM with security audit generated"
            else
                # Fallback to npm list
                npm list --json > "$output_file" 2>/dev/null || true
                log "Node.js SBOM generated (without security audit)"
            fi
        else
            log "Warning: npm not found, skipping Node.js dependencies"
        fi
        
        cd "$PROJECT_ROOT"
    else
        log "No package.json found, skipping Node.js dependencies"
    fi
    
    log "Node.js SBOM generated: $output_file"
}

# Generate system dependencies SBOM
generate_system_sbom() {
    log "Generating system dependencies SBOM..."
    
    local output_file="$OUTPUT_DIR/system_dependencies_$TIMESTAMP.json"
    
    {
        echo "{"
        echo "  \"metadata\": {"
        echo "    \"timestamp\": \"$(date -Iseconds)\","
        echo "    \"generator\": \"foresight-sbom-generator\","
        echo "    \"project\": \"foresight\","
        echo "    \"system\": \"$(uname -a)\""
        echo "  },"
        echo "  \"system_tools\": ["
        
        # Check for common tools
        tools=("python3" "node" "npm" "git" "docker" "kubectl" "ffmpeg" "adb")
        first=true
        
        for tool in "${tools[@]}"; do
            if command_exists "$tool"; then
                if [ "$first" = true ]; then
                    first=false
                else
                    echo ","
                fi
                
                version=$("$tool" --version 2>/dev/null | head -n1 || echo "unknown")
                echo "    {"
                echo "      \"name\": \"$tool\","
                echo "      \"version\": \"$version\","
                echo "      \"type\": \"system-tool\""
                echo -n "    }"
            fi
        done
        
        echo ""
        echo "  ]"
        echo "}"
    } > "$output_file"
    
    log "System SBOM generated: $output_file"
}

# Generate Docker dependencies SBOM
generate_docker_sbom() {
    log "Generating Docker dependencies SBOM..."
    
    local output_file="$OUTPUT_DIR/docker_dependencies_$TIMESTAMP.json"
    
    if [ -f "$PROJECT_ROOT/Dockerfile" ] || [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
        {
            echo "{"
            echo "  \"metadata\": {"
            echo "    \"timestamp\": \"$(date -Iseconds)\","
            echo "    \"generator\": \"foresight-sbom-generator\","
            echo "    \"project\": \"foresight\""
            echo "  },"
            echo "  \"docker_images\": ["
            
            # Extract base images from Dockerfile
            if [ -f "$PROJECT_ROOT/Dockerfile" ]; then
                grep -i "^FROM" "$PROJECT_ROOT/Dockerfile" | while read -r line; do
                    image=$(echo "$line" | awk '{print $2}')
                    echo "    {"
                    echo "      \"image\": \"$image\","
                    echo "      \"type\": \"base-image\""
                    echo "    },"
                done | sed '$ s/,$//'  # Remove trailing comma
            fi
            
            echo "  ]"
            echo "}"
        } > "$output_file"
        
        log "Docker SBOM generated: $output_file"
    else
        log "No Docker files found, skipping Docker SBOM"
    fi
}

# Generate consolidated SBOM
generate_consolidated_sbom() {
    log "Generating consolidated SBOM..."
    
    local output_file="$OUTPUT_DIR/foresight_sbom_$TIMESTAMP.json"
    
    {
        echo "{"
        echo "  \"bomFormat\": \"CycloneDX\","
        echo "  \"specVersion\": \"1.4\","
        echo "  \"serialNumber\": \"urn:uuid:$(uuidgen 2>/dev/null || echo 'generated-uuid')\","
        echo "  \"version\": 1,"
        echo "  \"metadata\": {"
        echo "    \"timestamp\": \"$(date -Iseconds)\","
        echo "    \"tools\": ["
        echo "      {"
        echo "        \"vendor\": \"Foresight\","
        echo "        \"name\": \"sbom-generator\","
        echo "        \"version\": \"1.0.0\""
        echo "      }"
        echo "    ],"
        echo "    \"component\": {"
        echo "      \"type\": \"application\","
        echo "      \"name\": \"foresight\","
        echo "      \"version\": \"1.0.0\""
        echo "    }"
        echo "  },"
        echo "  \"components\": [],"
        echo "  \"dependencies\": []"
        echo "}"
    } > "$output_file"
    
    log "Consolidated SBOM generated: $output_file"
}

# Generate license information
generate_license_info() {
    log "Generating license information..."
    
    local output_file="$OUTPUT_DIR/licenses_$TIMESTAMP.txt"
    
    {
        echo "# License Information for Foresight Project"
        echo "Generated on: $(date)"
        echo ""
        
        # Check for common license files
        for license_file in LICENSE LICENSE.txt LICENSE.md COPYING; do
            if [ -f "$PROJECT_ROOT/$license_file" ]; then
                echo "## Project License ($license_file)"
                cat "$PROJECT_ROOT/$license_file"
                echo ""
            fi
        done
        
        # Python package licenses (if pip-licenses is available)
        if command_exists pip-licenses; then
            echo "## Python Package Licenses"
            pip-licenses --format=plain
            echo ""
        fi
        
    } > "$output_file"
    
    log "License information generated: $output_file"
}

# Main function
main() {
    log "Starting SBOM generation for Foresight project..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Generate different types of SBOMs
    generate_python_sbom
    generate_nodejs_sbom
    generate_system_sbom
    generate_docker_sbom
    generate_consolidated_sbom
    generate_license_info
    
    # Create summary
    local summary_file="$OUTPUT_DIR/sbom_summary_$TIMESTAMP.txt"
    {
        echo "SBOM Generation Summary"
        echo "======================"
        echo "Generated on: $(date)"
        echo "Project: Foresight"
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        echo "Generated files:"
        ls -la "$OUTPUT_DIR"/*"$TIMESTAMP"*
    } > "$summary_file"
    
    log "SBOM generation completed successfully!"
    log "Output directory: $OUTPUT_DIR"
    log "Summary: $summary_file"
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help         Show this help message"
    echo "  -o, --output DIR   Output directory (default: ./sbom)"
    echo ""
    echo "This script generates a Software Bill of Materials (SBOM) for the Foresight project."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            echo "Unexpected argument: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main