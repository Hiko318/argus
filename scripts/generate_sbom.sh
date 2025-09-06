#!/bin/bash

# Software Bill of Materials (SBOM) Generation Script
# Foresight SAR System - Legal Compliance and Supply Chain Security

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/sbom"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
VERSION="$(git describe --tags --always --dirty 2>/dev/null || echo 'unknown')"

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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install required tools
install_tools() {
    log_info "Installing SBOM generation tools..."
    
    # Install Python tools
    if command_exists pip; then
        pip install --upgrade cyclonedx-bom pip-audit sbom-tool
        log_success "Python SBOM tools installed"
    else
        log_warning "pip not found, skipping Python tools"
    fi
    
    # Install Syft
    if ! command_exists syft; then
        log_info "Installing Syft..."
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
        log_success "Syft installed"
    else
        log_info "Syft already installed"
    fi
    
    # Install SPDX tools
    if command_exists npm; then
        npm install -g @microsoft/sbom-tool
        log_success "Microsoft SBOM tool installed"
    else
        log_warning "npm not found, skipping Microsoft SBOM tool"
    fi
}

# Create output directory
setup_output_dir() {
    log_info "Setting up output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    # Create subdirectories
    mkdir -p "$OUTPUT_DIR/python"
    mkdir -p "$OUTPUT_DIR/javascript"
    mkdir -p "$OUTPUT_DIR/docker"
    mkdir -p "$OUTPUT_DIR/comprehensive"
    mkdir -p "$OUTPUT_DIR/reports"
    
    log_success "Output directory structure created"
}

# Generate Python SBOM
generate_python_sbom() {
    log_info "Generating Python SBOM..."
    
    cd "$PROJECT_ROOT"
    
    # CycloneDX format
    if command_exists cyclonedx-py && [ -f "requirements.txt" ]; then
        log_info "Generating CycloneDX SBOM for Python dependencies..."
        cyclonedx-py -o "$OUTPUT_DIR/python/sbom-python-cyclonedx.json" --format json
        cyclonedx-py -o "$OUTPUT_DIR/python/sbom-python-cyclonedx.xml" --format xml
        log_success "CycloneDX Python SBOM generated"
    fi
    
    # pip-audit SBOM
    if command_exists pip-audit && [ -f "requirements.txt" ]; then
        log_info "Generating pip-audit SBOM..."
        pip-audit --format=cyclonedx-json --output="$OUTPUT_DIR/python/sbom-python-audit.json" || true
        pip-audit --format=cyclonedx-xml --output="$OUTPUT_DIR/python/sbom-python-audit.xml" || true
        log_success "pip-audit SBOM generated"
    fi
    
    # Generate requirements freeze
    if command_exists pip; then
        log_info "Generating pip freeze output..."
        pip freeze > "$OUTPUT_DIR/python/requirements-freeze.txt"
        log_success "Requirements freeze generated"
    fi
}

# Generate JavaScript SBOM
generate_javascript_sbom() {
    log_info "Generating JavaScript SBOM..."
    
    cd "$PROJECT_ROOT"
    
    if [ -f "package.json" ]; then
        # NPM SBOM
        if command_exists npm; then
            log_info "Generating npm SBOM..."
            npm list --json > "$OUTPUT_DIR/javascript/npm-list.json" 2>/dev/null || true
            npm list --prod --json > "$OUTPUT_DIR/javascript/npm-list-prod.json" 2>/dev/null || true
            log_success "npm SBOM generated"
        fi
        
        # License information
        if command_exists npx; then
            log_info "Generating JavaScript license information..."
            npx license-checker --json > "$OUTPUT_DIR/javascript/licenses.json" 2>/dev/null || true
            npx license-checker --csv > "$OUTPUT_DIR/javascript/licenses.csv" 2>/dev/null || true
            log_success "JavaScript license information generated"
        fi
    else
        log_warning "No package.json found, skipping JavaScript SBOM"
    fi
}

# Generate Docker SBOM
generate_docker_sbom() {
    log_info "Generating Docker SBOM..."
    
    cd "$PROJECT_ROOT"
    
    # Find Dockerfiles
    DOCKERFILES=()
    if [ -f "Dockerfile" ]; then
        DOCKERFILES+=("Dockerfile")
    fi
    if [ -f "deploy/jetson/Dockerfile" ]; then
        DOCKERFILES+=("deploy/jetson/Dockerfile")
    fi
    
    if [ ${#DOCKERFILES[@]} -eq 0 ]; then
        log_warning "No Dockerfiles found, skipping Docker SBOM"
        return
    fi
    
    for dockerfile in "${DOCKERFILES[@]}"; do
        log_info "Processing $dockerfile..."
        
        # Build image for SBOM generation
        image_name="foresight-sbom:$(basename "$dockerfile" .Dockerfile)"
        docker build -f "$dockerfile" -t "$image_name" . || {
            log_warning "Failed to build image from $dockerfile"
            continue
        }
        
        # Generate SBOM with Syft
        if command_exists syft; then
            log_info "Generating Syft SBOM for $dockerfile..."
            syft "$image_name" -o spdx-json="$OUTPUT_DIR/docker/sbom-$(basename "$dockerfile" .Dockerfile)-spdx.json"
            syft "$image_name" -o cyclonedx-json="$OUTPUT_DIR/docker/sbom-$(basename "$dockerfile" .Dockerfile)-cyclonedx.json"
            log_success "Syft SBOM generated for $dockerfile"
        fi
        
        # Clean up image
        docker rmi "$image_name" >/dev/null 2>&1 || true
    done
}

# Generate comprehensive SBOM
generate_comprehensive_sbom() {
    log_info "Generating comprehensive project SBOM..."
    
    cd "$PROJECT_ROOT"
    
    # Syft comprehensive scan
    if command_exists syft; then
        log_info "Running Syft comprehensive scan..."
        syft . -o spdx-json="$OUTPUT_DIR/comprehensive/sbom-project-spdx.json"
        syft . -o cyclonedx-json="$OUTPUT_DIR/comprehensive/sbom-project-cyclonedx.json"
        syft . -o table="$OUTPUT_DIR/comprehensive/sbom-project-table.txt"
        log_success "Syft comprehensive SBOM generated"
    fi
    
    # Microsoft SBOM tool
    if command_exists sbom-tool; then
        log_info "Running Microsoft SBOM tool..."
        sbom-tool generate -b "$OUTPUT_DIR/comprehensive" -bc "$PROJECT_ROOT" -pn "Foresight-SAR" -pv "$VERSION" -ps "Foresight" -nsb "https://foresight-sar.com" || {
            log_warning "Microsoft SBOM tool failed"
        }
        log_success "Microsoft SBOM generated"
    fi
}

# Generate vulnerability report
generate_vulnerability_report() {
    log_info "Generating vulnerability report..."
    
    cd "$PROJECT_ROOT"
    
    # Python vulnerabilities
    if command_exists safety && [ -f "requirements.txt" ]; then
        log_info "Scanning Python dependencies for vulnerabilities..."
        safety check --json --output "$OUTPUT_DIR/reports/python-vulnerabilities.json" || true
        safety check --short-report > "$OUTPUT_DIR/reports/python-vulnerabilities.txt" || true
    fi
    
    # pip-audit vulnerabilities
    if command_exists pip-audit && [ -f "requirements.txt" ]; then
        log_info "Running pip-audit vulnerability scan..."
        pip-audit --format=json --output="$OUTPUT_DIR/reports/pip-audit-vulnerabilities.json" || true
        pip-audit > "$OUTPUT_DIR/reports/pip-audit-vulnerabilities.txt" || true
    fi
    
    # JavaScript vulnerabilities
    if [ -f "package.json" ] && command_exists npm; then
        log_info "Scanning JavaScript dependencies for vulnerabilities..."
        npm audit --json > "$OUTPUT_DIR/reports/npm-vulnerabilities.json" 2>/dev/null || true
        npm audit > "$OUTPUT_DIR/reports/npm-vulnerabilities.txt" 2>/dev/null || true
    fi
    
    log_success "Vulnerability reports generated"
}

# Generate metadata
generate_metadata() {
    log_info "Generating SBOM metadata..."
    
    cat > "$OUTPUT_DIR/metadata.json" << EOF
{
  "project": {
    "name": "Foresight SAR System",
    "version": "$VERSION",
    "description": "AI-powered Search and Rescue System",
    "repository": "$(git remote get-url origin 2>/dev/null || echo 'unknown')",
    "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
  },
  "generation": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "generator": "generate_sbom.sh",
    "version": "1.0",
    "hostname": "$(hostname)",
    "user": "$(whoami)"
  },
  "tools": {
    "syft": "$(syft version 2>/dev/null | head -n1 || echo 'not installed')",
    "cyclonedx-py": "$(cyclonedx-py --version 2>/dev/null || echo 'not installed')",
    "pip-audit": "$(pip-audit --version 2>/dev/null || echo 'not installed')",
    "safety": "$(safety --version 2>/dev/null || echo 'not installed')"
  },
  "files": {
    "python": [
      "sbom-python-cyclonedx.json",
      "sbom-python-cyclonedx.xml",
      "sbom-python-audit.json",
      "requirements-freeze.txt"
    ],
    "javascript": [
      "npm-list.json",
      "npm-list-prod.json",
      "licenses.json",
      "licenses.csv"
    ],
    "docker": [
      "sbom-*-spdx.json",
      "sbom-*-cyclonedx.json"
    ],
    "comprehensive": [
      "sbom-project-spdx.json",
      "sbom-project-cyclonedx.json",
      "sbom-project-table.txt"
    ],
    "reports": [
      "python-vulnerabilities.json",
      "pip-audit-vulnerabilities.json",
      "npm-vulnerabilities.json"
    ]
  }
}
EOF

    log_success "Metadata generated"
}

# Create archive
create_archive() {
    log_info "Creating SBOM archive..."
    
    cd "$(dirname "$OUTPUT_DIR")"
    
    ARCHIVE_NAME="foresight-sar-sbom-${VERSION}-${TIMESTAMP}.tar.gz"
    tar -czf "$ARCHIVE_NAME" "$(basename "$OUTPUT_DIR")"
    
    # Generate checksums
    sha256sum "$ARCHIVE_NAME" > "${ARCHIVE_NAME}.sha256"
    
    log_success "Archive created: $ARCHIVE_NAME"
    log_info "SHA256: $(cat "${ARCHIVE_NAME}.sha256")"
}

# Generate summary report
generate_summary() {
    log_info "Generating summary report..."
    
    cat > "$OUTPUT_DIR/SBOM_SUMMARY.md" << EOF
# Software Bill of Materials (SBOM) Summary
## Foresight SAR System

**Generated:** $(date -u +%Y-%m-%dT%H:%M:%SZ)  
**Version:** $VERSION  
**Commit:** $(git rev-parse HEAD 2>/dev/null || echo 'unknown')  

## Overview

This directory contains comprehensive Software Bill of Materials (SBOM) files for the Foresight SAR System, generated for legal compliance, supply chain security, and vulnerability management.

## Directory Structure

- **python/**: Python dependency SBOMs and vulnerability reports
- **javascript/**: JavaScript/Node.js dependency information
- **docker/**: Container image SBOMs
- **comprehensive/**: Project-wide SBOM files
- **reports/**: Vulnerability and security reports

## File Formats

- **SPDX**: Software Package Data Exchange format (ISO/IEC 5962:2021)
- **CycloneDX**: OWASP CycloneDX format
- **JSON/XML**: Machine-readable formats
- **TXT**: Human-readable formats

## Key Files

### Python Dependencies
- \`sbom-python-cyclonedx.json\`: CycloneDX format Python SBOM
- \`sbom-python-audit.json\`: pip-audit generated SBOM with vulnerabilities
- \`requirements-freeze.txt\`: Exact versions of installed packages

### Comprehensive Project SBOM
- \`sbom-project-spdx.json\`: SPDX format project-wide SBOM
- \`sbom-project-cyclonedx.json\`: CycloneDX format project-wide SBOM

### Vulnerability Reports
- \`python-vulnerabilities.json\`: Known vulnerabilities in Python dependencies
- \`pip-audit-vulnerabilities.json\`: pip-audit vulnerability report
- \`npm-vulnerabilities.json\`: JavaScript dependency vulnerabilities

## Usage

### Legal Compliance
- Use SPDX files for license compliance reporting
- Reference vulnerability reports for security assessments
- Include in regulatory submissions and audits

### Supply Chain Security
- Monitor for new vulnerabilities using generated SBOMs
- Track dependency changes over time
- Verify integrity of software components

### Integration
- Import SBOMs into vulnerability management tools
- Use with dependency scanning pipelines
- Include in CI/CD security gates

## Tools Used

- **Syft**: Comprehensive SBOM generation
- **CycloneDX-py**: Python-specific SBOM generation
- **pip-audit**: Python vulnerability scanning
- **Safety**: Python security database
- **npm audit**: JavaScript vulnerability scanning

## Verification

To verify the integrity of this SBOM package:

\`\`\`bash
# Verify archive checksum
sha256sum -c foresight-sar-sbom-${VERSION}-${TIMESTAMP}.tar.gz.sha256

# Validate SPDX files
spdx-tools validate sbom-project-spdx.json

# Validate CycloneDX files
cyclonedx validate --input-file sbom-project-cyclonedx.json
\`\`\`

## Contact

For questions about this SBOM or security concerns:
- Security Team: security@foresight-sar.com
- Legal/Compliance: legal@foresight-sar.com

---

*This SBOM was automatically generated and should be reviewed by qualified personnel before use in legal or compliance contexts.*
EOF

    log_success "Summary report generated"
}

# Main execution
main() {
    log_info "Starting SBOM generation for Foresight SAR System"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Version: $VERSION"
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        log_warning "Not in a git repository, some metadata may be incomplete"
    fi
    
    # Install tools if requested
    if [[ "${1:-}" == "--install-tools" ]]; then
        install_tools
        shift
    fi
    
    # Setup
    setup_output_dir
    
    # Generate SBOMs
    generate_python_sbom
    generate_javascript_sbom
    generate_docker_sbom
    generate_comprehensive_sbom
    
    # Generate reports
    generate_vulnerability_report
    generate_metadata
    generate_summary
    
    # Create archive if requested
    if [[ "${1:-}" == "--archive" ]]; then
        create_archive
    fi
    
    log_success "SBOM generation completed successfully"
    log_info "Output available in: $OUTPUT_DIR"
    
    # Display summary
    echo
    log_info "Generated files:"
    find "$OUTPUT_DIR" -type f | sort | while read -r file; do
        echo "  - $(basename "$file")"
    done
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate Software Bill of Materials (SBOM) for Foresight SAR System

Options:
  --install-tools    Install required SBOM generation tools
  --archive         Create compressed archive of generated SBOMs
  --help            Show this help message

Examples:
  $0                           # Generate SBOMs with existing tools
  $0 --install-tools           # Install tools and generate SBOMs
  $0 --archive                 # Generate SBOMs and create archive
  $0 --install-tools --archive # Install tools, generate SBOMs, and archive

Output:
  SBOMs are generated in: $OUTPUT_DIR

Requirements:
  - Python 3.7+ with pip
  - Node.js with npm (optional, for JavaScript dependencies)
  - Docker (optional, for container SBOMs)
  - Git (for metadata)
  - curl (for tool installation)

EOF
}

# Parse command line arguments
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"