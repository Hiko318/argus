# Evidence Packaging System

The Evidence Packaging System provides legally compliant evidence collection, packaging, and verification for the Foresight SAR System. It ensures chain of custody, digital integrity, and legal admissibility of evidence collected during search and rescue operations.

## Overview

This system creates tamper-evident evidence packages with:

- **Digital Signatures**: HashiCorp Vault Transit Engine integration
- **Blockchain Timestamps**: OpenTimestamps (OTS) for immutable timestamping
- **Chain of Custody**: Comprehensive documentation and tracking
- **Integrity Verification**: Multiple hash algorithms and validation
- **Legal Compliance**: Structured for legal admissibility
- **Metadata Preservation**: Complete operational context

## Components

### 1. `evidence_packager.py`

Core evidence packaging functionality with:

- Evidence metadata management
- Digital signature creation and verification
- OpenTimestamps integration
- Chain of custody documentation
- Video processing with timestamp overlays
- Comprehensive integrity checking

### 2. `create_evidence_package.py`

Command-line interface and enhanced features:

- HashiCorp Vault Transit Engine integration
- Batch processing capabilities
- Configuration management
- Enhanced security features
- Verification utilities

## Features

### Core Capabilities

- **Multi-format Evidence**: Video, images, data files, metadata
- **Digital Signatures**: RSA-PSS and Vault Transit signatures
- **Blockchain Timestamps**: OpenTimestamps integration
- **Hash Verification**: SHA256, SHA1, MD5 integrity checks
- **Chain of Custody**: Complete audit trail
- **Legal Metadata**: Comprehensive case information

### Security Features

- **Tamper Detection**: Multiple integrity verification methods
- **Non-repudiation**: Digital signatures with timestamp proofs
- **Immutable Timestamps**: Blockchain-based time verification
- **Secure Storage**: Encrypted signature storage
- **Access Control**: Integration with enterprise security systems

### Legal Compliance

- **Evidence Standards**: Meets legal evidence requirements
- **Chain of Custody**: Complete documentation trail
- **Retention Management**: Configurable retention periods
- **Audit Trail**: Comprehensive logging and tracking
- **Classification Support**: Evidence classification levels

## Quick Start

### Basic Usage

```bash
# Create evidence package with video
python create_evidence_package.py \
    --case-id SAR-2024-001 \
    --operator OFFICER-123 \
    --location "Mountain Trail Area" \
    --video stream.mp4

# Create package with multiple evidence types
python create_evidence_package.py \
    --case-id SAR-2024-002 \
    --operator OFFICER-456 \
    --location "Forest Search Zone" \
    --video aerial_footage.mp4 \
    --image suspect_photo.jpg \
    --data gps_tracks.json \
    --equipment "DJI Mavic 3, Thermal Camera"
```

### With HashiCorp Vault Integration

```bash
# Create package with Vault signatures
python create_evidence_package.py \
    --case-id SAR-2024-003 \
    --operator OFFICER-789 \
    --location "Urban Search Area" \
    --video evidence.mp4 \
    --vault-url https://vault.agency.gov:8200 \
    --vault-token $VAULT_TOKEN \
    --vault-key evidence-signing-key
```

### Configuration File

```bash
# Create default configuration
python create_evidence_package.py --create-config

# Use configuration file
python create_evidence_package.py \
    --config evidence_config.json \
    --case-id SAR-2024-004 \
    --operator OFFICER-001 \
    --location "Coastal Search" \
    --video rescue_footage.mp4
```

## Configuration

### Evidence Configuration

```json
{
  "vault": {
    "url": "https://vault.example.com:8200",
    "token": "your-vault-token",
    "key_name": "evidence-signing-key"
  },
  "evidence": {
    "agency": "Search and Rescue Division",
    "default_operator": "OFFICER-001",
    "retention_period_years": 7,
    "evidence_classification": "UNCLASSIFIED"
  },
  "output": {
    "base_directory": "./evidence_packages",
    "include_timestamps": true,
    "compress_package": false
  }
}
```

### Vault Transit Engine Setup

```bash
# Enable Transit engine
vault auth -method=userpass username=evidence-user
vault secrets enable transit

# Create signing key
vault write -f transit/keys/evidence-signing-key type=rsa-4096

# Create policy
vault policy write evidence-signing - <<EOF
path "transit/sign/evidence-signing-key" {
  capabilities = ["update"]
}
path "transit/verify/evidence-signing-key" {
  capabilities = ["update"]
}
EOF
```

## Evidence Package Structure

```
evidence_SAR-2024-001_1234567890/
├── metadata.json                 # Case and operational metadata
├── manifest.json                 # File integrity manifest
├── manifest.sha256              # SHA256 checksums
├── package_summary.json         # Package overview
├── chain_of_custody.json        # Chain of custody documentation
├── security_manifest.json       # Security features summary
├── metadata.json.sig            # Digital signature (if enabled)
├── metadata.json.vault.sig      # Vault signature (if enabled)
├── metadata.json.ots.json       # OpenTimestamps proof
├── manifest.sha256.vault.sig    # Vault signature for manifest
├── manifest.sha256.ots.json     # OpenTimestamps for manifest
├── video_evidence/
│   └── evidence_video.mp4       # Processed video with timestamps
├── image_evidence/
│   ├── evidence_001_photo.jpg   # Evidence images
│   └── evidence_002_thermal.jpg
└── data_evidence/
    ├── data_001_gps_tracks.json # Data files
    └── data_002_sensor_data.csv
```

## Metadata Schema

### Evidence Metadata

```json
{
  "case_id": "SAR-2024-001",
  "operator_id": "OFFICER-123",
  "agency": "Search and Rescue Division",
  "incident_type": "Missing Person",
  "location": "Mountain Trail Area",
  "start_time": "2024-01-15T10:30:00Z",
  "end_time": "2024-01-15T14:45:00Z",
  "equipment_used": "DJI Mavic 3, Thermal Camera",
  "weather_conditions": "Clear, 10mph winds",
  "legal_authority": "Emergency Response Authorization",
  "chain_of_custody_officer": "SGT. SMITH",
  "evidence_classification": "UNCLASSIFIED",
  "retention_period_years": 7,
  "notes": "Aerial search for missing hiker"
}
```

### Chain of Custody Entry

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "action": "Evidence Collection Started",
  "officer_id": "OFFICER-123",
  "officer_name": "John Doe",
  "location": "Command Center",
  "notes": "Initiated aerial search and evidence collection",
  "signature": "digital_signature_hash"
}
```

## Digital Signatures

### RSA-PSS Signatures

Standard RSA-PSS signatures with SHA-256:

```python
from packaging.evidence_packager import EvidencePackager

packager = EvidencePackager(
    private_key_path="/path/to/private_key.pem",
    certificate_path="/path/to/certificate.pem"
)
```

### Vault Transit Signatures

HashiCorp Vault Transit Engine signatures:

```python
from packaging.create_evidence_package import VaultTransitSigner

signer = VaultTransitSigner(
    vault_url="https://vault.example.com:8200",
    vault_token="your-token",
    transit_key="evidence-signing-key"
)

signature = signer.sign_data(data, context="case_context")
```

## OpenTimestamps Integration

### Timestamp Creation

```python
from packaging.create_evidence_package import OpenTimestampsIntegration

ots = OpenTimestampsIntegration()
timestamp_proof = ots.create_timestamp(file_hash)
```

### Verification

```bash
# Install OpenTimestamps client
pip install opentimestamps-client

# Verify timestamp
ots verify evidence_file.mp4.ots

# Upgrade timestamp (after Bitcoin confirmation)
ots upgrade evidence_file.mp4.ots
```

## Verification

### Package Verification

```bash
# Verify evidence package
python create_evidence_package.py --verify /path/to/evidence_package

# Verify with Vault integration
python create_evidence_package.py \
    --verify /path/to/evidence_package \
    --config evidence_config.json
```

### Manual Verification

```bash
# Verify SHA256 checksums
sha256sum -c manifest.sha256

# Verify individual files
sha256sum evidence_video.mp4
```

### Programmatic Verification

```python
from packaging.evidence_packager import EvidencePackager

packager = EvidencePackager()
result = packager.verify_evidence_package("/path/to/package")

if result["success"]:
    print(f"Package verified with integrity score: {result['integrity_score']}")
else:
    print(f"Verification failed: {result['errors']}")
```

## Integration with Foresight

### Automatic Evidence Collection

```python
# In main SAR pipeline
from packaging.create_evidence_package import EnhancedEvidencePackager

# Create evidence package after operation
packager = EnhancedEvidencePackager(vault_config=vault_config)

result = packager.create_enhanced_evidence_package(
    metadata=operation_metadata,
    video_files=["stream_recording.mp4"],
    image_files=suspect_images,
    data_files=["detection_log.json", "gps_track.json"]
)
```

### UI Integration

```javascript
// In SAR interface
function createEvidencePackage() {
    const evidenceData = {
        case_id: getCurrentCaseId(),
        operator: getCurrentOperator(),
        location: getCurrentLocation(),
        video_files: getRecordedVideos(),
        image_files: getCapturedImages()
    };
    
    fetch('/api/create-evidence-package', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(evidenceData)
    });
}
```

## Legal Considerations

### Evidence Standards

- **Authenticity**: Digital signatures ensure evidence authenticity
- **Integrity**: Hash verification prevents tampering
- **Chain of Custody**: Complete documentation trail
- **Admissibility**: Structured for legal proceedings

### Compliance Features

- **FIPS 140-2**: Cryptographic standards compliance
- **RFC 3161**: Timestamp authority standards
- **ISO 27037**: Digital evidence handling guidelines
- **NIST SP 800-86**: Digital forensics integration

### Retention and Disposal

```python
# Set retention period
metadata = EvidenceMetadata(
    retention_period_years=7,  # Legal requirement
    evidence_classification="UNCLASSIFIED"
)

# Automated disposal (implement as needed)
def check_retention_expiry(package_path):
    # Check if retention period has expired
    # Implement secure disposal procedures
    pass
```

## Performance Optimization

### Large File Handling

```python
# Process large video files efficiently
packager = EvidencePackager()

# Use streaming for large files
result = packager.create_evidence_package(
    video_files=["large_video.mp4"],
    # Process in chunks to manage memory
)
```

### Batch Processing

```bash
# Process multiple cases
for case_dir in cases/*/; do
    python create_evidence_package.py \
        --config batch_config.json \
        --case-id $(basename "$case_dir") \
        --video "$case_dir"/*.mp4 \
        --output "evidence_packages/"
done
```

## Security Best Practices

### Key Management

- Store private keys securely (HSM, Vault)
- Use strong key generation (RSA-4096, ECDSA P-384)
- Implement key rotation policies
- Maintain key backup and recovery procedures

### Access Control

- Implement role-based access control
- Log all evidence package operations
- Use multi-factor authentication
- Maintain audit trails

### Network Security

- Use TLS for Vault communications
- Implement certificate validation
- Use VPN for remote operations
- Monitor network traffic

## Troubleshooting

### Common Issues

1. **Vault Connection Failures**
   ```bash
   # Check Vault connectivity
   curl -k $VAULT_ADDR/v1/sys/health
   
   # Verify token
   vault auth -method=userpass username=evidence-user
   ```

2. **OpenTimestamps Failures**
   ```bash
   # Check OTS calendar servers
   curl -I https://alice.btc.calendar.opentimestamps.org
   
   # Install OTS client
   pip install opentimestamps-client
   ```

3. **Video Processing Issues**
   ```bash
   # Install OpenCV
   pip install opencv-python
   
   # Check video codec support
   python -c "import cv2; print(cv2.getBuildInformation())"
   ```

### Debug Mode

```bash
# Enable verbose logging
python create_evidence_package.py \
    --verbose \
    --case-id DEBUG-001 \
    --operator DEBUG-USER \
    --location "Test Location" \
    --video test_video.mp4
```

### Validation

```python
# Test evidence package creation
from packaging.evidence_packager import create_sample_evidence_data

sample_data = create_sample_evidence_data()
packager = EvidencePackager()

result = packager.create_evidence_package(
    metadata=sample_data["metadata"],
    chain_entries=sample_data["chain_entries"]
)

print(f"Test package created: {result['success']}")
```

## API Reference

### EvidencePackager Class

```python
class EvidencePackager:
    def __init__(self, private_key_path=None, certificate_path=None)
    def create_evidence_package(self, metadata, video_files=None, ...)
    def verify_evidence_package(self, package_path)
    def _create_manifest(self, package_dir)
    def _sign_data(self, data, private_key_path=None)
    def _create_opentimestamp(self, file_path)
```

### VaultTransitSigner Class

```python
class VaultTransitSigner:
    def __init__(self, vault_url, vault_token, transit_key)
    def sign_data(self, data, context=None)
    def verify_signature(self, data, signature, context=None)
```

### Data Classes

```python
@dataclass
class EvidenceMetadata:
    case_id: str
    operator_id: str
    agency: str
    # ... additional fields

@dataclass
class ChainOfCustodyEntry:
    timestamp: str
    action: str
    officer_id: str
    # ... additional fields
```

## Contributing

When contributing to the evidence packaging system:

1. Follow legal evidence handling standards
2. Maintain cryptographic security
3. Add comprehensive tests
4. Update documentation
5. Consider legal implications

## License

This evidence packaging system is part of the Foresight SAR System and follows the same licensing terms.

## Support

For issues with evidence packaging:

1. Check the troubleshooting section
2. Verify system requirements
3. Test with sample data
4. Review security configurations
5. Contact system administrators for Vault/OTS issues

## Legal Disclaimer

This system is designed to meet common legal evidence standards, but users should:

- Consult with legal counsel for specific requirements
- Verify compliance with local laws and regulations
- Implement additional security measures as needed
- Maintain proper training and procedures
- Regular audit and validation of evidence packages