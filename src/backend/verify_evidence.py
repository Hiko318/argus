"""Evidence verification module for SAR operations.

This module provides functionality to verify the integrity and authenticity
of evidence packages created by the canonicalize module.
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from .canonicalize import VaultTransitSigner, LocalDevSigner, CanonicalMetadata

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of evidence verification."""
    is_valid: bool
    evidence_id: str
    verification_time: str
    checks_performed: List[str]
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    errors: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "evidence_id": self.evidence_id,
            "verification_time": self.verification_time,
            "checks_performed": self.checks_performed,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata
        }
    
    def add_check(self, check_name: str, passed: bool, message: str = "") -> None:
        """Add a verification check result."""
        self.checks_performed.append(check_name)
        
        if passed:
            self.checks_passed.append(check_name)
            if message:
                logger.info(f"✓ {check_name}: {message}")
        else:
            self.checks_failed.append(check_name)
            self.errors.append(f"{check_name}: {message}")
            logger.error(f"✗ {check_name}: {message}")
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(f"⚠ {message}")
    
    def finalize(self) -> None:
        """Finalize verification result."""
        self.is_valid = len(self.checks_failed) == 0
        
        if self.is_valid:
            logger.info(f"✓ Evidence {self.evidence_id} verification PASSED")
        else:
            logger.error(f"✗ Evidence {self.evidence_id} verification FAILED")


class EvidenceVerifier:
    """Evidence package verifier."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signer = None
        
        # Initialize signer for verification
        signer_type = config.get('signer_type', 'local_dev')
        
        if signer_type == 'vault_transit':
            vault_config = config.get('vault_transit', {})
            self.signer = VaultTransitSigner(
                vault_url=vault_config.get('url'),
                vault_token=vault_config.get('token'),
                key_name=vault_config.get('key_name')
            )
        elif signer_type == 'local_dev':
            local_config = config.get('local_dev', {})
            self.signer = LocalDevSigner(
                private_key_path=local_config.get('private_key_path'),
                public_key_path=local_config.get('public_key_path')
            )
        else:
            raise ValueError(f"Unknown signer type: {signer_type}")
        
        logger.info(f"Initialized evidence verifier with {signer_type} signer")
    
    def verify_evidence_package(self, package_dir: str) -> VerificationResult:
        """Verify a complete evidence package.
        
        Args:
            package_dir: Directory containing evidence package files
            
        Returns:
            VerificationResult with detailed verification status
        """
        package_path = Path(package_dir)
        
        # Initialize result
        result = VerificationResult(
            is_valid=False,
            evidence_id="unknown",
            verification_time=datetime.utcnow().isoformat(),
            checks_performed=[],
            checks_passed=[],
            checks_failed=[],
            warnings=[],
            errors=[]
        )
        
        try:
            logger.info(f"Starting verification of evidence package: {package_dir}")
            
            # 1. Check package structure
            package_files = self._check_package_structure(package_path, result)
            if not package_files:
                result.finalize()
                return result
            
            # 2. Load and validate metadata
            metadata = self._load_and_validate_metadata(package_files['metadata'], result)
            if not metadata:
                result.finalize()
                return result
            
            result.evidence_id = metadata.get('evidence_id', 'unknown')
            result.metadata = metadata
            
            # 3. Verify file integrity
            self._verify_file_integrity(package_files, result)
            
            # 4. Verify metadata hash
            self._verify_metadata_hash(metadata, result)
            
            # 5. Verify digital signature
            self._verify_digital_signature(package_files['metadata'], package_files['signature'], result)
            
            # 6. Verify manifest
            if 'manifest' in package_files:
                self._verify_manifest(package_files, result)
            
            # 7. Validate metadata content
            self._validate_metadata_content(metadata, result)
            
            # 8. Check chain of custody
            self._verify_chain_of_custody(metadata, result)
            
            # 9. Validate timestamps
            self._validate_timestamps(metadata, result)
            
            # 10. Check for tampering indicators
            self._check_tampering_indicators(package_files, result)
            
            result.finalize()
            return result
            
        except Exception as e:
            result.errors.append(f"Verification failed with exception: {str(e)}")
            logger.error(f"Evidence verification failed: {e}")
            result.finalize()
            return result
    
    def _check_package_structure(self, package_path: Path, result: VerificationResult) -> Optional[Dict[str, str]]:
        """Check if package has required files."""
        required_extensions = ['.mp4', '_metadata.json', '_metadata.json.sig']
        optional_extensions = ['_manifest.sha256', '.ots']
        
        # Find evidence files
        evidence_files = {}
        
        # Look for video file
        video_files = list(package_path.glob('*.mp4'))
        if not video_files:
            result.add_check('package_structure', False, 'No video file (.mp4) found')
            return None
        elif len(video_files) > 1:
            result.add_warning(f'Multiple video files found: {[f.name for f in video_files]}')
        
        video_file = video_files[0]
        base_name = video_file.stem
        evidence_files['video'] = str(video_file)
        
        # Check for metadata file
        metadata_file = package_path / f"{base_name}_metadata.json"
        if not metadata_file.exists():
            result.add_check('package_structure', False, f'Metadata file not found: {metadata_file.name}')
            return None
        evidence_files['metadata'] = str(metadata_file)
        
        # Check for signature file
        signature_file = package_path / f"{base_name}_metadata.json.sig"
        if not signature_file.exists():
            result.add_check('package_structure', False, f'Signature file not found: {signature_file.name}')
            return None
        evidence_files['signature'] = str(signature_file)
        
        # Check for optional files
        manifest_file = package_path / f"{base_name}_manifest.sha256"
        if manifest_file.exists():
            evidence_files['manifest'] = str(manifest_file)
        
        ots_file = package_path / f"{base_name}_metadata.json.ots"
        if ots_file.exists():
            evidence_files['ots'] = str(ots_file)
        
        result.add_check('package_structure', True, f'Found {len(evidence_files)} package files')
        return evidence_files
    
    def _load_and_validate_metadata(self, metadata_path: str, result: VerificationResult) -> Optional[Dict[str, Any]]:
        """Load and validate metadata JSON."""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check required fields
            required_fields = [
                'evidence_id', 'creation_time', 'format_version',
                'video_file', 'video_hash_sha256', 'frames',
                'mission_profile', 'model_version', 'metadata_hash'
            ]
            
            missing_fields = [field for field in required_fields if field not in metadata]
            if missing_fields:
                result.add_check('metadata_structure', False, f'Missing required fields: {missing_fields}')
                return None
            
            # Validate format version
            format_version = metadata.get('format_version')
            if format_version != '1.0':
                result.add_warning(f'Unknown format version: {format_version}')
            
            result.add_check('metadata_structure', True, 'Metadata structure is valid')
            return metadata
            
        except json.JSONDecodeError as e:
            result.add_check('metadata_structure', False, f'Invalid JSON: {str(e)}')
            return None
        except Exception as e:
            result.add_check('metadata_structure', False, f'Failed to load metadata: {str(e)}')
            return None
    
    def _verify_file_integrity(self, package_files: Dict[str, str], result: VerificationResult) -> None:
        """Verify integrity of package files."""
        try:
            # Check if all files exist and are readable
            for file_type, file_path in package_files.items():
                if not Path(file_path).exists():
                    result.add_check('file_integrity', False, f'{file_type} file missing: {file_path}')
                    return
                
                if not Path(file_path).is_file():
                    result.add_check('file_integrity', False, f'{file_type} is not a file: {file_path}')
                    return
                
                # Check file size
                file_size = Path(file_path).stat().st_size
                if file_size == 0:
                    result.add_check('file_integrity', False, f'{file_type} file is empty: {file_path}')
                    return
            
            result.add_check('file_integrity', True, 'All package files are present and readable')
            
        except Exception as e:
            result.add_check('file_integrity', False, f'File integrity check failed: {str(e)}')
    
    def _verify_metadata_hash(self, metadata: Dict[str, Any], result: VerificationResult) -> None:
        """Verify metadata hash."""
        try:
            stored_hash = metadata.get('metadata_hash')
            if not stored_hash:
                result.add_check('metadata_hash', False, 'No metadata hash found')
                return
            
            # Compute canonical hash
            canonical_data = {k: v for k, v in metadata.items() 
                             if k not in ['metadata_hash', 'signature']}
            canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
            computed_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
            
            if stored_hash == computed_hash:
                result.add_check('metadata_hash', True, 'Metadata hash is valid')
            else:
                result.add_check('metadata_hash', False, f'Hash mismatch: stored={stored_hash[:16]}..., computed={computed_hash[:16]}...')
            
        except Exception as e:
            result.add_check('metadata_hash', False, f'Hash verification failed: {str(e)}')
    
    def _verify_digital_signature(self, metadata_path: str, signature_path: str, result: VerificationResult) -> None:
        """Verify digital signature."""
        try:
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # Create canonical representation
            canonical_data = {k: v for k, v in metadata_dict.items() 
                             if k not in ['metadata_hash', 'signature']}
            canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
            
            # Load signature
            with open(signature_path, 'r', encoding='utf-8') as f:
                signature = f.read().strip()
            
            # Verify signature
            is_valid = self.signer.verify_signature(canonical_json, signature)
            
            if is_valid:
                result.add_check('digital_signature', True, 'Digital signature is valid')
            else:
                result.add_check('digital_signature', False, 'Digital signature verification failed')
            
        except Exception as e:
            result.add_check('digital_signature', False, f'Signature verification failed: {str(e)}')
    
    def _verify_manifest(self, package_files: Dict[str, str], result: VerificationResult) -> None:
        """Verify SHA-256 manifest."""
        try:
            manifest_path = package_files['manifest']
            
            # Load manifest
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_content = f.read().strip()
            
            # Parse manifest
            manifest_hashes = {}
            for line in manifest_content.split('\n'):
                if line.strip():
                    parts = line.split('  ', 1)
                    if len(parts) == 2:
                        file_hash, filename = parts
                        manifest_hashes[filename] = file_hash
            
            # Verify each file hash
            verification_failed = False
            for file_type, file_path in package_files.items():
                if file_type == 'manifest':  # Skip manifest itself
                    continue
                
                filename = Path(file_path).name
                if filename not in manifest_hashes:
                    result.add_warning(f'File not in manifest: {filename}')
                    continue
                
                # Compute file hash
                computed_hash = self._compute_file_hash(file_path)
                expected_hash = manifest_hashes[filename]
                
                if computed_hash != expected_hash:
                    result.add_check('manifest_verification', False, 
                                   f'Hash mismatch for {filename}: expected={expected_hash[:16]}..., computed={computed_hash[:16]}...')
                    verification_failed = True
            
            if not verification_failed:
                result.add_check('manifest_verification', True, f'All {len(manifest_hashes)} files verified against manifest')
            
        except Exception as e:
            result.add_check('manifest_verification', False, f'Manifest verification failed: {str(e)}')
    
    def _validate_metadata_content(self, metadata: Dict[str, Any], result: VerificationResult) -> None:
        """Validate metadata content for consistency and completeness."""
        try:
            # Validate evidence ID format
            evidence_id = metadata.get('evidence_id', '')
            if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', evidence_id):
                result.add_warning(f'Evidence ID format may be invalid: {evidence_id}')
            
            # Validate timestamps
            creation_time = metadata.get('creation_time')
            if creation_time:
                try:
                    datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                except ValueError:
                    result.add_warning(f'Invalid creation time format: {creation_time}')
            
            # Validate video hash format
            video_hash = metadata.get('video_hash_sha256', '')
            if not re.match(r'^[0-9a-f]{64}$', video_hash):
                result.add_warning(f'Video hash format may be invalid: {video_hash[:16]}...')
            
            # Validate frames data
            frames = metadata.get('frames', [])
            if not frames:
                result.add_warning('No frame metadata found')
            else:
                # Check frame sequence
                frame_seqs = [frame.get('frame_seq', 0) for frame in frames]
                if frame_seqs != sorted(frame_seqs):
                    result.add_warning('Frame sequence numbers are not in order')
                
                # Check for gaps in sequence
                if frame_seqs and (max(frame_seqs) - min(frame_seqs) + 1) != len(frame_seqs):
                    result.add_warning('Gaps detected in frame sequence')
            
            # Validate mission profile
            mission = metadata.get('mission_profile', {})
            if not mission.get('mission_id'):
                result.add_warning('Missing mission ID')
            if not mission.get('operator_id'):
                result.add_warning('Missing operator ID')
            
            # Validate model version
            model = metadata.get('model_version', {})
            if not model.get('model_name'):
                result.add_warning('Missing model name')
            if not model.get('version'):
                result.add_warning('Missing model version')
            
            result.add_check('metadata_content', True, 'Metadata content validation completed')
            
        except Exception as e:
            result.add_check('metadata_content', False, f'Metadata content validation failed: {str(e)}')
    
    def _verify_chain_of_custody(self, metadata: Dict[str, Any], result: VerificationResult) -> None:
        """Verify chain of custody information."""
        try:
            chain = metadata.get('chain_of_custody', [])
            
            if not chain:
                result.add_warning('No chain of custody information found')
                return
            
            # Validate chain entries
            for i, entry in enumerate(chain):
                required_fields = ['timestamp', 'action', 'operator']
                missing_fields = [field for field in required_fields if field not in entry]
                
                if missing_fields:
                    result.add_warning(f'Chain of custody entry {i} missing fields: {missing_fields}')
                
                # Validate timestamp format
                timestamp = entry.get('timestamp')
                if timestamp:
                    try:
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        result.add_warning(f'Invalid timestamp in chain entry {i}: {timestamp}')
            
            # Check chronological order
            timestamps = []
            for entry in chain:
                timestamp = entry.get('timestamp')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamps.append(dt)
                    except ValueError:
                        pass
            
            if len(timestamps) > 1 and timestamps != sorted(timestamps):
                result.add_warning('Chain of custody entries are not in chronological order')
            
            result.add_check('chain_of_custody', True, f'Chain of custody verified ({len(chain)} entries)')
            
        except Exception as e:
            result.add_check('chain_of_custody', False, f'Chain of custody verification failed: {str(e)}')
    
    def _validate_timestamps(self, metadata: Dict[str, Any], result: VerificationResult) -> None:
        """Validate timestamp consistency."""
        try:
            creation_time_str = metadata.get('creation_time')
            if not creation_time_str:
                result.add_warning('No creation time found')
                return
            
            creation_time = datetime.fromisoformat(creation_time_str.replace('Z', '+00:00'))
            current_time = datetime.utcnow().replace(tzinfo=creation_time.tzinfo)
            
            # Check if creation time is in the future
            if creation_time > current_time:
                result.add_warning(f'Creation time is in the future: {creation_time_str}')
            
            # Check if creation time is too old (more than 1 year)
            time_diff = current_time - creation_time
            if time_diff.days > 365:
                result.add_warning(f'Creation time is very old ({time_diff.days} days ago)')
            
            # Validate frame timestamps
            frames = metadata.get('frames', [])
            frame_times = []
            
            for frame in frames:
                timestamp_str = frame.get('timestamp_utc')
                if timestamp_str:
                    try:
                        frame_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        frame_times.append(frame_time)
                    except ValueError:
                        result.add_warning(f'Invalid frame timestamp: {timestamp_str}')
            
            # Check frame timestamp consistency
            if frame_times:
                if min(frame_times) < creation_time:
                    result.add_warning('Some frame timestamps are before creation time')
                
                if max(frame_times) > current_time:
                    result.add_warning('Some frame timestamps are in the future')
            
            result.add_check('timestamp_validation', True, 'Timestamp validation completed')
            
        except Exception as e:
            result.add_check('timestamp_validation', False, f'Timestamp validation failed: {str(e)}')
    
    def _check_tampering_indicators(self, package_files: Dict[str, str], result: VerificationResult) -> None:
        """Check for potential tampering indicators."""
        try:
            # Check file modification times
            file_times = {}
            for file_type, file_path in package_files.items():
                stat = Path(file_path).stat()
                file_times[file_type] = {
                    'mtime': stat.st_mtime,
                    'ctime': stat.st_ctime,
                    'size': stat.st_size
                }
            
            # Check if metadata file was modified after signature
            if 'metadata' in file_times and 'signature' in file_times:
                metadata_mtime = file_times['metadata']['mtime']
                signature_mtime = file_times['signature']['mtime']
                
                if metadata_mtime > signature_mtime + 1:  # Allow 1 second tolerance
                    result.add_warning('Metadata file appears to be modified after signature creation')
            
            # Check for suspicious file sizes
            if 'metadata' in file_times:
                metadata_size = file_times['metadata']['size']
                if metadata_size < 100:  # Very small metadata file
                    result.add_warning(f'Metadata file is suspiciously small: {metadata_size} bytes')
                elif metadata_size > 10 * 1024 * 1024:  # Very large metadata file
                    result.add_warning(f'Metadata file is suspiciously large: {metadata_size} bytes')
            
            if 'signature' in file_times:
                signature_size = file_times['signature']['size']
                if signature_size < 50:  # Very small signature
                    result.add_warning(f'Signature file is suspiciously small: {signature_size} bytes')
                elif signature_size > 10 * 1024:  # Very large signature
                    result.add_warning(f'Signature file is suspiciously large: {signature_size} bytes')
            
            result.add_check('tampering_indicators', True, 'Tampering indicator check completed')
            
        except Exception as e:
            result.add_check('tampering_indicators', False, f'Tampering indicator check failed: {str(e)}')
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def verify_single_file(self, file_path: str) -> VerificationResult:
        """Verify a single evidence file (metadata + signature).
        
        Args:
            file_path: Path to metadata JSON file
            
        Returns:
            VerificationResult for the single file
        """
        metadata_path = Path(file_path)
        signature_path = metadata_path.with_suffix(metadata_path.suffix + '.sig')
        
        result = VerificationResult(
            is_valid=False,
            evidence_id="unknown",
            verification_time=datetime.utcnow().isoformat(),
            checks_performed=[],
            checks_passed=[],
            checks_failed=[],
            warnings=[],
            errors=[]
        )
        
        try:
            # Check files exist
            if not metadata_path.exists():
                result.add_check('file_exists', False, f'Metadata file not found: {metadata_path}')
                result.finalize()
                return result
            
            if not signature_path.exists():
                result.add_check('file_exists', False, f'Signature file not found: {signature_path}')
                result.finalize()
                return result
            
            result.add_check('file_exists', True, 'Both metadata and signature files found')
            
            # Load and validate metadata
            metadata = self._load_and_validate_metadata(str(metadata_path), result)
            if not metadata:
                result.finalize()
                return result
            
            result.evidence_id = metadata.get('evidence_id', 'unknown')
            result.metadata = metadata
            
            # Verify metadata hash
            self._verify_metadata_hash(metadata, result)
            
            # Verify digital signature
            self._verify_digital_signature(str(metadata_path), str(signature_path), result)
            
            # Validate metadata content
            self._validate_metadata_content(metadata, result)
            
            result.finalize()
            return result
            
        except Exception as e:
            result.errors.append(f"Single file verification failed: {str(e)}")
            result.finalize()
            return result


# Example usage and CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    
    def main():
        parser = argparse.ArgumentParser(description='Verify evidence packages')
        parser.add_argument('path', help='Path to evidence package directory or metadata file')
        parser.add_argument('--config', help='Configuration file path')
        parser.add_argument('--output', help='Output verification report to file')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        args = parser.parse_args()
        
        # Setup logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                'signer_type': 'local_dev',
                'local_dev': {
                    'private_key_path': 'keys/evidence_signing_key.pem',
                    'public_key_path': 'keys/evidence_signing_key.pub'
                }
            }
        
        # Create verifier
        try:
            verifier = EvidenceVerifier(config)
        except Exception as e:
            print(f"Failed to initialize verifier: {e}")
            sys.exit(1)
        
        # Verify evidence
        path = Path(args.path)
        
        if path.is_dir():
            # Verify package directory
            result = verifier.verify_evidence_package(str(path))
        elif path.is_file() and path.suffix == '.json':
            # Verify single metadata file
            result = verifier.verify_single_file(str(path))
        else:
            print(f"Invalid path: {path}")
            sys.exit(1)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Verification report saved to: {args.output}")
        else:
            print(json.dumps(result.to_dict(), indent=2))
        
        # Exit with appropriate code
        sys.exit(0 if result.is_valid else 1)
    
    main()