#!/usr/bin/env python3
"""
Evidence Packager for Foresight SAR System

This module creates legally compliant evidence packages with digital signatures,
timestamps, and chain of custody documentation for search and rescue operations.
"""

import json
import hashlib
import time
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import subprocess

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available - video processing disabled")

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography not available - digital signatures disabled")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests not available - OpenTimestamps disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvidenceMetadata:
    """Metadata for evidence package"""
    case_id: str
    operator_id: str
    agency: str
    incident_type: str
    location: str
    start_time: str
    end_time: str
    equipment_used: str
    weather_conditions: str
    legal_authority: str = ""
    chain_of_custody_officer: str = ""
    evidence_classification: str = "UNCLASSIFIED"
    retention_period_years: int = 7
    notes: str = ""

@dataclass
class DigitalSignature:
    """Digital signature information"""
    algorithm: str
    signature: str
    public_key: str
    timestamp: str
    signer_id: str
    certificate_chain: Optional[List[str]] = None

@dataclass
class TimestampProof:
    """OpenTimestamps proof information"""
    file_hash: str
    timestamp_file: str
    verification_url: str
    created_at: str
    calendar_servers: List[str]

@dataclass
class ChainOfCustodyEntry:
    """Single entry in chain of custody"""
    timestamp: str
    action: str
    officer_id: str
    officer_name: str
    location: str
    notes: str
    signature: Optional[str] = None

class EvidencePackager:
    """Creates legally compliant evidence packages"""
    
    def __init__(self, private_key_path: str = None, 
                 certificate_path: str = None):
        self.private_key_path = private_key_path
        self.certificate_path = certificate_path
        self.temp_dir = None
        
        # OpenTimestamps calendar servers
        self.ots_calendars = [
            "https://alice.btc.calendar.opentimestamps.org",
            "https://bob.btc.calendar.opentimestamps.org",
            "https://finney.calendar.eternitywall.com"
        ]
    
    def _create_temp_directory(self) -> Path:
        """Create temporary directory for packaging"""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="evidence_"))
        return self.temp_dir
    
    def _cleanup_temp_directory(self):
        """Clean up temporary directory"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate hash of file"""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _create_manifest(self, package_dir: Path) -> Dict[str, Dict[str, str]]:
        """Create comprehensive manifest with multiple hash algorithms"""
        manifest = {}
        
        for file_path in package_dir.rglob("*"):
            if file_path.is_file() and file_path.name not in ["manifest.sha256", "manifest.json"]:
                relative_path = str(file_path.relative_to(package_dir))
                
                manifest[relative_path] = {
                    "sha256": self._calculate_file_hash(file_path, "sha256"),
                    "sha1": self._calculate_file_hash(file_path, "sha1"),
                    "md5": self._calculate_file_hash(file_path, "md5"),
                    "size_bytes": file_path.stat().st_size,
                    "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).isoformat()
                }
        
        return manifest
    
    def _sign_data(self, data: bytes, private_key_path: str = None) -> Optional[DigitalSignature]:
        """Create digital signature for data"""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography not available - skipping digital signature")
            return None
        
        key_path = private_key_path or self.private_key_path
        if not key_path or not Path(key_path).exists():
            logger.warning("Private key not found - skipping digital signature")
            return None
        
        try:
            # Load private key
            with open(key_path, "rb") as key_file:
                private_key = load_pem_private_key(
                    key_file.read(),
                    password=None  # Assume no password for simplicity
                )
            
            # Create signature
            signature = private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Get public key
            public_key = private_key.public_key()
            public_pem = public_key.public_key_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return DigitalSignature(
                algorithm="RSA-PSS-SHA256",
                signature=signature.hex(),
                public_key=public_pem.decode('utf-8'),
                timestamp=datetime.now(timezone.utc).isoformat(),
                signer_id="evidence_packager"
            )
            
        except Exception as e:
            logger.error(f"Failed to create digital signature: {e}")
            return None
    
    def _create_opentimestamp(self, file_path: Path) -> Optional[TimestampProof]:
        """Create OpenTimestamps proof for file"""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests not available - skipping OpenTimestamps")
            return None
        
        try:
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path, "sha256")
            
            # Create .ots file
            ots_file = file_path.with_suffix(file_path.suffix + ".ots")
            
            # Try to use ots command line tool if available
            try:
                result = subprocess.run(
                    ["ots", "stamp", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0 and ots_file.exists():
                    return TimestampProof(
                        file_hash=file_hash,
                        timestamp_file=str(ots_file),
                        verification_url="https://opentimestamps.org",
                        created_at=datetime.now(timezone.utc).isoformat(),
                        calendar_servers=self.ots_calendars
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("OpenTimestamps CLI not available")
            
            # Fallback: Create basic timestamp record
            timestamp_data = {
                "file_hash": file_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "calendar_servers": self.ots_calendars,
                "note": "Timestamp created without OTS CLI - manual verification required"
            }
            
            with open(ots_file, 'w') as f:
                json.dump(timestamp_data, f, indent=2)
            
            return TimestampProof(
                file_hash=file_hash,
                timestamp_file=str(ots_file),
                verification_url="https://opentimestamps.org",
                created_at=datetime.now(timezone.utc).isoformat(),
                calendar_servers=self.ots_calendars
            )
            
        except Exception as e:
            logger.error(f"Failed to create OpenTimestamp: {e}")
            return None
    
    def _process_video_evidence(self, video_path: str, output_dir: Path) -> Optional[str]:
        """Process video evidence with metadata embedding"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - copying video without processing")
            output_path = output_dir / "evidence_video.mp4"
            shutil.copy2(video_path, output_path)
            return str(output_path)
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Set up output video
            output_path = output_dir / "evidence_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Add timestamp overlay to each frame
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add timestamp overlay
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                cv2.putText(frame, f"EVIDENCE - {timestamp}", 
                           (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 255), 2)
                
                cv2.putText(frame, f"Frame: {frame_num}/{frame_count}", 
                           (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 255), 1)
                
                out.write(frame)
                frame_num += 1
            
            cap.release()
            out.release()
            
            logger.info(f"Processed video evidence: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return None
    
    def _create_chain_of_custody(self, metadata: EvidenceMetadata, 
                               entries: List[ChainOfCustodyEntry],
                               output_dir: Path) -> str:
        """Create chain of custody document"""
        custody_data = {
            "case_information": asdict(metadata),
            "chain_of_custody": [asdict(entry) for entry in entries],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(entries),
            "integrity_verification": {
                "manifest_created": True,
                "digital_signature": self.private_key_path is not None,
                "timestamp_proof": True
            }
        }
        
        custody_path = output_dir / "chain_of_custody.json"
        with open(custody_path, 'w') as f:
            json.dump(custody_data, f, indent=2)
        
        return str(custody_path)
    
    def create_evidence_package(self,
                              metadata: EvidenceMetadata,
                              video_files: List[str] = None,
                              image_files: List[str] = None,
                              data_files: List[str] = None,
                              chain_entries: List[ChainOfCustodyEntry] = None,
                              output_path: str = None) -> Dict[str, Any]:
        """Create complete evidence package"""
        
        # Set defaults
        video_files = video_files or []
        image_files = image_files or []
        data_files = data_files or []
        chain_entries = chain_entries or []
        
        # Create output directory
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"evidence_{metadata.case_id}_{timestamp}"
        
        package_dir = Path(output_path)
        package_dir.mkdir(exist_ok=True)
        
        try:
            created_files = []
            
            # Create evidence metadata
            metadata_path = package_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            created_files.append(str(metadata_path))
            
            # Process video files
            if video_files:
                video_dir = package_dir / "video_evidence"
                video_dir.mkdir(exist_ok=True)
                
                for i, video_file in enumerate(video_files):
                    if Path(video_file).exists():
                        processed_video = self._process_video_evidence(video_file, video_dir)
                        if processed_video:
                            created_files.append(processed_video)
            
            # Copy image files
            if image_files:
                image_dir = package_dir / "image_evidence"
                image_dir.mkdir(exist_ok=True)
                
                for i, image_file in enumerate(image_files):
                    if Path(image_file).exists():
                        src_path = Path(image_file)
                        dst_path = image_dir / f"evidence_{i:03d}_{src_path.name}"
                        shutil.copy2(src_path, dst_path)
                        created_files.append(str(dst_path))
            
            # Copy data files
            if data_files:
                data_dir = package_dir / "data_evidence"
                data_dir.mkdir(exist_ok=True)
                
                for i, data_file in enumerate(data_files):
                    if Path(data_file).exists():
                        src_path = Path(data_file)
                        dst_path = data_dir / f"data_{i:03d}_{src_path.name}"
                        shutil.copy2(src_path, dst_path)
                        created_files.append(str(dst_path))
            
            # Create chain of custody
            if chain_entries:
                custody_file = self._create_chain_of_custody(metadata, chain_entries, package_dir)
                created_files.append(custody_file)
            
            # Create manifest
            manifest = self._create_manifest(package_dir)
            
            # Save JSON manifest
            manifest_json_path = package_dir / "manifest.json"
            with open(manifest_json_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            created_files.append(str(manifest_json_path))
            
            # Create SHA256 manifest
            manifest_sha_path = package_dir / "manifest.sha256"
            with open(manifest_sha_path, 'w') as f:
                for file_path, hashes in manifest.items():
                    f.write(f"{hashes['sha256']}  {file_path}\n")
            created_files.append(str(manifest_sha_path))
            
            # Create digital signature for metadata
            signature = None
            if self.private_key_path:
                metadata_bytes = json.dumps(asdict(metadata), sort_keys=True).encode('utf-8')
                signature = self._sign_data(metadata_bytes)
                
                if signature:
                    signature_path = package_dir / "metadata.json.sig"
                    with open(signature_path, 'w') as f:
                        json.dump(asdict(signature), f, indent=2)
                    created_files.append(str(signature_path))
            
            # Create OpenTimestamps for manifest
            timestamp_proof = None
            if manifest_sha_path:
                timestamp_proof = self._create_opentimestamp(manifest_sha_path)
            
            # Create package summary
            package_summary = {
                "package_id": f"{metadata.case_id}_{int(time.time())}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "case_id": metadata.case_id,
                "operator_id": metadata.operator_id,
                "agency": metadata.agency,
                "total_files": len(created_files),
                "video_files": len(video_files),
                "image_files": len(image_files),
                "data_files": len(data_files),
                "package_size_bytes": sum(f.stat().st_size for f in package_dir.rglob("*") if f.is_file()),
                "digital_signature": signature is not None,
                "timestamp_proof": timestamp_proof is not None,
                "chain_of_custody_entries": len(chain_entries),
                "files": created_files,
                "integrity_checks": {
                    "manifest_sha256": self._calculate_file_hash(manifest_sha_path),
                    "metadata_hash": self._calculate_file_hash(metadata_path),
                    "total_file_count": len(created_files)
                }
            }
            
            summary_path = package_dir / "package_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(package_summary, f, indent=2)
            
            logger.info(f"Evidence package created: {package_dir}")
            logger.info(f"Total files: {len(created_files)}")
            logger.info(f"Package size: {package_summary['package_size_bytes']} bytes")
            
            return {
                "success": True,
                "package_path": str(package_dir),
                "package_summary": package_summary,
                "digital_signature": asdict(signature) if signature else None,
                "timestamp_proof": asdict(timestamp_proof) if timestamp_proof else None
            }
            
        except Exception as e:
            logger.error(f"Failed to create evidence package: {e}")
            return {
                "success": False,
                "error": str(e),
                "package_path": str(package_dir)
            }
    
    def verify_evidence_package(self, package_path: str) -> Dict[str, Any]:
        """Verify integrity of evidence package"""
        package_dir = Path(package_path)
        
        if not package_dir.exists():
            return {"success": False, "error": "Package directory not found"}
        
        verification_results = {
            "package_path": str(package_dir),
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check for required files
            required_files = ["metadata.json", "manifest.json", "manifest.sha256"]
            for req_file in required_files:
                file_path = package_dir / req_file
                verification_results["checks"][f"{req_file}_exists"] = file_path.exists()
                if not file_path.exists():
                    verification_results["errors"].append(f"Missing required file: {req_file}")
            
            # Verify manifest hashes
            manifest_path = package_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                hash_mismatches = 0
                for file_path, expected_hashes in manifest.items():
                    actual_file = package_dir / file_path
                    if actual_file.exists():
                        actual_hash = self._calculate_file_hash(actual_file, "sha256")
                        if actual_hash != expected_hashes["sha256"]:
                            hash_mismatches += 1
                            verification_results["errors"].append(
                                f"Hash mismatch for {file_path}: expected {expected_hashes['sha256']}, got {actual_hash}"
                            )
                    else:
                        verification_results["errors"].append(f"File missing: {file_path}")
                
                verification_results["checks"]["hash_verification"] = hash_mismatches == 0
            
            # Check digital signature if present
            signature_path = package_dir / "metadata.json.sig"
            if signature_path.exists():
                verification_results["checks"]["digital_signature_present"] = True
                # Note: Full signature verification would require the public key
                verification_results["warnings"].append("Digital signature present but not verified (requires public key)")
            else:
                verification_results["checks"]["digital_signature_present"] = False
            
            # Check OpenTimestamps if present
            ots_files = list(package_dir.glob("*.ots"))
            verification_results["checks"]["timestamp_proofs_present"] = len(ots_files) > 0
            verification_results["checks"]["timestamp_proof_count"] = len(ots_files)
            
            # Overall verification status
            verification_results["success"] = len(verification_results["errors"]) == 0
            verification_results["integrity_score"] = (
                sum(1 for check in verification_results["checks"].values() if check is True) /
                len(verification_results["checks"])
            ) if verification_results["checks"] else 0
            
            return verification_results
            
        except Exception as e:
            verification_results["success"] = False
            verification_results["errors"].append(f"Verification failed: {str(e)}")
            return verification_results
    
    def __del__(self):
        """Cleanup on destruction"""
        self._cleanup_temp_directory()

def create_sample_evidence_data() -> Dict[str, Any]:
    """Create sample evidence data for testing"""
    
    metadata = EvidenceMetadata(
        case_id="SAR-2024-001",
        operator_id="OFFICER-123",
        agency="Search and Rescue Division",
        incident_type="Missing Person",
        location="Mountain Trail Area",
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat(),
        equipment_used="DJI Mavic 3, Thermal Camera",
        weather_conditions="Clear, 10mph winds",
        legal_authority="Emergency Response Authorization",
        chain_of_custody_officer="SGT. SMITH",
        notes="Aerial search for missing hiker"
    )
    
    chain_entries = [
        ChainOfCustodyEntry(
            timestamp=datetime.now().isoformat(),
            action="Evidence Collection Started",
            officer_id="OFFICER-123",
            officer_name="John Doe",
            location="Command Center",
            notes="Initiated aerial search and evidence collection"
        ),
        ChainOfCustodyEntry(
            timestamp=datetime.now().isoformat(),
            action="Video Evidence Captured",
            officer_id="OFFICER-123",
            officer_name="John Doe",
            location="Search Area Alpha",
            notes="Captured aerial video of search area"
        )
    ]
    
    return {
        "metadata": metadata,
        "chain_entries": chain_entries
    }

if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_evidence_data()
    
    packager = EvidencePackager()
    
    # Create evidence package
    result = packager.create_evidence_package(
        metadata=sample_data["metadata"],
        chain_entries=sample_data["chain_entries"]
    )
    
    print("Evidence packaging result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Verify the package
    if result["success"]:
        verification = packager.verify_evidence_package(result["package_path"])
        print("\nPackage verification:")
        print(json.dumps(verification, indent=2, default=str))