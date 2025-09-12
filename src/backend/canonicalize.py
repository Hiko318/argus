"""Canonical metadata and evidence packaging for SAR operations.

This module provides functionality to create canonical, signed metadata packages
for video evidence with cryptographic verification and timestamping.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import uuid
import base64
import hmac

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    """Metadata for a single frame."""
    frame_seq: int
    pts_ms: float  # Presentation timestamp in milliseconds
    timestamp_utc: str
    detections: List[Dict[str, Any]]
    telemetry: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MissionProfile:
    """Mission profile information."""
    mission_id: str
    mission_type: str  # "search_and_rescue", "surveillance", "mapping"
    operator_id: str
    start_time: str
    location: Dict[str, float]  # {"lat": ..., "lon": ..., "alt": ...}
    weather_conditions: Optional[Dict[str, Any]] = None
    equipment_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelVersion:
    """Model version information."""
    model_name: str
    version: str
    hash_sha256: str
    training_date: str
    accuracy_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IMUData:
    """IMU sensor data."""
    timestamp: float
    accelerometer: Dict[str, float]  # {"x": ..., "y": ..., "z": ...}
    gyroscope: Dict[str, float]  # {"x": ..., "y": ..., "z": ...}
    magnetometer: Dict[str, float]  # {"x": ..., "y": ..., "z": ...}
    temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalMetadata:
    """Canonical metadata structure for evidence packaging."""
    # Core identification
    evidence_id: str
    creation_time: str
    format_version: str = "1.0"
    
    # Video information
    video_file: str
    video_hash_sha256: str
    video_duration_ms: float
    video_resolution: Dict[str, int]  # {"width": ..., "height": ...}
    video_fps: float
    video_codec: str
    
    # Frame-by-frame metadata
    frames: List[FrameMetadata]
    
    # Mission context
    mission_profile: MissionProfile
    
    # Model information
    model_version: ModelVersion
    
    # IMU data
    imu_data: List[IMUData]
    
    # Chain of custody
    chain_of_custody: List[Dict[str, Any]]
    
    # Verification
    metadata_hash: Optional[str] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic ordering."""
        data = asdict(self)
        # Ensure deterministic JSON serialization
        return self._sort_dict_recursively(data)
    
    def _sort_dict_recursively(self, obj: Any) -> Any:
        """Sort dictionary keys recursively for canonical representation."""
        if isinstance(obj, dict):
            return {k: self._sort_dict_recursively(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [self._sort_dict_recursively(item) for item in obj]
        else:
            return obj
    
    def to_canonical_json(self) -> str:
        """Convert to canonical JSON string."""
        data = self.to_dict()
        # Remove signature and hash for canonical representation
        canonical_data = {k: v for k, v in data.items() 
                         if k not in ['metadata_hash', 'signature']}
        
        return json.dumps(canonical_data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of canonical metadata."""
        canonical_json = self.to_canonical_json()
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    
    def update_hash(self) -> None:
        """Update the metadata hash."""
        self.metadata_hash = self.compute_hash()


class VaultTransitSigner:
    """HashiCorp Vault Transit engine signer."""
    
    def __init__(self, vault_url: str, vault_token: str, key_name: str):
        self.vault_url = vault_url.rstrip('/')
        self.vault_token = vault_token
        self.key_name = key_name
        self.session = None
        
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available for Vault integration")
        
        self.session = requests.Session()
        self.session.headers.update({
            'X-Vault-Token': vault_token,
            'Content-Type': 'application/json'
        })
        
        logger.info(f"Initialized Vault Transit signer with key: {key_name}")
    
    def sign_data(self, data: str) -> str:
        """Sign data using Vault Transit engine.
        
        Args:
            data: Data to sign
            
        Returns:
            Base64-encoded signature
        """
        try:
            # Encode data to base64
            encoded_data = base64.b64encode(data.encode('utf-8')).decode('utf-8')
            
            # Prepare request
            url = f"{self.vault_url}/v1/transit/sign/{self.key_name}"
            payload = {
                "input": encoded_data,
                "hash_algorithm": "sha2-256"
            }
            
            # Make request
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            # Extract signature
            result = response.json()
            signature = result['data']['signature']
            
            logger.info(f"Successfully signed data with Vault Transit")
            return signature
            
        except Exception as e:
            logger.error(f"Failed to sign data with Vault: {e}")
            raise
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify signature using Vault Transit engine.
        
        Args:
            data: Original data
            signature: Signature to verify
            
        Returns:
            True if signature is valid
        """
        try:
            # Encode data to base64
            encoded_data = base64.b64encode(data.encode('utf-8')).decode('utf-8')
            
            # Prepare request
            url = f"{self.vault_url}/v1/transit/verify/{self.key_name}"
            payload = {
                "input": encoded_data,
                "signature": signature,
                "hash_algorithm": "sha2-256"
            }
            
            # Make request
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            # Check result
            result = response.json()
            is_valid = result['data']['valid']
            
            logger.info(f"Signature verification result: {is_valid}")
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to verify signature with Vault: {e}")
            return False


class LocalDevSigner:
    """Local development signer using RSA keys."""
    
    def __init__(self, private_key_path: Optional[str] = None, public_key_path: Optional[str] = None):
        self.private_key_path = private_key_path
        self.public_key_path = public_key_path
        self.private_key = None
        self.public_key = None
        
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("cryptography library not available for local signing")
        
        # Generate or load keys
        if private_key_path and Path(private_key_path).exists():
            self._load_private_key()
        else:
            self._generate_keys()
        
        logger.info("Initialized local development signer")
    
    def _generate_keys(self) -> None:
        """Generate new RSA key pair."""
        logger.info("Generating new RSA key pair...")
        
        # Generate private key
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Get public key
        self.public_key = self.private_key.public_key()
        
        # Save keys if paths provided
        if self.private_key_path:
            self._save_private_key()
        if self.public_key_path:
            self._save_public_key()
        
        logger.info("RSA key pair generated")
    
    def _load_private_key(self) -> None:
        """Load private key from file."""
        try:
            with open(self.private_key_path, 'rb') as f:
                self.private_key = load_pem_private_key(
                    f.read(),
                    password=None  # No password for dev keys
                )
            
            self.public_key = self.private_key.public_key()
            logger.info(f"Loaded private key from {self.private_key_path}")
            
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
            raise
    
    def _save_private_key(self) -> None:
        """Save private key to file."""
        try:
            pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            os.makedirs(Path(self.private_key_path).parent, exist_ok=True)
            with open(self.private_key_path, 'wb') as f:
                f.write(pem)
            
            logger.info(f"Saved private key to {self.private_key_path}")
            
        except Exception as e:
            logger.error(f"Failed to save private key: {e}")
            raise
    
    def _save_public_key(self) -> None:
        """Save public key to file."""
        try:
            pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            os.makedirs(Path(self.public_key_path).parent, exist_ok=True)
            with open(self.public_key_path, 'wb') as f:
                f.write(pem)
            
            logger.info(f"Saved public key to {self.public_key_path}")
            
        except Exception as e:
            logger.error(f"Failed to save public key: {e}")
            raise
    
    def sign_data(self, data: str) -> str:
        """Sign data using RSA private key.
        
        Args:
            data: Data to sign
            
        Returns:
            Base64-encoded signature
        """
        try:
            signature = self.private_key.sign(
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            raise
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify signature using RSA public key.
        
        Args:
            data: Original data
            signature: Base64-encoded signature
            
        Returns:
            True if signature is valid
        """
        try:
            signature_bytes = base64.b64decode(signature)
            
            self.public_key.verify(
                signature_bytes,
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception:
            return False


class EvidencePackager:
    """Evidence packager for creating canonical signed metadata packages."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signer = None
        
        # Initialize signer based on configuration
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
        
        logger.info(f"Initialized evidence packager with {signer_type} signer")
    
    def create_evidence_package(
        self,
        video_path: str,
        frames_metadata: List[FrameMetadata],
        mission_profile: MissionProfile,
        model_version: ModelVersion,
        imu_data: List[IMUData],
        output_dir: str
    ) -> Dict[str, str]:
        """Create a complete evidence package.
        
        Args:
            video_path: Path to video file
            frames_metadata: Frame-by-frame metadata
            mission_profile: Mission information
            model_version: Model version info
            imu_data: IMU sensor data
            output_dir: Output directory for package
            
        Returns:
            Dictionary with paths to created files
        """
        try:
            logger.info(f"Creating evidence package for {video_path}")
            
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate evidence ID
            evidence_id = str(uuid.uuid4())
            
            # Get video information
            video_info = self._get_video_info(video_path)
            video_hash = self._compute_file_hash(video_path)
            
            # Create canonical metadata
            metadata = CanonicalMetadata(
                evidence_id=evidence_id,
                creation_time=datetime.now(timezone.utc).isoformat(),
                video_file=Path(video_path).name,
                video_hash_sha256=video_hash,
                video_duration_ms=video_info['duration_ms'],
                video_resolution=video_info['resolution'],
                video_fps=video_info['fps'],
                video_codec=video_info['codec'],
                frames=frames_metadata,
                mission_profile=mission_profile,
                model_version=model_version,
                imu_data=imu_data,
                chain_of_custody=[
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "action": "evidence_package_created",
                        "operator": mission_profile.operator_id,
                        "system": "foresight_sar",
                        "hash": video_hash
                    }
                ]
            )
            
            # Update metadata hash
            metadata.update_hash()
            
            # Sign metadata
            canonical_json = metadata.to_canonical_json()
            signature = self.signer.sign_data(canonical_json)
            metadata.signature = signature
            
            # Create file paths
            base_name = f"evidence_{evidence_id}"
            video_dest = output_path / f"{base_name}.mp4"
            metadata_path = output_path / f"{base_name}_metadata.json"
            signature_path = output_path / f"{base_name}_metadata.json.sig"
            manifest_path = output_path / f"{base_name}_manifest.sha256"
            
            # Copy video file
            import shutil
            shutil.copy2(video_path, video_dest)
            
            # Write metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=True)
            
            # Write signature
            with open(signature_path, 'w', encoding='utf-8') as f:
                f.write(signature)
            
            # Create manifest
            manifest_content = self._create_manifest({
                str(video_dest): self._compute_file_hash(str(video_dest)),
                str(metadata_path): self._compute_file_hash(str(metadata_path)),
                str(signature_path): self._compute_file_hash(str(signature_path))
            })
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                f.write(manifest_content)
            
            # Create package info
            package_files = {
                'video': str(video_dest),
                'metadata': str(metadata_path),
                'signature': str(signature_path),
                'manifest': str(manifest_path),
                'evidence_id': evidence_id
            }
            
            logger.info(f"Evidence package created: {evidence_id}")
            return package_files
            
        except Exception as e:
            logger.error(f"Failed to create evidence package: {e}")
            raise
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information."""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0
            
            cap.release()
            
            return {
                'duration_ms': duration_ms,
                'resolution': {'width': width, 'height': height},
                'fps': fps,
                'codec': 'h264',  # Default assumption
                'frame_count': frame_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            # Return default values
            return {
                'duration_ms': 0,
                'resolution': {'width': 1920, 'height': 1080},
                'fps': 30.0,
                'codec': 'unknown',
                'frame_count': 0
            }
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _create_manifest(self, file_hashes: Dict[str, str]) -> str:
        """Create SHA-256 manifest file content."""
        lines = []
        for file_path, file_hash in sorted(file_hashes.items()):
            filename = Path(file_path).name
            lines.append(f"{file_hash}  {filename}")
        
        return '\n'.join(lines) + '\n'
    
    def verify_signature(self, metadata_path: str, signature_path: str) -> bool:
        """Verify metadata signature.
        
        Args:
            metadata_path: Path to metadata JSON file
            signature_path: Path to signature file
            
        Returns:
            True if signature is valid
        """
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
            
            # Verify
            return self.signer.verify_signature(canonical_json, signature)
            
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'signer_type': 'local_dev',
        'local_dev': {
            'private_key_path': 'keys/evidence_signing_key.pem',
            'public_key_path': 'keys/evidence_signing_key.pub'
        }
    }
    
    # Create packager
    packager = EvidencePackager(config)
    
    # Example metadata
    frames = [
        FrameMetadata(
            frame_seq=1,
            pts_ms=33.33,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            detections=[{
                "class": "person",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 300]
            }]
        )
    ]
    
    mission = MissionProfile(
        mission_id="mission_001",
        mission_type="search_and_rescue",
        operator_id="operator_001",
        start_time=datetime.now(timezone.utc).isoformat(),
        location={"lat": 37.7749, "lon": -122.4194, "alt": 100.0}
    )
    
    model = ModelVersion(
        model_name="yolo11n",
        version="1.0.0",
        hash_sha256="abc123...",
        training_date="2024-01-01",
        accuracy_metrics={"mAP": 0.85}
    )
    
    imu = [
        IMUData(
            timestamp=time.time(),
            accelerometer={"x": 0.1, "y": 0.2, "z": 9.8},
            gyroscope={"x": 0.01, "y": 0.02, "z": 0.03},
            magnetometer={"x": 25.0, "y": 30.0, "z": 45.0}
        )
    ]
    
    print("Evidence packager example ready")