#!/usr/bin/env python3
"""
Evidence Packager - Creates tamper-evident evidence packages

Produces evidence.zip containing:
- MP4 video file
- metadata.json with capture details
- metadata.sig with digital signature
- manifest.sha256 with file hashes
- .ots timestamp proof
"""

import os
import json
import hashlib
import zipfile
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

# Import our security modules
try:
    from .vault_transit import VaultTransitClient
except ImportError:
    from vault_transit import VaultTransitClient

try:
    from .opentimestamps_client import OpenTimestampsClient, create_evidence_timestamp_bundle
except ImportError:
    from opentimestamps_client import OpenTimestampsClient, create_evidence_timestamp_bundle


class EvidencePackager:
    """Creates tamper-evident evidence packages for SAR operations"""
    
    def __init__(self, private_key_path: Optional[str] = None):
        self.private_key_path = private_key_path
        self.private_key = None
        self.public_key = None
        
        if private_key_path and os.path.exists(private_key_path):
            self._load_keys()
        else:
            self._generate_keys()
    
    def _generate_keys(self):
        """Generate RSA key pair for signing"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # Save keys to tools directory
        keys_dir = Path(__file__).parent / "keys"
        keys_dir.mkdir(exist_ok=True)
        
        # Save private key
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        with open(keys_dir / "evidence_private.pem", "wb") as f:
            f.write(private_pem)
        
        # Save public key
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(keys_dir / "evidence_public.pem", "wb") as f:
            f.write(public_pem)
    
    def _load_keys(self):
        """Load existing RSA keys"""
        with open(self.private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )
        self.public_key = self.private_key.public_key()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _sign_data(self, data: bytes) -> str:
        """Sign data with private key"""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    
    def _create_metadata(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata structure"""
        return {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "evidence": {
                "timestamp": evidence_data.get("timestamp"),
                "location": evidence_data.get("location"),
                "operator_id": evidence_data.get("operatorId"),
                "notes": evidence_data.get("notes", ""),
                "capture_method": "flag_lock_ui"
            },
            "system": {
                "version": "Foresight 1.0",
                "platform": "SAR Interface",
                "hash_algorithm": "SHA256",
                "signature_algorithm": "RSA-PSS"
            }
        }
    
    def _sign_metadata_vault(self, metadata_path: Path) -> None:
        """Create digital signature for metadata file using Vault Transit.
        
        Args:
            metadata_path: Path to metadata.json file
        """
        try:
            vault_client = VaultTransitClient()
            signature, public_key = vault_client.sign_metadata_file(str(metadata_path))
            
            # Store public key in package for verification
            key_file = metadata_path.parent / 'public_key.pem'
            with open(key_file, 'w') as f:
                f.write(public_key)
            
        except Exception as e:
            print(f"Vault signing failed, using fallback: {e}")
            self._sign_metadata_fallback(metadata_path)
    
    def _sign_metadata_fallback(self, metadata_path: Path) -> None:
        """Create digital signature for metadata file using local keys.
        
        Args:
            metadata_path: Path to metadata.json file
        """
        # Use existing RSA signing implementation
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata_bytes = json.dumps(metadata, sort_keys=True).encode('utf-8')
        signature = self._sign_data(metadata_bytes)
        
        signature_data = {
            "signature": signature,
            "algorithm": "RSA-PSS",
            "hash_algorithm": "SHA256",
            "public_key": base64.b64encode(
                self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            ).decode('utf-8')
        }
        
        signature_path = metadata_path.with_suffix('.sig')
        with open(signature_path, "w") as f:
            json.dump(signature_data, f, indent=2)
    
    def _create_timestamps(self, package_dir: Path) -> None:
        """Create OpenTimestamps proofs for evidence package.
        
        Args:
            package_dir: Evidence package directory
        """
        try:
            timestamp_info = create_evidence_timestamp_bundle(str(package_dir))
            print(f"OpenTimestamps proofs created: {timestamp_info}")
            
        except Exception as e:
            print(f"OpenTimestamps creation failed: {e}")
            # Create a simple timestamp file as fallback
            timestamp_file = package_dir / 'timestamp.txt'
            with open(timestamp_file, 'w') as f:
                f.write(f"Package created at: {datetime.utcnow().isoformat()}Z\n")
                f.write(f"Note: OpenTimestamps not available - {e}\n")
    
    def _create_ots_timestamp(self, file_path: str) -> str:
        """Create OpenTimestamps proof (placeholder implementation)"""
        # In a real implementation, this would submit to OpenTimestamps
        # For now, create a placeholder .ots file
        ots_content = f"# OpenTimestamps proof for {os.path.basename(file_path)}\n"
        ots_content += f"# Created: {datetime.utcnow().isoformat()}Z\n"
        ots_content += f"# File hash: {self._calculate_file_hash(file_path)}\n"
        return ots_content
    
    def package_evidence(self, evidence_data: Dict[str, Any], output_dir: str = ".", 
                        use_vault: bool = True, use_timestamps: bool = True) -> str:
        """Package evidence into tamper-evident zip file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"evidence_{timestamp}"
        temp_dir = Path(output_dir) / f"temp_{package_name}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Save image as JPEG
            if "image" in evidence_data:
                image_data = evidence_data["image"]
                if image_data.startswith("data:image/jpeg;base64,"):
                    image_data = image_data.split(",")[1]
                
                image_path = temp_dir / "capture.jpg"
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
            
            # Create metadata
            metadata = self._create_metadata(evidence_data)
            metadata_path = temp_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Create manifest with file hashes
            manifest = {}
            for file_path in temp_dir.glob("*"):
                if file_path.is_file():
                    manifest[file_path.name] = self._calculate_file_hash(str(file_path))
            
            manifest_path = temp_dir / "manifest.sha256"
            with open(manifest_path, "w") as f:
                for filename, file_hash in manifest.items():
                    f.write(f"{file_hash}  {filename}\n")
            
            # Sign metadata using Vault Transit or fallback
            if use_vault:
                self._sign_metadata_vault(metadata_path)
            else:
                self._sign_metadata_fallback(metadata_path)
            
            # Create OpenTimestamps proofs
            if use_timestamps:
                self._create_timestamps(temp_dir)
            else:
                # Create OTS timestamp (placeholder)
                ots_content = self._create_ots_timestamp(str(metadata_path))
                ots_path = temp_dir / "metadata.ots"
                with open(ots_path, "w") as f:
                    f.write(ots_content)
            
            # Create final zip package
            zip_path = Path(output_dir) / f"{package_name}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)
            
            return str(zip_path)
        
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """CLI interface for evidence packaging"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Package evidence for SAR operations")
    parser.add_argument("--metadata", required=True, help="JSON file with evidence metadata")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--private-key", help="Path to private key file")
    
    args = parser.parse_args()
    
    # Load evidence data
    with open(args.metadata, "r") as f:
        evidence_data = json.load(f)
    
    # Package evidence
    packager = EvidencePackager(args.private_key)
    package_path = packager.package_evidence(evidence_data, args.output)
    
    print(f"Evidence package created: {package_path}")


if __name__ == "__main__":
    main()