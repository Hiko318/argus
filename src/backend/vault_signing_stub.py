#!/usr/bin/env python3
"""
Vault Signing Dev-Stub for Foresight Evidence Signing

This module provides a dummy local signing client for development and testing.
It mimics the Vault Transit API behavior without requiring a real Vault instance.
"""

import os
import json
import base64
import hashlib
import time
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
import logging
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VaultSigningStub:
    """Development stub for Vault Transit signing functionality."""
    
    def __init__(self, key_storage_dir: str = "./dev_keys"):
        self.key_storage_dir = Path(key_storage_dir)
        self.key_storage_dir.mkdir(exist_ok=True)
        self.key_name = 'foresight-evidence-signing-dev'
        self.private_key_path = self.key_storage_dir / f"{self.key_name}.pem"
        self.public_key_path = self.key_storage_dir / f"{self.key_name}.pub"
        
        # Initialize or load keys
        self._ensure_keys_exist()
    
    def _ensure_keys_exist(self):
        """Generate RSA key pair if it doesn't exist."""
        if not self.private_key_path.exists():
            logger.info("Generating new RSA key pair for development...")
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Save private key
            with open(self.private_key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            public_key = private_key.public_key()
            with open(self.public_key_path, 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            logger.info(f"Keys generated and saved to {self.key_storage_dir}")
        else:
            logger.info("Using existing development keys")
    
    def _load_private_key(self):
        """Load private key from file."""
        with open(self.private_key_path, 'rb') as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    
    def _load_public_key(self):
        """Load public key from file."""
        with open(self.public_key_path, 'rb') as f:
            return serialization.load_pem_public_key(f.read())
    
    def initialize_transit_engine(self) -> bool:
        """Stub: Initialize Transit engine (always returns True)."""
        logger.info("[STUB] Transit engine initialized (development mode)")
        return True
    
    def get_public_key(self) -> str:
        """Return public key in PEM format."""
        with open(self.public_key_path, 'r') as f:
            return f.read()
    
    def sign_data(self, data: bytes) -> str:
        """Sign data using local RSA private key."""
        private_key = self._load_private_key()
        
        # Sign the data
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Return base64 encoded signature with vault-like prefix
        encoded_sig = base64.b64encode(signature).decode('utf-8')
        return f"vault:v1:{encoded_sig}"
    
    def verify_signature(self, data: bytes, signature: str) -> bool:
        """Verify signature using local RSA public key."""
        try:
            public_key = self._load_public_key()
            
            # Extract signature from vault format
            if signature.startswith("vault:v1:"):
                sig_data = signature[9:]  # Remove "vault:v1:" prefix
            else:
                sig_data = signature
            
            decoded_sig = base64.b64decode(sig_data)
            
            # Verify signature
            public_key.verify(
                decoded_sig,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
            
        except (InvalidSignature, Exception) as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def sign_metadata_file(self, metadata_path: str) -> Tuple[str, str]:
        """Sign metadata.json file and return signature and public key."""
        metadata_file = Path(metadata_path)
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Read and sign metadata
        with open(metadata_file, 'rb') as f:
            metadata_data = f.read()
        
        signature = self.sign_data(metadata_data)
        public_key = self.get_public_key()
        
        # Save signature file
        sig_file = metadata_file.with_suffix('.sig')
        signature_info = {
            'signature': signature,
            'algorithm': 'rsa-pss-sha256',
            'key_id': self.key_name,
            'signed_at': datetime.now(timezone.utc).isoformat(),
            'file_hash': hashlib.sha256(metadata_data).hexdigest(),
            'stub_mode': True,
            'note': 'Development signature - not for production use'
        }
        
        with open(sig_file, 'w') as f:
            json.dump(signature_info, f, indent=2)
        
        logger.info(f"Metadata signed: {sig_file}")
        return signature, public_key
    
    def verify_metadata_file(self, metadata_path: str, signature_path: str = None) -> bool:
        """Verify metadata file signature."""
        metadata_file = Path(metadata_path)
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Determine signature file path
        if signature_path:
            sig_file = Path(signature_path)
        else:
            sig_file = metadata_file.with_suffix('.sig')
        
        if not sig_file.exists():
            raise FileNotFoundError(f"Signature file not found: {sig_file}")
        
        # Load signature info
        with open(sig_file, 'r') as f:
            sig_info = json.load(f)
        
        # Read metadata
        with open(metadata_file, 'rb') as f:
            metadata_data = f.read()
        
        # Verify signature
        return self.verify_signature(metadata_data, sig_info['signature'])


def run_signing_test() -> bool:
    """Run signing test with sample metadata."""
    try:
        # Initialize stub client
        vault_stub = VaultSigningStub()
        
        # Test metadata path
        test_metadata_path = "demos/sample_mission/metadata.json"
        if not os.path.exists(test_metadata_path):
            logger.error(f"Test metadata file not found: {test_metadata_path}")
            return False
        
        # Sign metadata
        signature, public_key = vault_stub.sign_metadata_file(test_metadata_path)
        
        # Verify signature
        verification_result = vault_stub.verify_metadata_file(test_metadata_path)
        
        if verification_result:
            logger.info("Vault signing stub test passed!")
            logger.info(f"Signature: {signature[:50]}...")
            logger.info(f"Public key length: {len(public_key)} characters")
            return True
        else:
            logger.error("Signature verification failed")
            return False
            
    except Exception as e:
        logger.error(f"Signing test failed: {e}")
        return False


if __name__ == "__main__":
    # Run signing test
    success = run_signing_test()
    exit(0 if success else 1)