#!/usr/bin/env python3
"""
Vault Transit Integration for Foresight Evidence Signing

This module provides secure signing capabilities using HashiCorp Vault's Transit engine.
It handles metadata signing, key management, and public key storage for verification.
"""

import os
import json
import base64
import hashlib
import requests
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VaultTransitClient:
    """HashiCorp Vault Transit engine client for evidence signing."""
    
    def __init__(self, vault_url: str = None, vault_token: str = None):
        self.vault_url = vault_url or os.getenv('VAULT_ADDR', 'http://localhost:8200')
        self.vault_token = vault_token or os.getenv('VAULT_TOKEN')
        self.transit_path = 'transit'
        self.key_name = 'foresight-evidence-signing'
        
        if not self.vault_token:
            raise ValueError("Vault token required. Set VAULT_TOKEN environment variable.")
        
        self.headers = {
            'X-Vault-Token': self.vault_token,
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, method: str, path: str, data: Dict = None) -> Dict:
        """Make authenticated request to Vault API."""
        url = f"{self.vault_url}/v1/{path}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Vault API request failed: {e}")
            raise
    
    def initialize_transit_engine(self) -> bool:
        """Initialize Transit engine and create signing key."""
        try:
            # Enable Transit engine
            self._make_request('POST', 'sys/mounts/transit', {
                'type': 'transit',
                'description': 'Foresight evidence signing'
            })
            logger.info("Transit engine enabled")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                logger.info("Transit engine already enabled")
            else:
                raise
        
        # Create signing key
        try:
            self._make_request('POST', f'{self.transit_path}/keys/{self.key_name}', {
                'type': 'ecdsa-p256',
                'exportable': False,
                'allow_plaintext_backup': False
            })
            logger.info(f"Created signing key: {self.key_name}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                logger.info(f"Signing key {self.key_name} already exists")
            else:
                raise
        
        return True
    
    def get_public_key(self) -> str:
        """Retrieve public key for verification."""
        response = self._make_request('GET', f'{self.transit_path}/keys/{self.key_name}')
        
        # Extract latest public key
        keys = response['data']['keys']
        latest_version = max(keys.keys())
        public_key = keys[latest_version]['public_key']
        
        return public_key
    
    def sign_data(self, data: bytes) -> str:
        """Sign data using Vault Transit engine."""
        # Hash the data
        data_hash = hashlib.sha256(data).digest()
        
        # Base64 encode for Vault
        encoded_hash = base64.b64encode(data_hash).decode('utf-8')
        
        response = self._make_request('POST', f'{self.transit_path}/sign/{self.key_name}/sha2-256', {
            'input': encoded_hash
        })
        
        return response['data']['signature']
    
    def verify_signature(self, data: bytes, signature: str) -> bool:
        """Verify signature using Vault Transit engine."""
        # Hash the data
        data_hash = hashlib.sha256(data).digest()
        
        # Base64 encode for Vault
        encoded_hash = base64.b64encode(data_hash).decode('utf-8')
        
        try:
            response = self._make_request('POST', f'{self.transit_path}/verify/{self.key_name}/sha2-256', {
                'input': encoded_hash,
                'signature': signature
            })
            return response['data']['valid']
        except requests.exceptions.HTTPError:
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
        with open(sig_file, 'w') as f:
            json.dump({
                'signature': signature,
                'algorithm': 'ecdsa-p256-sha256',
                'key_id': self.key_name,
                'signed_at': metadata_data.decode('utf-8') if metadata_data else None
            }, f, indent=2)
        
        logger.info(f"Signed metadata: {metadata_path} -> {sig_file}")
        return signature, public_key

def store_public_key(public_key: str, repo_path: str = None) -> str:
    """Store public key in repository for verification."""
    if repo_path is None:
        repo_path = Path(__file__).parent.parent
    else:
        repo_path = Path(repo_path)
    
    keys_dir = repo_path / 'keys'
    keys_dir.mkdir(exist_ok=True)
    
    key_file = keys_dir / 'foresight-evidence-signing.pub'
    
    with open(key_file, 'w') as f:
        f.write(public_key)
    
    logger.info(f"Public key stored: {key_file}")
    return str(key_file)

def main():
    """CLI interface for Vault Transit operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vault Transit Evidence Signing')
    parser.add_argument('--init', action='store_true', help='Initialize Transit engine and keys')
    parser.add_argument('--sign', type=str, help='Sign metadata file')
    parser.add_argument('--verify', type=str, help='Verify signed metadata file')
    parser.add_argument('--export-key', action='store_true', help='Export public key to repository')
    
    args = parser.parse_args()
    
    try:
        client = VaultTransitClient()
        
        if args.init:
            client.initialize_transit_engine()
            public_key = client.get_public_key()
            store_public_key(public_key)
            print("Vault Transit initialized successfully")
        
        elif args.sign:
            signature, public_key = client.sign_metadata_file(args.sign)
            print(f"Metadata signed: {args.sign}")
            print(f"Signature: {signature}")
        
        elif args.verify:
            metadata_file = Path(args.verify)
            sig_file = metadata_file.with_suffix('.sig')
            
            if not sig_file.exists():
                print(f"Signature file not found: {sig_file}")
                return
            
            with open(metadata_file, 'rb') as f:
                metadata_data = f.read()
            
            with open(sig_file, 'r') as f:
                sig_data = json.load(f)
            
            is_valid = client.verify_signature(metadata_data, sig_data['signature'])
            print(f"Signature valid: {is_valid}")
        
        elif args.export_key:
            public_key = client.get_public_key()
            key_file = store_public_key(public_key)
            print(f"Public key exported: {key_file}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())