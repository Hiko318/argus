#!/usr/bin/env python3
"""Vault Transit integration for evidence signing.

This script provides functionality to sign and verify evidence packages
using HashiCorp Vault's Transit secrets engine.
"""

import argparse
import base64
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SigningResult:
    """Result of signing operation."""
    success: bool
    file_path: str
    signature_path: Optional[str] = None
    signature: Optional[str] = None
    key_version: Optional[int] = None
    signing_time: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "signature_path": self.signature_path,
            "signature": self.signature,
            "key_version": self.key_version,
            "signing_time": self.signing_time,
            "error_message": self.error_message
        }


@dataclass
class VerificationResult:
    """Result of signature verification."""
    success: bool
    file_path: str
    signature_path: str
    is_valid: bool
    key_version: Optional[int] = None
    verification_time: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "signature_path": self.signature_path,
            "is_valid": self.is_valid,
            "key_version": self.key_version,
            "verification_time": self.verification_time,
            "error_message": self.error_message
        }


class VaultTransitClient:
    """HashiCorp Vault Transit client for signing operations."""
    
    def __init__(self, vault_url: str, vault_token: str, key_name: str, mount_path: str = "transit"):
        """Initialize Vault Transit client.
        
        Args:
            vault_url: Vault server URL (e.g., https://vault.example.com:8200)
            vault_token: Vault authentication token
            key_name: Name of the signing key in Vault
            mount_path: Transit secrets engine mount path
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for Vault integration")
        
        self.vault_url = vault_url.rstrip('/')
        self.vault_token = vault_token
        self.key_name = key_name
        self.mount_path = mount_path
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'X-Vault-Token': self.vault_token,
            'Content-Type': 'application/json'
        })
        
        # Verify connection and key
        self._verify_connection()
        self._verify_key()
        
        logger.info(f"Initialized Vault Transit client for key: {key_name}")
    
    def _verify_connection(self) -> None:
        """Verify connection to Vault server."""
        try:
            response = self.session.get(f"{self.vault_url}/v1/sys/health", timeout=10)
            
            if response.status_code not in [200, 429, 472, 473, 501]:
                raise RuntimeError(f"Vault health check failed: {response.status_code}")
            
            logger.debug("Vault connection verified")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Vault: {e}")
    
    def _verify_key(self) -> None:
        """Verify that the signing key exists and is accessible."""
        try:
            url = f"{self.vault_url}/v1/{self.mount_path}/keys/{self.key_name}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                key_info = response.json()
                logger.debug(f"Signing key verified: {self.key_name}")
                logger.debug(f"Key type: {key_info.get('data', {}).get('type', 'unknown')}")
            elif response.status_code == 404:
                raise RuntimeError(f"Signing key not found: {self.key_name}")
            else:
                raise RuntimeError(f"Failed to access signing key: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to verify signing key: {e}")
    
    def sign_data(self, data: str, hash_algorithm: str = "sha2-256") -> SigningResult:
        """Sign data using Vault Transit.
        
        Args:
            data: Data to sign (will be base64 encoded)
            hash_algorithm: Hash algorithm to use
            
        Returns:
            SigningResult with signature information
        """
        try:
            # Encode data
            encoded_data = base64.b64encode(data.encode('utf-8')).decode('utf-8')
            
            # Prepare request
            url = f"{self.vault_url}/v1/{self.mount_path}/sign/{self.key_name}/{hash_algorithm}"
            payload = {
                "input": encoded_data
            }
            
            logger.debug(f"Signing data with Vault Transit: {len(data)} characters")
            
            # Make signing request
            response = self.session.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                signature = result['data']['signature']
                key_version = result['data'].get('key_version')
                
                logger.info(f"Data signed successfully with key version {key_version}")
                
                return SigningResult(
                    success=True,
                    file_path="<data>",
                    signature=signature,
                    key_version=key_version,
                    signing_time=datetime.utcnow().isoformat()
                )
            else:
                error_msg = f"Vault signing failed: {response.status_code}"
                if response.content:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('errors', [response.text])}"
                    except:
                        error_msg += f" - {response.text}"
                
                return SigningResult(
                    success=False,
                    file_path="<data>",
                    error_message=error_msg
                )
                
        except requests.exceptions.RequestException as e:
            return SigningResult(
                success=False,
                file_path="<data>",
                error_message=f"Network error during signing: {e}"
            )
        except Exception as e:
            return SigningResult(
                success=False,
                file_path="<data>",
                error_message=f"Signing failed: {e}"
            )
    
    def verify_signature(self, data: str, signature: str, hash_algorithm: str = "sha2-256") -> VerificationResult:
        """Verify signature using Vault Transit.
        
        Args:
            data: Original data that was signed
            signature: Signature to verify
            hash_algorithm: Hash algorithm used for signing
            
        Returns:
            VerificationResult with verification status
        """
        try:
            # Encode data
            encoded_data = base64.b64encode(data.encode('utf-8')).decode('utf-8')
            
            # Prepare request
            url = f"{self.vault_url}/v1/{self.mount_path}/verify/{self.key_name}/{hash_algorithm}"
            payload = {
                "input": encoded_data,
                "signature": signature
            }
            
            logger.debug(f"Verifying signature with Vault Transit")
            
            # Make verification request
            response = self.session.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                is_valid = result['data']['valid']
                
                logger.info(f"Signature verification: {'valid' if is_valid else 'invalid'}")
                
                return VerificationResult(
                    success=True,
                    file_path="<data>",
                    signature_path="<signature>",
                    is_valid=is_valid,
                    verification_time=datetime.utcnow().isoformat()
                )
            else:
                error_msg = f"Vault verification failed: {response.status_code}"
                if response.content:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('errors', [response.text])}"
                    except:
                        error_msg += f" - {response.text}"
                
                return VerificationResult(
                    success=False,
                    file_path="<data>",
                    signature_path="<signature>",
                    is_valid=False,
                    error_message=error_msg
                )
                
        except requests.exceptions.RequestException as e:
            return VerificationResult(
                success=False,
                file_path="<data>",
                signature_path="<signature>",
                is_valid=False,
                error_message=f"Network error during verification: {e}"
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                file_path="<data>",
                signature_path="<signature>",
                is_valid=False,
                error_message=f"Verification failed: {e}"
            )
    
    def sign_file(self, file_path: str, output_path: Optional[str] = None) -> SigningResult:
        """Sign a file using Vault Transit.
        
        Args:
            file_path: Path to file to sign
            output_path: Path to save signature. If None, saves as {file_path}.sig
            
        Returns:
            SigningResult with signature information
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return SigningResult(
                success=False,
                file_path=file_path,
                error_message=f"File not found: {file_path}"
            )
        
        try:
            # Read file content
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Sign content
            result = self.sign_data(content)
            
            if result.success:
                # Determine output path
                if output_path:
                    signature_path = Path(output_path)
                else:
                    signature_path = file_path_obj.with_suffix(file_path_obj.suffix + '.sig')
                
                # Save signature
                with open(signature_path, 'w', encoding='utf-8') as f:
                    f.write(result.signature)
                
                logger.info(f"Signature saved to: {signature_path}")
                
                return SigningResult(
                    success=True,
                    file_path=file_path,
                    signature_path=str(signature_path),
                    signature=result.signature,
                    key_version=result.key_version,
                    signing_time=result.signing_time
                )
            else:
                return SigningResult(
                    success=False,
                    file_path=file_path,
                    error_message=result.error_message
                )
                
        except Exception as e:
            return SigningResult(
                success=False,
                file_path=file_path,
                error_message=f"Failed to sign file: {e}"
            )
    
    def verify_file_signature(self, file_path: str, signature_path: str) -> VerificationResult:
        """Verify a file signature using Vault Transit.
        
        Args:
            file_path: Path to original file
            signature_path: Path to signature file
            
        Returns:
            VerificationResult with verification status
        """
        file_path_obj = Path(file_path)
        signature_path_obj = Path(signature_path)
        
        if not file_path_obj.exists():
            return VerificationResult(
                success=False,
                file_path=file_path,
                signature_path=signature_path,
                is_valid=False,
                error_message=f"File not found: {file_path}"
            )
        
        if not signature_path_obj.exists():
            return VerificationResult(
                success=False,
                file_path=file_path,
                signature_path=signature_path,
                is_valid=False,
                error_message=f"Signature file not found: {signature_path}"
            )
        
        try:
            # Read file content
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Read signature
            with open(signature_path_obj, 'r', encoding='utf-8') as f:
                signature = f.read().strip()
            
            # Verify signature
            result = self.verify_signature(content, signature)
            
            return VerificationResult(
                success=result.success,
                file_path=file_path,
                signature_path=signature_path,
                is_valid=result.is_valid,
                verification_time=result.verification_time,
                error_message=result.error_message
            )
            
        except Exception as e:
            return VerificationResult(
                success=False,
                file_path=file_path,
                signature_path=signature_path,
                is_valid=False,
                error_message=f"Failed to verify file signature: {e}"
            )


def sign_evidence_package(package_dir: str, vault_config: Dict[str, str]) -> List[SigningResult]:
    """Sign all metadata files in an evidence package.
    
    Args:
        package_dir: Directory containing evidence package
        vault_config: Vault configuration dictionary
        
    Returns:
        List of SigningResult for each file
    """
    package_path = Path(package_dir)
    
    if not package_path.exists() or not package_path.is_dir():
        return [SigningResult(
            success=False,
            file_path=package_dir,
            error_message=f"Package directory not found: {package_dir}"
        )]
    
    try:
        client = VaultTransitClient(
            vault_url=vault_config['vault_url'],
            vault_token=vault_config['vault_token'],
            key_name=vault_config['key_name'],
            mount_path=vault_config.get('mount_path', 'transit')
        )
    except Exception as e:
        return [SigningResult(
            success=False,
            file_path=package_dir,
            error_message=f"Failed to initialize Vault client: {e}"
        )]
    
    results = []
    
    # Find metadata files to sign
    metadata_files = list(package_path.glob('*_metadata.json'))
    
    if not metadata_files:
        return [SigningResult(
            success=False,
            file_path=package_dir,
            error_message="No metadata files found to sign"
        )]
    
    logger.info(f"Signing {len(metadata_files)} metadata files in package: {package_dir}")
    
    for metadata_file in metadata_files:
        result = client.sign_file(str(metadata_file))
        results.append(result)
        
        if result.success:
            logger.info(f"✓ Signed: {metadata_file.name}")
        else:
            logger.error(f"✗ Failed to sign: {metadata_file.name} - {result.error_message}")
    
    return results


def verify_evidence_signatures(package_dir: str, vault_config: Dict[str, str]) -> List[VerificationResult]:
    """Verify all signatures in an evidence package.
    
    Args:
        package_dir: Directory containing evidence package
        vault_config: Vault configuration dictionary
        
    Returns:
        List of VerificationResult for each verification
    """
    package_path = Path(package_dir)
    
    if not package_path.exists() or not package_path.is_dir():
        return [VerificationResult(
            success=False,
            file_path=package_dir,
            signature_path="",
            is_valid=False,
            error_message=f"Package directory not found: {package_dir}"
        )]
    
    try:
        client = VaultTransitClient(
            vault_url=vault_config['vault_url'],
            vault_token=vault_config['vault_token'],
            key_name=vault_config['key_name'],
            mount_path=vault_config.get('mount_path', 'transit')
        )
    except Exception as e:
        return [VerificationResult(
            success=False,
            file_path=package_dir,
            signature_path="",
            is_valid=False,
            error_message=f"Failed to initialize Vault client: {e}"
        )]
    
    results = []
    
    # Find signature files
    signature_files = list(package_path.glob('*.sig'))
    
    if not signature_files:
        return [VerificationResult(
            success=False,
            file_path=package_dir,
            signature_path="",
            is_valid=False,
            error_message="No signature files found to verify"
        )]
    
    logger.info(f"Verifying {len(signature_files)} signatures in package: {package_dir}")
    
    for signature_file in signature_files:
        # Determine original file path
        original_file = signature_file.with_suffix('')
        if original_file.suffix == '.sig':  # Handle double extension
            original_file = original_file.with_suffix('')
        
        if not original_file.exists():
            result = VerificationResult(
                success=False,
                file_path=str(original_file),
                signature_path=str(signature_file),
                is_valid=False,
                error_message=f"Original file not found: {original_file}"
            )
        else:
            result = client.verify_file_signature(str(original_file), str(signature_file))
        
        results.append(result)
        
        if result.success and result.is_valid:
            logger.info(f"✓ Verified: {signature_file.name}")
        elif result.success and not result.is_valid:
            logger.error(f"✗ Invalid signature: {signature_file.name}")
        else:
            logger.error(f"✗ Failed to verify: {signature_file.name} - {result.error_message}")
    
    return results


def main():
    """CLI interface for Vault Transit signing."""
    parser = argparse.ArgumentParser(description='Vault Transit integration for evidence signing')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Sign command
    sign_parser = subparsers.add_parser('sign', help='Sign file or package')
    sign_parser.add_argument('path', help='File or package directory to sign')
    sign_parser.add_argument('--vault-url', required=True, help='Vault server URL')
    sign_parser.add_argument('--vault-token', required=True, help='Vault authentication token')
    sign_parser.add_argument('--key-name', required=True, help='Vault Transit key name')
    sign_parser.add_argument('--mount-path', default='transit', help='Transit mount path')
    sign_parser.add_argument('--output', help='Output signature file path (for single file)')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify signature')
    verify_parser.add_argument('path', help='File or package directory to verify')
    verify_parser.add_argument('--signature', help='Signature file path (for single file)')
    verify_parser.add_argument('--vault-url', required=True, help='Vault server URL')
    verify_parser.add_argument('--vault-token', required=True, help='Vault authentication token')
    verify_parser.add_argument('--key-name', required=True, help='Vault Transit key name')
    verify_parser.add_argument('--mount-path', default='transit', help='Transit mount path')
    
    # Package commands
    package_sign_parser = subparsers.add_parser('package-sign', help='Sign entire evidence package')
    package_sign_parser.add_argument('package_dir', help='Evidence package directory')
    package_sign_parser.add_argument('--config', required=True, help='Vault configuration JSON file')
    
    package_verify_parser = subparsers.add_parser('package-verify', help='Verify all signatures in package')
    package_verify_parser.add_argument('package_dir', help='Evidence package directory')
    package_verify_parser.add_argument('--config', required=True, help='Vault configuration JSON file')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output-file', help='Output results to JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    try:
        results = []
        
        if args.command in ['sign', 'verify']:
            # Single file or package operations
            vault_config = {
                'vault_url': args.vault_url,
                'vault_token': args.vault_token,
                'key_name': args.key_name,
                'mount_path': args.mount_path
            }
            
            path = Path(args.path)
            
            if args.command == 'sign':
                if path.is_dir():
                    results = sign_evidence_package(str(path), vault_config)
                else:
                    client = VaultTransitClient(**vault_config)
                    result = client.sign_file(str(path), args.output)
                    results = [result]
            
            elif args.command == 'verify':
                if path.is_dir():
                    results = verify_evidence_signatures(str(path), vault_config)
                else:
                    if not args.signature:
                        signature_path = str(path) + '.sig'
                    else:
                        signature_path = args.signature
                    
                    client = VaultTransitClient(**vault_config)
                    result = client.verify_file_signature(str(path), signature_path)
                    results = [result]
        
        elif args.command in ['package-sign', 'package-verify']:
            # Load configuration from file
            with open(args.config, 'r') as f:
                vault_config = json.load(f)
            
            if args.command == 'package-sign':
                results = sign_evidence_package(args.package_dir, vault_config)
            elif args.command == 'package-verify':
                results = verify_evidence_signatures(args.package_dir, vault_config)
        
        # Output results
        results_dict = {
            "command": args.command,
            "timestamp": datetime.utcnow().isoformat(),
            "results": [result.to_dict() for result in results]
        }
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Results saved to: {args.output_file}")
        else:
            print(json.dumps(results_dict, indent=2))
        
        # Exit with appropriate code
        if args.command in ['sign', 'package-sign']:
            success_count = sum(1 for result in results if result.success)
        else:  # verify commands
            success_count = sum(1 for result in results if result.success and result.is_valid)
        
        if success_count == len(results):
            print(f"\n✓ All {len(results)} operations completed successfully")
            sys.exit(0)
        else:
            print(f"\n✗ {len(results) - success_count} of {len(results)} operations failed")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()