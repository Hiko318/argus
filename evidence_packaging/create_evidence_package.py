#!/usr/bin/env python3
"""
Evidence Package Creator for Foresight SAR System

This script creates legally compliant evidence packages with:
- HashiCorp Vault Transit Engine integration for digital signatures
- OpenTimestamps (OTS) for blockchain-based timestamping
- Chain of custody documentation
- Comprehensive integrity verification
- Legal compliance features

Usage:
    python create_evidence_package.py --case-id SAR-2024-001 --video stream.mp4 --operator OFFICER-123
    python create_evidence_package.py --config evidence_config.json --batch-mode
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# Configure logging
logger = logging.getLogger(__name__)

# Import our evidence packager
from evidence_packager import (
    EvidencePackager, EvidenceMetadata, ChainOfCustodyEntry,
    DigitalSignature, TimestampProof
)

try:
    import hvac  # HashiCorp Vault client
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    logging.warning("HashiCorp Vault client not available. Install with: pip install hvac")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests not available for OTS integration")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VaultTransitSigner:
    """HashiCorp Vault Transit Engine integration for digital signatures"""
    
    def __init__(self, vault_url: str, vault_token: str, transit_key: str):
        self.vault_url = vault_url
        self.vault_token = vault_token
        self.transit_key = transit_key
        self.client = None
        
        if VAULT_AVAILABLE:
            self._initialize_vault_client()
    
    def _initialize_vault_client(self):
        """Initialize Vault client and verify connection"""
        try:
            self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
            
            # Verify client is authenticated
            if not self.client.is_authenticated():
                raise Exception("Vault authentication failed")
            
            # Verify transit engine is enabled
            if 'transit/' not in self.client.sys.list_auth_methods():
                logger.warning("Transit engine may not be enabled")
            
            logger.info("Vault client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {e}")
            self.client = None
    
    def sign_data(self, data: bytes, context: str = None) -> Optional[Dict[str, Any]]:
        """Sign data using Vault Transit engine"""
        if not self.client:
            logger.error("Vault client not available")
            return None
        
        try:
            # Prepare signing request
            sign_data = {
                'input': data.hex(),
                'key_version': 'latest'
            }
            
            if context:
                sign_data['context'] = context
            
            # Sign data using Vault Transit
            response = self.client.secrets.transit.sign_data(
                name=self.transit_key,
                **sign_data
            )
            
            signature_data = response['data']
            
            return {
                'algorithm': 'vault-transit',
                'signature': signature_data['signature'],
                'key_version': signature_data.get('key_version'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signer_id': 'vault-transit',
                'vault_key': self.transit_key,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Vault signing failed: {e}")
            return None
    
    def verify_signature(self, data: bytes, signature: str, context: str = None) -> bool:
        """Verify signature using Vault Transit engine"""
        if not self.client:
            logger.error("Vault client not available")
            return False
        
        try:
            verify_data = {
                'input': data.hex(),
                'signature': signature
            }
            
            if context:
                verify_data['context'] = context
            
            response = self.client.secrets.transit.verify_signed_data(
                name=self.transit_key,
                **verify_data
            )
            
            return response['data']['valid']
            
        except Exception as e:
            logger.error(f"Vault signature verification failed: {e}")
            return False

class OpenTimestampsIntegration:
    """OpenTimestamps integration for blockchain timestamping"""
    
    def __init__(self):
        self.calendar_servers = [
            "https://alice.btc.calendar.opentimestamps.org",
            "https://bob.btc.calendar.opentimestamps.org",
            "https://finney.calendar.eternitywall.com"
        ]
    
    def create_timestamp(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Create OpenTimestamps proof for file hash"""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests not available for OTS integration")
            return None
        
        try:
            # Submit to calendar servers
            submitted_servers = []
            
            for server in self.calendar_servers:
                try:
                    response = requests.post(
                        f"{server}/digest",
                        data=bytes.fromhex(file_hash),
                        headers={'Content-Type': 'application/octet-stream'},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        submitted_servers.append(server)
                        logger.info(f"Submitted to OTS calendar: {server}")
                    
                except requests.RequestException as e:
                    logger.warning(f"Failed to submit to {server}: {e}")
            
            if submitted_servers:
                return {
                    'file_hash': file_hash,
                    'submitted_servers': submitted_servers,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'verification_url': 'https://opentimestamps.org',
                    'status': 'submitted'
                }
            else:
                logger.error("Failed to submit to any OTS calendar servers")
                return None
                
        except Exception as e:
            logger.error(f"OTS timestamp creation failed: {e}")
            return None

class EnhancedEvidencePackager(EvidencePackager):
    """Enhanced evidence packager with Vault and OTS integration"""
    
    def __init__(self, vault_config: Dict[str, str] = None, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize Vault signer if config provided
        self.vault_signer = None
        if vault_config and VAULT_AVAILABLE:
            self.vault_signer = VaultTransitSigner(
                vault_url=vault_config.get('url'),
                vault_token=vault_config.get('token'),
                transit_key=vault_config.get('key_name')
            )
        
        # Initialize OTS integration
        self.ots_integration = OpenTimestampsIntegration()
    
    def create_enhanced_evidence_package(self,
                                       metadata: EvidenceMetadata,
                                       video_files: List[str] = None,
                                       image_files: List[str] = None,
                                       data_files: List[str] = None,
                                       chain_entries: List[ChainOfCustodyEntry] = None,
                                       output_path: str = None,
                                       vault_context: str = None) -> Dict[str, Any]:
        """Create evidence package with enhanced security features"""
        
        # Create base evidence package
        result = self.create_evidence_package(
            metadata=metadata,
            video_files=video_files,
            image_files=image_files,
            data_files=data_files,
            chain_entries=chain_entries,
            output_path=output_path
        )
        
        if not result['success']:
            return result
        
        package_dir = Path(result['package_path'])
        enhanced_features = {
            'vault_signatures': [],
            'ots_timestamps': [],
            'security_level': 'enhanced'
        }
        
        try:
            # Create Vault signatures for critical files
            critical_files = ['metadata.json', 'manifest.sha256']
            
            for file_name in critical_files:
                file_path = package_dir / file_name
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    # Create Vault signature
                    if self.vault_signer:
                        vault_sig = self.vault_signer.sign_data(
                            file_data, 
                            context=vault_context or f"{metadata.case_id}_{file_name}"
                        )
                        
                        if vault_sig:
                            # Save signature
                            sig_path = package_dir / f"{file_name}.vault.sig"
                            with open(sig_path, 'w') as f:
                                json.dump(vault_sig, f, indent=2)
                            
                            enhanced_features['vault_signatures'].append({
                                'file': file_name,
                                'signature_file': str(sig_path),
                                'algorithm': vault_sig['algorithm'],
                                'timestamp': vault_sig['timestamp']
                            })
                            
                            logger.info(f"Created Vault signature for {file_name}")
                    
                    # Create OTS timestamp
                    file_hash = self._calculate_file_hash(file_path, 'sha256')
                    ots_proof = self.ots_integration.create_timestamp(file_hash)
                    
                    if ots_proof:
                        # Save OTS proof
                        ots_path = package_dir / f"{file_name}.ots.json"
                        with open(ots_path, 'w') as f:
                            json.dump(ots_proof, f, indent=2)
                        
                        enhanced_features['ots_timestamps'].append({
                            'file': file_name,
                            'timestamp_file': str(ots_path),
                            'file_hash': file_hash,
                            'created_at': ots_proof['created_at']
                        })
                        
                        logger.info(f"Created OTS timestamp for {file_name}")
            
            # Create enhanced security manifest
            security_manifest = {
                'package_id': result['package_summary']['package_id'],
                'security_features': enhanced_features,
                'vault_integration': self.vault_signer is not None,
                'ots_integration': True,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'compliance_level': 'legal_evidence',
                'verification_instructions': {
                    'vault_signatures': 'Use HashiCorp Vault Transit engine to verify signatures',
                    'ots_timestamps': 'Use OpenTimestamps client or web interface to verify timestamps',
                    'manifest_integrity': 'Verify SHA256 hashes in manifest.sha256'
                }
            }
            
            security_manifest_path = package_dir / 'security_manifest.json'
            with open(security_manifest_path, 'w') as f:
                json.dump(security_manifest, f, indent=2)
            
            # Update result with enhanced features
            result['enhanced_features'] = enhanced_features
            result['security_manifest'] = security_manifest
            result['vault_signatures_count'] = len(enhanced_features['vault_signatures'])
            result['ots_timestamps_count'] = len(enhanced_features['ots_timestamps'])
            
            logger.info("Enhanced evidence package created successfully")
            logger.info(f"Vault signatures: {len(enhanced_features['vault_signatures'])}")
            logger.info(f"OTS timestamps: {len(enhanced_features['ots_timestamps'])}")
            
        except Exception as e:
            logger.error(f"Enhanced features creation failed: {e}")
            result['enhanced_features_error'] = str(e)
        
        return result
    
    def verify_enhanced_package(self, package_path: str, vault_config: Dict[str, str] = None) -> Dict[str, Any]:
        """Verify enhanced evidence package with Vault and OTS verification"""
        
        # Start with base verification
        result = self.verify_evidence_package(package_path)
        
        package_dir = Path(package_path)
        enhanced_checks = {
            'vault_signatures_verified': [],
            'ots_timestamps_verified': [],
            'security_manifest_present': False
        }
        
        try:
            # Check for security manifest
            security_manifest_path = package_dir / 'security_manifest.json'
            if security_manifest_path.exists():
                enhanced_checks['security_manifest_present'] = True
                
                with open(security_manifest_path, 'r') as f:
                    security_manifest = json.load(f)
                
                # Verify Vault signatures if Vault is available
                if vault_config and VAULT_AVAILABLE:
                    vault_signer = VaultTransitSigner(
                        vault_url=vault_config.get('url'),
                        vault_token=vault_config.get('token'),
                        transit_key=vault_config.get('key_name')
                    )
                    
                    for sig_info in security_manifest['security_features']['vault_signatures']:
                        file_name = sig_info['file']
                        sig_file = sig_info['signature_file']
                        
                        # Load original file and signature
                        original_file = package_dir / file_name
                        signature_file = Path(sig_file)
                        
                        if original_file.exists() and signature_file.exists():
                            with open(original_file, 'rb') as f:
                                file_data = f.read()
                            
                            with open(signature_file, 'r') as f:
                                sig_data = json.load(f)
                            
                            # Verify signature
                            is_valid = vault_signer.verify_signature(
                                file_data,
                                sig_data['signature'],
                                context=sig_data.get('context')
                            )
                            
                            enhanced_checks['vault_signatures_verified'].append({
                                'file': file_name,
                                'valid': is_valid
                            })
                
                # Note: OTS verification would require the OTS client
                # For now, we just check that timestamp files exist
                for ots_info in security_manifest['security_features']['ots_timestamps']:
                    timestamp_file = Path(ots_info['timestamp_file'])
                    enhanced_checks['ots_timestamps_verified'].append({
                        'file': ots_info['file'],
                        'timestamp_file_exists': timestamp_file.exists(),
                        'note': 'Full OTS verification requires OpenTimestamps client'
                    })
        
        except Exception as e:
            result['enhanced_verification_error'] = str(e)
        
        result['enhanced_checks'] = enhanced_checks
        return result

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}

def create_default_config() -> Dict[str, Any]:
    """Create default configuration template"""
    return {
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
            "include_timestamps": True,
            "compress_package": False
        }
    }

def main():
    parser = argparse.ArgumentParser(
        description='Create legally compliant evidence packages for SAR operations'
    )
    
    # Required arguments
    parser.add_argument('--case-id', required=True,
                       help='Case ID for the evidence package')
    parser.add_argument('--operator', required=True,
                       help='Operator ID creating the evidence')
    
    # Evidence files
    parser.add_argument('--video', action='append',
                       help='Video evidence files (can be specified multiple times)')
    parser.add_argument('--image', action='append',
                       help='Image evidence files (can be specified multiple times)')
    parser.add_argument('--data', action='append',
                       help='Data evidence files (can be specified multiple times)')
    
    # Metadata
    parser.add_argument('--agency', default='Search and Rescue Division',
                       help='Agency conducting the operation')
    parser.add_argument('--incident-type', default='Search and Rescue',
                       help='Type of incident')
    parser.add_argument('--location', required=True,
                       help='Location of the operation')
    parser.add_argument('--equipment',
                       help='Equipment used in the operation')
    parser.add_argument('--weather',
                       help='Weather conditions during operation')
    parser.add_argument('--notes',
                       help='Additional notes')
    
    # Configuration
    parser.add_argument('--config',
                       help='Configuration file path')
    parser.add_argument('--output', '-o',
                       help='Output directory for evidence package')
    parser.add_argument('--vault-url',
                       help='HashiCorp Vault URL')
    parser.add_argument('--vault-token',
                       help='HashiCorp Vault token')
    parser.add_argument('--vault-key',
                       help='Vault Transit key name')
    
    # Options
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file')
    parser.add_argument('--verify',
                       help='Verify existing evidence package')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        config_path = 'evidence_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Default configuration created: {config_path}")
        return
    
    # Verify package if requested
    if args.verify:
        packager = EnhancedEvidencePackager()
        result = packager.verify_enhanced_package(args.verify)
        print("Verification Results:")
        print(json.dumps(result, indent=2, default=str))
        return
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Set up Vault configuration
    vault_config = None
    if args.vault_url or config.get('vault'):
        vault_config = {
            'url': args.vault_url or config.get('vault', {}).get('url'),
            'token': args.vault_token or config.get('vault', {}).get('token'),
            'key_name': args.vault_key or config.get('vault', {}).get('key_name')
        }
        
        if not all(vault_config.values()):
            logger.warning("Incomplete Vault configuration - signatures will be disabled")
            vault_config = None
    
    # Create evidence metadata
    current_time = datetime.now(timezone.utc).isoformat()
    
    metadata = EvidenceMetadata(
        case_id=args.case_id,
        operator_id=args.operator,
        agency=args.agency,
        incident_type=args.incident_type,
        location=args.location,
        start_time=current_time,
        end_time=current_time,
        equipment_used=args.equipment or "Foresight SAR System",
        weather_conditions=args.weather or "Not specified",
        legal_authority="Emergency Response Authorization",
        chain_of_custody_officer=args.operator,
        notes=args.notes or ""
    )
    
    # Create chain of custody entry
    chain_entries = [
        ChainOfCustodyEntry(
            timestamp=current_time,
            action="Evidence Package Created",
            officer_id=args.operator,
            officer_name=args.operator,
            location=args.location,
            notes="Automated evidence package creation via Foresight SAR System"
        )
    ]
    
    # Create enhanced evidence packager
    packager = EnhancedEvidencePackager(vault_config=vault_config)
    
    # Create evidence package
    result = packager.create_enhanced_evidence_package(
        metadata=metadata,
        video_files=args.video or [],
        image_files=args.image or [],
        data_files=args.data or [],
        chain_entries=chain_entries,
        output_path=args.output,
        vault_context=f"{args.case_id}_evidence"
    )
    
    # Output results
    if result['success']:
        print("Evidence package created successfully!")
        print(f"Package location: {result['package_path']}")
        print(f"Total files: {result['package_summary']['total_files']}")
        print(f"Package size: {result['package_summary']['package_size_bytes']} bytes")
        
        if 'vault_signatures_count' in result:
            print(f"Vault signatures: {result['vault_signatures_count']}")
        if 'ots_timestamps_count' in result:
            print(f"OTS timestamps: {result['ots_timestamps_count']}")
        
        if args.verbose:
            print("\nDetailed results:")
            print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Evidence package creation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == '__main__':
    main()