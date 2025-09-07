#!/usr/bin/env python3
"""
OpenTimestamps Integration for Foresight Evidence Anchoring

This module provides blockchain timestamping capabilities using OpenTimestamps
to create tamper-evident timestamps for evidence packages.
"""

import os
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenTimestampsClient:
    """OpenTimestamps client for evidence timestamping."""
    
    def __init__(self, ots_cli_path: str = None):
        """Initialize OpenTimestamps client.
        
        Args:
            ots_cli_path: Path to ots CLI tool. If None, assumes 'ots' is in PATH.
        """
        self.ots_cli = ots_cli_path or 'ots'
        self._verify_ots_installation()
    
    def _verify_ots_installation(self) -> None:
        """Verify OpenTimestamps CLI is installed and accessible."""
        try:
            result = subprocess.run(
                [self.ots_cli, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"OTS CLI not working: {result.stderr}")
            logger.info(f"OpenTimestamps CLI found: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(
                f"OpenTimestamps CLI not found or not working. "
                f"Install with: pip install opentimestamps-client. Error: {e}"
            )
    
    def create_timestamp(self, file_path: str) -> str:
        """Create OpenTimestamps proof for a file.
        
        Args:
            file_path: Path to file to timestamp
            
        Returns:
            Path to created .ots file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ots_file = file_path.with_suffix(file_path.suffix + '.ots')
        
        try:
            # Create timestamp
            result = subprocess.run(
                [self.ots_cli, 'stamp', str(file_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"OTS stamp failed: {result.stderr}")
            
            if not ots_file.exists():
                raise RuntimeError(f"OTS file not created: {ots_file}")
            
            logger.info(f"Created timestamp: {file_path} -> {ots_file}")
            return str(ots_file)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("OTS stamp operation timed out")
    
    def verify_timestamp(self, file_path: str, ots_file: str = None) -> Tuple[bool, str]:
        """Verify OpenTimestamps proof for a file.
        
        Args:
            file_path: Path to original file
            ots_file: Path to .ots file (optional, will be inferred if not provided)
            
        Returns:
            Tuple of (is_valid, verification_info)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if ots_file is None:
            ots_file = file_path.with_suffix(file_path.suffix + '.ots')
        else:
            ots_file = Path(ots_file)
        
        if not ots_file.exists():
            return False, f"OTS file not found: {ots_file}"
        
        try:
            # Verify timestamp
            result = subprocess.run(
                [self.ots_cli, 'verify', str(ots_file), str(file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            is_valid = result.returncode == 0
            info = result.stdout.strip() if is_valid else result.stderr.strip()
            
            logger.info(f"Timestamp verification: {is_valid} - {info}")
            return is_valid, info
            
        except subprocess.TimeoutExpired:
            return False, "OTS verify operation timed out"
    
    def upgrade_timestamp(self, ots_file: str) -> bool:
        """Upgrade incomplete timestamp to complete proof.
        
        Args:
            ots_file: Path to .ots file to upgrade
            
        Returns:
            True if upgrade successful or already complete
        """
        ots_file = Path(ots_file)
        if not ots_file.exists():
            raise FileNotFoundError(f"OTS file not found: {ots_file}")
        
        try:
            # Upgrade timestamp
            result = subprocess.run(
                [self.ots_cli, 'upgrade', str(ots_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # upgrade returns 0 for success or if already upgraded
            success = result.returncode == 0
            
            if success:
                logger.info(f"Timestamp upgraded: {ots_file}")
            else:
                logger.warning(f"Timestamp upgrade failed: {result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.error("OTS upgrade operation timed out")
            return False
    
    def get_timestamp_info(self, ots_file: str) -> dict:
        """Get information about a timestamp proof.
        
        Args:
            ots_file: Path to .ots file
            
        Returns:
            Dictionary with timestamp information
        """
        ots_file = Path(ots_file)
        if not ots_file.exists():
            raise FileNotFoundError(f"OTS file not found: {ots_file}")
        
        try:
            # Get info about timestamp
            result = subprocess.run(
                [self.ots_cli, 'info', str(ots_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            info = {
                'file': str(ots_file),
                'size': ots_file.stat().st_size,
                'created': ots_file.stat().st_ctime,
                'raw_info': result.stdout.strip() if result.returncode == 0 else result.stderr.strip(),
                'valid': result.returncode == 0
            }
            
            return info
            
        except subprocess.TimeoutExpired:
            return {
                'file': str(ots_file),
                'error': 'OTS info operation timed out'
            }

def timestamp_evidence_manifest(manifest_path: str, output_dir: str = None) -> str:
    """Create OpenTimestamps proof for evidence manifest.
    
    Args:
        manifest_path: Path to manifest.sha256 file
        output_dir: Directory to save .ots file (optional)
        
    Returns:
        Path to created .ots file
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    client = OpenTimestampsClient()
    ots_file = client.create_timestamp(str(manifest_file))
    
    # Move to output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        new_ots_file = output_dir / Path(ots_file).name
        Path(ots_file).rename(new_ots_file)
        ots_file = str(new_ots_file)
    
    return ots_file

def create_evidence_timestamp_bundle(evidence_dir: str) -> dict:
    """Create complete timestamp bundle for evidence package.
    
    Args:
        evidence_dir: Directory containing evidence files
        
    Returns:
        Dictionary with timestamp information
    """
    evidence_dir = Path(evidence_dir)
    if not evidence_dir.exists():
        raise FileNotFoundError(f"Evidence directory not found: {evidence_dir}")
    
    client = OpenTimestampsClient()
    timestamp_info = {
        'created_at': time.time(),
        'evidence_dir': str(evidence_dir),
        'timestamps': {}
    }
    
    # Find manifest file
    manifest_files = list(evidence_dir.glob('manifest.sha256'))
    if not manifest_files:
        raise FileNotFoundError("No manifest.sha256 found in evidence directory")
    
    manifest_file = manifest_files[0]
    
    # Create timestamp for manifest
    try:
        ots_file = client.create_timestamp(str(manifest_file))
        timestamp_info['timestamps']['manifest'] = {
            'file': str(manifest_file),
            'ots_file': ots_file,
            'status': 'created'
        }
        logger.info(f"Evidence manifest timestamped: {ots_file}")
    except Exception as e:
        timestamp_info['timestamps']['manifest'] = {
            'file': str(manifest_file),
            'error': str(e),
            'status': 'failed'
        }
        logger.error(f"Failed to timestamp manifest: {e}")
    
    # Save timestamp info
    info_file = evidence_dir / 'timestamp_info.json'
    with open(info_file, 'w') as f:
        json.dump(timestamp_info, f, indent=2)
    
    return timestamp_info

def main():
    """CLI interface for OpenTimestamps operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenTimestamps Evidence Anchoring')
    parser.add_argument('--stamp', type=str, help='Create timestamp for file')
    parser.add_argument('--verify', type=str, help='Verify timestamp for file')
    parser.add_argument('--upgrade', type=str, help='Upgrade timestamp proof')
    parser.add_argument('--info', type=str, help='Get timestamp info')
    parser.add_argument('--evidence-dir', type=str, help='Create timestamp bundle for evidence directory')
    
    args = parser.parse_args()
    
    try:
        client = OpenTimestampsClient()
        
        if args.stamp:
            ots_file = client.create_timestamp(args.stamp)
            print(f"Timestamp created: {ots_file}")
        
        elif args.verify:
            is_valid, info = client.verify_timestamp(args.verify)
            print(f"Timestamp valid: {is_valid}")
            print(f"Info: {info}")
        
        elif args.upgrade:
            success = client.upgrade_timestamp(args.upgrade)
            print(f"Upgrade successful: {success}")
        
        elif args.info:
            info = client.get_timestamp_info(args.info)
            print(json.dumps(info, indent=2))
        
        elif args.evidence_dir:
            bundle_info = create_evidence_timestamp_bundle(args.evidence_dir)
            print("Evidence timestamp bundle created:")
            print(json.dumps(bundle_info, indent=2))
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())