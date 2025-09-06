#!/usr/bin/env python3
"""OpenTimestamps integration for evidence timestamping.

This script provides functionality to create and verify OpenTimestamps
for evidence packages, ensuring temporal proof of existence.
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TimestampResult:
    """Result of timestamping operation."""
    success: bool
    file_path: str
    ots_path: Optional[str] = None
    timestamp_time: Optional[str] = None
    error_message: Optional[str] = None
    verification_status: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "ots_path": self.ots_path,
            "timestamp_time": self.timestamp_time,
            "error_message": self.error_message,
            "verification_status": self.verification_status
        }


class OTSTimestamper:
    """OpenTimestamps client wrapper."""
    
    def __init__(self, ots_cli_path: Optional[str] = None):
        """Initialize OTS timestamper.
        
        Args:
            ots_cli_path: Path to ots CLI binary. If None, assumes 'ots' is in PATH.
        """
        self.ots_cli_path = ots_cli_path or 'ots'
        self._check_ots_availability()
    
    def _check_ots_availability(self) -> None:
        """Check if OTS CLI is available."""
        try:
            result = subprocess.run(
                [self.ots_cli_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"OpenTimestamps CLI found: {result.stdout.strip()}")
            else:
                raise RuntimeError(f"OTS CLI check failed: {result.stderr}")
                
        except FileNotFoundError:
            raise RuntimeError(
                f"OpenTimestamps CLI not found at: {self.ots_cli_path}. "
                "Please install from https://github.com/opentimestamps/opentimestamps-client"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("OTS CLI check timed out")
    
    def stamp_file(self, file_path: str, output_dir: Optional[str] = None) -> TimestampResult:
        """Create OpenTimestamp for a file.
        
        Args:
            file_path: Path to file to timestamp
            output_dir: Directory to save .ots file. If None, saves next to original file.
            
        Returns:
            TimestampResult with operation status
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return TimestampResult(
                success=False,
                file_path=file_path,
                error_message=f"File not found: {file_path}"
            )
        
        # Determine output path
        if output_dir:
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)
            ots_path = output_dir_obj / f"{file_path_obj.name}.ots"
        else:
            ots_path = file_path_obj.with_suffix(file_path_obj.suffix + '.ots')
        
        try:
            logger.info(f"Creating timestamp for: {file_path}")
            
            # Run ots stamp command
            cmd = [self.ots_cli_path, 'stamp', str(file_path_obj)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=file_path_obj.parent
            )
            
            if result.returncode == 0:
                # Check if .ots file was created
                default_ots_path = file_path_obj.with_suffix(file_path_obj.suffix + '.ots')
                
                if default_ots_path.exists():
                    # Move to desired location if different
                    if ots_path != default_ots_path:
                        default_ots_path.rename(ots_path)
                    
                    logger.info(f"Timestamp created: {ots_path}")
                    
                    return TimestampResult(
                        success=True,
                        file_path=file_path,
                        ots_path=str(ots_path),
                        timestamp_time=datetime.utcnow().isoformat()
                    )
                else:
                    return TimestampResult(
                        success=False,
                        file_path=file_path,
                        error_message="OTS file was not created"
                    )
            else:
                return TimestampResult(
                    success=False,
                    file_path=file_path,
                    error_message=f"OTS stamp failed: {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            return TimestampResult(
                success=False,
                file_path=file_path,
                error_message="OTS stamp operation timed out"
            )
        except Exception as e:
            return TimestampResult(
                success=False,
                file_path=file_path,
                error_message=f"OTS stamp failed: {str(e)}"
            )
    
    def verify_timestamp(self, ots_path: str, original_file: Optional[str] = None) -> TimestampResult:
        """Verify an OpenTimestamp.
        
        Args:
            ots_path: Path to .ots file
            original_file: Path to original file. If None, inferred from .ots filename.
            
        Returns:
            TimestampResult with verification status
        """
        ots_path_obj = Path(ots_path)
        
        if not ots_path_obj.exists():
            return TimestampResult(
                success=False,
                file_path=ots_path,
                error_message=f"OTS file not found: {ots_path}"
            )
        
        # Determine original file path
        if original_file:
            original_file_obj = Path(original_file)
        else:
            # Remove .ots extension
            original_file_obj = ots_path_obj.with_suffix('')
            if original_file_obj.suffix == '.ots':  # Handle double extension
                original_file_obj = original_file_obj.with_suffix('')
        
        if not original_file_obj.exists():
            return TimestampResult(
                success=False,
                file_path=str(original_file_obj),
                ots_path=ots_path,
                error_message=f"Original file not found: {original_file_obj}"
            )
        
        try:
            logger.info(f"Verifying timestamp: {ots_path}")
            
            # Run ots verify command
            cmd = [self.ots_cli_path, 'verify', str(ots_path_obj), str(original_file_obj)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Parse verification output
                output = result.stdout.strip()
                
                # Look for timestamp information
                verification_status = "verified"
                if "Success!" in output:
                    verification_status = "verified_success"
                elif "pending" in output.lower():
                    verification_status = "pending"
                elif "incomplete" in output.lower():
                    verification_status = "incomplete"
                
                logger.info(f"Timestamp verification: {verification_status}")
                
                return TimestampResult(
                    success=True,
                    file_path=str(original_file_obj),
                    ots_path=ots_path,
                    verification_status=verification_status,
                    timestamp_time=datetime.utcnow().isoformat()
                )
            else:
                return TimestampResult(
                    success=False,
                    file_path=str(original_file_obj),
                    ots_path=ots_path,
                    error_message=f"OTS verify failed: {result.stderr}",
                    verification_status="failed"
                )
                
        except subprocess.TimeoutExpired:
            return TimestampResult(
                success=False,
                file_path=str(original_file_obj),
                ots_path=ots_path,
                error_message="OTS verify operation timed out"
            )
        except Exception as e:
            return TimestampResult(
                success=False,
                file_path=str(original_file_obj),
                ots_path=ots_path,
                error_message=f"OTS verify failed: {str(e)}"
            )
    
    def upgrade_timestamp(self, ots_path: str) -> TimestampResult:
        """Upgrade an OpenTimestamp (download newer proofs).
        
        Args:
            ots_path: Path to .ots file
            
        Returns:
            TimestampResult with upgrade status
        """
        ots_path_obj = Path(ots_path)
        
        if not ots_path_obj.exists():
            return TimestampResult(
                success=False,
                file_path=ots_path,
                error_message=f"OTS file not found: {ots_path}"
            )
        
        try:
            logger.info(f"Upgrading timestamp: {ots_path}")
            
            # Run ots upgrade command
            cmd = [self.ots_cli_path, 'upgrade', str(ots_path_obj)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # Upgrade can take longer
            )
            
            if result.returncode == 0:
                logger.info(f"Timestamp upgraded: {ots_path}")
                
                return TimestampResult(
                    success=True,
                    file_path=ots_path,
                    ots_path=ots_path,
                    timestamp_time=datetime.utcnow().isoformat()
                )
            else:
                return TimestampResult(
                    success=False,
                    file_path=ots_path,
                    error_message=f"OTS upgrade failed: {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            return TimestampResult(
                success=False,
                file_path=ots_path,
                error_message="OTS upgrade operation timed out"
            )
        except Exception as e:
            return TimestampResult(
                success=False,
                file_path=ots_path,
                error_message=f"OTS upgrade failed: {str(e)}"
            )
    
    def info_timestamp(self, ots_path: str) -> Dict[str, Any]:
        """Get information about an OpenTimestamp.
        
        Args:
            ots_path: Path to .ots file
            
        Returns:
            Dictionary with timestamp information
        """
        ots_path_obj = Path(ots_path)
        
        if not ots_path_obj.exists():
            return {
                "error": f"OTS file not found: {ots_path}"
            }
        
        try:
            logger.info(f"Getting timestamp info: {ots_path}")
            
            # Run ots info command
            cmd = [self.ots_cli_path, 'info', str(ots_path_obj)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "info": result.stdout.strip(),
                    "file_path": ots_path
                }
            else:
                return {
                    "success": False,
                    "error": f"OTS info failed: {result.stderr}",
                    "file_path": ots_path
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "OTS info operation timed out",
                "file_path": ots_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"OTS info failed: {str(e)}",
                "file_path": ots_path
            }


def timestamp_evidence_package(package_dir: str, ots_cli_path: Optional[str] = None) -> List[TimestampResult]:
    """Timestamp all files in an evidence package.
    
    Args:
        package_dir: Directory containing evidence package
        ots_cli_path: Path to OTS CLI binary
        
    Returns:
        List of TimestampResult for each file
    """
    package_path = Path(package_dir)
    
    if not package_path.exists() or not package_path.is_dir():
        return [TimestampResult(
            success=False,
            file_path=package_dir,
            error_message=f"Package directory not found: {package_dir}"
        )]
    
    timestamper = OTSTimestamper(ots_cli_path)
    results = []
    
    # Find evidence files to timestamp
    evidence_files = []
    
    # Look for metadata files (primary target)
    for metadata_file in package_path.glob('*_metadata.json'):
        evidence_files.append(metadata_file)
    
    # Also timestamp video files
    for video_file in package_path.glob('*.mp4'):
        evidence_files.append(video_file)
    
    # Timestamp manifest if present
    for manifest_file in package_path.glob('*_manifest.sha256'):
        evidence_files.append(manifest_file)
    
    if not evidence_files:
        return [TimestampResult(
            success=False,
            file_path=package_dir,
            error_message="No evidence files found to timestamp"
        )]
    
    logger.info(f"Timestamping {len(evidence_files)} files in package: {package_dir}")
    
    for file_path in evidence_files:
        result = timestamper.stamp_file(str(file_path))
        results.append(result)
        
        if result.success:
            logger.info(f"✓ Timestamped: {file_path.name}")
        else:
            logger.error(f"✗ Failed to timestamp: {file_path.name} - {result.error_message}")
    
    return results


def verify_evidence_timestamps(package_dir: str, ots_cli_path: Optional[str] = None) -> List[TimestampResult]:
    """Verify all timestamps in an evidence package.
    
    Args:
        package_dir: Directory containing evidence package
        ots_cli_path: Path to OTS CLI binary
        
    Returns:
        List of TimestampResult for each verification
    """
    package_path = Path(package_dir)
    
    if not package_path.exists() or not package_path.is_dir():
        return [TimestampResult(
            success=False,
            file_path=package_dir,
            error_message=f"Package directory not found: {package_dir}"
        )]
    
    timestamper = OTSTimestamper(ots_cli_path)
    results = []
    
    # Find .ots files
    ots_files = list(package_path.glob('*.ots'))
    
    if not ots_files:
        return [TimestampResult(
            success=False,
            file_path=package_dir,
            error_message="No .ots files found to verify"
        )]
    
    logger.info(f"Verifying {len(ots_files)} timestamps in package: {package_dir}")
    
    for ots_file in ots_files:
        result = timestamper.verify_timestamp(str(ots_file))
        results.append(result)
        
        if result.success:
            logger.info(f"✓ Verified: {ots_file.name} - {result.verification_status}")
        else:
            logger.error(f"✗ Failed to verify: {ots_file.name} - {result.error_message}")
    
    return results


def main():
    """CLI interface for OTS timestamping."""
    parser = argparse.ArgumentParser(description='OpenTimestamps integration for evidence packages')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stamp command
    stamp_parser = subparsers.add_parser('stamp', help='Create timestamp for file or package')
    stamp_parser.add_argument('path', help='File or package directory to timestamp')
    stamp_parser.add_argument('--ots-cli', help='Path to OTS CLI binary')
    stamp_parser.add_argument('--output-dir', help='Directory to save .ots files')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify timestamp')
    verify_parser.add_argument('path', help='OTS file or package directory to verify')
    verify_parser.add_argument('--original', help='Original file (if not inferred from .ots name)')
    verify_parser.add_argument('--ots-cli', help='Path to OTS CLI binary')
    
    # Upgrade command
    upgrade_parser = subparsers.add_parser('upgrade', help='Upgrade timestamp')
    upgrade_parser.add_argument('ots_file', help='OTS file to upgrade')
    upgrade_parser.add_argument('--ots-cli', help='Path to OTS CLI binary')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get timestamp information')
    info_parser.add_argument('ots_file', help='OTS file to get info for')
    info_parser.add_argument('--ots-cli', help='Path to OTS CLI binary')
    
    # Package commands
    package_stamp_parser = subparsers.add_parser('package-stamp', help='Timestamp entire evidence package')
    package_stamp_parser.add_argument('package_dir', help='Evidence package directory')
    package_stamp_parser.add_argument('--ots-cli', help='Path to OTS CLI binary')
    
    package_verify_parser = subparsers.add_parser('package-verify', help='Verify all timestamps in package')
    package_verify_parser.add_argument('package_dir', help='Evidence package directory')
    package_verify_parser.add_argument('--ots-cli', help='Path to OTS CLI binary')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', help='Output results to JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    try:
        results = []
        
        if args.command == 'stamp':
            path = Path(args.path)
            if path.is_dir():
                results = timestamp_evidence_package(str(path), args.ots_cli)
            else:
                timestamper = OTSTimestamper(args.ots_cli)
                result = timestamper.stamp_file(str(path), args.output_dir)
                results = [result]
        
        elif args.command == 'verify':
            path = Path(args.path)
            if path.is_dir():
                results = verify_evidence_timestamps(str(path), args.ots_cli)
            else:
                timestamper = OTSTimestamper(args.ots_cli)
                result = timestamper.verify_timestamp(str(path), args.original)
                results = [result]
        
        elif args.command == 'upgrade':
            timestamper = OTSTimestamper(args.ots_cli)
            result = timestamper.upgrade_timestamp(args.ots_file)
            results = [result]
        
        elif args.command == 'info':
            timestamper = OTSTimestamper(args.ots_cli)
            info = timestamper.info_timestamp(args.ots_file)
            print(json.dumps(info, indent=2))
            sys.exit(0 if info.get('success', False) else 1)
        
        elif args.command == 'package-stamp':
            results = timestamp_evidence_package(args.package_dir, args.ots_cli)
        
        elif args.command == 'package-verify':
            results = verify_evidence_timestamps(args.package_dir, args.ots_cli)
        
        # Output results
        results_dict = {
            "command": args.command,
            "timestamp": datetime.utcnow().isoformat(),
            "results": [result.to_dict() for result in results]
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print(json.dumps(results_dict, indent=2))
        
        # Exit with appropriate code
        success_count = sum(1 for result in results if result.success)
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