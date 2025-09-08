#!/usr/bin/env python3
"""
Foresight SAR Data Packager

Packages SAR mission data with metadata and integrity verification.
Generates metadata.json and SHA-256 manifest for data export.

Usage:
    from packager.packager import SARPackager
    
    packager = SARPackager()
    package_path = packager.create_package(
        mission_data=mission_data,
        output_dir="exports",
        package_name="mission_001"
    )

Author: Foresight AI Team
Date: 2024
"""

import json
import hashlib
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)

@dataclass
class MissionMetadata:
    """Mission metadata structure"""
    mission_id: str
    mission_name: str
    start_time: str
    end_time: str
    operator: str
    aircraft_type: str
    sensor_config: Dict[str, Any]
    geolocation_data: Dict[str, Any]
    detection_summary: Dict[str, Any]
    package_version: str = "1.0"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()

@dataclass
class FileManifestEntry:
    """File manifest entry with integrity data"""
    file_path: str
    file_size: int
    sha256_hash: str
    file_type: str
    created_at: str
    description: str = ""

class SARPackager:
    """SAR mission data packager with integrity verification"""
    
    def __init__(self, base_output_dir: str = "exports"):
        """
        Initialize SAR packager
        
        Args:
            base_output_dir: Base directory for package exports
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"SAR Packager initialized - Output dir: {self.base_output_dir}")
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def create_file_manifest(self, files: List[Path], base_dir: Path) -> List[FileManifestEntry]:
        """Create file manifest with integrity hashes"""
        manifest = []
        
        for file_path in files:
            if not file_path.exists() or not file_path.is_file():
                logger.warning(f"Skipping non-existent file: {file_path}")
                continue
            
            try:
                # Calculate relative path from base directory
                rel_path = file_path.relative_to(base_dir)
                
                # Calculate file hash
                file_hash = self.calculate_file_hash(file_path)
                
                # Determine file type
                file_type = self._determine_file_type(file_path)
                
                # Get file stats
                stat = file_path.stat()
                
                entry = FileManifestEntry(
                    file_path=str(rel_path),
                    file_size=stat.st_size,
                    sha256_hash=file_hash,
                    file_type=file_type,
                    created_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    description=f"{file_type} file"
                )
                
                manifest.append(entry)
                logger.debug(f"Added to manifest: {rel_path} ({file_type})")
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
        
        return manifest
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine file type based on extension and content"""
        suffix = file_path.suffix.lower()
        
        type_mapping = {
            '.mp4': 'video',
            '.avi': 'video',
            '.mov': 'video',
            '.mkv': 'video',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.bmp': 'image',
            '.tiff': 'image',
            '.json': 'metadata',
            '.xml': 'metadata',
            '.csv': 'data',
            '.txt': 'text',
            '.log': 'log',
            '.kml': 'geospatial',
            '.gpx': 'geospatial',
            '.shp': 'geospatial'
        }
        
        return type_mapping.get(suffix, 'unknown')
    
    def create_metadata_json(self, metadata: MissionMetadata, manifest: List[FileManifestEntry]) -> Dict[str, Any]:
        """Create complete metadata JSON structure"""
        return {
            "package_info": {
                "format_version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "packager_version": "1.0.0",
                "total_files": len(manifest),
                "total_size_bytes": sum(entry.file_size for entry in manifest)
            },
            "mission_metadata": asdict(metadata),
            "file_manifest": [asdict(entry) for entry in manifest],
            "integrity": {
                "manifest_hash": self._calculate_manifest_hash(manifest),
                "verification_method": "SHA-256"
            }
        }
    
    def _calculate_manifest_hash(self, manifest: List[FileManifestEntry]) -> str:
        """Calculate hash of the entire manifest for integrity verification"""
        # Create deterministic string representation of manifest
        manifest_str = json.dumps(
            [asdict(entry) for entry in manifest],
            sort_keys=True,
            separators=(',', ':')
        )
        
        return hashlib.sha256(manifest_str.encode('utf-8')).hexdigest()
    
    def create_package(self, 
                      mission_data: MissionMetadata,
                      files: List[Union[str, Path]],
                      package_name: str,
                      compress: bool = True) -> Path:
        """
        Create a complete SAR mission package
        
        Args:
            mission_data: Mission metadata
            files: List of files to include in package
            package_name: Name for the package
            compress: Whether to create a compressed archive
            
        Returns:
            Path to created package
        """
        logger.info(f"Creating SAR package: {package_name}")
        
        # Create package directory
        package_dir = self.base_output_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Convert file paths to Path objects
        file_paths = [Path(f) for f in files]
        
        # Copy files to package directory
        copied_files = []
        for file_path in file_paths:
            if file_path.exists():
                dest_path = package_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_files.append(dest_path)
                logger.debug(f"Copied: {file_path} -> {dest_path}")
            else:
                logger.warning(f"File not found, skipping: {file_path}")
        
        # Create file manifest
        manifest = self.create_file_manifest(copied_files, package_dir)
        
        # Create metadata JSON
        metadata_dict = self.create_metadata_json(mission_data, manifest)
        
        # Write metadata.json
        metadata_path = package_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created metadata.json with {len(manifest)} files")
        
        # Create SHA-256 manifest file
        manifest_path = package_dir / "SHA256SUMS"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for entry in manifest:
                f.write(f"{entry.sha256_hash}  {entry.file_path}\n")
            # Add metadata.json hash
            metadata_hash = self.calculate_file_hash(metadata_path)
            f.write(f"{metadata_hash}  metadata.json\n")
        
        logger.info(f"Created SHA256SUMS manifest")
        
        # Optionally create compressed archive
        if compress:
            archive_path = self.base_output_dir / f"{package_name}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir)
                        zipf.write(file_path, arcname)
            
            logger.info(f"Created compressed package: {archive_path}")
            return archive_path
        
        logger.info(f"Created package directory: {package_dir}")
        return package_dir
    
    def verify_package(self, package_path: Path) -> bool:
        """Verify package integrity using SHA-256 manifest"""
        logger.info(f"Verifying package: {package_path}")
        
        # Handle both directory and zip file packages
        if package_path.is_file() and package_path.suffix == '.zip':
            # Extract and verify zip package
            with zipfile.ZipFile(package_path, 'r') as zipf:
                temp_dir = package_path.parent / f"temp_{package_path.stem}"
                zipf.extractall(temp_dir)
                result = self._verify_directory(temp_dir)
                shutil.rmtree(temp_dir)
                return result
        elif package_path.is_dir():
            return self._verify_directory(package_path)
        else:
            logger.error(f"Invalid package path: {package_path}")
            return False
    
    def _verify_directory(self, package_dir: Path) -> bool:
        """Verify directory package integrity"""
        manifest_path = package_dir / "SHA256SUMS"
        
        if not manifest_path.exists():
            logger.error("SHA256SUMS manifest not found")
            return False
        
        # Read manifest
        verification_failed = False
        with open(manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('  ', 1)
                if len(parts) != 2:
                    logger.warning(f"Invalid manifest line: {line}")
                    continue
                
                expected_hash, file_path = parts
                full_path = package_dir / file_path
                
                if not full_path.exists():
                    logger.error(f"File missing: {file_path}")
                    verification_failed = True
                    continue
                
                actual_hash = self.calculate_file_hash(full_path)
                if actual_hash != expected_hash:
                    logger.error(f"Hash mismatch for {file_path}")
                    logger.error(f"Expected: {expected_hash}")
                    logger.error(f"Actual: {actual_hash}")
                    verification_failed = True
                else:
                    logger.debug(f"Verified: {file_path}")
        
        if verification_failed:
            logger.error("Package verification FAILED")
            return False
        else:
            logger.info("Package verification PASSED")
            return True

def create_sample_package():
    """Create a sample package for testing"""
    packager = SARPackager()
    
    # Create sample metadata
    metadata = MissionMetadata(
        mission_id="SAMPLE_001",
        mission_name="Test SAR Mission",
        start_time="2024-01-15T10:00:00Z",
        end_time="2024-01-15T12:00:00Z",
        operator="Test Operator",
        aircraft_type="Test Aircraft",
        sensor_config={"camera": "test_camera", "resolution": "1920x1080"},
        geolocation_data={"area": "test_area", "coordinates": [0, 0]},
        detection_summary={"humans_detected": 0, "total_frames": 100}
    )
    
    # Create sample files (empty for demo)
    sample_files = []
    
    # Create package
    package_path = packager.create_package(
        mission_data=metadata,
        files=sample_files,
        package_name="sample_mission",
        compress=True
    )
    
    logger.info(f"Sample package created: {package_path}")
    return package_path

def main():
    """Command line interface for SAR packager"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='SAR Mission Evidence Packager')
    parser.add_argument('--metadata', required=True, help='Path to mission metadata JSON file')
    parser.add_argument('--package-name', help='Name for the package (default: auto-generated)')
    parser.add_argument('--files', nargs='*', help='List of files to include in package')
    parser.add_argument('--compress', action='store_true', help='Create compressed ZIP package')
    parser.add_argument('--output-dir', help='Output directory for package (default: current directory)')
    parser.add_argument('--verify', action='store_true', help='Verify package after creation')
    
    args = parser.parse_args()
    
    try:
        # Load metadata from file
        with open(args.metadata, 'r') as f:
            metadata_dict = json.load(f)
        
        # Create metadata object
        metadata = MissionMetadata(
            mission_id=metadata_dict.get('mission_id', f'mission_{int(time.time())}'),
            mission_name=metadata_dict.get('mission_name', 'SAR Mission'),
            start_time=metadata_dict.get('start_time', datetime.now().isoformat()),
            end_time=metadata_dict.get('end_time', datetime.now().isoformat()),
            operator=metadata_dict.get('operator', 'Unknown'),
            aircraft_type=metadata_dict.get('aircraft_type', 'Drone'),
            sensor_config=metadata_dict.get('sensor_config', {}),
            geolocation_data=metadata_dict.get('geolocation_data', {}),
            detection_summary=metadata_dict.get('detection_summary', {})
        )
        
        # Initialize packager
        output_dir = args.output_dir or "exports"
        packager = SARPackager(base_output_dir=output_dir)
        
        print(f"Creating SAR package for mission: {metadata.mission_name}")
        
        # Create package
        package_path = packager.create_package(
            mission_data=metadata,
            files=args.files or [],
            package_name=args.package_name or f"mission_{metadata.mission_id}",
            compress=args.compress
        )
        
        if args.compress:
            print(f"Created compressed package: {package_path}")
        else:
            print(f"Created package directory: {package_path}")
        
        # Verify package if requested
        if args.verify:
            print("Verifying package...")
            if packager.verify_package(package_path):
                print("Package verification successful")
            else:
                print("Package verification failed")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error creating package: {e}")
        return 1


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())