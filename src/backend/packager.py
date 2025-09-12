#!/usr/bin/env python3
"""
Evidence Packager - Backend module for creating tamper-evident evidence packages

This module provides a simplified interface for creating evidence packages with:
- metadata.json with capture details
- manifest.sha256 with file hashes
- .ots timestamp proof (stub implementation)
- Digital signatures (stub implementation)
"""

import os
import json
import hashlib
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logger = logging.getLogger(__name__)


class EvidencePackager:
    """Simplified evidence packager for SAR operations"""
    
    def __init__(self, output_dir: str = "./evidence_packages"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _create_manifest(self, files: List[str]) -> Dict[str, str]:
        """Create manifest with file hashes"""
        manifest = {}
        for file_path in files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                file_hash = self._calculate_file_hash(file_path)
                manifest[filename] = file_hash
        return manifest
    
    def _create_ots_stub(self, metadata_path: str) -> str:
        """Create OpenTimestamps stub file"""
        ots_content = {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file": os.path.basename(metadata_path),
            "hash": self._calculate_file_hash(metadata_path),
            "proof": "stub_timestamp_proof_data",
            "note": "This is a stub implementation for development"
        }
        return json.dumps(ots_content, indent=2)
    
    def create_evidence_package(self, 
                              metadata: Dict[str, Any],
                              evidence_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create evidence package with metadata, manifest, and timestamp"""
        
        try:
            # Create package directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            case_id = metadata.get('case_id', 'unknown')
            package_name = f"evidence_{case_id}_{timestamp}"
            package_dir = self.output_dir / package_name
            package_dir.mkdir(exist_ok=True)
            
            # Create metadata.json
            metadata_path = package_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Copy evidence files if provided
            copied_files = [str(metadata_path)]
            if evidence_files:
                for file_path in evidence_files:
                    if os.path.exists(file_path):
                        src_path = Path(file_path)
                        dst_path = package_dir / src_path.name
                        shutil.copy2(src_path, dst_path)
                        copied_files.append(str(dst_path))
            
            # Create manifest.sha256
            manifest = self._create_manifest(copied_files)
            manifest_path = package_dir / "manifest.sha256"
            with open(manifest_path, 'w') as f:
                for filename, file_hash in manifest.items():
                    f.write(f"{file_hash}  {filename}\n")
            
            # Create .ots timestamp stub
            ots_content = self._create_ots_stub(str(metadata_path))
            ots_path = package_dir / "metadata.ots"
            with open(ots_path, 'w') as f:
                f.write(ots_content)
            
            # Calculate package summary
            total_size = sum(os.path.getsize(f) for f in copied_files if os.path.exists(f))
            
            result = {
                "success": True,
                "package_path": str(package_dir),
                "package_name": package_name,
                "files_count": len(copied_files),
                "total_size_bytes": total_size,
                "manifest_hash": self._calculate_file_hash(str(manifest_path)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": {
                    "metadata": str(metadata_path),
                    "manifest": str(manifest_path),
                    "timestamp": str(ots_path)
                }
            }
            
            logger.info(f"Evidence package created: {package_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create evidence package: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def verify_package(self, package_path: str) -> Dict[str, Any]:
        """Verify evidence package integrity"""
        try:
            package_dir = Path(package_path)
            if not package_dir.exists():
                return {"success": False, "error": "Package directory not found"}
            
            # Check required files
            required_files = ["metadata.json", "manifest.sha256", "metadata.ots"]
            missing_files = []
            for filename in required_files:
                if not (package_dir / filename).exists():
                    missing_files.append(filename)
            
            if missing_files:
                return {
                    "success": False,
                    "error": f"Missing required files: {missing_files}"
                }
            
            # Verify manifest hashes
            manifest_path = package_dir / "manifest.sha256"
            verification_errors = []
            
            with open(manifest_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('  ', 1)
                        if len(parts) == 2:
                            expected_hash, filename = parts
                            file_path = package_dir / filename
                            if file_path.exists():
                                actual_hash = self._calculate_file_hash(str(file_path))
                                if actual_hash != expected_hash:
                                    verification_errors.append(f"Hash mismatch for {filename}")
                            else:
                                verification_errors.append(f"File not found: {filename}")
            
            return {
                "success": len(verification_errors) == 0,
                "errors": verification_errors,
                "verified_files": len(required_files) - len(missing_files)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


def run_sanity_test() -> bool:
    """Run sanity test for evidence packager"""
    try:
        # Load test metadata
        test_metadata_path = "demos/sample_mission/metadata.json"
        if not os.path.exists(test_metadata_path):
            logger.error(f"Test metadata file not found: {test_metadata_path}")
            return False
        
        with open(test_metadata_path, 'r') as f:
            test_metadata = json.load(f)
        
        # Create packager instance
        packager = EvidencePackager("./test_evidence")
        
        # Create evidence package
        result = packager.create_evidence_package(
            metadata=test_metadata,
            evidence_files=[test_metadata_path]
        )
        
        if not result["success"]:
            logger.error(f"Failed to create evidence package: {result.get('error')}")
            return False
        
        # Verify the package
        verification = packager.verify_package(result["package_path"])
        
        if not verification["success"]:
            logger.error(f"Package verification failed: {verification.get('errors')}")
            return False
        
        logger.info("Evidence packager sanity test passed!")
        logger.info(f"Package created: {result['package_path']}")
        logger.info(f"Files: {result['files_count']}, Size: {result['total_size_bytes']} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"Sanity test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    
    # Run sanity test
    success = run_sanity_test()
    exit(0 if success else 1)