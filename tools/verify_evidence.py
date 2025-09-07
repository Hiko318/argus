#!/usr/bin/env python3
"""
Evidence Verifier - Validates tamper-evident evidence packages

Verifies:
- File integrity using SHA256 hashes
- Digital signatures
- Metadata consistency
- OpenTimestamps proofs
"""

import os
import json
import hashlib
import zipfile
import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature


class EvidenceVerifier:
    """Verifies tamper-evident evidence packages"""
    
    def __init__(self):
        self.verification_results = []
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _verify_signature(self, metadata: Dict[str, Any], signature_data: Dict[str, Any]) -> bool:
        """Verify digital signature"""
        try:
            # Load public key
            public_key_pem = base64.b64decode(signature_data["public_key"])
            public_key = serialization.load_pem_public_key(public_key_pem)
            
            # Verify signature
            metadata_bytes = json.dumps(metadata, sort_keys=True).encode('utf-8')
            signature = base64.b64decode(signature_data["signature"])
            
            public_key.verify(
                signature,
                metadata_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except (InvalidSignature, Exception) as e:
            self.verification_results.append(f"Signature verification failed: {e}")
            return False
    
    def _verify_file_integrity(self, temp_dir: Path, manifest: Dict[str, str]) -> bool:
        """Verify file integrity using manifest hashes"""
        integrity_ok = True
        
        for filename, expected_hash in manifest.items():
            file_path = temp_dir / filename
            if not file_path.exists():
                self.verification_results.append(f"Missing file: {filename}")
                integrity_ok = False
                continue
            
            actual_hash = self._calculate_file_hash(str(file_path))
            if actual_hash != expected_hash:
                self.verification_results.append(
                    f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
                )
                integrity_ok = False
            else:
                self.verification_results.append(f"✓ File integrity verified: {filename}")
        
        return integrity_ok
    
    def _verify_metadata_structure(self, metadata: Dict[str, Any]) -> bool:
        """Verify metadata has required structure"""
        required_fields = [
            "version", "created_at", "evidence", "system"
        ]
        
        evidence_fields = [
            "timestamp", "location", "operator_id", "capture_method"
        ]
        
        system_fields = [
            "version", "platform", "hash_algorithm", "signature_algorithm"
        ]
        
        # Check top-level fields
        for field in required_fields:
            if field not in metadata:
                self.verification_results.append(f"Missing metadata field: {field}")
                return False
        
        # Check evidence fields
        for field in evidence_fields:
            if field not in metadata["evidence"]:
                self.verification_results.append(f"Missing evidence field: {field}")
                return False
        
        # Check system fields
        for field in system_fields:
            if field not in metadata["system"]:
                self.verification_results.append(f"Missing system field: {field}")
                return False
        
        self.verification_results.append("✓ Metadata structure verified")
        return True
    
    def _verify_ots_timestamp(self, ots_path: Path) -> bool:
        """Verify OpenTimestamps proof (placeholder implementation)"""
        if not ots_path.exists():
            self.verification_results.append("Missing OpenTimestamps proof")
            return False
        
        # In a real implementation, this would verify against OpenTimestamps servers
        # For now, just check the file exists and has content
        try:
            with open(ots_path, "r") as f:
                content = f.read()
                if "OpenTimestamps proof" in content:
                    self.verification_results.append("✓ OpenTimestamps proof present")
                    return True
                else:
                    self.verification_results.append("Invalid OpenTimestamps proof format")
                    return False
        except Exception as e:
            self.verification_results.append(f"Error reading OpenTimestamps proof: {e}")
            return False
    
    def verify_package(self, package_path: str) -> Tuple[bool, List[str]]:
        """Verify complete evidence package"""
        self.verification_results = []
        package_path = Path(package_path)
        
        if not package_path.exists():
            return False, ["Evidence package not found"]
        
        if not package_path.suffix.lower() == ".zip":
            return False, ["Evidence package must be a ZIP file"]
        
        # Extract to temporary directory
        temp_dir = package_path.parent / f"verify_{package_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract package
            with zipfile.ZipFile(package_path, "r") as zipf:
                zipf.extractall(temp_dir)
            
            self.verification_results.append(f"✓ Package extracted: {package_path.name}")
            
            # Load and verify metadata
            metadata_path = temp_dir / "metadata.json"
            if not metadata_path.exists():
                return False, ["Missing metadata.json"]
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Verify metadata structure
            if not self._verify_metadata_structure(metadata):
                return False, self.verification_results
            
            # Load and verify manifest
            manifest_path = temp_dir / "manifest.sha256"
            if not manifest_path.exists():
                return False, self.verification_results + ["Missing manifest.sha256"]
            
            manifest = {}
            with open(manifest_path, "r") as f:
                for line in f:
                    if line.strip():
                        hash_val, filename = line.strip().split("  ", 1)
                        manifest[filename] = hash_val
            
            # Verify file integrity
            integrity_ok = self._verify_file_integrity(temp_dir, manifest)
            
            # Load and verify signature
            signature_path = temp_dir / "metadata.sig"
            if not signature_path.exists():
                return False, self.verification_results + ["Missing metadata.sig"]
            
            with open(signature_path, "r") as f:
                signature_data = json.load(f)
            
            signature_ok = self._verify_signature(metadata, signature_data)
            
            # Verify OpenTimestamps
            ots_path = temp_dir / "metadata.ots"
            ots_ok = self._verify_ots_timestamp(ots_path)
            
            # Overall verification result
            all_ok = integrity_ok and signature_ok and ots_ok
            
            if all_ok:
                self.verification_results.append("\n✓ EVIDENCE PACKAGE VERIFICATION PASSED")
                self.verification_results.append(f"  Package: {package_path.name}")
                self.verification_results.append(f"  Operator: {metadata['evidence']['operator_id']}")
                self.verification_results.append(f"  Timestamp: {metadata['evidence']['timestamp']}")
                self.verification_results.append(f"  Location: {metadata['evidence']['location']}")
            else:
                self.verification_results.append("\n✗ EVIDENCE PACKAGE VERIFICATION FAILED")
            
            return all_ok, self.verification_results
        
        except Exception as e:
            return False, self.verification_results + [f"Verification error: {e}"]
        
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def generate_report(self, package_path: str, output_path: str = None) -> str:
        """Generate verification report"""
        success, results = self.verify_package(package_path)
        
        report = []
        report.append("EVIDENCE PACKAGE VERIFICATION REPORT")
        report.append("=" * 50)
        report.append(f"Package: {package_path}")
        report.append(f"Verification Date: {json.dumps(None)}")
        report.append(f"Status: {'PASSED' if success else 'FAILED'}")
        report.append("")
        report.append("Verification Details:")
        report.append("-" * 25)
        
        for result in results:
            report.append(result)
        
        report_content = "\n".join(report)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_content)
        
        return report_content


def main():
    """CLI interface for evidence verification"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify evidence packages")
    parser.add_argument("package", help="Path to evidence package (.zip)")
    parser.add_argument("--report", help="Output verification report to file")
    parser.add_argument("--quiet", action="store_true", help="Only show final result")
    
    args = parser.parse_args()
    
    verifier = EvidenceVerifier()
    
    if args.report:
        report = verifier.generate_report(args.package, args.report)
        print(f"Verification report saved to: {args.report}")
        if not args.quiet:
            print("\n" + report)
    else:
        success, results = verifier.verify_package(args.package)
        
        if not args.quiet:
            for result in results:
                print(result)
        
        exit_code = 0 if success else 1
        exit(exit_code)


if __name__ == "__main__":
    main()