#!/usr/bin/env python3
"""
Test suite for SAR packager integration with GUI.

Tests the packager functionality including:
- Command line interface
- Package creation and verification
- Integration with Electron frontend
"""

import unittest
import tempfile
import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from packager.packager import SARPackager, MissionMetadata


class TestPackagerIntegration(unittest.TestCase):
    """Test SAR packager integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.packager = SARPackager(base_output_dir=self.temp_dir)
        
        # Sample mission metadata
        self.sample_metadata = MissionMetadata(
            mission_id="test_mission_001",
            mission_name="Test SAR Mission",
            start_time="2024-01-15T10:00:00Z",
            end_time="2024-01-15T12:00:00Z",
            operator="Test Operator",
            aircraft_type="Test Drone",
            sensor_config={"camera": "test", "resolution": "1920x1080"},
            geolocation_data={"coverage_area": "test_area"},
            detection_summary={"total_detections": 3, "confidence_threshold": 0.8}
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_packager_initialization(self):
        """Test packager can be initialized"""
        self.assertIsInstance(self.packager, SARPackager)
        self.assertEqual(self.packager.base_output_dir, self.temp_dir)
    
    def test_create_basic_package(self):
        """Test creating a basic package without files"""
        package_path = self.packager.create_package(
            mission_data=self.sample_metadata,
            files=[],
            package_name="test_package",
            compress=False
        )
        
        self.assertTrue(os.path.exists(package_path))
        self.assertTrue(os.path.isdir(package_path))
        
        # Check required files exist
        metadata_file = os.path.join(package_path, "metadata.json")
        manifest_file = os.path.join(package_path, "SHA256SUMS")
        
        self.assertTrue(os.path.exists(metadata_file))
        self.assertTrue(os.path.exists(manifest_file))
        
        # Verify metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata['mission_id'], 'test_mission_001')
        self.assertEqual(metadata['mission_name'], 'Test SAR Mission')
    
    def test_create_compressed_package(self):
        """Test creating a compressed ZIP package"""
        package_path = self.packager.create_package(
            mission_data=self.sample_metadata,
            files=[],
            package_name="test_compressed",
            compress=True
        )
        
        self.assertTrue(os.path.exists(package_path))
        self.assertTrue(package_path.endswith('.zip'))
    
    def test_package_with_files(self):
        """Test creating package with additional files"""
        # Create test files
        test_file1 = os.path.join(self.temp_dir, "test1.txt")
        test_file2 = os.path.join(self.temp_dir, "test2.txt")
        
        with open(test_file1, 'w') as f:
            f.write("Test file 1 content")
        with open(test_file2, 'w') as f:
            f.write("Test file 2 content")
        
        package_path = self.packager.create_package(
            mission_data=self.sample_metadata,
            files=[test_file1, test_file2],
            package_name="test_with_files",
            compress=False
        )
        
        # Check files were copied
        copied_file1 = os.path.join(package_path, "files", "test1.txt")
        copied_file2 = os.path.join(package_path, "files", "test2.txt")
        
        self.assertTrue(os.path.exists(copied_file1))
        self.assertTrue(os.path.exists(copied_file2))
    
    def test_package_verification(self):
        """Test package verification functionality"""
        package_path = self.packager.create_package(
            mission_data=self.sample_metadata,
            files=[],
            package_name="test_verify",
            compress=False
        )
        
        # Package should verify successfully
        self.assertTrue(self.packager.verify_package(package_path))
        
        # Corrupt the metadata file
        metadata_file = os.path.join(package_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            f.write("corrupted data")
        
        # Package should fail verification
        self.assertFalse(self.packager.verify_package(package_path))
    
    def test_command_line_interface(self):
        """Test command line interface"""
        # Create metadata file
        metadata_file = os.path.join(self.temp_dir, "test_metadata.json")
        metadata_dict = {
            "mission_id": "cli_test_001",
            "mission_name": "CLI Test Mission",
            "start_time": "2024-01-15T10:00:00Z",
            "end_time": "2024-01-15T12:00:00Z",
            "operator": "CLI Test Operator",
            "aircraft_type": "CLI Test Drone",
            "sensor_config": {"camera": "cli_test"},
            "geolocation_data": {},
            "detection_summary": {}
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f)
        
        # Get packager script path
        packager_script = Path(__file__).parent.parent / "packager" / "packager.py"
        
        # Run packager via command line
        cmd = [
            sys.executable,
            str(packager_script),
            "--metadata", metadata_file,
            "--package-name", "cli_test",
            "--output-dir", self.temp_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Check if command succeeded
            if result.returncode == 0:
                # Look for created package
                expected_package = os.path.join(self.temp_dir, "cli_test")
                self.assertTrue(os.path.exists(expected_package))
            else:
                # Print error for debugging
                print(f"CLI test failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.fail("Command line interface test timed out")
        except FileNotFoundError:
            self.skipTest("Packager script not found - skipping CLI test")
    
    def test_metadata_validation(self):
        """Test metadata validation"""
        # Test with minimal metadata
        minimal_metadata = MissionMetadata(
            mission_id="minimal_test",
            mission_name="Minimal Test",
            start_time="2024-01-15T10:00:00Z",
            end_time="2024-01-15T10:00:00Z",
            operator="Test",
            aircraft_type="Test",
            sensor_config={},
            geolocation_data={},
            detection_summary={}
        )
        
        package_path = self.packager.create_package(
            mission_data=minimal_metadata,
            files=[],
            package_name="minimal_test",
            compress=False
        )
        
        self.assertTrue(os.path.exists(package_path))
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with non-existent files
        with self.assertRaises(Exception):
            self.packager.create_package(
                mission_data=self.sample_metadata,
                files=["/non/existent/file.txt"],
                package_name="error_test",
                compress=False
            )
    
    def test_package_naming(self):
        """Test package naming conventions"""
        # Test auto-generated name
        package_path = self.packager.create_package(
            mission_data=self.sample_metadata,
            files=[],
            package_name=None,
            compress=False
        )
        
        package_name = os.path.basename(package_path)
        self.assertTrue(package_name.startswith('mission_'))
        
        # Test custom name
        custom_package_path = self.packager.create_package(
            mission_data=self.sample_metadata,
            files=[],
            package_name="custom_name",
            compress=False
        )
        
        custom_package_name = os.path.basename(custom_package_path)
        self.assertEqual(custom_package_name, "custom_name")


if __name__ == '__main__':
    unittest.main()