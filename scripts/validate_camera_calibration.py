#!/usr/bin/env python3
"""
Camera Calibration Validation Script

Validates camera intrinsics configuration and telemetry timestamp synchronization
for the Foresight SAR system.
"""

import sys
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.backend.telemetry_service import CameraIntrinsics
    from src.backend.camera_calibration import CameraCalibrationService, CalibrationResult
    REAL_MODULES = True
except ImportError:
    print("Warning: Using mock classes for validation")
    REAL_MODULES = False
    
    class MockCameraIntrinsics:
        def __init__(self, **kwargs):
            self.fx = kwargs.get('fx', 800.0)
            self.fy = kwargs.get('fy', 800.0)
            self.cx = kwargs.get('cx', 320.0)
            self.cy = kwargs.get('cy', 240.0)
            self.width = kwargs.get('width', 640)
            self.height = kwargs.get('height', 480)
            self.k1 = kwargs.get('k1', 0.0)
            self.k2 = kwargs.get('k2', 0.0)
            self.p1 = kwargs.get('p1', 0.0)
            self.p2 = kwargs.get('p2', 0.0)
            self.k3 = kwargs.get('k3', 0.0)
            self.model = kwargs.get('model', 'pinhole')
            self.camera_name = kwargs.get('camera_name', 'test_camera')
    
    CameraIntrinsics = MockCameraIntrinsics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraCalibrationValidator:
    """Validates camera calibration configuration and setup"""
    
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)
        self.validation_results = {}
        
    def validate_camera_config_file(self, config_file):
        """Validate camera calibration configuration file"""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            return {
                'status': 'FAIL',
                'error': f'Configuration file {config_file} not found',
                'file_path': str(config_path)
            }
        
        try:
            if config_path.suffix == '.yaml':
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                return {
                    'status': 'FAIL',
                    'error': f'Unsupported config format: {config_path.suffix}'
                }
            
            # Validate camera matrix structure
            validation_result = {
                'status': 'PASS',
                'file_path': str(config_path),
                'config_data': config_data,
                'checks': {}
            }
            
            # Check for camera matrix (either top-level or nested)
            matrix = None
            if 'camera_matrix' in config_data:
                matrix = config_data['camera_matrix']
            elif 'camera_intrinsics' in config_data and 'camera_matrix' in config_data['camera_intrinsics']:
                matrix = config_data['camera_intrinsics']['camera_matrix']
            
            if matrix is not None:
                if isinstance(matrix, list) and len(matrix) == 3:
                    validation_result['checks']['camera_matrix'] = 'PASS'
                    # Extract focal lengths
                    fx = matrix[0][0] if len(matrix[0]) > 0 else 0
                    fy = matrix[1][1] if len(matrix[1]) > 1 else 0
                    cx = matrix[0][2] if len(matrix[0]) > 2 else 0
                    cy = matrix[1][2] if len(matrix[1]) > 2 else 0
                    
                    validation_result['intrinsics'] = {
                        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
                    }
                    
                    # Validate reasonable values
                    if fx > 0 and fy > 0:
                        validation_result['checks']['focal_lengths'] = 'PASS'
                    else:
                        validation_result['checks']['focal_lengths'] = 'FAIL - Invalid focal lengths'
                        validation_result['status'] = 'WARN'
                else:
                    validation_result['checks']['camera_matrix'] = 'FAIL - Invalid matrix structure'
                    validation_result['status'] = 'FAIL'
            else:
                validation_result['checks']['camera_matrix'] = 'FAIL - Missing camera_matrix'
                validation_result['status'] = 'FAIL'
            
            # Check for distortion coefficients (either top-level or nested)
            dist_coeffs = None
            if 'dist_coeffs' in config_data:
                dist_coeffs = config_data['dist_coeffs']
            elif 'camera_intrinsics' in config_data and 'dist_coeffs' in config_data['camera_intrinsics']:
                dist_coeffs = config_data['camera_intrinsics']['dist_coeffs']
            
            if dist_coeffs is not None:
                validation_result['checks']['distortion_coeffs'] = 'PASS'
            else:
                validation_result['checks']['distortion_coeffs'] = 'WARN - Missing distortion coefficients'
            
            # Check FOV if present
            if 'fov_h_deg' in config_data and 'fov_v_deg' in config_data:
                fov_h = config_data['fov_h_deg']
                fov_v = config_data['fov_v_deg']
                if 10 <= fov_h <= 180 and 10 <= fov_v <= 180:
                    validation_result['checks']['fov'] = 'PASS'
                else:
                    validation_result['checks']['fov'] = 'WARN - Unusual FOV values'
            
            return validation_result
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': f'Failed to parse config file: {str(e)}',
                'file_path': str(config_path)
            }
    
    def validate_telemetry_timestamps(self, telemetry_file=None):
        """Validate telemetry timestamp synchronization"""
        if not telemetry_file:
            # Look for recent telemetry files
            log_dirs = list(Path('out/logs').glob('session_*')) if Path('out/logs').exists() else []
            if log_dirs:
                latest_session = max(log_dirs, key=lambda x: x.stat().st_mtime)
                telemetry_file = latest_session / 'telemetry.json'
        
        if not telemetry_file or not Path(telemetry_file).exists():
            return {
                'status': 'SKIP',
                'message': 'No telemetry file found for validation'
            }
        
        try:
            with open(telemetry_file, 'r') as f:
                # Read first few lines to check format
                lines = []
                for i, line in enumerate(f):
                    if i >= 10:  # Only check first 10 entries
                        break
                    if line.strip():
                        lines.append(json.loads(line.strip()))
            
            if not lines:
                return {
                    'status': 'FAIL',
                    'error': 'Empty telemetry file'
                }
            
            validation_result = {
                'status': 'PASS',
                'file_path': str(telemetry_file),
                'sample_count': len(lines),
                'checks': {}
            }
            
            # Check timestamp format
            timestamps = []
            for entry in lines:
                if 'timestamp' in entry:
                    try:
                        ts = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                        timestamps.append(ts)
                    except:
                        validation_result['checks']['timestamp_format'] = 'FAIL - Invalid timestamp format'
                        validation_result['status'] = 'FAIL'
                        break
            else:
                validation_result['checks']['timestamp_format'] = 'PASS'
            
            # Check timestamp intervals
            if len(timestamps) > 1:
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                           for i in range(len(timestamps)-1)]
                avg_interval = sum(intervals) / len(intervals)
                validation_result['avg_timestamp_interval'] = f"{avg_interval:.3f}s"
                
                if 0.01 <= avg_interval <= 1.0:  # 10ms to 1s seems reasonable
                    validation_result['checks']['timestamp_intervals'] = 'PASS'
                else:
                    validation_result['checks']['timestamp_intervals'] = 'WARN - Unusual timestamp intervals'
            
            # Check for camera intrinsics in telemetry
            has_intrinsics = any('camera_intrinsics' in entry for entry in lines)
            if has_intrinsics:
                validation_result['checks']['camera_intrinsics_in_telemetry'] = 'PASS'
                # Get sample intrinsics
                for entry in lines:
                    if 'camera_intrinsics' in entry:
                        validation_result['sample_intrinsics'] = entry['camera_intrinsics']
                        break
            else:
                validation_result['checks']['camera_intrinsics_in_telemetry'] = 'WARN - No camera intrinsics in telemetry'
            
            return validation_result
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': f'Failed to validate telemetry: {str(e)}',
                'file_path': str(telemetry_file)
            }
    
    def run_full_validation(self):
        """Run complete camera calibration validation"""
        logger.info("Starting camera calibration validation...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config_files': {},
            'telemetry_validation': {},
            'overall_status': 'PASS'
        }
        
        # Validate configuration files
        config_files = ['camera_calib.yaml', 'dji_o4_config.yaml', 'dji_config.json']
        
        for config_file in config_files:
            logger.info(f"Validating {config_file}...")
            result = self.validate_camera_config_file(config_file)
            results['config_files'][config_file] = result
            
            if result['status'] == 'FAIL':
                results['overall_status'] = 'FAIL'
            elif result['status'] == 'WARN' and results['overall_status'] == 'PASS':
                results['overall_status'] = 'WARN'
        
        # Validate telemetry timestamps
        logger.info("Validating telemetry timestamps...")
        telemetry_result = self.validate_telemetry_timestamps()
        results['telemetry_validation'] = telemetry_result
        
        if telemetry_result['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        elif telemetry_result['status'] == 'WARN' and results['overall_status'] == 'PASS':
            results['overall_status'] = 'WARN'
        
        return results
    
    def print_validation_report(self, results):
        """Print formatted validation report"""
        print("\n" + "="*60)
        print("CAMERA CALIBRATION VALIDATION REPORT")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Overall Status: {results['overall_status']}")
        
        print("\nConfiguration Files:")
        print("-" * 30)
        for config_file, result in results['config_files'].items():
            print(f"\n{config_file}: {result['status']}")
            if 'checks' in result:
                for check, status in result['checks'].items():
                    print(f"  - {check}: {status}")
            if 'intrinsics' in result:
                intrinsics = result['intrinsics']
                print(f"  - Focal lengths: fx={intrinsics['fx']}, fy={intrinsics['fy']}")
                print(f"  - Principal point: cx={intrinsics['cx']}, cy={intrinsics['cy']}")
            if 'error' in result:
                print(f"  - Error: {result['error']}")
        
        print("\nTelemetry Validation:")
        print("-" * 30)
        telemetry = results['telemetry_validation']
        print(f"Status: {telemetry['status']}")
        if 'checks' in telemetry:
            for check, status in telemetry['checks'].items():
                print(f"  - {check}: {status}")
        if 'avg_timestamp_interval' in telemetry:
            print(f"  - Average timestamp interval: {telemetry['avg_timestamp_interval']}")
        if 'sample_intrinsics' in telemetry:
            intrinsics = telemetry['sample_intrinsics']
            print(f"  - Sample intrinsics: fx={intrinsics.get('fx')}, fy={intrinsics.get('fy')}")
        if 'error' in telemetry:
            print(f"  - Error: {telemetry['error']}")
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Validate camera calibration setup')
    parser.add_argument('--config-dir', default='configs', help='Configuration directory')
    parser.add_argument('--telemetry-file', help='Specific telemetry file to validate')
    parser.add_argument('--output', help='Output validation results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = CameraCalibrationValidator(args.config_dir)
    results = validator.run_full_validation()
    
    validator.print_validation_report(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nValidation results saved to {args.output}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'FAIL':
        sys.exit(1)
    elif results['overall_status'] == 'WARN':
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()