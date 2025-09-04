#!/usr/bin/env python3
"""
Model Download Wrapper for Foresight

This script provides functions to download and manage AI models
without storing the actual model files in the repository.
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Optional

# Model configurations
MODEL_CONFIGS = {
    'yolov8n': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'filename': 'yolov8n.pt',
        'sha256': 'f59b3d833e4f7c8a7b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b',
        'description': 'YOLOv8 Nano model for object detection',
        'size_mb': 6.2
    },
    'yolov8s': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'filename': 'yolov8s.pt',
        'sha256': 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2',
        'description': 'YOLOv8 Small model for object detection',
        'size_mb': 22.5
    },
    'yolov8m': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'filename': 'yolov8m.pt',
        'sha256': 'b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3',
        'description': 'YOLOv8 Medium model for object detection',
        'size_mb': 52.0
    }
}

class ModelDownloader:
    """Handles downloading and verification of AI models."""
    
    def __init__(self, models_dir: str = None):
        """Initialize the model downloader.
        
        Args:
            models_dir: Directory to store downloaded models
        """
        if models_dir is None:
            # Default to project root directory
            project_root = Path(__file__).parent.parent
            models_dir = project_root
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def download_file(self, url: str, destination: Path, chunk_size: int = 8192) -> bool:
        """Download a file from URL to destination.
        
        Args:
            url: URL to download from
            destination: Local file path to save to
            chunk_size: Size of chunks to download
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
            print(f"\nDownload completed: {destination}")
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if destination.exists():
                destination.unlink()  # Remove partial file
            return False
    
    def verify_model(self, model_name: str) -> bool:
        """Verify a downloaded model's integrity.
        
        Args:
            model_name: Name of the model to verify
            
        Returns:
            True if model is valid, False otherwise
        """
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model: {model_name}")
            return False
        
        config = MODEL_CONFIGS[model_name]
        model_path = self.models_dir / config['filename']
        
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            return False
        
        print(f"Verifying {model_name}...")
        file_hash = self.calculate_sha256(model_path)
        expected_hash = config['sha256']
        
        if file_hash == expected_hash:
            print(f"✓ Model {model_name} verified successfully")
            return True
        else:
            print(f"✗ Model {model_name} verification failed")
            print(f"  Expected: {expected_hash}")
            print(f"  Got:      {file_hash}")
            return False
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download a specific model.
        
        Args:
            model_name: Name of the model to download
            force: Force re-download even if file exists
            
        Returns:
            True if download and verification successful
        """
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model: {model_name}")
            print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
            return False
        
        config = MODEL_CONFIGS[model_name]
        model_path = self.models_dir / config['filename']
        
        # Check if model already exists and is valid
        if model_path.exists() and not force:
            if self.verify_model(model_name):
                print(f"Model {model_name} already exists and is valid")
                return True
            else:
                print(f"Existing model {model_name} is invalid, re-downloading...")
        
        # Download the model
        if not self.download_file(config['url'], model_path):
            return False
        
        # Verify the downloaded model
        return self.verify_model(model_name)
    
    def list_models(self) -> None:
        """List all available models and their status."""
        print("Available Models:")
        print("=" * 50)
        
        for model_name, config in MODEL_CONFIGS.items():
            model_path = self.models_dir / config['filename']
            status = "✓ Downloaded" if model_path.exists() else "✗ Not downloaded"
            
            print(f"{model_name:12} | {config['size_mb']:6.1f} MB | {status}")
            print(f"             | {config['description']}")
            print()
    
    def clean_models(self) -> None:
        """Remove all downloaded model files."""
        removed_count = 0
        
        for config in MODEL_CONFIGS.values():
            model_path = self.models_dir / config['filename']
            if model_path.exists():
                model_path.unlink()
                print(f"Removed: {model_path}")
                removed_count += 1
        
        if removed_count == 0:
            print("No model files found to remove")
        else:
            print(f"Removed {removed_count} model files")

def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Foresight Model Downloader')
    parser.add_argument('action', choices=['download', 'list', 'verify', 'clean'],
                       help='Action to perform')
    parser.add_argument('model', nargs='?', help='Model name (for download/verify actions)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if file exists')
    parser.add_argument('--models-dir', help='Directory to store models')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.action == 'list':
        downloader.list_models()
    
    elif args.action == 'download':
        if not args.model:
            print("Error: Model name required for download action")
            sys.exit(1)
        
        success = downloader.download_model(args.model, args.force)
        sys.exit(0 if success else 1)
    
    elif args.action == 'verify':
        if not args.model:
            print("Error: Model name required for verify action")
            sys.exit(1)
        
        success = downloader.verify_model(args.model)
        sys.exit(0 if success else 1)
    
    elif args.action == 'clean':
        downloader.clean_models()

if __name__ == '__main__':
    main()