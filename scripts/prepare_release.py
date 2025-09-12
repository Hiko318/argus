#!/usr/bin/env python3
"""
Release preparation script for Foresight SAR System
Handles model asset preparation and GitHub release creation
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)

def prepare_model_assets() -> Dict:
    """Prepare model assets for release."""
    models_dir = Path(__file__).parent.parent / "models"
    assets = {}
    
    print("📦 Preparing model assets for release...")
    
    # Check for existing model files
    model_files = [
        "yolov8n.pt",
        "yolov8s.pt", 
        "yolov8m.pt",
        "yolov8n.onnx",
        "yolov8s.onnx"
    ]
    
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = get_file_size_mb(model_path)
            file_hash = calculate_file_hash(model_path)
            
            assets[model_file] = {
                "path": str(model_path),
                "size_mb": round(size_mb, 2),
                "sha256": file_hash,
                "upload_required": size_mb > 100  # GitHub has 100MB limit
            }
            
            print(f"  ✅ {model_file}: {size_mb:.1f}MB (SHA256: {file_hash[:16]}...)")
        else:
            print(f"  ⚠️  {model_file}: Not found")
    
    return assets

def create_release_manifest(assets: Dict, version: str) -> Dict:
    """Create release manifest with asset information."""
    manifest = {
        "version": version,
        "release_date": "2024-01-09",
        "assets": {},
        "download_instructions": {
            "small_models": "Download directly from GitHub Releases",
            "large_models": "Use download_models.py script or manual download from external storage"
        }
    }
    
    for filename, info in assets.items():
        if info["upload_required"]:
            # Large files - provide external download info
            manifest["assets"][filename] = {
                "size_mb": info["size_mb"],
                "sha256": info["sha256"],
                "download_method": "external",
                "note": "Too large for GitHub - use download_models.py script"
            }
        else:
            # Small files - can be uploaded to GitHub
            manifest["assets"][filename] = {
                "size_mb": info["size_mb"],
                "sha256": info["sha256"],
                "download_method": "github_release",
                "note": "Available as GitHub release asset"
            }
    
    return manifest

def generate_release_notes(version: str, assets: Dict) -> str:
    """Generate release notes for GitHub."""
    notes = f"""# Foresight SAR v{version} - Field-Ready Prototype

## 🚁 What's New

- **Production-Ready SAR System**: Complete search and rescue drone vision system
- **Real-time Human Detection**: YOLOv8-based detection with <100ms latency
- **Evidence Packaging**: Tamper-evident evidence collection with cryptographic signatures
- **Cross-Platform Support**: Windows and Linux installers available
- **Field Kit Documentation**: Complete deployment guide for field operations

## 📦 Installation

### Windows
1. Download `Foresight-SAR-Setup-v{version}.exe`
2. Run installer as Administrator
3. Follow setup wizard

### Linux
1. Download `Foresight-SAR-v{version}.AppImage`
2. Make executable: `chmod +x Foresight-SAR-v{version}.AppImage`
3. Run: `./Foresight-SAR-v{version}.AppImage`

## 🤖 AI Models

"""
    
    # Add model information
    for filename, info in assets.items():
        if filename.endswith('.pt'):
            model_name = filename.replace('.pt', '').upper()
            upload_method = "external storage" if info['upload_required'] else "GitHub release"
            notes += f"- **{model_name}**: {info['size_mb']}MB - Available via {upload_method}\n"
    
    notes += f"""

### Model Download

For automatic model download:
```bash
cd models
python download_models.py --model yolov8n --model yolov8s
```

## 🔧 System Requirements

### Minimum
- CPU: Intel i5-8400 / AMD Ryzen 5 3600
- RAM: 16GB DDR4
- GPU: NVIDIA GTX 1660 (CUDA support required)
- Storage: 500GB SSD

### Recommended
- CPU: Intel i7-10700K / AMD Ryzen 7 5800X
- RAM: 32GB DDR4
- GPU: NVIDIA RTX 3070 or better
- Storage: 1TB NVMe SSD

## 📋 Field Kit

See `packaging/field_kit.md` for complete hardware requirements and deployment procedures.

## 🔒 Security Features

- Evidence packaging with SHA256 manifests
- Cryptographic signatures for tamper detection
- Secure data transmission protocols
- Audit trail for all operations

## 🐛 Known Issues

- Electron-builder may fail on some Windows systems (use alternative build script)
- Large model files require external download due to GitHub size limits
- Some capture cards may require manual driver installation

## 📞 Support

- Documentation: See README.md and docs/ folder
- Issues: https://github.com/Hiko318/foresight/issues
- Field Kit Guide: packaging/field_kit.md

---

**Full Changelog**: https://github.com/Hiko318/foresight/compare/v0.8...v{version}
"""
    
    return notes

def create_github_release_script(version: str, assets: Dict) -> str:
    """Create a script for GitHub CLI release creation."""
    
    # Determine which assets can be uploaded to GitHub
    uploadable_assets = []
    for filename, info in assets.items():
        if not info["upload_required"]:
            uploadable_assets.append(info["path"])
    
    # Add installer files
    electron_dist = Path(__file__).parent.parent / "foresight-electron" / "dist"
    if (electron_dist / "Foresight SAR-win32-x64").exists():
        uploadable_assets.append(str(electron_dist / "Foresight SAR-win32-x64"))
    
    script = f"""#!/bin/bash
# GitHub Release Creation Script for Foresight SAR v{version}

set -e

echo "🚀 Creating GitHub release for Foresight SAR v{version}..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed"
    echo "Install from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub"
    echo "Run: gh auth login"
    exit 1
fi

echo "📝 Creating release..."
gh release create v{version} \\
    --title "Foresight SAR v{version} - Field-Ready Prototype" \\
    --notes-file release_notes.md \\
    --draft

echo "📦 Uploading assets..."
"""
    
    # Add asset uploads
    for asset_path in uploadable_assets:
        if os.path.exists(asset_path):
            script += f'gh release upload v{version} "{asset_path}"\n'
    
    script += f"""
echo "✅ Release created successfully!"
echo "📋 Next steps:"
echo "   1. Review the draft release at: https://github.com/Hiko318/foresight/releases"
echo "   2. Upload large model files manually or to external storage"
echo "   3. Update download_models.py with new URLs if needed"
echo "   4. Publish the release when ready"

echo "🔗 Release URL: https://github.com/Hiko318/foresight/releases/tag/v{version}"
"""
    
    return script

def main():
    parser = argparse.ArgumentParser(description="Prepare Foresight SAR release")
    parser.add_argument("--version", default="0.9", help="Release version")
    parser.add_argument("--output-dir", default="release_prep", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎯 Preparing release v{args.version}...")
    
    # Prepare model assets
    assets = prepare_model_assets()
    
    # Create release manifest
    manifest = create_release_manifest(assets, args.version)
    manifest_path = output_dir / "release_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"📄 Release manifest: {manifest_path}")
    
    # Generate release notes
    release_notes = generate_release_notes(args.version, assets)
    notes_path = output_dir / "release_notes.md"
    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write(release_notes)
    print(f"📝 Release notes: {notes_path}")
    
    # Create GitHub release script
    gh_script = create_github_release_script(args.version, assets)
    script_path = output_dir / "create_github_release.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(gh_script)
    script_path.chmod(0o755)  # Make executable
    print(f"🚀 GitHub release script: {script_path}")
    
    print(f"\n✅ Release preparation complete!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"\n📋 Next steps:")
    print(f"   1. Review files in {output_dir}/")
    print(f"   2. Run: cd {output_dir} && ./create_github_release.sh")
    print(f"   3. Upload large model files to external storage if needed")
    print(f"   4. Publish the GitHub release")

if __name__ == "__main__":
    main()