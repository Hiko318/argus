#!/usr/bin/env python3
"""Software Bill of Materials (SBOM) Generation Tool.

Generates comprehensive SBOM for the Foresight SAR System including:
- Python dependencies
- System dependencies
- Docker images
- Model files
- Configuration files
- Security vulnerability reports
"""

import json
import subprocess
import sys
import argparse
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import pkg_resources
import platform

try:
    import requests
except ImportError:
    requests = None

try:
    from cyclonedx.model import bom, component
    from cyclonedx.output import get_instance
    from cyclonedx.output.json import JsonV1Dot4
    from cyclonedx.output.xml import XmlV1Dot4
except ImportError:
    print("Warning: cyclonedx-python not installed. Install with: pip install cyclonedx-bom")
    bom = component = None


class SBOMGenerator:
    """Generate Software Bill of Materials for Foresight SAR System."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.sbom_data = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:foresight-sar-{self.timestamp}",
            "version": 1,
            "metadata": {
                "timestamp": self.timestamp,
                "tools": [
                    {
                        "vendor": "Foresight SAR Team",
                        "name": "SBOM Generator",
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "application",
                    "bom-ref": "foresight-sar",
                    "name": "Foresight SAR System",
                    "version": self._get_version(),
                    "description": "AI-powered Search and Rescue system with real-time object detection and tracking",
                    "licenses": [
                        {
                            "license": {
                                "name": "MIT"
                            }
                        }
                    ]
                }
            },
            "components": [],
            "vulnerabilities": []
        }
    
    def _get_version(self) -> str:
        """Get project version from various sources."""
        # Try to get version from git tag
        try:
            result = subprocess.run(
                ['git', 'describe', '--tags', '--abbrev=0'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass
        
        # Try to get version from setup.py or pyproject.toml
        setup_py = self.project_root / 'setup.py'
        if setup_py.exists():
            try:
                with open(setup_py, 'r') as f:
                    content = f.read()
                    import re
                    match = re.search(r'version=["\']([^"\']+)["\']', content)
                    if match:
                        return match.group(1)
            except Exception:
                pass
        
        # Default version
        return "1.0.0-dev"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return ""
    
    def generate_python_dependencies(self) -> List[Dict[str, Any]]:
        """Generate Python dependency components."""
        components = []
        
        # Get installed packages
        try:
            installed_packages = [d for d in pkg_resources.working_set]
            
            for package in installed_packages:
                component_data = {
                    "type": "library",
                    "bom-ref": f"python-{package.project_name}-{package.version}",
                    "name": package.project_name,
                    "version": package.version,
                    "purl": f"pkg:pypi/{package.project_name}@{package.version}",
                    "scope": "required"
                }
                
                # Add license information if available
                try:
                    metadata = package.get_metadata('METADATA')
                    if metadata:
                        for line in metadata.split('\n'):
                            if line.startswith('License:'):
                                license_name = line.split(':', 1)[1].strip()
                                if license_name and license_name != 'UNKNOWN':
                                    component_data['licenses'] = [
                                        {'license': {'name': license_name}}
                                    ]
                                break
                except Exception:
                    pass
                
                components.append(component_data)
        
        except Exception as e:
            print(f"Warning: Could not enumerate Python packages: {e}")
        
        return components
    
    def generate_system_dependencies(self) -> List[Dict[str, Any]]:
        """Generate system dependency components."""
        components = []
        
        # Operating system
        os_info = platform.uname()
        components.append({
            "type": "operating-system",
            "bom-ref": f"os-{os_info.system.lower()}-{os_info.release}",
            "name": os_info.system,
            "version": os_info.release,
            "description": f"{os_info.system} {os_info.release} on {os_info.machine}"
        })
        
        # Python runtime
        python_version = platform.python_version()
        components.append({
            "type": "framework",
            "bom-ref": f"python-{python_version}",
            "name": "Python",
            "version": python_version,
            "description": f"Python runtime {python_version}",
            "purl": f"pkg:generic/python@{python_version}"
        })
        
        # Check for CUDA if available
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                cuda_version = result.stdout.strip().split('\n')[0]
                components.append({
                    "type": "framework",
                    "bom-ref": f"cuda-{cuda_version}",
                    "name": "NVIDIA CUDA",
                    "version": cuda_version,
                    "description": "NVIDIA CUDA GPU computing platform"
                })
        except FileNotFoundError:
            pass
        
        # Check for Docker if available
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                docker_version = result.stdout.strip().split()[2].rstrip(',')
                components.append({
                    "type": "framework",
                    "bom-ref": f"docker-{docker_version}",
                    "name": "Docker",
                    "version": docker_version,
                    "description": "Docker container platform"
                })
        except FileNotFoundError:
            pass
        
        return components
    
    def generate_model_components(self) -> List[Dict[str, Any]]:
        """Generate ML model components."""
        components = []
        
        # Look for model files
        model_extensions = ['.pt', '.pth', '.onnx', '.trt', '.engine', '.h5', '.pb']
        model_dirs = ['models', 'weights', 'checkpoints']
        
        for model_dir in model_dirs:
            model_path = self.project_root / model_dir
            if model_path.exists():
                for model_file in model_path.rglob('*'):
                    if model_file.is_file() and model_file.suffix.lower() in model_extensions:
                        file_hash = self._calculate_file_hash(model_file)
                        file_size = model_file.stat().st_size
                        
                        components.append({
                            "type": "data",
                            "bom-ref": f"model-{model_file.stem}",
                            "name": model_file.name,
                            "version": "unknown",
                            "description": f"ML model file ({model_file.suffix})",
                            "hashes": [
                                {
                                    "alg": "SHA-256",
                                    "content": file_hash
                                }
                            ],
                            "properties": [
                                {
                                    "name": "file_size",
                                    "value": str(file_size)
                                },
                                {
                                    "name": "file_path",
                                    "value": str(model_file.relative_to(self.project_root))
                                }
                            ]
                        })
        
        return components
    
    def generate_configuration_components(self) -> List[Dict[str, Any]]:
        """Generate configuration file components."""
        components = []
        
        # Configuration files
        config_files = [
            'config.yaml', 'config.yml', 'config.json',
            'settings.yaml', 'settings.yml', 'settings.json',
            '.env.example', 'docker-compose.yml', 'Dockerfile'
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                file_hash = self._calculate_file_hash(config_path)
                
                components.append({
                    "type": "data",
                    "bom-ref": f"config-{config_file}",
                    "name": config_file,
                    "version": "1.0.0",
                    "description": f"Configuration file: {config_file}",
                    "hashes": [
                        {
                            "alg": "SHA-256",
                            "content": file_hash
                        }
                    ]
                })
        
        return components
    
    def generate_vulnerability_report(self) -> List[Dict[str, Any]]:
        """Generate vulnerability report using pip-audit."""
        vulnerabilities = []
        
        try:
            # Run pip-audit to get vulnerability information
            result = subprocess.run(
                ['pip-audit', '--format=json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                
                for vuln in audit_data.get('vulnerabilities', []):
                    vulnerability = {
                        "bom-ref": f"vuln-{vuln.get('id', 'unknown')}",
                        "id": vuln.get('id'),
                        "source": {
                            "name": "PyPI Advisory Database",
                            "url": "https://github.com/pypa/advisory-database"
                        },
                        "ratings": [
                            {
                                "source": {
                                    "name": "PyPI"
                                },
                                "severity": vuln.get('severity', 'unknown').upper()
                            }
                        ],
                        "description": vuln.get('description', ''),
                        "affects": [
                            {
                                "ref": f"python-{vuln.get('package')}-{vuln.get('installed_version')}"
                            }
                        ]
                    }
                    
                    if vuln.get('fix_versions'):
                        vulnerability['recommendation'] = f"Upgrade to version {', '.join(vuln['fix_versions'])}"
                    
                    vulnerabilities.append(vulnerability)
        
        except (FileNotFoundError, json.JSONDecodeError, subprocess.SubprocessError) as e:
            print(f"Warning: Could not generate vulnerability report: {e}")
        
        return vulnerabilities
    
    def generate_docker_components(self) -> List[Dict[str, Any]]:
        """Generate Docker image components."""
        components = []
        
        # Look for Dockerfiles
        dockerfiles = list(self.project_root.rglob('Dockerfile*'))
        
        for dockerfile in dockerfiles:
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                
                # Extract base images
                import re
                from_lines = re.findall(r'^FROM\s+([^\s]+)', content, re.MULTILINE)
                
                for base_image in from_lines:
                    if ':' in base_image:
                        name, tag = base_image.split(':', 1)
                    else:
                        name, tag = base_image, 'latest'
                    
                    components.append({
                        "type": "container",
                        "bom-ref": f"docker-{name.replace('/', '-')}-{tag}",
                        "name": name,
                        "version": tag,
                        "description": f"Docker base image: {base_image}",
                        "purl": f"pkg:docker/{name}@{tag}"
                    })
            
            except Exception as e:
                print(f"Warning: Could not parse Dockerfile {dockerfile}: {e}")
        
        return components
    
    def generate_full_sbom(self) -> Dict[str, Any]:
        """Generate complete SBOM."""
        print("Generating Software Bill of Materials...")
        
        # Collect all components
        all_components = []
        
        print("  - Python dependencies...")
        all_components.extend(self.generate_python_dependencies())
        
        print("  - System dependencies...")
        all_components.extend(self.generate_system_dependencies())
        
        print("  - ML models...")
        all_components.extend(self.generate_model_components())
        
        print("  - Configuration files...")
        all_components.extend(self.generate_configuration_components())
        
        print("  - Docker components...")
        all_components.extend(self.generate_docker_components())
        
        print("  - Vulnerability report...")
        vulnerabilities = self.generate_vulnerability_report()
        
        # Update SBOM data
        self.sbom_data['components'] = all_components
        self.sbom_data['vulnerabilities'] = vulnerabilities
        
        # Add statistics
        self.sbom_data['metadata']['properties'] = [
            {
                "name": "total_components",
                "value": str(len(all_components))
            },
            {
                "name": "total_vulnerabilities",
                "value": str(len(vulnerabilities))
            },
            {
                "name": "generation_time",
                "value": self.timestamp
            },
            {
                "name": "generator_platform",
                "value": f"{platform.system()} {platform.release()}"
            }
        ]
        
        print(f"Generated SBOM with {len(all_components)} components and {len(vulnerabilities)} vulnerabilities")
        
        return self.sbom_data
    
    def save_sbom(self, output_path: Path, format_type: str = 'json') -> None:
        """Save SBOM to file."""
        sbom_data = self.generate_full_sbom()
        
        if format_type.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(sbom_data, f, indent=2)
        elif format_type.lower() == 'xml':
            # Convert to CycloneDX XML format if library is available
            if bom and component:
                try:
                    # This would require more complex conversion
                    # For now, save as JSON
                    with open(output_path, 'w') as f:
                        json.dump(sbom_data, f, indent=2)
                    print("Note: XML format requires additional implementation")
                except Exception as e:
                    print(f"Error generating XML format: {e}")
                    # Fallback to JSON
                    json_path = output_path.with_suffix('.json')
                    with open(json_path, 'w') as f:
                        json.dump(sbom_data, f, indent=2)
            else:
                print("CycloneDX library not available, saving as JSON")
                json_path = output_path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(sbom_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print(f"SBOM saved to: {output_path}")
    
    def generate_summary_report(self) -> str:
        """Generate human-readable summary report."""
        sbom_data = self.sbom_data
        
        report = []
        report.append("# Foresight SAR System - Software Bill of Materials")
        report.append(f"Generated: {self.timestamp}")
        report.append(f"Version: {sbom_data['metadata']['component']['version']}")
        report.append("")
        
        # Component summary
        components = sbom_data.get('components', [])
        component_types = {}
        for comp in components:
            comp_type = comp.get('type', 'unknown')
            component_types[comp_type] = component_types.get(comp_type, 0) + 1
        
        report.append("## Component Summary")
        for comp_type, count in sorted(component_types.items()):
            report.append(f"- {comp_type.title()}: {count}")
        report.append(f"- **Total Components: {len(components)}**")
        report.append("")
        
        # Vulnerability summary
        vulnerabilities = sbom_data.get('vulnerabilities', [])
        if vulnerabilities:
            vuln_severities = {}
            for vuln in vulnerabilities:
                severity = 'unknown'
                if vuln.get('ratings'):
                    severity = vuln['ratings'][0].get('severity', 'unknown').lower()
                vuln_severities[severity] = vuln_severities.get(severity, 0) + 1
            
            report.append("## Security Vulnerabilities")
            for severity, count in sorted(vuln_severities.items()):
                report.append(f"- {severity.title()}: {count}")
            report.append(f"- **Total Vulnerabilities: {len(vulnerabilities)}**")
        else:
            report.append("## Security Vulnerabilities")
            report.append("No known vulnerabilities detected.")
        
        report.append("")
        
        # Python packages
        python_packages = [c for c in components if c.get('type') == 'library']
        if python_packages:
            report.append("## Python Dependencies")
            for pkg in sorted(python_packages, key=lambda x: x.get('name', '')):
                report.append(f"- {pkg.get('name', 'unknown')} {pkg.get('version', 'unknown')}")
            report.append("")
        
        # Models
        models = [c for c in components if c.get('type') == 'data' and 'model' in c.get('bom-ref', '')]
        if models:
            report.append("## ML Models")
            for model in models:
                report.append(f"- {model.get('name', 'unknown')}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate Software Bill of Materials for Foresight SAR System"
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd(),
        help='Project root directory (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('foresight-sar-sbom.json'),
        help='Output file path (default: foresight-sar-sbom.json)'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'xml'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Generate human-readable summary report'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if not args.project_root.exists():
        print(f"Error: Project root directory does not exist: {args.project_root}")
        sys.exit(1)
    
    try:
        generator = SBOMGenerator(args.project_root)
        
        # Generate and save SBOM
        generator.save_sbom(args.output, args.format)
        
        # Generate summary report if requested
        if args.summary:
            summary_path = args.output.with_suffix('.md')
            summary_report = generator.generate_summary_report()
            
            with open(summary_path, 'w') as f:
                f.write(summary_report)
            
            print(f"Summary report saved to: {summary_path}")
            
            if args.verbose:
                print("\n" + summary_report)
        
        print("\nSBOM generation completed successfully!")
        
    except Exception as e:
        print(f"Error generating SBOM: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()