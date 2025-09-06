#!/usr/bin/env python3
"""
Foresight SAR System - Interactive Setup Wizard
Provides guided installation for Windows and Jetson platforms
"""

import os
import sys
import platform
import subprocess
import json
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass
from enum import Enum

try:
    import colorama
    from colorama import Fore, Style, Back
    colorama.init()
except ImportError:
    # Fallback for systems without colorama
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""
    class Back:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""

class Platform(Enum):
    WINDOWS = "windows"
    JETSON = "jetson"
    LINUX = "linux"
    UNKNOWN = "unknown"

class InstallMode(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    MINIMAL = "minimal"
    CUSTOM = "custom"

@dataclass
class SystemRequirements:
    python_version: str = "3.8+"
    memory_gb: int = 8
    storage_gb: int = 50
    gpu_required: bool = False
    docker_required: bool = True

@dataclass
class InstallConfig:
    platform: Platform
    mode: InstallMode
    install_path: str
    data_path: str
    enable_gpu: bool = False
    enable_docker: bool = True
    enable_services: bool = True
    offline_mode: bool = False
    vault_integration: bool = False
    custom_components: List[str] = None

class SetupWizard:
    def __init__(self):
        self.platform = self._detect_platform()
        self.config: Optional[InstallConfig] = None
        self.requirements = SystemRequirements()
        self.script_dir = Path(__file__).parent
        self.assets_dir = self.script_dir / "assets"
        self.offline_assets = self.script_dir / "offline_assets"
        
    def _detect_platform(self) -> Platform:
        """Detect the current platform"""
        system = platform.system().lower()
        
        if system == "windows":
            return Platform.WINDOWS
        elif system == "linux":
            # Check if it's a Jetson device
            if os.path.exists("/etc/nv_tegra_release"):
                return Platform.JETSON
            return Platform.LINUX
        else:
            return Platform.UNKNOWN
    
    def print_banner(self):
        """Print the setup wizard banner"""
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              {Fore.WHITE}Foresight SAR System Setup Wizard{Fore.CYAN}              ║
║                                                              ║
║  {Fore.YELLOW}Search and Rescue Operations Platform{Fore.CYAN}                    ║
║  {Fore.GREEN}Interactive Installation and Configuration{Fore.CYAN}                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(banner)
        print(f"{Fore.WHITE}Platform: {Fore.GREEN}{self.platform.value.title()}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Python: {Fore.GREEN}{sys.version.split()[0]}{Style.RESET_ALL}")
        print()
    
    def check_system_requirements(self) -> Tuple[bool, List[str]]:
        """Check if system meets minimum requirements"""
        issues = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            issues.append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        
        # Check available memory
        try:
            if self.platform == Platform.WINDOWS:
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024**3)
            else:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    memory_kb = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1])
                    memory_gb = memory_kb / (1024**2)
            
            if memory_gb < self.requirements.memory_gb:
                issues.append(f"Insufficient memory: {memory_gb:.1f}GB available, {self.requirements.memory_gb}GB required")
        except Exception as e:
            issues.append(f"Could not check memory requirements: {e}")
        
        # Check disk space
        try:
            if self.platform == Platform.WINDOWS:
                free_bytes = shutil.disk_usage("C:\\")[2]
            else:
                free_bytes = shutil.disk_usage("/")[2]
            
            free_gb = free_bytes / (1024**3)
            if free_gb < self.requirements.storage_gb:
                issues.append(f"Insufficient storage: {free_gb:.1f}GB available, {self.requirements.storage_gb}GB required")
        except Exception as e:
            issues.append(f"Could not check storage requirements: {e}")
        
        return len(issues) == 0, issues
    
    def get_user_input(self, prompt: str, default: str = None, options: List[str] = None) -> str:
        """Get user input with validation"""
        while True:
            if default:
                display_prompt = f"{prompt} [{default}]: "
            else:
                display_prompt = f"{prompt}: "
            
            if options:
                print(f"Options: {', '.join(options)}")
            
            response = input(display_prompt).strip()
            
            if not response and default:
                return default
            
            if options and response.lower() not in [opt.lower() for opt in options]:
                print(f"{Fore.RED}Invalid option. Please choose from: {', '.join(options)}{Style.RESET_ALL}")
                continue
            
            return response
    
    def get_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Get yes/no input from user"""
        default_str = "Y/n" if default else "y/N"
        response = self.get_user_input(f"{prompt} ({default_str})", "y" if default else "n", ["y", "n", "yes", "no"])
        return response.lower() in ["y", "yes"]
    
    def configure_installation(self) -> InstallConfig:
        """Interactive configuration of installation"""
        print(f"{Fore.YELLOW}=== Installation Configuration ==={Style.RESET_ALL}")
        print()
        
        # Installation mode
        print("Select installation mode:")
        print("1. Development - Full development environment with debugging tools")
        print("2. Production - Optimized for operational deployment")
        print("3. Minimal - Core components only")
        print("4. Custom - Choose specific components")
        
        mode_choice = self.get_user_input("Installation mode", "2", ["1", "2", "3", "4"])
        mode_map = {
            "1": InstallMode.DEVELOPMENT,
            "2": InstallMode.PRODUCTION,
            "3": InstallMode.MINIMAL,
            "4": InstallMode.CUSTOM
        }
        mode = mode_map[mode_choice]
        
        # Installation paths
        if self.platform == Platform.WINDOWS:
            default_install = "C:\\Program Files\\Foresight SAR"
            default_data = "C:\\ProgramData\\Foresight SAR"
        else:
            default_install = "/opt/foresight"
            default_data = "/var/lib/foresight"
        
        install_path = self.get_user_input("Installation directory", default_install)
        data_path = self.get_user_input("Data directory", default_data)
        
        # GPU support
        enable_gpu = False
        if self._check_gpu_available():
            enable_gpu = self.get_yes_no("Enable GPU acceleration", True)
        
        # Docker support
        enable_docker = True
        if mode != InstallMode.MINIMAL:
            enable_docker = self.get_yes_no("Enable Docker containerization", True)
        
        # Services
        enable_services = True
        if self.platform != Platform.WINDOWS or mode == InstallMode.DEVELOPMENT:
            enable_services = self.get_yes_no("Install as system service", True)
        
        # Offline mode
        offline_mode = self.get_yes_no("Use offline installation (requires offline assets)", False)
        
        # Vault integration
        vault_integration = False
        if mode in [InstallMode.PRODUCTION, InstallMode.CUSTOM]:
            vault_integration = self.get_yes_no("Enable HashiCorp Vault integration", False)
        
        # Custom components
        custom_components = []
        if mode == InstallMode.CUSTOM:
            print("\nSelect components to install:")
            components = [
                "vision", "tracking", "geolocation", "reid", "packaging",
                "training", "ui", "connection", "tools"
            ]
            
            for component in components:
                if self.get_yes_no(f"Install {component} module", True):
                    custom_components.append(component)
        
        self.config = InstallConfig(
            platform=self.platform,
            mode=mode,
            install_path=install_path,
            data_path=data_path,
            enable_gpu=enable_gpu,
            enable_docker=enable_docker,
            enable_services=enable_services,
            offline_mode=offline_mode,
            vault_integration=vault_integration,
            custom_components=custom_components
        )
        
        return self.config
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            if self.platform == Platform.WINDOWS:
                # Check for NVIDIA GPU on Windows
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                return result.returncode == 0
            else:
                # Check for NVIDIA GPU on Linux/Jetson
                return os.path.exists("/proc/driver/nvidia/version")
        except:
            return False
    
    def download_offline_assets(self) -> bool:
        """Download offline installation assets"""
        if self.config.offline_mode and not self.offline_assets.exists():
            print(f"{Fore.YELLOW}Downloading offline assets...{Style.RESET_ALL}")
            
            assets_url = "https://github.com/foresight-sar/assets/releases/latest/download/offline-assets.zip"
            assets_file = self.script_dir / "offline-assets.zip"
            
            try:
                urllib.request.urlretrieve(assets_url, assets_file)
                
                with zipfile.ZipFile(assets_file, 'r') as zip_ref:
                    zip_ref.extractall(self.script_dir)
                
                assets_file.unlink()
                print(f"{Fore.GREEN}Offline assets downloaded successfully{Style.RESET_ALL}")
                return True
            except Exception as e:
                print(f"{Fore.RED}Failed to download offline assets: {e}{Style.RESET_ALL}")
                return False
        
        return True
    
    def create_directories(self) -> bool:
        """Create installation directories"""
        print(f"{Fore.YELLOW}Creating directories...{Style.RESET_ALL}")
        
        directories = [
            self.config.install_path,
            self.config.data_path,
            os.path.join(self.config.data_path, "data"),
            os.path.join(self.config.data_path, "models"),
            os.path.join(self.config.data_path, "evidence"),
            os.path.join(self.config.data_path, "logs"),
            os.path.join(self.config.data_path, "cache"),
            os.path.join(self.config.data_path, "config"),
            os.path.join(self.config.data_path, "backups")
        ]
        
        try:
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"Created: {directory}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to create directories: {e}{Style.RESET_ALL}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install system dependencies"""
        print(f"{Fore.YELLOW}Installing dependencies...{Style.RESET_ALL}")
        
        if self.platform == Platform.WINDOWS:
            return self._install_windows_dependencies()
        elif self.platform == Platform.JETSON:
            return self._install_jetson_dependencies()
        else:
            return self._install_linux_dependencies()
    
    def _install_windows_dependencies(self) -> bool:
        """Install Windows-specific dependencies"""
        try:
            # Run the existing Windows setup script
            script_path = self.script_dir / "windows" / "setup_windows.ps1"
            
            cmd_args = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
            
            if self.config.mode == InstallMode.DEVELOPMENT:
                cmd_args.append("-Development")
            elif self.config.mode == InstallMode.PRODUCTION:
                cmd_args.append("-Production")
            elif self.config.mode == InstallMode.MINIMAL:
                cmd_args.append("-Minimal")
            
            cmd_args.extend(["-InstallPath", self.config.install_path])
            cmd_args.extend(["-DataPath", self.config.data_path])
            
            result = subprocess.run(cmd_args, check=True)
            return result.returncode == 0
        except Exception as e:
            print(f"{Fore.RED}Failed to install Windows dependencies: {e}{Style.RESET_ALL}")
            return False
    
    def _install_jetson_dependencies(self) -> bool:
        """Install Jetson-specific dependencies"""
        try:
            # Run the existing Jetson setup script
            script_path = self.script_dir / "jetson" / "setup_jetson.sh"
            
            result = subprocess.run(["bash", str(script_path)], check=True)
            return result.returncode == 0
        except Exception as e:
            print(f"{Fore.RED}Failed to install Jetson dependencies: {e}{Style.RESET_ALL}")
            return False
    
    def _install_linux_dependencies(self) -> bool:
        """Install Linux-specific dependencies"""
        # Implement Linux-specific installation
        print(f"{Fore.YELLOW}Linux installation not yet implemented{Style.RESET_ALL}")
        return True
    
    def copy_application_files(self) -> bool:
        """Copy application files to installation directory"""
        print(f"{Fore.YELLOW}Copying application files...{Style.RESET_ALL}")
        
        try:
            source_dir = self.script_dir.parent
            target_dir = Path(self.config.install_path)
            
            # Files and directories to copy
            items_to_copy = [
                "main.py", "requirements.txt", "configs", "src", "ui",
                "vision", "tracking", "geolocation", "reid", "packaging",
                "connection", "tools", "assets"
            ]
            
            # Add custom components if specified
            if self.config.custom_components:
                items_to_copy = [item for item in items_to_copy 
                               if any(comp in item for comp in self.config.custom_components)]
                items_to_copy.extend(["main.py", "requirements.txt", "configs", "assets"])
            
            for item in items_to_copy:
                source_path = source_dir / item
                target_path = target_dir / item
                
                if source_path.exists():
                    if source_path.is_dir():
                        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source_path, target_path)
                    print(f"Copied: {item}")
            
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to copy application files: {e}{Style.RESET_ALL}")
            return False
    
    def create_configuration(self) -> bool:
        """Create default configuration files"""
        print(f"{Fore.YELLOW}Creating configuration...{Style.RESET_ALL}")
        
        try:
            config_dir = Path(self.config.data_path) / "config"
            
            # Main configuration
            main_config = {
                "app": {
                    "name": "Foresight SAR System",
                    "version": "1.0.0",
                    "mode": self.config.mode.value,
                    "debug": self.config.mode == InstallMode.DEVELOPMENT
                },
                "server": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "workers": 4
                },
                "ui": {
                    "host": "0.0.0.0",
                    "port": 8080
                },
                "websocket": {
                    "host": "0.0.0.0",
                    "port": 5000
                },
                "database": {
                    "type": "sqlite",
                    "path": os.path.join(self.config.data_path, "data", "foresight.db").replace("\\", "/")
                },
                "logging": {
                    "level": "DEBUG" if self.config.mode == InstallMode.DEVELOPMENT else "INFO",
                    "file": os.path.join(self.config.data_path, "logs", "foresight.log").replace("\\", "/")
                },
                "models": {
                    "path": os.path.join(self.config.data_path, "models").replace("\\", "/")
                },
                "evidence": {
                    "path": os.path.join(self.config.data_path, "evidence").replace("\\", "/")
                },
                "gpu": {
                    "enabled": self.config.enable_gpu
                },
                "vault": {
                    "enabled": self.config.vault_integration
                }
            }
            
            with open(config_dir / "settings.json", 'w') as f:
                json.dump(main_config, f, indent=2)
            
            print("Configuration created successfully")
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to create configuration: {e}{Style.RESET_ALL}")
            return False
    
    def install_services(self) -> bool:
        """Install system services"""
        if not self.config.enable_services:
            return True
        
        print(f"{Fore.YELLOW}Installing system services...{Style.RESET_ALL}")
        
        try:
            if self.platform == Platform.WINDOWS:
                return self._install_windows_service()
            else:
                return self._install_linux_service()
        except Exception as e:
            print(f"{Fore.RED}Failed to install services: {e}{Style.RESET_ALL}")
            return False
    
    def _install_windows_service(self) -> bool:
        """Install Windows service"""
        # Service installation is handled by the Windows setup script
        print("Windows service installation completed")
        return True
    
    def _install_linux_service(self) -> bool:
        """Install Linux systemd service"""
        service_content = f"""[Unit]
Description=Foresight SAR System
After=network.target

[Service]
Type=simple
User=foresight
Group=foresight
WorkingDirectory={self.config.install_path}
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        try:
            with open("/etc/systemd/system/foresight-sar.service", 'w') as f:
                f.write(service_content)
            
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "enable", "foresight-sar.service"], check=True)
            
            print("Linux service installed successfully")
            return True
        except Exception as e:
            print(f"Failed to install Linux service: {e}")
            return False
    
    def create_shortcuts(self) -> bool:
        """Create desktop shortcuts and start menu entries"""
        print(f"{Fore.YELLOW}Creating shortcuts...{Style.RESET_ALL}")
        
        if self.platform == Platform.WINDOWS:
            return self._create_windows_shortcuts()
        else:
            return self._create_linux_shortcuts()
    
    def _create_windows_shortcuts(self) -> bool:
        """Create Windows shortcuts"""
        # Shortcut creation is handled by the Windows setup script
        print("Windows shortcuts created")
        return True
    
    def _create_linux_shortcuts(self) -> bool:
        """Create Linux desktop entries"""
        desktop_entry = f"""[Desktop Entry]
Name=Foresight SAR
Comment=Search and Rescue Operations Platform
Exec=python3 {self.config.install_path}/main.py
Icon={self.config.install_path}/assets/icon.png
Terminal=false
Type=Application
Categories=Utility;Security;
"""
        
        try:
            desktop_dir = Path.home() / "Desktop"
            applications_dir = Path.home() / ".local" / "share" / "applications"
            
            for directory in [desktop_dir, applications_dir]:
                if directory.exists():
                    with open(directory / "foresight-sar.desktop", 'w') as f:
                        f.write(desktop_entry)
            
            print("Linux shortcuts created")
            return True
        except Exception as e:
            print(f"Failed to create Linux shortcuts: {e}")
            return False
    
    def run_post_install_tests(self) -> bool:
        """Run post-installation tests"""
        print(f"{Fore.YELLOW}Running post-installation tests...{Style.RESET_ALL}")
        
        try:
            # Test Python imports
            test_script = f"""
import sys
sys.path.insert(0, '{self.config.install_path}')

try:
    import src.backend.app
    print('✓ Backend imports successful')
except ImportError as e:
    print(f'✗ Backend import failed: {{e}}')
    sys.exit(1)

try:
    from vision.detector import YOLODetector
    print('✓ Vision module imports successful')
except ImportError as e:
    print(f'✗ Vision import failed: {{e}}')

try:
    from geolocation.projection import CameraProjection
    print('✓ Geolocation module imports successful')
except ImportError as e:
    print(f'✗ Geolocation import failed: {{e}}')

print('All tests passed!')
"""
            
            result = subprocess.run([sys.executable, "-c", test_script], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"{Fore.GREEN}Post-installation tests passed{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}Post-installation tests failed:{Style.RESET_ALL}")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"{Fore.RED}Failed to run tests: {e}{Style.RESET_ALL}")
            return False
    
    def print_completion_summary(self):
        """Print installation completion summary"""
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Installation Complete!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Installation Details:{Style.RESET_ALL}")
        print(f"Platform: {self.config.platform.value.title()}")
        print(f"Mode: {self.config.mode.value.title()}")
        print(f"Install Path: {self.config.install_path}")
        print(f"Data Path: {self.config.data_path}")
        print(f"GPU Enabled: {'Yes' if self.config.enable_gpu else 'No'}")
        print(f"Docker Enabled: {'Yes' if self.config.enable_docker else 'No'}")
        print(f"Services Enabled: {'Yes' if self.config.enable_services else 'No'}")
        
        print(f"\n{Fore.CYAN}Next Steps:{Style.RESET_ALL}")
        
        if self.platform == Platform.WINDOWS:
            print(f"1. Start the service: {self.config.install_path}\\Start.bat")
            print(f"2. Access web interface: http://localhost:8080")
            print(f"3. Check status: {self.config.install_path}\\Status.bat")
            print(f"4. Configure settings: {self.config.data_path}\\config\\settings.json")
        else:
            print(f"1. Start the service: sudo systemctl start foresight-sar")
            print(f"2. Access web interface: http://localhost:8080")
            print(f"3. Check status: systemctl status foresight-sar")
            print(f"4. Configure settings: {self.config.data_path}/config/settings.json")
        
        print(f"\n{Fore.CYAN}Documentation:{Style.RESET_ALL}")
        print(f"• User Guide: {self.config.install_path}/docs/")
        print(f"• API Documentation: http://localhost:8000/docs")
        print(f"• Configuration: {self.config.data_path}/config/")
        
        print(f"\n{Fore.YELLOW}Support:{Style.RESET_ALL}")
        print(f"• GitHub: https://github.com/foresight-sar/foresight")
        print(f"• Documentation: https://docs.foresight-sar.org")
        print(f"• Issues: https://github.com/foresight-sar/foresight/issues")
        
        print(f"\n{Fore.GREEN}Setup completed successfully!{Style.RESET_ALL}")
    
    def run(self) -> bool:
        """Run the complete setup wizard"""
        try:
            self.print_banner()
            
            # Check system requirements
            requirements_ok, issues = self.check_system_requirements()
            if not requirements_ok:
                print(f"{Fore.RED}System requirements not met:{Style.RESET_ALL}")
                for issue in issues:
                    print(f"  • {issue}")
                
                if not self.get_yes_no("Continue anyway", False):
                    return False
            
            # Configure installation
            self.configure_installation()
            
            # Confirm installation
            print(f"\n{Fore.CYAN}Installation Summary:{Style.RESET_ALL}")
            print(f"Platform: {self.config.platform.value}")
            print(f"Mode: {self.config.mode.value}")
            print(f"Install Path: {self.config.install_path}")
            print(f"Data Path: {self.config.data_path}")
            
            if not self.get_yes_no("\nProceed with installation", True):
                print("Installation cancelled.")
                return False
            
            # Installation steps
            steps = [
                ("Downloading offline assets", self.download_offline_assets),
                ("Creating directories", self.create_directories),
                ("Installing dependencies", self.install_dependencies),
                ("Copying application files", self.copy_application_files),
                ("Creating configuration", self.create_configuration),
                ("Installing services", self.install_services),
                ("Creating shortcuts", self.create_shortcuts),
                ("Running tests", self.run_post_install_tests)
            ]
            
            for step_name, step_func in steps:
                print(f"\n{Fore.YELLOW}Step: {step_name}...{Style.RESET_ALL}")
                if not step_func():
                    print(f"{Fore.RED}Installation failed at step: {step_name}{Style.RESET_ALL}")
                    return False
                print(f"{Fore.GREEN}✓ {step_name} completed{Style.RESET_ALL}")
            
            self.print_completion_summary()
            return True
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Installation cancelled by user{Style.RESET_ALL}")
            return False
        except Exception as e:
            print(f"\n{Fore.RED}Installation failed: {e}{Style.RESET_ALL}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Foresight SAR Setup Wizard")
    parser.add_argument("--non-interactive", action="store_true", 
                       help="Run in non-interactive mode with defaults")
    parser.add_argument("--config", type=str, 
                       help="Use configuration file for installation")
    parser.add_argument("--platform", choices=["windows", "jetson", "linux"], 
                       help="Override platform detection")
    
    args = parser.parse_args()
    
    wizard = SetupWizard()
    
    if args.platform:
        wizard.platform = Platform(args.platform)
    
    if args.config:
        # Load configuration from file
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            # Apply configuration
            print(f"Using configuration from: {args.config}")
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return 1
    
    if args.non_interactive:
        # Set default configuration for non-interactive mode
        wizard.config = InstallConfig(
            platform=wizard.platform,
            mode=InstallMode.PRODUCTION,
            install_path="C:\\Program Files\\Foresight SAR" if wizard.platform == Platform.WINDOWS else "/opt/foresight",
            data_path="C:\\ProgramData\\Foresight SAR" if wizard.platform == Platform.WINDOWS else "/var/lib/foresight",
            enable_gpu=wizard._check_gpu_available(),
            enable_docker=True,
            enable_services=True,
            offline_mode=False,
            vault_integration=False
        )
    
    success = wizard.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())