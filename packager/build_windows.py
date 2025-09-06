#!/usr/bin/env python3
"""
Windows packaging script for Foresight SAR
Creates standalone executable using PyInstaller and Inno Setup installer
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
import argparse
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WindowsPackager:
    def __init__(self, source_dir, output_dir, version="1.0.0"):
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.version = version
        self.temp_dir = None
        
        # Validate source directory
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.source_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Version: {self.version}")
    
    def check_dependencies(self):
        """Check if required tools are available"""
        logger.info("Checking dependencies...")
        
        # Check Python
        try:
            python_version = sys.version_info
            if python_version.major != 3 or python_version.minor < 8:
                raise RuntimeError(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            logger.info(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} ✓")
        except Exception as e:
            logger.error(f"Python check failed: {e}")
            return False
        
        # Check PyInstaller
        try:
            result = subprocess.run(['pyinstaller', '--version'], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"PyInstaller {result.stdout.strip()} ✓")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("PyInstaller not found. Install with: pip install pyinstaller")
            return False
        
        # Check Inno Setup (optional)
        inno_paths = [
            r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            r"C:\Program Files\Inno Setup 6\ISCC.exe",
            r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
            r"C:\Program Files\Inno Setup 5\ISCC.exe"
        ]
        
        inno_found = False
        for inno_path in inno_paths:
            if Path(inno_path).exists():
                self.inno_compiler = inno_path
                logger.info(f"Inno Setup found at {inno_path} ✓")
                inno_found = True
                break
        
        if not inno_found:
            logger.warning("Inno Setup not found. Installer creation will be skipped.")
            self.inno_compiler = None
        
        return True
    
    def create_spec_file(self):
        """Create PyInstaller spec file"""
        logger.info("Creating PyInstaller spec file...")
        
        spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Add source directory to path
sys.path.insert(0, r"{self.source_dir}")

block_cipher = None

# Main application analysis
a = Analysis(
    [r"{self.source_dir / 'src' / 'backend' / 'app.py'}"],
    pathex=[r"{self.source_dir}", r"{self.source_dir / 'src'}"],
    binaries=[
        # Add any binary dependencies here
    ],
    datas=[
        # Configuration files
        (r"{self.source_dir / 'config'}", "config"),
        # Models directory (if exists)
        (r"{self.source_dir / 'models'}", "models"),
        # Static files
        (r"{self.source_dir / 'src' / 'frontend' / 'static'}", "static"),
        # Templates
        (r"{self.source_dir / 'src' / 'frontend' / 'templates'}", "templates"),
    ],
    hiddenimports=[
        # Core dependencies
        'flask',
        'flask_cors',
        'flask_jwt_extended',
        'gunicorn',
        'gevent',
        # ML dependencies
        'torch',
        'torchvision',
        'ultralytics',
        'opencv-python',
        'pillow',
        'numpy',
        'scipy',
        'scikit-learn',
        'scikit-image',
        # Data processing
        'pandas',
        'h5py',
        'pyarrow',
        # Geospatial
        'pyproj',
        'shapely',
        'rasterio',
        # Cryptography
        'cryptography',
        'pycryptodome',
        'hashicorp-vault',
        # Utilities
        'pyyaml',
        'toml',
        'click',
        'tqdm',
        'psutil',
        'requests',
        'websockets',
        # Monitoring
        'prometheus_client',
        'structlog',
        # Testing
        'pytest',
        # Windows-specific
        'win32api',
        'win32con',
        'win32gui',
        'win32service',
        'win32serviceutil',
        'winsound',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib',
        'jupyter',
        'notebook',
        'IPython',
        'sphinx',
        'pytest',
        'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary files
a.datas = [x for x in a.datas if not any([
    'test' in x[0].lower(),
    'example' in x[0].lower(),
    'demo' in x[0].lower(),
    '.git' in x[0],
    '__pycache__' in x[0],
    '.pyc' in x[0],
])]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ForesightSAR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for windowed app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=r"{self.source_dir / 'assets' / 'icon.ico'}" if (self.source_dir / 'assets' / 'icon.ico').exists() else None,
    version_file=r"{self.output_dir / 'version_info.txt'}",
)

# Create distribution directory
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ForesightSAR',
)
'''
        
        spec_file = self.output_dir / 'ForesightSAR.spec'
        with open(spec_file, 'w') as f:
            f.write(spec_content)
        
        logger.info(f"Spec file created: {spec_file}")
        return spec_file
    
    def create_version_info(self):
        """Create version info file for Windows executable"""
        logger.info("Creating version info file...")
        
        version_parts = self.version.split('.')
        while len(version_parts) < 4:
            version_parts.append('0')
        
        version_info = f'''
# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
    # filevers and prodvers should be always a tuple with four items: (1, 2, 3, 4)
    # Set not needed items to zero 0.
    filevers=({version_parts[0]}, {version_parts[1]}, {version_parts[2]}, {version_parts[3]}),
    prodvers=({version_parts[0]}, {version_parts[1]}, {version_parts[2]}, {version_parts[3]}),
    # Contains a bitmask that specifies the valid bits 'flags'r
    mask=0x3f,
    # Contains a bitmask that specifies the Boolean attributes of the file.
    flags=0x0,
    # The operating system for which this file was designed.
    # 0x4 - NT and there is no need to change it.
    OS=0x4,
    # The general type of file.
    # 0x1 - the file is an application.
    fileType=0x1,
    # The function of the file.
    # 0x0 - the function is not defined for this fileType
    subtype=0x0,
    # Creation date and time stamp.
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'Foresight SAR Team'),
          StringStruct(u'FileDescription', u'Foresight Search and Rescue System'),
          StringStruct(u'FileVersion', u'{self.version}'),
          StringStruct(u'InternalName', u'ForesightSAR'),
          StringStruct(u'LegalCopyright', u'Copyright © 2024 Foresight SAR Team'),
          StringStruct(u'OriginalFilename', u'ForesightSAR.exe'),
          StringStruct(u'ProductName', u'Foresight SAR'),
          StringStruct(u'ProductVersion', u'{self.version}')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
        
        version_file = self.output_dir / 'version_info.txt'
        with open(version_file, 'w', encoding='utf-8') as f:
            f.write(version_info)
        
        logger.info(f"Version info created: {version_file}")
        return version_file
    
    def build_executable(self, spec_file):
        """Build executable using PyInstaller"""
        logger.info("Building executable with PyInstaller...")
        
        try:
            # Run PyInstaller
            cmd = [
                'pyinstaller',
                '--clean',
                '--noconfirm',
                '--distpath', str(self.output_dir / 'dist'),
                '--workpath', str(self.output_dir / 'build'),
                str(spec_file)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("PyInstaller completed successfully")
            logger.debug(f"PyInstaller output: {result.stdout}")
            
            # Check if executable was created
            exe_path = self.output_dir / 'dist' / 'ForesightSAR' / 'ForesightSAR.exe'
            if exe_path.exists():
                logger.info(f"Executable created: {exe_path}")
                return exe_path
            else:
                raise RuntimeError("Executable not found after build")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"PyInstaller failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def create_installer_script(self, exe_path):
        """Create Inno Setup script for installer"""
        if not self.inno_compiler:
            logger.warning("Inno Setup not available, skipping installer creation")
            return None
        
        logger.info("Creating Inno Setup script...")
        
        dist_dir = exe_path.parent
        
        installer_script = f'''
; Foresight SAR Installer Script
; Generated automatically by build script

#define MyAppName "Foresight SAR"
#define MyAppVersion "{self.version}"
#define MyAppPublisher "Foresight SAR Team"
#define MyAppURL "https://github.com/foresight-sar/foresight"
#define MyAppExeName "ForesightSAR.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}}
AppName={{#MyAppName}}
AppVersion={{#MyAppVersion}}
;AppVerName={{#MyAppName}} {{#MyAppVersion}}
AppPublisher={{#MyAppPublisher}}
AppPublisherURL={{#MyAppURL}}
AppSupportURL={{#MyAppURL}}
AppUpdatesURL={{#MyAppURL}}
DefaultDirName={{autopf}}\{{#MyAppName}}
DefaultGroupName={{#MyAppName}}
AllowNoIcons=yes
LicenseFile={self.source_dir / 'LICENSE'}
OutputDir={self.output_dir / 'installer'}
OutputBaseFilename=ForesightSAR-{self.version}-Setup
SetupIconFile={self.source_dir / 'assets' / 'icon.ico'}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{{cm:CreateQuickLaunchIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked; OnlyBelowVersion: 6.1
Name: "service"; Description: "Install as Windows Service"; GroupDescription: "Service Options"; Flags: unchecked

[Files]
Source: "{dist_dir}\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{{group}}\{{#MyAppName}}"; Filename: "{{app}}\{{#MyAppExeName}}"
Name: "{{group}}\{{cm:ProgramOnTheWeb,{{#MyAppName}}}}"; Filename: "{{#MyAppURL}}"
Name: "{{group}}\{{cm:UninstallProgram,{{#MyAppName}}}}"; Filename: "{{uninstallexe}}"
Name: "{{autodesktop}}\{{#MyAppName}}"; Filename: "{{app}}\{{#MyAppExeName}}"; Tasks: desktopicon
Name: "{{userappdata}}\Microsoft\Internet Explorer\Quick Launch\{{#MyAppName}}"; Filename: "{{app}}\{{#MyAppExeName}}"; Tasks: quicklaunchicon

[Run]
Filename: "{{app}}\{{#MyAppExeName}}"; Description: "{{cm:LaunchProgram,{{#StringChange(MyAppName, '&', '&&')}}}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{{app}}\logs"
Type: filesandordirs; Name: "{{app}}\cache"
Type: filesandordirs; Name: "{{app}}\temp"

[Code]
// Custom installation code
function InitializeSetup(): Boolean;
begin
  Result := True;
  // Check for .NET Framework or other prerequisites here
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Post-installation tasks
    if IsTaskSelected('service') then
    begin
      // Install Windows service
      Exec(ExpandConstant('{{app}}\{{#MyAppExeName}}'), '--install-service', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    // Stop and remove service if installed
    Exec(ExpandConstant('{{app}}\{{#MyAppExeName}}'), '--remove-service', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
'''
        
        script_file = self.output_dir / 'installer_script.iss'
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(installer_script)
        
        logger.info(f"Installer script created: {script_file}")
        return script_file
    
    def build_installer(self, script_file):
        """Build installer using Inno Setup"""
        if not self.inno_compiler or not script_file:
            logger.warning("Skipping installer build")
            return None
        
        logger.info("Building installer with Inno Setup...")
        
        try:
            # Create installer output directory
            installer_dir = self.output_dir / 'installer'
            installer_dir.mkdir(exist_ok=True)
            
            # Run Inno Setup compiler
            cmd = [self.inno_compiler, str(script_file)]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("Installer created successfully")
            logger.debug(f"Inno Setup output: {result.stdout}")
            
            # Find the created installer
            installer_files = list(installer_dir.glob('*.exe'))
            if installer_files:
                installer_path = installer_files[0]
                logger.info(f"Installer created: {installer_path}")
                return installer_path
            else:
                logger.warning("Installer file not found")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Inno Setup failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return None
    
    def create_portable_package(self, exe_path):
        """Create portable ZIP package"""
        logger.info("Creating portable package...")
        
        try:
            import zipfile
            
            dist_dir = exe_path.parent
            zip_path = self.output_dir / f'ForesightSAR-{self.version}-Portable.zip'
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from dist directory
                for file_path in dist_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(dist_dir)
                        zipf.write(file_path, arcname)
                
                # Add configuration files
                config_dir = self.source_dir / 'config'
                if config_dir.exists():
                    for config_file in config_dir.rglob('*'):
                        if config_file.is_file():
                            arcname = Path('config') / config_file.relative_to(config_dir)
                            zipf.write(config_file, arcname)
                
                # Add README for portable version
                readme_content = f'''
Foresight SAR - Portable Version {self.version}
================================================

This is a portable version of Foresight SAR that doesn't require installation.

To run:
1. Extract this ZIP file to any folder
2. Run ForesightSAR.exe

Configuration:
- Configuration files are in the 'config' folder
- Logs will be created in the 'logs' folder
- Data will be stored in the 'data' folder

System Requirements:
- Windows 10 or later (64-bit)
- 8GB RAM minimum, 16GB recommended
- NVIDIA GPU with CUDA support (optional but recommended)
- 10GB free disk space

For support and documentation:
https://github.com/foresight-sar/foresight
'''
                
                zipf.writestr('README.txt', readme_content)
            
            logger.info(f"Portable package created: {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Failed to create portable package: {e}")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files...")
        
        # Remove build directory
        build_dir = self.output_dir / 'build'
        if build_dir.exists():
            shutil.rmtree(build_dir)
            logger.info("Build directory removed")
        
        # Remove spec file
        spec_file = self.output_dir / 'ForesightSAR.spec'
        if spec_file.exists():
            spec_file.unlink()
            logger.info("Spec file removed")
    
    def package(self, create_installer=True, create_portable=True, cleanup=True):
        """Main packaging function"""
        logger.info("Starting Windows packaging process...")
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                raise RuntimeError("Dependency check failed")
            
            # Create version info
            self.create_version_info()
            
            # Create spec file
            spec_file = self.create_spec_file()
            
            # Build executable
            exe_path = self.build_executable(spec_file)
            
            results = {'executable': exe_path}
            
            # Create installer
            if create_installer:
                script_file = self.create_installer_script(exe_path)
                installer_path = self.build_installer(script_file)
                results['installer'] = installer_path
            
            # Create portable package
            if create_portable:
                portable_path = self.create_portable_package(exe_path)
                results['portable'] = portable_path
            
            # Cleanup
            if cleanup:
                self.cleanup()
            
            logger.info("Packaging completed successfully!")
            
            # Print results
            logger.info("\nPackaging Results:")
            logger.info("=" * 50)
            for package_type, path in results.items():
                if path:
                    logger.info(f"{package_type.capitalize()}: {path}")
                else:
                    logger.info(f"{package_type.capitalize()}: Not created")
            
            return results
            
        except Exception as e:
            logger.error(f"Packaging failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Package Foresight SAR for Windows')
    parser.add_argument('--source', '-s', default='.', help='Source directory (default: current directory)')
    parser.add_argument('--output', '-o', default='./dist', help='Output directory (default: ./dist)')
    parser.add_argument('--version', '-v', default='1.0.0', help='Version number (default: 1.0.0)')
    parser.add_argument('--no-installer', action='store_true', help='Skip installer creation')
    parser.add_argument('--no-portable', action='store_true', help='Skip portable package creation')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup of temporary files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        packager = WindowsPackager(
            source_dir=args.source,
            output_dir=args.output,
            version=args.version
        )
        
        results = packager.package(
            create_installer=not args.no_installer,
            create_portable=not args.no_portable,
            cleanup=not args.no_cleanup
        )
        
        print("\n✅ Packaging completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Packaging failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())