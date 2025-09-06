# -*- mode: python ; coding: utf-8 -*-
# Foresight SAR System - PyInstaller Specification
# Creates standalone Windows executable with all dependencies

import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Define paths
app_name = 'Foresight-SAR'
icon_path = str(project_root / 'assets' / 'icon.png')
main_script = str(project_root / 'main.py')

# Data files to include
datas = [
    # Configuration files
    (str(project_root / 'configs'), 'configs'),
    
    # UI assets
    (str(project_root / 'assets'), 'assets'),
    (str(project_root / 'foresight-electron'), 'ui'),
    
    # Models directory (empty but structure needed)
    (str(project_root / 'models'), 'models'),
    
    # Documentation
    (str(project_root / 'docs'), 'docs'),
    
    # Scripts
    (str(project_root / 'scripts'), 'scripts'),
]

# Hidden imports for dynamic loading
hiddenimports = [
    # Core dependencies
    'uvicorn',
    'fastapi',
    'websockets',
    'jinja2',
    'multipart',
    
    # ML/AI frameworks
    'torch',
    'torchvision',
    'ultralytics',
    'cv2',
    'PIL',
    'numpy',
    'scipy',
    
    # Geospatial
    'rasterio',
    'fiona',
    'shapely',
    'pyproj',
    'geopy',
    
    # Database
    'sqlalchemy',
    'psycopg2',
    'redis',
    
    # Cryptography
    'cryptography',
    'Crypto',
    
    # Utilities
    'yaml',
    'click',
    'rich',
    'tqdm',
    'psutil',
    'watchdog',
    'loguru',
    
    # Windows-specific
    'win32api',
    'win32con',
    'win32gui',
    'win32process',
    'winsound',
    
    # Application modules
    'src.backend.app',
    'src.backend.detection_service',
    'src.backend.geolocation_service',
    'src.backend.sar_service',
    'connection.dji_o4',
    'connection.phone_stream',
    'tracking.deepsort_tracker',
    'geolocation.geolocation_service',
    'ui.offline_maps',
    'ui.heatmap_generator',
    'ui.snapshot_packaging',
]

# Binaries to exclude (will be handled by conda/pip)
excludes = [
    'tkinter',
    'matplotlib',
    'IPython',
    'jupyter',
    'notebook',
    'sphinx',
    'pytest',
    'setuptools',
    'distutils',
]

# Analysis configuration
a = Analysis(
    [main_script],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate files
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for windowed app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path if os.path.exists(icon_path) else None,
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
    name=app_name,
)

# Optional: Create one-file executable (larger but portable)
# Uncomment the following to create a single executable file
"""
exe_onefile = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=f'{app_name}-Portable',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path if os.path.exists(icon_path) else None,
)
"""