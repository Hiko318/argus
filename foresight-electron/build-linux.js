#!/usr/bin/env node
/**
 * Linux build script for Foresight SAR Electron app
 * Creates AppImage and Debian packages
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('Building Foresight SAR for Linux...');

try {
  // Create dist directory if it doesn't exist
  const distDir = path.join(__dirname, 'dist');
  if (!fs.existsSync(distDir)) {
    fs.mkdirSync(distDir, { recursive: true });
  }

  // Install electron-packager if not already installed
  console.log('Ensuring electron-packager is available...');
  try {
    execSync('npx electron-packager --version', { stdio: 'pipe' });
  } catch {
    console.log('Installing electron-packager...');
    execSync('npm install --save-dev electron-packager', { stdio: 'inherit' });
  }

  console.log('Packaging application for Linux...');
  execSync('npx electron-packager . "Foresight SAR" --platform=linux --arch=x64 --out=dist --overwrite --app-version=0.9.0', { stdio: 'inherit' });

  console.log('\n✅ Linux packaging completed successfully!');
  console.log('📦 Packaged app available in: dist/Foresight SAR-linux-x64/');
  
  console.log('\n📋 Next steps for Linux distribution:');
  console.log('   1. Install electron-builder for AppImage/DEB creation:');
  console.log('      npm install --save-dev electron-builder');
  console.log('   2. Run: npm run dist:linux');
  console.log('   3. Or manually create AppImage using appimagetool');
  console.log('   4. Upload to GitHub Releases');

  console.log('\n🐧 Linux package structure:');
  const linuxDir = path.join(distDir, 'Foresight SAR-linux-x64');
  if (fs.existsSync(linuxDir)) {
    const files = fs.readdirSync(linuxDir);
    files.forEach(file => {
      console.log(`   - ${file}`);
    });
  }

} catch (error) {
  console.error('❌ Linux build failed:', error.message);
  console.log('\n💡 Note: This script is designed to run on Linux systems.');
  console.log('   For cross-platform building from Windows, use:');
  console.log('   - Docker with Linux container');
  console.log('   - WSL2 (Windows Subsystem for Linux)');
  console.log('   - GitHub Actions CI/CD pipeline');
  process.exit(1);
}