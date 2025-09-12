#!/usr/bin/env node
/**
 * Simple Windows build script for Foresight SAR Electron app
 * Alternative to electron-builder when app-builder.exe fails
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('Building Foresight SAR for Windows...');

try {
  // Create dist directory if it doesn't exist
  const distDir = path.join(__dirname, 'dist');
  if (!fs.existsSync(distDir)) {
    fs.mkdirSync(distDir, { recursive: true });
  }

  // Use electron-packager as alternative
  console.log('Installing electron-packager...');
  execSync('npm install --save-dev electron-packager', { stdio: 'inherit' });

  console.log('Packaging application...');
  execSync('npx electron-packager . "Foresight SAR" --platform=win32 --arch=x64 --out=dist --overwrite --app-version=0.9.0', { stdio: 'inherit' });

  console.log('\n✅ Build completed successfully!');
  console.log('📦 Packaged app available in: dist/Foresight SAR-win32-x64/');
  console.log('\n📋 Next steps:');
  console.log('   1. Test the packaged app by running the .exe file');
  console.log('   2. Create installer using Inno Setup or NSIS manually');
  console.log('   3. Upload to GitHub Releases');

} catch (error) {
  console.error('❌ Build failed:', error.message);
  process.exit(1);
}