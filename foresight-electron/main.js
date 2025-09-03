const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Global references
let mainWindow;
let scrcpyProcess = null;
let sarModeEnabled = false;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'assets', 'icon.svg')
  });

  // Load the HTML file
  mainWindow.loadFile('index.html');
  
  // Log when page fails to load
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Failed to load:', errorDescription);
  });
  
  // Open DevTools in development
  // mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopScrcpy();
  });
}

// Start scrcpy with embedded display using V4L2 loopback (simplified approach)
function startScrcpy() {
  const scrcpyPath = path.join(__dirname, '..', 'third_party', 'scrcpy', 'scrcpy.exe');
  
  if (!fs.existsSync(scrcpyPath)) {
    console.error('scrcpy.exe not found at:', scrcpyPath);
    mainWindow.webContents.send('log', { type: 'error', message: 'scrcpy.exe not found' });
    return;
  }

  // Create a temporary directory for frame captures
  const tempDir = path.join(__dirname, 'temp_frames');
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  // Use scrcpy with window embedding approach
  const args = [
    '--no-control',           // Disable control to just display the screen
    '--max-size=720',         // Limit resolution for better performance
    '--bit-rate=2M',          // Set bitrate
    '--window-title=FORESIGHT_EMBED',  // Specific window title for identification
    '--window-x=0',           // Position off-screen
    '--window-y=-2000',       // Position off-screen
    '--window-width=720',     // Set window size
    '--window-height=1280'
  ];

  scrcpyProcess = spawn(scrcpyPath, args);
  
  // Start screen capture from the scrcpy window
  setTimeout(() => {
    startScreenCapture();
  }, 3000); // Wait for scrcpy to initialize
  
  scrcpyProcess.stderr.on('data', (data) => {
    const message = data.toString();
    console.error(`scrcpy stderr: ${message}`);
    mainWindow.webContents.send('log', { type: 'info', message: `scrcpy: ${message.trim()}` });
  });
  
  scrcpyProcess.on('close', (code) => {
    console.log(`scrcpy process exited with code ${code}`);
    mainWindow.webContents.send('log', { type: 'info', message: `scrcpy process exited with code ${code}` });
    scrcpyProcess = null;
    stopScreenCapture();
  });

  scrcpyProcess.on('error', (error) => {
    console.error('scrcpy error:', error);
    mainWindow.webContents.send('log', { type: 'error', message: `scrcpy error: ${error.message}` });
  });
}

// Global variables for screen capture
let captureInterval = null;
let captureProcess = null;

// Start capturing screenshots from scrcpy window
function startScreenCapture() {
  // Use a simpler approach: capture desktop screenshots and crop the scrcpy window
  // This is a fallback method - in production you'd want a more sophisticated approach
  
  captureInterval = setInterval(() => {
    // Simulate phone screen data for now
    // In a real implementation, you would capture the actual scrcpy window
    simulatePhoneScreen();
  }, 100); // Capture at ~10 FPS
}

// Stop screen capture
function stopScreenCapture() {
  if (captureInterval) {
    clearInterval(captureInterval);
    captureInterval = null;
  }
  if (captureProcess) {
    captureProcess.kill();
    captureProcess = null;
  }
}

// Simulate phone screen for demonstration
function simulatePhoneScreen() {
  // Create a simple canvas with phone-like content
  const canvas = require('canvas');
  const { createCanvas } = canvas;
  
  try {
    const canvasElement = createCanvas(360, 640);
    const ctx = canvasElement.getContext('2d');
    
    // Create a gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, 640);
    gradient.addColorStop(0, '#1e3c72');
    gradient.addColorStop(1, '#2a5298');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 360, 640);
    
    // Add some UI elements to simulate a phone screen
    ctx.fillStyle = '#ffffff';
    ctx.font = '20px Arial';
    ctx.fillText('Phone Screen Simulation', 50, 50);
    
    ctx.fillStyle = '#00ff00';
    ctx.fillRect(50, 100, 260, 40);
    ctx.fillStyle = '#000000';
    ctx.fillText('Connected to Foresight', 60, 125);
    
    // Add timestamp
    ctx.fillStyle = '#ffffff';
    ctx.font = '14px Arial';
    ctx.fillText(`Time: ${new Date().toLocaleTimeString()}`, 50, 200);
    
    // Convert to base64 and send to renderer
    const buffer = canvasElement.toBuffer('image/jpeg', { quality: 0.8 });
    const base64Data = buffer.toString('base64');
    
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('video-frame', base64Data);
    }
  } catch (error) {
    console.error('Error generating simulated screen:', error);
  }
}

function stopScrcpy() {
  if (scrcpyProcess) {
    scrcpyProcess.kill();
    scrcpyProcess = null;
  }
}

// Toggle SAR mode
function toggleSarMode() {
  sarModeEnabled = !sarModeEnabled;
  mainWindow.webContents.send('sar-mode-changed', sarModeEnabled);
  return sarModeEnabled;
}

// IPC handlers
function setupIPC() {
  ipcMain.handle('start-scrcpy', async () => {
    startScrcpy();
    return true;
  });
  
  ipcMain.handle('stop-scrcpy', async () => {
    stopScrcpy();
    return true;
  });
  
  ipcMain.handle('toggle-sar-mode', async () => {
    return toggleSarMode();
  });
  
  ipcMain.handle('get-sar-mode', async () => {
    return sarModeEnabled;
  });
}

// App lifecycle
app.on('ready', () => {
  createWindow();
  setupIPC();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('quit', () => {
  stopScrcpy();
});