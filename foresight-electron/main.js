const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const fetch = require('node-fetch');

// Global references
let mainWindow;
let sarBackendProcess = null;
let sarServiceUrl = 'http://localhost:8004';
let isConnected = false;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false // Allow loading local resources
    },
    icon: path.join(__dirname, 'assets', 'icon.svg'),
    title: 'Foresight SAR System'
  });

  // Start SAR backend service
  startSARBackend();
  
  // Wait for backend to start, then load the interface
  setTimeout(() => {
    mainWindow.loadURL(sarServiceUrl);
  }, 3000);
  
  // Log when page fails to load
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Failed to load SAR interface:', errorDescription);
    // Fallback to local HTML if backend is not available
    mainWindow.loadFile('sar_interface.html');
  });
  
  // Open DevTools in development
  // mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopSARBackend();
  });
}

// Start SAR backend service
function startSARBackend() {
  const projectRoot = path.join(__dirname, '..');
  const pythonScript = path.join(projectRoot, 'src', 'backend', 'sar_service.py');
  
  if (!fs.existsSync(pythonScript)) {
    console.error('SAR service script not found at:', pythonScript);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('log', { type: 'error', message: 'SAR service script not found' });
    }
    return;
  }

  // Start the SAR backend service
  const args = ['-m', 'src.backend.sar_service', '--port', '8004'];
  
  sarBackendProcess = spawn('python', args, {
    cwd: projectRoot,
    stdio: ['pipe', 'pipe', 'pipe']
  });
  
  sarBackendProcess.stdout.on('data', (data) => {
    const message = data.toString();
    console.log(`SAR Backend: ${message}`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('log', { type: 'info', message: `SAR: ${message.trim()}` });
    }
    
    // Check if service is ready
    if (message.includes('Uvicorn running on')) {
      isConnected = true;
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('backend-ready', true);
      }
    }
  });
  
  sarBackendProcess.stderr.on('data', (data) => {
    const message = data.toString();
    console.error(`SAR Backend stderr: ${message}`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('log', { type: 'error', message: `SAR Error: ${message.trim()}` });
    }
  });
  
  sarBackendProcess.on('close', (code) => {
    console.log(`SAR backend process exited with code ${code}`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('log', { type: 'info', message: `SAR backend exited with code ${code}` });
    }
    sarBackendProcess = null;
    isConnected = false;
  });

  sarBackendProcess.on('error', (error) => {
    console.error('SAR backend error:', error);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('log', { type: 'error', message: `SAR backend error: ${error.message}` });
    }
  });
}

// Stop SAR backend service
function stopSARBackend() {
  if (sarBackendProcess) {
    sarBackendProcess.kill();
    sarBackendProcess = null;
    isConnected = false;
  }
}

// Check if SAR backend is running
async function checkSARBackend() {
  try {
    const response = await fetch(`${sarServiceUrl}/api/status`);
    return response.ok;
  } catch (error) {
    return false;
  }
}

// Restart SAR backend
function restartSARBackend() {
  stopSARBackend();
  setTimeout(() => {
    startSARBackend();
  }, 1000);
}

// IPC handlers
function setupIPC() {
  ipcMain.handle('start-sar-backend', async () => {
    startSARBackend();
    return true;
  });
  
  ipcMain.handle('stop-sar-backend', async () => {
    stopSARBackend();
    return true;
  });
  
  ipcMain.handle('restart-sar-backend', async () => {
    restartSARBackend();
    return true;
  });
  
  ipcMain.handle('check-sar-backend', async () => {
    return await checkSARBackend();
  });
  
  ipcMain.handle('get-connection-status', async () => {
    return isConnected;
  });
  
  ipcMain.handle('open-external', async (event, url) => {
    shell.openExternal(url);
    return true;
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
  stopSARBackend();
});

app.on('before-quit', () => {
  stopSARBackend();
});