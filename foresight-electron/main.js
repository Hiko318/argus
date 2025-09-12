const { app, BrowserWindow, ipcMain, shell, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const fetch = require('node-fetch');
const PackagerBridge = require('./packager-bridge');

// Global references
let mainWindow;
let sarBackendProcess = null;
let backendServiceUrl = 'http://localhost:8004';
let isConnected = false;
let packagerBridge = null;
let systemState = {
  connected: false,
  running: false,
  videoConnected: false,
  telemetryConnected: false,
  detectionRunning: false,
  geolocationActive: false,
  currentMode: 'regular'
};

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false // Allow loading local resources
    },
    icon: path.join(__dirname, 'assets', 'icon.svg'),
    title: 'Foresight SAR System',
    frame: false, // Remove default frame for custom header
    titleBarStyle: 'hidden',
    resizable: true,
    movable: true
  });

  // Load the local SAR interface
  mainWindow.loadFile('sar_interface.html');
  
  // Initialize backend connection check
  checkBackendConnection();
  
  // Open DevTools in development
  // mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopSARBackend();
  });
  
  // Handle window controls
  mainWindow.webContents.on('dom-ready', () => {
    setupWindowControls();
  });
}

// Backend connection check
async function checkBackendConnection() {
  try {
    const response = await fetch(`${backendServiceUrl}/api/status`);
    if (response.ok) {
      isConnected = true;
      console.log('✓ Backend service connected');
      mainWindow.webContents.send('backend-status', { connected: true });
    }
  } catch (error) {
    isConnected = false;
    console.log('✗ Backend service not available:', error.message);
    mainWindow.webContents.send('backend-status', { connected: false });
  }
}

// Window controls setup
function setupWindowControls() {
  // Handle window minimize, maximize, close
  ipcMain.handle('minimize-window', () => {
    mainWindow.minimize();
  });
  
  ipcMain.handle('maximize-window', () => {
    if (mainWindow.isFullScreen()) {
      mainWindow.setFullScreen(false);
    } else {
      mainWindow.setFullScreen(true);
    }
  });
  
  ipcMain.handle('close-window', () => {
    // Properly close the app
    stopSARBackend();
    app.quit();
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
    const response = await fetch(`${backendServiceUrl}/api/status`);
    return response.ok;
  } catch (error) {
    return false;
  }
}

// IPC handlers for SAR operations
ipcMain.handle('get-sar-status', async () => {
  try {
    const response = await fetch(`${backendServiceUrl}/api/status`);
    return await response.json();
  } catch (error) {
    return { error: error.message };
  }
});

// System state management
ipcMain.handle('get-system-state', () => {
  return systemState;
});

ipcMain.handle('update-system-state', (event, newState) => {
  systemState = { ...systemState, ...newState };
  mainWindow.webContents.send('system-state-updated', systemState);
  return systemState;
});

// Backend API handlers
ipcMain.handle('api-request', async (event, { endpoint, method = 'GET', data = null }) => {
  try {
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json'
      }
    };
    
    if (data && method !== 'GET') {
      options.body = JSON.stringify(data);
    }
    
    const response = await fetch(`${backendServiceUrl}${endpoint}`, options);
    const result = await response.json();
    
    return {
      success: response.ok,
      data: result,
      status: response.status
    };
  } catch (error) {
    return {
      success: false,
      error: error.message
    };
  }
});

// Restart SAR backend
function restartSARBackend() {
  stopSARBackend();
  setTimeout(() => {
    startSARBackend();
  }, 1000);
}

// IPC handlers
function setupIPC() {
  // Initialize packager bridge
  packagerBridge = new PackagerBridge();
  
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
  
  // Packager IPC handlers
  ipcMain.handle('package-evidence', async (event, missionData, files = []) => {
    try {
      console.log('Packaging evidence request received:', missionData);
      
      // Create progress callback to send updates to renderer
      const progressCallback = (progress) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('packaging-progress', progress);
        }
      };
      
      const result = await packagerBridge.packageEvidence(
        missionData, 
        files, 
        null, 
        progressCallback
      );
      
      console.log('Packaging completed:', result);
      return result;
      
    } catch (error) {
      console.error('Packaging failed:', error);
      throw error;
    }
  });
  
  ipcMain.handle('get-packager-status', async () => {
    return packagerBridge ? packagerBridge.getStatus() : { available: false };
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
