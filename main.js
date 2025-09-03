const { app, BrowserWindow, Menu } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let mainWindow;
let pythonServer;

// Enable live reload for development
if (process.env.NODE_ENV === 'development') {
  require('electron-reload')(__dirname, {
    electron: path.join(__dirname, '..', 'node_modules', '.bin', 'electron'),
    hardResetMethod: 'exit'
  });
}

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      webSecurity: false // Allow local file access for development
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    title: 'Foresight Phone Stream',
    show: false // Don't show until ready
  });

  // Remove default menu
  Menu.setApplicationMenu(null);

  // Load the app
  mainWindow.loadFile('index.html');

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Focus on the window
    if (process.platform === 'darwin') {
      mainWindow.focus();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
}

function startPythonServer() {
  return new Promise((resolve, reject) => {
    console.log('Starting Python server...');
    
    // Check if server.py exists
    const serverPath = path.join(__dirname, 'server.py');
    if (!fs.existsSync(serverPath)) {
      console.error('server.py not found at:', serverPath);
      reject(new Error('server.py not found'));
      return;
    }

    // Start Python server
    pythonServer = spawn('python', ['server.py'], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonServer.stdout.on('data', (data) => {
      console.log(`Python server: ${data}`);
      if (data.toString().includes('Server ready')) {
        resolve();
      }
    });

    pythonServer.stderr.on('data', (data) => {
      console.error(`Python server error: ${data}`);
    });

    pythonServer.on('close', (code) => {
      console.log(`Python server exited with code ${code}`);
    });

    pythonServer.on('error', (err) => {
      console.error('Failed to start Python server:', err);
      reject(err);
    });

    // Resolve after a short delay if no explicit ready message
    setTimeout(() => {
      resolve();
    }, 3000);
  });
}

function stopPythonServer() {
  if (pythonServer) {
    console.log('Stopping Python server...');
    pythonServer.kill('SIGTERM');
    pythonServer = null;
  }
}

// App event handlers
app.whenReady().then(async () => {
  try {
    // Start Python server first
    await startPythonServer();
    console.log('Python server started successfully');
    
    // Then create the window
    createWindow();
  } catch (error) {
    console.error('Failed to start Python server:', error);
    // Create window anyway for debugging
    createWindow();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopPythonServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopPythonServer();
});

// Handle app termination
process.on('SIGINT', () => {
  stopPythonServer();
  app.quit();
});

process.on('SIGTERM', () => {
  stopPythonServer();
  app.quit();
});

// Prevent navigation away from the app
app.on('web-contents-created', (event, contents) => {
  contents.on('will-navigate', (event, navigationUrl) => {
    const parsedUrl = new URL(navigationUrl);
    
    if (parsedUrl.origin !== 'http://localhost:8000') {
      event.preventDefault();
    }
  });
});