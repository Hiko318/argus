const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
    'ipcRenderer', {
        invoke: (channel, ...args) => {
            // Whitelist channels
            const validChannels = [
                'start-scrcpy',
                'stop-scrcpy',
                'toggle-sar-mode',
                'start-sar-backend',
                'stop-sar-backend',
                'restart-sar-backend',
                'check-sar-backend',
                'get-connection-status',
                'open-external',
                'package-evidence',
                'get-packager-status'
            ];
            if (validChannels.includes(channel)) {
                return ipcRenderer.invoke(channel, ...args);
            }
            return Promise.reject(new Error(`Unauthorized IPC channel: ${channel}`));
        },
        on: (channel, func) => {
            const validChannels = [
                'log',
                'screen-capture',
                'detection',
                'sar-mode-changed',
                'packaging-progress'
            ];
            if (validChannels.includes(channel)) {
                // Deliberately strip event as it includes `sender` 
                ipcRenderer.on(channel, (event, ...args) => func(...args));
            }
        },
        once: (channel, func) => {
            const validChannels = [
                'log',
                'screen-capture',
                'detection',
                'sar-mode-changed',
                'packaging-progress'
            ];
            if (validChannels.includes(channel)) {
                // Deliberately strip event as it includes `sender` 
                ipcRenderer.once(channel, (event, ...args) => func(...args));
            }
        }
    }
);

// Expose SAR-specific API
contextBridge.exposeInMainWorld(
    'electronAPI', {
        // SAR Backend controls
        startSARBackend: () => ipcRenderer.invoke('start-sar-backend'),
        stopSARBackend: () => ipcRenderer.invoke('stop-sar-backend'),
        restartSARBackend: () => ipcRenderer.invoke('restart-sar-backend'),
        checkSARBackend: () => ipcRenderer.invoke('check-sar-backend'),
        getConnectionStatus: () => ipcRenderer.invoke('get-connection-status'),
        
        // External links
        openExternal: (url) => ipcRenderer.invoke('open-external', url),
        
        // Evidence packaging
        packageEvidence: (missionData, files = []) => {
            return ipcRenderer.invoke('package-evidence', missionData, files);
        },
        getPackagerStatus: () => ipcRenderer.invoke('get-packager-status'),
        
        // Event listeners
        onPackagingProgress: (callback) => {
            ipcRenderer.on('packaging-progress', (event, progress) => callback(progress));
        },
        removePackagingProgressListener: () => {
            ipcRenderer.removeAllListeners('packaging-progress');
        }
    }
);