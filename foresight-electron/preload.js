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
                'toggle-sar-mode'
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
                'sar-mode-changed'
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
                'sar-mode-changed'
            ];
            if (validChannels.includes(channel)) {
                // Deliberately strip event as it includes `sender` 
                ipcRenderer.once(channel, (event, ...args) => func(...args));
            }
        }
    }
);