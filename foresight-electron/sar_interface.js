// Global functions for header controls
function toggleConnection() {
    console.log('Toggle connection');
    // Implementation for connection toggle
}

function startSystem() {
    console.log('Start system');
    // Implementation for system start
}

function stopSystem() {
    console.log('Stop system');
    // Implementation for system stop
}

function changeMode(mode) {
    console.log('Mode changed to:', mode);
    // Implementation for mode change
}

function exportMissionData() {
    console.log('Exporting mission data...');
    // Implementation for data export
}

function openSettings() {
    console.log('Open settings');
    // Implementation for settings panel
}

function closeApplication() {
    if (window.electronAPI) {
        window.electronAPI.closeApp();
    } else {
        window.close();
    }
}

// SAR Interface Class
class SARInterface {
    constructor() {
        this.baseUrl = 'http://localhost:8004';
        this.websocket = null;
        this.isConnected = false;
        this.currentMode = 'regular';
        this.detections = [];
        this.telemetryData = {};
        this.targetImage = null;
        this.isTracking = false;
        
        this.init();
    }
    
    init() {
        console.log('Initializing SAR Interface...');
        this.setupEventListeners();
        this.connectWebSocket();
    }
    
    setupEventListeners() {
        // Video feed error handling
        const videoFeed = document.getElementById('video-feed');
        if (videoFeed) {
            videoFeed.addEventListener('error', (e) => {
                console.error('Video feed error:', e);
            });
        }
        
        // Target image upload
        const targetInput = document.getElementById('target-input');
        if (targetInput) {
            targetInput.addEventListener('change', (e) => {
                this.handleTargetUpload(e);
            });
        }
    }
    
    connectWebSocket() {
        try {
            this.websocket = new WebSocket(`ws://localhost:8004/ws`);
            this.telemetryWs = new WebSocket('ws://localhost:8004/ws/telemetry');
            this.detectionWs = new WebSocket('ws://localhost:8004/ws/detections');
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.updateConnectionStatus();
            };
            
            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus();
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }
    
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'detection':
                    this.handleDetection(data.data);
                    break;
                case 'telemetry':
                    this.handleTelemetry(data.data);
                    break;
                case 'video_frame':
                    this.handleVideoFrame(data.data);
                    break;
                default:
                    console.log('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    handleDetection(detection) {
        this.detections.push(detection);
        this.addDetectionToUI(detection);
        this.drawDetectionOverlay(detection);
    }
    
    handleTelemetry(telemetry) {
        this.telemetryData = { ...this.telemetryData, ...telemetry };
        this.updateTelemetryUI(telemetry);
    }
    
    handleVideoFrame(frameData) {
        // Update video feed if needed
        const videoFeed = document.getElementById('video-feed');
        if (videoFeed && frameData.url) {
            videoFeed.src = frameData.url;
        }
    }
    
    addDetectionToUI(detection) {
        const detectionList = document.getElementById('detection-list');
        if (!detectionList) return;
        
        const detectionItem = document.createElement('div');
        detectionItem.className = 'detection-item';
        detectionItem.innerHTML = `
            <div>${detection.class || 'Object'} detected</div>
            <div class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</div>
            <div style="font-size: 12px; color: #95a5a6;">${new Date().toLocaleTimeString()}</div>
        `;
        
        detectionList.insertBefore(detectionItem, detectionList.firstChild);
        
        // Keep only the last 10 detections
        while (detectionList.children.length > 10) {
            detectionList.removeChild(detectionList.lastChild);
        }
    }
    
    updateTelemetryUI(telemetry) {
        if (telemetry.altitude !== undefined) {
            const altElement = document.getElementById('altitude');
            if (altElement) altElement.textContent = telemetry.altitude + ' m';
        }
        
        if (telemetry.speed !== undefined) {
            const speedElement = document.getElementById('speed');
            if (speedElement) speedElement.textContent = telemetry.speed + ' m/s';
        }
        
        if (telemetry.battery !== undefined) {
            const batteryElement = document.getElementById('battery');
            if (batteryElement) batteryElement.textContent = telemetry.battery + '%';
        }
        
        if (telemetry.signal !== undefined) {
            const signalElement = document.getElementById('signal');
            if (signalElement) signalElement.textContent = telemetry.signal;
        }
    }
    
    drawDetectionOverlay(detection) {
        const overlay = document.getElementById('video-overlay');
        if (!overlay || !detection.bbox) return;
        
        const bbox = document.createElement('div');
        bbox.style.position = 'absolute';
        bbox.style.border = '2px solid #ff0000';
        bbox.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
        bbox.style.left = detection.bbox.x + 'px';
        bbox.style.top = detection.bbox.y + 'px';
        bbox.style.width = detection.bbox.width + 'px';
        bbox.style.height = detection.bbox.height + 'px';
        bbox.style.pointerEvents = 'none';
        
        overlay.appendChild(bbox);
        
        // Remove overlay after 3 seconds
        setTimeout(() => {
            if (bbox.parentNode) {
                bbox.parentNode.removeChild(bbox);
            }
        }, 3000);
    }
    
    updateConnectionStatus() {
        const statusIndicator = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');
        
        if (statusIndicator && statusText) {
            if (this.isConnected) {
                statusIndicator.className = 'status-indicator status-connected';
                statusText.textContent = 'Connected';
            } else {
                statusIndicator.className = 'status-indicator status-disconnected';
                statusText.textContent = 'Disconnected';
            }
        }
    }
    
    handleTargetUpload(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const preview = document.getElementById('target-preview');
                if (preview) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }
                this.targetImage = e.target.result;
                this.uploadTargetImage(file);
            };
            reader.readAsDataURL(file);
        }
    }
    
    async uploadTargetImage(file) {
        try {
            const formData = new FormData();
            formData.append('image', file);
            
            const response = await fetch(`${this.baseUrl}/api/target`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Target image uploaded:', result);
            } else {
                console.error('Failed to upload target image');
            }
        } catch (error) {
            console.error('Error uploading target image:', error);
        }
    }
    
    setMode(mode) {
        this.currentMode = mode;
        console.log('Mode set to:', mode);
        
        // Send mode change to backend
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'mode_change',
                mode: mode
            }));
        }
    }
    
    async exportMissionData() {
        try {
            const response = await fetch(`${this.baseUrl}/api/export/mission`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `mission_data_${new Date().toISOString().slice(0, 10)}.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            } else {
                console.error('Failed to export mission data');
            }
        } catch (error) {
            console.error('Error exporting mission data:', error);
        }
    }
    
    showMissionSummary() {
        // Show mission summary dialog
        console.log('Showing mission summary...');
        this.exportMissionData();
    }
}

// Initialize SAR Interface when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.sarInterface = new SARInterface();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SARInterface;
}