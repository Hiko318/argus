// System state tracking
let systemState = {
    connected: false,
    running: false,
    videoConnected: false,
    telemetryConnected: false,
    detectionRunning: false,
    geolocationActive: false
};

// Global functions for header controls
function toggleConnection() {
    systemState.connected = !systemState.connected;
    
    if (systemState.connected) {
        updateSystemStatus('connected', 'Connected');
        updateVideoStatus('connected', 'Signal Active');
        updateTelemetryStatus('connected', 'Online');
        console.log('✓ System connected successfully');
        showStatusMessage('Connected to system', 'success');
    } else {
        updateSystemStatus('disconnected', 'Disconnected');
        updateVideoStatus('disconnected', 'No Signal');
        updateTelemetryStatus('disconnected', 'Offline');
        updateDetectionStatus('disconnected', 'Stopped');
        updateGeolocationStatus('disconnected', 'Offline');
        systemState.running = false;
        console.log('✗ System disconnected');
        showStatusMessage('Disconnected from system', 'error');
    }
}

function startSystem() {
    if (!systemState.connected) {
        showStatusMessage('Cannot start: System not connected', 'error');
        console.log('✗ Cannot start: System not connected');
        return;
    }
    
    if (systemState.running) {
        showStatusMessage('System already running', 'warning');
        console.log('⚠ System already running');
        return;
    }
    
    systemState.running = true;
    systemState.detectionRunning = true;
    systemState.geolocationActive = true;
    
    updateDetectionStatus('connected', 'Active');
    updateGeolocationStatus('connected', 'Active');
    
    console.log('✓ System started successfully');
    showStatusMessage('System started - All services active', 'success');
    
    // Simulate performance metrics
    startPerformanceMonitoring();
}

function stopSystem() {
    if (!systemState.running) {
        showStatusMessage('System already stopped', 'warning');
        console.log('⚠ System already stopped');
        return;
    }
    
    systemState.running = false;
    systemState.detectionRunning = false;
    systemState.geolocationActive = false;
    
    updateDetectionStatus('disconnected', 'Stopped');
    updateGeolocationStatus('disconnected', 'Offline');
    
    console.log('✓ System stopped');
    showStatusMessage('System stopped - All services offline', 'success');
    
    // Stop performance monitoring
    stopPerformanceMonitoring();
}

function toggleMode() {
    if (window.sarInterface) {
        window.sarInterface.toggleMode();
    }
}

function changeMode(mode) {
    if (window.sarInterface) {
        window.sarInterface.setMode(mode);
    }
}

function exportMissionData() {
    if (window.sarInterface) {
        window.sarInterface.showMissionSummary();
    }
}

function openSettings() {
    console.log('Open settings');
    // Implementation for settings panel
}

// Global functions for suspect-lock panel
function closeSuspectPanel() {
    const panel = document.getElementById('suspect-lock-panel');
    if (panel) {
        panel.style.display = 'none';
    }
    // Switch back to SAR mode
    if (window.sarInterface) {
        window.sarInterface.setMode('sar');
    }
}

// Flag/Lock functionality
let currentCapture = null;

function flagLockCapture() {
    // Capture current frame from video
    const video = document.getElementById('video-feed');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    // Get current timestamp and location
    const timestamp = new Date().toISOString();
    const lat = document.getElementById('lat-display').textContent;
    const location = `${lat}, ${document.getElementById('error-display').textContent}`;
    
    // Store capture data
    currentCapture = {
        image: canvas.toDataURL('image/jpeg', 0.9),
        timestamp: timestamp,
        location: location,
        operatorId: '',
        notes: ''
    };
    
    // Show modal
    showFlagLockModal();
}

function showFlagLockModal() {
    if (!currentCapture) return;
    
    // Populate modal with capture data
    document.getElementById('capture-preview-img').src = currentCapture.image;
    document.getElementById('capture-timestamp').textContent = currentCapture.timestamp;
    document.getElementById('capture-location').textContent = currentCapture.location;
    document.getElementById('operator-id').value = '';
    document.getElementById('capture-notes').value = '';
    
    // Show modal
    document.getElementById('flag-lock-modal').style.display = 'flex';
}

function closeFlagLockModal() {
    document.getElementById('flag-lock-modal').style.display = 'none';
    currentCapture = null;
}

// Status panel management
function toggleStatusPanel() {
    const panel = document.getElementById('statusPanel');
    const content = document.getElementById('statusContent');
    const toggle = document.getElementById('statusToggle');
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        toggle.textContent = '−';
    } else {
        content.style.display = 'none';
        toggle.textContent = '+';
    }
}

// Status update functions
function updateSystemStatus(status, text) {
    const indicator = document.getElementById('systemIndicator');
    const value = document.getElementById('systemValue');
    
    if (indicator) indicator.className = `status-indicator ${status}`;
    if (value) value.textContent = text;
}

function updateVideoStatus(status, text) {
    const indicator = document.getElementById('videoIndicator');
    const value = document.getElementById('videoValue');
    
    if (indicator) indicator.className = `status-indicator ${status}`;
    if (value) value.textContent = text;
}

function updateTelemetryStatus(status, text) {
    const indicator = document.getElementById('telemetryIndicator');
    const value = document.getElementById('telemetryValue');
    
    if (indicator) indicator.className = `status-indicator ${status}`;
    if (value) value.textContent = text;
}

function updateDetectionStatus(status, text) {
    const indicator = document.getElementById('detectionIndicator');
    const value = document.getElementById('detectionValue');
    
    if (indicator) indicator.className = `status-indicator ${status}`;
    if (value) value.textContent = text;
}

function updateGeolocationStatus(status, text) {
    const indicator = document.getElementById('geolocationIndicator');
    const value = document.getElementById('geolocationValue');
    
    if (indicator) indicator.className = `status-indicator ${status}`;
    if (value) value.textContent = text;
}

function updatePerformanceMetrics(metrics) {
    // Update status panel metrics
    const fpsEl = document.getElementById('fps-status');
    const latencyEl = document.getElementById('latency-display');
    
    if (fpsEl) {
        fpsEl.textContent = `${metrics.fps}`;
        // Color code based on thresholds
        fpsEl.style.color = getThresholdColor('fps', metrics.fps);
    }
    
    if (latencyEl) {
        latencyEl.textContent = `${metrics.latency} ms`;
        latencyEl.style.color = getThresholdColor('latency', metrics.latency);
    }
}

function updateHUDMetrics(metrics) {
    // Update HUD display elements
    const fpsDisplay = document.getElementById('fps-display');
    const gpuDisplay = document.getElementById('gpu-display');
    const latDisplay = document.getElementById('lat-display');
    const errorDisplay = document.getElementById('error-display');
    
    if (fpsDisplay) {
        fpsDisplay.textContent = metrics.fps;
        fpsDisplay.style.color = getThresholdColor('fps', metrics.fps);
    }
    
    if (gpuDisplay) {
        gpuDisplay.textContent = `${metrics.gpu_percent}%`;
        gpuDisplay.style.color = getThresholdColor('gpu', metrics.gpu_percent);
    }
    
    if (latDisplay) {
        latDisplay.textContent = `${metrics.latency}`;
        latDisplay.style.color = getThresholdColor('latency', metrics.latency);
    }
    
    if (errorDisplay) {
        errorDisplay.textContent = `±${metrics.error_margin}m`;
        errorDisplay.style.color = getThresholdColor('error', metrics.error_margin);
    }
    
    // Update signal status
    const signalElements = document.querySelectorAll('.signal-status');
    signalElements.forEach(el => {
        if (metrics.signal_status === 'NO SIGNAL') {
            el.textContent = 'NO SIGNAL...';
            el.style.color = '#ff4444';
        } else {
            el.textContent = 'CONNECTED';
            el.style.color = '#00ff00';
        }
    });
}

function getThresholdColor(metric, value) {
    const thresholds = {
        fps: { good: 20, warning: 15 },
        gpu: { good: 70, warning: 85 },
        latency: { good: 50, warning: 100 },
        error: { good: 5, warning: 10 },
        cpu: { good: 70, warning: 85 },
        memory: { good: 70, warning: 85 }
    };
    
    const threshold = thresholds[metric];
    if (!threshold) return '#00ff00'; // Default green
    
    if (metric === 'fps') {
        // For FPS, lower is worse
        if (value >= threshold.good) return '#00ff00'; // Green
        if (value >= threshold.warning) return '#ffaa00'; // Yellow
        return '#ff4444'; // Red
    } else {
        // For other metrics, higher is worse
        if (value <= threshold.good) return '#00ff00'; // Green
        if (value <= threshold.warning) return '#ffaa00'; // Yellow
        return '#ff4444'; // Red
    }
}

// Status message system
function showStatusMessage(message, type) {
    // Create or update a temporary status message
    let messageEl = document.getElementById('statusMessage');
    if (!messageEl) {
        messageEl = document.createElement('div');
        messageEl.id = 'statusMessage';
        messageEl.style.cssText = `
            position: absolute;
            top: 50px;
            left: 20px;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1001;
            transition: opacity 0.3s;
        `;
        document.body.appendChild(messageEl);
    }
    
    messageEl.textContent = message;
    messageEl.className = `status-message ${type}`;
    
    // Style based on type
    switch(type) {
        case 'success':
            messageEl.style.background = 'rgba(0, 255, 0, 0.1)';
            messageEl.style.border = '1px solid #00ff00';
            messageEl.style.color = '#00ff00';
            break;
        case 'error':
            messageEl.style.background = 'rgba(255, 68, 68, 0.1)';
            messageEl.style.border = '1px solid #ff4444';
            messageEl.style.color = '#ff4444';
            break;
        case 'warning':
            messageEl.style.background = 'rgba(255, 170, 0, 0.1)';
            messageEl.style.border = '1px solid #ffaa00';
            messageEl.style.color = '#ffaa00';
            break;
    }
    
    messageEl.style.opacity = '1';
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        messageEl.style.opacity = '0';
        setTimeout(() => {
            if (messageEl.parentNode) {
                messageEl.parentNode.removeChild(messageEl);
            }
        }, 300);
    }, 3000);
}

// Performance monitoring
let performanceInterval;

function startPerformanceMonitoring() {
    performanceInterval = setInterval(async () => {
        try {
            // Fetch real system metrics from API with cache busting
            const response = await fetch(`/api/system/metrics?t=${Date.now()}`);
            if (response.ok) {
                const metrics = await response.json();
                updatePerformanceMetrics(metrics);
                updateHUDMetrics(metrics);
            } else {
                console.error('Failed to fetch system metrics');
                // Fallback to simulated data
                const fallbackMetrics = {
                    fps: 0,
                    latency: 999,
                    cpu_percent: 0,
                    memory_percent: 0,
                    gpu_percent: 0,
                    error_margin: 0,
                    signal_status: 'ERROR'
                };
                updatePerformanceMetrics(fallbackMetrics);
                updateHUDMetrics(fallbackMetrics);
            }
        } catch (error) {
            console.error('Error fetching system metrics:', error);
        }
    }, 1000);
}

function stopPerformanceMonitoring() {
    if (performanceInterval) {
        clearInterval(performanceInterval);
        performanceInterval = null;
        
        // Reset to default values
        const resetMetrics = {
            fps: 0,
            latency: 0,
            cpu_percent: 0,
            memory_percent: 0,
            gpu_percent: 0,
            error_margin: 0,
            signal_status: 'DISCONNECTED'
        };
        updatePerformanceMetrics(resetMetrics);
        updateHUDMetrics(resetMetrics);
    }
}

// Make functions globally accessible
window.toggleStatusPanel = toggleStatusPanel;
window.toggleConnection = toggleConnection;
window.startSystem = startSystem;
window.stopSystem = stopSystem;

// Initialize status panel on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set initial status
    updateSystemStatus('disconnected', 'Disconnected');
    updateVideoStatus('disconnected', 'No Signal');
    updateTelemetryStatus('disconnected', 'Offline');
    updateDetectionStatus('disconnected', 'Stopped');
    updateGeolocationStatus('disconnected', 'Offline');
    
    // Initialize with default metrics
    const initialMetrics = {
        fps: 0,
        latency: 0,
        cpu_percent: 0,
        memory_percent: 0,
        gpu_percent: 0,
        error_margin: 0,
        signal_status: 'DISCONNECTED'
    };
    updatePerformanceMetrics(initialMetrics);
    updateHUDMetrics(initialMetrics);
});

// Retraining functionality
let currentRetrainData = null;

function openRetrainModal() {
    // Capture current scene data
    const timestamp = new Date().toISOString();
    const lat = document.getElementById('lat-display').textContent;
    const lon = document.getElementById('error-display').textContent;
    const location = `${lat}, ${lon}`;
    
    // Generate scene ID based on timestamp
    const sceneId = `SCENE_${timestamp.replace(/[:.]/g, '_')}`;
    
    // Store current scene data
    currentRetrainData = {
        sceneId: sceneId,
        timestamp: timestamp,
        location: location,
        video: document.getElementById('video-feed').src,
        currentFrame: getCurrentVideoFrame()
    };
    
    // Populate modal with scene data
    document.getElementById('retrain-scene-id').textContent = sceneId;
    document.getElementById('retrain-timestamp').textContent = new Date(timestamp).toLocaleString();
    document.getElementById('retrain-location').textContent = location;
    
    // Reset form fields
    document.getElementById('flag-reason').value = 'low_confidence';
    document.getElementById('predicted-class').value = '';
    document.getElementById('corrected-class').value = '';
    document.getElementById('confidence-score').value = '';
    document.getElementById('priority-level').value = '3';
    document.getElementById('retrain-notes').value = '';
    document.getElementById('retrain-operator-id').value = '';
    
    // Show modal
    document.getElementById('retrain-modal').style.display = 'flex';
}

function closeRetrainModal() {
    document.getElementById('retrain-modal').style.display = 'none';
    currentRetrainData = null;
}

function getCurrentVideoFrame() {
    const video = document.getElementById('video-feed');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.9);
}

function submitRetrainFlag() {
    if (!currentRetrainData) {
        alert('No scene data available');
        return;
    }
    
    // Collect form data
    const retrainFlag = {
        sceneId: currentRetrainData.sceneId,
        timestamp: currentRetrainData.timestamp,
        location: currentRetrainData.location,
        frameData: currentRetrainData.currentFrame,
        flagReason: document.getElementById('flag-reason').value,
        predictedClass: document.getElementById('predicted-class').value,
        correctedClass: document.getElementById('corrected-class').value,
        confidenceScore: parseFloat(document.getElementById('confidence-score').value) || null,
        priorityLevel: parseInt(document.getElementById('priority-level').value),
        notes: document.getElementById('retrain-notes').value,
        operatorId: document.getElementById('retrain-operator-id').value,
        flaggedAt: new Date().toISOString()
    };
    
    // Validate required fields
    if (!retrainFlag.operatorId) {
        alert('Please enter your operator ID');
        return;
    }
    
    if (!retrainFlag.correctedClass) {
        alert('Please specify the correct classification');
        return;
    }
    
    // Submit to active learning system
    submitToActiveLearning(retrainFlag)
        .then(response => {
            if (response.success) {
                alert(`Scene flagged successfully! Flag ID: ${response.flagId}`);
                closeRetrainModal();
                
                // Update UI to show flagged status
                updateRetrainStatus(retrainFlag.sceneId, 'flagged');
            } else {
                alert(`Failed to flag scene: ${response.error}`);
            }
        })
        .catch(error => {
            console.error('Error submitting retrain flag:', error);
            alert('Error submitting flag. Check console for details.');
        });
}

async function submitToActiveLearning(flagData) {
    try {
        // In a real implementation, this would send to the active learning API
        // For now, we'll simulate the submission and store locally
        
        // Store in localStorage for demonstration
        const existingFlags = JSON.parse(localStorage.getItem('retrainFlags') || '[]');
        const flagId = `FLAG_${Date.now()}`;
        
        const flagRecord = {
            ...flagData,
            flagId: flagId,
            status: 'pending',
            submittedAt: new Date().toISOString()
        };
        
        existingFlags.push(flagRecord);
        localStorage.setItem('retrainFlags', JSON.stringify(existingFlags));
        
        // Log to console for debugging
        console.log('Retrain flag submitted:', flagRecord);
        
        // Simulate API response
        return {
            success: true,
            flagId: flagId,
            message: 'Scene flagged for retraining'
        };
        
    } catch (error) {
        console.error('Error in submitToActiveLearning:', error);
        return {
            success: false,
            error: error.message
        };
    }
}

function updateRetrainStatus(sceneId, status) {
    // Update UI to show that scene has been flagged
    const statusElement = document.getElementById('tracking-status');
    if (statusElement) {
        statusElement.innerHTML += `<div class="retrain-flag">Scene ${sceneId} flagged for retraining (${status})</div>`;
    }
    
    // Add visual indicator to video feed
    const videoContainer = document.querySelector('.video-container');
    if (videoContainer && !videoContainer.querySelector('.retrain-indicator')) {
        const indicator = document.createElement('div');
        indicator.className = 'retrain-indicator';
        indicator.innerHTML = '🏷️ Flagged for Retraining';
        indicator.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: #ff9800;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1000;
        `;
        videoContainer.appendChild(indicator);
        
        // Remove indicator after 5 seconds
        setTimeout(() => {
            if (indicator.parentNode) {
                indicator.parentNode.removeChild(indicator);
            }
        }, 5000);
    }
}

// Function to view flagged scenes (for debugging/admin)
function viewRetrainFlags() {
    const flags = JSON.parse(localStorage.getItem('retrainFlags') || '[]');
    console.log('All retrain flags:', flags);
    return flags;
}

function exportAsCSV() {
    if (!currentCapture) return;
    
    const operatorId = document.getElementById('operator-id').value;
    const notes = document.getElementById('capture-notes').value;
    
    const csvData = [
        ['Field', 'Value'],
        ['Timestamp', currentCapture.timestamp],
        ['Location', currentCapture.location],
        ['Operator ID', operatorId],
        ['Notes', notes]
    ];
    
    const csvContent = csvData.map(row => row.map(field => `"${field}"`).join(',')).join('\n');
    downloadFile(csvContent, `evidence_${Date.now()}.csv`, 'text/csv');
}

function exportAsJSON() {
    if (!currentCapture) return;
    
    const operatorId = document.getElementById('operator-id').value;
    const notes = document.getElementById('capture-notes').value;
    
    const jsonData = {
        timestamp: currentCapture.timestamp,
        location: currentCapture.location,
        operatorId: operatorId,
        notes: notes,
        image: currentCapture.image
    };
    
    const jsonContent = JSON.stringify(jsonData, null, 2);
    downloadFile(jsonContent, `evidence_${Date.now()}.json`, 'application/json');
}

function packageEvidence() {
    if (!currentCapture) return;
    
    const operatorId = document.getElementById('operator-id').value;
    const notes = document.getElementById('capture-notes').value;
    
    // Send to backend for evidence packaging
    const evidenceData = {
        timestamp: currentCapture.timestamp,
        location: currentCapture.location,
        operatorId: operatorId,
        notes: notes,
        image: currentCapture.image
    };
    
    fetch('/api/package-evidence', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(evidenceData)
    })
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `evidence_${Date.now()}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    })
    .catch(error => {
        console.error('Error packaging evidence:', error);
        alert('Error packaging evidence. Please try again.');
    });
    
    closeFlagLockModal();
}

function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function selectTargetImage() {
    const input = document.getElementById('target-image-input');
    if (input) {
        input.click();
    }
}

function captureTargetImage() {
    // For now, just trigger file input (camera capture would need additional implementation)
    selectTargetImage();
}

function processTargetImage() {
    if (window.sarInterface) {
        window.sarInterface.processTargetImage();
    }
}

function clearTarget() {
    if (window.sarInterface) {
        window.sarInterface.clearTarget();
    }
}

function startTracking() {
    if (window.sarInterface) {
        window.sarInterface.startTracking();
    }
}

function stopTracking() {
    if (window.sarInterface) {
        window.sarInterface.stopTracking();
    }
}

function openSettings() {
    console.log('Open settings');
    // Implementation for settings
}

function closeApplication() {
    console.log('Close application');
    // Implementation for closing app
}

function switchTab(tabName) {
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    
    // Add active class to the correct tab
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        if (tab.textContent.toLowerCase().includes(tabName.toLowerCase())) {
            tab.classList.add('active');
        }
    });
    
    // Load tab content
    const tabContent = document.getElementById('tab-content');
    if (tabContent) {
        switch(tabName) {
            case 'detections':
                tabContent.innerHTML = '<div style="color: #00ff00; padding: 10px;">Detections panel - No active detections</div>';
                break;
            case 'suspect':
                tabContent.innerHTML = '<div style="color: #ffaa00; padding: 10px;">Suspect tracking - No suspects locked</div>';
                break;
            case 'maps':
                tabContent.innerHTML = '<div id="mini-map" style="height: 200px; background: #333; margin: 10px;">Map view</div>';
                break;
            case 'logs':
                tabContent.innerHTML = '<div style="color: #00ff00; padding: 10px; font-family: monospace; font-size: 10px;">System logs...<br>[INFO] SAR service started<br>[WARN] No video signal</div>';
                break;
        }
    }
}

class SARInterface {
    constructor() {
        this.websockets = {
            telemetry: null,
            detections: null,
            video: null
        };
        
        this.currentMode = 'sar'; // 'sar' or 'suspect-lock'
        this.detections = new Map();
        this.telemetryData = null;
        this.mappingDashboard = null;
        this.selectedDetection = null;
        this.missionStartTime = Date.now();
        this.frameCount = 0;
        this.lastFpsUpdate = Date.now();
        
        // Suspect-lock state
        this.targetImage = null;
        this.targetSignature = null;
        this.isTracking = false;
        this.currentTarget = null;
        
        // Statistics tracking
        this.stats = {
            totalDetections: 0,
            confirmedDetections: 0,
            searchArea: 0,
            flightDistance: 0,
            flightTime: 0
        };
        
        this.initializeInterface();
        this.initializeWebSockets();
        this.startUpdateLoop();
    }
    
    initializeInterface() {
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize UI state
        this.updateSystemTime();
        
        // Initialize default tab
        switchTab('detections');
        
        // Initialize mode dropdown
        this.updateModeIndicator();
        
        // Initialize target image handling
        this.initializeTargetImageHandling();
        
        // Make interface globally accessible
        window.sarInterface = this;
    }
    
    setupEventListeners() {
        // Video feed error handling
        const videoFeed = document.getElementById('video-feed');
        if (videoFeed) {
            videoFeed.addEventListener('error', () => {
                console.log('Video feed error');
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
    }
    
    toggleMode() {
        this.currentMode = this.currentMode === 'sar' ? 'suspect-lock' : 'sar';
        console.log('Mode switched to:', this.currentMode);
        // Update UI to reflect mode change
        this.updateModeDisplay();
    }
    
    updateModeDisplay() {
        // Update any mode-specific UI elements
        console.log('Current mode:', this.currentMode);
    }
    
    handleKeyboardShortcuts(e) {
        switch(e.key) {
            case 'f':
            case 'F':
                if (e.ctrlKey) {
                    e.preventDefault();
                    this.flagDetection();
                }
                break;
            case 'm':
            case 'M':
                if (e.ctrlKey) {
                    e.preventDefault();
                    this.toggleMode();
                }
                break;
            case 'h':
            case 'H':
                if (e.ctrlKey && this.selectedDetection) {
                    e.preventDefault();
                    this.initiateHandoff();
                }
                break;
        }
    }
    
    initializeMap() {
        // Map is now handled by MappingDashboard
        console.log('Map initialization delegated to MappingDashboard');
    }
    
    startUpdateLoop() {
        // Start the main update loop
        setInterval(() => {
            this.updateInterface();
        }, 100); // Update every 100ms
        
        // Update system time every second
        setInterval(() => {
            this.updateSystemTime();
        }, 1000);
    }
    
    startRealTimeUpdates() {
        // Simulate real-time stat updates like the React component
        setInterval(() => {
            // Update FPS with small random variations
            interfaceState.fps = parseFloat((interfaceState.fps + (Math.random() - 0.5) * 2).toFixed(1));
            interfaceState.fps = Math.max(8.0, Math.min(30.0, interfaceState.fps));
            
            // Update GPU usage
            interfaceState.gpu = Math.max(0, Math.min(100, interfaceState.gpu + Math.floor((Math.random() - 0.5) * 10)));
            
            // Update latency
            interfaceState.latency = Math.max(50, interfaceState.latency + Math.floor((Math.random() - 0.5) * 100));
            
            // Update HUD displays
            this.updateHUDDisplays();
        }, 2000);
    }
    
    updateHUDDisplays() {
        const fpsDisplay = document.getElementById('fps-display');
        const gpuDisplay = document.getElementById('gpu-display');
        
        if (fpsDisplay) {
            fpsDisplay.textContent = interfaceState.fps;
            // Color code FPS: red if low, green if good
            fpsDisplay.style.color = interfaceState.fps < 15 ? '#ff5555' : '#00ff00';
        }
        
        if (gpuDisplay) {
            gpuDisplay.textContent = `${interfaceState.gpu}%`;
            // Color code GPU: green if normal, yellow if high, red if very high
            if (interfaceState.gpu < 70) {
                gpuDisplay.style.color = '#00ff00';
            } else if (interfaceState.gpu < 90) {
                gpuDisplay.style.color = '#ffaa00';
            } else {
                gpuDisplay.style.color = '#ff5555';
            }
        }
        
        // Update status bar latency
        const statusItems = document.querySelectorAll('.status-item');
        statusItems.forEach(item => {
            if (item.textContent.includes('Latency:')) {
                item.innerHTML = `Latency: <span style="color: ${interfaceState.latency > 500 ? '#ff5555' : '#00ff00'}">${interfaceState.latency}ms</span>`;
            }
        });
    }
    
    updateInterface() {
        this.updateStatistics();
        this.updateMissionDuration();
    }
    
    updateSystemTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        const timeElement = document.getElementById('system-time');
        if (timeElement) {
            timeElement.textContent = timeString;
        }
    }
    
    updateMissionDuration() {
        const duration = Date.now() - this.missionStartTime;
        const hours = Math.floor(duration / 3600000);
        const minutes = Math.floor((duration % 3600000) / 60000);
        const seconds = Math.floor((duration % 60000) / 1000);
        
        const durationString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        const durationElement = document.getElementById('mission-duration');
        if (durationElement) {
            durationElement.textContent = durationString;
        }
    }
    
    // FPS tracking removed - now using real system metrics from API
    
    initializeWebSockets() {
        // Telemetry WebSocket
        this.connectTelemetryWS();
        
        // Detection WebSocket
        this.connectDetectionWS();
        
        // Video status monitoring
        this.monitorVideoFeed();
    }
    
    connectTelemetryWS() {
        try {
            this.telemetryWs = new WebSocket('ws://localhost:8004/ws/telemetry');
            
            this.telemetryWs.onopen = () => {
                console.log('Telemetry WebSocket connected');
                this.updateStatus('telemetry-status', 'connected');
            };
            
            this.telemetryWs.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.updateTelemetry(data);
            };
            
            this.telemetryWs.onclose = () => {
                console.log('Telemetry WebSocket disconnected');
                this.updateStatus('telemetry-status', 'error');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectTelemetryWS(), 5000);
            };
            
            this.telemetryWs.onerror = (error) => {
                console.error('Telemetry WebSocket error:', error);
                this.updateStatus('telemetry-status', 'error');
            };
        } catch (error) {
            console.error('Failed to connect telemetry WebSocket:', error);
            this.updateStatus('telemetry-status', 'error');
        }
    }
    
    connectDetectionWS() {
        try {
            this.detectionWs = new WebSocket('ws://localhost:8004/ws/detections');
            
            this.detectionWs.onopen = () => {
                console.log('Detection WebSocket connected');
                this.updateStatus('detection-status', 'connected');
            };
            
            this.detectionWs.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.updateDetections(data);
            };
            
            this.detectionWs.onclose = () => {
                console.log('Detection WebSocket disconnected');
                this.updateStatus('detection-status', 'error');
                setTimeout(() => this.connectDetectionWS(), 5000);
            };
            
            this.detectionWs.onerror = (error) => {
                console.error('Detection WebSocket error:', error);
                this.updateStatus('detection-status', 'error');
            };
        } catch (error) {
            console.error('Failed to connect detection WebSocket:', error);
            this.updateStatus('detection-status', 'error');
        }
    }
    
    monitorVideoFeed() {
        const video = document.getElementById('video-feed');
        
        video.addEventListener('loadstart', () => {
            this.updateStatus('video-status', 'warning');
        });
        
        video.addEventListener('canplay', () => {
            this.updateStatus('video-status', 'connected');
        });
        
        video.addEventListener('error', () => {
            this.updateStatus('video-status', 'error');
        });
        
        // Check if video is actually playing
        setInterval(() => {
            if (video.readyState >= 2 && !video.paused) {
                this.updateStatus('video-status', 'connected');
            } else {
                this.updateStatus('video-status', 'warning');
            }
        }, 5000);
    }
    
    updateTelemetry(data) {
        // Update HUD elements
        const hudElements = {
            'gps-coords': `${data.gps?.latitude?.toFixed(6) || '---'}°, ${data.gps?.longitude?.toFixed(6) || '---'}°`,
            'altitude': `${data.altitude?.toFixed(1) || '---'} m`,
            'heading': `${data.heading?.toFixed(0) || '---'}°`,
            'speed': `${data.speed?.toFixed(1) || '---'} m/s`,
            'battery': `${data.battery?.toFixed(0) || '---'}%`
        };
        
        Object.entries(hudElements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
                
                // Add warning/error classes based on values
                element.classList.remove('warning', 'error');
                if (id === 'battery') {
                    const battery = data.battery || 0;
                    if (battery < 20) element.classList.add('error');
                    else if (battery < 30) element.classList.add('warning');
                }
            }
        });
        
        // Update mapping dashboard
        if (this.mappingDashboard) {
            this.mappingDashboard.updateDronePosition(data);
        }
        
        this.telemetryData = data;
    }
    
    updateDetections(data) {
        const overlay = document.getElementById('video-overlay');
        const video = document.getElementById('video-feed');
        
        // Clear existing detection boxes
        overlay.innerHTML = '';
        
        if (data.detections && data.detections.length > 0) {
            this.updateStatus('geolocation-status', 'connected');
            
            data.detections.forEach((detection, index) => {
                // Create detection box on video
                this.createDetectionBox(detection, index);
                
                // Add/update detection marker on map
                if (detection.geolocation) {
                    this.updateDetectionMarker(detection);
                }
                
                // Store detection data
                this.detections.set(detection.id || index, detection);
            });
            
            // Update detection count
            document.getElementById('detection-count').textContent = 
                this.detections.size.toString();
        } else {
            this.updateStatus('geolocation-status', 'warning');
        }
    }
    
    createDetectionBox(detection, index) {
        const overlay = document.getElementById('video-overlay');
        const video = document.getElementById('video-feed');
        
        if (!detection.bbox) return;
        
        const [x1, y1, x2, y2] = detection.bbox;
        const videoRect = video.getBoundingClientRect();
        const overlayRect = overlay.getBoundingClientRect();
        
        // Calculate relative positions
        const scaleX = overlayRect.width / video.videoWidth;
        const scaleY = overlayRect.height / video.videoHeight;
        
        const box = document.createElement('div');
        box.className = 'detection-box';
        box.style.left = `${x1 * scaleX}px`;
        box.style.top = `${y1 * scaleY}px`;
        box.style.width = `${(x2 - x1) * scaleX}px`;
        box.style.height = `${(y2 - y1) * scaleY}px`;
        
        // Create label
        const label = document.createElement('div');
        label.className = 'detection-label';
        
        let labelText = `Person #${detection.id || index + 1}`;
        if (detection.confidence) {
            labelText += ` (${(detection.confidence * 100).toFixed(0)}%)`;
        }
        if (detection.distance) {
            labelText += ` - ${detection.distance.toFixed(0)}m`;
        }
        
        label.textContent = labelText;
        box.appendChild(label);
        
        // Add click handler
        box.addEventListener('click', (e) => {
            e.stopPropagation();
            this.showDetectionPopup(detection);
        });
        
        overlay.appendChild(box);
    }
    
    updateDetectionMarker(detection) {
        // Delegate to mapping dashboard
        if (this.mappingDashboard) {
            this.mappingDashboard.addDetectionMarker(detection);
        }
        
        // Update statistics
        this.stats.totalDetections++;
        this.updateStatistics();
    }
    
    updateStatistics() {
        // Update detection count
        const detectionElement = document.getElementById('detection-count');
        if (detectionElement) {
            detectionElement.textContent = this.stats.totalDetections.toString();
        }
        
        // Update search area from mapping dashboard
        if (this.mappingDashboard) {
            const searchArea = this.mappingDashboard.getSearchedArea();
            const areaElement = document.getElementById('area-covered');
            if (areaElement) {
                areaElement.textContent = searchArea.toFixed(2);
            }
            
            // Update flight distance
            const distance = this.mappingDashboard.getTotalDistance() / 1000; // Convert to km
            const distanceElement = document.getElementById('distance-flown');
            if (distanceElement) {
                distanceElement.textContent = distance.toFixed(2);
            }
        }
        
        // Update flight time
        const flightTime = Math.floor((Date.now() - this.missionStartTime) / 1000);
        const minutes = Math.floor(flightTime / 60);
        const seconds = flightTime % 60;
        const timeElement = document.getElementById('flight-time');
        if (timeElement) {
            timeElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }
    
    showDetectionPopup(detection) {
        const popup = document.getElementById('detection-popup');
        
        // Populate popup data
        document.getElementById('popup-person-id').textContent = 
            detection.id || 'Unknown';
        document.getElementById('popup-confidence').textContent = 
            `${(detection.confidence * 100).toFixed(0)}%`;
        
        if (detection.geolocation) {
            document.getElementById('popup-coordinates').textContent = 
                `${detection.geolocation.latitude.toFixed(6)}°, ${detection.geolocation.longitude.toFixed(6)}°`;
        } else {
            document.getElementById('popup-coordinates').textContent = 'Not available';
        }
        
        document.getElementById('popup-distance').textContent = 
            detection.distance ? `${detection.distance.toFixed(0)} m` : 'Unknown';
        
        document.getElementById('popup-timestamp').textContent = 
            new Date(detection.timestamp || Date.now()).toLocaleString();
        
        // Show popup
        popup.style.display = 'block';
        
        // Store current detection for actions
        this.currentDetection = detection;
    }
    
    closeDetectionPopup() {
        document.getElementById('detection-popup').style.display = 'none';
        this.currentDetection = null;
    }
    
    confirmSighting() {
        if (this.currentDetection) {
            console.log('Confirming sighting:', this.currentDetection);
            
            // Send confirmation to backend
            fetch('/api/confirm-sighting', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    detection: this.currentDetection,
                    timestamp: Date.now(),
                    operator: 'SAR_OPERATOR'
                })
            }).catch(error => console.error('Failed to confirm sighting:', error));
            
            this.closeDetectionPopup();
        }
    }
    
    async sendModeChange(mode) {
        try {
            const response = await fetch('/api/mode', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ mode })
            });
            
            if (!response.ok) {
                throw new Error('Failed to change mode');
            }
            
            console.log('Mode changed to:', mode);
            this.showNotification(`Switched to ${mode.toUpperCase().replace('-', ' ')} mode`, 'info');
        } catch (error) {
            console.error('Error changing mode:', error);
            this.showNotification('Failed to change mode', 'error');
        }
    }
    
    async sendFlagRequest(detection) {
        try {
            const response = await fetch('/api/flag', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    detection_id: detection.id,
                    mode: this.currentMode,
                    timestamp: Date.now()
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to flag detection');
            }
            
            console.log('Detection flagged successfully');
        } catch (error) {
            console.error('Error flagging detection:', error);
        }
    }
    
    async sendHandoffRequest(detection) {
        try {
            const response = await fetch('/api/handoff', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    detection: detection,
                    timestamp: Date.now(),
                    operator_notes: 'Confirmed detection - handoff to ground team'
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to send handoff request');
            }
            
            console.log('Handoff request sent successfully');
        } catch (error) {
            console.error('Error sending handoff request:', error);
            this.showNotification('Failed to send handoff request', 'error');
        }
    }
    
    async sendEmergencyStop() {
        try {
            const response = await fetch('/api/emergency-stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    timestamp: Date.now(),
                    reason: 'Manual emergency stop'
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to send emergency stop');
            }
            
            console.log('Emergency stop sent successfully');
        } catch (error) {
            console.error('Error sending emergency stop:', error);
        }
    }
    
    setMode(mode) {
        this.currentMode = mode;
        this.updateModeIndicator();
        this.sendModeChange(mode);
    }
    
    toggleMode() {
        const newMode = this.currentMode === 'sar' ? 'suspect-lock' : 'sar';
        this.setMode(newMode);
        
        // Update checkbox state
        const checkbox = document.getElementById('mode-checkbox');
        if (checkbox) {
            checkbox.checked = newMode === 'suspect-lock';
        }
    }
    
    updateModeIndicator() {
        const modeIndicator = document.getElementById('mode-indicator');
        const toggleSlider = document.getElementById('toggle-slider');
        const modeDropdown = document.getElementById('mode-dropdown');
        
        if (this.currentMode === 'sar') {
            if (modeIndicator) {
                modeIndicator.textContent = 'SAR MODE';
                modeIndicator.className = 'mode-indicator sar';
            }
            if (toggleSlider) {
                toggleSlider.textContent = 'SAR MODE';
            }
        } else {
            if (modeIndicator) {
                modeIndicator.textContent = 'SUSPECT LOCK';
                modeIndicator.className = 'mode-indicator suspect-lock';
            }
            if (toggleSlider) {
                toggleSlider.textContent = 'SUSPECT LOCK';
            }
        }
        
        // Update dropdown selection
        if (modeDropdown) {
            modeDropdown.value = this.currentMode;
        }
        
        // Show/hide suspect-lock panel based on mode
        const suspectPanel = document.getElementById('suspect-lock-panel');
        if (suspectPanel) {
            suspectPanel.style.display = this.currentMode === 'suspect-lock' ? 'block' : 'none';
        }
    }
    
    initializeTargetImageHandling() {
        const imageInput = document.getElementById('target-image-input');
        const uploadArea = document.getElementById('upload-area');
        
        if (imageInput) {
            imageInput.addEventListener('change', (e) => {
                this.handleImageSelection(e.target.files[0]);
            });
        }
        
        if (uploadArea) {
            uploadArea.addEventListener('click', () => {
                selectTargetImage();
            });
            
            // Drag and drop support
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('drag-over');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('drag-over');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleImageSelection(files[0]);
                }
            });
        }
    }
    
    handleImageSelection(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.targetImage = e.target.result;
            this.displayTargetPreview(e.target.result);
            this.enableProcessButton();
        };
        reader.readAsDataURL(file);
    }
    
    displayTargetPreview(imageSrc) {
        const preview = document.getElementById('target-preview');
        const placeholder = document.getElementById('upload-placeholder');
        
        if (preview && placeholder) {
            preview.src = imageSrc;
            preview.style.display = 'block';
            placeholder.style.display = 'none';
        }
    }
    
    enableProcessButton() {
        const processBtn = document.getElementById('process-target-btn');
        if (processBtn) {
            processBtn.disabled = false;
        }
    }
    
    async processTargetImage() {
        if (!this.targetImage) {
            this.showNotification('No target image selected', 'warning');
            return;
        }
        
        try {
            // Show processing state
            const processBtn = document.getElementById('process-target-btn');
            if (processBtn) {
                processBtn.disabled = true;
                processBtn.textContent = 'Processing...';
            }
            
            // Convert image to base64
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = async () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                
                // Send to backend for processing
                const response = await fetch('/api/process-target-image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image_data: imageData
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    this.targetSignature = result.signature_id;
                    this.displayTargetInfo(result);
                    this.showNotification('Target processed successfully', 'success');
                } else {
                    this.showNotification('Failed to process target image', 'error');
                }
                
                // Reset button state
                if (processBtn) {
                    processBtn.disabled = false;
                    processBtn.textContent = 'Process Target';
                }
            };
            
            img.src = this.targetImage;
            
        } catch (error) {
            console.error('Error processing target image:', error);
            this.showNotification('Error processing target image', 'error');
            
            // Reset button state
            const processBtn = document.getElementById('process-target-btn');
            if (processBtn) {
                processBtn.disabled = false;
                processBtn.textContent = 'Process Target';
            }
        }
    }
    
    displayTargetInfo(result) {
        const targetInfoSection = document.getElementById('target-info-section');
        const targetId = document.getElementById('target-id');
        const detectionMethod = document.getElementById('detection-method');
        const targetConfidence = document.getElementById('target-confidence');
        
        if (targetInfoSection) {
            targetInfoSection.style.display = 'block';
        }
        
        if (targetId) {
            targetId.textContent = result.target_id || 'TARGET-001';
        }
        
        if (detectionMethod) {
            detectionMethod.textContent = result.method || 'Face Recognition';
        }
        
        if (targetConfidence) {
            targetConfidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
        }
    }
    
    async clearTarget() {
        try {
            // Send clear target request to backend
            const response = await fetch('/api/clear-target', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.targetImage = null;
                this.targetSignature = null;
                this.currentTarget = null;
                
                // Clear tracking interval
                if (this.trackingInterval) {
                    clearInterval(this.trackingInterval);
                    this.trackingInterval = null;
                }
                
                // Reset UI
                const preview = document.getElementById('target-preview');
                const placeholder = document.getElementById('upload-placeholder');
                const targetInfoSection = document.getElementById('target-info-section');
                const trackingStatus = document.getElementById('tracking-status');
                const processBtn = document.getElementById('process-target-btn');
                
                if (preview) preview.style.display = 'none';
                if (placeholder) placeholder.style.display = 'block';
                if (targetInfoSection) targetInfoSection.style.display = 'none';
                if (trackingStatus) trackingStatus.style.display = 'none';
                if (processBtn) processBtn.disabled = true;
                
                const startBtn = document.getElementById('start-tracking-btn');
                const stopBtn = document.getElementById('stop-tracking-btn');
                if (startBtn) startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = true;
                
                // Stop tracking if active
                if (this.isTracking) {
                    this.stopTracking();
                }
                
                this.showNotification('Target cleared', 'info');
            } else {
                this.showNotification('Failed to clear target', 'error');
            }
        } catch (error) {
            console.error('Error clearing target:', error);
            this.showNotification('Error clearing target', 'error');
        }
    }
    
    async startTracking() {
        if (!this.targetSignature) {
            this.showNotification('No target signature available', 'warning');
            return;
        }
        
        try {
            // Send start tracking request to backend
            const response = await fetch('/api/start-tracking', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isTracking = true;
                
                // Update UI
                const startBtn = document.getElementById('start-tracking-btn');
                const stopBtn = document.getElementById('stop-tracking-btn');
                
                if (startBtn) startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = false;
                
                // Show tracking status
                const trackingStatus = document.getElementById('tracking-status');
                if (trackingStatus) {
                    trackingStatus.textContent = 'Active';
                    trackingStatus.className = 'status-indicator active';
                }
                
                this.showNotification('Tracking started', 'success');
                
                // Start periodic status updates
                this.trackingInterval = setInterval(() => {
                    this.checkTrackingStatus();
                }, 2000);
            } else {
                this.showNotification('Failed to start tracking', 'error');
            }
        } catch (error) {
            console.error('Error starting tracking:', error);
            this.showNotification('Error starting tracking', 'error');
        }
    }
    
    async stopTracking() {
        try {
            // Send stop tracking request to backend
            const response = await fetch('/api/stop-tracking', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isTracking = false;
                
                // Clear interval
                if (this.trackingInterval) {
                    clearInterval(this.trackingInterval);
                    this.trackingInterval = null;
                }
                
                // Update UI
                const startBtn = document.getElementById('start-tracking-btn');
                const stopBtn = document.getElementById('stop-tracking-btn');
                
                if (startBtn) startBtn.disabled = false;
                if (stopBtn) stopBtn.disabled = true;
                
                // Show tracking status
                const trackingStatus = document.getElementById('tracking-status');
                if (trackingStatus) {
                    trackingStatus.textContent = 'Inactive';
                    trackingStatus.className = 'status-indicator inactive';
                }
                
                this.showNotification('Tracking stopped', 'info');
            } else {
                this.showNotification('Failed to stop tracking', 'error');
            }
        } catch (error) {
            console.error('Error stopping tracking:', error);
            this.showNotification('Error stopping tracking', 'error');
        }
    }
    
    async checkTrackingStatus() {
        if (!this.isTracking) return;
        
        try {
            const response = await fetch('/api/tracking-status');
            const result = await response.json();
            
            if (result.status === 'success' && result.target_data) {
                this.updateTrackingInfo(result.target_data);
            }
        } catch (error) {
            console.error('Error checking tracking status:', error);
        }
    }
    
    updateTrackingInfo(targetData) {
        if (!this.isTracking || !targetData) return;
        
        const lastSeenTime = document.getElementById('last-seen-time');
        const targetLocation = document.getElementById('target-location');
        const matchScore = document.getElementById('match-score');
        
        if (lastSeenTime) {
            lastSeenTime.textContent = new Date().toLocaleTimeString();
        }
        
        if (targetLocation && targetData.location) {
            targetLocation.textContent = `${targetData.location.lat.toFixed(6)}, ${targetData.location.lng.toFixed(6)}`;
        }
        
        if (matchScore && targetData.confidence) {
            matchScore.textContent = `${(targetData.confidence * 100).toFixed(1)}%`;
        }
        
        this.currentTarget = targetData;
    }
    
    flagDetection() {
        if (this.detections.size === 0) {
            this.showNotification('No detections available to flag', 'warning');
            return;
        }
        
        // Get the most recent detection
        const latestDetection = Array.from(this.detections.values()).pop();
        this.selectedDetection = latestDetection;
        
        console.log('Flagging detection:', latestDetection.id, 'in', this.currentMode, 'mode');
        
        // Enable handoff button
        const handoffBtn = document.getElementById('handoff-btn');
        if (handoffBtn) {
            handoffBtn.disabled = false;
        }
        
        // Show detection popup
        this.showDetectionPopup(latestDetection);
        
        // Send flag request to backend
        this.sendFlagRequest(latestDetection);
    }
    
    initiateHandoff() {
        if (!this.selectedDetection) {
            this.showNotification('No detection selected for handoff', 'warning');
            return;
        }
        
        console.log('Initiating handoff for detection:', this.selectedDetection.id);
        
        // Show handoff confirmation
        const confirmed = confirm(`Confirm handoff of Person #${this.selectedDetection.id} to ground team?\n\nLocation: ${this.selectedDetection.geolocation?.latitude?.toFixed(6)}°, ${this.selectedDetection.geolocation?.longitude?.toFixed(6)}°`);
        
        if (confirmed) {
            this.sendHandoffRequest(this.selectedDetection);
            this.showNotification('Handoff request sent to ground team', 'success');
            
            // Update statistics
            this.stats.confirmedDetections++;
        }
    }
    
    emergencyStop() {
        const confirmed = confirm('EMERGENCY STOP: This will immediately halt all drone operations. Continue?');
        if (confirmed) {
            console.log('Emergency stop initiated');
            this.sendEmergencyStop();
            this.showNotification('Emergency stop activated', 'error');
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#ff4444' : type === 'warning' ? '#ff9800' : type === 'success' ? '#4CAF50' : '#2196F3'};
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            z-index: 2000;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'fadeOut 0.3s ease-out';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    captureVideoSnapshot() {
        const video = document.getElementById('video-feed');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        
        return canvas.toDataURL('image/jpeg', 0.8);
    }
    
    updateStatus(elementId, status) {
        const indicator = document.getElementById(elementId);
        if (indicator) {
            indicator.className = `status-indicator ${status === 'connected' ? '' : status}`;
        }
    }
    
    calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371000; // Earth's radius in meters
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }
    
    flagCurrentDetection() {
        // Flag the most recent or highest confidence detection
        if (this.detections.size > 0) {
            const latestDetection = Array.from(this.detections.values())
                .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))[0];
            
            this.showDetectionPopup(latestDetection);
        } else {
            alert('No detections to flag.');
        }
    }
    
    handoffToTeam() {
        if (this.currentDetection) {
            console.log('Handing off to team:', this.currentDetection);
            
            // Prepare handoff data
            const handoffData = {
                detection: this.currentDetection,
                timestamp: Date.now(),
                coordinates: this.currentDetection.geolocation,
                snapshot: this.captureVideoSnapshot(),
                operator: 'SAR_OPERATOR',
                priority: 'HIGH'
            };
            
            // Send handoff request
            fetch('/api/handoff-to-team', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(handoffData)
            }).then(response => {
                if (response.ok) {
                    alert('Handoff request sent successfully!');
                } else {
                    alert('Failed to send handoff request.');
                }
            }).catch(error => {
                console.error('Failed to handoff to team:', error);
                alert('Failed to send handoff request.');
            });
            
            this.closeDetectionPopup();
        }
    }
    
    handleVideoClick(e) {
        // Handle clicks on video overlay for manual marking
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        console.log(`Video clicked at: ${x}, ${y}`);
        // Could implement manual detection marking here
    }
    
    // Data Export and Mission Summary Methods
    async exportMissionLogs() {
        try {
            const response = await fetch('/api/export-logs');
            if (!response.ok) {
                throw new Error(`Export failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Create download links for exported data
            if (data.export_data) {
                this.downloadExportedData(data.export_data);
                alert('Mission logs exported successfully!');
            }
        } catch (error) {
            console.error('Failed to export mission logs:', error);
            alert('Failed to export mission logs: ' + error.message);
        }
    }
    
    async getStorageStats() {
        try {
            const response = await fetch('/api/storage-stats');
            if (!response.ok) {
                throw new Error(`Stats retrieval failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            return data.storage_stats;
        } catch (error) {
            console.error('Failed to get storage stats:', error);
            throw error;
        }
    }
    
    async createMissionArchive(format = 'zip', includeVideo = true) {
        try {
            const response = await fetch('/api/create-mission-archive', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    format: format,
                    include_video: includeVideo
                })
            });
            
            if (!response.ok) {
                throw new Error(`Archive creation failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            alert(`Mission archive created: ${data.archive_path}`);
            return data.archive_path;
        } catch (error) {
            console.error('Failed to create mission archive:', error);
            alert('Failed to create mission archive: ' + error.message);
            throw error;
        }
    }
    
    downloadExportedData(exportData) {
        // Download JSON data
        if (exportData.json_path) {
            this.downloadFile(exportData.json_path, 'mission_logs.json');
        }
        
        // Download CSV data
        if (exportData.csv_path) {
            this.downloadFile(exportData.csv_path, 'mission_summary.csv');
        }
        
        // Show summary information
        if (exportData.summary) {
            this.showMissionSummary(exportData.summary);
        }
    }
    
    downloadFile(filePath, fileName) {
        // Create a download link for the file
        const link = document.createElement('a');
        link.href = `/api/download/${encodeURIComponent(filePath)}`;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    showMissionSummary(summary) {
        const summaryText = `
Mission Summary:
- Total Detections: ${summary.total_detections || 0}
- Suspect Matches: ${summary.suspect_matches || 0}
- Confirmed Sightings: ${summary.confirmed_sightings || 0}
- Handoff Requests: ${summary.handoff_requests || 0}
- Session Duration: ${this.formatDuration(summary.session_duration || 0)}
- Data Storage Used: ${this.formatBytes(summary.storage_used || 0)}
        `;
        
        alert(summaryText);
    }
    
    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours}h ${minutes}m ${secs}s`;
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the SAR interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.sarInterface = new SARInterface();
});