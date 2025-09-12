#!/usr/bin/env python3
"""
FPV Live Detection Application

Integrates FPV video capture with real-time YOLO detection for SAR operations.
Provides a web interface to display live footage with bounding boxes.

Features:
- Real-time FPV video capture from multiple sources
- YOLO object detection with SAR-optimized filtering
- Live web streaming with detection overlays
- Performance monitoring and statistics
- WebSocket updates for real-time control

Author: Foresight AI Team
Date: 2024
"""

import asyncio
import cv2
import numpy as np
import json
import logging
import time
from typing import Optional, Dict, List
from pathlib import Path
import threading
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import our custom services
from services.fpv_capture import FPVCaptureService, CaptureConfig, CaptureSourceType, create_fpv_capture, list_available_sources
from services.yolo_wrapper import YOLOWrapper, YOLOConfig, ModelType, create_yolo_wrapper

logger = logging.getLogger(__name__)

class FPVLiveApp:
    """
    FPV Live Detection Application
    
    Orchestrates video capture, object detection, and web streaming
    for real-time SAR operations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.app = FastAPI(title="FPV Live Detection", version="1.0.0")
        
        # Services
        self.capture_service: Optional[FPVCaptureService] = None
        self.yolo_wrapper: Optional[YOLOWrapper] = None
        
        # State
        self.is_running = False
        self.detection_enabled = True
        self.current_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        # Statistics
        self.stats = {
            'fps': 0.0,
            'detection_fps': 0.0,
            'total_detections': 0,
            'people_detected': 0,
            'capture_resolution': '0x0',
            'model_loaded': False,
            'capture_connected': False,
            'uptime_seconds': 0,
            'start_time': None
        }
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.last_stats_update = time.time()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup FastAPI routes
        self._setup_routes()
        
        # Initialize services
        self._initialize_services()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load application configuration"""
        default_config = {
            'capture': {
                'auto_detect': True,
                'source_type': 'webcam',
                'source_id': '0',
                'width': 1920,
                'height': 1080,
                'fps': 30,
                'buffer_size': 1
            },
            'yolo': {
                'model_path': 'models/yolo11n.pt',
                'model_type': 'yolo11',
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'sar_mode': True,
                'device': 'auto'
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'stream_quality': 80,
                'stream_fps': 15
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for section, values in user_config.items():
                        if section in default_config:
                            default_config[section].update(values)
                        else:
                            default_config[section] = values
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Main page"""
            return self._get_html_template()
        
        @self.app.get("/video_feed")
        async def video_feed():
            """Video streaming endpoint"""
            return StreamingResponse(
                self._generate_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        
        @self.app.get("/api/stats")
        async def get_stats():
            """Get current statistics"""
            return self._get_current_stats()
        
        @self.app.get("/api/sources")
        async def get_sources():
            """Get available video sources"""
            return list_available_sources()
        
        @self.app.post("/api/start")
        async def start_capture():
            """Start video capture and detection"""
            success = self.start()
            return {"success": success, "message": "Started" if success else "Failed to start"}
        
        @self.app.post("/api/stop")
        async def stop_capture():
            """Stop video capture and detection"""
            self.stop()
            return {"success": True, "message": "Stopped"}
        
        @self.app.post("/api/toggle_detection")
        async def toggle_detection():
            """Toggle object detection on/off"""
            self.detection_enabled = not self.detection_enabled
            await self._broadcast_status()
            return {"success": True, "detection_enabled": self.detection_enabled}
        
        @self.app.post("/api/change_source")
        async def change_source(request: Request):
            """Change video source"""
            data = await request.json()
            source_type = data.get('source_type', 'webcam')
            source_id = data.get('source_id', '0')
            
            success = self._change_source(source_type, source_id)
            return {"success": success, "message": "Source changed" if success else "Failed to change source"}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic status updates
                    await asyncio.sleep(1)
                    await self._send_websocket_update(websocket)
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    def _initialize_services(self):
        """Initialize capture and detection services"""
        try:
            # Initialize YOLO
            yolo_config = YOLOConfig(
                model_path=self.config['yolo']['model_path'],
                model_type=ModelType(self.config['yolo']['model_type']),
                confidence_threshold=self.config['yolo']['confidence_threshold'],
                nms_threshold=self.config['yolo']['nms_threshold'],
                sar_mode=self.config['yolo']['sar_mode'],
                device=self.config['yolo']['device']
            )
            
            self.yolo_wrapper = YOLOWrapper(yolo_config)
            self.stats['model_loaded'] = self.yolo_wrapper.is_model_loaded()
            
            # Initialize capture
            if self.config['capture']['auto_detect']:
                self.capture_service = create_fpv_capture()
            else:
                capture_config = CaptureConfig(
                    source_type=CaptureSourceType(self.config['capture']['source_type']),
                    source_id=self.config['capture']['source_id'],
                    width=self.config['capture']['width'],
                    height=self.config['capture']['height'],
                    fps=self.config['capture']['fps'],
                    buffer_size=self.config['capture']['buffer_size']
                )
                self.capture_service = FPVCaptureService(capture_config)
            
            logger.info("Services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
    
    def start(self) -> bool:
        """Start the FPV live detection system"""
        if self.is_running:
            logger.warning("System already running")
            return True
        
        logger.info("Starting FPV live detection system...")
        
        # Start capture service
        if not self.capture_service or not self.capture_service.start():
            logger.error("Failed to start capture service")
            return False
        
        # Add frame callback for detection
        self.capture_service.add_frame_callback(self._process_frame)
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        self.stats['capture_connected'] = True
        
        logger.info("FPV live detection system started")
        return True
    
    def stop(self):
        """Stop the FPV live detection system"""
        if not self.is_running:
            return
        
        logger.info("Stopping FPV live detection system...")
        
        self.is_running = False
        
        if self.capture_service:
            self.capture_service.stop()
        
        self.stats['capture_connected'] = False
        
        logger.info("FPV live detection system stopped")
    
    def _process_frame(self, frame: np.ndarray):
        """Process captured frame with detection"""
        try:
            with self.frame_lock:
                self.current_frame = frame.copy()
                
                if self.detection_enabled and self.yolo_wrapper and self.yolo_wrapper.is_model_loaded():
                    # Run detection
                    detections = self.yolo_wrapper.detect(frame)
                    
                    # Draw detections
                    self.annotated_frame = self.yolo_wrapper.draw_detections(frame, detections)
                    
                    # Update statistics
                    self.detection_count += 1
                    self.stats['total_detections'] += len(detections)
                    
                    # Count people detections
                    people_count = sum(1 for d in detections if 'person' in d.class_name.lower())
                    self.stats['people_detected'] += people_count
                    
                else:
                    # No detection, use original frame
                    self.annotated_frame = frame.copy()
                
                self.frame_count += 1
                
                # Update FPS every second
                current_time = time.time()
                if current_time - self.last_stats_update >= 1.0:
                    time_diff = current_time - self.last_stats_update
                    self.stats['fps'] = self.frame_count / time_diff
                    self.stats['detection_fps'] = self.detection_count / time_diff
                    
                    # Update other stats
                    if self.capture_service:
                        width, height = self.capture_service.get_resolution()
                        self.stats['capture_resolution'] = f"{width}x{height}"
                    
                    if self.stats['start_time']:
                        self.stats['uptime_seconds'] = int((datetime.now() - self.stats['start_time']).total_seconds())
                    
                    # Reset counters
                    self.frame_count = 0
                    self.detection_count = 0
                    self.last_stats_update = current_time
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def _generate_frames(self):
        """Generate frames for video streaming"""
        while True:
            try:
                with self.frame_lock:
                    if self.annotated_frame is not None:
                        frame = self.annotated_frame.copy()
                    else:
                        # Generate a placeholder frame
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "No Video Signal", (200, 240), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add overlay information
                frame = self._add_overlay_info(frame)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, self.config['server']['stream_quality']])
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # Control streaming FPS
                time.sleep(1.0 / self.config['server']['stream_fps'])
                
            except Exception as e:
                logger.error(f"Error generating frame: {e}")
                time.sleep(0.1)
    
    def _add_overlay_info(self, frame: np.ndarray) -> np.ndarray:
        """Add overlay information to frame"""
        overlay_frame = frame.copy()
        
        # Add status information
        status_text = f"FPS: {self.stats['fps']:.1f} | Detection: {'ON' if self.detection_enabled else 'OFF'}"
        cv2.putText(overlay_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add detection count
        if self.detection_enabled:
            det_text = f"Detections: {self.stats['total_detections']} | People: {self.stats['people_detected']}"
            cv2.putText(overlay_frame, det_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(overlay_frame, timestamp, (10, overlay_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def _get_current_stats(self) -> Dict:
        """Get current system statistics"""
        stats = self.stats.copy()
        
        # Add service-specific stats
        if self.capture_service:
            capture_stats = self.capture_service.get_stats()
            stats.update({
                'capture_fps': capture_stats.get('avg_fps', 0),
                'frames_captured': capture_stats.get('frames_captured', 0),
                'frames_dropped': capture_stats.get('frames_dropped', 0)
            })
        
        if self.yolo_wrapper:
            yolo_stats = self.yolo_wrapper.get_stats()
            stats.update({
                'avg_inference_time_ms': yolo_stats.get('avg_inference_time_ms', 0),
                'total_inferences': yolo_stats.get('total_inferences', 0)
            })
        
        return stats
    
    def _change_source(self, source_type: str, source_id: str) -> bool:
        """Change video source"""
        try:
            was_running = self.is_running
            
            if was_running:
                self.stop()
            
            # Create new capture config
            capture_config = CaptureConfig(
                source_type=CaptureSourceType(source_type),
                source_id=source_id,
                width=self.config['capture']['width'],
                height=self.config['capture']['height'],
                fps=self.config['capture']['fps']
            )
            
            # Create new capture service
            self.capture_service = FPVCaptureService(capture_config)
            
            if was_running:
                return self.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to change source: {e}")
            return False
    
    async def _broadcast_status(self):
        """Broadcast status to all WebSocket connections"""
        if not self.websocket_connections:
            return
        
        status = {
            'type': 'status',
            'data': {
                'is_running': self.is_running,
                'detection_enabled': self.detection_enabled,
                'stats': self._get_current_stats()
            }
        }
        
        # Send to all connections
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(status))
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            if ws in self.websocket_connections:
                self.websocket_connections.remove(ws)
    
    async def _send_websocket_update(self, websocket: WebSocket):
        """Send update to specific WebSocket connection"""
        try:
            update = {
                'type': 'update',
                'data': self._get_current_stats()
            }
            await websocket.send_text(json.dumps(update))
        except:
            pass
    
    def _get_html_template(self) -> str:
        """Get HTML template for the web interface"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FPV Live Detection - Foresight SAR</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            align-items: start;
        }
        .video-section {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .video-container {
            position: relative;
            width: 100%;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        #videoFeed {
            width: 100%;
            height: auto;
            display: block;
        }
        .controls {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .control-group {
            margin-bottom: 25px;
        }
        .control-group h3 {
            margin: 0 0 15px 0;
            color: #fff;
            font-size: 1.2em;
        }
        .btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .btn.stop {
            background: linear-gradient(45deg, #f44336, #d32f2f);
        }
        .btn.toggle {
            background: linear-gradient(45deg, #ff9800, #f57c00);
        }
        .stats {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .stat-item:last-child {
            border-bottom: none;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-connected {
            background: #4CAF50;
            box-shadow: 0 0 10px #4CAF50;
        }
        .status-disconnected {
            background: #f44336;
        }
        .source-selector {
            margin: 15px 0;
        }
        .source-selector select {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: none;
            background: rgba(255,255,255,0.9);
            color: #333;
            font-size: 14px;
        }
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚁 FPV Live Detection</h1>
            <p>Real-time Search and Rescue Video Analysis</p>
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <h2>Live Video Feed</h2>
                <div class="video-container">
                    <img id="videoFeed" src="/video_feed" alt="Video Feed">
                </div>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <h3>System Control</h3>
                    <button class="btn" onclick="startSystem()">▶️ Start</button>
                    <button class="btn stop" onclick="stopSystem()">⏹️ Stop</button>
                    <button class="btn toggle" onclick="toggleDetection()">🎯 Toggle Detection</button>
                </div>
                
                <div class="control-group">
                    <h3>Video Source</h3>
                    <div class="source-selector">
                        <select id="sourceSelect">
                            <option value="webcam:0">Webcam (Device 0)</option>
                            <option value="uvc_capture:1">Capture Card (Device 1)</option>
                            <option value="uvc_capture:2">Capture Card (Device 2)</option>
                        </select>
                        <button class="btn" onclick="changeSource()" style="width: 100%; margin-top: 10px;">Change Source</button>
                    </div>
                </div>
                
                <div class="control-group">
                    <h3>System Status</h3>
                    <div class="stats">
                        <div class="stat-item">
                            <span>Connection:</span>
                            <span><span class="status-indicator" id="connectionStatus"></span><span id="connectionText">Disconnected</span></span>
                        </div>
                        <div class="stat-item">
                            <span>FPS:</span>
                            <span id="fpsValue">0.0</span>
                        </div>
                        <div class="stat-item">
                            <span>Detection:</span>
                            <span id="detectionStatus">OFF</span>
                        </div>
                        <div class="stat-item">
                            <span>Total Detections:</span>
                            <span id="totalDetections">0</span>
                        </div>
                        <div class="stat-item">
                            <span>People Found:</span>
                            <span id="peopleDetected">0</span>
                        </div>
                        <div class="stat-item">
                            <span>Resolution:</span>
                            <span id="resolution">0x0</span>
                        </div>
                        <div class="stat-item">
                            <span>Uptime:</span>
                            <span id="uptime">0s</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'update' || data.type === 'status') {
                    updateStats(data.data);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
            };
        }
        
        function updateStats(stats) {
            document.getElementById('fpsValue').textContent = stats.fps?.toFixed(1) || '0.0';
            document.getElementById('totalDetections').textContent = stats.total_detections || 0;
            document.getElementById('peopleDetected').textContent = stats.people_detected || 0;
            document.getElementById('resolution').textContent = stats.capture_resolution || '0x0';
            document.getElementById('uptime').textContent = formatUptime(stats.uptime_seconds || 0);
            
            // Update connection status
            const connectionStatus = document.getElementById('connectionStatus');
            const connectionText = document.getElementById('connectionText');
            if (stats.capture_connected) {
                connectionStatus.className = 'status-indicator status-connected';
                connectionText.textContent = 'Connected';
            } else {
                connectionStatus.className = 'status-indicator status-disconnected';
                connectionText.textContent = 'Disconnected';
            }
            
            // Update detection status
            const detectionStatus = document.getElementById('detectionStatus');
            if (stats.detection_enabled !== undefined) {
                detectionStatus.textContent = stats.detection_enabled ? 'ON' : 'OFF';
                detectionStatus.style.color = stats.detection_enabled ? '#4CAF50' : '#f44336';
            }
        }
        
        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;
            
            if (hours > 0) {
                return `${hours}h ${minutes}m ${secs}s`;
            } else if (minutes > 0) {
                return `${minutes}m ${secs}s`;
            } else {
                return `${secs}s`;
            }
        }
        
        async function startSystem() {
            try {
                const response = await fetch('/api/start', { method: 'POST' });
                const result = await response.json();
                console.log('Start result:', result);
            } catch (error) {
                console.error('Error starting system:', error);
            }
        }
        
        async function stopSystem() {
            try {
                const response = await fetch('/api/stop', { method: 'POST' });
                const result = await response.json();
                console.log('Stop result:', result);
            } catch (error) {
                console.error('Error stopping system:', error);
            }
        }
        
        async function toggleDetection() {
            try {
                const response = await fetch('/api/toggle_detection', { method: 'POST' });
                const result = await response.json();
                console.log('Toggle detection result:', result);
            } catch (error) {
                console.error('Error toggling detection:', error);
            }
        }
        
        async function changeSource() {
            const sourceSelect = document.getElementById('sourceSelect');
            const [sourceType, sourceId] = sourceSelect.value.split(':');
            
            try {
                const response = await fetch('/api/change_source', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        source_type: sourceType,
                        source_id: sourceId
                    })
                });
                const result = await response.json();
                console.log('Change source result:', result);
            } catch (error) {
                console.error('Error changing source:', error);
            }
        }
        
        // Initialize
        connectWebSocket();
        
        // Load available sources
        fetch('/api/sources')
            .then(response => response.json())
            .then(sources => {
                const select = document.getElementById('sourceSelect');
                select.innerHTML = '';
                sources.forEach(source => {
                    const option = document.createElement('option');
                    option.value = `${source.type}:${source.id}`;
                    option.textContent = `${source.name} (${source.resolution})`;
                    select.appendChild(option);
                });
            })
            .catch(error => console.error('Error loading sources:', error));
    </script>
</body>
</html>
        """
    
    def run(self):
        """Run the FPV live detection application"""
        host = self.config['server']['host']
        port = self.config['server']['port']
        
        logger.info(f"Starting FPV Live Detection server on {host}:{port}")
        
        try:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=False
            )
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop()

def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the application
    app = FPVLiveApp()
    
    # Auto-start if configured
    if app.config.get('auto_start', False):
        app.start()
    
    app.run()

if __name__ == "__main__":
    main()