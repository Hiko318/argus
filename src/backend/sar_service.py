from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Import our existing services
from .geolocation_pipeline import GeolocationPipeline
from .telemetry_service import get_telemetry_service
from .detection_pipeline import DetectionPipeline
from .identity_service import IdentityService
from .logging_service import get_logging_service
from .storage_service import get_storage_service

logger = logging.getLogger(__name__)

@dataclass
class SARDetection:
    """SAR-specific detection data structure"""
    id: str
    bbox: tuple
    confidence: float
    geolocation: Optional[Dict[str, float]] = None
    distance: Optional[float] = None
    timestamp: float = None
    class_name: str = "person"
    track_id: Optional[int] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class SARTelemetry:
    """SAR-specific telemetry data structure"""
    gps: Dict[str, float]
    altitude: float
    heading: float
    speed: float
    battery: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.telemetry_connections: List[WebSocket] = []
        self.detection_connections: List[WebSocket] = []
        self.video_connections: List[WebSocket] = []
    
    async def connect_telemetry(self, websocket: WebSocket):
        await websocket.accept()
        self.telemetry_connections.append(websocket)
        logger.info(f"Telemetry client connected. Total: {len(self.telemetry_connections)}")
    
    async def connect_detection(self, websocket: WebSocket):
        await websocket.accept()
        self.detection_connections.append(websocket)
        logger.info(f"Detection client connected. Total: {len(self.detection_connections)}")
    
    async def connect_video(self, websocket: WebSocket):
        await websocket.accept()
        self.video_connections.append(websocket)
        logger.info(f"Video client connected. Total: {len(self.video_connections)}")
    
    def disconnect_telemetry(self, websocket: WebSocket):
        if websocket in self.telemetry_connections:
            self.telemetry_connections.remove(websocket)
            logger.info(f"Telemetry client disconnected. Total: {len(self.telemetry_connections)}")
    
    def disconnect_detection(self, websocket: WebSocket):
        if websocket in self.detection_connections:
            self.detection_connections.remove(websocket)
            logger.info(f"Detection client disconnected. Total: {len(self.detection_connections)}")
    
    def disconnect_video(self, websocket: WebSocket):
        if websocket in self.video_connections:
            self.video_connections.remove(websocket)
            logger.info(f"Video client disconnected. Total: {len(self.video_connections)}")
    
    async def broadcast_telemetry(self, data: Dict[str, Any]):
        """Broadcast telemetry data to all connected clients"""
        if self.telemetry_connections:
            message = json.dumps(data)
            disconnected = []
            
            for connection in self.telemetry_connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending telemetry data: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                self.disconnect_telemetry(connection)
    
    async def broadcast_detections(self, data: Dict[str, Any]):
        """Broadcast detection data to all connected clients"""
        if self.detection_connections:
            message = json.dumps(data)
            disconnected = []
            
            for connection in self.detection_connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending detection data: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                self.disconnect_detection(connection)

class SARService:
    """Main SAR service that coordinates all components"""
    
    def __init__(self):
        self.app = FastAPI(title="Foresight SAR Service")
        self.connection_manager = ConnectionManager()
        self.current_mode = "sar"  # "sar" or "suspect"
        self.active_detections = {}
        self.confirmed_sightings = []
        self.handoff_requests = []
        
        # Suspect-Lock mode state
        self.current_target_signature = None
        self.tracking_active = False
        self.suspect_matches = []
        self.start_time = time.time()  # Track session start time
        
        # Initialize pipelines
        self.geolocation_pipeline = None
        self.detection_pipeline = None
        self.telemetry_service = None
        self.identity_service = IdentityService()
        
        # Initialize logging and storage services
        self.logging_service = get_logging_service()
        self.storage_service = get_storage_service()
        
        # Video recording for storage
        self.video_frames_buffer = []
        self.max_buffer_size = 300  # 10 seconds at 30fps
        
        self.setup_routes()
        self.setup_static_files()
        
    def setup_static_files(self):
        """Setup static file serving for the frontend"""
        frontend_path = Path(__file__).parent.parent / "frontend"
        if frontend_path.exists():
            self.app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    
    def setup_routes(self):
        """Setup all API routes and WebSocket endpoints"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_interface():
            """Serve the main SAR interface"""
            frontend_path = Path(__file__).parent.parent / "frontend" / "sar_interface.html"
            if frontend_path.exists():
                return HTMLResponse(content=frontend_path.read_text(), status_code=200)
            else:
                return HTMLResponse(content="<h1>SAR Interface Not Found</h1>", status_code=404)
        
        @self.app.websocket("/ws/telemetry")
        async def telemetry_websocket(websocket: WebSocket):
            await self.connection_manager.connect_telemetry(websocket)
            try:
                while True:
                    # Keep connection alive and handle any incoming messages
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connection_manager.disconnect_telemetry(websocket)
        
        @self.app.websocket("/ws/detections")
        async def detection_websocket(websocket: WebSocket):
            await self.connection_manager.connect_detection(websocket)
            try:
                while True:
                    # Keep connection alive and handle any incoming messages
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connection_manager.disconnect_detection(websocket)
        
        @self.app.get("/api/video-stream")
        async def video_stream():
            """Serve video stream (placeholder - integrate with your video source)"""
            def generate_frames():
                # This is a placeholder - integrate with your actual video source
                cap = cv2.VideoCapture(0)  # or your video source
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                cap.release()
            
            return StreamingResponse(
                generate_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        
        @self.app.post("/api/set-mode")
        async def set_mode(request: Dict[str, str]):
            """Set the current operating mode"""
            mode = request.get("mode")
            if mode in ["sar", "suspect"]:
                old_mode = self.current_mode
                self.current_mode = mode
                logger.info(f"Mode changed to: {mode}")
                
                # Log mode change event
                telemetry = self.telemetry_service.get_current_telemetry()
                self.logging_service.log_system_event(
                    event_type="mode_change",
                    description=f"Mode changed from {old_mode} to {mode}",
                    metadata={
                        "old_mode": old_mode,
                        "new_mode": mode,
                        "operator": request.get("operator", "unknown")
                    },
                    drone_gps=telemetry.gps_coordinates if telemetry else None
                )
                
                return {"status": "success", "mode": mode}
            else:
                raise HTTPException(status_code=400, detail="Invalid mode")
        
        @self.app.get("/api/mode")
        async def get_mode():
            """Get the current operating mode"""
            return {"mode": self.current_mode}
        
        @self.app.post("/api/confirm-sighting")
        async def confirm_sighting(request: Dict[str, Any]):
            """Confirm a sighting"""
            sighting = {
                "id": len(self.confirmed_sightings) + 1,
                "detection": request.get("detection"),
                "timestamp": request.get("timestamp", time.time()),
                "operator": request.get("operator", "unknown"),
                "confirmed_at": time.time()
            }
            
            self.confirmed_sightings.append(sighting)
            logger.info(f"Sighting confirmed: {sighting['id']}")
            
            # Log sighting confirmation event
            telemetry = self.telemetry_service.get_current_telemetry()
            detection = request.get("detection", {})
            self.logging_service.log_system_event(
                event_type="sighting_confirmed",
                description=f"Sighting {sighting['id']} confirmed by {sighting['operator']}",
                metadata={
                    "sighting_id": sighting["id"],
                    "operator": sighting["operator"],
                    "detection_confidence": detection.get("confidence"),
                    "bounding_box": detection.get("bbox"),
                    "suspect_lock_status": detection.get("is_suspect", False)
                },
                drone_gps=telemetry.gps_coordinates if telemetry else None
            )
            
            return {"status": "success", "sighting_id": sighting["id"]}
        
        @self.app.post("/api/handoff-to-team")
        async def handoff_to_team(request: Dict[str, Any]):
            """Create a handoff request to rescue teams"""
            handoff = {
                "id": len(self.handoff_requests) + 1,
                "detection": request.get("detection"),
                "coordinates": request.get("coordinates"),
                "snapshot": request.get("snapshot"),
                "timestamp": request.get("timestamp", time.time()),
                "operator": request.get("operator", "unknown"),
                "priority": request.get("priority", "MEDIUM"),
                "status": "PENDING",
                "created_at": time.time()
            }
            
            self.handoff_requests.append(handoff)
            logger.info(f"Handoff request created: {handoff['id']}")
            
            # Log handoff event
            telemetry = self.telemetry_service.get_current_telemetry()
            self.logging_service.log_handoff_event(
                handoff_id=handoff["id"],
                target_coordinates=handoff["coordinates"],
                priority=handoff["priority"],
                operator=handoff["operator"],
                detection_data=handoff["detection"],
                drone_gps=telemetry.gps_coordinates if telemetry else None,
                metadata={
                    "status": handoff["status"],
                    "snapshot_available": bool(handoff.get("snapshot")),
                    "timestamp": handoff["timestamp"]
                }
            )
            
            # Here you would typically send this to external systems
            # (radio, satellite communication, etc.)
            
            return {"status": "success", "handoff_id": handoff["id"]}
        
        @self.app.get("/api/sightings")
        async def get_sightings():
            """Get all confirmed sightings"""
            return {"sightings": self.confirmed_sightings}
        
        @self.app.get("/api/handoffs")
        async def get_handoffs():
            """Get all handoff requests"""
            return {"handoffs": self.handoff_requests}
        
        @self.app.get("/api/status")
        async def get_system_status():
            """Get overall system status"""
            return {
                "mode": self.current_mode,
                "active_detections": len(self.active_detections),
                "confirmed_sightings": len(self.confirmed_sightings),
                "pending_handoffs": len([h for h in self.handoff_requests if h["status"] == "PENDING"]),
                "telemetry_connected": len(self.connection_manager.telemetry_connections) > 0,
                "detection_connected": len(self.connection_manager.detection_connections) > 0,
                "timestamp": time.time()
            }
        
        # Data Export and Mission Summary Endpoints
        @self.app.get("/api/export-logs")
        async def export_logs():
            """Export mission logs in JSON and CSV format"""
            try:
                export_data = await self.logging_service.export_mission_summary()
                return {
                    "status": "success",
                    "export_data": export_data,
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Error exporting logs: {e}")
                raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
        
        @self.app.get("/api/storage-stats")
        async def get_storage_stats():
            """Get storage statistics and usage"""
            try:
                stats = await self.storage_service.get_storage_stats()
                return {
                    "status": "success",
                    "storage_stats": stats,
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Error getting storage stats: {e}")
                raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")
        
        @self.app.post("/api/create-mission-archive")
        async def create_mission_archive(request: Dict[str, Any]):
            """Create a mission archive for data transfer"""
            try:
                archive_format = request.get("format", "zip")  # zip or directory
                include_video = request.get("include_video", True)
                
                archive_path = await self.storage_service.create_mission_archive(
                    archive_format=archive_format,
                    include_video=include_video
                )
                
                return {
                    "status": "success",
                    "archive_path": str(archive_path),
                    "format": archive_format,
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Error creating mission archive: {e}")
                raise HTTPException(status_code=500, detail=f"Archive creation failed: {str(e)}")
        
        # Suspect-Lock Mode Endpoints
        @self.app.post("/api/process-target-image")
        async def process_target_image(request: Dict[str, Any]):
            """Process target image for Suspect-Lock mode"""
            try:
                image_data = request.get("image_data")
                if not image_data:
                    raise HTTPException(status_code=400, detail="No image data provided")
                
                # Generate unique signature ID
                signature_id = f"target_{int(time.time())}"
                
                # Process image using identity service
                import base64
                image_bytes = base64.b64decode(image_data)
                result = self.identity_service.process_target_image(image_bytes, signature_id)
                
                if result["success"]:
                    self.current_target_signature = signature_id
                    logger.info(f"Target signature created: {signature_id}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing target image: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/start-tracking")
        async def start_tracking():
            """Start suspect tracking"""
            if not self.current_target_signature:
                raise HTTPException(status_code=400, detail="No target signature available")
            
            self.tracking_active = True
            self.suspect_matches = []
            
            logger.info(f"Tracking started for signature: {self.current_target_signature}")
            
            return {
                "status": "success",
                "message": "Tracking started",
                "signature_id": self.current_target_signature
            }
        
        @self.app.post("/api/stop-tracking")
        async def stop_tracking():
            """Stop suspect tracking"""
            self.tracking_active = False
            
            logger.info("Tracking stopped")
            
            return {
                "status": "success",
                "message": "Tracking stopped"
            }
        
        @self.app.post("/api/clear-target")
        async def clear_target():
            """Clear current target signature"""
            if self.current_target_signature:
                self.identity_service.remove_target_signature(self.current_target_signature)
                self.current_target_signature = None
            
            self.tracking_active = False
            self.suspect_matches = []
            
            logger.info("Target signature cleared")
            
            return {
                "status": "success",
                "message": "Target cleared"
            }
        
        @self.app.get("/api/tracking-status")
        async def get_tracking_status():
            """Get current tracking status"""
            return {
                "tracking_active": self.tracking_active,
                "target_signature": self.current_target_signature,
                "total_matches": len(self.suspect_matches),
                "recent_matches": self.suspect_matches[-5:] if self.suspect_matches else []
            }
    
    async def initialize_pipelines(self):
        """Initialize the detection and geolocation pipelines"""
        try:
            # Initialize telemetry service
            from .telemetry_service import initialize_telemetry
            self.telemetry_service = initialize_telemetry({
                'sim_rate': 1.0,
                'enable_dji': False,
                'enable_mavlink': False
            })
            
            # Initialize geolocation pipeline
            self.geolocation_pipeline = GeolocationPipeline(
                camera_model="O4_4K",
                telemetry_source="simulated",
                terrain_model="flat"
            )
            
            # Initialize detection pipeline
            self.detection_pipeline = DetectionPipeline(
                model_path="models/yolov8n.pt",
                human_only=True,
                aerial_optimized=True
            )
            
            # Connect pipelines
            self.geolocation_pipeline.set_detection_pipeline(self.detection_pipeline)
            
            logger.info("SAR pipelines initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipelines: {e}")
            raise
    
    async def start_telemetry_broadcast(self):
        """Start broadcasting telemetry data"""
        while True:
            try:
                if self.telemetry_service:
                    # Get latest telemetry
                    telemetry_data = self.telemetry_service.get_latest_telemetry()
                    
                    if telemetry_data:
                        # Convert to SAR format
                        sar_telemetry = SARTelemetry(
                            gps={
                                "latitude": telemetry_data.latitude,
                                "longitude": telemetry_data.longitude
                            },
                            altitude=telemetry_data.altitude_agl,
                            heading=telemetry_data.yaw,
                            speed=getattr(telemetry_data, 'ground_speed', 0.0),
                            battery=getattr(telemetry_data, 'battery_level', 100)
                        )
                        
                        # Log telemetry data periodically
                        self.logging_service.log_telemetry(
                            telemetry_data={
                                "gps": sar_telemetry.gps,
                                "altitude": sar_telemetry.altitude,
                                "heading": sar_telemetry.heading,
                                "speed": sar_telemetry.speed,
                                "battery": sar_telemetry.battery,
                                "mode": self.current_mode,
                                "tracking_active": self.tracking_active
                            },
                            metadata={
                                "session_id": getattr(self.logging_service, 'session_id', 'unknown'),
                                "system_status": "operational"
                            }
                        )
                        
                        # Broadcast to connected clients
                        await self.connection_manager.broadcast_telemetry(
                            asdict(sar_telemetry)
                        )
                
                await asyncio.sleep(0.1)  # 10 Hz update rate
                
            except Exception as e:
                logger.error(f"Error in telemetry broadcast: {e}")
                await asyncio.sleep(1)
    
    async def start_detection_processing(self):
        """Start processing detections and broadcasting results"""
        # This would typically process video frames in real-time
        # For now, we'll simulate with a placeholder
        
        while True:
            try:
                if self.geolocation_pipeline:
                    # In a real implementation, you'd get frames from your video source
                    # For now, we'll create a dummy frame
                    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    
                    # Add frame to video buffer for storage
                    self.video_frames_buffer.append(dummy_frame.copy())
                    if len(self.video_frames_buffer) > self.max_buffer_size:
                        self.video_frames_buffer.pop(0)
                    
                    # Process frame through geolocation pipeline
                    results = self.geolocation_pipeline.process_frame(dummy_frame)
                    
                    if results and results.detections:
                        # Convert to SAR format
                        sar_detections = []
                        detection_events = []  # For logging
                        
                        # Get current telemetry for logging
                        current_telemetry = None
                        if self.telemetry_service:
                            current_telemetry = self.telemetry_service.get_current_telemetry()
                        
                        for detection in results.detections:
                            # Create geolocation dict if detection has valid coordinates
                            geolocation = None
                            if detection.latitude is not None and detection.longitude is not None:
                                geolocation = {
                                    "latitude": detection.latitude,
                                    "longitude": detection.longitude,
                                    "elevation": detection.elevation
                                }
                            
                            sar_detection = SARDetection(
                                id=str(detection.detection_id),
                                bbox=detection.bbox,
                                confidence=detection.confidence,
                                geolocation=geolocation,
                                distance=detection.ground_distance_m,
                                class_name=detection.class_name
                            )
                            
                            # Prepare drone GPS for logging
                            drone_gps = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
                            if current_telemetry:
                                drone_gps = {
                                    "lat": current_telemetry.get("gps", {}).get("latitude", 0.0),
                                    "lon": current_telemetry.get("gps", {}).get("longitude", 0.0),
                                    "alt": current_telemetry.get("altitude", 0.0)
                                }
                            
                            # Log detection event
                            self.logging_service.log_detection(
                                drone_gps=drone_gps,
                                bounding_box={
                                    "x": detection.bbox[0],
                                    "y": detection.bbox[1],
                                    "width": detection.bbox[2] - detection.bbox[0],
                                    "height": detection.bbox[3] - detection.bbox[1]
                                },
                                detection_confidence=detection.confidence,
                                geo_coordinates=geolocation,
                                suspect_lock_status=False,
                                event_type="detection",
                                additional_data={
                                    "detection_id": detection.detection_id,
                                    "distance_m": detection.ground_distance_m,
                                    "class_name": detection.class_name
                                }
                            )
                            
                            # Check for suspect match if tracking is active
                            if self.tracking_active and self.current_target_signature:
                                try:
                                    # Extract region from frame for identity matching
                                    x1, y1, x2, y2 = detection.bbox
                                    person_crop = dummy_frame[int(y1):int(y2), int(x1):int(x2)]
                                    
                                    if person_crop.size > 0:
                                        # Match against target signature
                                        match_result = self.identity_service.match_detection(
                                            person_crop, self.current_target_signature
                                        )
                                        
                                        if match_result["is_match"]:
                                            # Mark as suspect
                                            sar_detection.class_name = "suspect"
                                            sar_detection.track_id = 999  # Special ID for suspect
                                            
                                            # Generate masked target ID for privacy
                                            target_id = f"TARGET_{hash(str(self.current_target_signature)) % 10000:04d}"
                                            
                                            # Log suspect match event
                                            self.logging_service.log_suspect_match(
                                                drone_gps=drone_gps,
                                                bounding_box={
                                                    "x": detection.bbox[0],
                                                    "y": detection.bbox[1],
                                                    "width": detection.bbox[2] - detection.bbox[0],
                                                    "height": detection.bbox[3] - detection.bbox[1]
                                                },
                                                detection_confidence=detection.confidence,
                                                match_confidence=match_result["confidence"],
                                                target_id=target_id,
                                                geo_coordinates=geolocation
                                            )
                                            
                                            # Record video snippet for suspect match
                                            if len(self.video_frames_buffer) > 0:
                                                snippet_events = [{
                                                    "type": "suspect_match",
                                                    "detection_id": detection.detection_id,
                                                    "confidence": match_result["confidence"],
                                                    "target_id": target_id,
                                                    "timestamp": time.time()
                                                }]
                                                
                                                self.storage_service.record_video_snippet(
                                                    frames=self.video_frames_buffer.copy(),
                                                    detection_events=snippet_events,
                                                    duration_seconds=len(self.video_frames_buffer) / 30.0,
                                                    metadata={
                                                        "event_type": "suspect_match",
                                                        "target_id": target_id,
                                                        "match_confidence": match_result["confidence"],
                                                        "geolocation": geolocation
                                                    }
                                                )
                                            
                                            # Store match info
                                            match_info = {
                                                "timestamp": time.time(),
                                                "detection_id": detection.detection_id,
                                                "confidence": match_result["confidence"],
                                                "geolocation": geolocation,
                                                "bbox": detection.bbox,
                                                "target_id": target_id
                                            }
                                            self.suspect_matches.append(match_info)
                                            
                                            logger.info(f"Suspect match found: ID {detection.detection_id}, confidence {match_result['confidence']:.3f}, target {target_id}")
                                            
                                except Exception as e:
                                    logger.error(f"Error in suspect matching: {e}")
                            
                            sar_detections.append(asdict(sar_detection))
                        
                        # Broadcast to connected clients
                        await self.connection_manager.broadcast_detections({
                            "detections": sar_detections,
                            "timestamp": time.time(),
                            "frame_id": results.frame_id
                        })
                
                await asyncio.sleep(0.1)  # 10 Hz processing rate
                
            except Exception as e:
                logger.error(f"Error in detection processing: {e}")
                await asyncio.sleep(1)
    
    async def start_background_tasks(self):
        """Start all background tasks"""
        await self.initialize_pipelines()
        
        # Start background tasks
        asyncio.create_task(self.start_telemetry_broadcast())
        asyncio.create_task(self.start_detection_processing())
    
    async def shutdown(self):
        """Shutdown the SAR service and cleanup resources"""
        logger.info("Shutting down SAR service...")
        
        try:
            # Log system shutdown event
            if hasattr(self, 'logging_service') and self.logging_service:
                telemetry = self.telemetry_service.get_current_telemetry() if self.telemetry_service else None
                self.logging_service.log_system_event(
                    event_type="system_shutdown",
                    description="SAR service shutting down",
                    metadata={
                        "total_detections": len(getattr(self, 'all_detections', [])),
                        "suspect_matches": len(getattr(self, 'suspect_matches', [])),
                        "confirmed_sightings": len(getattr(self, 'confirmed_sightings', [])),
                        "handoff_requests": len(getattr(self, 'handoff_requests', [])),
                        "session_duration": time.time() - getattr(self, 'start_time', time.time())
                    },
                    drone_gps=telemetry.gps_coordinates if telemetry else None
                )
                
                # Shutdown logging service
                await self.logging_service.shutdown()
                logger.info("Logging service shutdown complete")
            
            # Shutdown storage service
            if hasattr(self, 'storage_service') and self.storage_service:
                await self.storage_service.shutdown()
                logger.info("Storage service shutdown complete")
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("SAR service shutdown complete")
        
        logger.info("SAR service background tasks started")

# Create the SAR service instance
sar_service = SARService()
app = sar_service.app

@app.on_event("startup")
async def startup_event():
    """Initialize the SAR service on startup"""
    logger.info("Starting SAR service...")
    await sar_service.start_background_tasks()
    logger.info("SAR service started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")