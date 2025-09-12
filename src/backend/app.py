from pathlib import Path
import asyncio
import cv2
import numpy as np
import base64
import time
import logging
from fastapi import FastAPI, WebSocket, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
import sys
import os

# Add the src directory to the path so we can import modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from connection.capture_screen import ScreenCapture

# Import our new FPV services
try:
    from services.fpv_capture import FPVCaptureService, create_fpv_capture, list_available_sources
    from services.yolo_wrapper import YOLOWrapper, create_yolo_wrapper
    FPV_SERVICES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"FPV services not available: {e}")
    FPV_SERVICES_AVAILABLE = False

# Import RTMP API router
try:
    from .rtmp_api import router as rtmp_router
    RTMP_API_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RTMP API not available: {e}")
    RTMP_API_AVAILABLE = False

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PUBLIC_DIR = BASE_DIR / "public"

app = FastAPI()

# Include RTMP API router
if RTMP_API_AVAILABLE:
    app.include_router(rtmp_router)
    logging.info("RTMP API endpoints registered")

# Global variables for video capture
capture = None
is_capturing = False
last_frame = None
sar_mode_enabled = False

# FPV capture system
fpv_capture_service = None
yolo_detector = None
use_fpv_capture = False
detection_enabled = True
fpv_stats = {
    'fps': 0.0,
    'detections': 0,
    'people_detected': 0,
    'source_type': 'screen_capture'
}

# Mount /public so the webapp is served
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# Serve the default page
@app.get("/")
async def root():
    return FileResponse(PUBLIC_DIR / "foresight-webapp.html")

# Serve the FPV live page
@app.get("/fpv")
async def fpv_live_page():
    return FileResponse("src/frontend/fpv_live.html")

# Stream video frames
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Toggle SAR mode
@app.get("/toggle_sar")
async def toggle_sar():
    global sar_mode_enabled
    sar_mode_enabled = not sar_mode_enabled
    return {"sar_mode": sar_mode_enabled}

# Get SAR mode status
@app.get("/sar_status")
async def sar_status():
    return {"sar_mode": sar_mode_enabled}

# FPV Capture endpoints
@app.get("/api/fpv/sources")
async def get_fpv_sources():
    """Get available FPV video sources"""
    if not FPV_SERVICES_AVAILABLE:
        return {"error": "FPV services not available"}
    try:
        sources = list_available_sources()
        return {"sources": sources}
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        return {"error": str(e)}

@app.post("/api/fpv/start")
async def start_fpv_capture():
    """Start FPV capture system"""
    global fpv_capture_service, yolo_detector, use_fpv_capture
    
    if not FPV_SERVICES_AVAILABLE:
        return {"success": False, "error": "FPV services not available"}
    
    try:
        # Initialize FPV capture if not already done
        if not fpv_capture_service:
            fpv_capture_service = create_fpv_capture()
        
        # Initialize YOLO detector if not already done
        if not yolo_detector:
            model_path = os.environ.get('YOLO_MODEL_PATH', 'yolo11n.pt')
            try:
                yolo_detector = create_yolo_wrapper(model_path, 'yolo11')
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}")
        
        # Start FPV capture
        if fpv_capture_service.start():
            use_fpv_capture = True
            fpv_stats['source_type'] = 'fpv_capture'
            return {"success": True, "message": "FPV capture started"}
        else:
            return {"success": False, "error": "Failed to start FPV capture"}
            
    except Exception as e:
        logger.error(f"Error starting FPV capture: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/fpv/stop")
async def stop_fpv_capture():
    """Stop FPV capture system"""
    global fpv_capture_service, use_fpv_capture
    
    try:
        if fpv_capture_service:
            fpv_capture_service.stop()
        use_fpv_capture = False
        fpv_stats['source_type'] = 'screen_capture'
        return {"success": True, "message": "FPV capture stopped"}
    except Exception as e:
        logger.error(f"Error stopping FPV capture: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/fpv/toggle_detection")
async def toggle_fpv_detection():
    """Toggle YOLO detection on/off"""
    global detection_enabled
    detection_enabled = not detection_enabled
    return {"success": True, "detection_enabled": detection_enabled}

@app.get("/api/fpv/stats")
async def get_fpv_stats():
    """Get FPV capture and detection statistics"""
    stats = fpv_stats.copy()
    
    if fpv_capture_service:
        capture_stats = fpv_capture_service.get_stats()
        stats.update({
            'capture_fps': capture_stats.get('avg_fps', 0),
            'frames_captured': capture_stats.get('frames_captured', 0),
            'source_connected': capture_stats.get('source_connected', False)
        })
    
    if yolo_detector:
        yolo_stats = yolo_detector.get_stats()
        stats.update({
            'inference_time_ms': yolo_stats.get('avg_inference_time_ms', 0),
            'total_inferences': yolo_stats.get('total_inferences', 0),
            'model_loaded': yolo_stats.get('model_loaded', False)
        })
    
    stats.update({
        'use_fpv_capture': use_fpv_capture,
        'detection_enabled': detection_enabled,
        'sar_mode': sar_mode_enabled
    })
    
    return stats

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Send stats and frame data
            await ws.send_json({
                "type": "stats", 
                "fps": 30, 
                "latency": 120, 
                "geo_error": 2.5, 
                "detections": [],
                "sar_mode": sar_mode_enabled
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("Client disconnected")

# Start video capture
def start_capture():
    global capture, is_capturing
    if not is_capturing:
        try:
            capture = ScreenCapture(title="FORESIGHT_FEED", target_fps=15)
            is_capturing = True
            print("Screen capture started")
        except Exception as e:
            print(f"Error starting capture: {e}")
            is_capturing = False

# Generate video frames
async def generate_frames():
    global capture, is_capturing, last_frame, fpv_capture_service, yolo_detector, use_fpv_capture, detection_enabled, fpv_stats
    
    # Start capture if not already running
    if not is_capturing and not use_fpv_capture:
        start_capture()
    
    while True:
        frame = None
        
        try:
            # Get frame from FPV capture or screen capture
            if use_fpv_capture and fpv_capture_service:
                frame = fpv_capture_service.get_frame()
                if frame is not None:
                    fpv_stats['fps'] = fpv_capture_service.get_stats().get('avg_fps', 0)
            elif is_capturing and capture:
                ret, frame = capture.read()
                if not ret:
                    frame = None
            
            if frame is not None:
                # Apply YOLO detection if enabled
                detections = []
                if (sar_mode_enabled or detection_enabled) and yolo_detector and yolo_detector.is_model_loaded():
                    try:
                        detections = yolo_detector.detect(frame)
                        
                        # Update statistics
                        fpv_stats['detections'] += len(detections)
                        people_count = sum(1 for d in detections if 'person' in d.class_name.lower())
                        fpv_stats['people_detected'] += people_count
                        
                        # Draw detections on frame
                        frame = yolo_detector.draw_detections(frame, detections)
                        
                    except Exception as e:
                        logger.error(f"Error in YOLO detection: {e}")
                
                # Add status overlay
                frame = add_status_overlay(frame, detections)
                
                # Store the last frame
                last_frame = frame.copy()
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    
                    # Yield the frame in the format expected by StreamingResponse
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    await asyncio.sleep(0.1)
            else:
                # No frame available, show placeholder
                placeholder_frame = create_placeholder_frame()
                ret, buffer = cv2.imencode('.jpg', placeholder_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in frame generation: {e}")
            await asyncio.sleep(0.5)
            
            # Try to restart capture if needed
            if not use_fpv_capture and not is_capturing:
                start_capture()

def add_status_overlay(frame, detections):
    """Add status information overlay to frame"""
    overlay_frame = frame.copy()
    
    # Add mode indicator
    mode_text = "FPV" if use_fpv_capture else "SCREEN"
    if sar_mode_enabled or detection_enabled:
        mode_text += " + SAR"
    
    cv2.putText(overlay_frame, mode_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add detection count
    if detections:
        det_text = f"Detections: {len(detections)}"
        people_count = sum(1 for d in detections if 'person' in d.class_name.lower())
        if people_count > 0:
            det_text += f" | People: {people_count}"
        
        cv2.putText(overlay_frame, det_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Add FPS if available
    if fpv_stats['fps'] > 0:
        fps_text = f"FPS: {fpv_stats['fps']:.1f}"
        cv2.putText(overlay_frame, fps_text, (10, overlay_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return overlay_frame

def create_placeholder_frame():
    """Create a placeholder frame when no video is available"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add text
    text_lines = [
        "No Video Signal",
        "Check camera connection",
        "or start FPV capture"
    ]
    
    y_start = 200
    for i, line in enumerate(text_lines):
        y_pos = y_start + (i * 40)
        cv2.putText(frame, line, (120, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

# Start capture on app startup
@app.on_event("startup")
async def startup_event():
    start_capture()

# Release resources on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    global capture, is_capturing
    if capture:
        capture.release()
    is_capturing = False
