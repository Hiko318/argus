#!/usr/bin/env python3
"""
Human Detection Service

FastAPI-based service for real-time human detection and tracking.
Provides endpoints for processing video frames, managing detection pipelines,
and streaming detection results.

Author: Foresight AI Team
Date: 2024
"""

import asyncio
import base64
import json
import logging
import time
from io import BytesIO
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from src.backend.detection_pipeline import DetectionPipeline, HumanDetection, DetectionFrame
from src.backend.detector import YOLODetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Human Detection Service",
    description="Real-time human detection and tracking API",
    version="1.0.0"
)

# Global pipeline instance
pipeline: Optional[DetectionPipeline] = None

# WebSocket connections for live streaming
active_connections: List[WebSocket] = []

# Request/Response Models
class DetectionConfig(BaseModel):
    """Configuration for detection pipeline"""
    model_path: str = "models/yolov8s.pt"
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    human_only: bool = True
    aerial_optimized: bool = False
    enable_tensorrt: bool = False
    max_disappeared: int = Field(30, ge=1, le=100)
    max_distance: float = Field(50.0, ge=1.0, le=200.0)
    min_hits: int = Field(3, ge=1, le=10)

class FrameRequest(BaseModel):
    """Request model for frame processing"""
    image_data: str  # Base64 encoded image
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None
    draw_annotations: bool = True

class OptimizationRequest(BaseModel):
    """Request model for performance optimization"""
    target_fps: float = Field(30.0, ge=1.0, le=60.0)

# Utility Functions
def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode numpy array image to base64"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Encode to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return ""

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global pipeline
    return {
        "status": "healthy",
        "service": "human-detection",
        "pipeline_loaded": pipeline is not None,
        "timestamp": time.time()
    }

@app.post("/initialize")
async def initialize_pipeline(config: DetectionConfig):
    """Initialize detection pipeline with configuration"""
    global pipeline
    
    try:
        logger.info(f"Initializing detection pipeline with config: {config.dict()}")
        
        pipeline = DetectionPipeline(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            human_only=config.human_only,
            aerial_optimized=config.aerial_optimized,
            enable_tensorrt=config.enable_tensorrt,
            max_disappeared=config.max_disappeared,
            max_distance=config.max_distance,
            min_hits=config.min_hits
        )
        
        return {
            "status": "success",
            "message": "Detection pipeline initialized successfully",
            "config": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {str(e)}")

@app.post("/detect")
async def detect_humans(request: FrameRequest):
    """Detect humans in a single frame"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized. Call /initialize first.")
    
    try:
        # Decode image
        frame = decode_base64_image(request.image_data)
        
        # Process frame
        detection_frame = pipeline.process_frame(
            frame, 
            frame_id=request.frame_id, 
            timestamp=request.timestamp
        )
        
        # Prepare response
        response = {
            "detection_frame": detection_frame.to_dict(),
            "performance": {
                "processing_time_ms": detection_frame.processing_time_ms,
                "fps": 1000 / detection_frame.processing_time_ms if detection_frame.processing_time_ms > 0 else 0
            }
        }
        
        # Add annotated image if requested
        if request.draw_annotations:
            annotated_frame = pipeline.draw_annotations(frame, detection_frame)
            response["annotated_image"] = encode_image_to_base64(annotated_frame)
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "detection_result",
            "data": response
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/stats")
async def get_performance_stats():
    """Get pipeline performance statistics"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.get_performance_stats()
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.post("/optimize")
async def optimize_performance(request: OptimizationRequest):
    """Optimize pipeline for target performance"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        optimization_result = pipeline.optimize_for_realtime(request.target_fps)
        return {
            "status": "success",
            "optimization_result": optimization_result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/reset")
async def reset_pipeline():
    """Reset pipeline state"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        pipeline.reset()
        return {
            "status": "success",
            "message": "Pipeline reset successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/upload")
async def upload_and_detect(file: UploadFile = File(...)):
    """Upload image file and detect humans"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process frame
        detection_frame = pipeline.process_frame(frame)
        
        # Draw annotations
        annotated_frame = pipeline.draw_annotations(frame, detection_frame)
        
        return {
            "detection_frame": detection_frame.to_dict(),
            "annotated_image": encode_image_to_base64(annotated_frame),
            "performance": {
                "processing_time_ms": detection_frame.processing_time_ms,
                "fps": 1000 / detection_frame.processing_time_ms if detection_frame.processing_time_ms > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Upload detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload detection failed: {str(e)}")

# WebSocket Endpoints
@app.websocket("/ws/live")
async def websocket_live_detection(websocket: WebSocket):
    """WebSocket endpoint for live detection streaming"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for frame data from client
            data = await websocket.receive_json()
            
            if data.get("type") == "frame" and pipeline is not None:
                try:
                    # Process frame
                    frame = decode_base64_image(data["image_data"])
                    detection_frame = pipeline.process_frame(frame)
                    
                    # Send result back
                    response = {
                        "type": "detection_result",
                        "data": {
                            "detection_frame": detection_frame.to_dict(),
                            "performance": {
                                "processing_time_ms": detection_frame.processing_time_ms,
                                "fps": 1000 / detection_frame.processing_time_ms if detection_frame.processing_time_ms > 0 else 0
                            }
                        }
                    }
                    
                    await websocket.send_json(response)
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Detection failed: {str(e)}"
                    })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Human Detection Service starting up...")
    
    # Initialize with default configuration
    default_config = DetectionConfig()
    try:
        await initialize_pipeline(default_config)
        logger.info("Default pipeline initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize default pipeline: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Human Detection Service shutting down...")
    global pipeline
    
    # Close all WebSocket connections
    for connection in manager.active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    
    # Reset pipeline
    if pipeline:
        pipeline.reset()
        pipeline = None

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Human Detection Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Human Detection Service on {args.host}:{args.port}")
    
    uvicorn.run(
        "detection_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )