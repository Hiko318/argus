from pathlib import Path
import asyncio
import cv2
import numpy as np
import base64
import time
from fastapi import FastAPI, WebSocket, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
import sys

# Add the src directory to the path so we can import modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.ingest.capture_screen import ScreenCapture

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PUBLIC_DIR = BASE_DIR / "public"

app = FastAPI()

# Global variables for video capture
capture = None
is_capturing = False
last_frame = None
sar_mode_enabled = False

# Mount /public so the webapp is served
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# Serve the default page
@app.get("/")
async def root():
    return FileResponse(PUBLIC_DIR / "foresight-webapp.html")

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
    global capture, is_capturing, last_frame
    
    # Start capture if not already running
    if not is_capturing:
        start_capture()
    
    while True:
        if is_capturing and capture:
            try:
                ret, frame = capture.read()
                if ret:
                    # Apply YOLO detection if SAR mode is enabled
                    if sar_mode_enabled:
                        # Placeholder for YOLO detection
                        # This will be implemented in the next step
                        cv2.putText(frame, "SAR MODE ACTIVE", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Store the last frame
                    last_frame = frame.copy()
                    
                    # Convert to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Yield the frame in the format expected by StreamingResponse
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # If no frame, wait a bit and try again
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in frame generation: {e}")
                await asyncio.sleep(0.5)
        else:
            # If not capturing, wait and try to start capture
            await asyncio.sleep(0.5)
            if not is_capturing:
                start_capture()

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
