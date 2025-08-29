# Keep your original app.py untouched. This file wires in the SAR router.
from .app import app  # your existing FastAPI app
from .routers import sar  # registers routes on import (side-effect)

__all__ = ["app"]
from fastapi.responses import StreamingResponse
import cv2

# OpenCV video capture (you can change to your camera index or stream source)
cap = cv2.VideoCapture(0)  # 0 = default webcam

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.get("/frame.jpg")
async def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
