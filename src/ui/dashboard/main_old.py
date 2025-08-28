from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import mss
import pytesseract
import loguru

app = FastAPI()
logger = loguru.logger

# Shared state (so frontend can pull updated info)
state = {
    "heading": 0,
    "home": {"lat": 6.1164, "lon": 125.1716}  # Default: General Santos City
}

# ✅ Root check
@app.get("/")
def root():
    return {"status": "server is running", "message": "Welcome to foresight!"}


# ✅ Screenshot OCR
@app.get("/capture")
def capture_screen():
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = np.array(sct.grab(monitor))
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
        text = pytesseract.image_to_string(gray)
        return {"captured_text": text.strip()}
    except Exception as e:
        logger.error(f"Capture error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ✅ Camera OCR
@app.get("/camera")
def capture_camera():
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return JSONResponse(status_code=500, content={"error": "Camera not available"})
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return {"camera_text": text.strip()}
    except Exception as e:
        logger.error(f"Camera error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ✅ Geolocation (pulled from state)
@app.get("/geolocate")
def geolocate_example():
    return {
        "lat": state["home"]["lat"],
        "lon": state["home"]["lon"],
        "confidence": "ok"
    }


# ✅ Set heading
@app.post("/heading")
async def set_heading(req: Request):
    try:
        data = await req.json()
        state["heading"] = data.get("heading", 0)
        logger.info(f"Heading updated to {state['heading']}")
        return {"ok": True, "heading": state["heading"]}
    except Exception as e:
        logger.error(f"Heading error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})


# ✅ Set home location
@app.post("/home")
async def set_home(req: Request):
    try:
        data = await req.json()
        state["home"] = {"lat": data.get("lat", 0), "lon": data.get("lon", 0)}
        logger.info(f"Home updated to {state['home']}")
        return {"ok": True, "home": state["home"]}
    except Exception as e:
        logger.error(f"Home error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})


# ✅ MJPEG video streaming (webcam feed)
@app.get("/mjpg")
def mjpg_stream():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, jpeg = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
from fastapi.responses import StreamingResponse

@app.get("/mjpg")
def mjpg_stream():
    cap = cv2.VideoCapture(0)  # try 1 if 0 doesn't work
    if not cap.isOpened():
        logger.error("❌ Could not open webcam. Try changing index (0 → 1 or 2).")
        return JSONResponse(status_code=500, content={"error": "Webcam not available"})

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("❌ Failed to grab frame from webcam.")
                break
            _, jpeg = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    logger.info("✅ MJPEG stream started on /mjpg")
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
