# src/backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from typing import Optional
import asyncio

app = FastAPI(title="Android Screen Capture API")

# Adjust origins for your dev environment (vite / react dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRAME_PATH = os.path.join("out", "frame.jpg")
MAX_RETRIES = 8
RETRY_DELAY = 0.02  # 20ms between attempts

def read_frame_stable() -> Optional[bytes]:
    """
    Read the JPEG file with retries so we don't return a partially-written file.
    Returns JPEG bytes or None.
    """
    if not os.path.exists(FRAME_PATH):
        return None

    for attempt in range(MAX_RETRIES):
        try:
            with open(FRAME_PATH, "rb") as f:
                data = f.read()
            # very quick sanity check for JPEG markers
            if len(data) < 10:
                time.sleep(RETRY_DELAY)
                continue
            if not (data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"):
                # incomplete jpeg or being written
                time.sleep(RETRY_DELAY)
                continue
            return data
        except (IOError, OSError):
            time.sleep(RETRY_DELAY)
            continue
    return None

@app.get("/")
async def root():
    frame_exists = os.path.exists(FRAME_PATH)
    frame_size = os.path.getsize(FRAME_PATH) if frame_exists else 0
    frame_age = time.time() - os.path.getmtime(FRAME_PATH) if frame_exists else None
    return {
        "status": "running",
        "frame_exists": frame_exists,
        "frame_size_bytes": frame_size,
        "frame_age_seconds": round(frame_age, 2) if frame_age is not None else None,
        "endpoints": {
            "preview": "/preview (cached-busted JPEG)",
            "frame": "/frame.jpg (raw file)",
            "stream": "/stream.mjpg (MJPEG stream)",
            "health": "/health (detailed status)",
        },
    }

@app.get("/preview")
async def preview():
    """
    Lightweight endpoint for the frontend.
    Returns raw JPEG bytes with a strict no-cache header.
    """
    data = read_frame_stable()
    if data is None:
        raise HTTPException(status_code=503, detail="Frame not available")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/frame.jpg")
async def frame_raw():
    if not os.path.exists(FRAME_PATH):
        raise HTTPException(status_code=404, detail="Frame file not found")
    data = read_frame_stable()
    if data is None:
        raise HTTPException(status_code=503, detail="Frame file in use / corrupted")
    return Response(content=data, media_type="image/jpeg")

@app.get("/stream.mjpg")
async def stream_mjpeg():
    """
    Simple MJPEG stream. Each multipart chunk uses a safe boundary and content-length.
    Keep connections alive; clients can use <img src="/stream.mjpg">.
    """
    async def generator():
        boundary = "frame"
        while True:
            data = read_frame_stable()
            if data:
                header = (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(data)}\r\n\r\n"
                ).encode("utf-8")
                yield header
                yield data
                yield b"\r\n"
            else:
                # no frame: small sleep so the loop doesn't busy-wait
                await asyncio.sleep(0.05)
                continue
            # pacing (adjust if you want faster/slower)
            await asyncio.sleep(0.05)

    return StreamingResponse(
        generator(),
        media_type=f"multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Connection": "keep-alive"},
    )

@app.get("/health")
async def health():
    frame_exists = os.path.exists(FRAME_PATH)
    info = {"timestamp": time.time(), "frame_file": {"exists": frame_exists, "path": FRAME_PATH}}
    if frame_exists:
        try:
            st = os.stat(FRAME_PATH)
            data = read_frame_stable()
            valid_jpeg = bool(data and len(data) > 10 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9")
            info["frame_file"].update({
                "size_bytes": st.st_size,
                "size_kb": round(st.st_size / 1024, 1),
                "modified_time": st.st_mtime,
                "age_seconds": round(time.time() - st.st_mtime, 2),
                "readable": data is not None,
                "valid_jpeg": valid_jpeg,
            })
        except Exception as e:
            info["frame_file"]["error"] = str(e)

    if info["frame_file"].get("valid_jpeg", False):
        age = info["frame_file"]["age_seconds"]
        info["status"] = "healthy" if age < 5 else ("stale" if age < 15 else "very_stale")
    else:
        info["status"] = "unhealthy"
    return info

if __name__ == "__main__":
    import uvicorn
    os.makedirs("out", exist_ok=True)
    print("Starting backend on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
