# src/backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import time
import asyncio

# --- config ---
FRAME_PATH = "out/frame.jpg"
MAX_RETRIES = 5
RETRY_DELAY = 0.01  # 10ms between retries

app = FastAPI(title="Android Screen Capture API")

# Allow your Vite dev server (use both localhost and 127.0.0.1 just in case)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("out", exist_ok=True)

def read_frame_stable() -> Optional[bytes]:
    """Read out/frame.jpg safely while FFmpeg writes it."""
    if not os.path.exists(FRAME_PATH):
        return None

    for _ in range(MAX_RETRIES):
        try:
            with open(FRAME_PATH, "rb") as f:
                data = f.read()

            # Require a complete JPEG (SOI/EOI markers)
            if len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9":
                return data

        except (OSError, IOError):
            pass

        time.sleep(RETRY_DELAY)

    return None

@app.get("/")
async def root():
    exists = os.path.exists(FRAME_PATH)
    size = os.path.getsize(FRAME_PATH) if exists else 0
    age = (time.time() - os.path.getmtime(FRAME_PATH)) if exists else None
    return {
        "status": "running",
        "frame_exists": exists,
        "frame_size_bytes": size,
        "frame_age_seconds": round(age, 2) if age is not None else None,
        "endpoints": {
            "preview": "/preview",
            "frame": "/frame.jpg",
            "stream": "/stream.mjpg",
            "health": "/health",
        },
    }

@app.get("/preview")
async def get_preview():
    data = read_frame_stable()
    if data is None:
        raise HTTPException(status_code=503, detail="Frame not available - check capture pipeline")
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
async def get_frame_raw():
    if not os.path.exists(FRAME_PATH):
        raise HTTPException(status_code=404, detail="Frame file not found")
    data = read_frame_stable()
    if data is None:
        raise HTTPException(status_code=503, detail="Frame file corrupted or in use")
    return Response(content=data, media_type="image/jpeg")

@app.get("/stream.mjpg")
async def mjpeg_stream():
    boundary = "frame"

    async def gen():
        while True:
            data = read_frame_stable()
            if data is not None:
                head = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(data)}\r\n\r\n"
                ).encode("utf-8")
                yield head
                yield data
                yield b"\r\n"
                await asyncio.sleep(0.1)  # ~10 fps cap
            else:
                await asyncio.sleep(0.1)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Connection": "keep-alive"},
    )

@app.get("/health")
async def health():
    exists = os.path.exists(FRAME_PATH)
    info = {"timestamp": time.time(), "frame_file": {"exists": exists, "path": FRAME_PATH}}
    if exists:
        try:
            st = os.stat(FRAME_PATH)
            data = read_frame_stable()
            valid = data is not None and len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
            age = time.time() - st.st_mtime
            info["frame_file"].update(
                {
                    "size_bytes": st.st_size,
                    "size_kb": round(st.st_size / 1024, 1),
                    "modified_time": st.st_mtime,
                    "age_seconds": round(age, 2),
                    "readable": data is not None,
                    "valid_jpeg": valid,
                }
            )
            info["status"] = "healthy" if valid and age < 5 else ("stale" if valid and age < 15 else "very_stale")
        except Exception as e:
            info["frame_file"]["error"] = str(e)
            info["status"] = "unhealthy"
    else:
        info["status"] = "unhealthy"
    return info

if __name__ == "__main__":
    import uvicorn
    print("Starting Android Screen Capture API…")
    uvicorn.run(app, host="127.0.0.1", port=8000)
