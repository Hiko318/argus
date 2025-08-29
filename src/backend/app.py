from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PUBLIC_DIR = BASE_DIR / "public"

app = FastAPI()

# Mount /public so the webapp is served
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# Serve the default page
@app.get("/")
async def root():
    return FileResponse(PUBLIC_DIR / "foresight-webapp.html")

# Example WebSocket (stub for now)
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "stats", "fps": 30, "latency": 120, "geo_error": 2.5, "detections": []})
