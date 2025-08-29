from __future__ import annotations
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import StreamingResponse, JSONResponse
from ..services.pipeline import SarPipeline
from ..app import app  # reuse your existing FastAPI app, do not replace it

router = APIRouter(prefix="")

# single global pipeline instance
PIPE = SarPipeline()

@router.on_event("startup")
async def _startup():
    # start stopped: let UI call /api/pipeline/start
    pass

@router.post("/api/pipeline/start")
async def start_pipeline():
    PIPE.start()
    return {"ok": True, "running": True}

@router.post("/api/pipeline/stop")
async def stop_pipeline():
    PIPE.stop()
    return {"ok": True, "running": False}

@router.get("/api/state")
async def get_state():
    return JSONResponse(PIPE.stats())

@router.post("/api/mode")
async def set_mode(payload: dict):
    PIPE.set_mode(payload.get("mode", "sar"))
    return PIPE.stats()

@router.post("/api/blur")
async def set_blur(payload: dict):
    PIPE.set_blur(bool(payload.get("enabled", True)))
    return PIPE.stats()

@router.get("/frame.jpg")
async def frame_jpg():
    jpeg = PIPE.snapshot_jpeg()
    if not jpeg:
        # tiny transparent 1x1 if nothing yet
        return Response(b"\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02L\x01\x00;", media_type="image/gif")
    return StreamingResponse(iter([jpeg]), media_type="image/jpeg")

@router.websocket("/ws/sar")
async def ws_sar(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json({"type": "stats", **PIPE.stats()})
            await asyncio.sleep(0.2)  # ~5 Hz
    except WebSocketDisconnect:
        pass

# Register on the existing app
app.include_router(router)
