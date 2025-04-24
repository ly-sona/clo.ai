# simulator.py (openroad is not implemented just yet)
import asyncio, random, time, json, shutil, subprocess, os, gzip
from pathlib import Path
from fastapi import WebSocket
from .jobs import get, update
from .utils import STATIC

async def simulate(job_id: str, ws: WebSocket):
    """
    Streams JSON lines:
      { "type":"progress", "value": nn }
      { "type":"log",      "text": "..." }
      { "type":"finished", "gifUrl": "/static/…" }
    """
    job = get(job_id)
    if not job:
        await ws.send_json({"type":"error", "msg":"job not found"}); return

    # Fake progress (replace w/ OpenROAD/KLayout CLI calls)
    await ws.accept()
    for pct in range(0,101,5):
        await asyncio.sleep(.3)
        await ws.send_json({"type":"progress", "value": pct})
        await ws.send_json({"type":"log", "text": f"Timing step {pct}% done"})
    # generate stub GIF
    gif = STATIC / f"{job_id}.gif"
    Path(gif).write_bytes(b"GIF89a")     # placeholder
    await ws.send_json({"type":"finished", "gifUrl": f"/static/{gif.name}"})
    update(job_id, routed=str(gif))