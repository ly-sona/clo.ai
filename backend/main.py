from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from . import schemas, jobs, optimizer, simulator, utils

app = FastAPI(title="Chip-Layout API")
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"), name="static")

# ────────────────────────────────  POST /optimize  ────────────────────────────────
@app.post("/optimize", response_model=schemas.JobCreated)
async def optimise_chip(file: UploadFile = File(...), bg: BackgroundTasks = None):
    """
    • store uploaded schematic for reference
    • kick off GA optimisation in the background
    """
    job_id = utils.new_id()
    schematic_path = (utils.TMP / f"{job_id}_{file.filename}")
    schematic_path.write_bytes(await file.read())
    jobs.create(job_id, schematic=str(schematic_path), status="queued", layouts=[])
    bg.add_task(optimizer.optimise, power_threshold=.5)
    return {"job_id": job_id}

# ────────────────────────────────  GET /layouts  ────────────────────────────────
@app.get("/layouts", response_model=list[schemas.LayoutMeta])
async def list_layouts():
    layouts = []
    for j in jobs.list_all():
        for l in j["layouts"]:
            layouts.append(schemas.LayoutMeta(**l))
    return layouts

# ────────────────────────  GET /layout/{id} (+ download)  ────────────────────────
@app.get("/layout/{layout_id}", response_model=schemas.LayoutDetail)
async def layout_detail(layout_id: str):
    for j in jobs.list_all():
        for l in j["layouts"]:
            if l["id"] == layout_id:
                return schemas.LayoutDetail(**l, wns=0.15, cells=1234, fullPng=l["thumb"])
    return {"detail": "not found"}

@app.get("/layout/{layout_id}/download")
async def download_def(layout_id: str):
    for j in jobs.list_all():
        for l in j["layouts"]:
            if l["id"] == layout_id:
                return FileResponse(l["def_path"], media_type="application/octet-stream",
                                    filename=f"{layout_id}.def")
    return {"detail": "not found"}

# ─────────────────────────  WebSocket /ws/simulate/{id}  ─────────────────────────
@app.websocket("/ws/simulate/{layout_id}")
async def ws_simulate(ws: WebSocket, layout_id: str):
    await simulator.simulate(layout_id, ws)