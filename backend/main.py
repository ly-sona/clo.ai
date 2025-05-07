from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import schemas
import jobs
import optimizer
import utils
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(title="Chip-Layout API")
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────  POST /optimize  ────────────────────────────────
@app.post("/optimize", response_model=schemas.JobCreated)
async def optimise_chip(bg: BackgroundTasks,                        # ← don't give it a default
    file: UploadFile = File(...)):
    """
    • store uploaded schematic for reference
    • kick off GA optimisation in the background
    """
    # Use filename as job_id, but remove extension and sanitize
    job_id = Path(file.filename).stem.replace(" ", "_")
    schematic_path = utils.TMP / f"{job_id}_{file.filename}"
    schematic_path.write_bytes(await file.read())

    jobs.create(
        job_id,
        schematic=str(schematic_path),
        status="queued",
        layouts=[],
    )

    # Pass the SAME job_id into the optimiser
    bg.add_task(optimizer.optimise, job_id=job_id, power_threshold=0.5)
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
    layout_dir = utils.LAYOUTS / layout_id
    if not layout_dir.exists():
        raise HTTPException(status_code=404, detail="Layout not found")
        
    metadata_path = layout_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Layout metadata not found")
        
    with open(metadata_path) as f:
        layout_data = json.load(f)
        
    return schemas.LayoutDetail(
        **layout_data,
        wns=0.15,  # These are placeholder values
        cells=1234,
        fullPng=layout_data["thumb"]
    )

@app.get("/layout/{layout_id}/download")
async def download_def(layout_id: str):
    def_path = utils.LAYOUTS / layout_id / f"{layout_id}.def"
    if not def_path.exists():
        raise HTTPException(status_code=404, detail="DEF file not found")
        
    return FileResponse(
        def_path,
        media_type="application/octet-stream",
        filename=f"{layout_id}.def"
    )

@app.get("/layout/{layout_id}/bench")
async def download_optimized_bench(layout_id: str):
    bench_path = utils.LAYOUTS / layout_id / f"{layout_id}_optimized.bench"
    if not bench_path.exists():
        raise HTTPException(status_code=404, detail="Optimized .bench file not found")
    return FileResponse(bench_path, media_type="text/plain", filename=f"{layout_id}_optimized.bench")

@app.get("/layout/{layout_id}/lef")
async def get_lef(layout_id: str):
    """Get the LEF file content for a layout."""
    try:
        with open("lib/nangate45.lef", "r") as f:
            return Response(content=f.read(), media_type="text/plain")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="LEF file not found")

@app.get("/def/{layout_id}")
async def get_def(layout_id: str):
    try:
        def_path = utils.LAYOUTS / layout_id / f"{layout_id}.def"
        with open(def_path, "r") as f:
            return Response(content=f.read(), media_type="text/plain")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="DEF file not found")