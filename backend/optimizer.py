import numpy as np, asyncio, json
from .utils import new_id, save_thumb, save_layout_metadata, save_optimization_log
from .jobs import update
from models.deap_model import run_ga, GRID_WIDTH, GRID_HEIGHT
from models.def_writer import write_def

async def optimise(job_id: str, power_threshold: float = 0.5, n_layouts: int = 3):
    update(job_id, status="running")
    
    # Run genetic algorithm optimization
    pop, logbook, hof, fitness_history = run_ga(threshold=power_threshold)
    best = hof[0]
    layout_xy = np.array(best).reshape(-1, 2)

    # Save DEF file
    def_path = utils.TMP / f"{job_id}.def"
    write_def(layout_xy, GRID_WIDTH, GRID_HEIGHT, outfile=def_path)

    # Save thumbnail
    thumb_url = save_thumb(layout_xy, job_id)

    # Save layout metadata
    layout_data = {
        "id": job_id,
        "power": float(best.fitness.values[0]),
        "grid_size": {"width": GRID_WIDTH, "height": GRID_HEIGHT},
        "coordinates": layout_xy.tolist(),
        "def_path": str(def_path),
        "thumbnail": thumb_url
    }
    save_layout_metadata(job_id, layout_data)

    # Save optimization progress
    for gen, stats in enumerate(logbook):
        save_optimization_log(job_id, gen, stats)

    # Update job status
    update(
        job_id,
        status="done",
        layouts=[{
            "id": job_id,
            "thumb": thumb_url,
            "power": float(best.fitness.values[0]),
            "def_path": str(def_path),
        }],
    )
