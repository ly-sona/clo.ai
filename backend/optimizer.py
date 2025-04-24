import numpy as np, asyncio, json
from .utils import new_id, save_thumb
from .jobs import update
from deap_model import run_ga, GRID_WIDTH, GRID_HEIGHT

async def optimise(n_layouts: int = 3, power_threshold: float = .5):
    """
    Fire-and-forget async task:
      • runs the GA
      • stores DEF + thumbnail
      • pushes results into the job record
    """
    job_id = new_id()
    update(job_id, status="running")
    pop, _, hof, _ = run_ga(threshold=power_threshold)
    best_ind = hof[0]
    layout_xy = np.array(best_ind).reshape(-1,2)

    # write DEF + thumb
    from def_writer import write_def
    def_path = (TMP := __import__("pathlib").Path("tmp")) / f"{job_id}.def"
    write_def(layout_xy, GRID_WIDTH, GRID_HEIGHT, outfile=def_path)

    thumb = save_thumb(layout_xy, job_id)
    update(job_id,
           status="done",
           layouts=[{"id": job_id, "thumb": thumb, "power": hof[0].fitness.values[0],
                     "def_path": str(def_path)}])