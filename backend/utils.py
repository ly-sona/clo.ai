from pathlib import Path
import uuid, matplotlib.pyplot as plt, numpy as np

ROOT   = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
TMP    = ROOT / "tmp"
STATIC.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)

def new_id() -> str:
    return uuid.uuid4().hex[:12]

def save_thumb(layout_xy: np.ndarray, job_id: str) -> str:
    """Scatter-plot → PNG, return URL path `/static/…png`."""
    fig, ax = plt.subplots()
    ax.scatter(layout_xy[:,0], layout_xy[:,1])
    ax.set_aspect("equal"); ax.axis("off")
    p = STATIC / f"{job_id}.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return f"/static/{p.name}"