from pathlib import Path
import uuid, matplotlib.pyplot as plt, numpy as np
import os
import json
import time

ROOT   = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
TMP    = ROOT / "tmp"
LAYOUTS = ROOT / "layouts"

# Create necessary directories
for dir_path in [STATIC, TMP, LAYOUTS]:
    dir_path.mkdir(exist_ok=True)
    (dir_path / "thumbnails").mkdir(exist_ok=True)
    (dir_path / "optimization_logs").mkdir(exist_ok=True)

def new_id() -> str:
    return uuid.uuid4().hex[:12]

def save_thumb(layout_xy: np.ndarray, job_id: str) -> str:
    """Save a thumbnail visualization of the layout."""
    plt.figure(figsize=(6, 6))
    plt.scatter(layout_xy[:, 0], layout_xy[:, 1], c='blue', alpha=0.6)
    plt.grid(True)
    plt.title(f"Layout {job_id[:8]}")
    
    # Save thumbnail
    thumb_path = STATIC / "thumbnails" / f"{job_id}_thumb.png"
    plt.savefig(thumb_path)
    plt.close()
    
    return f"/static/thumbnails/{job_id}_thumb.png"

def save_layout_metadata(job_id: str, layout_data: dict):
    """Save layout metadata including power metrics."""
    metadata_path = LAYOUTS / job_id / "metadata.json"
    metadata_path.parent.mkdir(exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(layout_data, f, indent=2)

def save_optimization_log(job_id: str, generation: int, stats: dict):
    """Save optimization progress logs."""
    log_path = STATIC / "optimization_logs" / f"{job_id}_gen_{generation:03d}.json"
    
    with open(log_path, 'w') as f:
        json.dump(stats, f, indent=2)

def cleanup_old_files(max_age_days: int = 7):
    """Clean up old temporary files."""
    current_time = time.time()
    
    for dir_path in [TMP, STATIC / "thumbnails", STATIC / "optimization_logs"]:
        if not dir_path.exists():
            continue
            
        for file_path in dir_path.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_days * 86400:  # 86400 seconds = 1 day
                    file_path.unlink()