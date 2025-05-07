from pathlib import Path
import uuid, matplotlib.pyplot as plt, numpy as np
import os
import json
import time
import shutil

ROOT   = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
TMP    = ROOT / "tmp"
LAYOUTS = ROOT / "layouts"

# Create necessary directories
for dir_path in [STATIC, TMP, LAYOUTS]:
    dir_path.mkdir(exist_ok=True)
    (dir_path / "thumbnails").mkdir(exist_ok=True)
    (dir_path / "optimization_logs").mkdir(exist_ok=True)

# Add LEF file handling
LEF_PATH = Path(__file__).parent / "lib" / "nangate45.lef"

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
    layout_dir = LAYOUTS / job_id
    layout_dir.mkdir(exist_ok=True)
    
    # Save metadata
    metadata_path = layout_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(layout_data, f, indent=2)
    
    # Save DEF file if it exists
    if "def_path" in layout_data:
        def_path = Path(layout_data["def_path"])
        if def_path.exists():
            shutil.copy2(def_path, layout_dir / f"{job_id}.def")
            layout_data["def_path"] = str(layout_dir / f"{job_id}.def")

def save_optimization_log(job_id: str, generation: int, stats: dict):
    """Save optimization progress logs."""
    log_path = STATIC / "optimization_logs" / f"{job_id}_gen_{generation:03d}.json"
    
    with open(log_path, 'w') as f:
        json.dump(stats, f, indent=2)

def cleanup_old_files(max_age_days: int = 7):
    """Clean up old temporary files."""
    current_time = time.time()
    
    for dir_path in [TMP, STATIC / "thumbnails", STATIC / "optimization_logs", LAYOUTS]:
        if not dir_path.exists():
            continue
            
        for file_path in dir_path.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_days * 86400:  # 86400 seconds = 1 day
                    file_path.unlink()
            elif file_path.is_dir():
                # Clean up old layout directories
                dir_age = current_time - file_path.stat().st_mtime
                if dir_age > max_age_days * 86400:
                    shutil.rmtree(file_path)

def get_lef_content():
    """Get the content of the LEF file if it exists."""
    if LEF_PATH.exists():
        return LEF_PATH.read_text()
    return None

def parse_lef(content: str) -> dict:
    """Parse LEF content into a structured format."""
    cells = {}
    current_cell = None
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Parse MACRO (cell) definitions
        if line.startswith('MACRO'):
            current_cell = line.split()[1]
            cells[current_cell] = {
                'name': current_cell,
                'size': {'width': 0, 'height': 0},
                'pins': []
            }
        elif current_cell and 'SIZE' in line:
            # Parse cell size
            size_parts = line.split()
            width_idx = size_parts.index('SIZE') + 1
            cells[current_cell]['size']['width'] = float(size_parts[width_idx])
            cells[current_cell]['size']['height'] = float(size_parts[width_idx + 2])
        elif current_cell and 'PIN' in line:
            # Parse pin definitions
            pin_name = line.split()[1]
            cells[current_cell]['pins'].append({
                'name': pin_name,
                'x': 0,  # These will be updated when we parse PORT
                'y': 0
            })
            
    return cells