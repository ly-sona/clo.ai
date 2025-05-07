import numpy as np, asyncio, json
import utils
import jobs
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to Python path
from models.circuit_optimizer import CircuitOptimizer

# Constants for grid size
GRID_WIDTH = 100
GRID_HEIGHT = 100

async def optimise(job_id: str, power_threshold: float = 0.5, n_layouts: int = 3):
    jobs.update(job_id, status="running")
    
    # Initialize circuit optimizer
    optimizer = CircuitOptimizer()
    
    # Get the job details to find the correct file path
    job = jobs.get(job_id)
    if not job:
        jobs.update(job_id, status="failed")
        return
        
    # Get the circuit file path from the job's schematic field
    circuit_file = Path(job["schematic"])
    if not circuit_file.exists():
        jobs.update(job_id, status="failed")
        return
    
    # Run optimization
    try:
        # Get optimized circuit and sizes, and also original/optimized metrics
        result = optimizer.optimize_circuit(str(circuit_file))
        if isinstance(result, tuple) and len(result) == 2:
            optimized_circuit, optimized_sizes = result
            # For backward compatibility, set dummy values
            original_power = None
            original_delay = None
            new_delay = None
            new_power = float(sum(optimized_sizes.values()))
        elif isinstance(result, tuple) and len(result) == 6:
            optimized_circuit, optimized_sizes, original_power, original_delay, new_power, new_delay = result
        else:
            jobs.update(job_id, status="failed")
            return
        if optimized_circuit is None:
            jobs.update(job_id, status="failed")
            return

        # Save DEF file
        def_path = utils.TMP / f"{job_id}.def"
        with open(def_path, 'w') as f:
            f.write(optimized_circuit)

        # Save optimized .bench file
        bench_path = utils.LAYOUTS / job_id / f"{job_id}_optimized.bench"
        bench_path.parent.mkdir(exist_ok=True)
        with open(bench_path, 'w') as f:
            f.write(optimized_circuit)

        # --- Generate real circuit layout coordinates ---
        # Parse the netlist again to get connections and gates
        iscas85_content = optimizer.load_iscas85(str(circuit_file))
        inputs, outputs, gates, connections, gate_types = optimizer.parse_iscas85(iscas85_content)
        all_nodes = inputs + gates + outputs
        coords_dict = optimizer.generate_gate_coordinates(connections, all_nodes, inputs)
        layout_xy = np.array([coords_dict.get(n, (0, 0)) for n in all_nodes])
        thumb_url = utils.save_thumb(layout_xy, job_id)

        # Save layout metadata
        layout_data = {
            "id": job_id,
            "power": new_power,
            "original_power": original_power,
            "original_delay": original_delay,
            "new_delay": new_delay,
            "grid_size": {"width": GRID_WIDTH, "height": GRID_HEIGHT},
            "coordinates": layout_xy.tolist(),
            "gate_names": all_nodes,
            "def_path": str(def_path),
            "thumb": thumb_url,
            "optimized_circuit": optimized_circuit
        }
        
        # Save layout metadata and files
        utils.save_layout_metadata(job_id, layout_data)

        # Update job status with all metrics
        jobs.update(
            job_id,
            status="done",
            layouts=[{
                "id": job_id,
                "thumb": thumb_url,
                "power": new_power,
                "original_power": original_power,
                "original_delay": original_delay,
                "new_delay": new_delay,
                "def_path": str(utils.LAYOUTS / job_id / f"{job_id}.def"),
                "optimized_circuit": optimized_circuit
            }],
        )
    except Exception as e:
        print(f"Optimization failed: {e}")
        jobs.update(job_id, status="failed")
