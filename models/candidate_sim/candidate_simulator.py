def simulate_layout(candidate_id, layout_path):
    """
    Placeholder simulation routine for validating layout.
    """
    print(f"[SIMULATION] Running simulation for Candidate {candidate_id} at {layout_path}...")
    # Here you could later call OpenROAD via subprocess or API
    return {
        "candidate_id": candidate_id,
        "status": "success",
        "timing_score": 0.75,  # Dummy value
        "congestion_score": 0.5
    }

import gdstk
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import random
from matplotlib.widgets import Button


# Path to sample gds layout file
gds_path = "C:/Users/orvin/clo.ai/models/candidate_sim/nand2.gds2"
gds_path2 = "C:/Users/orvin/clo.ai/models/candidate_sim/xor.gds2"
gds_paths = [gds_path, gds_path2]
# Keep everything in memory instead of using file paths
overlay_data = []

for i in range(3):
    # Generate fake macros
    macros = []
    for j in range(3):
        macros.append({
            "name": f"MACRO{i}_{j}",
            "x": random.randint(5, 125),
            "y": random.randint(5, 125),
            "width": random.randint(20, 40),
            "height": random.randint(15, 30)
        })

    # Generate fake congestion
    congestion = pd.DataFrame({
        'x': [random.randint(5, 125) for _ in range(5)],
        'y': [random.randint(5, 125) for _ in range(5)],
        'congestion_level': [round(random.uniform(0.1, 1.0), 2) for _ in range(5)]
    })

    overlay_data.append((macros, congestion))
    
    def visualize_candidate_overlay(
    gds_path,
    macros=None,
    congestion_df=None,
    max_polygons=1000,
    show_macros=True,
    show_congestion=True
):
        # Same logic as before, just skips file loading
        lib = gdstk.read_gds(gds_path)
        top_cell = lib.top_level()[0]
        polygons = top_cell.get_polygons()

        fig, ax = plt.subplots(figsize=(20, 20))
        all_xs, all_ys = [], []

        for i, polygon in enumerate(polygons):
            if i >= max_polygons:
                break
            xs, ys = polygon.points[:, 0], polygon.points[:, 1]
            ax.fill(xs, ys, alpha=0.7)
            all_xs.extend(xs)
            all_ys.extend(ys)

        if show_macros and macros:
            for macro in macros:
                rect = plt.Rectangle(
                    (macro['x'], macro['y']),
                    macro['width'],
                    macro['height'],
                    linewidth=2,
                    edgecolor='blue',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    macro['x'],
                    macro['y'] + macro['height'] + 5,
                    macro['name'],
                    fontsize=8,
                    color='blue'
                )
                all_xs.extend([macro['x'], macro['x'] + macro['width']])
                all_ys.extend([macro['y'], macro['y'] + macro['height']])

        if show_congestion and congestion_df is not None and not congestion_df.empty:
            for _, row in congestion_df.iterrows():
                ax.scatter(
                    row['x'],
                    row['y'],
                    color='red',
                    s=row['congestion_level'] * 300,
                    alpha=0.6,
                    edgecolors='black'
                )
                all_xs.append(row['x'])
                all_ys.append(row['y'])

        if all_xs and all_ys:
            margin = 25
            ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
            ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)

        ax.set_title("Candidate Layout View")
        ax.set_xlabel("Microns")
        ax.set_ylabel("Microns")
        ax.set_aspect("equal")
        ax.grid(True)
        plt.show()
        
        return fig, ax
    
current_index = 0
cur_path = 0

def show_current_candidate():
    macros, congestion_df = overlay_data[current_index]
    gdsp = gds_paths[current_index]
    fig, ax = visualize_candidate_overlay(
        gds_path=gdsp,
        macros=macros,
        congestion_df=congestion_df,
        show_macros=True,
        show_congestion=True
    )
    

def on_prev_clicked(b):
    global current_index
    print(current_index)
    show_current_candidate()
    current_index = (current_index - 1) % len(gds_paths)

def on_next_clicked(b):
    global current_index
    print(current_index)
    show_current_candidate()
    current_index = (current_index + 1) % len(gds_paths)

# Add buttons to the plot
ax_button_prev = plt.axes([0.1, 0.01, 0.1, 0.05])
ax_button_next = plt.axes([0.8, 0.01, 0.1, 0.05])

button_prev = Button(ax_button_prev, '⬅ Previous')
button_next = Button(ax_button_next, 'Next ➡')

button_prev.on_clicked(on_prev_clicked)
button_next.on_clicked(on_next_clicked)

plt.show()

# Display the buttons and the initial layout
show_current_candidate() 