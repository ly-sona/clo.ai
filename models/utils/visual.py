#!/usr/bin/env python
# utils/visual.py
"""Heat‑map display and simple 2‑D gate‑graph drawing."""

import numpy as np, matplotlib.pyplot as plt, networkx as nx
from pathlib import Path

# Light colour map per gate family
GATE_COLOURS = {
    "nand":  "#64b5f6", "nor": "#ff8a65", "and": "#9575cd",
    "or":    "#4db6ac", "inv": "#e57373", "buf": "#ffd54f"
}

def plot_heatmap(arr2d: np.ndarray, title="", out: Path | None = None):
    plt.figure(figsize=(10, 8))
    plt.title(title); plt.imshow(arr2d, cmap="viridis")
    plt.axis("off")
    if out: plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

def draw_gate_graph(conns: dict[str, list[str]],
                    types: dict[str, str],
                    sizes: dict[str, float],
                    title="Optimised circuit",
                    out: Path | None = None):
    """Render a simple DAG with node size ∝ gate size."""
    G = nx.DiGraph()
    for dst, srcs in conns.items():
        for src in srcs: G.add_edge(src, dst)

    # Graphviz 'dot' if available; else spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 9))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=[GATE_COLOURS.get(types.get(n, ""), "#cccccc") for n in G.nodes],
        node_size=[400 * sizes.get(n, 1.0) for n in G.nodes],
        arrowsize=12, linewidths=0.6, edge_color="#888888"
    )
    plt.title(title); plt.axis("off")
    if out: plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()