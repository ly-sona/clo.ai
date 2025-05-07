# features.py
import numpy as np

def extract_features(layout_array):
    """
    Converts a flattened layout or 2D array of shape (N,2) into a feature vector:
    [center_of_mass_x, center_of_mass_y, average_pairwise_distance]
    """
    arr = np.array(layout_array)
    # If 1D, reshape to Nx2
    if arr.ndim == 1:
        layout = arr.reshape(-1, 2)
    else:
        layout = arr
    com = np.mean(layout, axis=0)
    # Compute all pairwise distances
    dists = [np.linalg.norm(layout[i] - layout[j])
             for i in range(len(layout)) for j in range(i+1, len(layout))]
    avg_dist = np.mean(dists)
    return np.hstack([com, avg_dist])