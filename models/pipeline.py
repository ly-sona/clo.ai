import os
import json
import argparse
import numpy as np


def load_manifest(manifest_path):
    """
    Load manifest file defining samples and their input/output files.
    Expected format (JSON list of dicts):
    [
      {
        "design_id": "RISCY-a-1-c2-u0",
        "feature_maps": ["macro_region", "cell_density", "rudy", "pin_rudy", "congestion", "instance_power"],
        "label_map": "instance_power"
      },
      ...
    ]
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    return manifest


def load_sample(entry, data_dir, preserve_spatial=True, flatten=False):
    """
    Load one sample's feature maps and label map as arrays.

    Args:
      entry (dict): manifest entry with design_id, feature_maps, label_map
      data_dir (str): base path containing design directories
      preserve_spatial (bool): keep HxW structure (with channels)
      flatten (bool): flatten output to 1D vector

    Returns:
      X (np.ndarray): feature array (C,H,W) or (H*W*C,)
      Y (np.ndarray): label array (H,W) or (H*W,)
    """
    design_dir = os.path.join(data_dir, entry['design_id'])

    # load feature maps
    feats = []
    for fm in entry['feature_maps']:
        path = os.path.join(design_dir, f"{fm}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature map not found: {path}")
        feats.append(np.load(path))
    # stack along channel dimension
    X = np.stack(feats, axis=0)  # shape (C, H, W)

    # load label
    label_name = entry.get('label_map')
    if label_name:
        label_path = os.path.join(design_dir, f"{label_name}.npy")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label map not found: {label_path}")
        Y = np.load(label_path)  # shape (H, W)
    else:
        Y = None

    # optionally flatten
    if flatten:
        C, H, W = X.shape
        X = X.reshape(C * H * W)
        if Y is not None:
            Y = Y.reshape(H * W)

    return X, Y


def load_dataset(manifest, data_dir, preserve_spatial=True, flatten=False, max_samples=None):
    """
    Load multiple samples as per the manifest.

    Returns:
      X_list: list of np.ndarray
      Y_list: list of np.ndarray or None
    """
    X_list, Y_list = [], []
    for i, entry in enumerate(manifest):
        if max_samples and i >= max_samples:
            break
        X, Y = load_sample(entry, data_dir, preserve_spatial, flatten)
        X_list.append(X)
        if Y is not None:
            Y_list.append(Y)
    return X_list, Y_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for CircuitNet')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing design subdirectories')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to manifest JSON file')
    parser.add_argument('--preserve_spatial', action='store_true',
                        help='Keep 2D spatial structure (C,H,W)')
    parser.add_argument('--flatten', action='store_true',
                        help='Flatten arrays to 1D vectors')
    parser.add_argument('--max_samples', type=int, default=5,
                        help='Load only a small subset for validation')
    args = parser.parse_args()

    # Load manifest
    manifest = load_manifest(args.manifest)
    print(f"Loaded manifest with {len(manifest)} entries.")

    # Load small subset and validate shapes
    X_list, Y_list = load_dataset(
        manifest, args.data_dir,
        preserve_spatial=args.preserve_spatial,
        flatten=args.flatten,
        max_samples=args.max_samples
    )

    print("Sample shapes:")
    for idx, X in enumerate(X_list):
        print(f"Sample {idx}: X shape = {X.shape}")
        if Y_list:
            print(f"          Y shape = {Y_list[idx].shape}")
    print("Preprocessing validation complete.")


# Example Colab notebook cells:
# ```python
# # Mount Google Drive (if needed)
# from google.colab import drive
# drive.mount('/content/drive')
# 
# # Install dependencies
# !pip install numpy
# 
# # Copy pipeline script and manifest to Colab
# !cp pipeline.py /content/
# !cp manifest.json /content/
# 
# # Run preprocessing validation
# !python pipeline.py --data_dir /content/data/CircuitNet --manifest /content/manifest.json --preserve_spatial
# ```
# 
# ```python
# # In notebook: load processed samples
# import numpy as np
# import json
# 
# # Example load
# with open('manifest.json') as f:
#     manifest = json.load(f)
# X_list, Y_list = [], []
# for entry in manifest[:3]:
#     X, Y = load_sample(entry, '/content/data/CircuitNet')
#     X_list.append(X)
#     Y_list.append(Y)
# print([x.shape for x in X_list], [y.shape for y in Y_list])
# 
