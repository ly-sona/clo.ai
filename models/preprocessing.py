import os
import json
import numpy as np

def load_dataset(input_dirs, output_dirs=None, manifest_path=None, preserve_shape=False):
    """
    Load and combine dataset from the given input (and output) file locations.
    
    Parameters:
        input_dirs (str or list): Path(s) to directory or directories containing input .npy files for features.
                                  Can be a single directory or a list of multiple feature directories.
        output_dirs (str or list, optional): Path(s) to directory or directories for output/label .npy files.
                                  Can be None (if no labels available or for inference only).
        manifest_path (str, optional): Path to a manifest file (JSON or text) listing input-output file pairs.
        preserve_shape (bool): If True, preserve multi-channel 2D structure (do not flatten the feature arrays).
                               If False, flatten each feature array and concatenate for each sample.
    
    Returns:
        X, Y: If output_dirs/manifest provides labels, returns a tuple (X, Y).
              X is a list or array of input features per sample, Y is a list or array of outputs per sample.
              If no outputs are provided, returns X and None.
    """
    # Normalize input_dirs and output_dirs to list form for consistency
    if isinstance(input_dirs, str):
        # Allow comma-separated string for multiple dirs
        if "," in input_dirs:
            input_dir_list = [d.strip() for d in input_dirs.split(",") if d.strip()]
        else:
            input_dir_list = [input_dirs]
    else:
        input_dir_list = list(input_dirs)
    
    if output_dirs:
        if isinstance(output_dirs, str):
            if "," in output_dirs:
                output_dir_list = [d.strip() for d in output_dirs.split(",") if d.strip()]
            else:
                output_dir_list = [output_dirs]
        else:
            output_dir_list = list(output_dirs)
    else:
        output_dir_list = []
    
    data_X = []
    data_Y = []
    
    # If a manifest is provided, use it to load files
    if manifest_path:
        # Determine manifest format (JSON or text)
        if manifest_path.endswith(".json"):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            # The manifest could be a list or dict; handle common structures
            if isinstance(manifest, dict) and "samples" in manifest:
                # e.g., {"samples": [ {...}, {...} ]}
                entries = manifest["samples"]
            elif isinstance(manifest, list):
                entries = manifest
            else:
                # If manifest has an unexpected format, wrap it uniformly
                entries = [manifest]
        else:
            # Text manifest: each line with input and output paths (possibly multiple inputs)
            entries = []
            with open(manifest_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Split by whitespace or comma
                    parts = [p for p in line.replace(',', ' ').split() if p]
                    if len(parts) < 2:
                        continue  # skip lines that don't have at least input and output
                    # If more than 2 parts, assume all but last are input paths and last is output
                    if len(parts) > 2:
                        in_paths = parts[:-1]
                        out_path = parts[-1]
                    else:
                        in_paths = [parts[0]]
                        out_path = parts[1]
                    entries.append({"inputs": in_paths, "output": out_path})
        
        # Iterate through manifest entries
        for entry in entries:
            # Each entry can specify input(s) and output(s)
            if isinstance(entry, dict):
                # Possible keys: 'input'/'inputs', 'output'/'outputs'
                if "inputs" in entry:
                    in_files = entry["inputs"]
                elif "input" in entry:
                    # single input given as 'input'
                    in_files = [entry["input"]]
                else:
                    # If keys not found, try interpreting the entry itself as a mapping of feature name to path,
                    # in which case treat all except maybe a key named 'output' as inputs.
                    in_files = []
                    for k,v in entry.items():
                        if k.lower().startswith("out"):
                            out_file = v
                        else:
                            in_files.append(v)
                
                if "outputs" in entry:
                    out_files = entry["outputs"]
                elif "output" in entry:
                    out_files = [entry["output"]]
                else:
                    # no output specified (maybe inference-only manifest)
                    out_files = []
            elif isinstance(entry, list) or isinstance(entry, tuple):
                # e.g., ["input_path", "output_path"]
                entry_list = list(entry)
                if len(entry_list) > 1:
                    in_files = entry_list[:-1]
                    out_files = [entry_list[-1]]
                else:
                    in_files = entry_list
                    out_files = []
            else:
                # Unrecognized format, skip
                continue
            
            # Load input feature files for this sample
            feature_arrays = []
            for in_path in in_files:
                arr = np.load(in_path, allow_pickle=True)
                feature_arrays.append(arr)
            # Combine multiple input features if present
            if len(feature_arrays) == 1:
                X_sample = feature_arrays[0]
                if not preserve_shape:
                    # Flatten the single feature
                    X_sample = X_sample.reshape(-1)
            else:
                # Multiple input features: ensure they have compatible shapes for stacking/concatenation
                if preserve_shape:
                    # e.g., stack as channels (assumes 2D feature maps with same HxW)
                    try:
                        X_sample = np.stack(feature_arrays, axis=0)
                    except Exception as e:
                        # If stacking fails (shape mismatch), fall back to treating as list
                        X_sample = feature_arrays  
                else:
                    # Flatten each and concatenate into one vector
                    flat_feats = [feat.reshape(-1) for feat in feature_arrays]
                    X_sample = np.concatenate(flat_feats)
            data_X.append(X_sample)
            
            # Load output/label files if specified
            if out_files:
                label_arrays = []
                for out_path in out_files:
                    y_arr = np.load(out_path, allow_pickle=True)
                    label_arrays.append(y_arr)
                if len(label_arrays) == 1:
                    Y_sample = label_arrays[0]
                    if not preserve_shape:
                        Y_sample = Y_sample.reshape(-1)
                else:
                    if preserve_shape:
                        try:
                            Y_sample = np.stack(label_arrays, axis=0)
                        except Exception as e:
                            Y_sample = label_arrays
                    else:
                        flat_labels = [lab.reshape(-1) for lab in label_arrays]
                        Y_sample = np.concatenate(flat_labels)
                data_Y.append(Y_sample)
        # End for each manifest entry
    
    else:
        # No manifest: assume input_dirs contains input files, and output_dirs (if provided) contains output files with matching names
        # Gather all input file paths
        input_files = []
        for d in input_dir_list:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Input directory not found: {d}")
            # get all .npy files in directory
            files = [f for f in os.listdir(d) if f.endswith(".npy")]
            for fname in files:
                input_files.append(os.path.join(d, fname))
        input_files.sort()  # sort for consistency (assumes corresponding output will sort similarly)
        
        # If multiple input directories were provided, the above collected *all* files from all dirs.
        # In that case, we actually need to group features by sample. 
        # It's better to handle multiple input_dirs by manifest to know grouping. 
        # Here, if multiple input dirs, we assume they actually contain the *same filenames* (sample IDs) so grouping by name:
        grouped_inputs = {}
        if len(input_dir_list) > 1:
            # Group files by sample name (filename without directory)
            for path in input_files:
                fname = os.path.basename(path)
                grouped_inputs.setdefault(fname, []).append(path)
        else:
            # Only one input directory, each file is a group by itself
            for path in input_files:
                fname = os.path.basename(path)
                grouped_inputs[fname] = [path]
        
        # Similarly, prepare output files if output_dirs given
        grouped_outputs = {}
        if output_dir_list:
            output_files = []
            for d in output_dir_list:
                if not os.path.isdir(d):
                    raise FileNotFoundError(f"Output directory not found: {d}")
                files = [f for f in os.listdir(d) if f.endswith(".npy")]
                for fname in files:
                    output_files.append(os.path.join(d, fname))
            output_files.sort()
            if len(output_dir_list) > 1:
                # Group outputs by filename
                for path in output_files:
                    fname = os.path.basename(path)
                    grouped_outputs.setdefault(fname, []).append(path)
            else:
                for path in output_files:
                    fname = os.path.basename(path)
                    grouped_outputs[fname] = [path]
        # else: no outputs
        
        # Now iterate through samples (grouped by filename)
        for fname, in_paths in grouped_inputs.items():
            # Load all input features for this sample name
            feature_arrays = [np.load(p, allow_pickle=True) for p in in_paths]
            if len(feature_arrays) == 1:
                X_sample = feature_arrays[0]
                if not preserve_shape:
                    X_sample = X_sample.reshape(-1)
            else:
                if preserve_shape:
                    try:
                        X_sample = np.stack(feature_arrays, axis=0)
                    except Exception as e:
                        X_sample = feature_arrays
                else:
                    flat_feats = [feat.reshape(-1) for feat in feature_arrays]
                    X_sample = np.concatenate(flat_feats)
            data_X.append(X_sample)
            # If a matching output exists for this sample, load it
            if fname in grouped_outputs:
                out_paths = grouped_outputs[fname]
                label_arrays = [np.load(p, allow_pickle=True) for p in out_paths]
                if len(label_arrays) == 1:
                    Y_sample = label_arrays[0]
                    if not preserve_shape:
                        Y_sample = Y_sample.reshape(-1)
                else:
                    if preserve_shape:
                        try:
                            Y_sample = np.stack(label_arrays, axis=0)
                        except Exception as e:
                            Y_sample = label_arrays
                    else:
                        flat_labels = [lab.reshape(-1) for lab in label_arrays]
                        Y_sample = np.concatenate(flat_labels)
                data_Y.append(Y_sample)
        # end for each sample name
    # end if/else manifest
    
    # Convert lists to numpy arrays if possible (uniform shape)
    X_array = None
    Y_array = None
    if data_X:
        # Check if all X have same shape (for stacking into one array)
        try:
            X_array = np.stack(data_X, axis=0)
        except Exception as e:
            # If shapes differ, we keep as list
            X_array = data_X
    if data_Y:
        try:
            Y_array = np.stack(data_Y, axis=0)
        except Exception as e:
            Y_array = data_Y
    else:
        Y_array = None
    
    return X_array, Y_array

# Example usage (if running as a script):
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load CircuitNet dataset features into memory.")
    parser.add_argument("--input_dirs", required=True, 
                        help="Path to input .npy files directory (or multiple directories separated by commas) containing features.")
    parser.add_argument("--output_dirs", required=False, 
                        help="Path to output/label .npy files directory (or multiple, separated by commas).")
    parser.add_argument("--manifest", required=False, 
                        help="Path to a manifest file listing input and output file pairs (JSON or text).")
    parser.add_argument("--preserve_shape", action="store_true", 
                        help="Preserve spatial shape and multi-channel structure of features (do not flatten).")
    args = parser.parse_args()
    
    X, Y = load_dataset(args.input_dirs, output_dirs=args.output_dirs, manifest_path=args.manifest, preserve_shape=args.preserve_shape)
    print(f"Loaded {len(X)} samples from {args.input_dirs}.")
    if isinstance(X, np.ndarray):
        print(f"Input array shape: {X.shape}")
    else:
        # X is a list (varying shapes)
        first_shape = X[0].shape if hasattr(X[0], 'shape') else type(X[0])
        print(f"Input is a list of length {len(X)}. Example element shape/type: {first_shape}")
    if Y is not None:
        if isinstance(Y, np.ndarray):
            print(f"Output array shape: {Y.shape}")
        else:
            first_y_shape = Y[0].shape if hasattr(Y[0], 'shape') else type(Y[0])
            print(f"Output is a list of length {len(Y)}. Example element shape/type: {first_y_shape}")