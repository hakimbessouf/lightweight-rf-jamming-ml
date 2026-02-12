# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 01:20:29 2026

@author: Alienware
"""

import os
import pandas as pd
import numpy as np

ROOT = r"D:\test"                  # root where stitched_features.csv files are located
OUT_SUBFOLDER = "merged_features"  # new subfolder name
SLICE_SIZE = 90000

def find_feature_files(root):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower() == "stitched_features.csv":
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)

def interleave_and_save(files, root, out_subfolder, slice_size):
    if not files:
        print("No stitched_features.csv files found.")
        return

    out_dir = os.path.join(root, out_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    # Load all feature CSVs
    dfs = [pd.read_csv(f) for f in files]
    lengths = [len(df) for df in dfs]
    min_len = min(lengths)
    if len(set(lengths)) != 1:
        print(f"WARNING: different lengths {lengths}, using min_len={min_len}")

    arrays = [df.iloc[:min_len].to_numpy() for df in dfs]
    num_files = len(arrays)
    num_rows = min_len
    num_cols = arrays[0].shape[1]

    # Interleave rows: row0 of all files, row1 of all files, ...
    interleaved = np.empty((num_rows * num_files, num_cols), dtype=arrays[0].dtype)
    idx = 0
    for r in range(num_rows):
        for f in range(num_files):
            interleaved[idx, :] = arrays[f][r, :]
            idx += 1

    columns = dfs[0].columns

    total_rows = interleaved.shape[0]
    part = 1
    for start in range(0, total_rows, slice_size):
        end = min(start + slice_size, total_rows)
        slice_arr = interleaved[start:end, :]
        out_df = pd.DataFrame(slice_arr, columns=columns)
        out_path = os.path.join(out_dir, f"merged_features_part{part}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} (rows {start}–{end-1})")
        part += 1

if __name__ == "__main__":
    feature_files = find_feature_files(ROOT)
    print("Found stitched_features.csv files:")
    for f in feature_files:
        print("  ", f)
    interleave_and_save(feature_files, ROOT, OUT_SUBFOLDER, SLICE_SIZE)
