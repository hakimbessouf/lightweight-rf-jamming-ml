# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 01:46:48 2026

@author: Alienware
"""

import os
import numpy as np
import pandas as pd

ROOT = r"D:\test\merged_features\scaled"
OUT_SUBFOLDER = "cleaned"

DROP_CONSTANT_COLS = True  # set False if you want to keep them

def find_scaled_files(root):
    files = []
    for fname in os.listdir(root):
        if fname.lower().endswith("_scaled.csv"):
            files.append(os.path.join(root, fname))
    return sorted(files)

def clean_file(path, out_dir):
    print(f"Cleaning {path}")
    df = pd.read_csv(path)

    # Separate numeric / non-numeric
    num_cols = df.select_dtypes(include=[np.number]).columns
    non_num_cols = [c for c in df.columns if c not in num_cols]

    # Replace +/-inf with NaN, then drop rows with NaN in numeric columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=num_cols)
    after = len(df)
    print(f"  Dropped {before - after} rows with NaN/inf.")

    # Optionally drop constant numeric columns
    if DROP_CONSTANT_COLS:
        const_cols = [c for c in num_cols if df[c].nunique() <= 1]
        if const_cols:
            print("  Dropping constant columns:", const_cols)
            df = df.drop(columns=const_cols)
        else:
            print("  No constant columns to drop.")

    # Save cleaned file
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, name + "_cleaned" + ext)
    df.to_csv(out_path, index=False)
    print(f"  Saved cleaned file: {out_path}")

def main():
    files = find_scaled_files(ROOT)
    if not files:
        print("No *_scaled.csv files found.")
        return

    out_dir = os.path.join(ROOT, OUT_SUBFOLDER)
    os.makedirs(out_dir, exist_ok=True)

    for f in files:
        clean_file(f, out_dir)

if __name__ == "__main__":
    main()
