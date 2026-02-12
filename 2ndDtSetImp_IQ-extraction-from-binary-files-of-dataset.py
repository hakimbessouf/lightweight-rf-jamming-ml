# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 23:10:20 2026

@author: Alienware
"""

import os
import numpy as np

# Root folder of the dataset
ROOT = r"D:\test"

# Optional: limit number of samples per file for testing (None = all)
MAX_SAMPLES = None   # e.g., 1_000_000

def convert_iq_file(iq_path):
    # Read raw interleaved int16 IQ (fc16)
    raw = np.fromfile(iq_path, dtype=np.int16)
    if raw.size % 2 != 0:
        print(f"[WARN] Odd number of samples in {iq_path}, skipping.")
        return

    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)

    # Normalize (optional but useful for ML)
    i = i / 32768.0
    q = q / 32768.0

    if MAX_SAMPLES is not None:
        i = i[:MAX_SAMPLES]
        q = q[:MAX_SAMPLES]

    # Stack into two-column array: I,Q
    data = np.column_stack((i, q))

    # CSV path: same as .iq but with .csv
    csv_path = os.path.splitext(iq_path)[0] + ".csv"

    # Save without header, comma-separated
    np.savetxt(csv_path, data, delimiter=",", fmt="%.6f")

    print(f"Saved {csv_path}  (samples: {data.shape[0]})")

def walk_and_convert(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".iq"):
                iq_path = os.path.join(dirpath, fname)
                convert_iq_file(iq_path)

if __name__ == "__main__":
    walk_and_convert(ROOT)
