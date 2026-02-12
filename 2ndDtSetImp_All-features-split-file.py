# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 00:39:49 2026

@author: Alienware
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import scipy.signal as signal  # not used now but kept if needed later

ROOT = r"D:\test"      # root of your dataset tree
OUT_SUFFIX = "_features.csv"   # suffix for output files

def process_stitched_csv(csv_path):
    print(f"Processing {csv_path}")

    # Load IQ data; adjust header=None if no header line
    try:
        df_iq = pd.read_csv(csv_path)
    except Exception:
        # If there is no header, uncomment this:
        # df_iq = pd.read_csv(csv_path, header=None, names=["I", "Q"])
        raise

    # Expect columns named 'I' and 'Q' or first two columns as I,Q
    if "I" in df_iq.columns and "Q" in df_iq.columns:
        i_samples = df_iq["I"].to_numpy(dtype=np.float32)
        q_samples = df_iq["Q"].to_numpy(dtype=np.float32)
    else:
        i_samples = df_iq.iloc[:, 0].to_numpy(dtype=np.float32)
        q_samples = df_iq.iloc[:, 1].to_numpy(dtype=np.float32)

    length = len(i_samples)
    if length == 0:
        print("  Empty file, skipping.")
        return

    # Per-sample features
    amplitude = np.sqrt(i_samples**2 + q_samples**2)
    phase = np.arctan2(q_samples, i_samples)
    power = amplitude**2
    phase_diff = np.diff(phase)
    instantaneous_frequency = np.concatenate(([0.0], phase_diff))

    # Global statistics per file
    kurt_i = kurtosis(i_samples)
    skew_i = skew(i_samples)
    kurt_q = kurtosis(q_samples)
    skew_q = skew(q_samples)

    features = {
        "Amplitude": amplitude,
        "Phase": phase,
        "Power": power,
        "Instantaneous_Frequency": instantaneous_frequency,
        "Kurtosis_I": np.full(length, kurt_i, dtype=np.float32),
        "Skewness_I": np.full(length, skew_i, dtype=np.float32),
        "Kurtosis_Q": np.full(length, kurt_q, dtype=np.float32),
        "Skewness_Q": np.full(length, skew_q, dtype=np.float32),
    }

    df_feat = pd.DataFrame(features)

    out_path = os.path.splitext(csv_path)[0] + OUT_SUFFIX
    df_feat.to_csv(out_path, index=False)
    print(f"  Saved {out_path}  (rows: {length})")

def walk_and_process(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            # match stitched.csv exactly; change if needed
            if fname.lower() == "stitched.csv":
                csv_path = os.path.join(dirpath, fname)
                process_stitched_csv(csv_path)

if __name__ == "__main__":
    walk_and_process(ROOT)
