# -*- coding: utf-8 -*-
"""
Extended IQ feature extraction from stitched.csv:
Per-sample:
  - Amplitude, Phase, Power, Instantaneous_Frequency
Global per-file (repeated on each row):
  - Kurtosis_I, Skewness_I, Kurtosis_Q, Skewness_Q
  - Mean_I, Std_I, RMS_I, PAPR_I
  - Mean_Q, Std_Q, RMS_Q, PAPR_Q
  - Mean_Amplitude, Std_Amplitude, RMS_Amplitude, PAPR_Amplitude
  - Spectral_Centroid, Spectral_Bandwidth, Spectral_Flatness
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, gmean

ROOT = r"D:\test"          # root of your dataset tree
OUT_SUFFIX = "_features.csv"

def compute_basic_stats(x):
    """Return mean, std, RMS, PAPR for a 1D array."""
    mean = np.mean(x)
    std = np.std(x)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x)) + 1e-12
    papr = (peak**2) / (rms**2 + 1e-12)
    return mean, std, rms, papr

def compute_spectral_features(x, fs=1.0):
    """
    Compute simple spectral features from a real or complex 1D signal x:
      - spectral centroid
      - spectral bandwidth
      - spectral flatness
    fs is a nominal sampling rate (used only to scale frequencies).
    """
    N = len(x)
    if N < 4:
        return 0.0, 0.0, 0.0

    # Remove DC bias
    x0 = x - np.mean(x)

    # Hann window using NumPy (avoids SciPy hann issues)
    window = np.hanning(N).astype(np.float32)

    X = np.fft.rfft(x0 * window)
    mag = np.abs(X) + 1e-12  # avoid log(0)

    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # Spectral centroid
    centroid = np.sum(freqs * mag) / np.sum(mag)

    # Spectral bandwidth (std around centroid)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag))

    # Spectral flatness (geometric mean / arithmetic mean)
    flatness = gmean(mag) / np.mean(mag)

    return centroid, bandwidth, flatness

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

    # ---------- Per-sample features ----------
    amplitude = np.sqrt(i_samples**2 + q_samples**2)
    phase = np.arctan2(q_samples, i_samples)
    power = amplitude**2
    phase_diff = np.diff(phase)
    instantaneous_frequency = np.concatenate(([0.0], phase_diff))

    # ---------- Global statistics per file ----------
    kurt_i = kurtosis(i_samples)
    skew_i = skew(i_samples)
    kurt_q = kurtosis(q_samples)
    skew_q = skew(q_samples)

    mean_i, std_i, rms_i, papr_i = compute_basic_stats(i_samples)
    mean_q, std_q, rms_q, papr_q = compute_basic_stats(q_samples)
    mean_a, std_a, rms_a, papr_a = compute_basic_stats(amplitude)

    # Spectral features computed on amplitude (can change to I, Q, or complex IQ)
    fs_nominal = 1.0  # if you know the real sampling rate, put it here
    spec_centroid, spec_bw, spec_flat = compute_spectral_features(amplitude, fs=fs_nominal)

    # Repeat global features for every row
    features = {
        "Amplitude": amplitude,
        "Phase": phase,
        "Power": power,
        "Instantaneous_Frequency": instantaneous_frequency,
        "Kurtosis_I": np.full(length, kurt_i, dtype=np.float32),
        "Skewness_I": np.full(length, skew_i, dtype=np.float32),
        "Kurtosis_Q": np.full(length, kurt_q, dtype=np.float32),
        "Skewness_Q": np.full(length, skew_q, dtype=np.float32),
        "Mean_I": np.full(length, mean_i, dtype=np.float32),
        "Std_I": np.full(length, std_i, dtype=np.float32),
        "RMS_I": np.full(length, rms_i, dtype=np.float32),
        "PAPR_I": np.full(length, papr_i, dtype=np.float32),
        "Mean_Q": np.full(length, mean_q, dtype=np.float32),
        "Std_Q": np.full(length, std_q, dtype=np.float32),
        "RMS_Q": np.full(length, rms_q, dtype=np.float32),
        "PAPR_Q": np.full(length, papr_q, dtype=np.float32),
        "Mean_Amplitude": np.full(length, mean_a, dtype=np.float32),
        "Std_Amplitude": np.full(length, std_a, dtype=np.float32),
        "RMS_Amplitude": np.full(length, rms_a, dtype=np.float32),
        "PAPR_Amplitude": np.full(length, papr_a, dtype=np.float32),
        "Spectral_Centroid": np.full(length, spec_centroid, dtype=np.float32),
        "Spectral_Bandwidth": np.full(length, spec_bw, dtype=np.float32),
        "Spectral_Flatness": np.full(length, spec_flat, dtype=np.float32),
    }

    df_feat = pd.DataFrame(features)

    out_path = os.path.splitext(csv_path)[0] + OUT_SUFFIX
    df_feat.to_csv(out_path, index=False)
    print(f"  Saved {out_path}  (rows: {length})")

def walk_and_process(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower() == "stitched.csv":
                csv_path = os.path.join(dirpath, fname)
                process_stitched_csv(csv_path)

if __name__ == "__main__":
    walk_and_process(ROOT)
