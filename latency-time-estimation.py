import joblib
import time
import numpy as np
import pandas as pd
import os
import warnings

# --- 1. SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")

# --- 2. CONFIGURATION ---
models_dir = r"D:\Low-Ber-Detection_1\models"  # Your path

# Feature Definitions
FEATURES = {
    "Cloud": [
        "dph_p50", "pwr_kurt", "bp_07", "bp_06", "bp_02", "bp_01_norm", "bp_04",
        "Q_skew", "mag_skew", "bp_01", "bp_08", "bp_08_norm", "bp_10_norm", "I_min",
        "spec_flatness", "bp_03_norm", "I_max", "spec_centroid", "iq_corr", "crest",
        "bp_05", "I_std", "bp_03", "bp_00_norm", "bp_11", "mag_mean", "bp_09",
        "spec_bandwidth", "iq_mean_offset", "I_mean"
    ],
    "Fog": [
        "iq_corr", "crest", "bp_05", "I_std", "bp_03", "bp_00_norm", "bp_11",
        "mag_mean", "bp_09", "spec_bandwidth", "iq_mean_offset", "I_mean"
    ],
    "Edge": [
        "bp_03", "bp_00_norm", "bp_11", "mag_mean", "bp_09", "spec_bandwidth",
        "iq_mean_offset", "I_mean"
    ]
}

# --- SCALING FACTORS (3 Edge, 2 Fog, 1 Cloud) ---
SCALING = {
    # EDGE CATEGORIES
    "Edge_Powerful": 15.0,  # Raspberry Pi 4 / Jetson Nano
    "Edge_Medium":   25.0,  # Raspberry Pi 3 / Zero 2 W
    "Edge_Weak":     50.0,  # ESP32 / Older Microcontrollers
    
    # FOG CATEGORIES
    "Fog_Laptop":    1.2,   # High-end Gateway / Laptop (i5/i7 U-series)
    "Fog_Embedded":  5.0,   # Low-power Gateway (Atom / Celeron / High-end ARM)

    # CLOUD CATEGORY
    "Cloud_Server":  0.8    # Dedicated Server (Xeon / EPYC)
}

# Model Files Map
model_files = {
    # EDGE
    "EDGE_Random_Forest.joblib":       "Edge",
    "EDGE_Naive_Bayes.joblib":         "Edge",
    "EDGE_Logistic_Regression.joblib": "Edge",
    # FOG
    "FOG_Random_Forest.joblib":        "Fog",
    "FOG_Naive_Bayes.joblib":          "Fog",
    "FOG_Logistic_Regression.joblib":  "Fog",
    # CLOUD
    "CLOUD_LightGBM.joblib":           "Cloud",
    "CLOUD_XGBoost.joblib":            "Cloud",
    "CLOUD_Random_Forest.joblib":      "Cloud"
}

# --- 3. BENCHMARK FUNCTION ---
def measure_latency(filename, tier):
    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        return None, "File not found"

    try:
        # Load model
        model = joblib.load(path)
        
        # Create Dummy Data
        feats = FEATURES[tier]
        X_test = pd.DataFrame(np.random.rand(1, len(feats)), columns=feats)
        
        # Warm-up
        model.predict(X_test)

        # Measure
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            model.predict(X_test)
        end = time.perf_counter()

        # Base Latency on PC
        avg_pc_ms = ((end - start) / iterations) * 1000
        
        # Build Results Dictionary
        results_dict = {
            "Model": filename.replace(".joblib", ""),
            "Tier": tier,
            "PC_Latency": avg_pc_ms
        }
        
        if tier == "Edge":
            results_dict["Powerful (Pi4)"] = avg_pc_ms * SCALING["Edge_Powerful"]
            results_dict["Medium (Pi3)"]   = avg_pc_ms * SCALING["Edge_Medium"]
            results_dict["Weak (ESP32)"]   = avg_pc_ms * SCALING["Edge_Weak"]
        elif tier == "Fog":
            results_dict["Laptop (Gateway)"] = avg_pc_ms * SCALING["Fog_Laptop"]
            results_dict["Embedded (Atom)"]  = avg_pc_ms * SCALING["Fog_Embedded"]
        elif tier == "Cloud":
            results_dict["Server (Xeon)"]    = avg_pc_ms * SCALING["Cloud_Server"]

        return results_dict, None

    except Exception as e:
        return None, str(e)

# --- 4. RUN & PRINT TABLES ---

print("=== 1. EDGE TIER (3 Device Classes) ===")
print(f"{'Model':<30} | {'PC (ms)':<8} | {'Powerful':<10} | {'Medium':<10} | {'Weak':<10}")
print("-" * 85)

fog_results = []
cloud_results = []

for fname, tier in model_files.items():
    res, err = measure_latency(fname, tier)
    if res:
        if tier == "Edge":
            print(f"{res['Model']:<30} | {res['PC_Latency']:<8.4f} | {res['Powerful (Pi4)']:<10.4f} | {res['Medium (Pi3)']:<10.4f} | {res['Weak (ESP32)']:<10.4f}")
        elif tier == "Fog":
            fog_results.append(res)
        elif tier == "Cloud":
            cloud_results.append(res)
    elif "File not found" not in str(err):
        print(f"{fname:<30} | ERROR: {err}")

print("\n\n=== 2. FOG TIER (2 Device Classes) ===")
print(f"{'Model':<30} | {'PC (ms)':<8} | {'Laptop Gtwy':<12} | {'Embedded Gtwy':<15}")
print("-" * 75)
for res in fog_results:
    print(f"{res['Model']:<30} | {res['PC_Latency']:<8.4f} | {res['Laptop (Gateway)']:<12.4f} | {res['Embedded (Atom)']:<15.4f}")

print("\n\n=== 3. CLOUD TIER (1 Device Class) ===")
print(f"{'Model':<30} | {'PC (ms)':<8} | {'Server':<10}")
print("-" * 55)
for res in cloud_results:
    print(f"{res['Model']:<30} | {res['PC_Latency']:<8.4f} | {res['Server (Xeon)']:<10.4f}")
print("-" * 55)
