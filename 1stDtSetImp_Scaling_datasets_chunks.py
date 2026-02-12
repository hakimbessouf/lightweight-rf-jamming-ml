# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 08:35:37 2025

@author: Alienware
"""
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 📍 Input and output directories
merged_directory = r"D:\dataset\extracted_features_w3\merged_parts"
output_directory = os.path.join(merged_directory, "scaled")
os.makedirs(output_directory, exist_ok=True)

# 📍 Detect all merged_part*.csv files
files_to_process = sorted([f for f in os.listdir(merged_directory) if f.startswith("merged_part") and f.endswith(".csv")])

for file in files_to_process:
    file_path = os.path.join(merged_directory, file)
    output_path = os.path.join(output_directory, f"scaled_{file}")
    
    print(f"\n🔹 Scaling {file} ...")
    
    # Load file
    df = pd.read_csv(file_path, low_memory=False)
    
    # Separate features and labels
    X = df.drop(columns=["Condition"])
    y = df["Condition"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Rebuild DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled["Condition"] = y
    
    # Save output
    df_scaled.to_csv(output_path, index=False)
    print(f"✅ Saved scaled file: {output_path}")

print("\n🎉 All merged parts have been normalized and saved!")
