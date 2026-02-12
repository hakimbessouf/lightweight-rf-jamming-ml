# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 00:58:17 2025

@author: SURFACE
"""
import os
import pandas as pd

# Base directory where W1, W3, W7, W13 folders are stored
base_directory = r"E:\extracted-attrib"

# List of selected folders
selected_folders = ["W1", "W3", "W7", "W13"]

# Output directory for merged files
merged_directory = os.path.join(base_directory, "merged_data")
os.makedirs(merged_directory, exist_ok=True)

# Define chunk size to prevent memory issues
chunk_size = 50000  

# Function to merge all CSVs inside a folder
def merge_csvs_in_folder(folder_name, folder_path, output_file):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    all_files.sort()  # Ensure files are read in order

    # Create output file
    with open(output_file, 'w', newline='') as outfile:
        writer = None  # Initialize CSV writer

        for file in all_files:
            file_path = os.path.join(folder_path, file)

            # Read in chunks
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                condition = file.split("_")[3]  # Extract condition (Gaussian, Nojamming, Sine)
                chunk["Condition"] = condition
                chunk["Source"] = folder_name  # Add folder name as "Source"

                # ✅ Append chunk to output file
                if writer is None:
                    chunk.to_csv(outfile, mode='a', index=False)
                    writer = True  # Set flag after first write
                else:
                    chunk.to_csv(outfile, mode='a', index=False, header=False)

    print(f"✅ Merged data saved: {output_file}")

# Process each folder separately
for folder in selected_folders:
    folder_path = os.path.join(base_directory, folder)
    output_file = os.path.join(merged_directory, f"merged_{folder}.csv")

    if os.path.exists(folder_path):
        merge_csvs_in_folder(folder, folder_path, output_file)
    else:
        print(f"⚠️ Folder not found: {folder_path}")

print("✅ All datasets merged into separate files.")
