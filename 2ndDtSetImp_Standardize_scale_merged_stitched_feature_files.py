import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = r"D:\test\merged_features"
OUT_SUBFOLDER = "scaled"

def find_merged_files(root):
    files = []
    for fname in os.listdir(root):
        if fname.lower().startswith("merged_features_part") and fname.lower().endswith(".csv"):
            files.append(os.path.join(root, fname))
    return sorted(files)

def scale_each_file_independently(files, root, out_subfolder):
    out_dir = os.path.join(root, out_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    for path in files:
        print(f"Processing {path}")
        df = pd.read_csv(path)

        # Select numeric columns only
        num_cols = df.select_dtypes(include=[np.number]).columns
        non_num_cols = [c for c in df.columns if c not in num_cols]

        print("  Numeric columns to scale:", list(num_cols))
        print("  Non-numeric columns kept as is:", non_num_cols)

        X = df[num_cols].values.astype(np.float32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        df_scaled_num = pd.DataFrame(X_scaled, columns=num_cols, index=df.index)

        # Recombine: non-numeric columns unchanged, numeric scaled
        df_out = pd.concat([df[non_num_cols], df_scaled_num], axis=1)
        # Preserve original column order
        df_out = df_out[df.columns]

        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(out_dir, name + "_scaled" + ext)

        df_out.to_csv(out_path, index=False)
        print(f"  Saved scaled file: {out_path}")

if __name__ == "__main__":
    merged_files = find_merged_files(ROOT)
    if not merged_files:
        print("No merged_features_part*.csv files found.")
    else:
        scale_each_file_independently(merged_files, ROOT, OUT_SUBFOLDER)
