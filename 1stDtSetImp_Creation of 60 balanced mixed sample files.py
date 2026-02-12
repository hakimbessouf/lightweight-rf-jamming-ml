# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:36:19 2025

@author: Alienware
"""

import os
import pandas as pd
import numpy as np

# === 📂 Répertoire contenant les fichiers d'entrée
input_dir = r"D:\dataset\extracted_features_w1\merged_parts\scaled"
output_dir = os.path.join(input_dir, "sampled_60")
os.makedirs(output_dir, exist_ok=True)

# === 🔍 Recherche des fichiers standardized_scaled_merged_partX.csv
files = sorted([f for f in os.listdir(input_dir) if f.startswith("standardized_scaled_merged_part") and f.endswith(".csv")])
print(f"🔹 {len(files)} fichiers trouvés pour échantillonnage.")

# === ⚙️ Paramètres
num_output_files = 60        # Nombre de fichiers finaux à générer
sampling_step = 20           # Pas de sélection (1ère, 20e, 40e ligne, etc.)
target_rows_per_file = 360_000  # Environ 360k lignes par fichier

# === 🧠 Initialisation des fichiers de sortie
buffers = [[] for _ in range(num_output_files)]

for idx, file in enumerate(files):
    file_path = os.path.join(input_dir, file)
    print(f"📖 Lecture de : {file}")

    # Chargement partiel pour ne pas exploser la RAM
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"❌ Erreur lecture {file}: {e}")
        continue

    # Sélection périodique : lignes 0, 20, 40, 60, etc.
    sampled_df = df.iloc[::sampling_step, :]

    # Découper les échantillons dans 60 buffers
    chunks = np.array_split(sampled_df, num_output_files)

    for i in range(num_output_files):
        buffers[i].append(chunks[i])

# === 💾 Sauvegarde des 60 fichiers finaux
for i in range(num_output_files):
    combined_df = pd.concat(buffers[i], ignore_index=True)
    out_path = os.path.join(output_dir, f"sampled_cloud_part{i+1}.csv")

    # Si trop petit ou trop grand → ajustement
    if len(combined_df) > target_rows_per_file:
        combined_df = combined_df.sample(n=target_rows_per_file, random_state=42).reset_index(drop=True)

    combined_df.to_csv(out_path, index=False)
    print(f"✅ Fichier sauvegardé : {out_path} ({len(combined_df)} lignes)")

print("\n🎉 Création des 60 fichiers d'échantillons terminée avec succès !")
