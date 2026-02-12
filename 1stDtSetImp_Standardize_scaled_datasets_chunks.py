# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 22:30:34 2025

@author: Alienware
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 📍 Répertoire contenant les fichiers scaled_merged_partX.csv
merged_directory = r"D:\dataset\extracted_features_w1\merged_parts\scaled"

# 📍 Chercher les bons fichiers
merged_files = [f for f in os.listdir(merged_directory) if f.startswith("scaled_merged_part") and f.endswith(".csv")]

print(f"🔹 {len(merged_files)} fichiers trouvés pour normalisation/standardisation.")

# 📍 Initialiser le standard scaler
scaler = StandardScaler()

for file in merged_files:
    file_path = os.path.join(merged_directory, file)
    print(f"   ➝ Traitement de {file}...")

    # Charger le CSV
    df = pd.read_csv(file_path)

    # Vérifier que la colonne 'Condition' existe
    if "Condition" not in df.columns:
        print(f"❌ La colonne 'Condition' est absente dans {file}, saut de ce fichier.")
        continue

    # Séparer la cible
    condition = df["Condition"]
    features = df.drop(columns=["Condition"])

    # Appliquer StandardScaler uniquement sur les features numériques
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Réassembler dataset
    df_scaled = pd.concat([features_scaled, condition], axis=1)

    # Sauvegarder fichier
    output_file = os.path.join(merged_directory, f"standardized_{file}")
    df_scaled.to_csv(output_file, index=False)
    print(f"✅ Sauvegardé : {output_file}")

print("🎉 Tous les fichiers ont été normalisés/standardisés avec succès !")
