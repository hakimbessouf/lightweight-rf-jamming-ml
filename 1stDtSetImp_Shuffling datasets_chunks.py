# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 07:58:32 2025

@author: Alienware
"""

import os
import pandas as pd

# 📍 Répertoire où sont stockés les fichiers CSV extraits
input_dir = r"D:\\dataset\\extracted_features_w3"
output_dir = os.path.join(input_dir, "merged_parts")
os.makedirs(output_dir, exist_ok=True)

# 📍 Nombre total de parties (ajuste selon ton cas : ici 1461)
num_parts = 1461  

for part in range(1, num_parts + 1):
    try:
        print(f"🔹 Fusion de la partie {part}...")

        # 📍 Chemins des fichiers Gaussian, Nojamming, Sine
        gaussian_file = os.path.join(input_dir, f"iq_features_data_Gaussian_part{part}.csv")
        nojamming_file = os.path.join(input_dir, f"iq_features_data_Nojamming_part{part}.csv")
        sine_file = os.path.join(input_dir, f"iq_features_data_Sine_part{part}.csv")

        # ✅ Vérifie que les trois fichiers existent
        if not (os.path.exists(gaussian_file) and os.path.exists(nojamming_file) and os.path.exists(sine_file)):
            print(f"❌ Partie {part} incomplète (un ou plusieurs fichiers manquants).")
            continue

        # 📍 Charger les trois fichiers
        df_gaussian = pd.read_csv(gaussian_file, low_memory=False)
        df_nojamming = pd.read_csv(nojamming_file, low_memory=False)
        df_sine = pd.read_csv(sine_file, low_memory=False)

        # ✅ Ajouter la colonne "Condition"
        df_gaussian["Condition"] = "Gaussian"
        df_nojamming["Condition"] = "Nojamming"
        df_sine["Condition"] = "Sine"

        # 📍 Concaténer les données
        df_merged = pd.concat([df_gaussian, df_nojamming, df_sine], ignore_index=True)

        # 📍 Mélanger les lignes
        df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

        # 📍 Sauvegarder le fichier fusionné
        output_file = os.path.join(output_dir, f"merged_part{part}.csv")
        df_merged.to_csv(output_file, index=False)

        print(f"✅ Partie {part} fusionnée et sauvegardée : {output_file}")

    except Exception as e:
        print(f"⚠️ Erreur lors du traitement de la partie {part} : {e}")

print("🎉 Fusion et mélange terminés pour toutes les parties !")
