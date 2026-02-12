# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:12:03 2025

@author: Alienware
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

# === 📂 Chemin vers tes fichiers standardisés
data_dir = r"D:\dataset\extracted_features_w1\merged_parts\scaled\sampled_60"
files = [f for f in os.listdir(data_dir) if f.startswith("Sampled_part_") and f.endswith(".csv")]

# === ⚙️ Features utilisées au niveau Cloud (Edge + un peu plus)
features_to_use = ["Amplitude", "Phase", "Power", "Instantaneous_Frequency", "Kurtosis_I", "Kurtosis_Q"]

# === Initialisation des encodeurs et structures de stockage
label_encoder = LabelEncoder()
label_encoder.fit(["Gaussian", "Nojamming", "Sine"])

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=6, learning_rate=0.1, n_estimators=150),
    "LightGBM": LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42)
}

results = []
all_metrics = []

# === Répertoire pour graphiques
output_dir = os.path.join(data_dir, "cloud_results")
os.makedirs(output_dir, exist_ok=True)

for file in files[:60]:  # Prends 60 fichiers pour une simulation réaliste
    file_path = os.path.join(data_dir, file)
    print(f"\n🔹 Traitement du fichier : {file}")
    df = pd.read_csv(file_path)
    
    X = df[features_to_use]
    y = label_encoder.transform(df["Condition"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    for name, model in models.items():
        print(f"   ➝ Entraînement du modèle : {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

        print(f"      ➝ Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        
        results.append([file, name, acc, prec, rec, f1, auc])

        # === Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f"Confusion Matrix - {name}\n({file})")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_{name}_{file}.png"))
        plt.close()

        # === Importance des features (Random Forest ou XGBoost ou LGBM)
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': features_to_use, 'Importance': importances})
        feat_imp.sort_values(by="Importance", ascending=False, inplace=True)
        plt.figure(figsize=(7, 4))
        sns.barplot(x="Importance", y="Feature", data=feat_imp, palette="viridis")
        plt.title(f"Feature Importance - {name}\n({file})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_importance_{name}_{file}.png"))
        plt.close()

# === Sauvegarde CSV
df_results = pd.DataFrame(results, columns=["File", "Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])
df_results.to_csv(os.path.join(output_dir, "cloud_results_summary.csv"), index=False)

# === Graphique comparatif global
plt.figure(figsize=(8, 5))
sns.boxplot(x="Model", y="Accuracy", data=df_results, palette="Set2")
plt.title("Comparaison des modèles Cloud - Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparison_accuracy.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Model", y="F1", data=df_results, palette="Set3")
plt.title("Comparaison des modèles Cloud - F1-score")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparison_f1.png"))
plt.close()

print(f"\n✅ Résultats enregistrés dans : {output_dir}")
print(f"📊 CSV des résultats : {os.path.join(output_dir, 'cloud_results_summary.csv')}")
