# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 00:37:05 2025

@author: Alienware
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud-level model evaluation with inter-file validation.
→ Train on a subset of files, test on distinct unseen files.
→ Models: Random Forest, XGBoost, LightGBM.
→ Outputs: confusion matrices, feature importances, boxplots, CSV summary.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# === 📂 Chemin vers les fichiers Cloud
data_dir = r"D:\dataset\extracted_features_w1\merged_parts\scaled\sampled_60"
files = sorted([f for f in os.listdir(data_dir) if f.startswith("Sampled_part_") and f.endswith(".csv")])

# === 📊 Caractéristiques utilisées (Edge + Kurtosis)
features_to_use = ["Amplitude", "Phase", "Power", "Instantaneous_Frequency", "Kurtosis_I", "Kurtosis_Q"]

# === 🔠 Encodeur pour les classes
label_encoder = LabelEncoder()
label_encoder.fit(["Gaussian", "Nojamming", "Sine"])

# === ⚙️ Modèles à évaluer
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=6, learning_rate=0.1, n_estimators=150),
    "LightGBM": LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42)
}

# === 📁 Répertoire de sortie
output_dir = os.path.join(data_dir, "cloud_results_interfile")
os.makedirs(output_dir, exist_ok=True)

# === 🧠 Split fichiers : 80% train / 20% test
n_train = int(len(files) * 0.8)
train_files = files[:n_train]
test_files = files[n_train:]

print(f"\n📘 Training on {len(train_files)} files")
print(f"📗 Testing on {len(test_files)} files\n")

# === 🔹 Charger et combiner les fichiers
def load_files(file_list):
    dfs = []
    for f in file_list:
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df_train = load_files(train_files)
df_test = load_files(test_files)

# === 🔢 Préparer X, y
X_train = df_train[features_to_use]
y_train = label_encoder.transform(df_train["Condition"])

X_test = df_test[features_to_use]
y_test = label_encoder.transform(df_test["Condition"])

# === 📊 Stockage résultats
results = []

# === Boucle sur les modèles
for name, model in models.items():
    print(f"🚀 Training {name} ...")

    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 📈 Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print(f"   ➝ Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    results.append([name, acc, prec, rec, f1, auc])

    # === 🔹 Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_{name}.png"), dpi=300)
    plt.close()

    # === 🔹 Importance des features (si disponible)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': features_to_use, 'Importance': importances})
        feat_imp.sort_values(by="Importance", ascending=False, inplace=True)
        plt.figure(figsize=(7, 4))
        sns.barplot(x="Importance", y="Feature", data=feat_imp, palette="viridis")
        plt.title(f"Feature Importance - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_importance_{name}.png"), dpi=300)
        plt.close()

    # === 🔹 Sauvegarde du modèle
    joblib.dump(model, os.path.join(output_dir, f"{name.replace(' ', '_')}_model.pkl"))

# === 📁 Résumé des résultats
df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])
df_results.to_csv(os.path.join(output_dir, "cloud_results_interfile_summary.csv"), index=False)

# === 📊 Graphique comparatif global
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=df_results, palette="Set2")
for i, v in enumerate(df_results["Accuracy"]):
    plt.text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=9)
plt.ylim(0.6, 1.0)
plt.title("Cloud Models (Inter-file Validation) - Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparison_accuracy_interfile.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="F1", data=df_results, palette="Set3")
for i, v in enumerate(df_results["F1"]):
    plt.text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=9)
plt.ylim(0.6, 1.0)
plt.title("Cloud Models (Inter-file Validation) - F1-score")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparison_f1_interfile.png"), dpi=300)
plt.close()

print("\n✅ Cloud inter-file validation complete.")
print(f"📊 Results CSV: {os.path.join(output_dir, 'cloud_results_interfile_summary.csv')}")
print(f"📈 Figures saved in: {output_dir}")
