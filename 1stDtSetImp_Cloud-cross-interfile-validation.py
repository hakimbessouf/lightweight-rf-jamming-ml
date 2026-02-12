# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 02:18:18 2025

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

# === 📂 Répertoires
data_dir = r"D:\dataset\extracted_features_w1\merged_parts\scaled\sampled_60"
output_dir = os.path.join(data_dir, "cloud_results_interfile_cv")
os.makedirs(output_dir, exist_ok=True)

# === ⚙️ Paramètres et features
features_to_use = ["Amplitude", "Phase", "Power", "Instantaneous_Frequency", "Kurtosis_Q"]
label_encoder = LabelEncoder()
label_encoder.fit(["Gaussian", "Nojamming", "Sine"])
classes = label_encoder.classes_

# === Modèles Cloud
models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=6, learning_rate=0.1, n_estimators=200),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
}

# === Chargement de tous les fichiers
files = sorted([f for f in os.listdir(data_dir) if f.startswith("Sampled_part_") and f.endswith(".csv")])
if len(files) == 0:
    raise FileNotFoundError("Aucun fichier trouvé dans le répertoire spécifié.")

# === Séparation inter-fichiers (80% pour train / 20% pour test)
n_train = int(0.8 * len(files))
train_files = files[:n_train]
test_files = files[n_train:]

print(f"📂 Entraînement sur {len(train_files)} fichiers, test sur {len(test_files)} fichiers.")

# === Chargement et concaténation des fichiers
def load_files(file_list):
    dfs = []
    for f in file_list:
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

train_df = load_files(train_files)
test_df = load_files(test_files)

# === Extraction des features et labels
X_train_full = train_df[features_to_use].values
y_train_full = label_encoder.transform(train_df["Condition"])
X_test = test_df[features_to_use].values
y_test = label_encoder.transform(test_df["Condition"])

# === Stockage des résultats
results = []

# === Cross-validation interne (K-Fold sur l'ensemble train)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\n🧠 Modèle : {model_name}")
    fold_scores = []

    # === K-Fold Cross-validation sur les fichiers d'entraînement
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_full, y_train_full), 1):
        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        auc = roc_auc_score(label_binarize(y_val, classes=np.arange(len(classes))), y_proba, multi_class="ovr")

        fold_scores.append([acc, prec, rec, f1, auc])
        print(f"   ➝ Fold {fold_idx}: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

    # === Moyenne des folds
    mean_scores = np.mean(fold_scores, axis=0)
    print(f"   ✅ Moyenne (CV): Acc={mean_scores[0]:.4f}, F1={mean_scores[3]:.4f}, AUC={mean_scores[4]:.4f}")

    # === Test inter-fichiers (données jamais vues)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)
    acc_t = accuracy_score(y_test, y_pred_test)
    prec_t = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
    rec_t = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
    f1_t = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    auc_t = roc_auc_score(label_binarize(y_test, classes=np.arange(len(classes))), y_proba_test, multi_class="ovr")

    print(f"   🧩 Test inter-fichiers: Acc={acc_t:.4f}, F1={f1_t:.4f}, AUC={auc_t:.4f}")

    results.append([model_name, *mean_scores, acc_t, prec_t, rec_t, f1_t, auc_t])

# === Résumé global
columns = [
    "Model", "CV_Accuracy", "CV_Precision", "CV_Recall", "CV_F1", "CV_AUC",
    "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_AUC"
]
df_results = pd.DataFrame(results, columns=columns)

# === Sauvegarde CSV
csv_path = os.path.join(output_dir, "cloud_interfile_cv_summary.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n✅ Résultats sauvegardés dans : {csv_path}")

# === Graphiques
plt.figure(figsize=(8,5))
sns.barplot(data=df_results.melt(id_vars="Model", value_vars=["Test_Accuracy","Test_F1","Test_AUC"]),
            x="Model", y="value", hue="variable", palette="Set2")
plt.ylim(0.6, 1.0)
plt.title("Cloud Models Performance (Inter-file Test)")
plt.ylabel("Score")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cloud_interfile_cv_results.png"), dpi=300)
plt.close()

print(f"📊 Graphique sauvegardé dans : {output_dir}")
