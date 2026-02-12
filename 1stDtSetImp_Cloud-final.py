# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 19:28:11 2025

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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# === 📂 Répertoires
data_dir = r"D:\dataset\extracted_features_w1\merged_parts\scaled\sampled_60"
output_dir = os.path.join(data_dir, "cloud_results_interfile_cv")
os.makedirs(output_dir, exist_ok=True)

# === ⚙️ Paramètres et features
features_to_use = ["Amplitude", "Phase", "Power", "Instantaneous_Frequency", "Kurtosis_Q", "Kurtosis_I"]
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

# === Séparation inter-fichiers (80% train / 20% test)
n_train = int(0.8 * len(files))
train_files = files[:n_train]
test_files = files[n_train:]

print(f"📂 Entraînement sur {len(train_files)} fichiers, test sur {len(test_files)} fichiers.")

# === Chargement et concaténation
def load_files(file_list):
    dfs = []
    for f in file_list:
        df = pd.read_csv(os.path.join(data_dir, f))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

train_df = load_files(train_files)
test_df = load_files(test_files)

X_train_full = train_df[features_to_use].values
y_train_full = label_encoder.transform(train_df["Condition"])
X_test = test_df[features_to_use].values
y_test = label_encoder.transform(test_df["Condition"])

# === Résultats
results = []
feature_importances = {}
confusion_matrices = {}

# === K-Fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\n🧠 Modèle : {model_name}")
    fold_scores = []
    feature_imp_folds = []
    cm_total = np.zeros((3,3), dtype=float)

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
        cm_total += confusion_matrix(y_val, y_pred, labels=np.arange(3))

        if hasattr(model, "feature_importances_"):
            feature_imp_folds.append(model.feature_importances_)

        print(f"   ➝ Fold {fold_idx}: acc={acc:.4f}, f1={f1:.4f}")

    mean_scores = np.mean(fold_scores, axis=0)
    std_scores = np.std(fold_scores, axis=0)

    # === Test inter-fichiers
    y_pred_t = model.predict(X_test)
    y_proba_t = model.predict_proba(X_test)
    acc_t = accuracy_score(y_test, y_pred_t)
    prec_t = precision_score(y_test, y_pred_t, average="weighted", zero_division=0)
    rec_t = recall_score(y_test, y_pred_t, average="weighted", zero_division=0)
    f1_t = f1_score(y_test, y_pred_t, average="weighted", zero_division=0)
    auc_t = roc_auc_score(label_binarize(y_test, classes=np.arange(len(classes))), y_proba_t, multi_class="ovr")

    print(f"   🧩 Test inter-fichiers: Acc={acc_t:.4f}, F1={f1_t:.4f}")

    results.append([
        model_name, *mean_scores, *std_scores, acc_t, prec_t, rec_t, f1_t, auc_t
    ])

    # Confusion matrix (test)
    cm = confusion_matrix(y_test, y_pred_t, labels=np.arange(3))
    confusion_matrices[model_name] = cm / cm.sum()

    # Feature importances
    if feature_imp_folds:
        feature_importances[model_name] = np.mean(np.vstack(feature_imp_folds), axis=0)

# === Résumé
columns = [
    "Model", "CV_Acc", "CV_Prec", "CV_Rec", "CV_F1", "CV_AUC",
    "STD_Acc", "STD_Prec", "STD_Rec", "STD_F1", "STD_AUC",
    "Test_Acc", "Test_Prec", "Test_Rec", "Test_F1", "Test_AUC"
]
df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv(os.path.join(output_dir, "cloud_results_summary.csv"), index=False)

# === STYLE GRAPHIQUE
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "axes.linewidth": 0.6,
})

# === Fig 1 : CV Results
fig1 = plt.figure(figsize=(8,5))
sns.barplot(data=df_results.melt(id_vars="Model", value_vars=["CV_Acc","CV_F1","CV_AUC"]),
            x="Model", y="value", hue="variable", palette="tab10")
plt.ylim(0.6, 1.0)
plt.title("Cross-validation Results (Cloud Models)")
plt.ylabel("Score")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Fig1_CV_Results.png"), dpi=300)
plt.close(fig1)

# === Fig 2 : Test Results
fig2 = plt.figure(figsize=(8,5))
sns.barplot(data=df_results.melt(id_vars="Model", value_vars=["Test_Acc","Test_F1","Test_AUC"]),
            x="Model", y="value", hue="variable", palette="Set2")
plt.ylim(0.6, 1.0)
plt.title("Inter-File Test Results (Cloud Models)")
plt.ylabel("Score")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Fig2_Test_Results.png"), dpi=300)
plt.close(fig2)

# === Fig 3 : Confusion matrices
for model_name, cm in confusion_matrices.items():
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=classes, yticklabels=classes, fmt=".2f")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Fig3_Confusion_{model_name.replace(' ','_')}.png"), dpi=300)
    plt.close()

# === Fig 4 : Global Mean ± Std
metrics = ["CV_Acc","CV_F1","CV_AUC"]
plt.figure(figsize=(8,5))
for i, metric in enumerate(metrics):
    plt.errorbar(df_results["Model"], df_results[metric], 
                 yerr=df_results[f"STD_{metric.split('_')[1]}"], fmt='o-', label=metric)
plt.ylim(0.6, 1.0)
plt.title("Global Mean ± Std (Cross-Validation)")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Fig4_Global_Mean_STD.png"), dpi=300)
plt.close()

# === Fig 5 : Feature importance
for model_name, imps in feature_importances.items():
    feat_df = pd.DataFrame({"Feature": features_to_use, "Importance": imps}).sort_values("Importance", ascending=False)
    plt.figure(figsize=(6,4))
    sns.barplot(data=feat_df, y="Feature", x="Importance", palette="mako")
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Fig5_Feature_Importance_{model_name.replace(' ','_')}.png"), dpi=300)
    plt.close()

print(f"\n✅ Tous les graphiques et CSV sont disponibles dans : {output_dir}")
