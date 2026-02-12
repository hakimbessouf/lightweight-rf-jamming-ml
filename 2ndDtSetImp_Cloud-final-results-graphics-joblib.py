# -*- coding: utf-8 -*-
"""
Cloud-layer detailed CV for weak-jamming dataset
Uses per-file K-fold CV, saves detailed metrics, confusion matrices, ROC data,
aggregated figures, and per-file trained models.

Directory: D:\test\merged_features\scaled\cleaned
Files: merged_features_part*_scaled_cleaned.csv
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# ========================= CONFIGURATION =========================
data_dir = r"D:\test\merged_features\scaled\cleaned"

output_dir = os.path.join(data_dir, "cloud_results_detailed_cv")
os.makedirs(output_dir, exist_ok=True)

# Adjust feature list to your CSV columns
features_to_use = [
    "Amplitude",
    "Phase",
    "Power",
    "Instantaneous_Frequency",
    "DistJam"
    # "Kurtosis_I",
    # "Skewness_I",
    # "Kurtosis_Q",
    # "Skewness_Q"
]

label_column = "Condition"   # change if your label has another name

# Adapt label set to your dataset (example: Jam / NoJam)
label_encoder = LabelEncoder()
label_encoder.fit(["Jam", "NoJam"])  # edit if needed
classes = label_encoder.classes_

n_splits = 5
random_state = 42

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=random_state
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric='mlogloss',
        max_depth=6, learning_rate=0.1, n_estimators=200,
        random_state=random_state
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, max_depth=10, learning_rate=0.1,
        random_state=random_state, verbose=-1
    )
}

# ========================= PLOTTING STYLE =========================
plt.style.use('classic')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 0.6,
})

# ========================= LOAD DATA FILES =========================
print("="*60)
print("CLOUD LAYER - DETAILED CROSS-VALIDATION ANALYSIS")
print("="*60)

file_pattern = os.path.join(data_dir, "merged_features_part*_scaled_cleaned.csv")
files = sorted(glob.glob(file_pattern))

if len(files) == 0:
    raise FileNotFoundError(f"No files found matching {file_pattern}")

print(f"\nFound {len(files)} data files")
print(f"Files: {[os.path.basename(f) for f in files[:3]]}{'...' if len(files) > 3 else ''}")

all_results = []

# ========================= PER-FILE CROSS-VALIDATION =========================
for file_path in files:
    file_name = os.path.basename(file_path)
    print(f"\n{'='*60}")
    print(f"Processing: {file_name}")
    print(f"{'='*60}")

    df = pd.read_csv(file_path)

    # Check features and label
    for feat in features_to_use:
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' not found in {file_name}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {file_name}")

    X = df[features_to_use].values.astype(np.float32)
    y_raw = df[label_column].values
    y = label_encoder.transform(y_raw)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for model_name, model in models.items():
        print(f"\n  Model: {model_name}")

        fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}
        agg_cm = np.zeros((len(classes), len(classes)), dtype=np.float64)
        probs_list, true_list = [], []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"    Fold {fold_idx} training failed: {e}")
                continue

            y_pred = model.predict(X_val)
            try:
                y_proba = model.predict_proba(X_val)
            except Exception:
                y_proba = None

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

            if y_proba is not None and len(np.unique(y_val)) > 1:
                try:
                    y_val_bin = label_binarize(y_val, classes=np.arange(len(classes)))
                    auc = roc_auc_score(y_val_bin, y_proba, multi_class="ovr", average="weighted")
                except Exception:
                    auc = np.nan
                probs_list.append(y_proba)
                true_list.append(y_val)
            else:
                auc = np.nan

            fold_metrics["accuracy"].append(acc)
            fold_metrics["precision"].append(prec)
            fold_metrics["recall"].append(rec)
            fold_metrics["f1"].append(f1)
            fold_metrics["auc"].append(auc)

            cm_fold = confusion_matrix(y_val, y_pred, labels=np.arange(len(classes)))
            agg_cm += cm_fold

            print(f"    Fold {fold_idx}: Acc={acc:.4f}, Prec={prec:.4f}, "
                  f"Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        acc_mean = np.mean(fold_metrics["accuracy"])
        acc_std = np.std(fold_metrics["accuracy"])
        prec_mean = np.mean(fold_metrics["precision"])
        prec_std = np.std(fold_metrics["precision"])
        rec_mean = np.mean(fold_metrics["recall"])
        rec_std = np.std(fold_metrics["recall"])
        f1_mean = np.mean(fold_metrics["f1"])
        f1_std = np.std(fold_metrics["f1"])
        auc_mean = np.nanmean(fold_metrics["auc"])
        auc_std = np.nanstd(fold_metrics["auc"])

        agg_cm_norm = agg_cm / agg_cm.sum() if agg_cm.sum() > 0 else agg_cm

        print(f"\n  Summary for {model_name} on {file_name}:")
        print(f"     Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"     Precision: {prec_mean:.4f} ± {prec_std:.4f}")
        print(f"     Recall:    {rec_mean:.4f} ± {rec_std:.4f}")
        print(f"     F1-Score:  {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"     AUC:       {auc_mean:.4f} ± {auc_std:.4f}")

        # ---------- save final fitted model for this file & algorithm ----------
        model_out_path = os.path.join(
            output_dir,
            f"model_{model_name.replace(' ', '_')}_{os.path.splitext(file_name)[0]}.joblib"
        )
        joblib.dump(model, model_out_path)
        print(f"  Saved model to: {model_out_path}")
        # ----------------------------------------------------------------------

        all_results.append({
            "file": file_name,
            "model": model_name,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "precision_mean": prec_mean,
            "precision_std": prec_std,
            "recall_mean": rec_mean,
            "recall_std": rec_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "confusion_matrix_counts": agg_cm.tolist(),
            "confusion_matrix_norm": agg_cm_norm.tolist()
        })

        # Per-file confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(agg_cm_norm, annot=True, fmt=".3f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted", fontweight='bold')
        ax.set_ylabel("Actual", fontweight='bold')
        ax.set_title(f"Confusion Matrix\n{model_name} - {file_name}", fontweight='bold')
        plt.tight_layout()
        cm_fname = os.path.join(
            output_dir, f"cm_{model_name.replace(' ', '_')}_{file_name}.png"
        )
        fig.savefig(cm_fname, dpi=300)
        plt.close(fig)

        # ROC data
        if len(probs_list) > 0:
            Ys = np.concatenate(true_list, axis=0)
            Probas = np.vstack(probs_list)
            roc_file = os.path.join(
                output_dir, f"roc_data_{model_name.replace(' ', '_')}_{file_name}.npz"
            )
            np.savez(roc_file, y=Ys, proba=Probas)

# ========================= SAVE RESULTS & FIGURES =========================
results_df = pd.DataFrame(all_results)
results_csv = os.path.join(output_dir, "cloud_model_cv_summary_full.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nSaved detailed CSV: {results_csv}")

model_stats = results_df.groupby('model').agg({
    'accuracy_mean': ['mean', 'std', 'min', 'max'],
    'precision_mean': ['mean', 'std', 'min', 'max'],
    'recall_mean': ['mean', 'std', 'min', 'max'],
    'f1_mean': ['mean', 'std', 'min', 'max'],
    'auc_mean': ['mean', 'std', 'min', 'max']
}).reset_index()

model_stats.columns = ['Model',
                       'Accuracy_Mean', 'Accuracy_Std', 'Accuracy_Min', 'Accuracy_Max',
                       'Precision_Mean', 'Precision_Std', 'Precision_Min', 'Precision_Max',
                       'Recall_Mean', 'Recall_Std', 'Recall_Min', 'Recall_Max',
                       'F1_Mean', 'F1_Std', 'F1_Min', 'F1_Max',
                       'AUC_Mean', 'AUC_Std', 'AUC_Min', 'AUC_Max']

summary_csv = os.path.join(output_dir, "cloud_model_summary_statistics.csv")
model_stats.to_csv(summary_csv, index=False)
print(f"Saved summary statistics: {summary_csv}")

# (You can paste your figure-generation section after this, reusing
# `model_stats`, `results_df`, and `classes` as before.)
