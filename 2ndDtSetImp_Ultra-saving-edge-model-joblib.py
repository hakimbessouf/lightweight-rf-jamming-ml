# -*- coding: utf-8 -*-
"""
Ultra-Lightweight Edge ML Models - Power Saving Mode
on weak-jamming dataset (Jam / NoJam)

Directory: D:\test\merged_features\scaled\cleaned
Files: merged_features_part*_scaled_cleaned.csv

Per-file StratifiedKFold CV, ultra-light RF / NB / LR,
detailed metrics, confusion matrices, ROC data,
and Edge vs Cloud comparison plots.
"""

# ============================================================================
# ULTRA-LIGHTWEIGHT MODEL CONFIGURATION - POWER SAVING MODE
# ============================================================================
# These configurations are optimized for:
# - Minimal energy consumption
# - Low memory footprint
# - Fast inference time (<10ms target)
# - Edge deployment on resource-constrained devices (e.g., IoT nodes, MANETs)
#
# Model Complexity Reduction Summary:
# ┌─────────────────────┬──────────────┬─────────────────┬─────────────────┐
# │ Model               │ Standard     │ Ultra-Light     │ Reduction       │
# ├─────────────────────┼──────────────┼─────────────────┼─────────────────┤
# │ Random Forest       │ 100 trees    │ 10 trees        │ 90% fewer trees │
# │                     │ depth=10     │ depth=5         │ 50% less depth  │
# │ Logistic Regression │ 1000 iters   │ 100 iters       │ 90% fewer iters │
# │ Naive Bayes         │ (minimal)    │ (no change)     │ already optimal │
# └─────────────────────┴──────────────┴─────────────────┴─────────────────┘
#
# Expected Performance Trade-off:
# - Accuracy: ~1-3% decrease compared to standard configuration
# - Energy Consumption: ~85-90% reduction
# - Inference Time: ~80-85% reduction
# - Model Size: ~90% reduction (Random Forest)
# ============================================================================

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- USER CONFIG ---------------------------------
data_dir = r"D:\test\merged_features\scaled\cleaned"

results_dir = os.path.join(data_dir, "edge_results_cv")
os.makedirs(results_dir, exist_ok=True)

# Use light feature set (you can add more from your extended set)
features_to_use = [
    #"Amplitude",
    #"Phase",
    #"Power"
    "Instantaneous_Frequency"
    #"DistJam"
    #"Kurtosis_I",
    #"Skewness_I",
    #"Kurtosis_Q",
    #"Skewness_Q",
    #"Mean_I",
    #"Std_I",
    #"RMS_I",
    #"PAPR_I",
    #"Mean_Q",
    #"Std_Q",
    #"RMS_Q",
    #"PAPR_Q",
    #"Mean_Amplitude",
    #"Std_Amplitude",
    #"RMS_Amplitude",
    #"PAPR_Amplitude",
    #"Spectral_Centroid",
    #"Spectral_Bandwidth",
    #"Spectral_Flatness",
]

label_column = "Condition"  # expected values: "Jam", "NoJam"

n_splits = 5
random_state = 42

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=1
    ),
    "Naive Bayes": GaussianNB(var_smoothing=1e-8),
    "Logistic Regression": LogisticRegression(
        max_iter=100,
        solver='lbfgs',
        C=1.0,
        random_state=random_state,
        n_jobs=1
    )
}

# Cloud placeholders (for comparison plots)
cloud_table4 = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost", "LightGBM"],
    "Mean Accuracy": [0.999, 0.999, 0.999],
    "Std Dev": [0.0004, 0.0005, 0.0003],
    "Min": [0.998, 0.998, 0.998],
    "Max": [1.000, 1.000, 1.000]
})

# ---------------------------- STYLE ----------------------------------------
plt.style.use('classic')
sns.set_style("whitegrid", {'axes.grid': False})

# ---------------------------- LOAD FILES -----------------------------------
file_pattern = os.path.join(data_dir, "merged_features_part*_scaled_cleaned.csv")
files = sorted(glob.glob(file_pattern))
if len(files) == 0:
    print(f"No files found matching {file_pattern}.")
    sys.exit(1)

print(f"Found {len(files)} cleaned scaled files.")

# Label encoder for Jam / NoJam
label_encoder = LabelEncoder()
label_encoder.fit(["Jam", "NoJam"])
classes_labels = label_encoder.classes_

all_results = []

# ==================== PER-FILE CROSS-VALIDATION ============================
for file_path in files:
    file_name = os.path.basename(file_path)
    print(f"\nProcessing file: {file_name}")

    df = pd.read_csv(file_path)

    # Validate columns
    for feat in features_to_use:
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' not found in {file_name}. "
                             f"Available: {df.columns.tolist()}")

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not in {file_name}.")

    X = df[features_to_use].values.astype(np.float32)
    y_raw = df[label_column].values
    y = label_encoder.transform(y_raw)
    classes = np.unique(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for model_name, model in models.items():
        print(f"  Model: {model_name}")
        fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}
        agg_cm = np.zeros((len(classes), len(classes)), dtype=np.float64)
        probs_list, true_list = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"    [Fold {fold_idx}] Training failed: {e}")
                continue

            y_pred = model.predict(X_test)

            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:
                    y_proba = None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            if y_proba is not None and len(np.unique(y_test)) > 1:
                try:
                    auc = roc_auc_score(
                        label_binarize(y_test, classes=classes),
                        y_proba,
                        multi_class="ovr",
                        average="weighted"
                    )
                except Exception:
                    auc = np.nan
                probs_list.append((y_test, y_proba))
                true_list.append(y_test)
            else:
                auc = np.nan

            fold_metrics["accuracy"].append(acc)
            fold_metrics["precision"].append(prec)
            fold_metrics["recall"].append(rec)
            fold_metrics["f1"].append(f1)
            fold_metrics["auc"].append(auc)

            cm = confusion_matrix(y_test, y_pred, labels=classes)
            agg_cm += cm

            print(f"    Fold {fold_idx}: acc={acc:.4f}, f1={f1:.4f}, "
                  f"auc={np.nan if np.isnan(auc) else auc:.4f}")

        def mean_std(lst):
            arr = np.array([x for x in lst if not (x is None or (isinstance(x, float) and np.isnan(x)))])
            if arr.size == 0:
                return np.nan, np.nan
            return arr.mean(), arr.std(ddof=0)

        acc_mean, acc_std = mean_std(fold_metrics["accuracy"])
        prec_mean, prec_std = mean_std(fold_metrics["precision"])
        rec_mean, rec_std = mean_std(fold_metrics["recall"])
        f1_mean, f1_std = mean_std(fold_metrics["f1"])
        auc_vals = [v for v in fold_metrics["auc"] if not (v is None or np.isnan(v))]
        auc_mean, auc_std = mean_std(auc_vals)

        total_samples = agg_cm.sum()
        agg_cm_norm = agg_cm / total_samples if total_samples > 0 else agg_cm

        # -------- save final fitted edge model for this file & algorithm -----
        model_out_path = os.path.join(
            results_dir,
            f"model_{model_name.replace(' ', '_')}_{os.path.splitext(file_name)[0]}.joblib"
        )
        joblib.dump(model, model_out_path)
        print(f"  Saved model to: {model_out_path}")
        # ---------------------------------------------------------------------

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

        # Save confusion matrix figure (normalized)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(
            agg_cm_norm, annot=True, fmt=".3f", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Aggregated Confusion Matrix\n{model_name} - {file_name} (CV k={n_splits})")
        plt.tight_layout()
        cm_fname = os.path.join(results_dir, f"cm_{model_name.replace(' ', '_')}_{file_name}.png")
        fig.savefig(cm_fname, dpi=300)
        plt.close(fig)

        # Save ROC data
        if len(probs_list) > 0:
            Ys = np.concatenate([y_t for (y_t, _) in probs_list], axis=0)
            Probas = np.vstack([p for (_, p) in probs_list])
            roc_csv_path = os.path.join(results_dir, f"roc_data_{model_name.replace(' ', '_')}_{file_name}.npz")
            np.savez(roc_csv_path, y=Ys, proba=Probas)

# ================= SAVE SUMMARY AND PLOTS ==================================
results_df = pd.DataFrame(all_results)
results_csv = os.path.join(results_dir, "edge_model_cv_summary_full.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nSaved CSV results to: {results_csv}")

plot_metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "auc_mean"]
summary_per_model = results_df.groupby("model")[plot_metrics].mean().reset_index()
summary_per_model = summary_per_model.rename(columns={
    "accuracy_mean":"Accuracy",
    "precision_mean":"Precision",
    "recall_mean":"Recall",
    "f1_mean":"F1",
    "auc_mean":"AUC"
})

cloud_plot = pd.DataFrame({
    "Model": cloud_table4["Model"],
    "Accuracy": cloud_table4["Mean Accuracy"],
    "Precision": cloud_table4["Mean Accuracy"],
    "Recall": cloud_table4["Mean Accuracy"],
    "F1": cloud_table4["Mean Accuracy"],
    "AUC": [0.9999, 0.9999, 0.9999]
})

edge_rows = summary_per_model.copy()
edge_rows["Layer"] = "Edge"
cloud_rows = cloud_plot.copy()
cloud_rows["Layer"] = "Cloud"
combined_plot_df = pd.concat([edge_rows, cloud_rows], ignore_index=True)

metrics = ["Accuracy","Precision","Recall","F1","AUC"]
vals = combined_plot_df[metrics].values
labels = [f"{m}\n({l})" for m,l in zip(combined_plot_df["Model"], combined_plot_df["Layer"])]

fig, ax = plt.subplots(figsize=(12,6))
n_groups, n_metrics = vals.shape
width = 0.13
indices = np.arange(n_groups)
for j in range(n_metrics):
    ax.bar(indices + (j - n_metrics/2)*width + width/2, vals[:,j], width, label=metrics[j])
ax.set_xticks(indices)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel("Metric value")
ax.set_ylim(0,1.05)
ax.set_title("Combined metrics: Edge (light) vs Cloud (heavy) models")
ax.legend(loc='upper left', bbox_to_anchor=(1.01,1), frameon=False)
plt.tight_layout()
combined_path = os.path.join(results_dir, "fig_combined_metrics_edge_cloud.png")
fig.savefig(combined_path, dpi=300)
plt.close(fig)
