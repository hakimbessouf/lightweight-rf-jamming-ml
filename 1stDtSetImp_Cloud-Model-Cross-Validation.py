# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 00:07:44 2025

@author: Alienware
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud-level evaluation with Stratified K-Fold CV (complete script)
- Loads Sampled_part_*.csv files from data_dir
- For each file: performs StratifiedKFold CV (k folds)
- Trains cloud models (RandomForest, XGBoost, LightGBM) per fold
- Computes per-fold metrics and aggregates mean/std
- Aggregates confusion matrix across folds and across files for each model
- Generates and saves:
    - per-file aggregated confusion matrix images
    - aggregated CSV summary (per file per model: mean +/- std)
    - boxplots and combined bar charts (Accuracy, Precision, Recall, F1, AUC)
    - ROC curves using concatenated CV probabilities
    - aggregated feature importance plots (mean importance across folds/files)
"""
import os
import glob
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ------------------ USER CONFIG ------------------
#data_dir = "/mnt/data/sampled_60"   # <-- change to your folder containing Sampled_part_*.csv
data_dir = r"D:\dataset\extracted_features_w1\merged_parts\scaled\sampled_60"
results_dir = os.path.join(data_dir, "cloud_results_cv")
os.makedirs(results_dir, exist_ok=True)

# Features used at cloud level (edge features + extras)
features_to_use = ["Amplitude", "Phase", "Power", "Instantaneous_Frequency", "Kurtosis_I", "Kurtosis_Q"]

label_column = "Condition"   # expected labels: "Gaussian","Nojamming","Sine"

# CV configuration
n_splits = 5
random_state = 42

# Models to evaluate (cloud/heavy)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=random_state),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=6, learning_rate=0.1, n_estimators=200, verbosity=0),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=random_state)
}

# ------------------ END CONFIG --------------------

sns.set_style("white")
plt.style.use('classic')

# Helper to safely compute mean/std ignoring NaNs
def mean_std_safe(lst):
    arr = np.array([x for x in lst if not (x is None or (isinstance(x, float) and np.isnan(x)))])
    if arr.size == 0:
        return (np.nan, np.nan)
    return (float(arr.mean()), float(arr.std(ddof=0)))

# Collect file list
file_pattern = os.path.join(data_dir, "Sampled_part_*.csv")
files = sorted(glob.glob(file_pattern))
if len(files) == 0:
    print("No files found. Check data_dir and filenames.")
    sys.exit(1)

# Initialize label encoder
label_encoder = LabelEncoder()
label_encoder.fit(["Gaussian", "Nojamming", "Sine"])
classes = label_encoder.classes_

# Storage for all file-model results
all_results = []

# Accumulators for aggregated feature importances across files
feature_importances_acc = {name: [] for name in models.keys()}

# Main loop over files
for file_path in files[:60]:  # use first 60 files (adjust if needed)
    file_name = os.path.basename(file_path)
    print(f"\nProcessing (cloud) {file_name}")

    df = pd.read_csv(file_path)

    # Validate features and labels
    for feat in features_to_use:
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' not found in {file_name}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {file_name}")

    X = df[features_to_use].values
    y_raw = df[label_column].values
    y = label_encoder.transform(y_raw)
    y_binarized_full = label_binarize(y, classes=np.arange(len(classes)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for model_name, model in models.items():
        print(f"  Model: {model_name}")
        # Per-fold storage
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        agg_cm = np.zeros((len(classes), len(classes)), dtype=float)
        probs_all = []
        y_all = []

        # Per-fold training
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit model
            model.fit(X_train, y_train)

            # Predict labels + probs (if available)
            y_pred = model.predict(X_test)
            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:
                    y_proba = None

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            if y_proba is not None:
                try:
                    # Use label_binarize on y_test for AUC
                    y_test_b = label_binarize(y_test, classes=np.arange(len(classes)))
                    auc = roc_auc_score(y_test_b, y_proba, multi_class="ovr", average="weighted")
                except Exception:
                    auc = np.nan
            else:
                auc = np.nan

            accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1); aucs.append(auc)

            # confusion matrix counts
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))
            agg_cm += cm

            if y_proba is not None:
                probs_all.append(y_proba)
                y_all.append(y_test)

            print(f"    Fold {fold_idx}: acc={acc:.4f}, f1={f1:.4f}, auc={np.nan if np.isnan(auc) else auc:.4f}")

        # Aggregate fold metrics
        acc_mean, acc_std = mean_std_safe(accs)
        prec_mean, prec_std = mean_std_safe(precs)
        rec_mean, rec_std = mean_std_safe(recs)
        f1_mean, f1_std = mean_std_safe(f1s)
        auc_mean, auc_std = mean_std_safe([v for v in aucs if not np.isnan(v)])

        # Normalize aggregated confusion matrix
        total_samples = agg_cm.sum()
        agg_cm_norm = agg_cm / total_samples if total_samples > 0 else agg_cm

        # Save per-file per-model result
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

        # Save per-file aggregated confusion matrix figure
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(agg_cm_norm, annot=True, fmt=".3f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Aggregated Confusion Matrix\n{model_name} - {file_name} (CV k={n_splits})")
        plt.tight_layout()
        cm_fname = os.path.join(results_dir, f"cm_{model_name.replace(' ','_')}_{file_name}.png")
        fig.savefig(cm_fname, dpi=300)
        plt.close(fig)

        # Aggregate feature importance if available
        try:
            if hasattr(model, "feature_importances_"):
                importances = np.array(model.feature_importances_, dtype=float)
                feature_importances_acc[model_name].append(importances)
        except Exception:
            pass

        # Concatenate probs for ROC across folds and save ROC data for file+model
        if len(probs_all) > 0:
            Y_concat = np.concatenate(y_all, axis=0)
            P_concat = np.vstack(probs_all)
            # Save per-file ROC arrays (npz)
            npz_path = os.path.join(results_dir, f"roc_data_{model_name.replace(' ','_')}_{file_name}.npz")
            np.savez(npz_path, y=Y_concat, proba=P_concat)

# Save results CSV
results_df = pd.DataFrame(all_results)
csv_path = os.path.join(results_dir, "cloud_model_cv_summary_full.csv")
results_df.to_csv(csv_path, index=False)
print("\nSaved cloud CV results CSV:", csv_path)

# ------------------ Summary across files (per model) ------------------
metrics_cols = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "auc_mean"]
summary_by_model = results_df.groupby("model")[metrics_cols].agg(['mean','std']).reset_index()
# Flatten multiindex columns
summary_by_model.columns = ['_'.join(col).strip('_') for col in summary_by_model.columns.values]
summary_by_model_path = os.path.join(results_dir, "cloud_summary_by_model_mean_std.csv")
summary_by_model.to_csv(summary_by_model_path, index=False)
print("Saved summary per model (mean/std across files):", summary_by_model_path)

# ------------------ Plots: combined and separate ------------------
# Prepare plotting DataFrame: average metrics per model (edge NOT included here; cloud only)
plot_df = results_df.groupby("model")[metrics_cols].mean().reset_index()
plot_df = plot_df.rename(columns={
    "accuracy_mean":"Accuracy",
    "precision_mean":"Precision",
    "recall_mean":"Recall",
    "f1_mean":"F1",
    "auc_mean":"AUC"
})

# Combined bar chart (cloud models)
metrics = ["Accuracy","Precision","Recall","F1","AUC"]
labels = plot_df["model"].tolist()
vals = plot_df[metrics].values

fig, ax = plt.subplots(figsize=(10,5))
n_groups, n_metrics = vals.shape[0], vals.shape[1]
width = 0.13
indices = np.arange(n_groups)
for j in range(n_metrics):
    ax.bar(indices + (j - n_metrics/2)*width + width/2, vals[:,j], width, label=metrics[j])
ax.set_xticks(indices)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel("Metric value")
ax.set_ylim(0,1.05)
ax.set_title("Cloud models - mean metrics (averaged across files)")
ax.legend(loc='upper left', bbox_to_anchor=(1.01,1), frameon=False)
plt.tight_layout()
combined_cloud_path = os.path.join(results_dir, "fig_combined_cloud_metrics.png")
fig.savefig(combined_cloud_path, dpi=300)
plt.close(fig)
print("Saved combined cloud metrics chart:", combined_cloud_path)

# Separate charts per metric
separate_paths = {}
for metric in metrics:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(labels, plot_df[metric].values)
    ax.set_ylim(0,1.05)
    ax.set_ylabel(metric)
    ax.set_title(f"Cloud models - {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    p = os.path.join(results_dir, f"fig_cloud_{metric.lower()}.png")
    fig.savefig(p, dpi=300)
    separate_paths[metric] = p
    plt.close(fig)
print("Saved separate cloud metric charts:", separate_paths)

# ------------------ Aggregated feature importances (mean across files) ------------------
for model_name, flist in feature_importances_acc.items():
    if len(flist) == 0:
        continue
    arr = np.vstack(flist)  # shape (n_files, n_features)
    mean_imp = np.nanmean(arr, axis=0)
    std_imp = np.nanstd(arr, axis=0)
    feat_df = pd.DataFrame({
        "Feature": features_to_use,
        "MeanImportance": mean_imp,
        "StdImportance": std_imp
    }).sort_values(by="MeanImportance", ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x="MeanImportance", y="Feature", data=feat_df, ax=ax, palette="Greys")
    ax.set_title(f"Aggregated Feature Importance - {model_name} (mean across files)")
    plt.tight_layout()
    pth = os.path.join(results_dir, f"feature_importance_aggregated_{model_name.replace(' ','_')}.png")
    fig.savefig(pth, dpi=300)
    plt.close(fig)
    print("Saved aggregated feature importance for", model_name, "->", pth)

# ------------------ ROC curves aggregated across files for each model (if prob data exists) ------------------
# We'll attempt to build a representative ROC by concatenating per-file prob arrays saved earlier
for model_name in models.keys():
    roc_files = sorted(glob.glob(os.path.join(results_dir, f"roc_data_{model_name.replace(' ','_')}_*.npz")))
    if len(roc_files) == 0:
        print(f"No ROC prob files for {model_name} (skipping ROC plot).")
        continue
    Ys_list, Probas_list = [], []
    for f in roc_files:
        data = np.load(f)
        Ys_list.append(data['y'])
        Probas_list.append(data['proba'])
    Ys = np.concatenate(Ys_list, axis=0)
    Probas = np.vstack(Probas_list)
    try:
        auc_val = roc_auc_score(label_binarize(Ys, classes=np.arange(len(classes))), Probas, multi_class="ovr", average="weighted")
    except Exception:
        auc_val = np.nan
    # Simple illustrative ROC: plot diagonal + mention AUC (detailed per-class ROC curves omitted here)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot([0,1],[0,1],'--', linewidth=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (aggregated) - {model_name} - AUC ≈ {auc_val:.4f}")
    roc_out = os.path.join(results_dir, f"roc_aggregated_{model_name.replace(' ','_')}.png")
    fig.savefig(roc_out, dpi=300)
    plt.close(fig)
    print("Saved aggregated ROC approx for", model_name, "->", roc_out)

print("\nCloud CV processing complete. Results in:", results_dir)
print("Main CSV:", csv_path)
print("Summary per model (mean/std):", summary_by_model_path)
