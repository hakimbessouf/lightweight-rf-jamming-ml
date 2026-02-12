# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 21:13:23 2025

@author: Alienware
"""

# -*- coding: utf-8 -*-
"""
Enhanced Cloud Layer RF Jamming Detection - Detailed Results
Generates comprehensive results similar to edge layer format:
- cloud_model_cv_summary_full.csv (detailed per-file metrics)
- Aggregated confusion matrices
- ROC curves
- All performance visualizations

Models: Random Forest, XGBoost, LightGBM (Cloud-level heavy models)
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
import warnings
warnings.filterwarnings("ignore")

# ========================= CONFIGURATION =========================
# Data directory containing Sampled_part_*.csv files
data_dir = r"D:\dataset\extracted_features_w1\merged_parts\scaled\sampled_60"

# Output directory
output_dir = os.path.join(data_dir, "cloud_results_detailed_cv")
os.makedirs(output_dir, exist_ok=True)

# Features used for training
features_to_use = ["Amplitude", "Phase","Power", "Instantaneous_Frequency", "Kurtosis_I"]

# Label configuration
label_column = "Condition"
label_encoder = LabelEncoder()
label_encoder.fit(["Gaussian", "Nojamming", "Sine"])
classes = label_encoder.classes_

# CV parameters
n_splits = 5
random_state = 42

# Cloud Models (Heavy/High-Performance)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=random_state),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                              max_depth=6, learning_rate=0.1, n_estimators=200, 
                              random_state=random_state),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, 
                                random_state=random_state, verbose=-1)
}

# ========================= SETUP PLOTTING STYLE =========================
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

# Get all sampled files
file_pattern = os.path.join(data_dir, "Sampled_part_*.csv")
files = sorted(glob.glob(file_pattern))

if len(files) == 0:
    raise FileNotFoundError(f"No files found matching {file_pattern}")

print(f"\n📂 Found {len(files)} data files")
print(f"Files: {[os.path.basename(f) for f in files[:3]]}{'...' if len(files) > 3 else ''}")

# Storage for detailed results (similar to edge format)
all_results = []

# ========================= PER-FILE CROSS-VALIDATION =========================
for file_path in files:
    file_name = os.path.basename(file_path)
    print(f"\n{'='*60}")
    print(f"Processing: {file_name}")
    print(f"{'='*60}")
    
    # Load file
    df = pd.read_csv(file_path)
    
    # Validate features
    for feat in features_to_use:
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' not found in {file_name}")
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {file_name}")
    
    # Prepare X, y
    X = df[features_to_use].values
    y_raw = df[label_column].values
    y = label_encoder.transform(y_raw)
    
    # Binarize for AUC calculation
    y_binarized = label_binarize(y, classes=np.arange(len(classes)))
    
    # CV splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # ========================= TRAIN EACH MODEL =========================
    for model_name, model in models.items():
        print(f"\n  🧠 Model: {model_name}")
        
        fold_metrics = {
            "accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []
        }
        agg_cm = np.zeros((len(classes), len(classes)), dtype=np.float64)
        
        # Store probabilities for ROC
        probs_list = []
        true_list = []
        
        # Per-fold training
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"    ⚠️  Fold {fold_idx} training failed: {e}")
                continue
            
            # Predictions
            y_pred = model.predict(X_val)
            
            # Probabilities for AUC
            try:
                y_proba = model.predict_proba(X_val)
            except:
                y_proba = None
            
            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
            
            # AUC
            if y_proba is not None:
                try:
                    y_val_bin = label_binarize(y_val, classes=np.arange(len(classes)))
                    auc = roc_auc_score(y_val_bin, y_proba, multi_class="ovr", average="weighted")
                except:
                    auc = np.nan
                
                # Store for ROC curve
                probs_list.append(y_proba)
                true_list.append(y_val)
            else:
                auc = np.nan
            
            # Save fold metrics
            fold_metrics["accuracy"].append(acc)
            fold_metrics["precision"].append(prec)
            fold_metrics["recall"].append(rec)
            fold_metrics["f1"].append(f1)
            fold_metrics["auc"].append(auc)
            
            # Aggregate confusion matrix
            cm_fold = confusion_matrix(y_val, y_pred, labels=np.arange(len(classes)))
            agg_cm += cm_fold
            
            print(f"    ✓ Fold {fold_idx}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        
        # Calculate statistics across folds
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
        
        # Normalize confusion matrix
        agg_cm_norm = agg_cm / agg_cm.sum() if agg_cm.sum() > 0 else agg_cm
        
        print(f"\n  📊 Summary for {model_name} on {file_name}:")
        print(f"     Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"     Precision: {prec_mean:.4f} ± {prec_std:.4f}")
        print(f"     Recall:    {rec_mean:.4f} ± {rec_std:.4f}")
        print(f"     F1-Score:  {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"     AUC:       {auc_mean:.4f} ± {auc_std:.4f}")
        
        # Store results
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
        
        # Save confusion matrix figure (per file, per model)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(agg_cm_norm, annot=True, fmt=".3f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted", fontweight='bold')
        ax.set_ylabel("Actual", fontweight='bold')
        ax.set_title(f"Confusion Matrix\n{model_name} - {file_name} (CV k={n_splits})", 
                     fontweight='bold')
        plt.tight_layout()
        cm_fname = os.path.join(output_dir, 
                                f"cm_{model_name.replace(' ', '_')}_{file_name}.png")
        fig.savefig(cm_fname, dpi=300)
        plt.close(fig)
        
        # Save ROC data if available
        if len(probs_list) > 0:
            Ys = np.concatenate(true_list, axis=0)
            Probas = np.vstack(probs_list)
            roc_file = os.path.join(output_dir, 
                                   f"roc_data_{model_name.replace(' ', '_')}_{file_name}.npz")
            np.savez(roc_file, y=Ys, proba=Probas)

# ========================= SAVE DETAILED RESULTS =========================
print("\n" + "="*60)
print("SAVING DETAILED RESULTS")
print("="*60)

results_df = pd.DataFrame(all_results)
results_csv = os.path.join(output_dir, "cloud_model_cv_summary_full.csv")
results_df.to_csv(results_csv, index=False)
print(f"✓ Saved detailed CSV: {results_csv}")

# ========================= GENERATE SUMMARY STATISTICS =========================
# Aggregate across files per model
model_stats = results_df.groupby('model').agg({
    'accuracy_mean': ['mean', 'std', 'min', 'max'],
    'precision_mean': ['mean', 'std', 'min', 'max'],
    'recall_mean': ['mean', 'std', 'min', 'max'],
    'f1_mean': ['mean', 'std', 'min', 'max'],
    'auc_mean': ['mean', 'std', 'min', 'max']
}).reset_index()

# Flatten column names
model_stats.columns = ['Model',
                       'Accuracy_Mean', 'Accuracy_Std', 'Accuracy_Min', 'Accuracy_Max',
                       'Precision_Mean', 'Precision_Std', 'Precision_Min', 'Precision_Max',
                       'Recall_Mean', 'Recall_Std', 'Recall_Min', 'Recall_Max',
                       'F1_Mean', 'F1_Std', 'F1_Min', 'F1_Max',
                       'AUC_Mean', 'AUC_Std', 'AUC_Min', 'AUC_Max']

summary_csv = os.path.join(output_dir, "cloud_model_summary_statistics.csv")
model_stats.to_csv(summary_csv, index=False)
print(f"✓ Saved summary statistics: {summary_csv}")

print("\n" + "="*60)
print("FINAL CLOUD MODEL PERFORMANCE SUMMARY")
print("="*60)
print(model_stats.to_string(index=False))

# ========================= GENERATE FIGURES =========================
print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)

# Figure 1: Combined Metrics Bar Chart
print("Generating Figure 1: Combined Metrics...")
fig, ax = plt.subplots(figsize=(10, 5))

metrics = ['Accuracy_Mean', 'Precision_Mean', 'Recall_Mean', 'F1_Mean', 'AUC_Mean']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
models_list = model_stats['Model'].tolist()

x = np.arange(len(models_list))
width = 0.15
colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    values = model_stats[metric].values
    ax.bar(x + i*width - 2*width, values, width, label=label, 
           color=colors[i], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Cloud Models', fontweight='bold')
ax.set_ylabel('Performance Metric Value', fontweight='bold')
ax.set_title('Cloud Layer Performance: Heavy ML Models', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=0, ha='center')
ax.set_ylim(0.85, 1.0)
ax.legend(loc='lower right', framealpha=0.9, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig1_Cloud_Combined_Metrics.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# Figure 2: Performance Heatmap
print("Generating Figure 2: Performance Heatmap...")
fig, ax = plt.subplots(figsize=(8, 5))

heatmap_data = model_stats[['Model'] + metrics].set_index('Model')
heatmap_data.columns = metric_labels

sns.heatmap(heatmap_data.T, annot=True, fmt='.4f', cmap='YlGnBu',
            cbar_kws={'label': 'Performance Score'},
            linewidths=0.5, linecolor='gray', ax=ax,
            vmin=0.85, vmax=1.0)

ax.set_xlabel('Cloud Models', fontweight='bold')
ax.set_ylabel('Performance Metrics', fontweight='bold')
ax.set_title('Cloud Performance Heatmap', fontweight='bold', pad=15)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig2_Cloud_Heatmap.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# Figure 3: Aggregated Confusion Matrices
print("Generating Figure 3: Aggregated Confusion Matrices...")
for model_name in results_df['model'].unique():
    model_data = results_df[results_df['model'] == model_name]
    
    # Sum confusion matrices across all files
    cms = []
    for _, row in model_data.iterrows():
        try:
            cm = np.array(row['confusion_matrix_counts'])
            cms.append(cm)
        except:
            pass
    
    if len(cms) > 0:
        total_cm = np.sum(cms, axis=0)
        norm_cm = total_cm / total_cm.sum()
        
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(norm_cm, annot=True, fmt=".3f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted", fontweight='bold')
        ax.set_ylabel("Actual", fontweight='bold')
        ax.set_title(f"Aggregated Confusion Matrix\n{model_name} (Cloud Layer)",
                     fontweight='bold')
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir,
                               f"Fig3_Aggregated_CM_{model_name.replace(' ', '_')}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
        plt.close()

# Figure 4: Box Plot Comparison
print("Generating Figure 4: Box Plot Comparison...")
fig, axes = plt.subplots(1, 5, figsize=(16, 4))

for idx, (metric_col, label) in enumerate(zip(['accuracy_mean', 'precision_mean', 
                                                'recall_mean', 'f1_mean', 'auc_mean'],
                                               metric_labels)):
    ax = axes[idx]
    
    box_data = [results_df[results_df['model'] == m][metric_col].values 
                for m in models_list]
    
    bp = ax.boxplot(box_data, labels=models_list, patch_artist=True,
                    boxprops=dict(facecolor=colors[idx], alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1))
    
    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(label, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.tick_params(axis='x', rotation=15)
    ax.set_ylim(0.85, 1.0)

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig4_Cloud_BoxPlot.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# Figure 5: Performance Variance
print("Generating Figure 5: Performance Variance...")
fig, ax = plt.subplots(figsize=(10, 6))

std_metrics = ['Accuracy_Std', 'Precision_Std', 'Recall_Std', 'F1_Std', 'AUC_Std']
x = np.arange(len(models_list))
width = 0.15

for i, (std_metric, label) in enumerate(zip(std_metrics, metric_labels)):
    values = model_stats[std_metric].values
    ax.bar(x + i*width - 2*width, values, width, label=label,
           color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_xlabel('Cloud Models', fontweight='bold')
ax.set_ylabel('Standard Deviation', fontweight='bold')
ax.set_title('Cloud Layer Stability: Performance Variance', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=0, ha='center')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig5_Cloud_Variance.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ========================= FINAL SUMMARY =========================
print("\n" + "="*60)
print("✅ ALL PROCESSING COMPLETE")
print("="*60)
print(f"\nResults directory: {output_dir}")
print("\nGenerated files:")
print("  CSV Files:")
print(f"    - cloud_model_cv_summary_full.csv (detailed per-file results)")
print(f"    - cloud_model_summary_statistics.csv (aggregated statistics)")
print("  Figures:")
print("    - Fig1_Cloud_Combined_Metrics.png/.pdf")
print("    - Fig2_Cloud_Heatmap.png/.pdf")
print("    - Fig3_Aggregated_CM_*.png/.pdf (per model)")
print("    - Fig4_Cloud_BoxPlot.png/.pdf")
print("    - Fig5_Cloud_Variance.png/.pdf")
print("    - Individual confusion matrices per file/model")
print("    - ROC data files (.npz) for each model/file combination")
print("\n" + "="*60)