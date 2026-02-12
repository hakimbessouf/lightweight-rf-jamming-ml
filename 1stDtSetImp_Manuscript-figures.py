#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manuscript Figure Generation Script
Generates publication-ready figures from edge_model_cv_summary_full.csv
for RF jamming detection research paper

Figures generated:
1. Performance comparison bar chart (all metrics)
2. Individual metric bar charts
3. Aggregated confusion matrices per model
4. Model comparison heatmap
5. Performance variance visualization
6. Metric correlation matrix
7. Statistical summary table
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
# Input file path
csv_file = "D:\dataset\extracted_features_w1\merged_parts\scaled\sampled_60\cloud_model_cv_summary_adjusted.csv"  # Change this to your file path

# Output directory
output_dir = "manuscript_figures"
os.makedirs(output_dir, exist_ok=True)

# IEEE-style plotting configuration
plt.style.use('classic')
sns.set_palette("Set2")
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True
})

# Class labels (adjust if needed)
class_labels = ["Gaussian", "Nojamming", "Sine"]

# ===================== LOAD DATA =====================
print(f"Loading data from {csv_file}...")
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} records")
print(f"Models found: {df['model'].unique()}")
print(f"Files processed: {df['file'].unique() if 'file' in df.columns else 'N/A'}")

# Aggregate across files per model
metrics_to_aggregate = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'auc_mean',
                        'accuracy_std', 'precision_std', 'recall_std', 'f1_std', 'auc_std']

# Calculate overall statistics per model
model_stats = df.groupby('model').agg({
    'accuracy_mean': ['mean', 'std', 'min', 'max'],
    'precision_mean': ['mean', 'std', 'min', 'max'],
    'recall_mean': ['mean', 'std', 'min', 'max'],
    'f1_mean': ['mean', 'std', 'min', 'max'],
    'auc_mean': ['mean', 'std', 'min', 'max']
}).reset_index()

model_stats.columns = ['_'.join(col).strip('_') for col in model_stats.columns.values]
model_stats.columns = ['Model', 
                       'Accuracy_Mean', 'Accuracy_Std', 'Accuracy_Min', 'Accuracy_Max',
                       'Precision_Mean', 'Precision_Std', 'Precision_Min', 'Precision_Max',
                       'Recall_Mean', 'Recall_Std', 'Recall_Min', 'Recall_Max',
                       'F1_Mean', 'F1_Std', 'F1_Min', 'F1_Max',
                       'AUC_Mean', 'AUC_Std', 'AUC_Min', 'AUC_Max']

print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)
print(model_stats.to_string(index=False))

# ===================== FIGURE 1: COMBINED METRICS BAR CHART =====================
print("\nGenerating Figure 1: Combined Metrics Comparison...")

fig, ax = plt.subplots(figsize=(10, 5))

metrics = ['Accuracy_Mean', 'Precision_Mean', 'Recall_Mean', 'F1_Mean', 'AUC_Mean']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
models = model_stats['Model'].tolist()

x = np.arange(len(models))
width = 0.15

colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    values = model_stats[metric].values
    ax.bar(x + i*width - 2*width, values, width, label=label, color=colors[i], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Machine Learning Models', fontweight='bold')
ax.set_ylabel('Performance Metric Value', fontweight='bold')
ax.set_title('Performance Comparison of Lightweight Edge ML Models', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0, ha='center')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', framealpha=0.9, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig1_Combined_Metrics_Comparison.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ===================== FIGURE 2: INDIVIDUAL METRIC BAR CHARTS =====================
print("\nGenerating Figure 2: Individual Metric Charts...")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    
    values = model_stats[metric].values
    std_metric = metric.replace('_Mean', '_Std')
    errors = model_stats[std_metric].values if std_metric in model_stats.columns else None
    
    bars = ax.bar(models, values, color=colors[idx], edgecolor='black', linewidth=0.7, alpha=0.8)
    
    if errors is not None:
        ax.errorbar(models, values, yerr=errors, fmt='none', ecolor='black', 
                   capsize=4, capthick=1.5, linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel(label, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title(f'{label} Performance', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.tick_params(axis='x', rotation=15)

# Remove extra subplot
axes[-1].remove()

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig2_Individual_Metrics.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ===================== FIGURE 3: CONFUSION MATRICES =====================
print("\nGenerating Figure 3: Aggregated Confusion Matrices...")

# Parse and aggregate confusion matrices
n_models = len(model_stats)
fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 3.5))
if n_models == 1:
    axes = [axes]

for idx, model_name in enumerate(models):
    model_data = df[df['model'] == model_name]
    
    # Aggregate confusion matrices across files
    cms = []
    for _, row in model_data.iterrows():
        if pd.notna(row.get('confusion_matrix_norm')):
            try:
                cm = eval(row['confusion_matrix_norm'])
                cms.append(np.array(cm))
            except:
                pass
    
    if len(cms) > 0:
        # Average confusion matrices
        avg_cm = np.mean(cms, axis=0)
        
        ax = axes[idx]
        im = ax.imshow(avg_cm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('Normalized Frequency', rotation=270, labelpad=15, fontweight='bold')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)
        
        # Add text annotations
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                text = ax.text(j, i, f'{avg_cm[i, j]:.3f}',
                             ha="center", va="center", 
                             color="white" if avg_cm[i, j] > 0.5 else "black",
                             fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig3_Confusion_Matrices.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ===================== FIGURE 4: PERFORMANCE HEATMAP =====================
print("\nGenerating Figure 4: Model Performance Heatmap...")

fig, ax = plt.subplots(figsize=(8, 5))

# Prepare data for heatmap
heatmap_data = model_stats[['Model'] + metrics].set_index('Model')
heatmap_data.columns = metric_labels

sns.heatmap(heatmap_data.T, annot=True, fmt='.4f', cmap='YlGnBu', 
            cbar_kws={'label': 'Performance Score'}, 
            linewidths=0.5, linecolor='gray', ax=ax,
            vmin=0, vmax=1)

ax.set_xlabel('Machine Learning Models', fontweight='bold')
ax.set_ylabel('Performance Metrics', fontweight='bold')
ax.set_title('Performance Heatmap: Model × Metric', fontweight='bold', pad=15)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig4_Performance_Heatmap.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ===================== FIGURE 5: PERFORMANCE VARIANCE =====================
print("\nGenerating Figure 5: Performance Variance Analysis...")

fig, ax = plt.subplots(figsize=(10, 6))

std_metrics = ['Accuracy_Std', 'Precision_Std', 'Recall_Std', 'F1_Std', 'AUC_Std']
x = np.arange(len(models))
width = 0.15

for i, (std_metric, label) in enumerate(zip(std_metrics, metric_labels)):
    if std_metric in model_stats.columns:
        values = model_stats[std_metric].values
        ax.bar(x + i*width - 2*width, values, width, label=label, 
               color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_xlabel('Machine Learning Models', fontweight='bold')
ax.set_ylabel('Standard Deviation', fontweight='bold')
ax.set_title('Performance Stability Analysis: Standard Deviation Across Folds', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0, ha='center')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig5_Performance_Variance.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ===================== FIGURE 6: BOX PLOT COMPARISON =====================
print("\nGenerating Figure 6: Box Plot Comparison...")

fig, axes = plt.subplots(1, 5, figsize=(16, 4))

for idx, (metric_col, label) in enumerate(zip(['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'auc_mean'],
                                               metric_labels)):
    ax = axes[idx]
    
    # Prepare data for box plot
    box_data = [df[df['model'] == model][metric_col].values for model in models]
    
    bp = ax.boxplot(box_data, labels=models, patch_artist=True,
                    boxprops=dict(facecolor=colors[idx], alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1))
    
    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(label, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.tick_params(axis='x', rotation=15)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig6_BoxPlot_Comparison.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ===================== FIGURE 7: RADAR CHART =====================
print("\nGenerating Figure 7: Radar Chart Performance...")

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for idx, model in enumerate(models):
    values = model_stats[model_stats['Model'] == model][metrics].values[0].tolist()
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx % len(colors)])
    ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title('Performance Radar Chart: Model Comparison', fontweight='bold', 
             pad=20, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9, edgecolor='black')

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig7_Radar_Chart.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ===================== TABLE 1: STATISTICAL SUMMARY =====================
print("\nGenerating Table 1: Statistical Summary...")

# Create formatted table
table_data = []
for _, row in model_stats.iterrows():
    table_data.append({
        'Model': row['Model'],
        'Accuracy': f"{row['Accuracy_Mean']:.4f} ± {row['Accuracy_Std']:.4f}",
        'Precision': f"{row['Precision_Mean']:.4f} ± {row['Precision_Std']:.4f}",
        'Recall': f"{row['Recall_Mean']:.4f} ± {row['Recall_Std']:.4f}",
        'F1-Score': f"{row['F1_Mean']:.4f} ± {row['F1_Std']:.4f}",
        'AUC': f"{row['AUC_Mean']:.4f} ± {row['AUC_Std']:.4f}"
    })

table_df = pd.DataFrame(table_data)

# Save as CSV
table_csv = os.path.join(output_dir, "Table1_Performance_Summary.csv")
table_df.to_csv(table_csv, index=False)
print(f"✓ Saved: {table_csv}")

# Create visual table
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=table_df.values,
                colLabels=table_df.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.17, 0.17, 0.17, 0.17, 0.17])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(len(table_df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style cells
for i in range(1, len(table_df) + 1):
    for j in range(len(table_df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

plt.title('Table 1: Cross-Validation Performance Summary (Mean ± Std)', 
          fontweight='bold', fontsize=12, pad=20)
plt.tight_layout()

fig_path = os.path.join(output_dir, "Table1_Performance_Summary.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ===================== FIGURE 8: MIN-MAX RANGE VISUALIZATION =====================
print("\nGenerating Figure 8: Min-Max Performance Range...")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.15

# Map display labels to actual column names
column_mapping = {
    'Accuracy': 'Accuracy',
    'Precision': 'Precision',
    'Recall': 'Recall',
    'F1-Score': 'F1',
    'AUC': 'AUC'
}

for i, (metric_label) in enumerate(metric_labels):
    base_name = column_mapping[metric_label]
    mean_col = f"{base_name}_Mean"
    min_col = f"{base_name}_Min"
    max_col = f"{base_name}_Max"
    
    means = model_stats[mean_col].values
    mins = model_stats[min_col].values
    maxs = model_stats[max_col].values
    
    positions = x + i*width - 2*width
    
    # Plot error bars showing min-max range
    ax.errorbar(positions, means, 
                yerr=[means - mins, maxs - means],
                fmt='o', markersize=8, capsize=5, capthick=2,
                label=metric_label, color=colors[i], linewidth=2)

ax.set_xlabel('Machine Learning Models', fontweight='bold')
ax.set_ylabel('Performance Score', fontweight='bold')
ax.set_title('Performance Range Analysis: Mean with Min-Max Bounds', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0, ha='center')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', framealpha=0.9, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
fig_path = os.path.join(output_dir, "Fig8_MinMax_Range.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# ===================== SUMMARY REPORT =====================
print("\n" + "="*60)
print("FIGURE GENERATION COMPLETE")
print("="*60)
print(f"All figures saved to: {output_dir}/")
print("\nGenerated files:")
print("  - Fig1_Combined_Metrics_Comparison.png/.pdf")
print("  - Fig2_Individual_Metrics.png/.pdf")
print("  - Fig3_Confusion_Matrices.png/.pdf")
print("  - Fig4_Performance_Heatmap.png/.pdf")
print("  - Fig5_Performance_Variance.png/.pdf")
print("  - Fig6_BoxPlot_Comparison.png/.pdf")
print("  - Fig7_Radar_Chart.png/.pdf")
print("  - Fig8_MinMax_Range.png/.pdf")
print("  - Table1_Performance_Summary.png/.pdf/.csv")
print("\nAll figures are publication-ready at 300 DPI in both PNG and PDF formats.")
print("="*60)