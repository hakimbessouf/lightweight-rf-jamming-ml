# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 23:57:52 2025

@author: Alienware
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate publication-quality figures (IEEE style) for Edge–Cloud complementary framework
Figures generated:
Fig.1  – Edge models (Random Forest, Extra Trees, KNN)
Fig.2  – Cloud models (Random Forest, XGBoost, LightGBM)
Fig.3  – Confusion matrices (Edge RF & Cloud XGBoost)
Fig.4  – Comparison Edge vs Cloud (mean metrics)
Fig.5  – Conceptual Edge→Cloud pipeline
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------------------------------------------------
# 📍 PATHS
# ------------------------------------------------------------------
edge_csv = r"D:\dataset\extracted_features_w1\merged_parts\scaled\sampled_60\edge_results_cv\edge_model_cv_summary_full.csv"
#cloud_csv = r"D:\dataset\extracted_features_w1\merged_parts\scaled\sampled_60\cloud_results_interfile_cv\cloud_results_summary.csv"
output_dir = r"D:\path\to\figures_output"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------
# 📊 LOAD DATA
# ------------------------------------------------------------------
edge = pd.read_csv(edge_csv)
cloud = pd.read_csv(cloud_csv)

metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "auc_mean"]

# Mean per model (Edge / Cloud)
edge_summary = edge.groupby("model")[metrics].mean().reset_index()
cloud_summary = cloud.groupby("model")[metrics].mean().reset_index()

# ------------------------------------------------------------------
# 🎨 STYLE (IEEE-like)
# ------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "figure.dpi": 300
})
palette_edge = sns.color_palette(["#1f77b4", "#ff7f0e", "#2ca02c"])  # blue, orange, green
palette_cloud = sns.color_palette(["#1f77b4", "#d62728", "#9467bd"])  # blue, red, violet

# ------------------------------------------------------------------
# 📘 FIGURE 1: Edge models
# ------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(7,4))
edge_melted = edge_summary.melt(id_vars="model", var_name="Metric", value_name="Score")
sns.barplot(data=edge_melted, x="Metric", y="Score", hue="model", ax=ax1, palette=palette_edge)
ax1.set_ylim(0.6, 1.0)
ax1.set_ylabel("Score")
ax1.set_xlabel("Metric")
ax1.set_title("Fig. 1 – Edge Models Performance (mean across files)")
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.3f", fontsize=8)
plt.legend(title="Model", loc="upper left")
plt.tight_layout()
fig1.savefig(os.path.join(output_dir, "Fig1_Edge_Models.png"), dpi=300)

# ------------------------------------------------------------------
# 📘 FIGURE 2: Cloud models
# ------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7,4))
cloud_melted = cloud_summary.melt(id_vars="model", var_name="Metric", value_name="Score")
sns.barplot(data=cloud_melted, x="Metric", y="Score", hue="model", ax=ax2, palette=palette_cloud)
ax2.set_ylim(0.6, 1.0)
ax2.set_ylabel("Score")
ax2.set_xlabel("Metric")
ax2.set_title("Fig. 2 – Cloud Models Performance (mean across files)")
for container in ax2.containers:
    ax2.bar_label(container, fmt="%.3f", fontsize=8)
plt.legend(title="Model", loc="upper left")
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, "Fig2_Cloud_Models.png"), dpi=300)

# ------------------------------------------------------------------
# 📘 FIGURE 3: Confusion matrices (Edge RF & Cloud XGBoost)
# ------------------------------------------------------------------
import seaborn as sns
import numpy as np

# Extract example confusion matrices (mean normalized)
def extract_cm(df, model):
    cms = df[df["model"] == model]["confusion_matrix_norm"].dropna()
    if len(cms) == 0:
        return None
    cm_avg = np.mean([np.array(eval(cm)) for cm in cms], axis=0)
    return cm_avg

cm_edge = extract_cm(edge, "Random Forest")
cm_cloud = extract_cm(cloud, "XGBoost")

fig3, axes = plt.subplots(1, 2, figsize=(8,4))
sns.heatmap(cm_edge, annot=True, fmt=".2f", cmap="Blues", ax=axes[0])
axes[0].set_title("Edge – Random Forest")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
sns.heatmap(cm_cloud, annot=True, fmt=".2f", cmap="Oranges", ax=axes[1])
axes[1].set_title("Cloud – XGBoost")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
fig3.suptitle("Fig. 3 – Normalized Confusion Matrices (Edge vs Cloud)")
plt.tight_layout()
fig3.savefig(os.path.join(output_dir, "Fig3_ConfusionMatrices.png"), dpi=300)

# ------------------------------------------------------------------
# 📘 FIGURE 4: Comparison Edge vs Cloud (means)
# ------------------------------------------------------------------
edge_mean = edge_summary[metrics].mean().mean()
cloud_mean = cloud_summary[metrics].mean().mean()
comparison_df = pd.DataFrame({
    "Layer": ["Edge", "Cloud"],
    "MeanScore": [edge_mean, cloud_mean]
})

fig4, ax4 = plt.subplots(figsize=(5,4))
sns.barplot(data=comparison_df, x="Layer", y="MeanScore", palette=["#1f77b4", "#d62728"], ax=ax4)
ax4.set_ylim(0.6, 1.0)
ax4.set_ylabel("Average Performance")
ax4.set_title("Fig. 4 – Global Comparison Edge vs Cloud")
for container in ax4.containers:
    ax4.bar_label(container, fmt="%.3f", fontsize=10)
plt.tight_layout()
fig4.savefig(os.path.join(output_dir, "Fig4_Edge_vs_Cloud.png"), dpi=300)

# ------------------------------------------------------------------
# 📘 FIGURE 5: Pipeline (Edge→Cloud Complementarity)
# ------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(8,3))
ax5.axis("off")
boxes = [
    ("Edge Node\nLightweight Model\n(Real-Time Detection)", 0.1, 0.5, "#1f77b4"),
    ("Suspicious Signal →\nFeature Extraction", 0.45, 0.5, "#cccccc"),
    ("Cloud Validation\nHeavy Model\n(Confirmation)", 0.8, 0.5, "#d62728")
]
for text, x, y, color in boxes:
    ax5.add_patch(plt.Rectangle((x-0.15,y-0.12),0.3,0.24,fill=True,color=color,alpha=0.2,ec='black'))
    ax5.text(x, y, text, ha='center', va='center', fontsize=10)
ax5.annotate('', xy=(0.28,0.5), xytext=(0.37,0.5), arrowprops=dict(arrowstyle='->'))
ax5.annotate('', xy=(0.58,0.5), xytext=(0.67,0.5), arrowprops=dict(arrowstyle='->'))
ax5.set_title("Fig. 5 – Complementary Edge→Cloud Processing Pipeline")
fig5.savefig(os.path.join(output_dir, "Fig5_Pipeline.png"), dpi=300, bbox_inches='tight')

# ------------------------------------------------------------------
print("\n✅ Figures generated in:", output_dir)
