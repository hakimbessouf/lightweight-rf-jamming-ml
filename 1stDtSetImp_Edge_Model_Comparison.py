# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:59:41 2025

@author: Alienware
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =============================
# 📍 CONFIGURATION GÉNÉRALE
# =============================
data_dir = r"D:\dataset\extracted_features_w1\merged_parts\scaled"
files = [f for f in os.listdir(data_dir) if f.startswith("standardized_scaled_merged_part") and f.endswith(".csv")]

output_dir = os.path.join(data_dir, "Edge_Results_Comparison")
os.makedirs(output_dir, exist_ok=True)

features = ["Amplitude", "Phase", "Power", "Instantaneous_Frequency"]

label_encoder = LabelEncoder()
classes = ["Gaussian", "Nojamming", "Sine"]
label_encoder.fit(classes)

# =============================
# 🧠 Modèles à comparer
# =============================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, C=5, gamma='scale', random_state=42)
}

results = []

# =============================
# 📊 ENTRAÎNEMENT & ÉVALUATION
# =============================
file_path = os.path.join(data_dir, files[0])
print(f"🔹 Chargement du fichier : {file_path}")
df = pd.read_csv(file_path)

# Standardisation des features (important pour le SVM)
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = label_encoder.transform(df["Condition"])
y_bin = label_binarize(y, classes=[0, 1, 2])  # Pour ROC multi-classes

# Découpage train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for model_name, model in models.items():
    print(f"\n🚀 Entraînement du modèle : {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc_score = roc_auc_score(y_bin, model.predict_proba(X) if hasattr(model, "predict_proba") else np.zeros_like(y_bin), multi_class="ovr")

    results.append([model_name, acc, prec, rec, f1, auc_score])

    print(f"   ➝ Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # =============================
    # 🔹 MATRICE DE CONFUSION
    # =============================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ConfusionMatrix_{model_name}.png"))
    plt.close()

    # =============================
    # 🔹 ROC MULTICLASSE
    # =============================
    plt.figure(figsize=(7, 6))
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    for i, class_name in enumerate(label_encoder.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ROC_Multiclass_{model_name}.png"))
    plt.close()

# =============================
# 📈 COMPARAISON DES MODÈLES
# =============================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"])
print("\n✅ Résumé des performances Edge :")
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
            x="Metric", y="Score", hue="Model")
plt.title("📊 Performance Comparison of Edge Models")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Edge_Model_Comparison.png"))
plt.show()

print(f"\n📁 Tous les graphiques sauvegardés dans : {output_dir}")
