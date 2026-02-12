# lightweight-rf-jamming-ml
# Lightweight Machine Learning for RF Jamming Detection: Edge–Fog–Cloud Computing Architecture

This repository contains the code for **RF jamming detection** using lightweight machine learning within an **Edge–Fog–Cloud** computing architecture. Two datasets are implemented, each with its own full pipeline for preprocessing, feature extraction, model training, and result generation.

---

## Datasets

This project uses two publicly available RF jamming datasets:

- **1st dataset – Indoor wireless jamming measurements**  
  *A Dataset of physical-layer measurements in indoor wireless jamming scenarios* (Scientific Data, 2022).  
  Link: https://www.sciencedirect.com/science/article/pii/S2352340922009763  

  This dataset provides physical‑layer measurements collected in controlled indoor wireless jamming scenarios. It includes multiple jamming types such as **Gaussian noise**, **sinusoidal jamming**, and **no‑jamming** baseline conditions.

- **2nd dataset – Weak jamming in IEEE 802.11 networks**  
  *Weak-Jamming Detection in IEEE 802.11 Networks: Techniques, Scenarios and Mobility*.  
  Link: https://www.researchgate.net/publication/392105159_Weak-Jamming_Detection_in_IEEE_80211_Networks_Techniques_Scenarios_and_Mobility  

  This dataset focuses on **weak RF jamming** in Wi‑Fi (IEEE 802.11) networks, with multiple scenarios and mobility patterns. The main detection task is **binary classification** between **Jam** and **NoJam** conditions.

The folder:

- `1st dataset implementation/` contains the pipeline built on the **indoor wireless jamming** dataset (three classes: Gaussian, Sine, Nojamming).  
- `2nd dataset implementation/` contains the pipeline built on the **weak‑jamming IEEE 802.11** dataset (two classes: Jam / NoJam), including IQ feature extraction and the Edge–Fog–Cloud models.

---

## 1st dataset implementation

The **1st dataset implementation** folder holds the original pipeline designed for the indoor wireless jamming dataset.

### Main steps

1. **Data preparation**
   - Merge raw CSV files per scenario and folder.  
   - Scale and standardize dataset chunks.  
   - Shuffle chunks to randomize sample order.  
   - Create balanced mixed sample files for training and evaluation.

2. **Edge layer**
   - Train compressed, lightweight edge models on preprocessed features.  
   - Use ultra‑light configurations to reduce computational cost while preserving accuracy.  
   - Compare edge models and generate metrics/figures.

3. **Fog and Cloud layers**
   - Train balanced fog models with moderate complexity.  
   - Train cloud‑level models (e.g., XGBoost and other heavier classifiers).  
   - Perform inter‑file and cross‑validation experiments at the cloud layer.  
   - Produce final performance summaries, including accuracy, precision, recall, F1‑score and AUC.

4. **Figure generation**
   - Generate figures used in the related manuscript (performance plots, confusion matrices, and comparison graphs).

---

## 2nd dataset implementation

The **2nd dataset implementation** folder contains the updated pipeline for the weak‑jamming IEEE 802.11 dataset, based on IQ samples and extended feature sets. It is organized to support Edge–Fog–Cloud deployment, with models saved using `joblib`.

### Main steps

1. **IQ extraction and feature engineering**
   - Extract I/Q samples from raw binary files of the dataset.  
   - Compute extended IQ features, including amplitude, phase, power, instantaneous frequency, higher‑order statistics (kurtosis, skewness), and spectral features (e.g., spectral centroid, bandwidth, flatness).  
   - Merge stitched feature files into per‑scenario or per‑part CSVs.  
   - Standardize and scale merged feature files.  
   - Clean NaN values and prepare the final `merged_features_part*_scaled_cleaned.csv` files.  
   - Optionally split all features into separate train/test or scenario‑specific files.

2. **Edge and fog layers (lightweight models)**
   - Train balanced and ultra‑light edge/fog models (Random Forest, Naive Bayes, Logistic Regression) using a small subset of features for low energy consumption.  
   - Use Stratified K‑fold cross‑validation per file to estimate performance stability.  
   - Compute per‑fold and per‑file metrics (accuracy, precision, recall, F1‑score, AUC where defined).  
   - Aggregate confusion matrices across folds and save them as figures.  
   - Save trained edge and fog models as `.joblib` files for deployment.

3. **Cloud layer (heavy models)**
   - Train cloud‑level models such as Random Forest, XGBoost, and LightGBM on richer feature sets.  
   - Perform detailed cross‑validation per file and across files.  
   - Compute detailed statistics (mean and standard deviation of metrics, min/max per file).  
   - Save confusion matrices, ROC data, and performance plots.  
   - Export one model per file and optionally global models (trained on sampled or concatenated data) as `.joblib`.

4. **Results and figures**
   - Generate figures comparing Edge, Fog, and Cloud performance (combined metric bar charts, heatmaps, boxplots, ROC approximations).  
   - Provide visual summaries of stability (variance of metrics) and the overall processing pipeline.

---

## Requirements

Create a Python environment (e.g., with `conda` or `venv`) and install the main dependencies:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn joblib scipy

