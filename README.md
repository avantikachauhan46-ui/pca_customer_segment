# 📊 Day 34 — PCA Customer Segmentation

A beginner-friendly ML project using **PCA + K-Means** for customer segmentation.

## Features
- Data loading and feature selection
- StandardScaler
- PCA dimensionality reduction
- Explained variance
- K-Means clustering
- Silhouette Score
- Cluster visualization
- Streamlit dashboard
- CSV export

## Run in VS Code

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Workflow

```text
Dataset
  ↓
Feature Selection
  ↓
StandardScaler
  ↓
PCA
  ↓
2 Principal Components
  ↓
K-Means
  ↓
Silhouette Score
  ↓
Cluster Visualization
```

> The included customer data is synthetic and is for learning/practice only.
