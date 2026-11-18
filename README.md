# KNN — Minimal Optimized

A compact, NumPy-only **K-Nearest Neighbors (KNN)** project. Exactly two files: `KNN.py` and this `README.md`.

## Setup

Requires Python 3.10+ and NumPy:

```powershell
python -m pip install numpy
```

## Run

```powershell
python KNN.py
```

## Example

```python
from KNN import knn_predict, KNNClassifier, KNNRegressor

X = [[0, 0], [1, 1], [8, 8], [9, 9]]
labels = ["A", "A", "B", "B"]

print(knn_predict(X, labels, [0.2, 0.3], k=3))
print(KNNClassifier(k=3, weights="distance").fit(X, labels).predict([[0.2, 0.3]]))
print(KNNRegressor(k=2).fit(X, [1, 2, 8, 9]).predict([[0.2, 0.3]]))
```

## What is included

- Basic `knn_predict(X, y, sample, k)` API returning predicted label and nearest labels
- `KNNClassifier`: classification, `predict_proba()`, accuracy `score()`
- `KNNRegressor`: numerical prediction and R² `score()`
- Distance methods: Euclidean, Manhattan, Minkowski, Chebyshev, Cosine
- Uniform or inverse-distance weighting, optional feature standardization, batched queries
- Validation and deterministic behavior when neighbor distances tie

## What was removed

The previous repository's `knn_advanced.py`, `knn_kdtree.py`, `knn_plot.py`, `knn_regression.py`, `test_knn.py`, `requirements.txt`, and `.gitignore` are intentionally **not included** in this minimal version. Advanced classification and regression are consolidated into `KNN.py`; KD-tree and plotting are removed. If existing code imports the removed modules, update those imports to `from KNN import KNNClassifier, KNNRegressor`.

NumPy is the only dependency. This version prioritizes clarity and compactness; it does not claim a benchmarked speedup over the previous implementation.

## Update an existing GitHub repository

After extracting, copy the **two files** into your existing cloned repo (keep its hidden `.git` folder). To make your GitHub repo contain only these two tracked files, remove the old tracked `.py` modules and other unwanted files from the local clone, review `git status`, then commit and push. Do not delete `.git`.
