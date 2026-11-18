# KNN — Optimized NumPy implementation

A learning-oriented K-Nearest Neighbors project, supporting classification, regression, custom distance metrics, KD-tree search and visualization. This version retains the five original Python examples and improves reliability and memory efficiency.

## Requirements

Python 3.10+ and NumPy. Matplotlib is used only for plots.

```bash
python -m pip install -r requirements.txt
```

## Run examples

```bash
python KNN.py
python knn_advanced.py
python knn_plot.py output
python knn_regression.py
python knn_kdtree.py
```

Note: The last two examples include larger performance demonstrations and may take longer. Use the automated tests for a quick check.

## Quick usage

```python
import numpy as np
from knn_advanced import KNNClassifier
from knn_regression import KNNRegressor

X = np.array([[0, 0], [1, 1], [9, 9], [10, 10]], dtype=float)
classifier = KNNClassifier(k=2, weights="distance", standardize=True, batch_size=128)
classifier.fit(X, np.array(["A", "A", "B", "B"]))
print(classifier.predict([[0.3, 0.2]]))

regressor = KNNRegressor(k=2, weights="distance", algorithm="brute")
regressor.fit(X, np.array([1., 2., 9., 10.]))
print(regressor.predict([[0.3, 0.2]]))
```

## Improvements

- Classification processes query vectors in batches (`batch_size`), avoiding one large query-by-training distance matrix.
- Consistent validation for positive `k`, data shape and finite numeric inputs.
- Distance-weighted voting handles exact duplicate positions by considering only zero-distance neighbors, consistent with nearest-neighbor semantics.
- Fixed cross-validation's maximum permitted `k` for the smallest training fold and explicit validation when none is usable.
- Fixed numerical rounding for cosine distance and added deterministic index-based tie ordering in brute-force classification.
- KD-tree performs less aggressive pruning at equal-distance boundaries and validates query input.
- Basic KNN example has no import-time side effects; plotting creates its output directory.
- Regression and classification keep their existing public APIs.

For `cosine`, zero-length vectors use zero-normalized representations. KD-tree is not available for cosine.

## Run tests

```bash
python -m unittest discover -v
```

## Notes

These are exact nearest-neighbor algorithms, not an approximate nearest-neighbor index. `algorithm="auto"` for regression can choose KD-tree for larger low-dimensional datasets. For high-dimensional datasets, brute-force search can be faster. Speed varies with dataset and CPU; no universal speedup percentage is claimed.
