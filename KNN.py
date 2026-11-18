"""Small, NumPy-only K-nearest-neighbors classification and regression.

Run ``python KNN.py`` to see an example, or import KNNClassifier/KNNRegressor.
"""
from __future__ import annotations

import numpy as np

_METRICS = {"euclidean", "manhattan", "chebyshev", "minkowski", "cosine"}


def _neighbors(train: np.ndarray, queries: np.ndarray, k: int, metric: str, p: float):
    """Return nearest distances and indices, sorting equal-distance ties by index."""
    if metric == "euclidean":
        # Work in float64 to avoid overflow when inputs are uint8 or large integers.
        delta = queries[:, None, :] - train[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    elif metric == "cosine":
        train_norm = np.linalg.norm(train, axis=1)
        query_norm = np.linalg.norm(queries, axis=1)
        product = queries @ train.T
        denom = query_norm[:, None] * train_norm[None, :]
        similarity = np.zeros_like(product)
        np.divide(product, denom, out=similarity, where=denom > 0)
        distances = np.clip(1.0 - similarity, 0.0, 2.0)
    else:
        delta = np.abs(queries[:, None, :] - train[None, :, :])
        if metric == "manhattan":
            distances = delta.sum(axis=2)
        elif metric == "chebyshev":
            distances = delta.max(axis=2)
        else:
            distances = np.power(np.power(delta, p).sum(axis=2), 1.0 / p)

    # stable argsort gives deterministic tie behavior, including across the kth boundary.
    idx = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(distances, idx, axis=1), idx


class _BaseKNN:
    def __init__(self, k=3, *, metric="euclidean", weights="uniform",
                 standardize=False, p=2, batch_size=64):
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)}")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be uniform or distance")
        if not np.isfinite(p) or p < 1:
            raise ValueError("p must be finite and >= 1")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.k, self.metric, self.weights = int(k), metric, weights
        self.standardize, self.p, self.batch_size = standardize, float(p), batch_size

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or not X.size or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1D array matching X rows")
        if self.k > len(X):
            raise ValueError("k cannot exceed training sample count")
        self.mean_ = X.mean(axis=0) if self.standardize else np.zeros(X.shape[1])
        self.scale_ = X.std(axis=0) if self.standardize else np.ones(X.shape[1])
        self.scale_[self.scale_ == 0] = 1.0
        self.X_ = (X - self.mean_) / self.scale_
        self.y_ = y.copy()
        return self

    def kneighbors(self, X):
        if not hasattr(self, "X_"):
            raise RuntimeError("Call fit before predict")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1] or not np.isfinite(X).all():
            raise ValueError("Queries must have the same number of finite features as X")
        X = (X - self.mean_) / self.scale_
        distances, indices = [], []
        # For non-Euclidean metrics, intermediate delta has 3 dimensions.
        max_batch = max(1, 2_000_000 // max(1, self.X_.size))
        chunk_size = min(self.batch_size, max_batch)
        for start in range(0, len(X), chunk_size):
            d, i = _neighbors(self.X_, X[start:start + chunk_size], self.k,
                              self.metric, self.p)
            distances.append(d)
            indices.append(i)
        if not distances:
            return np.empty((0, self.k)), np.empty((0, self.k), dtype=int)
        return np.vstack(distances), np.vstack(indices)

    def _weight_matrix(self, distances):
        if self.weights == "uniform":
            return np.ones_like(distances)
        exact = distances <= 1e-12
        # Only identical observations vote if at least one exact match is present.
        return np.where(exact.any(axis=1, keepdims=True), exact.astype(float),
                        1.0 / np.maximum(distances, 1e-12))


class KNNClassifier(_BaseKNN):
    """KNN classifier with deterministic label tie-breaking and predict_proba."""

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_, self.encoded_ = np.unique(self.y_, return_inverse=True)
        return self

    def predict_proba(self, X):
        distances, indices = self.kneighbors(X)
        weights = self._weight_matrix(distances)
        votes = np.zeros((len(indices), len(self.classes_)), dtype=float)
        for row in range(len(indices)):
            np.add.at(votes[row], self.encoded_[indices[row]], weights[row])
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        y = np.asarray(y)
        pred = self.predict(X)
        if y.shape != pred.shape or not len(y):
            raise ValueError("y must match the query count")
        return float(np.mean(pred == y))


class KNNRegressor(_BaseKNN):
    """KNN regression using the same neighbor engine as classification."""

    def fit(self, X, y):
        super().fit(X, y)
        self.y_ = np.asarray(y, dtype=float)
        if not np.isfinite(self.y_).all():
            raise ValueError("Regression targets must be finite numbers")
        return self

    def predict(self, X):
        distances, indices = self.kneighbors(X)
        weights = self._weight_matrix(distances)
        return (self.y_[indices] * weights).sum(axis=1) / weights.sum(axis=1)

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        if y.shape != pred.shape or not len(y) or not np.isfinite(y).all():
            raise ValueError("y must match the query count")
        total = np.square(y - y.mean()).sum()
        residual = np.square(y - pred).sum()
        return float(1 - residual / total) if total else float(residual == 0)


def knn_predict(X, y, sample, k=3):
    """Backwards-compatible basic API: (predicted_label, neighbor_labels)."""
    model = KNNClassifier(k=k).fit(X, y)
    _, indices = model.kneighbors(sample)
    if len(indices) != 1:
        raise ValueError("sample must be a single 1D vector")
    return model.predict(sample)[0], model.y_[indices[0]]


if __name__ == "__main__":
    X = np.array([[1, 2], [2, 2], [2, 1], [5, 5], [5, 6], [6, 5]])
    y = np.array(["A", "A", "A", "B", "B", "B"])
    label, neighbors = knn_predict(X, y, [2, 4], k=3)
    print("Nearest labels:", neighbors)
    print("Predicted class:", label)
    reg = KNNRegressor(k=3, weights="distance").fit(X, [1., 1.5, 2., 8., 9., 10.])
    print("Regression:", round(float(reg.predict([2, 4])[0]), 3))
