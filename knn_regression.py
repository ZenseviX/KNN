"""
KNN สำหรับ Regression (ทำนายค่าตัวเลขต่อเนื่อง) NumPy ล้วน

หลักการ: หา k เพื่อนบ้านที่ใกล้ที่สุด แล้วเฉลี่ยค่า y ของเพื่อนบ้านเหล่านั้น
  - weights='uniform'  : ค่าเฉลี่ยธรรมดา
  - weights='distance' : ค่าเฉลี่ยถ่วงน้ำหนัก (ยิ่งใกล้ ยิ่งมีน้ำหนักมาก)

ฟีเจอร์:
  - เลือก metric ได้ (euclidean, manhattan, chebyshev, minkowski, cosine)
  - standardize ฟีเจอร์ก่อนคำนวณระยะ
  - algorithm='kd_tree' ใช้ KDTree จาก knn_kdtree.py เมื่อข้อมูลใหญ่
  - ตัวชี้วัด R^2, MAE, RMSE และ cross-validation + หา k ที่ดีที่สุด
"""

import numpy as np

from knn_kdtree import KD_METRICS, KDTree


def _brute_neighbors(X_train, Q, k, metric, p):
    """หาเพื่อนบ้านแบบเทียบทุกจุด (แบ่งเป็นก้อนเพื่อไม่กินหน่วยความจำมาก)"""
    n, d = X_train.shape
    if metric in ("euclidean", "cosine"):
        chunk = max(1, int(2e7 // max(n, 1)))
    else:
        chunk = max(1, int(2e7 // max(n * d, 1)))

    out_d, out_i = [], []
    for s in range(0, len(Q), chunk):
        A = Q[s:s + chunk]
        if metric == "euclidean":
            sq = (A ** 2).sum(1)[:, None] + (X_train ** 2).sum(1)[None, :] - 2 * A @ X_train.T
            D = np.sqrt(np.maximum(sq, 0.0))
        elif metric == "cosine":
            An = np.linalg.norm(A, axis=1, keepdims=True)
            Bn = np.linalg.norm(X_train, axis=1, keepdims=True)
            An[An == 0] = 1.0
            Bn[Bn == 0] = 1.0
            D = 1.0 - (A / An) @ (X_train / Bn).T
        else:
            diff = np.abs(A[:, None, :] - X_train[None, :, :])
            if metric == "manhattan":
                D = diff.sum(axis=2)
            elif metric == "chebyshev":
                D = diff.max(axis=2)
            elif metric == "minkowski":
                D = (diff ** p).sum(axis=2) ** (1.0 / p)
            else:
                raise ValueError(f"ไม่รู้จัก metric: {metric}")

        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        dist = np.take_along_axis(D, idx, axis=1)
        order = np.argsort(dist, axis=1, kind="stable")
        out_d.append(np.take_along_axis(dist, order, axis=1))
        out_i.append(np.take_along_axis(idx, order, axis=1))
    return np.vstack(out_d), np.vstack(out_i)


class KNNRegressor:
    def __init__(self, k=5, metric="euclidean", weights="uniform",
                 standardize=False, p=3, algorithm="auto", leaf_size=30):
        """
        algorithm : 'brute' | 'kd_tree' | 'auto'
                    auto = ใช้ kd_tree เมื่อข้อมูลฝึกมาก (>= 5000 แถว),
                           มิติไม่เกิน 15 และ metric รองรับ KD-tree
        """
        if weights not in ("uniform", "distance"):
            raise ValueError("weights ต้องเป็น 'uniform' หรือ 'distance'")
        if algorithm not in ("auto", "brute", "kd_tree"):
            raise ValueError("algorithm ต้องเป็น 'auto', 'brute' หรือ 'kd_tree'")
        self.k = k
        self.metric = metric
        self.weights = weights
        self.standardize = standardize
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float)
        if len(X) != len(y):
            raise ValueError("X และ y ต้องมีจำนวนแถวเท่ากัน")
        if self.k > len(X):
            raise ValueError(f"k={self.k} มากกว่าจำนวนข้อมูลฝึก ({len(X)})")

        if self.standardize:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
            X = (X - self.mean_) / self.std_

        self.X_, self.y_ = X, y

        algo = self.algorithm
        if algo == "auto":
            use_kd = (len(X) >= 5000 and X.shape[1] <= 15
                      and self.metric in KD_METRICS)
            algo = "kd_tree" if use_kd else "brute"
        if algo == "kd_tree" and self.metric not in KD_METRICS:
            raise ValueError(f"KD-tree ไม่รองรับ metric '{self.metric}'")
        self.algorithm_ = algo
        self.tree_ = (KDTree(X, self.leaf_size, self.metric, self.p)
                      if algo == "kd_tree" else None)
        return self

    def kneighbors(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            # จุดเดียว: ถ้าโมเดลมีหลายฟีเจอร์ให้ถือเป็น 1 แถว ถ้ามีฟีเจอร์เดียวให้ถือเป็นหลายแถว
            X = X[None, :] if self.X_.shape[1] > 1 else X[:, None]
        if self.standardize:
            X = (X - self.mean_) / self.std_
        if self.tree_ is not None:
            return self.tree_.query(X, self.k)
        return _brute_neighbors(self.X_, X, self.k, self.metric, self.p)

    def predict(self, X):
        dist, idx = self.kneighbors(X)
        neighbor_y = self.y_[idx]                       # (n_query, k)
        if self.weights == "distance":
            w = 1.0 / np.maximum(dist, 1e-12)
        else:
            w = np.ones_like(dist)
        return (w * neighbor_y).sum(axis=1) / w.sum(axis=1)

    def score(self, X, y):
        """R^2 (1 = สมบูรณ์แบบ, 0 = ไม่ดีไปกว่าเดาค่าเฉลี่ย)"""
        return r2_score(y, self.predict(X))


# ----------------------------------------------------------------------
# ตัวชี้วัด
# ----------------------------------------------------------------------
def r2_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ----------------------------------------------------------------------
# Cross-validation และหา k ที่ดีที่สุด
# ----------------------------------------------------------------------
def cross_val_score(X, y, n_splits=5, seed=0, **params):
    """คืน R^2 ของแต่ละ fold"""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(X)), n_splits)
    scores = []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
        model = KNNRegressor(**params).fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[test_idx], y[test_idx]))
    return np.array(scores)


def find_best_k(X, y, k_values=range(1, 31), n_splits=5, seed=0, **params):
    max_train = len(X) - len(X) // n_splits - 1
    results = {k: cross_val_score(X, y, n_splits, seed, k=k, **params).mean()
               for k in k_values if k <= max_train}
    return max(results, key=results.get), results


# ----------------------------------------------------------------------
# ตัวอย่างใช้งาน
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time

    rng = np.random.default_rng(1)

    # --- 1) ข้อมูลเล็ก ๆ ดูผลทำนาย ---
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([1.2, 1.9, 3.2, 3.8, 5.1, 6.0])
    m = KNNRegressor(k=3, weights="distance").fit(X, y)
    print("=== ตัวอย่างเล็ก ===")
    for q in (2.5, 4.2):
        print(f"x={q} -> ทำนาย y = {m.predict([q])[0]:.3f}")

    # --- 2) ข้อมูลซับซ้อน: sin(x) + noise ---
    n = 300
    X = rng.uniform(0, 10, size=(n, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.25, size=n)

    print("\n=== sin(x) + noise: เลือก k ด้วย cross-validation ===")
    best_k, res = find_best_k(X, y, k_values=range(1, 31, 2), weights="distance")
    for k, r2 in res.items():
        print(f"  k={k:2d}  R^2={r2:.3f}" + ("  <-- ดีที่สุด" if k == best_k else ""))

    perm = rng.permutation(n)
    tr, te = perm[:225], perm[225:]
    model = KNNRegressor(k=best_k, weights="distance").fit(X[tr], y[tr])
    pred = model.predict(X[te])
    print(f"\nTest (k={best_k}):  R^2={r2_score(y[te], pred):.3f}  "
          f"MAE={mae(y[te], pred):.3f}  RMSE={rmse(y[te], pred):.3f}")

    # --- 3) KD-tree กับ brute ให้ผลตรงกัน และเร็วกว่าเมื่อข้อมูลใหญ่ ---
    N = 100_000
    Xb = rng.uniform(size=(N, 2))
    yb = np.sin(6 * Xb[:, 0]) + np.cos(4 * Xb[:, 1])
    Qb = rng.uniform(size=(500, 2))
    print(f"\n=== ข้อมูลใหญ่ {N:,} แถว ทำนาย 500 จุด ===")
    preds = {}
    for algo in ("brute", "kd_tree"):
        t = time.perf_counter()
        mdl = KNNRegressor(k=5, algorithm=algo).fit(Xb, yb)
        preds[algo] = mdl.predict(Qb)
        print(f"  {algo:8s} {time.perf_counter() - t:6.2f}s")
    print("  ผลทำนายตรงกัน:", np.allclose(preds["brute"], preds["kd_tree"]))

    # --- 4) วาดกราฟ (ถ้ามี matplotlib) ---
    out = sys.argv[1] if len(sys.argv) > 1 else "knn_regression.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = np.linspace(0, 10, 400)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, k in zip(axes, (1, 5, 30)):
            fit = KNNRegressor(k=k, weights="uniform").fit(X, y)
            ax.scatter(X[:, 0], y, s=12, alpha=0.5, label="ข้อมูล")
            ax.plot(grid, np.sin(grid), "g--", lw=1.5, label="sin(x) จริง")
            ax.plot(grid, fit.predict(grid), "r-", lw=2, label=f"KNN k={k}")
            ax.set_title(f"k = {k}")
            ax.legend(loc="upper right")
        fig.suptitle("KNN Regression: k น้อย = ตามสัญญาณรบกวน (overfit), k มาก = เรียบเกินไป")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nบันทึกกราฟแล้ว: {out}")
    except ImportError:
        print("\nไม่มี matplotlib จึงข้ามการวาดกราฟ (pip install matplotlib)")
