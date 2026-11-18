"""
KNN แบบ Advanced (ใช้ NumPy ล้วน ๆ)

ฟีเจอร์เพิ่มจากเวอร์ชันพื้นฐาน:
  1. เขียนเป็นคลาส (fit / predict / predict_proba / score) ใช้งานแบบ scikit-learn
  2. เลือกวิธีวัดระยะได้: euclidean, manhattan, minkowski, chebyshev, cosine
  3. Weighted voting: เพื่อนบ้านที่ใกล้กว่ามีน้ำหนักโหวตมากกว่า
  4. Standardization (z-score) เพื่อให้ฟีเจอร์ที่สเกลต่างกันมีผลเท่ากัน
  5. คำนวณระยะแบบ vectorized ทำนายหลายจุดพร้อมกัน (ไม่มี for loop ต่อ sample)
  6. predict_proba บอกความมั่นใจของแต่ละคลาส
  7. Tie-breaking ที่แน่นอน (เสมอกัน -> เลือกคลาสที่รวมระยะใกล้กว่า)
  8. K-Fold Cross-Validation + ค้นหาค่า k ที่ดีที่สุดอัตโนมัติ
  9. Confusion matrix และ accuracy
"""

import numpy as np


# ----------------------------------------------------------------------
# ฟังก์ชันวัดระยะ: คืนเมทริกซ์ขนาด (n_query, n_train)
# ----------------------------------------------------------------------
def pairwise_distances(A, B, metric="euclidean", p=3):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("A and B must be 2D with the same feature count")
    if metric == "minkowski" and (not np.isfinite(p) or p < 1):
        raise ValueError("Minkowski p must be finite and >= 1")

    if metric == "euclidean":
        # ใช้สูตร |a-b|^2 = |a|^2 + |b|^2 - 2a.b  (เร็วกว่าการวนลูป)
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(sq, 0.0))  # กัน error ปัดเศษติดลบ
    if metric == "manhattan":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
    if metric == "chebyshev":
        return np.abs(A[:, None, :] - B[None, :, :]).max(axis=2)
    if metric == "minkowski":
        return (np.abs(A[:, None, :] - B[None, :, :]) ** p).sum(axis=2) ** (1.0 / p)
    if metric == "cosine":
        An = np.linalg.norm(A, axis=1, keepdims=True)
        Bn = np.linalg.norm(B, axis=1, keepdims=True)
        An[An == 0] = 1.0
        Bn[Bn == 0] = 1.0
        return np.clip(1.0 - (A / An) @ (B / Bn).T, 0.0, 2.0)
    raise ValueError(f"ไม่รู้จัก metric: {metric}")


class KNNClassifier:
    def __init__(self, k=3, metric="euclidean", weights="uniform",
                 standardize=False, p=3, batch_size=512):
        """
        k           : จำนวนเพื่อนบ้าน
        metric      : euclidean | manhattan | minkowski | chebyshev | cosine
        weights     : 'uniform' (โหวตเท่ากัน) | 'distance' (น้ำหนัก = 1/ระยะ)
        standardize : True = ทำ z-score ก่อนคำนวณระยะ
        p           : ค่า p สำหรับ minkowski
        """
        if weights not in ("uniform", "distance"):
            raise ValueError("weights ต้องเป็น 'uniform' หรือ 'distance'")
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1:
            raise ValueError("k must be a positive integer")
        if metric not in ("euclidean", "manhattan", "minkowski", "chebyshev", "cosine"):
            raise ValueError("Unknown metric")
        if metric == "minkowski" and (not np.isfinite(p) or p < 1):
            raise ValueError("Minkowski p must be finite and >= 1")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.k = k
        self.metric = metric
        self.weights = weights
        self.standardize = standardize
        self.p = p

    # ---------------- ฝึกโมเดล ----------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array")
        if y.ndim != 1 or len(y) == 0:
            raise ValueError("y must be a nonempty 1D array")
        if len(X) != len(y):
            raise ValueError("X และ y ต้องมีจำนวนแถวเท่ากัน")
        if self.k > len(X):
            raise ValueError(f"k={self.k} มากกว่าจำนวนข้อมูลฝึก ({len(X)})")

        if self.standardize:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1.0  # กันหารศูนย์
            X = (X - self.mean_) / self.std_

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        # แปลง label เป็น index 0..C-1 เพื่อใช้นับแบบ vectorized
        self._y_idx = np.searchsorted(self.classes_, y)
        return self

    # ---------------- หาเพื่อนบ้าน ----------------
    def kneighbors(self, X):
        if not hasattr(self, "X_"):
            raise RuntimeError("Call fit before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1] or not np.isfinite(X).all():
            raise ValueError("Query must be finite and have the same feature count as training data")
        if self.standardize:
            X = (X - self.mean_) / self.std_

        all_dist, all_idx = [], []
        for start in range(0, len(X), self.batch_size):
            D = pairwise_distances(X[start:start + self.batch_size], self.X_, self.metric, self.p)
            # kth cutoff with deterministic ordering of equal distances by training index.
            idx = np.argpartition(D, self.k - 1, axis=1)[:, :self.k]
            if self.k < self.X_.shape[0]:
                threshold = np.max(np.take_along_axis(D, idx, axis=1), axis=1)
                # Only revisit rows whose kth distance has a tie across the cutoff.
                ties = np.sum(D <= threshold[:, None], axis=1) > self.k
                for r in np.flatnonzero(ties):
                    idx[r] = np.lexsort((np.arange(D.shape[1]), D[r]))[:self.k]
            d = np.take_along_axis(D, idx, axis=1)
            order = np.lexsort((idx, d), axis=1)
            all_dist.append(np.take_along_axis(d, order, axis=1))
            all_idx.append(np.take_along_axis(idx, order, axis=1))
        if not all_dist:
            return np.empty((0, self.k)), np.empty((0, self.k), dtype=int)
        return np.vstack(all_dist), np.vstack(all_idx)

    def _votes(self, dist, idx):
        if self.weights == "distance":
            zeros = dist <= 1e-12
            w = np.where(zeros.any(axis=1, keepdims=True), zeros.astype(float),
                         1.0 / np.maximum(dist, 1e-12))
        else:
            w = np.ones_like(dist)
        votes = np.zeros((len(dist), len(self.classes_)), dtype=float)
        labels = self._y_idx[idx]
        np.add.at(votes, (np.repeat(np.arange(len(dist)), self.k), labels.ravel()), w.ravel())
        return votes, labels

    # ---------------- ความน่าจะเป็นของแต่ละคลาส ----------------
    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        votes, _ = self._votes(dist, idx)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        dist, idx = self.kneighbors(X)
        votes, labels = self._votes(dist, idx)
        predictions = np.argmax(votes, axis=1)
        for i in range(len(votes)):
            tied = np.flatnonzero(np.isclose(votes[i], votes[i].max()))
            if len(tied) > 1:
                sums = np.array([dist[i, labels[i] == c].sum() for c in tied])
                predictions[i] = tied[np.argmin(sums)]
        return self.classes_[predictions]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# ----------------------------------------------------------------------
# เครื่องมือประเมินผล
# ----------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def k_fold_indices(n, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    if not isinstance(n_splits, int) or not 2 <= n_splits <= n:
        raise ValueError("n_splits must be between 2 and n_samples")
    perm = rng.permutation(n)
    return np.array_split(perm, n_splits)


def cross_val_score(X, y, n_splits=5, seed=0, **knn_params):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    folds = k_fold_indices(len(X), n_splits, seed)
    scores = []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
        model = KNNClassifier(**knn_params).fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[test_idx], y[test_idx]))
    return np.array(scores)


def find_best_k(X, y, k_values=range(1, 16), n_splits=5, seed=0, **knn_params):
    """ลองหลายค่า k ด้วย cross-validation แล้วคืน k ที่ให้ accuracy เฉลี่ยสูงสุด"""
    if not 2 <= n_splits <= len(X):
        raise ValueError("Invalid n_splits")
    n_train = len(X) - int(np.ceil(len(X) / n_splits))
    results = {}
    for k in k_values:
        if k > n_train:
            continue
        results[k] = cross_val_score(X, y, n_splits, seed, k=k, **knn_params).mean()
    if not results:
        raise ValueError("No valid k values for this dataset")
    best_k = max(results, key=results.get)
    return best_k, results


# ----------------------------------------------------------------------
# ตัวอย่างใช้งาน
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- 1) ข้อมูลชุดเดิมจาก KNN.py ---
    X = np.array([[1, 2], [2, 2], [2, 1], [5, 5], [5, 6], [6, 5]])
    y = np.array(["A", "A", "A", "B", "B", "B"])

    model = KNNClassifier(k=3, metric="euclidean", weights="distance").fit(X, y)
    sample = np.array([[2, 4], [4.5, 4.5], [3.5, 3.5]])

    print("=== ข้อมูลชุดเดิม ===")
    print("Predicted :", model.predict(sample))
    print("Proba     :")
    for s, p in zip(sample, model.predict_proba(sample)):
        print(f"  {s} -> " + ", ".join(f"{c}={v:.2f}" for c, v in zip(model.classes_, p)))
    dist, idx = model.kneighbors(sample[0])
    print("เพื่อนบ้านของ [2, 4]:", y[idx[0]], "ระยะ:", np.round(dist[0], 3))

    # --- 2) ข้อมูลสังเคราะห์ 3 คลาส + ฟีเจอร์คนละสเกล เพื่อโชว์ standardize / หา k ---
    rng = np.random.default_rng(42)
    n = 60
    centers = np.array([[0, 0], [3, 3], [0, 4]])
    Xs, ys = [], []
    for c, ctr in zip("ABC", centers):
        pts = rng.normal(ctr, 1.0, size=(n, 2))
        Xs.append(pts)
        ys += [c] * n
    Xs = np.vstack(Xs)
    ys = np.array(ys)
    Xs[:, 1] *= 1000            # ทำให้ฟีเจอร์ที่ 2 สเกลใหญ่กว่ามาก

    print("\n=== ข้อมูล 3 คลาส (ฟีเจอร์สเกลต่างกัน) ===")
    for std in (False, True):
        acc = cross_val_score(Xs, ys, n_splits=5, k=5, standardize=std).mean()
        print(f"standardize={std!s:5}  CV accuracy = {acc:.3f}")

    best_k, res = find_best_k(Xs, ys, k_values=range(1, 21, 2), standardize=True)
    print("\nCV accuracy ตามค่า k:")
    for k, a in res.items():
        print(f"  k={k:2d}  {a:.3f}" + ("  <-- ดีที่สุด" if k == best_k else ""))

    # แบ่ง train/test แล้วดู confusion matrix
    perm = rng.permutation(len(Xs))
    cut = int(0.75 * len(Xs))
    tr, te = perm[:cut], perm[cut:]
    final = KNNClassifier(k=best_k, weights="distance", standardize=True).fit(Xs[tr], ys[tr])
    pred = final.predict(Xs[te])
    print(f"\nTest accuracy (k={best_k}): {final.score(Xs[te], ys[te]):.3f}")
    print("Confusion matrix (แถว=จริง, คอลัมน์=ทำนาย):", [str(c) for c in final.classes_])
    print(confusion_matrix(ys[te], pred, final.classes_))
