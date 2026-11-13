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
        return 1.0 - (A / An) @ (B / Bn).T
    raise ValueError(f"ไม่รู้จัก metric: {metric}")


class KNNClassifier:
    def __init__(self, k=3, metric="euclidean", weights="uniform",
                 standardize=False, p=3):
        """
        k           : จำนวนเพื่อนบ้าน
        metric      : euclidean | manhattan | minkowski | chebyshev | cosine
        weights     : 'uniform' (โหวตเท่ากัน) | 'distance' (น้ำหนัก = 1/ระยะ)
        standardize : True = ทำ z-score ก่อนคำนวณระยะ
        p           : ค่า p สำหรับ minkowski
        """
        if weights not in ("uniform", "distance"):
            raise ValueError("weights ต้องเป็น 'uniform' หรือ 'distance'")
        self.k = k
        self.metric = metric
        self.weights = weights
        self.standardize = standardize
        self.p = p

    # ---------------- ฝึกโมเดล ----------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
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
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if self.standardize:
            X = (X - self.mean_) / self.std_

        D = pairwise_distances(X, self.X_, self.metric, self.p)
        # argpartition เร็วกว่า argsort เต็ม ๆ แล้วค่อยเรียงเฉพาะ k ตัว
        idx = np.argpartition(D, self.k - 1, axis=1)[:, : self.k]
        d = np.take_along_axis(D, idx, axis=1)
        order = np.argsort(d, axis=1, kind="stable")
        return np.take_along_axis(d, order, axis=1), np.take_along_axis(idx, order, axis=1)

    # ---------------- ความน่าจะเป็นของแต่ละคลาส ----------------
    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)

        if self.weights == "distance":
            w = 1.0 / np.maximum(dist, 1e-12)   # ระยะ 0 -> น้ำหนักสูงมาก
        else:
            w = np.ones_like(dist)

        n_query, n_classes = len(dist), len(self.classes_)
        votes = np.zeros((n_query, n_classes))
        labels = self._y_idx[idx]               # (n_query, k)
        rows = np.repeat(np.arange(n_query), self.k)
        np.add.at(votes, (rows, labels.ravel()), w.ravel())

        return votes / votes.sum(axis=1, keepdims=True)

    # ---------------- ทำนาย ----------------
    def predict(self, X):
        proba = self.predict_proba(X)
        dist, idx = self.kneighbors(X)
        labels = self._y_idx[idx]

        preds = []
        for i in range(len(proba)):
            best = np.flatnonzero(np.isclose(proba[i], proba[i].max()))
            if len(best) > 1:
                # เสมอกัน -> เลือกคลาสที่ระยะรวมของเพื่อนบ้านน้อยที่สุด
                totals = [dist[i][labels[i] == c].sum() for c in best]
                best = [best[int(np.argmin(totals))]]
            preds.append(self.classes_[best[0]])
        return np.array(preds)

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
    n_train = len(X) - len(X) // n_splits - 1      # กัน k > จำนวนข้อมูลฝึกใน fold
    results = {}
    for k in k_values:
        if k > n_train:
            continue
        results[k] = cross_val_score(X, y, n_splits, seed, k=k, **knn_params).mean()
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
