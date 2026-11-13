"""
KD-tree สำหรับ KNN (NumPy ล้วน) ใช้ค้นหาเพื่อนบ้านเร็วขึ้นเมื่อข้อมูลฝึกมีจำนวนมาก

แนวคิด:
  - แบ่งพื้นที่ซ้ำ ๆ ด้วยมิติที่ข้อมูลกระจายมากที่สุด ตัดที่ค่ามัธยฐาน
  - ตอนค้นหาจะข้ามกิ่งที่ "ไกลเกินกว่าเพื่อนบ้านที่ดีที่สุดตอนนี้" (pruning)
    จึงไม่ต้องเทียบกับข้อมูลทุกจุด

รองรับ metric: euclidean, manhattan, chebyshev, minkowski (ตั้งค่า p)
(cosine ใช้กับ KD-tree ไม่ได้ เพราะไม่ใช่ระยะแบบ norm)
"""

import heapq
import time

import numpy as np

KD_METRICS = ("euclidean", "manhattan", "chebyshev", "minkowski")


def point_distances(pts, q, metric="euclidean", p=3):
    """ระยะจากจุด q ไปยังทุกแถวของ pts"""
    diff = np.abs(pts - q)
    if metric == "euclidean":
        return np.sqrt((diff ** 2).sum(axis=1))
    if metric == "manhattan":
        return diff.sum(axis=1)
    if metric == "chebyshev":
        return diff.max(axis=1)
    if metric == "minkowski":
        return (diff ** p).sum(axis=1) ** (1.0 / p)
    raise ValueError(f"KD-tree ไม่รองรับ metric: {metric}")


def brute_force_knn(X, Q, k, metric="euclidean", p=3):
    """ค้นหาแบบเทียบทุกจุด ใช้เป็นตัวอ้างอิงตรวจความถูกต้อง"""
    all_d, all_i = [], []
    for q in Q:
        d = point_distances(X, q, metric, p)
        idx = np.argsort(d, kind="stable")[:k]
        all_d.append(d[idx])
        all_i.append(idx)
    return np.array(all_d), np.array(all_i)


class KDTree:
    def __init__(self, X, leaf_size=30, metric="euclidean", p=3):
        if metric not in KD_METRICS:
            raise ValueError(f"metric ต้องเป็นหนึ่งใน {KD_METRICS}")
        self.X = np.asarray(X, dtype=float)
        self.leaf_size = max(1, int(leaf_size))
        self.metric = metric
        self.p = p

        n = len(self.X)
        self.idx = np.arange(n)          # ลำดับจุดหลังจัดเรียงตามต้นไม้
        # ข้อมูลของแต่ละ node เก็บเป็นลิสต์ขนาน (index = หมายเลข node)
        self.dim, self.val, self.left, self.right = [], [], [], []
        self.start, self.end = [], []
        self._build(0, n)

    # ---------------- สร้างต้นไม้ ----------------
    def _build(self, lo, hi):
        node = len(self.dim)
        self.dim.append(-1)
        self.val.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.start.append(lo)
        self.end.append(hi)

        if hi - lo <= self.leaf_size:
            return node                  # leaf

        pts = self.X[self.idx[lo:hi]]
        d = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
        mid = (lo + hi) // 2
        order = np.argpartition(pts[:, d], mid - lo)
        self.idx[lo:hi] = self.idx[lo:hi][order]

        self.dim[node] = d
        self.val[node] = float(self.X[self.idx[mid], d])
        self.left[node] = self._build(lo, mid)
        self.right[node] = self._build(mid, hi)
        return node

    # ---------------- ค้นหา 1 จุด ----------------
    def _query_one(self, q, k):
        heap = []                        # max-heap เก็บ (-ระยะ, index)
        stack = [(0, 0.0)]               # (node, ขอบเขตระยะต่ำสุดไปยัง node นี้)

        while stack:
            node, bound = stack.pop()
            if len(heap) == k and bound >= -heap[0][0]:
                continue                 # กิ่งนี้ไกลเกินไป ตัดทิ้ง

            # ลงไปหา leaf โดยเลือกฝั่งใกล้ก่อน และฝากฝั่งไกลไว้ใน stack
            while self.dim[node] != -1:
                diff = q[self.dim[node]] - self.val[node]
                if diff <= 0:
                    near, far = self.left[node], self.right[node]
                else:
                    near, far = self.right[node], self.left[node]
                # |diff| คือขอบเขตล่างของระยะไปฝั่งไกล (ใช้ได้กับทุก p-norm)
                stack.append((far, max(bound, abs(diff))))
                node = near

            ids = self.idx[self.start[node]:self.end[node]]
            dists = point_distances(self.X[ids], q, self.metric, self.p)
            for dist, i in zip(dists.tolist(), ids.tolist()):
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, i))
                elif dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist, i))

        result = sorted((-nd, i) for nd, i in heap)
        return [r[0] for r in result], [r[1] for r in result]

    # ---------------- ค้นหาหลายจุด ----------------
    def query(self, Q, k=1):
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 1:
            Q = Q[None, :]
        if k > len(self.X):
            raise ValueError(f"k={k} มากกว่าจำนวนข้อมูล ({len(self.X)})")
        out_d = np.empty((len(Q), k))
        out_i = np.empty((len(Q), k), dtype=int)
        for r, q in enumerate(Q):
            d, i = self._query_one(q, k)
            out_d[r], out_i[r] = d, i
        return out_d, out_i


# ----------------------------------------------------------------------
# ทดสอบความถูกต้อง + วัดความเร็ว
# ----------------------------------------------------------------------
def _timeit(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k, n_query = 5, 500

    print("1) ตรวจว่า KD-tree ให้ผลตรงกับ brute force")
    X = rng.normal(size=(3000, 3))
    Q = rng.normal(size=(200, 3))
    for metric in KD_METRICS:
        tree = KDTree(X, leaf_size=20, metric=metric)
        d_kd, _ = tree.query(Q, k)
        d_bf, _ = brute_force_knn(X, Q, k, metric)
        ok = np.allclose(d_kd, d_bf)
        print(f"   {metric:10s} ระยะตรงกัน: {ok}")

    print("\n2) เทียบเวลา (2 มิติ, ค้นหา 500 จุด, k=5)")
    print(f"   {'n_train':>8} | {'สร้างต้นไม้':>10} | {'KD-tree':>8} | {'brute':>8}")
    for n in (2_000, 20_000, 100_000):
        X = rng.uniform(size=(n, 2))
        Q = rng.uniform(size=(n_query, 2))
        tree, t_build = _timeit(lambda: KDTree(X, leaf_size=30))
        _, t_kd = _timeit(lambda: tree.query(Q, k))
        _, t_bf = _timeit(lambda: brute_force_knn(X, Q, k))
        print(f"   {n:>8} | {t_build:>9.3f}s | {t_kd:>7.3f}s | {t_bf:>7.3f}s")
