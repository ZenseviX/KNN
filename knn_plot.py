"""
วาดกราฟ Decision Boundary ของ KNN ด้วย matplotlib

ต้องมีไฟล์ KNN_advanced.py อยู่โฟลเดอร์เดียวกัน (ใช้คลาส KNNClassifier)
ติดตั้ง matplotlib:  pip install matplotlib

ใช้งานจากไฟล์อื่น:
    from knn_plot import plot_decision_boundary
    model = KNNClassifier(k=5).fit(X, y)
    plot_decision_boundary(model, X, y)
    plt.show()
"""

import sys

import numpy as np

from knn_advanced import KNNClassifier
PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]


def plot_decision_boundary(model, X, y, ax=None, resolution=150, padding=0.5,
                           title=None, feature_names=("x1", "x2")):
    """
    วาดพื้นที่ที่โมเดลทำนายเป็นแต่ละคลาส (สีพื้น) พร้อมจุดข้อมูลจริง
    ใช้ได้กับข้อมูล 2 ฟีเจอร์เท่านั้น
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.shape[1] != 2:
        raise ValueError("ต้องมี 2 ฟีเจอร์เท่านั้นถึงจะวาด decision boundary ได้")
    if ax is None:
        ax = plt.gca()

    classes = model.classes_
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]

    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    pred = model.predict(grid)
    zz = np.searchsorted(classes, pred).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=np.arange(-0.5, len(classes), 1),
                colors=colors, alpha=0.25)
    ax.contour(xx, yy, zz, levels=np.arange(0.5, len(classes) - 1, 1),
               colors="k", linewidths=0.8, alpha=0.6)
    for c, col in zip(classes, colors):
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], c=col, edgecolor="k", s=28, label=str(c))

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax


def make_moons(n=200, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = rng.uniform(0, np.pi, n_out)
    t_in = rng.uniform(0, np.pi, n_in)
    outer = np.c_[np.cos(t_out), np.sin(t_out)]
    inner = np.c_[1 - np.cos(t_in), 0.5 - np.sin(t_in)]
    X = np.vstack([outer, inner]) + rng.normal(0, noise, size=(n, 2))
    y = np.array(["A"] * n_out + ["B"] * n_in)
    return X, y


def make_blobs(n_per_class=60, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [3, 3], [0, 4]])
    X = np.vstack([rng.normal(c, 1.0, size=(n_per_class, 2)) for c in centers])
    y = np.repeat(np.array(["A", "B", "C"]), n_per_class)
    return X, y


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")           # บันทึกเป็นไฟล์ได้แม้ไม่มีหน้าจอ
    import matplotlib.pyplot as plt

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # --- รูปที่ 1: เปรียบเทียบค่า k บนข้อมูลรูปพระจันทร์เสี้ยว ---
    X, y = make_moons(200, noise=0.3, seed=3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, k in zip(axes, (1, 7, 50)):
        model = KNNClassifier(k=k).fit(X, y)
        acc = model.score(X, y)
        plot_decision_boundary(model, X, y, ax=ax,
                               title=f"k = {k}  (train acc = {acc:.2f})")
    fig.suptitle("ผลของค่า k: k=1 ขอบหยัก (overfit), k=50 เรียบเกินไป (underfit)")
    fig.tight_layout()
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p1 = str(Path(out_dir) / "knn_decision_boundary_k.png")
    fig.savefig(p1, dpi=120)
    print("บันทึกแล้ว:", p1)
    plt.close(fig)

    # --- รูปที่ 2: เปรียบเทียบ uniform vs distance และ metric (3 คลาส) ---
    X, y = make_blobs(60, seed=1)
    configs = [
        dict(k=15, weights="uniform", metric="euclidean"),
        dict(k=15, weights="distance", metric="euclidean"),
        dict(k=15, weights="uniform", metric="manhattan"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, cfg in zip(axes, configs):
        model = KNNClassifier(**cfg).fit(X, y)
        plot_decision_boundary(model, X, y, ax=ax,
                               title=f"{cfg['metric']}, {cfg['weights']}, k={cfg['k']}")
    fig.suptitle("เปรียบเทียบ weights และ metric บนข้อมูล 3 คลาส")
    fig.tight_layout()
    p2 = str(Path(out_dir) / "knn_decision_boundary_config.png")
    fig.savefig(p2, dpi=120)
    print("บันทึกแล้ว:", p2)
    plt.close(fig)
