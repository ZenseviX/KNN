import numpy as np

# ฟังก์ชัน KNN
def knn_predict(X, y, sample, k=3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    sample = np.asarray(sample, dtype=float)
    if X.ndim != 2 or sample.shape != (X.shape[1],) or len(y) != len(X):
        raise ValueError("Invalid input shapes")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and number of samples")
    # 1. คำนวณระยะห่าง
    distances = np.linalg.norm(X - sample, axis=1)

    # 2. หา k จุดที่ใกล้ที่สุด
    nearest = np.argpartition(distances, k - 1)[:k]
    nearest = nearest[np.lexsort((nearest, distances[nearest]))]

    # 3. เอา Label ของจุดที่ใกล้ที่สุด
    nearest_labels = y[nearest]

    # 4. นับจำนวนแต่ละ Class
    labels, counts = np.unique(nearest_labels, return_counts=True)

    # 5. เลือก Class ที่มีจำนวนมากที่สุด
    prediction = labels[np.argmax(counts)]

    return prediction, nearest_labels


if __name__ == "__main__":
    X = np.array([[1, 2], [2, 2], [2, 1], [5, 5], [5, 6], [6, 5]])
    y = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    sample = np.array([2, 4])
    prediction, neighbors = knn_predict(X, y, sample, k=3)
    print("Neighbors:", neighbors)
    print("Predicted class:", prediction)
