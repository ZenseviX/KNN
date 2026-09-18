import numpy as np

# ข้อมูลฝึก
X = np.array([
    [1, 2],
    [2, 2],
    [2, 1],
    [5, 5],
    [5, 6],
    [6, 5]
])

# Label
y = np.array(['A', 'A', 'A', 'B', 'B', 'B'])


# ฟังก์ชัน KNN
def knn_predict(X, y, sample, k=3):
    # 1. คำนวณระยะห่าง
    distances = np.linalg.norm(X - sample, axis=1)

    # 2. หา k จุดที่ใกล้ที่สุด
    nearest = np.argsort(distances)[:k]

    # 3. เอา Label ของจุดที่ใกล้ที่สุด
    nearest_labels = y[nearest]

    # 4. นับจำนวนแต่ละ Class
    labels, counts = np.unique(nearest_labels, return_counts=True)

    # 5. เลือก Class ที่มีจำนวนมากที่สุด
    prediction = labels[np.argmax(counts)]

    return prediction, nearest_labels


# ข้อมูลใหม่
sample = np.array([2, 4])

# Function
prediction, neighbors = knn_predict(X, y, sample, k=3)

print("Neighbors:", neighbors)
print("Predicted class:", prediction)