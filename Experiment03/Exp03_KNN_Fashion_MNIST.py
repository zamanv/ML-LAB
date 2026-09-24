import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

X, y = fetch_openml("fashion-mnist", version=1, as_frame=False, parser="auto")
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000,
                                                    random_state=42, stratify=y)

X_train = X_train / 255.0
X_test = X_test / 255.0

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

k_values = [1, 3, 5, 7, 10, 15, 20]
accuracies = []
times = []

for k in k_values:
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start
    accuracies.append(acc)
    times.append(elapsed)
    print(f"K = {k:2d} -> Accuracy: {acc:.4f}  Time: {elapsed:.2f}s")

best_k = k_values[int(np.argmax(accuracies))]
print("\nBest K:", best_k, "with accuracy:", max(accuracies))

knn_best = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)
print("\nClassification report (K =", best_k, "):")
print(classification_report(y_test, y_pred_best))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(k_values, accuracies, marker="o", color="blue")
ax1.set_xlabel("K")
ax1.set_ylabel("Test Accuracy")
ax1.set_title("Accuracy vs K")
ax2.plot(k_values, times, marker="o", color="red")
ax2.set_xlabel("K")
ax2.set_ylabel("Runtime (s)")
ax2.set_title("Prediction Time vs K")
plt.tight_layout()
plt.savefig("knn_analysis.png", dpi=150)
plt.show()