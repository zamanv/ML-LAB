import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

data = load_iris()
X = data.data[:, :2]
y = (data.target == 0).astype(int)

print("Setosa vs Non-Setosa binary labels. Class distribution:")
print(np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svm = SVC(kernel="linear", C=1.0)
svm.fit(X_train_s, y_train)

y_pred = svm.predict(X_test_s)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Non-Setosa", "Setosa"]))

w = svm.coef_[0]
b = svm.intercept_[0]
margin = 1.0 / np.linalg.norm(w)
print(f"\nWeight vector w: {w}")
print(f"Intercept b: {b}")
print(f"Margin width: {margin:.4f}")
print(f"Support vectors: {len(svm.support_)}")

xx, yy = np.meshgrid(np.linspace(X_train_s[:, 0].min() - 0.5, X_train_s[:, 0].max() + 0.5, 300),
                     np.linspace(X_train_s[:, 1].min() - 0.5, X_train_s[:, 1].max() + 0.5, 300))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(9, 7))
plt.contourf(xx, yy, Z, levels=10, cmap="RdBu", alpha=0.6)
contour = plt.contour(xx, yy, Z, levels=[-1, 0, 1],
                      colors=["orange", "black", "orange"],
                      linestyles=["--", "-", "--"], linewidths=[1.5, 2, 1.5])
plt.clabel(contour, fmt={-1: "margin", 0: "decision", 1: "margin"}, fontsize=10,
           inline=True, colors="black")

plt.scatter(X_train_s[:, 0], X_train_s[:, 1], c=y_train, cmap="bwr",
            edgecolors="k", s=40, label="Train data")
plt.scatter(X_train_s[svm.support_, 0], X_train_s[svm.support_, 1],
            s=160, facecolors="none", edgecolors="green", linewidths=2,
            label="Support vectors")
plt.xlabel("Sepal length (scaled)")
plt.ylabel("Sepal width (scaled)")
plt.title(f"Linear SVM - Setosa vs Non-Setosa (margin = {margin:.3f})")
plt.legend()
plt.savefig("svm_decision_boundary.png", dpi=150)
plt.show()