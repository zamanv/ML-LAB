import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, header=None, names=columns)

print("Dataset shape:", df.shape)
print(df.head())

zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop(columns=["Outcome"]).values
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic(X, y, lr=0.01, epochs=1000):
    X = np.c_[np.ones(X.shape[0]), X]
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        grad = (1 / n) * X.T.dot(sigmoid(X.dot(w)) - y)
        w = w - lr * grad
    return w

def predict_prob(X, w):
    X = np.c_[np.ones(X.shape[0]), X]
    return sigmoid(X.dot(w))

def predict(X, w):
    return (predict_prob(X, w) >= 0.5).astype(int)

def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1, cm

results = {}

print("\n=== WITHOUT feature scaling ===")
w_raw = train_logistic(X_train, y_train, lr=0.001, epochs=3000)
y_pred_raw = predict(X_test, w_raw)
acc_r, prec_r, rec_r, f1_r, cm_r = evaluate(y_test, y_pred_raw)
results["Without Scaling"] = (acc_r, prec_r, rec_r, f1_r, cm_r)
print("Accuracy:", acc_r, "| Precision:", prec_r, "| Recall:", rec_r, "| F1:", f1_r)
print("Confusion Matrix:\n", cm_r)

print("\n=== WITH feature scaling ===")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
w_scaled = train_logistic(X_train_s, y_train, lr=0.1, epochs=3000)
y_pred_scaled = predict(X_test_s, w_scaled)
acc_s, prec_s, rec_s, f1_s, cm_s = evaluate(y_test, y_pred_scaled)
results["With Scaling"] = (acc_s, prec_s, rec_s, f1_s, cm_s)
print("Accuracy:", acc_s, "| Precision:", prec_s, "| Recall:", rec_s, "| F1:", f1_s)
print("Confusion Matrix:\n", cm_s)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (label, (acc, prec, rec, f1, cm)) in zip(axes, results.items()):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{label}\nAcc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("logistic_comparison.png", dpi=150)
plt.show()