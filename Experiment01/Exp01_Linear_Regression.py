import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing(as_frame=True)
df = data.frame
X = df[["AveRooms"]].values
y = df["MedHouseVal"].values

print("Dataset shape:", df.shape)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

def predict(X, w, b):
    return X.dot(w) + b

def compute_cost(X, y, w, b):
    m = len(y)
    cost = (1 / (2 * m)) * np.sum((predict(X, w, b) - y) ** 2)
    return cost

def gradient_descent(X, y, w, b, lr, epochs):
    m = len(y)
    history = []
    for i in range(epochs):
        error = predict(X, w, b) - y
        dw = (1 / m) * X.T.dot(error)
        db = (1 / m) * np.sum(error)
        w = w - lr * dw
        b = b - lr * db
        history.append(compute_cost(X, y, w, b))
    return w, b, history

w = np.zeros(1)
b = 0.0
lr = 0.1
epochs = 500
w_gd, b_gd, cost_hist = gradient_descent(X_train_s, y_train, w, b, lr, epochs)

print("Gradient Descent params: w =", w_gd, ", b =", b_gd)
print("Final training cost:", cost_hist[-1])

X_design = np.c_[np.ones(X_train_s.shape[0]), X_train_s]
w_normal = np.linalg.pinv(X_design.T.dot(X_design)).dot(X_design.T).dot(y_train)
b_normal = w_normal[0]
w_normal_slope = w_normal[1]
print("Normal Equation params: w =", w_normal_slope, ", b =", b_normal)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

y_pred_gd = predict(X_test_s, w_gd, b_gd)
y_pred_ne = X_test_s.dot(w_normal_slope) + b_normal

print("Gradient Descent  -> MSE:", mse(y_test, y_pred_gd), " R2:", r2(y_test, y_pred_gd))
print("Normal Equation   -> MSE:", mse(y_test, y_pred_ne), " R2:", r2(y_test, y_pred_ne))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), cost_hist, color="blue")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Cost History")

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.4, s=8, label="Data points")
xs = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
xs_s = scaler.transform(xs)
plt.plot(xs, predict(xs_s, w_gd, b_gd), color="red", label="GD fit")
plt.plot(xs, xs_s.dot(w_normal_slope) + b_normal, color="green", linestyle="--", label="Normal eq. fit")
plt.xlabel("AveRooms")
plt.ylabel("MedHouseVal")
plt.title("Linear Fit: Rooms vs House Price")
plt.legend()

plt.tight_layout()
plt.savefig("linear_fit.png", dpi=150)
plt.show()