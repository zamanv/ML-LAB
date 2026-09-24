import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
try:
    df = pd.read_csv(url, delim_whitespace=True, header=None,
                     names=["mpg", "cyl", "disp", "hp", "weight", "accel",
                            "model_year", "origin", "name"])
except Exception:
    df = pd.read_csv("auto-mpg.csv")

df = df.replace("?", np.nan)
df["hp"] = pd.to_numeric(df["hp"], errors="coerce")
df = df.dropna()

X = df[["disp"]].values
y = df["mpg"].values

print("Dataset shape after cleaning:", df.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = [1, 2, 3, 5, 9]
models = {}
results = []

for deg in degrees:
    model = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
    model.fit(X_train, y_train)
    models[deg] = model
    y_pred = model.predict(X_test)
    results.append({
        "degree": deg,
        "train_mse": mean_squared_error(y_train, model.predict(X_train)),
        "test_mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for deg, model in models.items():
    xs = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    plt.plot(xs, model.predict(xs), label=f"degree={deg}")
plt.scatter(X_train, y_train, alpha=0.3, s=8, color="black", label="data")
plt.xlabel("Displacement (cubic inches)")
plt.ylabel("MPG")
plt.title("Polynomial Fits on Auto MPG")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(res_df["degree"], res_df["test_mse"], marker="o", label="Test MSE")
plt.plot(res_df["degree"], res_df["train_mse"], marker="s", label="Train MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("MSE vs Degree (overfitting check)")
plt.legend()
plt.xticks(degrees)

plt.tight_layout()
plt.savefig("poly_fit.png", dpi=150)
plt.show()