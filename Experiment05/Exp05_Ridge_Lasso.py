import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X, y = data.data, data.target

print("Dataset shape:", X.shape)
print("Features:", data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

def report(name, model):
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:12s} -> MSE: {mse:8.2f}  R2: {r2:.4f}")
    return mse, r2

lr_mse, lr_r2 = report("Linear", LinearRegression())

ridge_params = {"alpha": np.logspace(-3, 3, 50)}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring="neg_mean_squared_error")
ridge_grid.fit(X_train_s, y_train)
best_ridge = ridge_grid.best_estimator_
ridge_pred = best_ridge.predict(X_test_s)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
print(f"Ridge        -> MSE: {ridge_mse:8.2f}  R2: {ridge_r2:.4f}  (best alpha = {ridge_grid.best_params_['alpha']:.4f})")

lasso_params = {"alpha": np.logspace(-4, 0, 50)}
lasso_grid = GridSearchCV(Lasso(max_iter=100000), lasso_params, cv=5,
                          scoring="neg_mean_squared_error")
lasso_grid.fit(X_train_s, y_train)
best_lasso = lasso_grid.best_estimator_
lasso_pred = best_lasso.predict(X_test_s)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
print(f"Lasso        -> MSE: {lasso_mse:8.2f}  R2: {lasso_r2:.4f}  (best alpha = {lasso_grid.best_params_['alpha']:.4f})")

print("\nLasso coefficients (non-zero features retained):")
for name, coef in zip(data.feature_names, best_lasso.coef_):
    print(f"  {name:10s}: {coef:8.4f}")

labels = ["Linear", "Ridge", "Lasso"]
mses = [lr_mse, ridge_mse, lasso_mse]
r2s = [lr_r2, ridge_r2, lasso_r2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.bar(labels, mses, color=["gray", "blue", "green"])
ax1.set_ylabel("Test MSE")
ax1.set_title("MSE Comparison")
ax2.bar(labels, r2s, color=["gray", "blue", "green"])
ax2.set_ylabel("Test R-squared")
ax2.set_title("R-squared Comparison")
plt.tight_layout()
plt.savefig("ridge_lasso_comparison.png", dpi=150)
plt.show()