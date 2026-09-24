import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta as beta_dist

df = sns.load_dataset("titanic")
survived = df["survived"].dropna().astype(int).values

n = len(survived)
k = int(survived.sum())
print(f"Total passengers: {n}, Survivors: {k}")

mle = k / n
print(f"\nMLE estimate of survival probability: {mle:.4f}")

priors = [(1, 1), (2, 2), (5, 2), (2, 5)]
print("\nMAP estimates with different Beta priors:")
print(f"{'Prior':>10} | {'Prior mean':>10} | {'MAP estimate':>12}")
map_estimates = {}
for a, b in priors:
    prior_mean = a / (a + b)
    map_est = (k + a - 1) / (n + a + b - 2)
    map_estimates[(a, b)] = map_est
    print(f"Beta({a},{b}) | {prior_mean:10.4f} | {map_est:12.4f}")

theta = np.linspace(0, 1, 500)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
colors = ["red", "blue", "green", "orange"]
for (a, b), color in zip(priors, colors):
    plt.plot(theta, beta_dist.pdf(theta, a, b), color=color, label=f"Beta({a},{b})")
plt.axvline(mle, color="black", linestyle="--", label=f"MLE = {mle:.3f}")
plt.xlabel("Survival probability")
plt.ylabel("Density")
plt.title("Beta Priors")
plt.legend()

plt.subplot(1, 2, 2)
labels = ["MLE"] + [f"MAP Beta({a},{b})" for a, b in priors]
values = [mle] + [map_estimates[p] for p in priors]
plt.barh(labels, values, color=["black"] + colors)
plt.axvline(mle, color="black", linestyle="--")
plt.xlabel("Estimated survival probability")
plt.title("MLE vs MAP Estimates")
plt.tight_layout()
plt.savefig("mle_map_titanic.png", dpi=150)
plt.show()