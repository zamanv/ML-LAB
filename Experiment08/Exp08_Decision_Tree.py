import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
try:
    df = pd.read_csv(url, header=None, names=columns)
except Exception:
    df = pd.read_csv("car.data", header=None, names=columns)

print("Dataset shape:", df.shape)
print(df["class"].value_counts())

encoders = {}
df_enc = df.copy()
for col in columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df_enc.drop(columns=["class"]).values
y = df_enc["class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
target_names = encoders["class"].classes_
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

print("\nFeature Importances:")
feat_names = columns[:-1]
for name, imp in sorted(zip(feat_names, clf.feature_importances_),
                        key=lambda x: x[1], reverse=True):
    print(f"  {name:10s}: {imp:.4f}")

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feat_names, class_names=target_names,
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree (Entropy) - Car Evaluation")
plt.savefig("car_decision_tree.png", dpi=120, bbox_inches="tight")
plt.show()