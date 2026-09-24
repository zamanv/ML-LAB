import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

categories = ["alt.atheism", "comp.graphics", "sci.med", "soc.religion.christian"]

train = fetch_20newsgroups(subset="train", categories=categories,
                           remove=("headers", "footers", "quotes"), random_state=42)
test = fetch_20newsgroups(subset="test", categories=categories,
                          remove=("headers", "footers", "quotes"), random_state=42)

print("Train documents:", len(train.data), "| Test documents:", len(test.data))
print("Categories:", train.target_names)

mnb = Pipeline([
    ("vect", CountVectorizer(stop_words="english")),
    ("nb", MultinomialNB()),
])

bnb = Pipeline([
    ("vect", CountVectorizer(stop_words="english", binary=True)),
    ("nb", BernoulliNB()),
])

mnb.fit(train.data, train.target)
mnb_pred = mnb.predict(test.data)

bnb.fit(train.data, train.target)
bnb_pred = bnb.predict(test.data)

mnb_acc = accuracy_score(test.target, mnb_pred)
bnb_acc = accuracy_score(test.target, bnb_pred)
mnb_f1 = f1_score(test.target, mnb_pred, average="weighted")
bnb_f1 = f1_score(test.target, bnb_pred, average="weighted")

print(f"\nMultinomial NB -> Accuracy: {mnb_acc:.4f}  F1 (weighted): {mnb_f1:.4f}")
print(f"Bernoulli NB   -> Accuracy: {bnb_acc:.4f}  F1 (weighted): {bnb_f1:.4f}")

print("\nMultinomial NB classification report:")
print(classification_report(test.target, mnb_pred, target_names=train.target_names))
print("Bernoulli NB classification report:")
print(classification_report(test.target, bnb_pred, target_names=train.target_names))

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, [mnb_acc, mnb_f1], width, label="Multinomial")
ax.bar(x + width/2, [bnb_acc, bnb_f1], width, label="Bernoulli")
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "F1 (weighted)"])
ax.set_ylabel("Score")
ax.set_title("Multinomial vs Bernoulli Naive Bayes")
ax.legend()
plt.tight_layout()
plt.savefig("naive_bayes_comparison.png", dpi=150)
plt.show()