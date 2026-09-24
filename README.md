# ML LAB

Machine Learning lab experiments — regression, classification, and probability estimation implemented in Python with scikit-learn, NumPy, and matplotlib.

## Experiments

| # | Topic | Dataset | Files |
|---|-------|---------|-------|
| 01 | Linear Regression (Gradient Descent & Normal Equation) | California Housing | `Experiment01/` |
| 02 | Polynomial Regression vs Linear Regression | Auto MPG | `Experiment02/` |
| 03 | K-Nearest Neighbors image classification | Fashion MNIST | `Experiment03/` |
| 04 | Logistic Regression with/without feature scaling | Pima Indians Diabetes | `Experiment04/` |
| 05 | Ridge & Lasso Regularization | Diabetes (sklearn) | `Experiment05/` |
| 06 | Multinomial vs Bernoulli Naive Bayes | 20 Newsgroups | `Experiment06/` |
| 07 | MLE & MAP estimation of survival probability | Titanic | `Experiment07/` |
| 08 | Decision Tree with entropy criterion | Car Evaluation | `Experiment08/` |
| 09 | Linear SVM decision boundary & margin | Iris | `Experiment09/` |

Each experiment folder contains:
- `Exp*_*.py` — the complete program
- `Exp*_*.txt` — question, algorithm, and result analysis

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn seaborn scipy
```

## Running

```bash
python Experiment01/Exp01_Linear_Regression.py
```

Note: Experiments 2, 3, 6, and 8 download their datasets on first run; Experiment 3 (KNN on Fashion MNIST) is compute-heavy and may take several minutes.
