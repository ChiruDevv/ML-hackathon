"""Model zoo for candidate scoring.

Owner: Meenakshi P (PES2UG24CS811)

Every model maps one (fix, candidate) feature row to P(candidate is the
reference arc). Models from seven algorithm families are compared.

Training responsibility (each member trains their models on their own laptop
with ``scripts/03_train_compare.py --models ...``):
  Harsh Mannan       - LogisticRegression, GaussianNB
  Chiranthan Shankar - KNN, DecisionTree
  Meenakshi P        - RandomForest, XGBoost (+ Bayesian optimisation)
  Hansika R          - LightGBM, MLP
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42

# name -> (family, constructor, human-readable key configuration, max training rows)
MODEL_SPECS = {
    "LogisticRegression": (
        "Linear / probabilistic",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "StandardScaler + L2, C=1.0, lbfgs", None),
    "GaussianNB": (
        "Probabilistic (generative)",
        lambda: make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-9)),
        "var_smoothing=1e-9", None),
    "KNN": (
        "Instance-based",
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2)),
        "k=25, distance-weighted, Euclidean on scaled features, 60k-row training sample", 60_000),
    "DecisionTree": (
        "Tree-based",
        lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED),
        "max_depth=12, min_samples_leaf=50", None),
    "RandomForest": (
        "Ensemble (bagging)",
        lambda: RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=5,
                                       max_samples=0.5, n_jobs=2, random_state=SEED),
        "100 trees, max_depth=18, min_leaf=5, max_samples=0.5", None),
    "XGBoost": (
        "Boosting",
        lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist", n_jobs=2,
                              eval_metric="logloss", random_state=SEED),
        "300 trees, depth=6, lr=0.1, subsample=0.8", None),
    "LightGBM": (
        "Boosting",
        lambda: LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31,
                               n_jobs=2, random_state=SEED, verbose=-1),
        "300 trees, lr=0.1, num_leaves=31 (library defaults otherwise)", None),
    "MLP": (
        "Neural network",
        None,  # trained epoch by epoch, see train_mlp
        "StandardScaler + MLP(64,32), ReLU, Adam lr=1e-3, batch=1024, 25 epochs", None),
}


class MLPWrapper:
    """Scaler + MLP trained with partial_fit so train/val loss can be logged per epoch."""

    def __init__(self, hidden=(64, 32), epochs=25, lr=1e-3, batch=1024, alpha=1e-4, seed=SEED):
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden, learning_rate_init=lr, batch_size=batch,
                                 alpha=alpha, random_state=seed, max_iter=1)
        self.epochs = epochs
        self.history = {"train_loss": [], "val_loss": []}
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.metrics import log_loss
        Xs = self.scaler.fit_transform(X)
        Xv = self.scaler.transform(X_val) if X_val is not None else None
        for ep in range(self.epochs):
            idx = self.rng.permutation(len(Xs))
            for s in range(0, len(idx), 50_000):
                b = idx[s:s + 50_000]
                self.mlp.partial_fit(Xs[b], y[b], classes=[0, 1])
            self.history["train_loss"].append(float(log_loss(y, self.mlp.predict_proba(Xs)[:, 1], labels=[0, 1])))
            if Xv is not None:
                self.history["val_loss"].append(float(log_loss(y_val, self.mlp.predict_proba(Xv)[:, 1], labels=[0, 1])))
        return self

    def predict_proba(self, X):
        return self.mlp.predict_proba(self.scaler.transform(X))


def build(name):
    fam, ctor, conf, cap = MODEL_SPECS[name]
    return MLPWrapper() if name == "MLP" else ctor()
