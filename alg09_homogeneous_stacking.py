"""Algorithm 9 -- Homogeneous stacking baseline (MS-YieldStackNet, Ali et al., 2026).

Three identical feed-forward networks (64/32, ReLU) trained on the concatenated
multi-source feature vector, combined by a random-forest meta-learner fitted on
OUT-OF-FOLD level-0 predictions. The base networks differ only in their random
seed. If you have the original MS-YieldStackNet code, prefer it and report that
the baseline was re-executed from it (Section 2.4 of the manuscript).
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

import config as C
from data_utils import main_holdout
from torch_utils import FFNNRegressor


class HomogeneousStacking:
    def __init__(self, seed=C.SEED, n_base=3, hidden=(64, 32), meta_rf=None, folds=5, ffnn=None):
        self.seed, self.n_base, self.hidden, self.folds = seed, n_base, hidden, folds
        self.meta_rf = meta_rf or dict(n_estimators=400, max_depth=15)
        self.ffnn = ffnn or C.BASELINES["ffnn"]

    def _base(self, k):
        f = self.ffnn
        return FFNNRegressor(hidden=self.hidden, lr=f["lr"], batch_size=f["batch_size"],
                             epochs=f["epochs"], patience=f["patience"], seed=self.seed + 101 * (k + 1))

    def fit(self, X, y):
        X, y = np.asarray(X, np.float32), np.asarray(y, float)
        oof = np.zeros((len(X), self.n_base))
        for tr, va in KFold(self.folds, shuffle=True, random_state=self.seed).split(X):
            for k in range(self.n_base):
                oof[va, k] = self._base(k).fit(X[tr], y[tr]).predict(X[va])
        self.oof_base_ = oof
        self.meta_ = RandomForestRegressor(random_state=self.seed, n_jobs=-1, **self.meta_rf).fit(oof, y)
        self.meta_.set_params(n_jobs=1)   # serial prediction -> reproducible
        self.bases_ = [self._base(k).fit(X, y) for k in range(self.n_base)]
        return self

    def base_predictions(self, X):
        X = np.asarray(X, np.float32)
        return np.column_stack([b.predict(X) for b in self.bases_])

    def predict(self, X):
        return self.meta_.predict(self.base_predictions(X))


def build_model(seed=C.SEED):
    p = C.BASELINES["homogeneous_stacking"]
    return HomogeneousStacking(seed=seed, n_base=p["n_base"], hidden=p["hidden"],
                               meta_rf=p["meta_rf"], folds=p["folds"])


if __name__ == "__main__":
    main_holdout(build_model, "HomogeneousStacking")
