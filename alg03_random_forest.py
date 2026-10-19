"""Algorithm 3 -- Random Forest regressor baseline (Table 6)."""
from sklearn.ensemble import RandomForestRegressor
import config as C
from data_utils import main_holdout


class _RF(RandomForestRegressor):
    """Parallel fit, serial predict (parallel prediction sums trees in a random order)."""
    def fit(self, X, y, **kw):
        super().fit(X, y, **kw)
        self.n_jobs = 1
        return self


def build_model(seed=C.SEED):
    return _RF(random_state=seed, n_jobs=-1, **C.BASELINES["random_forest"])


if __name__ == "__main__":
    main_holdout(build_model, "RandomForest")
