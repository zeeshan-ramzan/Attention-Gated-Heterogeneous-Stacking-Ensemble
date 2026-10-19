"""Algorithm 4 -- Gradient Boosting regressor baseline (Table 6)."""
from sklearn.ensemble import GradientBoostingRegressor
import config as C
from data_utils import main_holdout


def build_model(seed=C.SEED):
    return GradientBoostingRegressor(random_state=seed, **C.BASELINES["gradient_boosting"])


if __name__ == "__main__":
    main_holdout(build_model, "GradientBoosting")
