"""Algorithm 5 -- XGBoost regressor baseline (Table 6)."""
from xgboost import XGBRegressor
import config as C
from data_utils import main_holdout


def build_model(seed=C.SEED):
    return XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist", **C.BASELINES["xgboost"])


if __name__ == "__main__":
    main_holdout(build_model, "XGBoost")
