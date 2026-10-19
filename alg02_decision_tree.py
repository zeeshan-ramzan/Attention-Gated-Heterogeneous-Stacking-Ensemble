"""Algorithm 2 -- Decision Tree regressor baseline (Table 6)."""
from sklearn.tree import DecisionTreeRegressor
import config as C
from data_utils import main_holdout


def build_model(seed=C.SEED):
    return DecisionTreeRegressor(random_state=seed, **C.BASELINES["decision_tree"])


if __name__ == "__main__":
    main_holdout(build_model, "DecisionTree")
