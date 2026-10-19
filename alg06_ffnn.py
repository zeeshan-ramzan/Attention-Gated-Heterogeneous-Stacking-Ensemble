"""Algorithm 6 -- Feed-forward neural network baseline (2 hidden layers, 64/32, ReLU, Adam)."""
import config as C
from data_utils import main_holdout
from torch_utils import FFNNRegressor


def build_model(seed=C.SEED):
    p = C.BASELINES["ffnn"]
    return FFNNRegressor(hidden=p["hidden"], lr=p["lr"], batch_size=p["batch_size"],
                         epochs=p["epochs"], patience=p["patience"], seed=seed)


if __name__ == "__main__":
    main_holdout(build_model, "FFNN")
