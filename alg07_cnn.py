"""Algorithm 7 -- 1-D CNN baseline.

Each field-season record (p standardised features, no sub-seasonal time steps) is
reshaped to a p x 1 pseudo-sequence whose order is the fixed column order of the
feature matrix; kernels therefore convolve over adjacent columns, not dates
(Section 2.4). Kept only for continuity with Ali et al. (2026).
"""
import config as C
from data_utils import main_holdout
from torch_utils import CNNRegressor


def build_model(seed=C.SEED):
    p = C.BASELINES["cnn"]
    return CNNRegressor(filters=p["filters"], kernel=p["kernel"], conv_layers=p["conv_layers"],
                        dense=p["dense"], lr=p["lr"], batch_size=p["batch_size"],
                        epochs=p["epochs"], patience=p["patience"], seed=seed)


if __name__ == "__main__":
    main_holdout(build_model, "CNN")
