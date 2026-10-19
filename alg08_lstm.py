"""Algorithm 8 -- LSTM baseline.

The recurrence runs over feature positions of the p x 1 pseudo-sequence, not over
time (Section 2.4). Kept only for continuity with Ali et al. (2026).
"""
import config as C
from data_utils import main_holdout
from torch_utils import LSTMRegressor


def build_model(seed=C.SEED):
    p = C.BASELINES["lstm"]
    return LSTMRegressor(units=p["units"], dense=p["dense"], lr=p["lr"], batch_size=p["batch_size"],
                         epochs=p["epochs"], patience=p["patience"], seed=seed)


if __name__ == "__main__":
    main_holdout(build_model, "LSTM")
