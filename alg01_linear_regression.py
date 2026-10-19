"""Algorithm 1 -- Linear Regression baseline (ordinary least squares, Table 6)."""
from sklearn.linear_model import LinearRegression
import config as C
from data_utils import main_holdout


def build_model(seed=C.SEED):
    return LinearRegression()


if __name__ == "__main__":
    main_holdout(build_model, "LinearRegression")
