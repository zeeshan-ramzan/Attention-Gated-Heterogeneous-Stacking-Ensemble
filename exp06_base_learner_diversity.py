"""
Experiment 6 -- Table 12 (Section 3.5): diversity of the base-learner tier.

For the heterogeneous tier (AG-StackNet: FFNN, XGBoost, RF) and the homogeneous
tier (baseline: 3 x FFNN) on the held-out partition:
  * pairwise Pearson correlation of prediction ERRORS
  * mean pairwise correlation of PREDICTIONS
  * ensemble ambiguity (Krogh & Vedelsby, 1995): mean squared deviation of the
    individual predictions from the ensemble mean, in kg^2/ha^2
"""
import itertools

import numpy as np
import pandas as pd

import config as C
from data_utils import load_dataset, holdout_split, run_split
from models_registry import MODELS


def diversity(P, y):
    E = P - y[:, None]
    pairs = list(itertools.combinations(range(P.shape[1]), 2))
    err = {f"M{i + 1} vs. M{j + 1}": np.corrcoef(E[:, i], E[:, j])[0, 1] for i, j in pairs}
    pred = np.mean([np.corrcoef(P[:, i], P[:, j])[0, 1] for i, j in pairs])
    amb = np.mean((P - P.mean(1, keepdims=True)) ** 2)
    return err, np.mean(list(err.values())), pred, amb


def main():
    df = load_dataset()
    tr, te = holdout_split(df)
    het = run_split(MODELS["AG-StackNet"], df, tr, te, return_model=True)
    hom = run_split(MODELS["Homogeneous Stacking"], df, tr, te, return_model=True)
    y = het["y_true"]
    out = {}
    for tag, r in (("Heterogeneous tier", het), ("Homogeneous tier", hom)):
        P = r["model"].base_predictions(r["X_test"])
        err, merr, mpred, amb = diversity(P, y)
        out[tag] = {**err, "Mean pairwise error correlation": merr,
                    "Mean pairwise prediction correlation": mpred,
                    "Ensemble ambiguity (kg2/ha2)": amb}
        for k in range(P.shape[1]):
            out[tag][f"M{k + 1} RMSE (kg/ha)"] = float(np.sqrt(np.mean((P[:, k] - y) ** 2)))
    tab = pd.DataFrame(out)
    tab.to_csv(C.RESULTS_DIR / "table12_diversity.csv")
    print(tab.round(3).to_string())
    a, b = out["Heterogeneous tier"], out["Homogeneous tier"]
    print(f"\nMean error correlation {b['Mean pairwise error correlation']:.2f} (homogeneous) -> "
          f"{a['Mean pairwise error correlation']:.2f} (heterogeneous); ambiguity "
          f"{b['Ensemble ambiguity (kg2/ha2)']:.1f} -> {a['Ensemble ambiguity (kg2/ha2)']:.1f} kg2/ha2")


if __name__ == "__main__":
    main()
