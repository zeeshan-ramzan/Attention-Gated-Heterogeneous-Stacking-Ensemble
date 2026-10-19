"""
Experiment 7 -- Section 2.2: sensitivity of AG-StackNet to the |r| screening
threshold, and the leakage check (training-only screen vs. full-data screen).
"""
import numpy as np
import pandas as pd

import config as C
from data_utils import load_dataset, holdout_split, run_split, Preprocessor
from models_registry import MODELS

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]


def main():
    df = load_dataset()
    tr, te = holdout_split(df)
    rows = []
    for t in THRESHOLDS:
        r = run_split(MODELS["AG-StackNet"], df, tr, te, threshold=t)
        rows.append(dict(threshold=t, RMSE=r["metrics"]["RMSE"], R2=r["metrics"]["R2"],
                         n_features=len(r["features"]), features=";".join(r["features"])))
        print(f"|r| > {t:.2f}: RMSE = {rows[-1]['RMSE']:.2f} kg/ha ({rows[-1]['n_features']} columns)")
    tab = pd.DataFrame(rows)
    tab.to_csv(C.RESULTS_DIR / "threshold_sensitivity.csv", index=False)
    print(f"Spread across thresholds: {tab.RMSE.max() - tab.RMSE.min():.2f} kg/ha")

    # leakage check: which features does the screen keep on train only vs all data?
    X = df.drop(columns=[C.TARGET])
    p_tr = Preprocessor().fit(X.iloc[tr], df[C.TARGET].values[tr])
    p_all = Preprocessor().fit(X, df[C.TARGET].values)
    a, b = set(p_tr.selected_), set(p_all.selected_)
    print("Training-only screen keeps :", sorted(a))
    print("Full-data screen keeps     :", sorted(b))
    print("Identical" if a == b else f"Differences -> only train: {sorted(a - b)}, only full: {sorted(b - a)}")
    pd.DataFrame({"feature": list(p_all.correlations_),
                  "r_abs_full_data": list(p_all.correlations_.values()),
                  "r_abs_train_only": [p_tr.correlations_.get(k, np.nan) for k in p_all.correlations_]}
                 ).to_csv(C.RESULTS_DIR / "feature_screening_correlations.csv", index=False)


if __name__ == "__main__":
    main()
