"""
Experiment 8 -- Section 2.2: outlier assessment with the 1.5 x IQR rule on each
standardised numeric feature, and the effect of excluding flagged records from
TRAINING (the test partition is kept identical so the RMSE is comparable).
"""
import numpy as np
import pandas as pd

import config as C
from data_utils import load_dataset, holdout_split, run_split
from models_registry import MODELS


def iqr_flags(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def main():
    df = load_dataset()
    num = [c for c in df.columns if c not in C.CATEGORICAL + ["latitude", "longitude", "season"]
           and np.issubdtype(df[c].dtype, np.number)]
    Z = (df[num] - df[num].mean()) / df[num].std()
    F = pd.DataFrame({c: iqr_flags(Z[c]) for c in num})
    per = F.sum().sort_values(ascending=False)
    rec = F.any(axis=1)
    feats = [c for c in num if c != C.TARGET]
    max_z = Z[feats][F[feats]].abs().max().max()
    print(per.to_string())
    print(f"\nTotal flagged values (covariates): {int(F[feats].values.sum())}; "
          f"records affected: {int(F[feats].any(axis=1).sum())} of {len(df)} "
          f"({100 * F[feats].any(axis=1).mean():.1f}%)")
    print(f"Features without flags: {[c for c in feats if per[c] == 0]}")
    print(f"Most extreme flagged value: {max_z:.2f} SD from the feature mean")
    print(f"Target (yield) records flagged: {int(F[C.TARGET].sum())}; "
          f"yield range {df[C.TARGET].min():.0f}-{df[C.TARGET].max():.0f} kg/ha")
    per.to_frame("n_flagged").to_csv(C.RESULTS_DIR / "outliers_per_feature.csv")

    tr, te = holdout_split(df)
    keep_tr = np.array([i for i in tr if not rec.iloc[i]])
    full = run_split(MODELS["AG-StackNet"], df, tr, te)["metrics"]["RMSE"]
    excl = run_split(MODELS["AG-StackNet"], df, keep_tr, te)["metrics"]["RMSE"]
    print(f"AG-StackNet holdout RMSE: retained {full:.2f} kg/ha; flagged training records excluded "
          f"({len(tr) - len(keep_tr)} removed) {excl:.2f} kg/ha")


if __name__ == "__main__":
    main()
