"""
Experiment 1 -- Table 7: all ten models on the identical 80/20 holdout (seed 42).
Also prints the relative RMSE/MAE reductions of AG-StackNet vs. homogeneous stacking
and saves every model's test predictions (used by exp11 for Figures 2-4).
"""
import numpy as np
import pandas as pd

import config as C
from data_utils import load_dataset, holdout_split, run_split
from models_registry import MODELS


def main(models=None):
    df = load_dataset()
    tr, te = holdout_split(df)
    print(f"n = {len(df)} records ({df.attrs.get('n_dropped_null', 0)} dropped for nulls); "
          f"train = {len(tr)}, test = {len(te)}")
    rows, preds = [], {"y_true": df.iloc[te][C.TARGET].values}
    for name, build in (models or MODELS).items():
        res = run_split(build, df, tr, te)
        m = res["metrics"]
        rows.append(dict(Model=name, **{k: m[k] for k in ["MAE", "MSE", "RMSE", "MAPE", "R2", "NSE"]},
                         fit_seconds=res["fit_seconds"]))
        preds[name] = res["y_pred"]
        print(f"{name:22s} MAE={m['MAE']:8.2f}  MSE={m['MSE']:10.2f}  RMSE={m['RMSE']:7.2f}  "
              f"MAPE={m['MAPE']:5.2f}  R2={m['R2']:.3f}")
    tab = pd.DataFrame(rows)
    tab.to_csv(C.RESULTS_DIR / "table7_holdout.csv", index=False)
    pd.DataFrame(preds).to_csv(C.RESULTS_DIR / "holdout_predictions_all_models.csv", index=False)

    if {"AG-StackNet", "Homogeneous Stacking"} <= set(tab.Model):
        a = tab.set_index("Model").loc["AG-StackNet"]
        b = tab.set_index("Model").loc["Homogeneous Stacking"]
        sd = df[C.TARGET].std()
        print("\nAG-StackNet vs homogeneous stacking (holdout):")
        print(f"  RMSE {b.RMSE:.2f} -> {a.RMSE:.2f}: -{b.RMSE - a.RMSE:.2f} kg/ha "
              f"(relative reduction {100 * (b.RMSE - a.RMSE) / b.RMSE:.2f}%)")
        print(f"  MAE  {b.MAE:.2f} -> {a.MAE:.2f}: relative reduction {100 * (b.MAE - a.MAE) / b.MAE:.2f}%")
        print(f"  MAPE {b.MAPE:.2f} -> {a.MAPE:.2f};  R2 {b.R2:.3f} -> {a.R2:.3f}")
        print(f"  RMSE / SD(yield): {b.RMSE / sd:.2f} -> {a.RMSE / sd:.2f}")
    return tab


if __name__ == "__main__":
    main()
