"""
Experiment 3 -- Tables 9 and 10 (Section 3.3): generalisation to unseen seasons
and unseen locations.

  * leave-one-season-out (LOSO): each season withheld in turn
  * chronological: train on all earlier seasons, test on the last season
    (NB: this is the same split as the last LOSO fold, so the numbers must coincide)
  * field-grouped 5-fold CV: every record of a latitude-longitude location in one fold
  * AG-StackNet without spatio-temporal inputs under LOSO and grouped CV
"""
import numpy as np
import pandas as pd

import config as C
from data_utils import (load_dataset, leave_one_season_out, chronological_split,
                        field_grouped_kfold, run_split, holdout_split)
from models_registry import MODELS

AG, HS = "AG-StackNet", "Homogeneous Stacking"
VARIANTS = [(AG, ()), (HS, ()), (AG + " w/o spatio-temporal", ("st",))]


def _build(name):
    return MODELS[name.replace(" w/o spatio-temporal", "")]


def main():
    df = load_dataset()
    rows, loso_rows = [], []

    # Random holdout for reference
    tr, te = holdout_split(df)
    for name, drop in VARIANTS[:2]:
        m = run_split(_build(name), df, tr, te, drop_modalities=drop)["metrics"]
        rows.append(dict(Protocol="Random 80/20 holdout", Model=name, RMSE=m["RMSE"], RMSE_sd=np.nan,
                         MAE=m["MAE"], R2=m["R2"]))

    # LOSO
    per = {}
    for season, tr, te in leave_one_season_out(df):
        for name, drop in VARIANTS:
            m = run_split(_build(name), df, tr, te, drop_modalities=drop)["metrics"]
            per.setdefault(name, []).append(m)
            if name in (AG, HS):
                loso_rows.append(dict(season=season, n_test=len(te), model=name, RMSE=m["RMSE"], R2=m["R2"]))
            print(f"LOSO {season} {name:35s} RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}")
    for name, ms in per.items():
        r = np.array([m["RMSE"] for m in ms])
        rows.append(dict(Protocol="Leave-one-season-out", Model=name, RMSE=r.mean(), RMSE_sd=r.std(ddof=1),
                         MAE=np.mean([m["MAE"] for m in ms]), R2=np.mean([m["R2"] for m in ms])))

    # Chronological
    tr, te, last = chronological_split(df)
    for name, drop in VARIANTS[:2]:
        m = run_split(_build(name), df, tr, te, drop_modalities=drop)["metrics"]
        rows.append(dict(Protocol=f"Chronological (test {last})", Model=name, RMSE=m["RMSE"], RMSE_sd=np.nan,
                         MAE=m["MAE"], R2=m["R2"]))

    # Field-grouped CV
    per = {}
    for f, tr, te in field_grouped_kfold(df):
        for name, drop in VARIANTS:
            m = run_split(_build(name), df, tr, te, drop_modalities=drop)["metrics"]
            per.setdefault(name, []).append(m)
            print(f"Grouped fold {f} {name:35s} RMSE={m['RMSE']:.2f}")
    for name, ms in per.items():
        r = np.array([m["RMSE"] for m in ms])
        rows.append(dict(Protocol="Field-grouped five-fold", Model=name, RMSE=r.mean(), RMSE_sd=r.std(ddof=1),
                         MAE=np.mean([m["MAE"] for m in ms]), R2=np.mean([m["R2"] for m in ms])))

    tab = pd.DataFrame(rows)
    base = tab[tab.Model == HS].set_index("Protocol").RMSE
    tab["dRMSE_vs_baseline"] = [r.RMSE - base.get(r.Protocol, np.nan) if r.Model != HS else np.nan
                                for r in tab.itertuples()]
    tab.to_csv(C.RESULTS_DIR / "table9_validation.csv", index=False)

    loso = pd.DataFrame(loso_rows).pivot_table(index=["season", "n_test"], columns="model", values="RMSE").reset_index()
    loso.to_csv(C.RESULTS_DIR / "table10_loso_per_season.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(2).to_string(index=False))
    print(loso.round(2).to_string(index=False))
    wins = int((loso[AG] < loso[HS]).sum())
    print(f"\nAG-StackNet better in {wins} of {len(loso)} withheld seasons. "
          f"LOSO mean +/- SD (sample SD): AG {loso[AG].mean():.2f} +/- {loso[AG].std(ddof=1):.2f}; "
          f"baseline {loso[HS].mean():.2f} +/- {loso[HS].std(ddof=1):.2f}")


if __name__ == "__main__":
    main()
