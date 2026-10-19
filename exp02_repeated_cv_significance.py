"""
Experiment 2 -- Table 8 and Section 3.2: repeated stratified 5-fold CV
(CV_REPEATS x CV_FOLDS fold estimates), with every pipeline component (screening,
scaling, model) re-fitted inside each training fold.

Outputs
  results/cv_fold_results.csv   one row per (model, repeat, fold)  -- resumable
  results/table8_cv.csv         mean +/- SD, 95% CI, paired tests vs AG-StackNet
  results/section3_2_stats.txt  the numbers quoted in the text (incl. holdout bootstrap)
"""
import argparse

import numpy as np
import pandas as pd

import config as C
import stats_utils as S
from data_utils import load_dataset, repeated_stratified_kfold, run_split, holdout_split
from models_registry import MODELS

DEFAULT = ["AG-StackNet", "Homogeneous Stacking", "LSTM", "CNN", "FFNN", "XGBoost"]
REF = "AG-StackNet"


def run_folds(models, df, out_csv):
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["model", "repeat", "fold"])
    have = set(zip(done.model, done.repeat, done.fold))
    for r, f, tr, te in repeated_stratified_kfold(df):
        for name in models:
            if (name, r, f) in have:
                continue
            res = run_split(MODELS[name], df, tr, te)
            row = dict(model=name, repeat=r, fold=f, n_train=len(tr), n_test=len(te), **res["metrics"])
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            print(f"repeat {r} fold {f} {name:22s} RMSE={res['metrics']['RMSE']:.2f}")
    return pd.read_csv(out_csv)


def summarise(folds):
    folds = folds.sort_values(["repeat", "fold"])
    ref = folds[folds.model == REF].set_index(["repeat", "fold"])
    n_tr, n_te = ref.n_train.mean(), ref.n_test.mean()
    rows, lines = [], []
    for name, g in folds.groupby("model", sort=False):
        g = g.set_index(["repeat", "fold"]).loc[ref.index]
        lo, hi = S.mean_ci(g.RMSE)
        row = dict(Model=name, RMSE_mean=g.RMSE.mean(), RMSE_sd=g.RMSE.std(ddof=1), CI_low=lo, CI_high=hi,
                   R2_mean=g.R2.mean(), R2_sd=g.R2.std(ddof=1), MAE_mean=g.MAE.mean(), J=len(g))
        if name != REF:
            # difference defined as AG-StackNet minus comparator (negative = AG-StackNet better)
            md, clo, chi, t, p_t = S.corrected_resampled_t(ref.RMSE.values, g.RMSE.values, n_tr, n_te)
            w, p_w = S.wilcoxon(ref.RMSE.values, g.RMSE.values)
            dz = S.cohens_dz(g.RMSE.values, ref.RMSE.values)
            wins = int((ref.RMSE.values < g.RMSE.values).sum())
            row.update(dRMSE=md, dRMSE_CI_low=clo, dRMSE_CI_high=chi, corr_t=t, p_corr_t=p_t,
                       wilcoxon_W=w, p_wilcoxon=p_w, cohens_dz=dz, folds_won_by_AG=wins)
            if name == "Homogeneous Stacking":
                diffs = g.RMSE.values - ref.RMSE.values
                lines += [
                    f"J = {len(g)} fold estimates ({folds.repeat.nunique()} repeats x {folds.fold.nunique()} folds); "
                    f"seeds = {C.CV_SEEDS}",
                    f"AG-StackNet RMSE = {ref.RMSE.mean():.2f} +/- {ref.RMSE.std(ddof=1):.2f}; "
                    f"baseline = {g.RMSE.mean():.2f} +/- {g.RMSE.std(ddof=1):.2f} kg/ha",
                    f"Mean fold-level difference (baseline - AG) = {diffs.mean():.2f} +/- {diffs.std(ddof=1):.2f} kg/ha",
                    f"Wilcoxon W = {w:.1f}, p = {p_w:.3g}",
                    f"Cohen's d_z = {dz:.2f} ({S.interpret_dz(dz)})",
                    f"AG-StackNet lower error in {wins} of {len(g)} folds",
                    f"Corrected resampled t (Nadeau-Bengio): t = {t:.2f}, p = {p_t:.3g}, "
                    f"95% CI of (AG - baseline) = [{clo:.2f}, {chi:.2f}] kg/ha",
                    f"Significant at 5%: {'yes' if p_t < 0.05 and p_w < 0.05 else 'no'}",
                ]
        rows.append(row)
    return pd.DataFrame(rows), lines


def holdout_bootstrap(df):
    f = C.RESULTS_DIR / "holdout_predictions_all_models.csv"
    if not f.exists():
        tr, te = holdout_split(df)
        a = run_split(MODELS["AG-StackNet"], df, tr, te)
        b = run_split(MODELS["Homogeneous Stacking"], df, tr, te)
        y, pa, pb = a["y_true"], a["y_pred"], b["y_pred"]
    else:
        p = pd.read_csv(f)
        y, pa, pb = p.y_true, p["AG-StackNet"], p["Homogeneous Stacking"]
    full, lo, hi = S.paired_bootstrap_rmse_diff(y, pa, pb, reps=C.BOOTSTRAP_REPS, seed=C.SEED)
    return f"Holdout RMSE difference (baseline - AG) = {full:.2f} kg/ha, paired bootstrap 95% CI " \
           f"[{lo:.2f}, {hi:.2f}] ({C.BOOTSTRAP_REPS} replicates)"


def main(models=DEFAULT):
    df = load_dataset()
    out_csv = C.RESULTS_DIR / "cv_fold_results.csv"
    folds = run_folds(models, df, out_csv)
    folds = folds[folds.model.isin(models)]
    tab, lines = summarise(folds)
    lines.append(holdout_bootstrap(df))
    tab.to_csv(C.RESULTS_DIR / "table8_cv.csv", index=False)
    (C.RESULTS_DIR / "section3_2_stats.txt").write_text("\n".join(lines))
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string(index=False))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT)
    main(ap.parse_args().models)
