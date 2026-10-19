"""
Experiment 11 -- Figures 2-7 and the diagnostics quoted in Section 3.8, all
generated from actual data / predictions (no approximations):

  Fig 2  RMSE bar chart                    <- results/table7_holdout.csv (run exp01 first)
  Fig 3  predicted vs observed, all test records
  Fig 4  residuals vs fitted, all test records
  Fig 5  histogram of the raw yield values (all records)
  Fig 6  training / validation loss of the gradient-trained part of AG-StackNet
         (encoders + gate + FFNN head), in kg^2/ha^2; the ridge meta-learner is closed-form
  Fig 7  |Pearson r| of each covariate with yield, dashed line at the 0.30 threshold
  + Breusch-Pagan test for AG-StackNet and the baseline, residual mean, tercile spread ratio,
    yield skewness, converged validation MSE.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew
from statsmodels.stats.diagnostic import het_breuschpagan

import config as C
from data_utils import load_dataset, holdout_split, run_split, Preprocessor
from models_registry import MODELS

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def bp_test(y, p):
    resid = y - p
    exog = np.column_stack([np.ones_like(p), p])
    lm, lm_p, _, _ = het_breuschpagan(resid, exog)
    return lm, lm_p


def main():
    df = load_dataset()
    tr, te = holdout_split(df)
    ag = run_split(MODELS["AG-StackNet"], df, tr, te, return_model=True)
    hs = run_split(MODELS["Homogeneous Stacking"], df, tr, te)
    y, p = ag["y_true"], ag["y_pred"]
    lines = []

    # Fig 2
    t7 = C.RESULTS_DIR / "table7_holdout.csv"
    if t7.exists():
        t = pd.read_csv(t7).sort_values("RMSE", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(t.Model, t.RMSE, color=["#c0392b" if m == "AG-StackNet" else "#7f8c8d" for m in t.Model])
        ax.set_xlabel("RMSE (kg/ha)")
        for i, v in enumerate(t.RMSE):
            ax.text(v + 1, i, f"{v:.2f}", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "agstack_fig2_rmse.png", dpi=300); plt.close(fig)

    # Fig 3
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, p, s=14, alpha=0.7)
    lo, hi = min(y.min(), p.min()) - 20, max(y.max(), p.max()) + 20
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed yield (kg/ha)"); ax.set_ylabel("Predicted yield (kg/ha)")
    ax.set_title(f"AG-StackNet, n = {len(y)} test records")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "agstack_fig3_scatter.png", dpi=300); plt.close(fig)

    # Fig 4
    r = y - p
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(p, r, s=14, alpha=0.7); ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Fitted yield (kg/ha)"); ax.set_ylabel("Residual (kg/ha)")
    ax.set_title(f"AG-StackNet residuals, n = {len(y)}")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "agstack_fig4_residual.png", dpi=300); plt.close(fig)

    # Fig 5
    yy = df[C.TARGET].values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(yy, bins=30, edgecolor="white")
    ax.set_xlabel("Wheat yield (kg/ha)"); ax.set_ylabel("Number of field-seasons")
    ax.set_title(f"Observed yield, n = {len(yy)}")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "agstack_fig5_histogram.png", dpi=300); plt.close(fig)

    # Fig 6 (loss in standardised units x y_sd^2 -> kg^2/ha^2)
    m = ag["model"]
    sd2 = m.level0_["y_sd"] ** 2
    h = m.history_
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.array(h["train_loss"]) * sd2, label="Training")
    ax.plot(np.array(h["val_loss"]) * sd2, label="Validation (inner early-stopping split)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (kg$^2$/ha$^2$)"); ax.legend()
    ax.set_title("Encoders + attention gate + FFNN head")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "agstack_fig6_training.png", dpi=300); plt.close(fig)
    best_val = np.min(h["val_loss"]) * sd2

    # Fig 7
    pre = Preprocessor().fit(df.drop(columns=[C.TARGET]), yy)
    corr = {}
    for c in df.columns:
        if c in (C.TARGET, "latitude", "longitude", "season") or c in C.CATEGORICAL:
            continue
        corr[c] = np.corrcoef(df[c].astype(float), yy)[0, 1]
    cs = pd.Series(corr).sort_values(key=np.abs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(cs.index, cs.abs(), color=["#2e86c1" if v > 0 else "#e67e22" for v in cs])
    ax.axvline(C.CORR_THRESHOLD, color="k", ls="--", lw=1)
    ax.set_xlabel("|Pearson r| with yield (blue: positive, orange: negative)")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "agstack_fig7_correlation.png", dpi=300); plt.close(fig)
    cs.to_csv(C.RESULTS_DIR / "covariate_correlations_full_data.csv", header=["r"])

    # Diagnostics
    lm_ag, p_ag = bp_test(y, p)
    lm_hs, p_hs = bp_test(hs["y_true"], hs["y_pred"])
    order = np.argsort(y)
    terc = np.array_split(order, 3)
    ratio = np.std(r[terc[2]], ddof=1) / np.std(r[terc[0]], ddof=1)
    lines += [
        f"Residual mean (AG-StackNet) = {r.mean():.2f} kg/ha",
        f"Breusch-Pagan LM: AG-StackNet = {lm_ag:.2f} (p = {p_ag:.3g}); baseline = {lm_hs:.2f} (p = {p_hs:.3g})",
        f"Residual SD upper / lower observed-yield tercile = {ratio:.2f}",
        f"Records above the 90th percentile of test yield ({np.percentile(y, 90):.0f} kg/ha): {int((y > np.percentile(y, 90)).sum())}",
        f"Yield skewness = {skew(yy):.2f} (mean {yy.mean():.2f}, SD {yy.std(ddof=1):.2f})",
        f"Best (converged) inner-validation MSE of the gradient-trained part = {best_val:.1f} kg2/ha2 "
        f"vs test MSE of full AG-StackNet = {ag['metrics']['MSE']:.1f}",
        "Strongest |r| with yield: " + ", ".join(f"{k} {v:+.2f}" for k, v in cs.iloc[::-1].head(5).items()),
        "Retained by the 0.30 screen (full data, descriptive): " + ", ".join(pre.selected_),
    ]
    (C.RESULTS_DIR / "section3_8_diagnostics.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Figures written to {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
