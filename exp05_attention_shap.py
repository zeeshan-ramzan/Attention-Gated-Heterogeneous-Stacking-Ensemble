"""
Experiment 5 -- Table 11 and Section 3.4: learned attention weights and SHAP
modality attribution on the 161 held-out records.

  * attention weights: mean, SD, min, max per modality, and seasonal means
  * SHAP: permutation explainer with an interventional (independent) masker
    -- background = BACKGROUND_N records sampled from the TRAINING partition --
    applied to the complete pipeline (standardised inputs -> final prediction);
    modality share = sum |SHAP| of the modality's columns / total |SHAP| per record
  * Spearman correlation between per-record attention weight and SHAP share
  * check that SHAP leaves predictions unchanged (answers the old SHAP-ablation issue)
"""
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

import config as C
from data_utils import load_dataset, holdout_split, run_split, modality_index
from models_registry import MODELS

BACKGROUND_N = 100


def main():
    df = load_dataset()
    tr, te = holdout_split(df)
    res = run_split(MODELS["AG-StackNet"], df, tr, te, return_model=True)
    model, X_tr, X_te = res["model"], res["X_train"], res["X_test"]
    mods = model.modalities_
    names = [C.MODALITY_NAMES[m] for m in mods]

    # ---- attention weights ----
    A = model.attention(X_te)
    seasons = df.iloc[te]["season"].values
    rows = []
    for j, m in enumerate(mods):
        r = dict(Modality=C.MODALITY_NAMES[m], Mean=A[:, j].mean(), SD=A[:, j].std(ddof=1),
                 Min=A[:, j].min(), Max=A[:, j].max())
        for s in sorted(np.unique(seasons)):
            r[str(int(s))] = A[seasons == s, j].mean()
        rows.append(r)
    att = pd.DataFrame(rows)
    att.to_csv(C.RESULTS_DIR / "table11_attention.csv", index=False)
    pd.DataFrame(A, columns=names).assign(season=seasons).to_csv(C.RESULTS_DIR / "attention_per_record.csv", index=False)

    # ---- SHAP on the whole pipeline ----
    pred_before = model.predict(X_te)
    bg = X_tr.sample(min(BACKGROUND_N, len(X_tr)), random_state=C.SEED)
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    explainer = shap.explainers.Permutation(model.predict, masker, seed=C.SEED)
    sv = explainer(X_te, max_evals=2 * X_te.shape[1] + 1 + 200).values      # (n_test, n_features)
    pred_after = model.predict(X_te)
    unchanged = np.allclose(pred_before, pred_after)

    idx = modality_index(list(X_te.columns))
    absv = np.abs(sv)
    share = np.column_stack([absv[:, idx[m]].sum(1) for m in mods]) / absv.sum(1, keepdims=True)
    feat_imp = pd.Series(absv.mean(0), index=X_te.columns).sort_values(ascending=False)
    feat_imp.to_csv(C.RESULTS_DIR / "shap_feature_importance.csv", header=["mean_abs_shap"])
    pd.DataFrame(sv, columns=X_te.columns).to_csv(C.RESULTS_DIR / "shap_values_test.csv", index=False)

    lines = ["Attention weights (held-out records):", att.round(3).to_string(index=False), "",
             "Ordering by mean attention: " + " > ".join(att.sort_values("Mean", ascending=False).Modality), "",
             "SHAP modality share of total |SHAP| (mean over records):"]
    for j, n in enumerate(names):
        rho, p = spearmanr(A[:, j], share[:, j])
        lines.append(f"  {n:16s} SHAP share = {100 * share[:, j].mean():5.1f}%   "
                     f"Spearman(attention, SHAP share) rho = {rho:.2f} (p = {p:.3g})")
    lines.append("Ordering by SHAP share: " + " > ".join(np.array(names)[np.argsort(-share.mean(0))]))
    lines.append(f"Predictions identical before/after SHAP computation: {unchanged}")
    lines.append(f"Explainer: permutation, interventional masker, background {len(bg)} training records")
    (C.RESULTS_DIR / "section3_4_attention_shap.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(sv, X_te, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(C.FIGURES_DIR / "shap_summary.png", dpi=300)
        plt.close()
    except Exception as e:      # plotting is optional
        print("SHAP plot skipped:", e)


if __name__ == "__main__":
    main()
