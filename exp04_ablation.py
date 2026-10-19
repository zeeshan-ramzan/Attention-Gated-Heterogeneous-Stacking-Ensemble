"""
Experiment 4 -- Table 13 (Section 3.6): single, pairwise and triple component
removals on the identical holdout, plus interaction terms.

Interaction(A,B) = [RMSE(-A,-B) - RMSE(full)] - {[RMSE(-A) - RMSE(full)] + [RMSE(-B) - RMSE(full)]}
Negative values mean the components overlap (sub-additive).
Note: there is deliberately NO 'without SHAP' row -- SHAP is post hoc and cannot
change predictions (see exp05, which verifies this).
Use --cv to repeat the ablation over the repeated 5-fold protocol as well.
"""
import argparse
import itertools

import numpy as np
import pandas as pd

import config as C
from alg10_ag_stacknet import AGStackNet
from data_utils import load_dataset, holdout_split, run_split, repeated_stratified_kfold
from models_registry import MODELS

COMPONENTS = {"gate": "use_gate", "learners": "heterogeneous", "residual": "residual"}
LABEL = {"gate": "attention gate", "learners": "heterogeneous learners", "residual": "residual corrector"}


def configs():
    yield "Full AG-StackNet", ()
    for r in (1, 2, 3):
        for combo in itertools.combinations(COMPONENTS, r):
            name = "Without " + " and ".join(LABEL[c] for c in combo) if r < 3 else "Without all three components"
            yield name, combo


def builder(removed):
    kw = {COMPONENTS[c]: False for c in removed}
    return lambda: AGStackNet(seed=C.SEED, **kw)


def main(cv=False):
    df = load_dataset()
    splits = [(0, 0, *holdout_split(df))] if not cv else list(repeated_stratified_kfold(df))
    rows = []
    for name, removed in configs():
        ms = [run_split(builder(removed), df, tr, te)["metrics"] for _, _, tr, te in splits]
        rows.append(dict(Configuration=name, removed="+".join(removed),
                         RMSE=np.mean([m["RMSE"] for m in ms]), MAE=np.mean([m["MAE"] for m in ms]),
                         R2=np.mean([m["R2"] for m in ms])))
        print(f"{name:60s} RMSE={rows[-1]['RMSE']:.2f}")
    ms = [run_split(MODELS["Homogeneous Stacking"], df, tr, te)["metrics"] for _, _, tr, te in splits]
    rows.append(dict(Configuration="Homogeneous stacking baseline", removed="baseline",
                     RMSE=np.mean([m["RMSE"] for m in ms]), MAE=np.mean([m["MAE"] for m in ms]),
                     R2=np.mean([m["R2"] for m in ms])))
    tab = pd.DataFrame(rows)
    full = tab.RMSE.iloc[0]
    tab["penalty_vs_full"] = tab.RMSE - full
    pen = dict(zip(tab.removed, tab.penalty_vs_full))
    lines = []
    for a, b in itertools.combinations(COMPONENTS, 2):
        inter = pen[f"{a}+{b}"] - (pen[a] + pen[b])
        lines.append(f"Interaction {a} x {b}: {inter:+.2f} kg/ha "
                     f"(joint {pen[f'{a}+{b}']:.2f} vs sum of singles {pen[a] + pen[b]:.2f})")
    lines.append(f"Sum of single-removal penalties = {pen['gate'] + pen['learners'] + pen['residual']:.2f} kg/ha; "
                 f"full-vs-baseline gap = {tab.RMSE.iloc[-1] - full:.2f} kg/ha")
    tag = "cv" if cv else "holdout"
    tab.to_csv(C.RESULTS_DIR / f"table13_ablation_{tag}.csv", index=False)
    (C.RESULTS_DIR / f"ablation_interactions_{tag}.txt").write_text("\n".join(lines))
    print(tab.round(2).to_string(index=False))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="average over the repeated 5-fold protocol")
    main(ap.parse_args().cv)
