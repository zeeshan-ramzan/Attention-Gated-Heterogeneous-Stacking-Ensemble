"""exp.py -- runs every experiment reported in the revised manuscript; writes results/*.json."""
import json, sys, time, pickle, tracemalloc, warnings, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, GroupKFold
from agx import *

OUT = "results"; os.makedirs(OUT, exist_ok=True)
THR = 0.30
d = load(); y = d.Yield.values.astype(float); tr, te = holdout(d)
loc = d.Latitude.round(6).astype(str) + "_" + d.Longitude.round(6).astype(str)
which = sys.argv[1] if len(sys.argv) > 1 else "all"


def save(name, obj):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)


MODELS = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost",
          "FFNN", "CNN", "LSTM", "Homogeneous Stacking", "AG-StackNet"]

# 1 -------------------------------------------------------------- holdout + leaky replication
if which in ("all", "holdout"):
    res, preds = {}, {"y": y[te].tolist()}
    for n in MODELS:
        r = run_split(n, d, tr, te, threshold=THR); res[n] = r["metrics"]; preds[n] = r["p"].tolist()
        print("holdout", n, round(r["metrics"]["RMSE"], 2), flush=True)
    pre = Prep(THR).fit(d.iloc[tr], y[tr])
    leak_p = HomStack().leaky_test_score(pre.transform(d.iloc[tr]), y[tr], pre.transform(d.iloc[te]), y[te])
    res["Homogeneous Stacking (published protocol: meta-learner fitted on test set)"] = metrics(y[te], leak_p)
    agl = run_split("AG-StackNet", d, tr, te, threshold=THR, ret=True)
    # leaky AG analogue: ridge meta + corrector refitted on test base predictions and labels
    m = agl["model"]; B, A, T = m._pred0(m.l0, agl["Xte"]); MF = m._mf(B, A)
    from sklearn.ensemble import RandomForestRegressor as RFR
    leak_ag = RFR(n_estimators=100, random_state=42).fit(MF, y[te]).predict(MF)
    res["AG-StackNet (published protocol: meta-learner fitted on test set)"] = metrics(y[te], leak_ag)
    res["_retained_features"] = pre.cols_
    save("holdout", res); save("holdout_preds", preds)

# 2 -------------------------------------------------------------- repeated CV
if which in ("all", "cv"):
    CVM = ["AG-StackNet", "Homogeneous Stacking", "LSTM", "CNN", "FFNN", "XGBoost"]
    strata = pd.qcut(y, 5, labels=False, duplicates="drop")
    rows = []
    for rep in range(10):
        for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), strata)):
            for n in CVM:
                mm = run_split(n, d, a, b, threshold=THR)["metrics"]
                rows.append(dict(model=n, rep=rep, fold=f, n_tr=len(a), n_te=len(b), **mm))
            print("cv", rep, f, flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/cv_folds.csv", index=False)

# 3 -------------------------------------------------------------- season / location validation
if which in ("all", "val"):
    out = {}
    cfgs = [("AG-StackNet", {}), ("Homogeneous Stacking", {}), ("AG-StackNet w/o ST", dict(drop=("st",))), ("XGBoost", {})]
    for name, kw in cfgs:
        mname = "AG-StackNet" if name.startswith("AG") else name
        loso = {}
        for s in sorted(d.Year.unique()):
            a, b = np.where(d.Year != s)[0], np.where(d.Year == s)[0]
            loso[int(s)] = run_split(mname, d, a, b, threshold=THR, **kw)["metrics"]
        grp = [run_split(mname, d, a, b, threshold=THR, **kw)["metrics"]
               for a, b in GroupKFold(5).split(np.zeros(len(y)), groups=loc)]
        out[name] = dict(loso=loso, grouped=grp)
        print("val", name, flush=True)
    save("validation", out)

# 4 -------------------------------------------------------------- ablation (5-fold x 2 CV)
if which in ("all", "abl"):
    cfgs = {"Full AG-StackNet": {}, "Without attention gate (static concat.)": dict(use_gate=False),
            "Without heterogeneous base learners (3xFFNN)": dict(heterogeneous=False),
            "Without residual corrector": dict(residual=False),
            "Without attention gate and heterogeneous learners": dict(use_gate=False, heterogeneous=False),
            "Without attention gate and residual corrector": dict(use_gate=False, residual=False),
            "Without heterogeneous learners and residual corrector": dict(heterogeneous=False, residual=False),
            "Without all three components": dict(use_gate=False, heterogeneous=False, residual=False)}
    strata = pd.qcut(y, 5, labels=False, duplicates="drop"); rows = []
    for rep in range(2):
        for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), strata)):
            for n, kw in cfgs.items():
                rows.append(dict(cfg=n, rep=rep, fold=f, **run_split("AG-StackNet", d, a, b, threshold=THR, **kw)["metrics"]))
            rows.append(dict(cfg="Homogeneous stacking baseline", rep=rep, fold=f,
                             **run_split("Homogeneous Stacking", d, a, b, threshold=THR)["metrics"]))
            print("abl", rep, f, flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/ablation_folds.csv", index=False)

# 5 -------------------------------------------------------------- attention, SHAP, diversity, diagnostics
if which in ("all", "interp"):
    import shap
    out = {}
    # four-modality model (no screening) for attention / SHAP
    r4 = run_split("AG-StackNet", d, tr, te, threshold=0.0, ret=True)
    m4 = r4["model"]; A = m4.attention(r4["Xte"]); mods = [k for k in MOD_ORDER if k in r4["pre"].mod_idx_]
    seas = d.Year.values[te]
    out["four_modality_holdout"] = r4["metrics"]
    out["attention"] = {k: dict(mean=A[:, j].mean(), sd=A[:, j].std(ddof=1), min=A[:, j].min(), max=A[:, j].max(),
                                seasons={int(s): A[seas == s, j].mean() for s in sorted(set(seas))})
                        for j, k in enumerate(mods)}
    bg = r4["Xtr"][np.random.RandomState(42).choice(len(r4["Xtr"]), 100, replace=False)]
    p_before = m4.predict(r4["Xte"])
    ex = shap.explainers.Permutation(m4.predict, shap.maskers.Independent(bg, max_samples=100), seed=42)
    sv = ex(r4["Xte"], max_evals=2 * r4["Xte"].shape[1] + 1 + 200).values
    out["shap_unchanged"] = bool(np.allclose(p_before, m4.predict(r4["Xte"])))
    absv = np.abs(sv); tot = absv.sum(1, keepdims=True)
    share = {k: float((absv[:, r4["pre"].mod_idx_[k]].sum(1, keepdims=True) / tot).mean()) for k in mods}
    out["shap_share"] = share
    out["shap_feature"] = dict(zip(r4["pre"].cols_, absv.mean(0).tolist()))
    # thresholded main model: diversity + residual diagnostics + convergence
    ra = run_split("AG-StackNet", d, tr, te, threshold=THR, ret=True)
    rh = run_split("Homogeneous Stacking", d, tr, te, threshold=THR, ret=True)
    out["main_attention_modalities"] = [k for k in MOD_ORDER if k in ra["pre"].mod_idx_]
    Aa = ra["model"].attention(ra["Xte"]); out["main_attention_mean"] = Aa.mean(0).tolist()
    def div(B, yt):
        E = yt[:, None] - B
        ce = np.corrcoef(E.T); cp = np.corrcoef(B.T); iu = np.triu_indices(3, 1)
        return dict(err_pairs=ce[iu].tolist(), err_mean=ce[iu].mean(), pred_mean=cp[iu].mean(),
                    ambiguity=float(((B - B.mean(1, keepdims=True)) ** 2).mean()))
    out["diversity_het"] = div(ra["model"].base_predictions(ra["Xte"]), y[te])
    out["diversity_hom"] = div(rh["model"].base_predictions(rh["Xte"]), y[te])
    from statsmodels.stats.diagnostic import het_breuschpagan
    def bp(yt, p):
        lm, lp, _, _ = het_breuschpagan(yt - p, np.column_stack([np.ones_like(p), p])); return lm, lp
    out["bp_ag"] = bp(y[te], ra["p"]); out["bp_hs"] = bp(y[te], rh["p"])
    out["resid_mean_ag"] = float((y[te] - ra["p"]).mean())
    h = ra["model"].history; sd2 = ra["model"].l0["sd"] ** 2
    out["conv"] = dict(best_epoch=h["best_epoch"], n_epochs=len(h["val"]), best_val_mse=min(h["val"]) * sd2,
                       final_train_mse=h["train"][h["best_epoch"] - 1] * sd2)
    out["gate_params"] = ra["model"].l0["net"].n_params()
    out["gate_params_4mod"] = m4.l0["net"].n_params()
    save("interp", out)
    print("interp done", flush=True)

# 6 -------------------------------------------------------------- threshold, outliers, data summary, compute
if which in ("all", "misc"):
    out = {}
    out["threshold"] = {t: run_split("AG-StackNet", d, tr, te, threshold=t)["metrics"]["RMSE"] for t in [0.0, 0.2, 0.25, 0.3, 0.35, 0.4]}
    out["threshold_xgb"] = {t: run_split("XGBoost", d, tr, te, threshold=t)["metrics"]["RMSE"] for t in [0.0, 0.3]}
    num = sum(MOD.values(), []); Z = (d[num] - d[num].mean()) / d[num].std()
    q1, q3 = Z.quantile(.25), Z.quantile(.75); iqr = q3 - q1
    F = (Z < q1 - 1.5 * iqr) | (Z > q3 + 1.5 * iqr)
    fl = F[[c for c in num if c not in MOD["st"]]]
    out["outliers"] = dict(total=int(fl.values.sum()), records=int(fl.any(axis=1).sum()), per_feature=fl.sum().sort_values(ascending=False).to_dict(),
                           max_z=float(Z[fl.columns][fl].abs().max().max()), yield_flagged=int(((y < np.quantile(y, .25) - 1.5 * (np.quantile(y, .75) - np.quantile(y, .25))) | (y > np.quantile(y, .75) + 1.5 * (np.quantile(y, .75) - np.quantile(y, .25)))).sum()))
    spl = d.groupby(loc).Year.nunique()
    out["data"] = dict(n=len(d), dropped=d.attrs["dropped"], locations=int(loc.nunique()), seasons_per_loc=spl.value_counts().to_dict(),
                       per_season={int(s): dict(n=int((d.Year == s).sum()), locs=int(loc[d.Year == s].nunique()), mean=float(y[d.Year == s].mean())) for s in sorted(d.Year.unique())},
                       test_shared=int(loc.iloc[te].isin(set(loc.iloc[tr])).sum()), ymean=float(y.mean()), ysd=float(y.std(ddof=1)),
                       ymin=float(y.min()), ymax=float(y.max()), skew=float(stats.skew(y)))
    out["corr"] = {c: float(np.corrcoef(d[c], y)[0, 1]) for c in num}
    # compute cost (CPU only)
    import psutil
    proc = psutil.Process()
    comp = {}
    for n in ["AG-StackNet", "Homogeneous Stacking"]:
        rss0 = proc.memory_info().rss
        r = run_split(n, d, tr, te, threshold=THR, ret=True)
        comp[n] = dict(fit_s=r["fit_s"], pred_s=r["pred_s"], per_rec_ms=1000 * r["pred_s"] / len(te),
                       rss_gb=proc.memory_info().rss / 1e9, size_mb=len(pickle.dumps(r["model"])) / 1e6)
    out["compute"] = comp
    save("misc", out)
    print("misc done", flush=True)
