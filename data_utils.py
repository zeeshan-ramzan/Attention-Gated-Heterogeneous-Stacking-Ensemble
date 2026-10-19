"""
data_utils.py -- data loading, leakage-safe preprocessing, splitting and metrics.

The preprocessing order follows Section 2.2 ("Order of operations and leakage
control") of the revised manuscript:
  1. drop records with null vegetation indices / yield  (no response information used)
  2. split into train / test (or train / validation fold)
  3. Pearson screening |r| > threshold on the TRAINING part only (+ agronomic overrides)
  4. z-score scaling fitted on the TRAINING part only
  5. model fitting on the training part
  6. a single evaluation on the held-out part
Every resampling experiment re-runs steps 3-5 inside each training split via
`run_split`, so no held-out record ever influences screening, scaling or fitting.
"""
from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config as C


# ==========================================================================
# Loading
# ==========================================================================
def _norm(s: str) -> str:
    return re.sub(r"[\s_\-\.\(\)/%]", "", str(s)).lower()


def load_dataset(path=None) -> pd.DataFrame:
    """Load the CSV/XLSX and rename columns to the canonical names of config.py."""
    path = C.DATA_PATH if path is None else path
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    lookup = {_norm(c): c for c in raw.columns}
    rename, missing = {}, []
    for canon, aliases in C.COLUMN_ALIASES.items():
        hit = next((lookup[_norm(a)] for a in aliases if _norm(a) in lookup), None)
        if hit is None:
            missing.append(canon)
        else:
            rename[hit] = canon
    df = raw.rename(columns=rename)[list(rename.values())].copy()
    if C.TARGET not in df.columns:
        raise ValueError(f"Target column not found. Add your yield header to COLUMN_ALIASES['yield']. "
                         f"Columns in file: {list(raw.columns)}")
    if missing:
        warnings.warn(f"Columns not found and therefore skipped: {missing}. "
                      f"Edit COLUMN_ALIASES in config.py if they exist under another name.")

    # season may be stored as '2018-19', 'Rabi 2018' etc. -> first 4-digit year
    if "season" in df.columns and not np.issubdtype(df["season"].dtype, np.number):
        df["season"] = df["season"].astype(str).str.extract(r"(\d{4})")[0].astype(float)

    n0 = len(df)
    must_have = [c for c in ["ndvi", "dvi", C.TARGET] if c in df.columns]
    df = df.dropna(subset=must_have).reset_index(drop=True)
    df.attrs["n_dropped_null"] = n0 - len(df)
    return df


def location_key(df: pd.DataFrame, decimals: int = 5) -> pd.Series:
    """Latitude-longitude pair used as the location identifier for grouped validation."""
    return (df["latitude"].round(decimals).astype(str) + "_" + df["longitude"].round(decimals).astype(str))


# ==========================================================================
# Preprocessing (fitted on training data only)
# ==========================================================================
@dataclass
class Preprocessor:
    threshold: float = C.CORR_THRESHOLD
    drop_modalities: tuple = ()               # e.g. ("st",) for the w/o spatio-temporal variants
    selected_: list = field(default_factory=list)
    correlations_: dict = field(default_factory=dict)
    cat_levels_: dict = field(default_factory=dict)
    mean_: pd.Series = None
    std_: pd.Series = None
    columns_: list = field(default_factory=list)

    @staticmethod
    def _modality_of(col):
        for m, cols in C.MODALITIES.items():
            if col in cols:
                return m
        return None

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        feats = [c for c in df.columns if c != C.TARGET and self._modality_of(c) not in self.drop_modalities]
        sel = []
        for c in feats:
            if c in C.CATEGORICAL:
                # categorical texture: keep (association is screened via one-hot correlation max)
                dummies = pd.get_dummies(df[c].astype(str))
                r = max((abs(np.corrcoef(dummies[k].astype(float), y)[0, 1]) for k in dummies.columns
                         if dummies[k].std() > 0), default=0.0)
            else:
                x = df[c].astype(float)
                r = abs(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0
            self.correlations_[c] = float(r)
            if r > self.threshold or c in C.FORCE_KEEP or c in C.ALWAYS_KEEP:
                sel.append(c)
        self.selected_ = sel
        X = self._encode(df, fit=True)
        self.mean_, self.std_ = X.mean(), X.std(ddof=0).replace(0, 1.0)
        self.columns_ = list(X.columns)
        return self

    def _encode(self, df, fit=False):
        parts = []
        for c in self.selected_:
            m = self._modality_of(c)
            if c in C.CATEGORICAL:
                if fit:
                    self.cat_levels_[c] = sorted(df[c].astype(str).unique())[1:]   # drop first level
                for lv in self.cat_levels_[c]:
                    parts.append(pd.Series((df[c].astype(str) == lv).astype(float).values,
                                           name=f"{m}__{c}_{lv}"))
            else:
                parts.append(pd.Series(df[c].astype(float).values, name=f"{m}__{c}"))
        return pd.concat(parts, axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._encode(df)
        X = X.reindex(columns=self.columns_, fill_value=0.0)
        return (X - self.mean_) / self.std_

    def fit_transform(self, df, y):
        return self.fit(df, y).transform(df)


def modality_index(columns) -> dict:
    """Map modality -> list of column positions, from the 'm__name' column prefixes."""
    out = {}
    for i, c in enumerate(columns):
        out.setdefault(c.split("__")[0], []).append(i)
    return out


# ==========================================================================
# Splits
# ==========================================================================
def holdout_split(df, seed=C.SEED, test_size=C.TEST_SIZE):
    idx = np.arange(len(df))
    return train_test_split(idx, test_size=test_size, random_state=seed)


def repeated_stratified_kfold(df, n_splits=C.CV_FOLDS, seeds=C.CV_SEEDS):
    """Yield (repeat, fold, train_idx, test_idx); stratified on yield quintiles."""
    strata = pd.qcut(df[C.TARGET], 5, labels=False, duplicates="drop").values
    for r, s in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(df)), strata)):
            yield r, f, tr, te


def leave_one_season_out(df):
    for s in sorted(df["season"].unique()):
        te = np.where(df["season"] == s)[0]
        tr = np.where(df["season"] != s)[0]
        yield int(s), tr, te


def chronological_split(df):
    last = df["season"].max()
    return np.where(df["season"] < last)[0], np.where(df["season"] == last)[0], int(last)


def field_grouped_kfold(df, n_splits=C.CV_FOLDS):
    groups = location_key(df).values
    for f, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(df)), groups=groups)):
        yield f, tr, te


# ==========================================================================
# Metrics
# ==========================================================================
def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = mean_squared_error(y_true, y_pred)
    return dict(
        MAE=mean_absolute_error(y_true, y_pred),
        MSE=mse,
        RMSE=float(np.sqrt(mse)),
        MAPE=float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        R2=r2_score(y_true, y_pred),
        NSE=1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2),
    )


# ==========================================================================
# One leakage-safe train/evaluate cycle
# ==========================================================================
def run_split(build_model, df, train_idx, test_idx, drop_modalities=(), return_model=False,
              threshold=C.CORR_THRESHOLD):
    """Fit preprocessing + model on train_idx only, evaluate on test_idx."""
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    y_tr, y_te = tr[C.TARGET].values.astype(float), te[C.TARGET].values.astype(float)
    pre = Preprocessor(threshold=threshold, drop_modalities=tuple(drop_modalities))
    X_tr = pre.fit_transform(tr.drop(columns=[C.TARGET]), y_tr)
    X_te = pre.transform(te.drop(columns=[C.TARGET]))
    model = build_model()
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_te)
    pred_s = time.perf_counter() - t0
    out = dict(metrics=metrics(y_te, pred), y_true=y_te, y_pred=np.asarray(pred, float),
               fit_seconds=fit_s, predict_seconds=pred_s, features=pre.columns_,
               correlations=pre.correlations_)
    if return_model:
        out.update(model=model, preprocessor=pre, X_train=X_tr, X_test=X_te, y_train=y_tr)
    return out


def evaluate_holdout(build_model, df=None, name="model", verbose=True, **kw):
    df = load_dataset() if df is None else df
    tr, te = holdout_split(df)
    res = run_split(build_model, df, tr, te, **kw)
    if verbose:
        m = res["metrics"]
        print(f"{name:28s} MAE={m['MAE']:.2f}  MSE={m['MSE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.3f}  (n_test={len(te)})")
    return res


def main_holdout(build_model, name):
    """Entry point used by every alg*.py script: holdout evaluation + saved predictions."""
    df = load_dataset()
    res = evaluate_holdout(build_model, df, name=name)
    out = C.RESULTS_DIR / f"holdout_{name}.csv"
    pd.DataFrame({"y_true": res["y_true"], "y_pred": res["y_pred"]}).to_csv(out, index=False)
    pd.DataFrame([res["metrics"]]).to_csv(C.RESULTS_DIR / f"holdout_metrics_{name}.csv", index=False)
    print(f"  retained features: {res['features']}")
    print(f"  saved {out.name}")
    return res
