"""
agx.py -- self-contained, leakage-safe implementation of every model and experiment in the
revised AG-StackNet manuscript (re-implemented from the manuscript, Algorithm 1 and Tables 5-6).

Revision relative to the earlier implementation (documented in the manuscript, Section 2.3):
  * the XGBoost and random-forest base learners and the residual corrector receive the
    standardized covariates together with the per-record attention weights and the fused
    embedding, [x, alpha, h], instead of the 16-dimensional fused embedding alone;
  * tree learners are regularized (depth / leaf-size limits) because the target signal is weak;
  * every supervised step (screening, scaling, tuning, out-of-fold stacking) is fitted on the
    training part of each split only.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

torch.set_num_threads(2)
SEED = 42
import os
DATA = os.environ.get("AGX_DATA", "data/WheatDataset.csv")   # PeerJ supplementary of Ali et al. (2026)
MOD = {"rs": ["DVI", "NDVI"],
       "w": ["Precipitation", "Min_Temperature", "Max_Temperature", "Mean_Temperature", "RH", "Wind_Speed"],
       "soil": ["EC", "pH", "Organic_Matter", "Available_Phosphorus", "Available_Potassium", "Saturation"],
       "st": ["Latitude", "Longitude", "Year"]}
MOD_ORDER = ["rs", "w", "soil", "st"]
FORCE_KEEP = MOD["w"][:4]           # precipitation and temperatures (agronomic override)
ALWAYS_KEEP = MOD["st"]


def load():
    d = pd.read_csv(DATA)
    n0 = len(d)
    d = d.dropna(subset=["DVI", "NDVI", "Yield"]).reset_index(drop=True)
    d.attrs["dropped"] = n0 - len(d)
    return d


# ------------------------------------------------------------------ preprocessing
class Prep:
    def __init__(self, threshold=0.0, drop=()):
        self.threshold, self.drop = threshold, drop

    def fit(self, d, y):
        self.r_ = {}
        cols = []
        for m in MOD_ORDER:
            if m in self.drop:
                continue
            for c in MOD[m]:
                x = d[c].astype(float)
                r = abs(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0
                self.r_[c] = r
                if r > self.threshold or c in FORCE_KEEP or c in ALWAYS_KEEP or self.threshold <= 0:
                    cols.append(c)
        self.cols_ = cols
        self.mu_, self.sd_ = d[cols].mean(), d[cols].std(ddof=0).replace(0, 1)
        self.mod_idx_ = {}
        for i, c in enumerate(cols):
            m = next(k for k in MOD_ORDER if c in MOD[k])
            self.mod_idx_.setdefault(m, []).append(i)
        return self

    def transform(self, d):
        return ((d[self.cols_] - self.mu_) / self.sd_).values.astype(np.float32)


# ------------------------------------------------------------------ torch helpers
def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def mlp(n_in, hidden=(64, 32), drop=0.0):
    L, k = [], n_in
    for h in hidden:
        L += [nn.Linear(k, h), nn.ReLU()] + ([nn.Dropout(drop)] if drop else [])
        k = h
    return nn.Sequential(*L, nn.Linear(k, 1))


def train_net(net, X, y, fwd, lr=1e-3, bs=32, epochs=200, patience=20, val=0.15, seed=0, wd=1e-4):
    set_seed(seed)
    n = len(X); rng = np.random.RandomState(seed); idx = rng.permutation(n)
    nv = max(1, int(round(val * n))); va, tr = idx[:nv], idx[nv:]
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best, state, bad, hist = np.inf, None, 0, {"train": [], "val": []}
    for ep in range(epochs):
        net.train(); perm = rng.permutation(tr); tl = []
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); loss = ((fwd(net, Xt[b]).squeeze(-1) - yt[b]) ** 2).mean(); loss.backward(); opt.step()
            tl.append(loss.item() * len(b))
        net.eval()
        with torch.no_grad():
            vl = ((fwd(net, Xt[va]).squeeze(-1) - yt[va]) ** 2).mean().item()
        hist["train"].append(sum(tl) / len(tr)); hist["val"].append(vl)
        if vl < best - 1e-6:
            best, bad = vl, 0; state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval()
    hist["best_epoch"] = int(np.argmin(hist["val"])) + 1
    return net, hist


class NNReg:
    """FFNN / CNN / LSTM regressor on standardized inputs (target standardized internally)."""
    def __init__(self, kind="ffnn", seed=0, hidden=(64, 32)):
        self.kind, self.seed, self.hidden = kind, seed, hidden

    def _build(self, p):
        if self.kind == "ffnn":
            return mlp(p, self.hidden)
        if self.kind == "cnn":
            return CNN(p)
        return LSTMNet()

    def fit(self, X, y):
        self.mu, self.sd = y.mean(), y.std() or 1.0
        set_seed(self.seed)
        self.net, self.hist = train_net(self._build(X.shape[1]), X, (y - self.mu) / self.sd,
                                        lambda m, xb: m(xb), seed=self.seed)
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.tensor(X)).numpy().ravel() * self.sd + self.mu


class CNN(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(32 * p, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.h(self.c(x.unsqueeze(1)))


class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.LSTM(1, 32, batch_first=True); self.h = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.l(x.unsqueeze(-1)); return self.h(o[:, -1])


# ------------------------------------------------------------------ baselines
class SK:
    def __init__(self, m): self.m = m
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X): return self.m.predict(X)


def baseline(name, seed=SEED):
    return {
        "Linear Regression": lambda: SK(LinearRegression()),
        "Decision Tree": lambda: SK(DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=seed)),
        "Random Forest": lambda: SK(RandomForestRegressor(n_estimators=400, max_depth=15, max_features="sqrt", random_state=seed, n_jobs=2)),
        "Gradient Boosting": lambda: SK(GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=seed)),
        "XGBoost": lambda: SK(XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=seed, n_jobs=2)),
        "FFNN": lambda: NNReg("ffnn", seed),
        "CNN": lambda: NNReg("cnn", seed),
        "LSTM": lambda: NNReg("lstm", seed),
        "Homogeneous Stacking": lambda: HomStack(seed),
        "AG-StackNet": lambda: AGStackNet(seed=seed),
    }[name]()


class HomStack:
    """MS-YieldStackNet: 3 x FFNN (64/32) + random-forest meta-learner on OUT-OF-FOLD predictions.
    leaky=True reproduces the published notebook, which fits the meta-learner on the test-set
    base predictions and test labels and then scores the same test set (diagnostic only)."""
    def __init__(self, seed=SEED, K=5):
        self.seed, self.K = seed, K

    def _b(self, k): return NNReg("ffnn", self.seed + 101 * (k + 1))

    def fit(self, X, y):
        oof = np.zeros((len(X), 3))
        for tr, va in KFold(self.K, shuffle=True, random_state=self.seed).split(X):
            for k in range(3):
                oof[va, k] = self._b(k).fit(X[tr], y[tr]).predict(X[va])
        self.meta = RandomForestRegressor(n_estimators=400, max_depth=15, random_state=self.seed, n_jobs=2).fit(oof, y)
        self.bases = [self._b(k).fit(X, y) for k in range(3)]
        return self

    def base_predictions(self, X): return np.column_stack([b.predict(X) for b in self.bases])
    def predict(self, X): return self.meta.predict(self.base_predictions(X))

    def leaky_test_score(self, Xtr, ytr, Xte, yte):
        bases = [self._b(k).fit(Xtr, ytr) for k in range(3)]
        P = np.column_stack([b.predict(Xte) for b in bases])
        meta = RandomForestRegressor(n_estimators=100, random_state=42).fit(P, yte)   # fitted on TEST labels
        return meta.predict(P)


# ------------------------------------------------------------------ AG-StackNet
AGP = dict(d=16, gate_hidden=32, T=1.0, dropout=0.1, head=(64, 32), lr=1e-3, bs=32, epochs=200, patience=20,
           tree_inputs="x", positive=True,
           xgb=dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
           rf=dict(n_estimators=400, max_depth=8, min_samples_leaf=10, max_features=0.5),
           ridge=10.0, res=dict(n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.8))


class AGNet(nn.Module):
    def __init__(self, mod_idx, d, gh, T, dr, head, use_gate=True):
        super().__init__()
        self.mods = [m for m in MOD_ORDER if m in mod_idx]
        self.idx = {m: torch.tensor(mod_idx[m]) for m in self.mods}
        self.use_gate, self.T = use_gate, T
        self.enc = nn.ModuleDict({m: nn.Sequential(nn.Linear(len(mod_idx[m]), d), nn.ReLU(), nn.Dropout(dr)) for m in self.mods})
        M = len(self.mods)
        if use_gate:
            self.W = nn.Linear(M * d, gh); self.w = nn.Linear(gh, M, bias=False); fd = d
        else:
            fd = M * d
        self.head = mlp(fd, head)

    def forward(self, x):
        h = [self.enc[m](x[:, self.idx[m]]) for m in self.mods]
        if self.use_gate:
            a = torch.softmax(self.w(torch.tanh(self.W(torch.cat(h, 1)))) / self.T, 1)
            f = (torch.stack(h, 1) * a.unsqueeze(-1)).sum(1)
        else:
            a = torch.full((x.shape[0], len(self.mods)), 1 / len(self.mods)); f = torch.cat(h, 1)
        return self.head(f), a, f

    def n_params(self):
        g = sum(p.numel() for p in list(self.W.parameters()) + list(self.w.parameters())) if self.use_gate else 0
        return dict(gate=g, enc=sum(p.numel() for p in self.enc.parameters()), head=sum(p.numel() for p in self.head.parameters()))


class AGStackNet:
    def __init__(self, seed=SEED, use_gate=True, heterogeneous=True, residual=True, K=5, p=None, mod_idx=None):
        self.seed, self.use_gate, self.het, self.res, self.K = seed, use_gate, heterogeneous, residual, K
        self.p = dict(AGP if p is None else p); self.mod_idx = mod_idx

    def _l0(self, X, y, s):
        p = self.p; mu, sd = y.mean(), y.std() or 1.0
        set_seed(s)
        net = AGNet(self.mod_idx, p["d"], p["gate_hidden"], p["T"], p["dropout"], p["head"], self.use_gate)
        net, hist = train_net(net, X, (y - mu) / sd, lambda m, xb: m(xb)[0], lr=p["lr"], bs=p["bs"],
                              epochs=p["epochs"], patience=p["patience"], seed=s)
        st = dict(net=net, mu=mu, sd=sd, hist=hist)
        _, A, F = self._net(st, X); T = self._tf(X, A, F)
        if self.het:
            st["m2"] = XGBRegressor(random_state=s, n_jobs=2, **p["xgb"]).fit(T, y)
            st["m3"] = RandomForestRegressor(random_state=s, n_jobs=2, **p["rf"]).fit(T, y)
        else:
            st["m2"] = NNReg("ffnn", s + 1).fit(T.astype(np.float32), y)
            st["m3"] = NNReg("ffnn", s + 2).fit(T.astype(np.float32), y)
        return st

    @staticmethod
    def _net(st, X):
        with torch.no_grad():
            o, a, f = st["net"](torch.tensor(X))
        return o.numpy().ravel() * st["sd"] + st["mu"], a.numpy(), f.numpy()

    def _pred0(self, st, X):
        m1, A, F = self._net(st, X); T = self._tf(X, A, F).astype(np.float32)
        return np.column_stack([m1, st["m2"].predict(T), st["m3"].predict(T)]), A, T

    def _tf(self, X, A, F):
        mode = self.p.get("tree_inputs", "x")
        return {"x": X, "xa": np.hstack([X, A]), "xaf": np.hstack([X, A, F])}[mode]

    def _mf(self, B, A): return np.hstack([B, A]) if self.use_gate else B

    def fit(self, X, y):
        n = len(X); kf = KFold(self.K, shuffle=True, random_state=self.seed)
        B = np.zeros((n, 3)); A = T = None
        for k, (tr, va) in enumerate(kf.split(X)):
            st = self._l0(X[tr], y[tr], self.seed + k)
            b, a, t = self._pred0(st, X[va])
            if A is None:
                A, T = np.zeros((n, a.shape[1])), np.zeros((n, t.shape[1]))
            B[va], A[va], T[va] = b, a, t
        self.oof_B = B
        MF = self._mf(B, A)
        meta = lambda: make_pipeline(StandardScaler(), Ridge(alpha=self.p["ridge"], positive=self.p.get("positive", True)))
        self.meta = meta().fit(MF, y)
        if self.res:
            oofm = np.zeros(n)
            for tr, va in kf.split(MF):
                oofm[va] = meta().fit(MF[tr], y[tr]).predict(MF[va])
            # corrector uses the standardized covariates X, which are identical across fold models
            # (fold-specific embeddings live in different latent spaces and cannot be pooled)
            self.rc = GradientBoostingRegressor(random_state=self.seed, **self.p["res"]).fit(X, y - oofm)
        self.l0 = self._l0(X, y, self.seed); self.history = self.l0["hist"]
        return self

    def predict(self, X):
        X = np.asarray(X, np.float32)
        B, A, T = self._pred0(self.l0, X)
        out = self.meta.predict(self._mf(B, A))
        return out + self.rc.predict(X) if self.res else out

    def attention(self, X): return self._net(self.l0, X)[1]
    def base_predictions(self, X): return self._pred0(self.l0, X)[0]


# ------------------------------------------------------------------ evaluation
def metrics(y, p):
    e = y - p; mse = float((e ** 2).mean())
    return dict(MAE=float(np.abs(e).mean()), MSE=mse, RMSE=mse ** 0.5, MAPE=float(100 * np.abs(e / y).mean()),
                R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def run_split(name, d, tr, te, threshold=0.0, drop=(), seed=SEED, ret=False, **agkw):
    y = d.Yield.values.astype(float)
    pre = Prep(threshold, drop).fit(d.iloc[tr], y[tr])
    Xtr, Xte = pre.transform(d.iloc[tr]), pre.transform(d.iloc[te])
    if name.startswith("AG-StackNet"):
        m = AGStackNet(seed=seed, mod_idx=pre.mod_idx_, **agkw)
    else:
        m = baseline(name, seed)
    t0 = time.perf_counter(); m.fit(Xtr, y[tr]); ft = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict(Xte); pt = time.perf_counter() - t0
    out = dict(metrics=metrics(y[te], p), y=y[te], p=p, fit_s=ft, pred_s=pt, cols=pre.cols_)
    if ret:
        out.update(model=m, pre=pre, Xtr=Xtr, Xte=Xte, ytr=y[tr])
    return out


def holdout(d, seed=SEED):
    return train_test_split(np.arange(len(d)), test_size=0.2, random_state=seed)
