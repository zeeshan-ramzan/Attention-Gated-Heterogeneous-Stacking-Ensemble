"""
Algorithm 10 -- AG-StackNet (proposed): Attention-Gated Heterogeneous Stacking Ensemble.

Implements Section 2.3 and Algorithm 1 of the manuscript:

  * modality encoders  h_m = ReLU(W_m x_m + b_m)        (one dense layer per modality, dim d)
  * attention gate     s_m = w_m^T tanh(W z + b),  z = [h_rs, h_w, h_soil, h_st]
                       alpha = softmax(s / T)                                   (Eq. 1)
  * fusion             x = sum_m alpha_m h_m                                     (Eq. 2)
  * heterogeneous base learners on x:  M1 = FFNN head (trained jointly with
    encoders + gate by back-propagation), M2 = XGBoost, M3 = random forest
  * ridge meta-learner on [M1, M2, M3, alpha]  -- fitted on OUT-OF-FOLD predictions
  * residual gradient-boosting corrector on OUT-OF-FOLD meta residuals            (Eq. 3)
  * SHAP is applied post hoc in exp05_attention_shap.py (it never changes predictions)

Ablation switches (Table 13):
  use_gate=False        -> static concatenation of the modality embeddings (no attention)
  heterogeneous=False   -> three FFNN base learners instead of FFNN + XGBoost + RF
  residual=False        -> no residual corrector
The 'without spatio-temporal inputs' variant is obtained by dropping modality 'st'
in preprocessing (run_split(..., drop_modalities=('st',))).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config as C
from data_utils import main_holdout, modality_index
from torch_utils import DEVICE, FFNNRegressor, mlp, set_seed, train_module

MOD_ORDER = ["rs", "w", "soil", "st"]


# --------------------------------------------------------------------------
# Neural part: encoders + attention gate + FFNN head (M1)
# --------------------------------------------------------------------------
class AGNet(nn.Module):
    def __init__(self, mod_idx: dict, d=16, gate_hidden=32, temperature=1.0, dropout=0.1,
                 head_layers=(64, 32), use_gate=True):
        super().__init__()
        self.mods = [m for m in MOD_ORDER if m in mod_idx]
        self.idx = {m: torch.tensor(mod_idx[m], dtype=torch.long) for m in self.mods}
        self.use_gate, self.T, self.d = use_gate, temperature, d
        M = len(self.mods)
        self.enc = nn.ModuleDict({m: nn.Sequential(nn.Linear(len(mod_idx[m]), d), nn.ReLU(), nn.Dropout(dropout))
                                  for m in self.mods})
        if use_gate:
            self.gate_W = nn.Linear(M * d, gate_hidden)          # W z + b
            self.gate_w = nn.Linear(gate_hidden, M, bias=False)  # rows are w_m
            fused_dim = d
        else:
            fused_dim = M * d                                    # static concatenation
        self.head = mlp(fused_dim, head_layers)

    def embed(self, x):
        return [self.enc[m](x[:, self.idx[m].to(x.device)]) for m in self.mods]

    def forward(self, x):
        h = self.embed(x)
        if self.use_gate:
            z = torch.cat(h, dim=1)
            s = self.gate_w(torch.tanh(self.gate_W(z)))           # (n, M)
            alpha = torch.softmax(s / self.T, dim=1)              # Eq. 1
            fused = (torch.stack(h, dim=1) * alpha.unsqueeze(-1)).sum(1)   # Eq. 2
        else:
            alpha = torch.full((x.shape[0], len(self.mods)), 1.0 / len(self.mods), device=x.device)
            fused = torch.cat(h, dim=1)
        return self.head(fused), alpha, fused

    def n_gate_params(self):
        if not self.use_gate:
            return 0
        return sum(p.numel() for p in list(self.gate_W.parameters()) + list(self.gate_w.parameters()))

    def n_encoder_params(self):
        return sum(p.numel() for p in self.enc.parameters())

    def n_head_params(self):
        return sum(p.numel() for p in self.head.parameters())


# --------------------------------------------------------------------------
# Full AG-StackNet estimator
# --------------------------------------------------------------------------
class AGStackNet:
    def __init__(self, seed=C.SEED, use_gate=True, heterogeneous=True, residual=True,
                 K=C.INNER_FOLDS, params=None):
        self.seed, self.use_gate, self.heterogeneous, self.residual, self.K = seed, use_gate, heterogeneous, residual, K
        self.p = dict(C.AG if params is None else params)

    # ---------- level 0: encoders + gate + three base learners ----------
    def _fit_level0(self, X, y, seed):
        p = self.p
        set_seed(seed)
        y_mu, y_sd = y.mean(), y.std() or 1.0
        net = AGNet(self.mod_idx_, d=p["embed_dim"], gate_hidden=p["gate_hidden"], temperature=p["temperature"],
                    dropout=p["dropout"], head_layers=p["head_layers"], use_gate=self.use_gate)
        net, hist = train_module(net, X, (y - y_mu) / y_sd, lr=p["lr"], batch_size=p["batch_size"],
                                 epochs=p["epochs"], patience=p["patience"], val_fraction=p["val_fraction"],
                                 seed=seed, forward=lambda m, xb: m(xb)[0])
        st = dict(net=net, y_mu=y_mu, y_sd=y_sd, hist=hist)
        _, _, F = self._net_out(st, X)
        if self.heterogeneous:
            st["m2"] = XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist", **p["xgb"]).fit(F, y)
            st["m3"] = RandomForestRegressor(random_state=seed, n_jobs=-1, **p["rf"]).fit(F, y)
            st["m3"].set_params(n_jobs=1)   # serial prediction -> bit-for-bit reproducible sums
        else:   # homogeneous tier: two further FFNNs on the same fused representation
            kw = dict(hidden=p["head_layers"], lr=p["lr"], batch_size=p["batch_size"],
                      epochs=p["epochs"], patience=p["patience"])
            st["m2"] = FFNNRegressor(seed=seed + 1, **kw).fit(F, y)
            st["m3"] = FFNNRegressor(seed=seed + 2, **kw).fit(F, y)
        return st

    @staticmethod
    def _net_out(st, X):
        with torch.no_grad():
            out, a, f = st["net"](torch.tensor(np.asarray(X, np.float32), device=DEVICE))
        return out.cpu().numpy().ravel() * st["y_sd"] + st["y_mu"], a.cpu().numpy(), f.cpu().numpy()

    def _level0_predict(self, st, X):
        m1, A, F = self._net_out(st, X)
        B = np.column_stack([m1, st["m2"].predict(F), st["m3"].predict(F)])
        return B, A, F

    def _meta_features(self, B, A):
        return np.hstack([B, A]) if self.use_gate else B

    # ---------- training (Algorithm 1) ----------
    def fit(self, X, y):
        self.columns_ = list(X.columns)
        self.mod_idx_ = modality_index(self.columns_)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, float)
        n = len(X)
        kf = KFold(self.K, shuffle=True, random_state=self.seed)

        # (i) out-of-fold level-0 predictions, attention weights and fused representations
        B_oof = np.zeros((n, 3))
        A_oof, F_oof = None, None
        for k, (tr, va) in enumerate(kf.split(X)):
            st = self._fit_level0(X[tr], y[tr], self.seed + k)
            B, A, F = self._level0_predict(st, X[va])
            if A_oof is None:
                A_oof, F_oof = np.zeros((n, A.shape[1])), np.zeros((n, F.shape[1]))
            B_oof[va], A_oof[va], F_oof[va] = B, A, F
        self.oof_base_, self.oof_alpha_ = B_oof, A_oof

        # (ii) ridge meta-learner on out-of-fold meta-features
        MF = self._meta_features(B_oof, A_oof)
        meta = make_pipeline(StandardScaler(), Ridge(alpha=self.p["ridge_alpha"]))
        self.meta_ = clone(meta).fit(MF, y)

        # (iii)-(iv) residual corrector on out-of-fold meta residuals
        if self.residual:
            meta_oof = cross_val_predict(clone(meta), MF, y, cv=kf)
            r = y - meta_oof
            self.res_ = GradientBoostingRegressor(random_state=self.seed, **self.p["residual"]).fit(F_oof, r)

        # (v) refit level 0 on the whole training partition
        self.level0_ = self._fit_level0(X, y, self.seed)
        self.history_ = self.level0_["hist"]
        net = self.level0_["net"]
        self.n_gate_params_, self.n_encoder_params_, self.n_head_params_ = (
            net.n_gate_params(), net.n_encoder_params(), net.n_head_params())
        return self

    # ---------- inference ----------
    def predict(self, X):
        B, A, F = self._level0_predict(self.level0_, np.asarray(X, np.float32))
        pred = self.meta_.predict(self._meta_features(B, A))
        if self.residual:
            pred = pred + self.res_.predict(F)
        return pred

    def attention(self, X):
        """Per-record modality attention weights (columns follow self.modalities_)."""
        return self._net_out(self.level0_, np.asarray(X, np.float32))[1]

    @property
    def modalities_(self):
        return [m for m in MOD_ORDER if m in self.mod_idx_]

    def base_predictions(self, X):
        return self._level0_predict(self.level0_, np.asarray(X, np.float32))[0]


def build_model(seed=C.SEED, **kw):
    return AGStackNet(seed=seed, **kw)


if __name__ == "__main__":
    res = main_holdout(build_model, "AG-StackNet")
