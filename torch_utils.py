"""
torch_utils.py -- shared PyTorch training loop (Adam, MSE, mini-batches, early
stopping on an inner validation split, standardised target) and the three
neural architectures used as baselines: FFNN, 1-D CNN and LSTM.
"""
from __future__ import annotations

import copy
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:     # pragma: no cover
    raise ImportError("PyTorch is required: pip install torch") from e


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_module(module, X, y, lr=1e-3, batch_size=32, epochs=200, patience=20,
                 val_fraction=0.15, seed=42, forward=None):
    """Train `module` to regress standardised y on X; returns (module, history).

    `forward(module, xb)` lets callers customise the forward pass (used by
    AG-StackNet, whose module returns several tensors).
    """
    set_seed(seed)
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    va, tr = idx[:n_val], idx[n_val:]
    Xt = torch.tensor(np.asarray(X, np.float32), device=DEVICE)
    yt = torch.tensor(np.asarray(y, np.float32).reshape(-1, 1), device=DEVICE)
    fwd = forward or (lambda m, xb: m(xb))
    module.to(DEVICE)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best, best_state, wait = np.inf, None, 0
    hist = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        module.train()
        perm = rng.permutation(tr)
        tl = 0.0
        for b in range(0, len(perm), batch_size):
            bi = perm[b:b + batch_size]
            opt.zero_grad()
            loss = lossf(fwd(module, Xt[bi]), yt[bi])
            loss.backward()
            opt.step()
            tl += loss.item() * len(bi)
        module.eval()
        with torch.no_grad():
            vl = lossf(fwd(module, Xt[va]), yt[va]).item()
        hist["train_loss"].append(tl / len(tr))
        hist["val_loss"].append(vl)
        if vl < best - 1e-6:
            best, best_state, wait = vl, copy.deepcopy(module.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    return module, hist


class TorchRegressor:
    """sklearn-style wrapper: standardises y, trains, inverse-transforms predictions."""

    def __init__(self, lr=1e-3, batch_size=32, epochs=200, patience=20, seed=42, **kw):
        self.lr, self.batch_size, self.epochs, self.patience, self.seed = lr, batch_size, epochs, patience, seed
        self.kw = kw

    def build(self, n_features):            # overridden
        raise NotImplementedError

    def _prep(self, X):
        return np.asarray(X, np.float32)

    def fit(self, X, y):
        X = self._prep(X)
        y = np.asarray(y, float)
        self.y_mu_, self.y_sd_ = y.mean(), y.std() or 1.0
        set_seed(self.seed)
        self.module_ = self.build(X.shape[-1] if X.ndim == 2 else X.shape[1])
        self.module_, self.history_ = train_module(
            self.module_, X, (y - self.y_mu_) / self.y_sd_, lr=self.lr,
            batch_size=self.batch_size, epochs=self.epochs, patience=self.patience, seed=self.seed)
        return self

    def predict(self, X):
        X = torch.tensor(self._prep(X), device=DEVICE)
        with torch.no_grad():
            out = self.module_(X).cpu().numpy().ravel()
        return out * self.y_sd_ + self.y_mu_


def mlp(n_in, hidden=(64, 32), dropout=0.0):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class FFNNRegressor(TorchRegressor):
    def build(self, n):
        return mlp(n, self.kw.get("hidden", (64, 32)))


class _CNN(nn.Module):
    def __init__(self, p, filters=32, kernel=3, conv_layers=2, dense=32):
        super().__init__()
        layers, c = [], 1
        for _ in range(conv_layers):
            layers += [nn.Conv1d(c, filters, kernel, padding=kernel // 2), nn.ReLU()]
            c = filters
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(filters * p, dense), nn.ReLU(), nn.Linear(dense, 1))

    def forward(self, x):                   # x: (batch, p) -> (batch, 1, p)
        return self.head(self.conv(x.unsqueeze(1)))


class CNNRegressor(TorchRegressor):
    """1-D CNN over the p x 1 pseudo-sequence of standardised features (Section 2.4)."""
    def build(self, p):
        return _CNN(p, **{k: self.kw[k] for k in ("filters", "kernel", "conv_layers", "dense") if k in self.kw})


class _LSTM(nn.Module):
    def __init__(self, units=32, dense=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=units, batch_first=True)
        self.head = nn.Sequential(nn.Linear(units, dense), nn.ReLU(), nn.Linear(dense, 1))

    def forward(self, x):                   # x: (batch, p) -> (batch, p, 1); recurrence over feature positions
        _, (h, _) = self.lstm(x.unsqueeze(-1))
        return self.head(h[-1])


class LSTMRegressor(TorchRegressor):
    """LSTM over the p x 1 pseudo-sequence (feature order carries no temporal meaning)."""
    def build(self, p):
        return _LSTM(self.kw.get("units", 32), self.kw.get("dense", 32))
