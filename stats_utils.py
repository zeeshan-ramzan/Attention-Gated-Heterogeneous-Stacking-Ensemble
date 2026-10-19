"""stats_utils.py -- significance tests and effect sizes used in Section 3.2."""
import numpy as np
from scipy import stats


def wilcoxon(a, b):
    """Two-sided paired Wilcoxon signed-rank test on per-fold errors."""
    res = stats.wilcoxon(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def cohens_dz(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean() / d.std(ddof=1))


def corrected_resampled_t(a, b, n_train, n_test, alpha=0.05):
    """Nadeau & Bengio (2003) corrected resampled t-test for J = len(a) paired estimates.

    Variance of the mean difference is inflated by (1/J + n_test/n_train) to allow for
    the overlap between training sets. Returns (mean_diff, ci_low, ci_high, t, p).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / J + n_test / n_train) * var)
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    h = stats.t.ppf(1 - alpha / 2, df=J - 1) * se
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h), float(t), float(p)


def mean_ci(x, alpha=0.05):
    """t-based CI for a mean of J fold estimates (descriptive; ignores fold overlap)."""
    x = np.asarray(x, float)
    h = stats.t.ppf(1 - alpha / 2, df=len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - h), float(x.mean() + h)


def paired_bootstrap_rmse_diff(y, p_a, p_b, reps=2000, seed=0, alpha=0.05):
    """CI for RMSE(b) - RMSE(a) by resampling test records with replacement."""
    rng = np.random.RandomState(seed)
    y, p_a, p_b = map(lambda v: np.asarray(v, float), (y, p_a, p_b))
    n = len(y)
    out = np.empty(reps)
    for i in range(reps):
        s = rng.randint(0, n, n)
        out[i] = np.sqrt(np.mean((y[s] - p_b[s]) ** 2)) - np.sqrt(np.mean((y[s] - p_a[s]) ** 2))
    full = np.sqrt(np.mean((y - p_b) ** 2)) - np.sqrt(np.mean((y - p_a) ** 2))
    return float(full), float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def interpret_dz(dz):
    a = abs(dz)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"
