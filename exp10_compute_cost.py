"""
Experiment 10 -- Table 15 (Section 4.3): measured computational cost.

Each measurement runs in a fresh subprocess so peak memory is not contaminated
by earlier runs. 'cpu' forces CPU-only execution (CUDA hidden); 'auto' uses the
GPU if one is available. GPU utilisation is best read with `nvidia-smi dmon`
running in parallel; peak GPU memory is reported from torch.
"""
import json
import os
import pickle
import resource
import subprocess
import sys
import time

import config as C

MODELS_TO_TIME = ["AG-StackNet", "Homogeneous Stacking"]


def worker(name, device):
    import numpy as np
    from data_utils import load_dataset, holdout_split, Preprocessor
    from models_registry import MODELS
    import torch
    df = load_dataset()
    tr, te = holdout_split(df)
    y_tr = df.iloc[tr][C.TARGET].values
    pre = Preprocessor()
    X_tr = pre.fit_transform(df.iloc[tr].drop(columns=[C.TARGET]), y_tr)
    X_te = pre.transform(df.iloc[te].drop(columns=[C.TARGET]))
    import psutil
    rss_before = psutil.Process().memory_info().rss / 2**20
    model = MODELS[name]()
    t0 = time.perf_counter(); model.fit(X_tr, y_tr); fit_s = time.perf_counter() - t0
    peak_fit_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # Linux: KiB -> MiB
    t0 = time.perf_counter()
    for _ in range(10):
        model.predict(X_te)
    pred_s = (time.perf_counter() - t0) / 10
    size_mb = len(pickle.dumps(model)) / 1e6
    gpu_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    print(json.dumps(dict(model=name, device=device, fit_seconds=fit_s, peak_rss_mb=peak_fit_mb, rss_before_fit_mb=rss_before,
                          fit_memory_increase_mb=peak_fit_mb - rss_before,
                          predict_seconds_161=pred_s, latency_ms_per_record=1000 * pred_s / len(te),
                          model_size_mb=size_mb, peak_gpu_mem_mb=gpu_mb, n_test=len(te))))


def main():
    rows = []
    for name in MODELS_TO_TIME:
        for device in ("auto", "cpu"):
            env = dict(os.environ)
            if device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            out = subprocess.run([sys.executable, __file__, "--worker", name, device], env=env,
                                 capture_output=True, text=True, cwd=str(C.ROOT))
            line = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(out.stderr[-2000:]); continue
            rows.append(json.loads(line[-1]))
            print(rows[-1])
    import pandas as pd
    tab = pd.DataFrame(rows)
    tab.to_csv(C.RESULTS_DIR / "table15_compute_cost.csv", index=False)
    print(tab.round(3).to_string(index=False))
    print("\nTotal time of the repeated CV protocol: read the timestamps of results/cv_fold_results.csv "
          "or time `python exp02_repeated_cv_significance.py`.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main()
