# AG-StackNet – code for every algorithm and experiment in the manuscript

Ready-to-run Python implementation of all models in
*"AG-StackNet: An Attention-Gated Heterogeneous Stacking Ensemble for Explainable Wheat
Yield Prediction from Multi-Source Data"*, plus the scripts that produce every table and
figure requested in the revision.

## 1. Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.9+ is required. A GPU is optional; everything runs on CPU.

## 2. Data
Put the MS-YieldStackNet Faisalabad dataset (Ali et al., 2026, PeerJ CS supplementary file)
at `data/faisalabad_wheat.csv` (or `.xlsx`, then change `DATA_PATH` in `config.py`).
Column headers are matched automatically (case and punctuation are ignored). If a column is
reported as "not found", add your header spelling to `COLUMN_ALIASES` in `config.py`.

Smoke test without the real data (results are meaningless, never report them):
```bash
python make_synthetic_data.py
# then set DATA_PATH in config.py to data/synthetic_test_data.csv
python run_all.py --quick
```

## 3. Algorithms (one file each)
| File | Algorithm |
|---|---|
| `alg01_linear_regression.py` | Linear regression |
| `alg02_decision_tree.py` | Decision tree |
| `alg03_random_forest.py` | Random forest |
| `alg04_gradient_boosting.py` | Gradient boosting |
| `alg05_xgboost.py` | XGBoost |
| `alg06_ffnn.py` | Feed-forward neural network (64/32) |
| `alg07_cnn.py` | 1-D CNN on the p×1 pseudo-sequence |
| `alg08_lstm.py` | LSTM on the p×1 pseudo-sequence |
| `alg09_homogeneous_stacking.py` | Homogeneous stacking baseline (3×FFNN + RF meta-learner, MS-YieldStackNet) |
| `alg10_ag_stacknet.py` | **AG-StackNet (proposed)**: modality encoders, attention gate, FFNN + XGBoost + RF, ridge meta-learner, residual GBM corrector, out-of-fold stacking |

Each file runs on its own and evaluates the model on the 80/20 holdout (seed 42):
```bash
python alg10_ag_stacknet.py
```

## 4. Experiments → manuscript tables and figures
| Script | Produces |
|---|---|
| `exp09_dataset_summary.py` | Table 3, location recurrence and train/test overlap (Sec. 2.1.1), gate parameter counts |
| `exp01_holdout_benchmark.py` | Table 7 (all models, holdout) and relative reductions |
| `exp07_threshold_sensitivity.py` | |r| threshold sensitivity, training-only vs full-data screen (Sec. 2.2) |
| `exp08_outliers.py` | IQR outlier counts and RMSE without flagged records (Sec. 2.2) |
| `exp02_repeated_cv_significance.py` | Table 8: 10×5 repeated CV, Wilcoxon, Nadeau–Bengio corrected t, Cohen's d_z, bootstrap CI (Sec. 3.2) |
| `exp03_season_and_field_validation.py` | Tables 9–10: leave-one-season-out, chronological, field-grouped CV, w/o spatio-temporal inputs |
| `exp04_ablation.py` | Table 13: single, pairwise and triple removals, interaction terms (`--cv` for CV version) |
| `exp05_attention_shap.py` | Table 11: attention statistics by season, SHAP modality shares, Spearman agreement |
| `exp06_base_learner_diversity.py` | Table 12: error/prediction correlations, ensemble ambiguity |
| `exp11_figures_and_diagnostics.py` | Figures 2–7 from real predictions, Breusch–Pagan, residual diagnostics |
| `exp10_compute_cost.py` | Table 15: training/inference time, memory, model size |
| `exp12_hyperparameter_tuning.py` | Optional inner-CV tuning on the training partition only |
| `run_all.py` | Runs everything in order (`--quick` = 2 CV repeats) |

All outputs are written to `results/` (CSV/TXT) and `figures/` (PNG, 300 dpi).
`exp02` writes fold results incrementally and resumes if interrupted.

## 5. Leakage control (as described in the revised Section 2.2)
`data_utils.run_split` is the only route from data to a score. For every split it
(1) fits the |r| > 0.30 screen and (2) the z-score scaler on the training part only,
(3) fits the model on the training part, and (4) scores the held-out part once.
Inside AG-StackNet, the meta-learner and the residual corrector are fitted on
out-of-fold predictions (K = 5) before the level-0 models are refitted on the full
training partition, exactly as in Algorithm 1.

## 6. Points to check against the manuscript after running
* `exp03`: the chronological result must equal the last-season LOSO fold (same split).
* `exp09`: report the printed parameter counts (the gate with d = 16 and hidden 32 has 2,208 parameters).
* `exp05`: confirms predictions are identical with and without SHAP.
* If your original MS-YieldStackNet implementation differs from `alg09`, use the original
  and state so in Section 2.4.
* Report the hyper-parameters actually used (config.py) in Tables 5 and 6.
