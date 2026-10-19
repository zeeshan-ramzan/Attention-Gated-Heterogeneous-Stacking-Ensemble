"""
run_all.py -- reproduces every table and figure of the revised manuscript in order.

    python run_all.py            # everything (the repeated CV is the slow part)
    python run_all.py --quick    # 2 CV repeats instead of 10, for a first check

Outputs go to results/ (CSV + TXT) and figures/ (PNG, 300 dpi).
"""
import argparse
import time

import config as C


def main(quick=False):
    if quick:
        C.CV_REPEATS = 2
        C.CV_SEEDS[:] = list(range(2))
    import exp09_dataset_summary, exp01_holdout_benchmark, exp07_threshold_sensitivity, exp08_outliers
    import exp02_repeated_cv_significance, exp03_season_and_field_validation, exp04_ablation
    import exp05_attention_shap, exp06_base_learner_diversity, exp11_figures_and_diagnostics, exp10_compute_cost
    steps = [
        ("Dataset summary (Table 3, Section 2.1.1)", exp09_dataset_summary.main),
        ("Holdout benchmark (Table 7)", exp01_holdout_benchmark.main),
        ("Threshold sensitivity + leakage check (Section 2.2)", exp07_threshold_sensitivity.main),
        ("Outliers (Section 2.2)", exp08_outliers.main),
        ("Repeated CV + significance (Table 8)", exp02_repeated_cv_significance.main),
        ("Season / field validation (Tables 9-10)", exp03_season_and_field_validation.main),
        ("Ablation (Table 13)", exp04_ablation.main),
        ("Attention + SHAP (Table 11)", exp05_attention_shap.main),
        ("Base-learner diversity (Table 12)", exp06_base_learner_diversity.main),
        ("Figures + diagnostics (Figures 2-7, Section 3.8)", exp11_figures_and_diagnostics.main),
        ("Computational cost (Table 15)", exp10_compute_cost.main),
    ]
    for title, fn in steps:
        print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
        t0 = time.time()
        fn()
        print(f"[{title}] done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
