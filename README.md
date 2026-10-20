# AG-StackNet: code and results for the revised manuscript

Place `WheatDataset.csv` (PeerJ supplementary of Ali et al., 2026) in `data/`.

| File | Purpose |
|---|---|
| `agx.py` | Leakage-safe preprocessing, all ten models, AG-StackNet (revised implementation) |
| `exp.py` | Every experiment: `python exp.py holdout`, `cv`, `val`, `abl`, `interp`, `misc` |
| `cvsum.py` | Table 8 statistics (Wilcoxon, Nadeau-Bengio corrected t, Cohen's d_z) from `results/cv_folds.csv` |
| `leak_check.py` | Reproduces the original MS-YieldStackNet protocol (meta-learner fitted on test labels) |
| `results/` | Raw outputs behind Tables 3, 7-13 and 15 and Sections 2.2, 3.4, 3.8 |
| `figure_data/` | Data for regenerating Figures 2-7 |

Revisions to AG-StackNet relative to the earlier code: tree learners receive the standardized covariates;
the residual corrector uses the standardized covariates (fold-specific embeddings do not share a latent space);
ridge meta-learner with non-negative coefficients; regularized random forest.
Repeated CV uses 5 repeats x 5 folds (seeds 0-4); the CV log in results/ also contains extra folds of repeat 5, which are excluded.
