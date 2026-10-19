"""
Experiment 12 (optional) -- inner-CV hyper-parameter search on the TRAINING
partition only (the test partition is never touched).

  * tree baselines: randomised search, 5-fold inner CV
  * AG-StackNet: grid over gate hidden width and softmax temperature (Section 2.5)
Copy the selected values into config.py and report them in Tables 5 and 6.
If you use this random search, describe it as such in Section 2.5 (the current
text says "Bayesian optimization"; change one or the other so they agree).
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import config as C
from data_utils import load_dataset, holdout_split, Preprocessor, metrics
from alg10_ag_stacknet import AGStackNet

SPACES = {
    "decision_tree": (DecisionTreeRegressor(random_state=C.SEED),
                      {"max_depth": [3, 5, 8, 10, 15, None], "min_samples_leaf": [1, 2, 5, 10]}),
    "random_forest": (RandomForestRegressor(random_state=C.SEED, n_jobs=-1),
                      {"n_estimators": [200, 400, 600], "max_depth": [8, 12, 15, None],
                       "max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 2, 4]}),
    "gradient_boosting": (GradientBoostingRegressor(random_state=C.SEED),
                          {"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.05, 0.1],
                           "max_depth": [2, 3, 4]}),
    "xgboost": (XGBRegressor(random_state=C.SEED, tree_method="hist", n_jobs=-1),
                {"n_estimators": [200, 500, 800], "learning_rate": [0.01, 0.05, 0.1],
                 "max_depth": [3, 4, 6], "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.7, 0.8, 1.0]}),
}


def main(n_iter=30):
    df = load_dataset()
    tr, _ = holdout_split(df)
    d = df.iloc[tr]
    y = d[C.TARGET].values
    X = Preprocessor().fit_transform(d.drop(columns=[C.TARGET]), y)
    cv = KFold(C.INNER_FOLDS, shuffle=True, random_state=C.SEED)
    for name, (est, space) in SPACES.items():
        s = RandomizedSearchCV(est, space, n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
                               random_state=C.SEED, n_jobs=-1).fit(X, y)
        print(f"{name:18s} best inner-CV RMSE = {-s.best_score_:.2f}  params = {s.best_params_}")

    rows = []
    for gh, T in itertools.product([16, 32, 64], [0.5, 1.0, 2.0]):
        p = dict(C.AG, gate_hidden=gh, temperature=T)
        rmse = []
        for itr, iva in cv.split(X):
            m = AGStackNet(seed=C.SEED, params=p).fit(X.iloc[itr], y[itr])
            rmse.append(metrics(y[iva], m.predict(X.iloc[iva]))["RMSE"])
        rows.append(dict(gate_hidden=gh, temperature=T, inner_cv_rmse=np.mean(rmse)))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "ag_gate_tuning.csv", index=False)


if __name__ == "__main__":
    main()
