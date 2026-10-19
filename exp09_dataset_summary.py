"""
Experiment 9 -- Table 3 and Section 2.1.1: dataset composition, location recurrence,
train/test location overlap, and AG-StackNet parameter counts (Section 2.1.1).
"""
import numpy as np
import pandas as pd

import config as C
from data_utils import load_dataset, holdout_split, location_key, run_split
from models_registry import MODELS


def main():
    df = load_dataset()
    df["loc"] = location_key(df)
    seasons_per_loc = df.groupby("loc")["season"].nunique()
    recurring = set(seasons_per_loc[seasons_per_loc > 1].index)

    rows = []
    for s, g in df.groupby("season"):
        rows.append(dict(Season=int(s), Records=len(g), Distinct_locations=g["loc"].nunique(),
                         Recurring_locations=len(set(g["loc"]) & recurring),
                         Mean_yield=g[C.TARGET].mean()))
    tab = pd.DataFrame(rows)
    total = dict(Season="Total", Records=len(df), Distinct_locations=df["loc"].nunique(),
                 Recurring_locations=len(recurring), Mean_yield=df[C.TARGET].mean())
    tab = pd.concat([tab, pd.DataFrame([total])], ignore_index=True)
    tab.to_csv(C.RESULTS_DIR / "table3_season_composition.csv", index=False)
    print(tab.round(2).to_string(index=False))
    print("NB: the per-season 'Recurring_locations' counts location-seasons, so their sum is "
          "at least 2 x the total number of recurring locations.")

    dup_within = (df.groupby(["season", "loc"]).size() > 1).sum()
    print(f"\nDistinct locations: {df['loc'].nunique()}; recurring in >1 season: {len(recurring)} "
          f"({100 * len(recurring) / df['loc'].nunique():.1f}%)")
    print("Locations observed in k seasons:", seasons_per_loc.value_counts().sort_index().to_dict())
    print(f"Most frequent location contributes {df['loc'].value_counts().max()} records")
    print(f"Location-season pairs with more than one record (same coordinates, same season): {dup_within}")

    tr, te = holdout_split(df)
    shared = df.iloc[te]["loc"].isin(set(df.iloc[tr]["loc"])).sum()
    print(f"Holdout: {shared} of {len(te)} test records share a location with a training record")

    res = run_split(MODELS["AG-StackNet"], df.drop(columns=["loc"]), tr, te, return_model=True)
    m = res["model"]
    print(f"\nAG-StackNet parameter counts: attention gate {m.n_gate_params_}, modality encoders "
          f"{m.n_encoder_params_}, FFNN head {m.n_head_params_}")


if __name__ == "__main__":
    main()
