"""
make_synthetic_data.py -- writes a SYNTHETIC dataset with the same columns and
ranges as Table 4, ONLY to check that the code runs end to end.

!! Never report results obtained on this file. Use the real MS-YieldStackNet data. !!
"""
import numpy as np
import pandas as pd

import config as C


def main(n_loc=300, seed=0):
    rng = np.random.RandomState(seed)
    lat = rng.uniform(31.2, 31.7, n_loc); lon = rng.uniform(72.8, 73.4, n_loc)
    soil = dict(ec=rng.uniform(2, 4, n_loc), ph=rng.uniform(7, 8, n_loc), organic_matter=rng.uniform(.4, 2, n_loc),
                phosphorus=rng.uniform(8, 20, n_loc), potassium=rng.uniform(80, 120, n_loc),
                saturation=rng.uniform(35, 45, n_loc), texture=rng.choice(["Loam", "Clay loam", "Sandy loam"], n_loc))
    rows = []
    for s in range(2018, 2023):
        w = dict(precipitation=rng.uniform(.03, .69), min_temp=rng.uniform(7.6, 13), max_temp=rng.uniform(21, 27.8),
                 rel_humidity=rng.uniform(53, 61.5), wind_speed=rng.uniform(1.2, 2.2))
        w["mean_temp"] = (w["min_temp"] + w["max_temp"]) / 2
        for i in rng.choice(n_loc, 161, replace=False):
            ndvi = np.clip(rng.normal(.41, .14), .13, .72)
            dvi = np.clip(.35 * ndvi + rng.normal(0, .03), .05, .34)
            y = (1300 + 600 * ndvi + 80 * soil["organic_matter"][i] + 8 * (soil["phosphorus"][i] - 14)
                 + 150 * w["precipitation"] - 15 * (w["max_temp"] - 25) + 12 * (s - 2018) + rng.normal(0, 60))
            rows.append(dict(Latitude=lat[i], Longitude=lon[i], Season=s, **{k: v for k, v in w.items()},
                             NDVI=ndvi, DVI=dvi, **{k: v[i] for k, v in soil.items()}, Yield=np.clip(y, 1320, 2200)))
    df = pd.DataFrame(rows)
    C.DATA_PATH.parent.mkdir(exist_ok=True)
    out = C.DATA_PATH.parent / "synthetic_test_data.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows) -- for code testing only")


if __name__ == "__main__":
    main()
