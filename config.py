"""
config.py -- single place to adapt the code to your copy of the dataset.

Edit DATA_PATH and, if your column headers differ, the COLUMN_ALIASES below.
Every other script imports its settings from here, so a change here propagates
to all models and experiments.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "faisalabad_wheat.csv"   # <-- put the MS-YieldStackNet CSV/XLSX here
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility and protocol (must match the manuscript, Table 5)
# --------------------------------------------------------------------------
SEED = 42                     # seed of the 80/20 split used by Ali et al. (2026)
TEST_SIZE = 0.20
CORR_THRESHOLD = 0.30         # |r| screening threshold (Section 2.2)
CV_FOLDS = 5                  # outer folds of the repeated CV
CV_REPEATS = 10               # repeats of the 5-fold CV (10 x 5 = 50 fold estimates)
CV_SEEDS = list(range(CV_REPEATS))   # one seed per repeat; reported in the paper
INNER_FOLDS = 5               # K for out-of-fold stacking / residual correction
BOOTSTRAP_REPS = 2000         # paired bootstrap replicates on the holdout

# --------------------------------------------------------------------------
# Column mapping.
# Each canonical name maps to a list of accepted header spellings
# (matching is case-insensitive and ignores spaces, '_', '-', '.', '(', ')').
# --------------------------------------------------------------------------
COLUMN_ALIASES = {
    # target
    "yield":        ["yield", "yield_kg_ha", "yieldkgha", "wheat_yield", "yield(kg/ha)"],
    # weather modality
    "precipitation": ["precipitation", "precip", "rainfall", "prcp", "rain"],
    "min_temp":     ["min_temp", "mintemp", "tmin", "minimumtemperature", "mintemperature", "t2m_min"],
    "max_temp":     ["max_temp", "maxtemp", "tmax", "maximumtemperature", "maxtemperature", "t2m_max"],
    "mean_temp":    ["mean_temp", "meantemp", "tmean", "temperature", "avgtemp", "t2m", "meantemperature"],
    "rel_humidity": ["rel_humidity", "rh", "humidity", "relativehumidity", "rh2m"],
    "wind_speed":   ["wind_speed", "windspeed", "wind", "ws", "ws2m"],
    # remote-sensing modality
    "ndvi":         ["ndvi"],
    "dvi":          ["dvi"],
    # soil modality
    "ec":           ["ec", "electricalconductivity"],
    "ph":           ["ph", "soilph"],
    "organic_matter": ["organic_matter", "organicmatter", "om", "soc"],
    "phosphorus":   ["phosphorus", "availablephosphorus", "p", "availp"],
    "potassium":    ["potassium", "availablepotassium", "k", "availk"],
    "saturation":   ["saturation", "sat", "saturationpercentage"],
    "texture":      ["texture", "soiltexture", "textureclass"],
    # spatio-temporal modality
    "latitude":     ["latitude", "lat"],
    "longitude":    ["longitude", "lon", "long", "lng"],
    "season":       ["season", "year", "seasonyear", "season_year"],
}

TARGET = "yield"

MODALITIES = {
    "rs":   ["ndvi", "dvi"],
    "w":    ["precipitation", "min_temp", "max_temp", "mean_temp", "rel_humidity", "wind_speed"],
    "soil": ["ec", "ph", "organic_matter", "phosphorus", "potassium", "saturation", "texture"],
    "st":   ["latitude", "longitude", "season"],
}
MODALITY_NAMES = {"rs": "Remote sensing", "w": "Weather", "soil": "Soil", "st": "Spatio-temporal"}

# Variables retained regardless of their correlation (agronomic override, Section 2.2)
FORCE_KEEP = ["precipitation", "min_temp", "max_temp", "mean_temp"]
# The spatio-temporal identifiers are always passed to the models (they are inputs in the paper)
ALWAYS_KEEP = ["latitude", "longitude", "season"]
CATEGORICAL = ["texture"]     # one-hot encoded inside the preprocessing pipeline

# --------------------------------------------------------------------------
# Model hyper-parameters (defaults = values to be reported in Tables 5 and 6;
# change them here if your tuned values differ, then report what you used)
# --------------------------------------------------------------------------
AG = dict(
    embed_dim=16, gate_hidden=32, temperature=1.0, dropout=0.10,
    head_layers=(64, 32), lr=1e-3, batch_size=32, epochs=200, patience=20,
    val_fraction=0.15,
    xgb=dict(n_estimators=500, max_depth=4, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8),
    rf=dict(n_estimators=400, max_depth=15, min_samples_leaf=2),
    ridge_alpha=1.0,
    residual=dict(n_estimators=200, max_depth=3, learning_rate=0.05),
)

BASELINES = dict(
    decision_tree=dict(max_depth=10, min_samples_leaf=5),
    random_forest=dict(n_estimators=400, max_depth=15, max_features="sqrt"),
    gradient_boosting=dict(n_estimators=200, learning_rate=0.10, max_depth=3),
    xgboost=dict(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8),
    ffnn=dict(hidden=(64, 32), lr=1e-3, batch_size=32, epochs=200, patience=20),
    cnn=dict(filters=32, kernel=3, conv_layers=2, dense=32, lr=1e-3,
             batch_size=32, epochs=200, patience=20),
    lstm=dict(units=32, dense=32, lr=1e-3, batch_size=32, epochs=200, patience=20),
    homogeneous_stacking=dict(n_base=3, hidden=(64, 32),
                              meta_rf=dict(n_estimators=400, max_depth=15),
                              folds=5),
)
