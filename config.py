"""
Central configuration for the Urban Energy Consumption Forecasting pipeline.
All hyperparameters, paths, and feature settings are defined here so that
train.py, serve.py, and the notebooks share consistent defaults.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (resolved relative to this file's location)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Directory layout (mirrors the structure in README)
# ---------------------------------------------------------------------------
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR         = ROOT_DIR / "models"
NOTEBOOKS_DIR      = ROOT_DIR / "notebooks"

# Ensure directories exist at import time
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data / feature settings
# ---------------------------------------------------------------------------

# UCI dataset filename (place the downloaded CSV here)
RAW_DATA_FILE = DATA_RAW_DIR / "household_power_consumption.csv"

# Processed artefacts
PROCESSED_DATA_FILE  = DATA_PROCESSED_DIR / "energy_processed.csv"
SCALER_X_FILE        = DATA_PROCESSED_DIR / "scaler_X.pkl"
SCALER_Y_FILE        = DATA_PROCESSED_DIR / "scaler_y.pkl"

# Input features fed into the LSTM
FEATURE_COLUMNS = [
    "Global_active_power",  # target is also used as a lagged feature
    "temperature",
    "humidity",
    "wind_speed",
    "solar_irradiance",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
]

TARGET_COLUMN = "Global_active_power"   # kWh per interval

# ---------------------------------------------------------------------------
# Sequence / windowing
# ---------------------------------------------------------------------------
LOOKBACK  = 48   # hours of history fed to the LSTM  (--lookback CLI arg)
HORIZON   = 24   # hours ahead to predict            (multi-step output)

# ---------------------------------------------------------------------------
# Model hyperparameters (match the README architecture diagram)
# ---------------------------------------------------------------------------
LSTM_UNITS_1  = 128
LSTM_UNITS_2  = 64
DENSE_UNITS   = 32
DROPOUT_RATE  = 0.2

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
EPOCHS        = 50        # overridden by --epochs CLI arg
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.10
TEST_SPLIT       = 0.10   # final hold-out fraction

# Model checkpoint / save path
MODEL_SAVE_PATH = MODELS_DIR / "lstm_v2.h5"

# ---------------------------------------------------------------------------
# Serving / API settings
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# ---------------------------------------------------------------------------
# Weather API (OpenWeatherMap — optional, needed to fetch live features)
# ---------------------------------------------------------------------------
OWM_API_KEY  = os.getenv("OWM_API_KEY", "")
OWM_CITY_IDS = os.getenv("OWM_CITY_IDS", "").split(",")  # comma-separated city IDs

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
