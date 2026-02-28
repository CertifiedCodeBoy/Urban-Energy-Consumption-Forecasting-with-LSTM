"""
src/preprocess.py
-----------------
End-to-end data preparation pipeline for the Urban Energy Forecasting project.

Key responsibilities
--------------------
1. Load the raw UCI household power-consumption CSV.
2. Resample to hourly averages.
3. Merge any available weather features (temperature, humidity, wind speed,
   solar irradiance).  If a weather CSV is present in data/raw/ those columns
   are joined; otherwise synthetic weather is generated so the pipeline still
   runs in demo / offline mode.
4. Engineer cyclical time features (hour, day-of-week) and a public-holiday flag.
5. Min–Max scale features and target independently.
6. Build sliding-window sequences (X, y) ready for the LSTM.
7. Split into train / validation / test sets (chronologically).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    SCALER_X_FILE,
    SCALER_Y_FILE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LOOKBACK,
    HORIZON,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _cyclic_encode(series: pd.Series, period: float) -> Tuple[pd.Series, pd.Series]:
    """Return sin and cos encodings of a periodic numeric series."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def _add_synthetic_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate plausible weather-like signals when no real weather data is
    available.  Not meant to be realistic — purely for demonstration /
    offline development so models can be trained without an API key.
    """
    np.random.seed(RANDOM_SEED)
    n = len(df)
    hours = df.index.hour.values.astype(float)

    # Temperature: ~15°C mean with diurnal cycle + noise
    df["temperature"] = (
        15.0
        + 8.0 * np.sin(2 * np.pi * (hours - 6) / 24)
        + np.random.normal(0, 1.5, n)
    )
    # Humidity: inversely correlated with temperature, roughy
    df["humidity"] = (
        60.0
        - 20.0 * np.sin(2 * np.pi * (hours - 6) / 24)
        + np.random.normal(0, 5.0, n)
    ).clip(20, 100)
    # Wind speed: random-ish, log-normal
    df["wind_speed"] = np.abs(np.random.randn(n) * 3 + 4)
    # Solar irradiance: zero at night, bell-shaped during the day
    solar = np.zeros(n)
    daytime = (hours >= 6) & (hours <= 20)
    solar[daytime] = 800 * np.sin(np.pi * (hours[daytime] - 6) / 14) ** 2
    solar += np.random.normal(0, 30, n)
    df["solar_irradiance"] = solar.clip(0)

    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical hour / day-of-week encodings and an is_holiday flag."""
    try:
        import holidays as hols
        country_holidays = hols.country_holidays("US")
        df["is_holiday"] = df.index.normalize().isin(country_holidays).astype(float)
    except Exception:
        df["is_holiday"] = 0.0

    sin_h, cos_h = _cyclic_encode(pd.Series(df.index.hour, index=df.index), 24)
    sin_d, cos_d = _cyclic_encode(pd.Series(df.index.dayofweek, index=df.index), 7)

    df["hour_sin"] = sin_h.values
    df["hour_cos"] = cos_h.values
    df["dow_sin"]  = sin_d.values
    df["dow_cos"]  = cos_d.values
    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """Full preprocessing pipeline."""

    def __init__(
        self,
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
        val_split: float = VALIDATION_SPLIT,
        test_split: float = TEST_SPLIT,
    ) -> None:
        self.lookback   = lookback
        self.horizon    = horizon
        self.val_split  = val_split
        self.test_split = test_split

        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_raw(self, filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
        """
        Load the UCI household power consumption dataset.
        Returns a DataFrame indexed by a datetime column, resampled to 1-hour means.
        """
        logger.info("Loading raw data from %s ...", filepath)

        if not Path(filepath).exists():
            logger.warning(
                "Raw data file not found at %s — generating synthetic dataset for demonstration.",
                filepath,
            )
            return self._generate_synthetic_dataset()

        df = pd.read_csv(
            filepath,
            sep=";",
            na_values=["?"],
            parse_dates={"datetime": ["Date", "Time"]},
            dayfirst=True,
            low_memory=False,
        )
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        # Keep only the power column; convert to numeric
        df = df[["Global_active_power"]].copy()
        df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
        df.dropna(inplace=True)

        # Resample to hourly sums (kWh ≈ kW × 1h); divide by 60 to convert from kW·min
        df = df.resample("h").mean()
        df.dropna(inplace=True)

        logger.info("Raw data loaded: %d hourly records (%s → %s)",
                    len(df), df.index[0], df.index[-1])
        return df

    def add_features(self, df: pd.DataFrame, weather_file: Path | None = None) -> pd.DataFrame:
        """Merge weather data (if available) and build all engineered features."""
        if weather_file and Path(weather_file).exists():
            logger.info("Merging weather data from %s ...", weather_file)
            weather = pd.read_csv(weather_file, index_col=0, parse_dates=True)
            weather = weather.resample("h").mean().interpolate()
            df = df.join(weather[["temperature", "humidity", "wind_speed", "solar_irradiance"]],
                         how="left")
            df[["temperature", "humidity", "wind_speed", "solar_irradiance"]] = \
                df[["temperature", "humidity", "wind_speed", "solar_irradiance"]].interpolate()
        else:
            logger.info("No weather file provided — generating synthetic weather signals.")
            df = _add_synthetic_weather(df)

        df = _add_time_features(df)
        df.dropna(inplace=True)
        return df

    def scale(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale features with MinMaxScaler.

        Parameters
        ----------
        df  : DataFrame with FEATURE_COLUMNS present
        fit : If True (training), fit the scalers then transform;
              otherwise only transform (inference / evaluation).

        Returns
        -------
        scaled : np.ndarray of shape (n_samples, n_features)
        """
        X = df[FEATURE_COLUMNS].values
        y = df[[TARGET_COLUMN]].values

        if fit:
            X_scaled = self.scaler_X.fit_transform(X)
            self.scaler_y.fit(y)
        else:
            X_scaled = self.scaler_X.transform(X)

        return X_scaled

    def make_sequences(
        self, scaled: np.ndarray, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build overlapping sliding-window sequences for the LSTM.

        Returns
        -------
        X : (n_windows, lookback, n_features)
        y : (n_windows, horizon)   — multi-step target, *unscaled*

        The target is the first column in FEATURE_COLUMNS (Global_active_power).
        """
        target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
        raw_target = df[TARGET_COLUMN].values

        X_list, y_list = [], []
        total = len(scaled) - self.lookback - self.horizon + 1

        for i in range(total):
            X_list.append(scaled[i : i + self.lookback])
            y_list.append(raw_target[i + self.lookback : i + self.lookback + self.horizon])

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def scale_y(self, y: np.ndarray) -> np.ndarray:
        """Scale a (n, horizon) target array using the fitted scaler_y."""
        return self.scaler_y.transform(y.reshape(-1, 1)).reshape(y.shape)

    def inverse_scale_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform predictions back to kWh."""
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)

    def train_val_test_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """
        Chronological split into train / val / test.
        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        n = len(X)
        n_test = int(n * self.test_split)
        n_val  = int(n * self.val_split)
        n_train = n - n_val - n_test

        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val   = X[n_train : n_train + n_val]
        y_val   = y[n_train : n_train + n_val]
        X_test  = X[n_train + n_val :]
        y_test  = y[n_train + n_val :]

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(X_train), len(X_val), len(X_test),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_scalers(self) -> None:
        """Persist fitted scalers to disk."""
        SCALER_X_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SCALER_X_FILE, "wb") as f:
            pickle.dump(self.scaler_X, f)
        with open(SCALER_Y_FILE, "wb") as f:
            pickle.dump(self.scaler_y, f)
        logger.info("Scalers saved to %s", SCALER_X_FILE.parent)

    def load_scalers(self) -> None:
        """Load previously fitted scalers from disk."""
        with open(SCALER_X_FILE, "rb") as f:
            self.scaler_X = pickle.load(f)
        with open(SCALER_Y_FILE, "rb") as f:
            self.scaler_y = pickle.load(f)
        logger.info("Scalers loaded.")

    def save_processed(self, df: pd.DataFrame) -> None:
        """Save the fully processed DataFrame for reproducibility."""
        df.to_csv(PROCESSED_DATA_FILE)
        logger.info("Processed data saved to %s", PROCESSED_DATA_FILE)

    # ------------------------------------------------------------------
    # Convenience: full pipeline in one call
    # ------------------------------------------------------------------

    def run(
        self,
        raw_filepath: Path = RAW_DATA_FILE,
        weather_filepath: Path | None = None,
        save: bool = True,
    ) -> Tuple[np.ndarray, ...]:
        """
        Execute the complete preprocessing pipeline.

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        (all np.float32 arrays, target already scaled)
        """
        df = self.load_raw(raw_filepath)
        df = self.add_features(df, weather_file=weather_filepath)

        if save:
            self.save_processed(df)

        scaled = self.scale(df, fit=True)
        X, y   = self.make_sequences(scaled, df)
        y_s    = self.scale_y(y)

        splits = self.train_val_test_split(X, y_s)

        if save:
            self.save_scalers()

        return splits

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_synthetic_dataset(n_hours: int = 35_040) -> pd.DataFrame:
        """
        Generate a plausible 4-year hourly power consumption series
        (used when no real UCI file is present).
        """
        np.random.seed(RANDOM_SEED)
        rng = pd.date_range("2019-01-01", periods=n_hours, freq="h")

        hours = rng.hour.values.astype(float)
        dow   = rng.dayofweek.values.astype(float)

        # Diurnal pattern + weekly pattern + trend + noise
        diurnal = 1.2 + 0.8 * np.sin(2 * np.pi * (hours - 8) / 24)
        weekly  = 1.0 - 0.15 * ((dow >= 5).astype(float))  # weekends slightly lower
        trend   = np.linspace(1.0, 1.1, n_hours)          # slight upward drift
        noise   = np.random.normal(0, 0.1, n_hours)

        power = (diurnal * weekly * trend + noise).clip(0.1)

        df = pd.DataFrame({"Global_active_power": power}, index=rng)
        return df
