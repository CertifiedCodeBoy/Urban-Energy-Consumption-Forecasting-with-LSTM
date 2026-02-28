"""
src/model.py
------------
LSTM model definition for multi-step urban energy consumption forecasting.

Architecture (matches README diagram):
    Input Sequence (LOOKBACK × n_features)
            │
       [LSTM Layer 1]  — 128 units, return_sequences=True
            │
       [Dropout 0.2]
            │
       [LSTM Layer 2]  — 64 units
            │
       [Dense Layer]   — 32 units, ReLU
            │
       [Output Layer]  — HORIZON units (next 24 h forecast)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LOOKBACK,
    HORIZON,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DENSE_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    FEATURE_COLUMNS,
    MODEL_SAVE_PATH,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_lstm_model(
    n_features: int = len(FEATURE_COLUMNS),
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    lstm_units_1: int = LSTM_UNITS_1,
    lstm_units_2: int = LSTM_UNITS_2,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> "tf.keras.Model":  # type: ignore[name-defined]
    """
    Build and compile the stacked LSTM model.

    Parameters
    ----------
    n_features    : number of input features per time-step
    lookback      : sequence length (time steps)
    horizon       : number of future hours to predict
    lstm_units_1  : units in the first LSTM layer
    lstm_units_2  : units in the second LSTM layer
    dense_units   : units in the intermediate Dense layer
    dropout_rate  : dropout fraction between LSTM layers
    learning_rate : Adam learning rate

    Returns
    -------
    Compiled tf.keras.Model
    """
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    inputs = tf.keras.Input(shape=(lookback, n_features), name="sequence_input")

    x = tf.keras.layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        name="lstm_1",
    )(inputs)

    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = tf.keras.layers.LSTM(
        lstm_units_2,
        return_sequences=False,
        name="lstm_2",
    )(x)

    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_1")(x)

    outputs = tf.keras.layers.Dense(horizon, activation="linear", name="forecast_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="urban_energy_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_model(
    model: "tf.keras.Model",  # type: ignore[name-defined]
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    save_path: Path = MODEL_SAVE_PATH,
) -> "tf.keras.callbacks.History":  # type: ignore[name-defined]
    """
    Train the model with early stopping and model checkpointing.

    Returns the Keras History object.
    """
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info(
        "Starting training — epochs=%d, batch_size=%d, train_samples=%d",
        epochs, batch_size, len(X_train),
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Training complete. Best model saved to %s", save_path)
    return history


# ---------------------------------------------------------------------------
# Loading helper
# ---------------------------------------------------------------------------

def load_model(path: Path = MODEL_SAVE_PATH) -> "tf.keras.Model":  # type: ignore[name-defined]
    """Load a saved Keras model from disk."""
    import tensorflow as tf

    logger.info("Loading model from %s ...", path)
    model = tf.keras.models.load_model(str(path))
    return model
