"""
train.py
--------
Entry point for training the Urban Energy Consumption LSTM.

Usage examples
--------------
# Basic run (uses defaults from config.py):
    python train.py

# Custom hyperparameters:
    python train.py --epochs 100 --lookback 72 --batch_size 128

# Skip reprocessing if processed data already exists:
    python train.py --skip_preprocess
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering during training

from config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    MODEL_SAVE_PATH,
    LOOKBACK,
    HORIZON,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    FEATURE_COLUMNS,
    MODELS_DIR,
)
from src.preprocess import DataPreprocessor
from src.model import build_lstm_model, train_model
from src.evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Urban Energy Consumption LSTM model."
    )
    p.add_argument("--epochs",         type=int,   default=EPOCHS,
                   help=f"Training epochs (default: {EPOCHS})")
    p.add_argument("--lookback",       type=int,   default=LOOKBACK,
                   help=f"Sequence lookback window in hours (default: {LOOKBACK})")
    p.add_argument("--horizon",        type=int,   default=HORIZON,
                   help=f"Forecast horizon in hours (default: {HORIZON})")
    p.add_argument("--batch_size",     type=int,   default=BATCH_SIZE,
                   help=f"Mini-batch size (default: {BATCH_SIZE})")
    p.add_argument("--lr",             type=float, default=LEARNING_RATE,
                   help=f"Adam learning rate (default: {LEARNING_RATE})")
    p.add_argument("--raw_data",       type=str,   default=str(RAW_DATA_FILE),
                   help="Path to raw UCI CSV file")
    p.add_argument("--weather_data",   type=str,   default=None,
                   help="Optional path to weather CSV file")
    p.add_argument("--skip_preprocess", action="store_true",
                   help="Skip preprocessing if processed data already exists")
    p.add_argument("--save_plots",     action="store_true",
                   help="Save evaluation plots to models/ directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Urban Energy Consumption Forecasting — Training")
    logger.info("=" * 60)
    logger.info("Config: lookback=%d, horizon=%d, epochs=%d, batch=%d, lr=%.5f",
                args.lookback, args.horizon, args.epochs, args.batch_size, args.lr)

    # ---------------------------------------------------------------
    # 1. Preprocessing
    # ---------------------------------------------------------------
    preprocessor = DataPreprocessor(
        lookback=args.lookback,
        horizon=args.horizon,
    )

    if args.skip_preprocess and PROCESSED_DATA_FILE.exists():
        logger.info("Skipping preprocessing — loading from %s", PROCESSED_DATA_FILE)
        import pandas as pd
        df = pd.read_csv(PROCESSED_DATA_FILE, index_col=0, parse_dates=True)
        scaled = preprocessor.scale(df, fit=True)
        X, y   = preprocessor.make_sequences(scaled, df)
        y_s    = preprocessor.scale_y(y)
        preprocessor.save_scalers()
        splits = preprocessor.train_val_test_split(X, y_s)
    else:
        splits = preprocessor.run(
            raw_filepath=Path(args.raw_data),
            weather_filepath=Path(args.weather_data) if args.weather_data else None,
            save=True,
        )

    X_train, X_val, X_test, y_train, y_val, y_test = splits

    logger.info("Tensor shapes — X_train: %s  y_train: %s", X_train.shape, y_train.shape)

    # ---------------------------------------------------------------
    # 2. Build model
    # ---------------------------------------------------------------
    n_features = X_train.shape[2]
    model = build_lstm_model(
        n_features=n_features,
        lookback=args.lookback,
        horizon=args.horizon,
        learning_rate=args.lr,
    )
    model.summary(print_fn=logger.info)

    # ---------------------------------------------------------------
    # 3. Train
    # ---------------------------------------------------------------
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=MODEL_SAVE_PATH,
    )

    # ---------------------------------------------------------------
    # 4. Evaluate
    # ---------------------------------------------------------------
    evaluator = ModelEvaluator(model=model, preprocessor=preprocessor)
    metrics   = evaluator.evaluate(X_test, y_test)

    # ---------------------------------------------------------------
    # 5. (Optional) Save plots
    # ---------------------------------------------------------------
    if args.save_plots:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        evaluator.plot_predictions(X_test, y_test,
                                   save_path=MODELS_DIR / "predictions.png")
        evaluator.plot_horizon_errors(X_test, y_test,
                                      save_path=MODELS_DIR / "horizon_errors.png")
        evaluator.plot_residuals(X_test, y_test,
                                 save_path=MODELS_DIR / "residuals.png")
        logger.info("Evaluation plots saved to %s", MODELS_DIR)

    # ---------------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("  MAE  = %.4f kWh", metrics["MAE"])
    logger.info("  RMSE = %.4f kWh", metrics["RMSE"])
    logger.info("  R²   = %.4f",     metrics["R2"])
    logger.info("  Model saved → %s", MODEL_SAVE_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
