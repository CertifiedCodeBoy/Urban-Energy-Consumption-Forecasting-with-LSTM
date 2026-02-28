"""
src/evaluate.py
---------------
Evaluation utilities for the Urban Energy Consumption Forecasting LSTM.

Provides:
- Regression metrics (MAE, RMSE, R², MAPE)
- ModelEvaluator class that orchestrates prediction + metric reporting
- Plotting helpers (actual vs predicted, residuals, horizon error profile)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (avoids division by zero)."""
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of all regression metrics."""
    return {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2":   r2(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Wraps a trained Keras model and a fitted DataPreprocessor to provide
    prediction, metric computation, and visualization in one place.
    """

    def __init__(self, model, preprocessor) -> None:
        """
        Parameters
        ----------
        model        : trained tf.keras.Model
        preprocessor : fitted DataPreprocessor instance (scalers must be loaded)
        """
        self.model        = model
        self.preprocessor = preprocessor

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference and inverse-scale predictions to kWh.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, lookback, n_features) — scaled

        Returns
        -------
        np.ndarray of shape (n_samples, horizon) — kWh
        """
        y_scaled = self.model.predict(X, verbose=0)
        return self.preprocessor.inverse_scale_y(y_scaled)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test_scaled: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Compute metrics on the test set.

        Parameters
        ----------
        X_test        : scaled input sequences
        y_test_scaled : scaled target sequences (shape n × horizon)
        verbose       : print a formatted report

        Returns
        -------
        dict with MAE, RMSE, R2, MAPE
        """
        y_pred_kw = self.predict(X_test)
        y_true_kw = self.preprocessor.inverse_scale_y(y_test_scaled)

        metrics = compute_all_metrics(y_true_kw.ravel(), y_pred_kw.ravel())

        if verbose:
            print("\n" + "=" * 50)
            print("  TEST SET EVALUATION RESULTS")
            print("=" * 50)
            print(f"  MAE   : {metrics['MAE']:.4f} kWh")
            print(f"  RMSE  : {metrics['RMSE']:.4f} kWh")
            print(f"  R²    : {metrics['R2']:.4f}")
            print(f"  MAPE  : {metrics['MAPE']:.2f} %")
            print("=" * 50 + "\n")

        return metrics

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_predictions(
        self,
        X_test: np.ndarray,
        y_test_scaled: np.ndarray,
        n_samples: int = 168,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot actual vs predicted energy consumption over a rolling horizon.

        Uses the first element of each multi-step prediction (1-hour-ahead)
        for a continuous time-series comparison.
        """
        y_pred_kw = self.predict(X_test)[:, 0]
        y_true_kw = self.preprocessor.inverse_scale_y(y_test_scaled)[:, 0]

        n = min(n_samples, len(y_true_kw))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(y_true_kw[:n], label="Actual", linewidth=1.2, color="#1f77b4")
        ax.plot(y_pred_kw[:n], label="Predicted", linewidth=1.2,
                color="#ff7f0e", linestyle="--")
        ax.set_title("Energy Consumption — Actual vs Predicted (1-step ahead)", fontsize=13)
        ax.set_xlabel("Hour index")
        ax.set_ylabel("Global Active Power (kWh)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info("Plot saved to %s", save_path)
        return fig

    def plot_horizon_errors(
        self,
        X_test: np.ndarray,
        y_test_scaled: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Show how MAE grows with forecast horizon (step 1 → HORIZON).
        Useful for communicating model reliability at longer lead times.
        """
        y_pred_kw = self.predict(X_test)
        y_true_kw = self.preprocessor.inverse_scale_y(y_test_scaled)

        horizon = y_pred_kw.shape[1]
        step_mae = [mae(y_true_kw[:, h], y_pred_kw[:, h]) for h in range(horizon)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(1, horizon + 1), step_mae, color="#2ca02c", alpha=0.8)
        ax.set_title("Forecast MAE by Horizon Step", fontsize=13)
        ax.set_xlabel("Forecast Step (h ahead)")
        ax.set_ylabel("MAE (kWh)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info("Plot saved to %s", save_path)
        return fig

    def plot_residuals(
        self,
        X_test: np.ndarray,
        y_test_scaled: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Residual distribution (1-step forecasts)."""
        y_pred_kw = self.predict(X_test)[:, 0]
        y_true_kw = self.preprocessor.inverse_scale_y(y_test_scaled)[:, 0]
        residuals = y_true_kw - y_pred_kw

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(residuals, bins=60, color="#9467bd", edgecolor="white", alpha=0.85)
        axes[0].axvline(0, color="red", linewidth=1.2, linestyle="--")
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual (kWh)")
        axes[0].set_ylabel("Count")

        # Q-Q style scatter: predicted vs actual
        axes[1].scatter(y_true_kw, y_pred_kw, alpha=0.3, s=6, color="#8c564b")
        lim = [min(y_true_kw.min(), y_pred_kw.min()),
               max(y_true_kw.max(), y_pred_kw.max())]
        axes[1].plot(lim, lim, "r--", linewidth=1.2)
        axes[1].set_title("Predicted vs Actual")
        axes[1].set_xlabel("Actual (kWh)")
        axes[1].set_ylabel("Predicted (kWh)")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info("Plot saved to %s", save_path)
        return fig
