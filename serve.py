"""
serve.py
--------
FastAPI REST endpoint for Urban Energy Consumption Forecasting.

Endpoints
---------
GET  /health          — liveness probe
GET  /model/info      — model metadata
POST /predict         — 24-h consumption forecast for one district
POST /predict/batch   — forecast for multiple districts

Start the server:
    python serve.py
    # or via uvicorn directly:
    uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from config import (
    MODEL_SAVE_PATH,
    LOOKBACK,
    HORIZON,
    FEATURE_COLUMNS,
    API_HOST,
    API_PORT,
)
from src.preprocess import DataPreprocessor
from src.model import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state — loaded once on startup
# ---------------------------------------------------------------------------

class AppState:
    model        = None
    preprocessor = None
    model_loaded = False

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts on startup, release on shutdown."""
    logger.info("Loading model from %s ...", MODEL_SAVE_PATH)
    try:
        state.preprocessor = DataPreprocessor()
        state.preprocessor.load_scalers()
        state.model = load_model(MODEL_SAVE_PATH)
        state.model_loaded = True
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.warning(
            "Could not load model (%s). /predict will return 503 until a model is trained.",
            exc,
        )
        state.model_loaded = False
    yield
    # Shutdown — nothing to clean up for this service
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Urban Energy Consumption Forecasting API",
    description=(
        "Predicts hourly energy consumption for urban districts "
        "using a stacked LSTM model. "
        "Returns a 24-hour forecast vector in kWh."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HourlyReading(BaseModel):
    """One hour of sensor / weather readings for a district."""
    power_kwh:        float = Field(..., ge=0,          description="Observed power (kWh)")
    temperature:      float = Field(...,                 description="°C")
    humidity:         float = Field(..., ge=0, le=100,  description="%")
    wind_speed:       float = Field(..., ge=0,          description="m/s")
    solar_irradiance: float = Field(..., ge=0,          description="W/m²")
    hour:             int   = Field(..., ge=0, le=23,   description="Hour of day (0-23)")
    day_of_week:      int   = Field(..., ge=0, le=6,    description="0=Mon … 6=Sun")
    is_holiday:       float = Field(0.0, ge=0, le=1,   description="1 if public holiday")


class PredictRequest(BaseModel):
    """Sequence of LOOKBACK hourly readings for a single district."""
    district_id: Optional[str] = Field(None, description="District identifier (optional)")
    history:     List[HourlyReading] = Field(
        ...,
        min_length=LOOKBACK,
        max_length=LOOKBACK,
        description=f"Exactly {LOOKBACK} hourly readings (oldest first)",
    )

    @field_validator("history")
    @classmethod
    def check_length(cls, v: list) -> list:
        if len(v) != LOOKBACK:
            raise ValueError(f"history must contain exactly {LOOKBACK} readings, got {len(v)}")
        return v


class ForecastStep(BaseModel):
    hour_ahead:     int
    forecast_kwh:   float


class PredictResponse(BaseModel):
    district_id:   Optional[str]
    forecast:      List[ForecastStep]
    horizon_hours: int = HORIZON


class BatchPredictRequest(BaseModel):
    districts: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


# ---------------------------------------------------------------------------
# Helper: readings → scaled numpy sequence
# ---------------------------------------------------------------------------

def _readings_to_sequence(history: List[HourlyReading]) -> np.ndarray:
    """
    Convert a list of HourlyReading objects to a scaled (1, LOOKBACK, n_features)
    numpy array ready for model.predict().
    """
    import pandas as pd

    rows = []
    for r in history:
        # Cyclic encodings
        hr_sin = float(np.sin(2 * np.pi * r.hour / 24))
        hr_cos = float(np.cos(2 * np.pi * r.hour / 24))
        dw_sin = float(np.sin(2 * np.pi * r.day_of_week / 7))
        dw_cos = float(np.cos(2 * np.pi * r.day_of_week / 7))

        rows.append({
            "Global_active_power": r.power_kwh,
            "temperature":         r.temperature,
            "humidity":            r.humidity,
            "wind_speed":          r.wind_speed,
            "solar_irradiance":    r.solar_irradiance,
            "hour_sin":            hr_sin,
            "hour_cos":            hr_cos,
            "dow_sin":             dw_sin,
            "dow_cos":             dw_cos,
            "is_holiday":          r.is_holiday,
        })

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    scaled = state.preprocessor.scaler_X.transform(df.values)
    return scaled[np.newaxis, :, :]   # (1, LOOKBACK, n_features)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": state.model_loaded,
    }


@app.get("/model/info", tags=["System"])
def model_info() -> dict:
    return {
        "model_path":  str(MODEL_SAVE_PATH),
        "lookback":    LOOKBACK,
        "horizon":     HORIZON,
        "n_features":  len(FEATURE_COLUMNS),
        "features":    FEATURE_COLUMNS,
        "loaded":      state.model_loaded,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Return a 24-hour energy consumption forecast for a single district.

    The EMS calls this endpoint when it needs to schedule renewable reserves
    or trigger demand-response programs before an anticipated peak.
    """
    if not state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first with train.py.",
        )

    X = _readings_to_sequence(request.history)

    try:
        y_scaled = state.model.predict(X, verbose=0)                        # (1, HORIZON)
        y_kwh    = state.preprocessor.inverse_scale_y(y_scaled).ravel()     # (HORIZON,)
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    forecast = [
        ForecastStep(hour_ahead=h + 1, forecast_kwh=round(float(y_kwh[h]), 4))
        for h in range(HORIZON)
    ]

    return PredictResponse(
        district_id=request.district_id,
        forecast=forecast,
        horizon_hours=HORIZON,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Forecasting"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Forecast for multiple districts in a single round-trip."""
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for district_req in request.districts:
        resp = predict(district_req)
        results.append(resp)

    return BatchPredictResponse(results=results)


# ---------------------------------------------------------------------------
# Dev-server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host=API_HOST, port=API_PORT, reload=True)
