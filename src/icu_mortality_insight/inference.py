"""Inference helpers for trained ICU mortality models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from icu_mortality_insight.config import DEFAULT_MODEL_PATH, FEATURE_COLUMNS
from icu_mortality_insight.features import add_clinical_features


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")
    return joblib.load(model_path)


def risk_band(probability: float) -> str:
    if probability >= 0.65:
        return "high"
    if probability >= 0.35:
        return "moderate"
    return "low"


def predict_mortality_risk(patient: dict, model_bundle: dict) -> dict:
    frame = add_clinical_features(pd.DataFrame([patient]))
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    model = model_bundle["model"]
    threshold = float(model_bundle.get("threshold", 0.35))
    probability = float(model.predict_proba(frame[FEATURE_COLUMNS])[:, 1][0])
    return {
        "mortality_probability": round(probability, 4),
        "risk_band": risk_band(probability),
        "alert": bool(probability >= threshold),
        "threshold": threshold,
        "model_name": model_bundle.get("model_name", "unknown"),
    }

