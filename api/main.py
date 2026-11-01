"""FastAPI service for fraud prediction and local explanation."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_fraud_data
from src.evaluation import optimize_threshold
from src.explainability import compute_shap, top_reason_codes
from src.features import build_features, chronological_split
from src.modeling import train_models


app = FastAPI(
    title="Explainable E-Commerce Fraud Detection API",
    version="1.0.0",
    description="API de scoring et d'explication locale du modèle supervisé.",
)


class FraudRequest(BaseModel):
    features: dict[str, Any]
    threshold: float | None = Field(default=None)


@lru_cache(maxsize=1)
def get_bundle():
    data = load_fraud_data()
    featured = build_features(data)
    split = chronological_split(featured)
    results = train_models(split)
    supervised = results["Régression logistique"]

    recommendation = optimize_threshold(
        split.y_validation,
        supervised.validation_scores,
        split.validation_raw["transaction_amount"],
        alert_cost=5.0,
    )
    return split, supervised, recommendation


def _to_frame(features: dict[str, Any]) -> pd.DataFrame:
    split, _, _ = get_bundle()
    missing = [
        column
        for column in split.X_train.columns
        if column not in features
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail="Variables préparées manquantes : " + ", ".join(missing),
        )

    row = {
        column: features[column]
        for column in split.X_train.columns
    }
    return pd.DataFrame([row])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "fraud-detection-xai",
    }


@app.post("/predict")
def predict(request: FraudRequest):
    _, supervised, recommendation = get_bundle()
    row = _to_frame(request.features)

    probability = float(
        supervised.estimator.predict_proba(row)[0, 1]
    )
    threshold = (
        float(request.threshold)
        if request.threshold is not None
        else float(recommendation["threshold"])
    )

    return {
        "model": "Régression logistique",
        "fraud_score": probability,
        "threshold": threshold,
        "alert": bool(probability >= threshold),
    }


@app.post("/explain")
def explain(request: FraudRequest):
    split, supervised, recommendation = get_bundle()
    row = _to_frame(request.features)

    probability = float(
        supervised.estimator.predict_proba(row)[0, 1]
    )
    threshold = (
        float(request.threshold)
        if request.threshold is not None
        else float(recommendation["threshold"])
    )

    shap_values, feature_names = compute_shap(
        supervised.estimator,
        split.X_train,
        row,
    )
    reasons = top_reason_codes(
        shap_values[0],
        feature_names,
        top_n=5,
    )

    return {
        "model": "Régression logistique",
        "fraud_score": probability,
        "threshold": threshold,
        "alert": bool(probability >= threshold),
        "reason_codes": reasons.to_dict(orient="records"),
    }
