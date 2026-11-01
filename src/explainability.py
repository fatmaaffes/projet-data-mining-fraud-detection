"""SHAP explanations for the supervised fraud detector."""

from __future__ import annotations

import numpy as np
import pandas as pd


def transformed_data(model_pipeline, X: pd.DataFrame):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["model"]
    transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    return classifier, transformed, feature_names


def compute_shap(
    model_pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
):
    import shap

    classifier, background_t, feature_names = transformed_data(
        model_pipeline,
        X_background,
    )
    _, explain_t, _ = transformed_data(
        model_pipeline,
        X_explain,
    )

    if len(background_t) > 200:
        rng = np.random.default_rng(42)
        selected = rng.choice(
            len(background_t),
            size=200,
            replace=False,
        )
        background_t = background_t[selected]

    explainer = shap.Explainer(
        classifier,
        background_t,
        feature_names=feature_names,
    )
    values = explainer(explain_t)
    return values, feature_names


def top_reason_codes(
    shap_row,
    feature_names: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    values = np.asarray(shap_row.values).reshape(-1)
    order = np.argsort(np.abs(values))[::-1][:top_n]

    rows = []
    for index in order:
        contribution = float(values[index])
        rows.append(
            {
                "variable": feature_names[index],
                "contribution": contribution,
                "motif": (
                    "Augmente le risque d'alerte"
                    if contribution > 0
                    else "Réduit le risque d'alerte"
                ),
            }
        )
    return pd.DataFrame(rows)
