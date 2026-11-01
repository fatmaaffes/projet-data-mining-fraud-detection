"""Fraud-specific evaluation metrics and business threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_recall_at_k(
    y_true,
    scores,
    k: int,
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    k = int(max(1, min(k, len(y))))
    order = np.argsort(scores)[::-1][:k]
    selected = y[order]
    precision = float(selected.mean())
    total_frauds = int(y.sum())
    recall = (
        float(selected.sum() / total_frauds)
        if total_frauds
        else 0.0
    )
    return precision, recall


def business_cost(
    y_true,
    scores,
    amounts,
    threshold: float,
    alert_cost: float,
) -> float:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    alerts = scores >= threshold
    missed_fraud = (y == 1) & (~alerts)
    return float(
        amounts[missed_fraud].sum()
        + alert_cost * alerts.sum()
    )


def optimize_threshold(
    y_true,
    scores,
    amounts,
    alert_cost: float = 5.0,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)

    quantiles = np.linspace(0.0, 1.0, 201)
    candidates = np.unique(np.quantile(scores, quantiles))

    rows = []
    for threshold in candidates:
        cost = business_cost(
            y_true,
            scores,
            amounts,
            threshold,
            alert_cost,
        )
        rows.append((float(cost), float(threshold)))

    best_cost, best_threshold = min(rows, key=lambda item: item[0])
    return {
        "threshold": best_threshold,
        "cost": best_cost,
    }


def metrics_at_threshold(
    y_true,
    scores,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    predictions = (np.asarray(scores) >= threshold).astype(int)

    return {
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y,
            predictions,
            beta=2,
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(y, predictions),
        "alerts": int(predictions.sum()),
    }


def comparison_table(
    results,
    split,
    alert_cost: float,
    k: int,
) -> pd.DataFrame:
    """Choose every threshold on validation, then report final test metrics."""

    rows = []
    validation_amounts = split.validation_raw["transaction_amount"].to_numpy()
    test_amounts = split.test_raw["transaction_amount"].to_numpy()

    for name, result in results.items():
        threshold_info = optimize_threshold(
            split.y_validation,
            result.validation_scores,
            validation_amounts,
            alert_cost,
        )
        threshold = threshold_info["threshold"]

        threshold_metrics = metrics_at_threshold(
            split.y_test,
            result.test_scores,
            threshold,
        )
        p_at_k, r_at_k = precision_recall_at_k(
            split.y_test,
            result.test_scores,
            k,
        )
        test_cost = business_cost(
            split.y_test,
            result.test_scores,
            test_amounts,
            threshold,
            alert_cost,
        )

        rows.append(
            {
                "Méthode": name,
                "PR-AUC test": average_precision_score(
                    split.y_test,
                    result.test_scores,
                ),
                "ROC-AUC test": roc_auc_score(
                    split.y_test,
                    result.test_scores,
                ),
                f"Précision@{k}": p_at_k,
                f"Rappel@{k}": r_at_k,
                "F2 test": threshold_metrics["f2"],
                "MCC test": threshold_metrics["mcc"],
                "Seuil validation": threshold,
                "Coût validation": threshold_info["cost"],
                "Coût test": test_cost,
                "Alertes test": threshold_metrics["alerts"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["PR-AUC test", "Coût test"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
