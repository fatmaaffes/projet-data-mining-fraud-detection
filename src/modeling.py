"""Supervised and unsupervised fraud models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FraudSplit


@dataclass
class FraudModelResult:
    name: str
    estimator: object
    validation_scores: np.ndarray
    test_scores: np.ndarray
    supervised: bool
    strategy: str


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_models(split: FraudSplit) -> dict[str, FraudModelResult]:
    """Train one supervised model and two unsupervised anomaly detectors."""

    # Compare weighted vs unweighted logistic regression on validation PR-AUC.
    candidates: list[tuple[str, Pipeline]] = []
    for label, class_weight in [
        ("Sans pondération", None),
        ("Pondération balanced", "balanced"),
    ]:
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        split.numeric_features,
                        split.categorical_features,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight=class_weight,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(split.X_train, split.y_train)
        candidates.append((label, pipeline))

    scored_candidates = []
    for label, pipeline in candidates:
        val_scores = pipeline.predict_proba(split.X_validation)[:, 1]
        pr_auc = average_precision_score(
            split.y_validation,
            val_scores,
        )
        scored_candidates.append((pr_auc, label, pipeline))

    best_pr_auc, selected_strategy, supervised_model = max(
        scored_candidates,
        key=lambda item: item[0],
    )
    strategy_scores = {
        label: float(pr_auc)
        for pr_auc, label, _ in scored_candidates
    }
    strategy_description = (
        f"{selected_strategy} retenue sur validation "
        f"(PR-AUC={best_pr_auc:.4f}). "
        + "Comparaison : "
        + " ; ".join(
            f"{label}={score:.4f}"
            for label, score in strategy_scores.items()
        )
    )

    supervised_validation = supervised_model.predict_proba(
        split.X_validation
    )[:, 1]
    supervised_test = supervised_model.predict_proba(
        split.X_test
    )[:, 1]

    # Fit a shared preprocessing transformation for anomaly detectors.
    anomaly_preprocessor = build_preprocessor(
        split.numeric_features,
        split.categorical_features,
    )
    X_train_t = anomaly_preprocessor.fit_transform(split.X_train)
    X_validation_t = anomaly_preprocessor.transform(split.X_validation)
    X_test_t = anomaly_preprocessor.transform(split.X_test)

    isolation = IsolationForest(
        n_estimators=200,
        max_samples=min(256, len(split.X_train)),
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    isolation.fit(X_train_t)

    lof = LocalOutlierFactor(
        n_neighbors=20,
        novelty=True,
        contamination="auto",
        n_jobs=-1,
    )
    # LOF is neighbor-based and expensive on large training sets.
    # A deterministic subset keeps the Streamlit demo responsive.
    rng = np.random.default_rng(42)
    lof_size = min(5000, len(X_train_t))
    lof_indices = rng.choice(
        len(X_train_t),
        size=lof_size,
        replace=False,
    )
    lof.fit(X_train_t[lof_indices])

    # Higher score = more suspicious for every method.
    isolation_validation = -isolation.score_samples(X_validation_t)
    isolation_test = -isolation.score_samples(X_test_t)
    lof_validation = -lof.score_samples(X_validation_t)
    lof_test = -lof.score_samples(X_test_t)

    return {
        "Régression logistique": FraudModelResult(
            name="Régression logistique",
            estimator=supervised_model,
            validation_scores=supervised_validation,
            test_scores=supervised_test,
            supervised=True,
            strategy=strategy_description,
        ),
        "Isolation Forest": FraudModelResult(
            name="Isolation Forest",
            estimator={
                "preprocessor": anomaly_preprocessor,
                "model": isolation,
            },
            validation_scores=isolation_validation,
            test_scores=isolation_test,
            supervised=False,
            strategy="Détection d'anomalies sans labels",
        ),
        "LOF": FraudModelResult(
            name="LOF",
            estimator={
                "preprocessor": anomaly_preprocessor,
                "model": lof,
            },
            validation_scores=lof_validation,
            test_scores=lof_test,
            supervised=False,
            strategy="Détection locale d'anomalies sans labels",
        ),
    }
