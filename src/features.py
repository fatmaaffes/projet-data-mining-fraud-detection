"""Leakage-safe feature engineering and chronological splitting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


DROP_FROM_MODEL = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "ip_address",
    "shipping_address",
    "billing_address",
    "is_fraud",
]


@dataclass
class FraudSplit:
    train_raw: pd.DataFrame
    validation_raw: pd.DataFrame
    test_raw: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]


def _add_customer_history(group: pd.DataFrame) -> pd.DataFrame:
    """Compute behavior variables using only transactions strictly in the past."""

    group = group.sort_values("transaction_date").copy()
    one_hour: deque[tuple[pd.Timestamp, float]] = deque()
    one_day: deque[tuple[pd.Timestamp, float]] = deque()
    thirty_days: deque[tuple[pd.Timestamp, float]] = deque()
    sum_30d = 0.0

    count_1h: list[int] = []
    count_24h: list[int] = []
    mean_30d: list[float] = []
    time_since_prev: list[float] = []
    previous_time: pd.Timestamp | None = None

    for row in group.itertuples():
        current_time = row.transaction_date
        amount = float(row.transaction_amount)

        while one_hour and (
            current_time - one_hour[0][0]
        ).total_seconds() > 3600:
            one_hour.popleft()

        while one_day and (
            current_time - one_day[0][0]
        ).total_seconds() > 86400:
            one_day.popleft()

        while thirty_days and (
            current_time - thirty_days[0][0]
        ).total_seconds() > 30 * 86400:
            _, old_amount = thirty_days.popleft()
            sum_30d -= old_amount

        count_1h.append(len(one_hour))
        count_24h.append(len(one_day))
        mean_30d.append(
            sum_30d / len(thirty_days)
            if thirty_days
            else np.nan
        )
        time_since_prev.append(
            (current_time - previous_time).total_seconds() / 3600
            if previous_time is not None
            else np.nan
        )

        one_hour.append((current_time, amount))
        one_day.append((current_time, amount))
        thirty_days.append((current_time, amount))
        sum_30d += amount
        previous_time = current_time

    group["customer_tx_count_1h"] = count_1h
    group["customer_tx_count_24h"] = count_24h
    group["customer_amount_mean_30d_past"] = mean_30d
    group["hours_since_previous_tx"] = time_since_prev
    group["amount_vs_30d_mean"] = (
        group["transaction_amount"]
        / group["customer_amount_mean_30d_past"].replace(0, np.nan)
    )
    return group


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal and behavioral features without future information."""

    result = df.copy().sort_values("transaction_date")
    result["hour"] = result["transaction_date"].dt.hour
    result["day_of_week"] = result["transaction_date"].dt.dayofweek
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)

    if {
        "shipping_address",
        "billing_address",
    }.issubset(result.columns):
        result["shipping_billing_match"] = (
            result["shipping_address"].astype(str)
            == result["billing_address"].astype(str)
        ).astype(int)

    engineered_parts = []
    for _, group in result.groupby("customer_id", sort=False):
        engineered_parts.append(_add_customer_history(group))

    result = (
        pd.concat(engineered_parts, ignore_index=True)
        .sort_values("transaction_date")
        .reset_index(drop=True)
    )

    # New customers have no historical behavior yet. Zero means "no past event".
    result["customer_tx_count_1h"] = result["customer_tx_count_1h"].fillna(0)
    result["customer_tx_count_24h"] = result["customer_tx_count_24h"].fillna(0)

    return result


def chronological_split(
    featured: pd.DataFrame,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
) -> FraudSplit:
    """Split in time order: 60% train, 20% validation, 20% test."""

    data = featured.sort_values("transaction_date").reset_index(drop=True)
    n = len(data)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))

    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:validation_end].copy()
    test = data.iloc[validation_end:].copy()

    feature_columns = [
        column
        for column in data.columns
        if column not in DROP_FROM_MODEL
    ]

    X_train = train[feature_columns].copy()
    X_validation = validation[feature_columns].copy()
    X_test = test[feature_columns].copy()

    numeric_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_features = [
        column
        for column in X_train.columns
        if column not in numeric_features
    ]

    return FraudSplit(
        train_raw=train,
        validation_raw=validation,
        test_raw=test,
        X_train=X_train,
        y_train=train["is_fraud"].astype(int),
        X_validation=X_validation,
        y_validation=validation["is_fraud"].astype(int),
        X_test=X_test,
        y_test=test["is_fraud"].astype(int),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
