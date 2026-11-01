"""Data loading and basic EDA helpers for the e-commerce fraud dataset."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "anirudhakolay/Fradulent_E-commerce_Transaction_ClassificationModel/"
    "main/Fraudulent_E-Commerce_Transaction_Data_2.csv"
)

KAGGLE_SOURCE = (
    "https://www.kaggle.com/datasets/shriyashjagtap/"
    "fraudulent-e-commerce-transactions"
)

COLUMN_NAMES = {
    "Transaction ID": "transaction_id",
    "Customer ID": "customer_id",
    "Transaction Amount": "transaction_amount",
    "Transaction Date": "transaction_date",
    "Payment Method": "payment_method",
    "Product Category": "product_category",
    "Quantity": "quantity",
    "Customer Age": "customer_age",
    "Customer Location": "customer_location",
    "Device Used": "device_used",
    "IP Address": "ip_address",
    "Shipping Address": "shipping_address",
    "Billing Address": "billing_address",
    "Is Fraudulent": "is_fraud",
    "Account Age Days": "account_age_days",
    "Transaction Hour": "transaction_hour",
}


@lru_cache(maxsize=1)
def load_fraud_data() -> pd.DataFrame:
    """Load the 23k-row public sample and return a cleaned dataframe."""

    df = pd.read_csv(DATA_URL)
    df = df.rename(columns=COLUMN_NAMES)

    expected = {
        "transaction_id",
        "customer_id",
        "transaction_amount",
        "transaction_date",
        "payment_method",
        "product_category",
        "quantity",
        "customer_age",
        "customer_location",
        "device_used",
        "is_fraud",
        "account_age_days",
    }
    missing = sorted(expected.difference(df.columns))
    if missing:
        raise ValueError(f"Colonnes attendues absentes du dataset : {missing}")

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"],
        errors="coerce",
    )
    df["is_fraud"] = pd.to_numeric(
        df["is_fraud"],
        errors="coerce",
    ).fillna(0).astype(int)

    numeric_columns = [
        "transaction_amount",
        "quantity",
        "customer_age",
        "account_age_days",
    ]
    if "transaction_hour" in df.columns:
        numeric_columns.append("transaction_hour")

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["transaction_date", "is_fraud"])
        .sort_values("transaction_date")
        .reset_index(drop=True)
    )
    return df


def fraud_summary(df: pd.DataFrame) -> dict[str, float | int]:
    total = len(df)
    frauds = int(df["is_fraud"].sum())
    return {
        "transactions": total,
        "frauds": frauds,
        "normal": total - frauds,
        "fraud_rate": frauds / total if total else 0.0,
        "amount_total": float(df["transaction_amount"].sum()),
        "fraud_amount": float(
            df.loc[df["is_fraud"].eq(1), "transaction_amount"].sum()
        ),
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "variable": df.columns,
            "valeurs_manquantes": df.isna().sum().values,
            "pourcentage": (100 * df.isna().mean()).round(2).values,
        }
    )
    return result.sort_values(
        ["valeurs_manquantes", "variable"],
        ascending=[False, True],
    ).reset_index(drop=True)
