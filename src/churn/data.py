"""Loading, cleaning and feature engineering for the IBM Telco churn dataset."""

import numpy as np
import pandas as pd

from churn.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    ID_COL,
    SERVICE_COLS,
    TARGET,
)


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and known data-quality issues.

    `TotalCharges` is blank for 11 brand-new customers (tenure == 0); they have not been
    billed yet, so 0 is the correct value rather than an imputed mean.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
    # Stored as 0/1 in the source but semantically categorical, like its siblings.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No", "Yes": "Yes", "No": "No"})
    if TARGET in df:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (
        df["InternetService"] != "No"
    ).astype(int)
    # Average historical bill vs. current bill: a positive drift means the price went up.
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"]
    )
    df["charge_drift"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"],
    ).astype(str)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw (or partially raw) customer records -> model-ready feature frame."""
    df = engineer_features(clean(df))
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    return X


def load_dataset(path=DATA_PATH):
    """Return (X, y, ids) for the full dataset."""
    raw = load_raw(path)
    df = clean(raw)
    return prepare_features(raw), df[TARGET], df[ID_COL]
