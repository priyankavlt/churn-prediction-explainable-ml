import numpy as np
import pandas as pd

from churn.config import FEATURES
from churn.data import clean, prepare_features


def test_blank_total_charges_become_zero_for_new_customers(raw_df):
    blank = raw_df["TotalCharges"].str.strip() == ""
    assert blank.sum() == 11 and (raw_df.loc[blank, "tenure"] == 0).all()
    cleaned = clean(raw_df)
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert (cleaned.loc[blank, "TotalCharges"] == 0).all()


def test_target_is_binary(raw_df):
    assert set(clean(raw_df)["Churn"].unique()) == {0, 1}


def test_prepare_features_shape_and_no_nans(raw_df):
    X = prepare_features(raw_df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(raw_df)
    assert not X.isna().any().any()


def test_engineered_features(customer):
    X = prepare_features(pd.DataFrame([customer]))
    row = X.iloc[0]
    # Phone + fiber + StreamingTV + StreamingMovies
    assert row["num_services"] == 4
    assert np.isclose(row["avg_monthly_spend"], 280.5 / 3)
    assert row["tenure_band"] == "0-6m"


def test_senior_citizen_accepts_int_or_string(customer):
    as_int = prepare_features(pd.DataFrame([{**customer, "SeniorCitizen": 1}]))
    as_str = prepare_features(pd.DataFrame([{**customer, "SeniorCitizen": "Yes"}]))
    assert as_int["SeniorCitizen"].iloc[0] == as_str["SeniorCitizen"].iloc[0] == "Yes"
