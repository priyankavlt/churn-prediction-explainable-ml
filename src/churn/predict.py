"""Inference: churn probability, SHAP explanation and retention recommendation.

Shared by the FastAPI service and the Streamlit app so both give identical answers.
"""

import json
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool

from churn.config import (
    CATEGORICAL_FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    RetentionEconomics,
)
from churn.data import prepare_features

RISK_TIERS = [(0.30, "low"), (0.60, "medium"), (1.01, "high")]

# Actionable drivers -> what a retention team could do about them.
# Each rule fires only when that feature is pushing *this* customer's churn risk up.
ACTIONS = {
    "Contract": (
        lambda v: v == "Month-to-month",
        "Offer a discounted 1- or 2-year contract upgrade.",
    ),
    "OnlineSecurity": (lambda v: v == "No", "Bundle a free 3-month Online Security trial."),
    "TechSupport": (lambda v: v == "No", "Offer complimentary Tech Support onboarding."),
    "PaymentMethod": (
        lambda v: v == "Electronic check",
        "Incentivise switching to automatic card/bank payment.",
    ),
    "InternetService": (
        lambda v: v == "Fiber optic",
        "Proactive fiber service-quality check-in.",
    ),
    "MonthlyCharges": (lambda v: v > 70, "Review plan fit; propose a right-sized bundle."),
    "tenure": (lambda v: v <= 12, "Enrol in new-customer success / onboarding outreach."),
    "PaperlessBilling": (lambda v: v == "Yes", "Send clear, itemised e-bills with usage summary."),
}


def risk_tier(p: float) -> str:
    return next(name for cutoff, name in RISK_TIERS if p < cutoff)


@dataclass
class Contribution:
    feature: str
    value: object
    shap_value: float


@dataclass
class Prediction:
    churn_probability: float
    risk_tier: str
    expected_value_of_contact: float
    recommend_contact: bool
    base_value: float | None = None
    contributions: list[Contribution] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


class ChurnPredictor:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))
        self.metadata = json.loads(metadata_path.read_text())
        self.economics = RetentionEconomics(**self.metadata["economics"])

    @cached_property
    def explainer(self):
        return shap.TreeExplainer(self.model)

    def _features(self, records) -> pd.DataFrame:
        df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
        return prepare_features(df)

    def predict_proba(self, records) -> np.ndarray:
        return self.model.predict_proba(self._features(records))[:, 1]

    def score(self, records) -> pd.DataFrame:
        """Batch scoring: probability, tier, expected value and contact decision per row."""
        X = self._features(records)
        proba = self.model.predict_proba(X)[:, 1]
        ev = self.economics.expected_value(proba, X["MonthlyCharges"].to_numpy())
        return pd.DataFrame(
            {
                "churn_probability": proba.round(4),
                "risk_tier": [risk_tier(p) for p in proba],
                "expected_value_of_contact": ev.round(2),
                "recommend_contact": ev > 0,
            },
            index=X.index,
        )

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        return self.explainer.shap_values(Pool(X, cat_features=CATEGORICAL_FEATURES))

    def explain(self, record: dict, top_k: int | None = None) -> Prediction:
        """Single-customer prediction with per-feature SHAP contributions (log-odds scale)."""
        X = self._features([record])
        proba = float(self.model.predict_proba(X)[0, 1])
        ev = float(self.economics.expected_value(proba, float(X["MonthlyCharges"].iloc[0])))
        sv = self.shap_values(X)[0]
        row = X.iloc[0]
        contributions = sorted(
            (
                Contribution(f, _py(row[f]), round(float(s), 4))
                for f, s in zip(X.columns, sv, strict=True)
            ),
            key=lambda c: abs(c.shap_value),
            reverse=True,
        )
        actions = [
            ACTIONS[c.feature][1]
            for c in contributions
            if c.shap_value > 0 and c.feature in ACTIONS and ACTIONS[c.feature][0](c.value)
        ]
        return Prediction(
            churn_probability=round(proba, 4),
            risk_tier=risk_tier(proba),
            expected_value_of_contact=round(ev, 2),
            recommend_contact=ev > 0,
            base_value=round(float(np.ravel(self.explainer.expected_value)[0]), 4),
            contributions=contributions[:top_k] if top_k else contributions,
            actions=actions[:3],
        )


def _py(v):
    """numpy scalar -> plain Python for JSON serialisation."""
    return v.item() if hasattr(v, "item") else v
