"""Project paths, feature definitions and business assumptions."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "catboost_churn.cbm"
METADATA_PATH = MODEL_DIR / "metadata.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42

# Raw input columns a caller must provide (everything in the dataset except ID and target).
CATEGORICAL_INPUTS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_INPUTS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Features derived in `data.engineer_features`.
ENGINEERED_NUMERIC = ["num_services", "avg_monthly_spend", "charge_drift"]
ENGINEERED_CATEGORICAL = ["tenure_band"]

CATEGORICAL_FEATURES = CATEGORICAL_INPUTS + ENGINEERED_CATEGORICAL
NUMERIC_FEATURES = NUMERIC_INPUTS + ENGINEERED_NUMERIC
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


@dataclass(frozen=True)
class RetentionEconomics:
    """Assumptions used to turn churn probabilities into a targeting decision.

    A customer we contact receives an offer costing `offer_cost`. If they were going to
    churn, the offer saves them with probability `save_rate`, and we keep
    `value_horizon_months` of their monthly revenue.
    """

    offer_cost: float = 50.0
    save_rate: float = 0.35
    value_horizon_months: int = 12

    def expected_value(self, churn_prob, monthly_charges):
        """Expected profit of contacting a customer (vectorised)."""
        saved_revenue = monthly_charges * self.value_horizon_months
        return churn_prob * self.save_rate * saved_revenue - self.offer_cost


ECONOMICS = RetentionEconomics()
