"""Model evaluation: statistical metrics plus business (campaign profit) metrics."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import ECONOMICS, RetentionEconomics


def classification_metrics(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "lift_top_decile": lift_at(y_true, proba, 0.10),
        "recall_top_20pct": capture_at(y_true, proba, 0.20),
    }


def lift_at(y_true, proba, fraction) -> float:
    """Churn rate among the top `fraction` highest-risk customers, relative to the base rate."""
    y_true = np.asarray(y_true)
    k = max(1, int(round(len(y_true) * fraction)))
    top = np.argsort(-proba)[:k]
    return y_true[top].mean() / y_true.mean()


def capture_at(y_true, proba, fraction) -> float:
    """Share of all churners found in the top `fraction` highest-risk customers."""
    y_true = np.asarray(y_true)
    k = max(1, int(round(len(y_true) * fraction)))
    top = np.argsort(-proba)[:k]
    return y_true[top].sum() / y_true.sum()


def best_f1_threshold(y_true, proba) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def realized_profit(y_true, contacted, monthly_charges, econ: RetentionEconomics = ECONOMICS):
    """Expected campaign profit given who actually churned.

    For each contacted customer we pay the offer cost; for each contacted churner we
    recover `save_rate` x their value over the horizon.
    """
    y_true = np.asarray(y_true)
    contacted = np.asarray(contacted, dtype=bool)
    value = np.asarray(monthly_charges) * econ.value_horizon_months
    gain = (y_true * econ.save_rate * value)[contacted].sum()
    return float(gain - econ.offer_cost * contacted.sum())


def campaign_comparison(y_true, proba, monthly_charges, f1_threshold, econ=ECONOMICS) -> dict:
    """Profit of several targeting policies on the same customers."""
    n = len(y_true)
    ev = econ.expected_value(proba, np.asarray(monthly_charges))
    policies = {
        "contact_nobody": np.zeros(n, dtype=bool),
        "contact_everyone": np.ones(n, dtype=bool),
        "f1_threshold": proba >= f1_threshold,
        "expected_value_positive": ev > 0,
    }
    return {
        name: {
            "customers_contacted": int(mask.sum()),
            "profit": realized_profit(y_true, mask, monthly_charges, econ),
        }
        for name, mask in policies.items()
    }


def profit_curve(y_true, proba, monthly_charges, econ=ECONOMICS, points=101):
    """Profit when contacting the top-x% customers ranked by expected value."""
    ev = econ.expected_value(proba, np.asarray(monthly_charges))
    order = np.argsort(-ev)
    fractions = np.linspace(0, 1, points)
    profits = []
    for f in fractions:
        mask = np.zeros(len(order), dtype=bool)
        mask[order[: int(round(f * len(order)))]] = True
        profits.append(realized_profit(y_true, mask, monthly_charges, econ))
    return fractions, np.array(profits)
