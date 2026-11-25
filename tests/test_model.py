import numpy as np
from conftest import requires_model

pytestmark = requires_model


def _predictor():
    from churn.predict import ChurnPredictor

    return ChurnPredictor()


def test_quality_gate():
    """Fail CI if a retrained model regresses below an agreed bar."""
    m = _predictor().metadata["metrics"]["catboost"]
    assert m["roc_auc"] >= 0.83
    assert m["pr_auc"] >= 0.62
    assert m["brier"] <= 0.145


def test_probabilities_in_range(customers):
    proba = _predictor().predict_proba(customers.sample(500, random_state=0))
    assert ((proba >= 0) & (proba <= 1)).all()


def test_shap_values_sum_to_model_output(customer):
    """SHAP is additive: base value + contributions = model log-odds."""
    p = _predictor()
    e = p.explain(customer)
    log_odds = e.base_value + sum(c.shap_value for c in e.contributions)
    assert np.isclose(1 / (1 + np.exp(-log_odds)), e.churn_probability, atol=1e-3)


def test_longer_contract_lowers_risk(customer):
    """Behavioural sanity check: same customer, longer commitment -> lower churn risk."""
    p = _predictor()
    probs = [
        p.predict_proba([{**customer, "Contract": c}])[0]
        for c in ["Month-to-month", "One year", "Two year"]
    ]
    assert probs[0] > probs[1] > probs[2]


def test_actions_follow_drivers(customer):
    e = _predictor().explain(customer)
    assert any("contract" in a.lower() for a in e.actions)
    loyal = {**customer, "Contract": "Two year", "tenure": 60, "TotalCharges": 5700.0}
    assert not any("contract" in a.lower() for a in _predictor().explain(loyal).actions)
