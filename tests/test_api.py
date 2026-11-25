import pytest
from conftest import requires_model
from fastapi.testclient import TestClient

pytestmark = requires_model


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["metrics"]["catboost"]["roc_auc"] > 0.8
    assert "Contract" in body["global_importance"]


def test_predict(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_tier"] in {"low", "medium", "high"}


def test_total_charges_optional(client, customer):
    customer.pop("TotalCharges")
    assert client.post("/predict", json=customer).status_code == 200


def test_explain(client, customer):
    body = client.post("/explain?top_k=5", json=customer).json()
    assert len(body["contributions"]) == 5
    assert body["actions"]


def test_batch(client, customer):
    r = client.post("/predict/batch", json={"customers": [customer] * 3})
    assert len(r.json()["predictions"]) == 3


@pytest.mark.parametrize(
    "patch",
    [
        {"Contract": "Weekly"},
        {"tenure": -1},
        {"PhoneService": "No", "MultipleLines": "Yes"},
        {"InternetService": "No"},  # add-ons still say Yes/No
    ],
)
def test_rejects_invalid_input(client, customer, patch):
    assert client.post("/predict", json={**customer, **patch}).status_code == 422
