"""FastAPI service exposing the churn model.

Run: uvicorn api.main:app --reload   (docs at http://localhost:8000/docs)
"""

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    BatchIn,
    BatchOut,
    Customer,
    ExplanationOut,
    ModelInfo,
    PredictionOut,
)
from churn import __version__
from churn.predict import ChurnPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = ChurnPredictor()
    _ = app.state.predictor.explainer  # build SHAP explainer at startup
    yield


app = FastAPI(
    title="Customer Churn API",
    version=__version__,
    description="CatBoost churn probabilities, SHAP explanations and retention recommendations.",
    lifespan=lifespan,
)


def _predictor(request: Request) -> ChurnPredictor:
    return request.app.state.predictor


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "predictor")}


@app.get("/model/info", response_model=ModelInfo)
def model_info(request: Request):
    return _predictor(request).metadata


@app.post("/predict", response_model=PredictionOut)
def predict(customer: Customer, request: Request):
    return _predictor(request).score([customer.model_dump()]).iloc[0].to_dict()


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchIn, request: Request):
    scored = _predictor(request).score([c.model_dump() for c in batch.customers])
    return {"predictions": scored.to_dict(orient="records")}


@app.post("/explain", response_model=ExplanationOut)
def explain(customer: Customer, request: Request, top_k: int = 10):
    if not 1 <= top_k <= 50:
        raise HTTPException(422, "top_k must be between 1 and 50")
    return asdict(_predictor(request).explain(customer.model_dump(), top_k=top_k))
