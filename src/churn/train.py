"""Train, evaluate and persist the churn model.

Usage: python -m churn.train
"""

import hashlib
import json
from datetime import UTC, datetime
from itertools import product

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import __version__, plots
from churn.config import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_INPUTS,
    DATA_PATH,
    ECONOMICS,
    FEATURES,
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    NUMERIC_INPUTS,
    RANDOM_STATE,
)
from churn.data import load_dataset
from churn.evaluate import best_f1_threshold, campaign_comparison, classification_metrics

PARAM_GRID = {"depth": [4, 6], "l2_leaf_reg": [3, 10], "learning_rate": [0.03]}


def split(X, y):
    """60/20/20 stratified train / validation / test split."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_catboost(iterations=3000, early_stopping_rounds=200, **params) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        eval_metric="AUC",
        loss_function="Logloss",
        cat_features=CATEGORICAL_FEATURES,
        random_seed=RANDOM_STATE,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
        allow_writing_files=False,
        **params,
    )


def tune(X_train, y_train, X_val, y_val):
    """Small grid search on validation log loss (we care about calibrated probabilities)."""
    best = None
    val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL_FEATURES)
    for values in product(*PARAM_GRID.values()):
        params = dict(zip(PARAM_GRID, values, strict=True))
        model = make_catboost(**params)
        model.fit(X_train, y_train, eval_set=val_pool, use_best_model=True)
        proba = model.predict_proba(X_val)[:, 1]
        score = classification_metrics(y_val, proba)["log_loss"]
        print(f"  {params} -> val log loss {score:.4f}, iters {model.get_best_iteration()}")
        if best is None or score < best[0]:
            best = (score, params, model)
    return best[1], best[2]


def baseline(X_train, y_train):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return make_pipeline(pre, LogisticRegression(max_iter=2000)).fit(X_train, y_train)


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _round(d):
    if isinstance(d, dict):
        return {k: _round(v) for k, v in d.items()}
    if isinstance(d, float | np.floating):
        return round(float(d), 4)
    return d


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    print(
        f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)} | churn rate {y.mean():.1%}"
    )

    print("Tuning CatBoost...")
    params, model = tune(X_train, y_train, X_val, y_val)
    n_iter = model.get_best_iteration() + 1

    # Robustness check: 5-fold CV ROC-AUC on train+val with the chosen settings.
    X_dev, y_dev = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = np.array(
        [
            roc_auc_score(
                y_dev.iloc[te],
                make_catboost(iterations=n_iter, early_stopping_rounds=None, **params)
                .fit(X_dev.iloc[tr], y_dev.iloc[tr])
                .predict_proba(X_dev.iloc[te])[:, 1],
            )
            for tr, te in cv.split(X_dev, y_dev)
        ]
    )
    print(f"5-fold CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_proba)

    test_proba = model.predict_proba(X_test)[:, 1]
    lr = baseline(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_threshold = best_f1_threshold(y_val, lr.predict_proba(X_val)[:, 1])

    metrics = classification_metrics(y_test, test_proba, threshold)
    baseline_metrics = classification_metrics(y_test, lr_proba, lr_threshold)
    campaign = campaign_comparison(y_test, test_proba, X_test["MonthlyCharges"], threshold)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Pool(X_test, cat_features=CATEGORICAL_FEATURES))
    importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
        .sort_values(ascending=False)
        .round(4)
    )

    plots.roc_pr(y_test, test_proba, lr_proba, FIGURES_DIR / "roc_pr.png")
    plots.calibration(y_test, test_proba, FIGURES_DIR / "calibration.png")
    plots.gains(y_test, test_proba, FIGURES_DIR / "gains.png")
    plots.profit(y_test, test_proba, X_test["MonthlyCharges"], FIGURES_DIR / "profit_curve.png")
    plots.shap_summary(
        shap_values, X_test, FIGURES_DIR / "shap_beeswarm.png", FIGURES_DIR / "shap_bar.png"
    )

    model.save_model(MODEL_PATH)
    metadata = {
        "model_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_sha256": file_sha256(DATA_PATH),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "base_churn_rate": float(y.mean()),
        "params": {**params, "iterations": n_iter},
        "threshold": threshold,
        "economics": ECONOMICS.__dict__,
        "metrics": {"catboost": metrics, "logistic_regression": baseline_metrics},
        "cv_roc_auc": {"mean": cv_auc.mean(), "std": cv_auc.std()},
        "campaign_test_set": campaign,
        "global_importance": importance.to_dict(),
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        # Used by the UI and API to offer valid choices / sensible ranges.
        "category_levels": {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL_INPUTS},
        "numeric_ranges": {
            c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
            for c in NUMERIC_INPUTS
        },
        "expected_value": float(np.mean(explainer.expected_value)),
    }
    METADATA_PATH.write_text(json.dumps(_round(metadata), indent=2))

    print("\nTest-set results")
    print(pd.DataFrame({"CatBoost": metrics, "LogReg": baseline_metrics}).round(4).to_string())
    print("\nCampaign profit on test set")
    print(pd.DataFrame(campaign).T.to_string())
    print(f"\nSaved model to {MODEL_PATH.relative_to(MODEL_DIR.parent)}")


if __name__ == "__main__":
    main()
