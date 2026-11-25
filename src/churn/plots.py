"""Static report figures written during training."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from churn.evaluate import profit_curve  # noqa: E402

BLUE, GREY, ORANGE = "#2563eb", "#9ca3af", "#ea580c"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def roc_pr(y, proba_model, proba_baseline, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for proba, label, color in [
        (proba_model, "CatBoost", BLUE),
        (proba_baseline, "Logistic regression", GREY),
    ]:
        fpr, tpr, _ = roc_curve(y, proba)
        axes[0].plot(fpr, tpr, label=label, color=color, lw=2)
        prec, rec, _ = precision_recall_curve(y, proba)
        axes[1].plot(rec, prec, label=label, color=color, lw=2)
    axes[0].plot([0, 1], [0, 1], ls="--", color="#d1d5db")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].axhline(np.mean(y), ls="--", color="#d1d5db", label="Base rate")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    for ax in axes:
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def calibration(y, proba, path):
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#d1d5db", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", color=BLUE, lw=2, label="CatBoost")
    ax.set(
        xlabel="Predicted churn probability",
        ylabel="Observed churn rate",
        title="Calibration (test set)",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def gains(y, proba, path):
    y = np.asarray(y)
    order = np.argsort(-proba)
    cum = np.cumsum(y[order]) / y.sum()
    x = np.arange(1, len(y) + 1) / len(y)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(x, cum, color=BLUE, lw=2, label="CatBoost")
    ax.plot([0, 1], [0, 1], ls="--", color="#d1d5db", label="Random targeting")
    ax.set(
        xlabel="Share of customers contacted (highest risk first)",
        ylabel="Share of churners reached",
        title="Cumulative gains",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def profit(y, proba, monthly_charges, path):
    fractions, profits = profit_curve(y, proba, monthly_charges)
    best = int(np.argmax(profits))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(fractions * 100, profits, color=BLUE, lw=2)
    ax.axhline(0, color="#d1d5db")
    ax.scatter([fractions[best] * 100], [profits[best]], color=ORANGE, zorder=3)
    ax.annotate(
        f"Max ${profits[best]:,.0f}\ncontacting top {fractions[best]:.0%}",
        (fractions[best] * 100, profits[best]),
        xytext=(12, -30),
        textcoords="offset points",
    )
    ax.set(
        xlabel="% of customers contacted (ranked by expected value)",
        ylabel="Campaign profit ($)",
        title="Retention campaign profit curve (test set)",
    )
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def shap_summary(shap_values, X, path_beeswarm, path_bar):
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    _save(plt.gcf(), path_beeswarm)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=15, color=BLUE)
    _save(plt.gcf(), path_bar)
