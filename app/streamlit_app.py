"""Interactive churn explorer.

Run: streamlit run app/streamlit_app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn.config import DATA_PATH, FIGURES_DIR, ID_COL, TARGET
from churn.predict import ChurnPredictor

st.set_page_config(page_title="Churn Explorer", page_icon="📉", layout="wide")

TIER_COLOR = {"low": "#16a34a", "medium": "#d97706", "high": "#dc2626"}
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
PRETTY = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "num_services": "Number of services",
    "avg_monthly_spend": "Avg historical bill",
    "charge_drift": "Bill change vs. average",
    "tenure_band": "Tenure band",
}


@st.cache_resource
def load_predictor():
    p = ChurnPredictor()
    _ = p.explainer  # warm up the SHAP explainer
    return p


@st.cache_data
def load_customers():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=[TARGET]), df[TARGET]


predictor = load_predictor()
meta = predictor.metadata
levels = meta["category_levels"]
customers, labels = load_customers()


# --------------------------------------------------------------------------- input form
def set_customer(record: dict):
    for k, v in record.items():
        if k in levels or k in ("tenure", "MonthlyCharges"):
            st.session_state[k] = (
                int(v) if k == "tenure" else (float(v) if k == "MonthlyCharges" else str(v))
            )
    st.session_state["SeniorCitizen"] = {"1": "Yes", "0": "No"}.get(
        str(record["SeniorCitizen"]), str(record["SeniorCitizen"])
    )


def dependent_select(label, key, options):
    """Selectbox whose options depend on another field; reset stale values before rendering."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[-1]
    st.selectbox(label, options, key=key)


if "Contract" not in st.session_state:
    set_customer(customers.iloc[0].to_dict())

with st.sidebar:
    st.header("Customer profile")
    c1, c2 = st.columns(2)
    if c1.button("🎲 Random", use_container_width=True):
        set_customer(customers.sample(1).iloc[0].to_dict())
    if c2.button("🔥 High-risk", use_container_width=True):
        high = customers[labels == "Yes"]
        set_customer(high.sample(1).iloc[0].to_dict())

    with st.expander("Account", expanded=True):
        st.selectbox("Contract", levels["Contract"], key="Contract")
        st.slider("Tenure (months)", 0, 72, key="tenure")
        st.number_input("Monthly charges ($)", 18.0, 120.0, step=0.5, key="MonthlyCharges")
        st.selectbox("Payment method", levels["PaymentMethod"], key="PaymentMethod")
        st.selectbox("Paperless billing", levels["PaperlessBilling"], key="PaperlessBilling")
    with st.expander("Services"):
        st.selectbox("Phone service", ["Yes", "No"], key="PhoneService")
        phone = st.session_state["PhoneService"] == "Yes"
        dependent_select(
            "Multiple lines", "MultipleLines", ["Yes", "No"] if phone else ["No phone service"]
        )
        st.selectbox("Internet service", levels["InternetService"], key="InternetService")
        internet = st.session_state["InternetService"] != "No"
        for addon in INTERNET_ADDONS:
            dependent_select(addon, addon, ["Yes", "No"] if internet else ["No internet service"])
    with st.expander("Demographics"):
        for col in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
            st.selectbox(col, levels[col], key=col)


def current_customer() -> dict:
    keys = list(levels) + ["tenure", "MonthlyCharges"]
    rec = {k: st.session_state[k] for k in keys}
    rec["TotalCharges"] = rec["tenure"] * rec["MonthlyCharges"]
    return rec


# --------------------------------------------------------------------------- helpers
def gauge(p: float, tier: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": TIER_COLOR[tier]},
                "steps": [
                    {"range": [0, 30], "color": "#f0fdf4"},
                    {"range": [30, 60], "color": "#fffbeb"},
                    {"range": [60, 100], "color": "#fef2f2"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(t=20, b=0, l=20, r=20))
    return fig


def waterfall(pred, k=10):
    """SHAP contributions in log-odds, largest first, remainder folded into 'other'."""
    top, rest = pred.contributions[:k], pred.contributions[k:]

    def label(c):
        value = f"{c.value:.1f}" if isinstance(c.value, float) else c.value
        return f"{PRETTY.get(c.feature, c.feature)} = {value}"

    labels_ = [label(c) for c in top]
    values = [c.shap_value for c in top]
    if rest:
        labels_.append(f"{len(rest)} other features")
        values.append(sum(c.shap_value for c in rest))
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=["Average customer"] + labels_[::-1] + ["This customer"],
            x=[pred.base_value] + values[::-1] + [0],
            measure=["absolute"] + ["relative"] * len(values) + ["total"],
            increasing={"marker": {"color": "#dc2626"}},
            decreasing={"marker": {"color": "#2563eb"}},
            totals={"marker": {"color": "#6b7280"}},
            connector={"line": {"color": "#e5e7eb"}},
        )
    )
    fig.update_layout(
        height=110 + 34 * len(values),
        margin=dict(t=10, b=30, l=10, r=10),
        xaxis_title="Churn log-odds  (red pushes risk up, blue pushes it down)",
        showlegend=False,
    )
    return fig


def what_if_scenarios(rec: dict) -> list[tuple[str, dict]]:
    """Plausible retention levers for this customer."""
    out = []
    if rec["Contract"] == "Month-to-month":
        out.append(("Upgrade to 1-year contract", {"Contract": "One year"}))
        out.append(("Upgrade to 2-year contract", {"Contract": "Two year"}))
    if rec["InternetService"] != "No":
        for addon in ["OnlineSecurity", "TechSupport"]:
            if rec[addon] == "No":
                out.append((f"Add {addon}", {addon: "Yes"}))
    if rec["PaymentMethod"] == "Electronic check":
        out.append(("Switch to auto-pay (card)", {"PaymentMethod": "Credit card (automatic)"}))
    if rec["MonthlyCharges"] > 30:
        out.append(
            ("10% price discount", {"MonthlyCharges": round(rec["MonthlyCharges"] * 0.9, 2)})
        )
    return out


# --------------------------------------------------------------------------- layout
st.title("📉 Customer Churn Explorer")
st.caption(
    f"CatBoost model v{meta['model_version']} · test ROC-AUC "
    f"{meta['metrics']['catboost']['roc_auc']:.3f} · explanations via SHAP"
)

tab_single, tab_batch, tab_model = st.tabs(
    ["🔍 Score a customer", "📋 Batch scoring", "📊 Model insights"]
)

with tab_single:
    rec = current_customer()
    pred = predictor.explain(rec)

    left, right = st.columns([1, 1.6])
    with left:
        st.plotly_chart(gauge(pred.churn_probability, pred.risk_tier), use_container_width=True)
        m1, m2 = st.columns(2)
        m1.metric("Risk tier", pred.risk_tier.title())
        m2.metric(
            "Value of a retention offer",
            f"${pred.expected_value_of_contact:,.0f}",
            help=(
                f"Expected profit of a ${predictor.economics.offer_cost:.0f} offer that saves "
                f"{predictor.economics.save_rate:.0%} of would-be churners, valuing a saved "
                f"customer at {predictor.economics.value_horizon_months} months of revenue."
            ),
        )
        if pred.recommend_contact:
            st.success("**Recommend contacting** this customer: the offer pays for itself.")
        else:
            st.info("**Don't contact**: expected savings don't cover the offer cost.")
        if pred.actions:
            st.subheader("Suggested actions")
            for a in pred.actions:
                st.markdown(f"- {a}")

    with right:
        st.subheader("Why this prediction?")
        st.plotly_chart(waterfall(pred), use_container_width=True)

    st.subheader("What-if: which lever moves the needle?")
    scenarios = what_if_scenarios(rec)
    if scenarios:
        variants = [{**rec, **change} for _, change in scenarios]
        for v in variants:
            v["TotalCharges"] = v["tenure"] * v["MonthlyCharges"]
        new = predictor.predict_proba(variants)
        df = pd.DataFrame(
            {
                "Scenario": [s for s, _ in scenarios],
                "New churn probability": new,
                "Change": new - pred.churn_probability,
            }
        ).sort_values("Change")
        st.dataframe(
            df.style.format(
                {"New churn probability": "{:.1%}", "Change": "{:+.1%}"}
            ).background_gradient(subset=["Change"], cmap="RdYlGn_r"),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Model-based what-ifs show association, not guaranteed causal effect. "
            "Validate levers with an A/B test before rolling out."
        )
    else:
        st.write("This customer is already on the most retention-friendly options.")

with tab_batch:
    st.write(
        "Upload a CSV with the same columns as the Telco dataset (extra columns are kept) "
        "or score a sample of existing customers."
    )
    up = st.file_uploader("Customer CSV", type="csv")
    n = st.slider("…or sample size from the dataset", 50, len(customers), 500, step=50)
    df = pd.read_csv(up) if up else customers.sample(n, random_state=7)
    try:
        scored = pd.concat(
            [df.reset_index(drop=True), predictor.score(df.reset_index(drop=True))], axis=1
        ).sort_values("expected_value_of_contact", ascending=False)
    except KeyError as e:
        st.error(f"Missing required column: {e}")
        st.stop()

    contact = scored[scored["recommend_contact"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{len(scored):,}")
    c2.metric("High risk", f"{(scored['risk_tier'] == 'high').mean():.0%}")
    c3.metric("Recommended contacts", f"{len(contact):,}")
    c4.metric("Expected campaign profit", f"${contact['expected_value_of_contact'].sum():,.0f}")

    cols = [ID_COL] if ID_COL in scored else []
    cols += [
        "churn_probability",
        "risk_tier",
        "expected_value_of_contact",
        "recommend_contact",
        "Contract",
        "tenure",
        "MonthlyCharges",
    ]
    st.dataframe(
        scored[cols].style.format(
            {"churn_probability": "{:.1%}", "expected_value_of_contact": "${:,.0f}"}
        ),
        hide_index=True,
        use_container_width=True,
        height=420,
    )
    st.download_button(
        "⬇️ Download scored CSV",
        scored.to_csv(index=False),
        "churn_scores.csv",
        "text/csv",
    )

with tab_model:
    cb, lr = meta["metrics"]["catboost"], meta["metrics"]["logistic_regression"]
    m = st.columns(5)
    m[0].metric("ROC-AUC", f"{cb['roc_auc']:.3f}", f"{cb['roc_auc'] - lr['roc_auc']:+.3f} vs LR")
    m[1].metric("PR-AUC", f"{cb['pr_auc']:.3f}", f"{cb['pr_auc'] - lr['pr_auc']:+.3f} vs LR")
    m[2].metric(
        "Brier score",
        f"{cb['brier']:.3f}",
        f"{cb['brier'] - lr['brier']:+.3f}",
        delta_color="inverse",
    )
    m[3].metric("Top-decile lift", f"{cb['lift_top_decile']:.2f}×")
    m[4].metric("Churners in top 20%", f"{cb['recall_top_20pct']:.0%}")

    st.subheader("Business impact (held-out test set)")
    camp = pd.DataFrame(meta["campaign_test_set"]).T
    camp.index = camp.index.map(
        {
            "contact_nobody": "Contact nobody",
            "contact_everyone": "Contact everyone",
            "f1_threshold": f"Classifier threshold ({meta['threshold']:.2f})",
            "expected_value_positive": "Expected-value targeting (ours)",
        }
    )
    a, b = st.columns([1, 1.3])
    a.dataframe(
        camp.style.format({"customers_contacted": "{:,.0f}", "profit": "${:,.0f}"}),
        use_container_width=True,
    )
    b.image(str(FIGURES_DIR / "profit_curve.png"))

    st.subheader("What drives churn overall?")
    a, b = st.columns(2)
    a.image(str(FIGURES_DIR / "shap_beeswarm.png"), caption="SHAP summary (test set)")
    b.image(str(FIGURES_DIR / "shap_bar.png"), caption="Mean |SHAP| feature importance")

    st.subheader("Discrimination & calibration")
    a, b, c = st.columns([2, 1, 1])
    a.image(str(FIGURES_DIR / "roc_pr.png"))
    b.image(str(FIGURES_DIR / "calibration.png"))
    c.image(str(FIGURES_DIR / "gains.png"))
