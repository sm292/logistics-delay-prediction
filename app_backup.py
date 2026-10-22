
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

PREDICTIONS_FILE = DATA_DIR / "scored_test_orders.csv"
COMPARISON_FILE = DATA_DIR / "model_comparison.csv"
METADATA_FILE = MODEL_DIR / "metadata.json"

st.set_page_config(
    page_title="Logistics Intelligence",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Logistics Delay Intelligence")
st.markdown(
    "### Supply Chain Risk Monitoring & Machine Learning"
)
st.caption(
    "Historical test-set analysis | Predictive risk scores | "
    "Operations intelligence"
)

# --------------------------------------------------
# 1. Check required files
# --------------------------------------------------

missing_files = [
    str(path.relative_to(ROOT))
    for path in [PREDICTIONS_FILE, COMPARISON_FILE, METADATA_FILE]
    if not path.exists()
]

if missing_files:
    st.error("Some project files are missing:")
    for filename in missing_files:
        st.write(f"- {filename}")
    st.info("Run the data preparation and model training scripts first.")
    st.stop()

# --------------------------------------------------
# 2. Load project outputs
# --------------------------------------------------

@st.cache_data
def load_predictions(path):
    data = pd.read_csv(
        path,
        parse_dates=["order_purchase_timestamp"],
    )
    required = {
        "order_id",
        "order_purchase_timestamp",
        "is_late",
        "late_risk_probability",
        "predicted_late",
        "risk_band",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")

    if data.empty:
        raise ValueError("The prediction file contains no records.")

    if not data["late_risk_probability"].between(0, 1).all():
        raise ValueError("Risk probabilities must be between 0 and 1.")

    return data


@st.cache_data
def load_comparison(path):
    return pd.read_csv(path)


try:
    df = load_predictions(str(PREDICTIONS_FILE))
    comparison = load_comparison(str(COMPARISON_FILE))
except (ValueError, KeyError, OSError) as exc:
    st.error(f"Could not load project outputs: {exc}")
    st.stop()

# --------------------------------------------------
# 3. Sidebar filters
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

risk_options = st.sidebar.multiselect(
    "Risk category",
    ["Low", "Medium", "High"],
    default=["Low", "Medium", "High"],
)

date_min = df["order_purchase_timestamp"].min().date()
date_max = df["order_purchase_timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Purchase date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

threshold = st.sidebar.slider(
    "Late-risk decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help=(
        "Changes the predicted-late classification for the dashboard. "
        "This is a decision threshold, not a retrained model."
    ),
)

# --------------------------------------------------
# 4. Apply filters
# --------------------------------------------------

filtered = df[df["risk_band"].isin(risk_options)].copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        filtered["order_purchase_timestamp"].dt.date.between(
            start_date, end_date
        )
    ]

filtered["threshold_prediction"] = (
    filtered["late_risk_probability"] >= threshold
).astype(int)

# --------------------------------------------------
# 5. Key performance indicators
# --------------------------------------------------

total_orders = len(filtered)

actual_late_rate = (
    filtered["is_late"].mean() if total_orders else 0
)

predicted_late_count = int(
    filtered["threshold_prediction"].sum()
)

high_risk_count = int(
    (filtered["risk_band"] == "High").sum()
)

actual_late_count = int(filtered["is_late"].sum())

recall = (
    (
        (filtered["is_late"] == 1)
        & (filtered["threshold_prediction"] == 1)
    ).sum() / actual_late_count
    if actual_late_count else 0
)

st.subheader("Executive Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Evaluated Orders", f"{total_orders:,}")
col2.metric("Actual Late Rate", f"{actual_late_rate:.1%}")
col3.metric("High-Risk Orders", f"{high_risk_count:,}")
col4.metric("Predicted Late", f"{predicted_late_count:,}")

st.caption(
    f"Current decision threshold: {threshold:.0%} | "
    f"Recall within the filtered dataset: {recall:.1%}"
)

st.divider()

# --------------------------------------------------
# 6. Risk analysis charts
# --------------------------------------------------

st.subheader("Delivery Risk Analytics")

left, right = st.columns(2)

with left:
    st.markdown("**Orders by Risk Category**")

    if total_orders:
        risk_counts = (
            filtered["risk_band"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk Category")
            .reset_index(name="Orders")
        )

        fig = px.bar(
            risk_counts,
            x="Risk Category",
            y="Orders",
            text="Orders",
            title="Predicted Risk Distribution",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records match your filters.")

with right:
    st.markdown("**Actual vs Predicted Outcomes**")

    if total_orders:
        chart_data = filtered.copy()

        chart_data["Actual Outcome"] = chart_data["is_late"].map({
            0: "On time",
            1: "Late",
        })

        chart_data["Predicted Outcome"] = chart_data[
            "threshold_prediction"
        ].map({
            0: "Predicted on time",
            1: "Predicted late",
        })

        summary = (
            chart_data.groupby(
                ["Actual Outcome", "Predicted Outcome"],
                observed=False,
            )
            .size()
            .reset_index(name="Orders")
        )

        fig = px.bar(
            summary,
            x="Actual Outcome",
            y="Orders",
            color="Predicted Outcome",
            barmode="group",
            title="Classification Outcomes",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records match your filters.")

# --------------------------------------------------
# 7. Model comparison
# --------------------------------------------------

st.divider()
st.subheader("Machine Learning Model Comparison")

metric_columns = [
    column for column in [
        "model",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
    ]
    if column in comparison.columns
]

st.dataframe(
    comparison[metric_columns].round(4),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Model comparison uses the original 0.50 threshold. "
    "The interactive threshold changes the dashboard classifications "
    "without retraining the model."
)

# --------------------------------------------------
# 8. Order-level risk register
# --------------------------------------------------

st.divider()
st.subheader("Order-Level Risk Register")

if total_orders:
    risk_register = filtered.sort_values(
        "late_risk_probability",
        ascending=False,
    ).copy()

    risk_register["late_risk_probability"] = (
        risk_register["late_risk_probability"].map(
            lambda probability: f"{probability:.1%}"
        )
    )

    risk_register["Actual Outcome"] = risk_register["is_late"].map({
        0: "On time",
        1: "Late",
    })

    risk_register["Current Prediction"] = risk_register[
        "threshold_prediction"
    ].map({
        0: "Predicted on time",
        1: "Predicted late",
    })

    display_columns = [
        "order_id",
        "order_purchase_timestamp",
        "late_risk_probability",
        "risk_band",
        "Actual Outcome",
        "Current Prediction",
    ]

    st.dataframe(
        risk_register[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    download_data = filtered.copy()
    download_data["threshold_prediction"] = (
        download_data["late_risk_probability"] >= threshold
    ).astype(int)

    st.download_button(
        label="Download filtered risk register (CSV)",
        data=download_data.to_csv(index=False).encode("utf-8"),
        file_name="logistics_risk_register.csv",
        mime="text/csv",
    )
else:
    st.info("Adjust the sidebar filters to see orders.")

st.divider()

st.caption(
    "Portfolio prototype using historical e-commerce data. "
    "Risk scores are estimates, not guarantees. "
    "Actual outcomes are shown for retrospective evaluation."
)
