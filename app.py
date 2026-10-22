
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.score_new_orders import score_and_store


# =========================================================
# 1. CONFIGURATION AND PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

HISTORICAL_PATH = PROCESSED_DIR / "scored_test_orders.csv"
COMPARISON_PATH = PROCESSED_DIR / "model_comparison.csv"
LIVE_PATH = PROCESSED_DIR / "live_scored_orders.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Logistics Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PAGE HEADER
# =========================================================

st.title("🚚 Logistics Delay Prediction")
st.subheader("Supply Chain Intelligence Dashboard")

st.markdown(
    """
    Monitor delivery-delay risk, upload new logistics records,
    generate predictions, and track scored orders over time.
    """
)

st.caption(
    "Model predictions are estimates, not guarantees of delivery outcomes."
)


# =========================================================
# 3. DATA LOADING HELPERS
# =========================================================

@st.cache_data(ttl=30)
def load_csv(path_string, modified_time):
    """Load a CSV and cache it until its file modification time changes."""
    return pd.read_csv(path_string)


def read_optional_csv(path):
    """Read a CSV when available."""
    if not path.exists():
        return pd.DataFrame()

    return load_csv(str(path), path.stat().st_mtime)


def read_uploaded_orders(uploaded_file):
    """
    Read CSV or JSON uploads.

    Supported JSON formats:
    1. An array of order objects.
    2. An object containing a 'records' array.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".json"):
        data = json.load(uploaded_file)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict) and isinstance(
            data.get("records"), list
        ):
            return pd.DataFrame(data["records"])

        raise ValueError(
            "Unsupported JSON structure. Use an array of order "
            "objects or an object containing a 'records' array."
        )

    raise ValueError("Upload a CSV or JSON file.")


def prepare_historical_data():
    """Load the original test-set predictions."""
    df = read_optional_csv(HISTORICAL_PATH)

    if df.empty:
        return df

    df = df.copy()

    if "is_late" in df.columns:
        df["actual_is_late"] = pd.to_numeric(
            df["is_late"], errors="coerce"
        )

    if "actual_is_late" not in df.columns:
        df["actual_is_late"] = float("nan")

    if "risk_band" not in df.columns:
        if "late_risk_probability" in df.columns:
            df["risk_band"] = pd.cut(
                pd.to_numeric(
                    df["late_risk_probability"], errors="coerce"
                ),
                bins=[-0.001, 0.30, 0.60, 1.0],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            ).astype("string")
        else:
            df["risk_band"] = "Unknown"

    df["record_source"] = "Historical test set"

    return df


def prepare_live_data():
    """Load the persisted predictions from uploaded orders."""
    df = read_optional_csv(LIVE_PATH)

    if df.empty:
        return df

    df = df.copy()

    if "actual_is_late" not in df.columns:
        df["actual_is_late"] = float("nan")

    if "risk_band" not in df.columns:
        df["risk_band"] = "Unknown"

    df["record_source"] = "Uploaded orders"

    return df


def normalise_actuals(df):
    """Standardise known outcomes to numeric 0/1 values."""
    df = df.copy()

    if "actual_is_late" not in df.columns:
        df["actual_is_late"] = float("nan")

    df["actual_is_late"] = pd.to_numeric(
        df["actual_is_late"], errors="coerce"
    )

    df.loc[
        ~df["actual_is_late"].isin([0, 1]),
        "actual_is_late",
    ] = float("nan")

    return df


# =========================================================
# 4. SIDEBAR: REFRESH CONTROLS
# =========================================================

st.sidebar.title("Dashboard Controls")

refresh_seconds = st.sidebar.selectbox(
    "Automatic refresh interval",
    options=[15, 30, 60, 120],
    index=1,
    format_func=lambda seconds: f"Every {seconds} seconds",
)

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="logistics_auto_refresh",
)

st.sidebar.caption(
    "Refresh updates the dashboard from saved files. "
    "It does not retrain the model or fetch external shipment data."
)

if st.sidebar.button("Refresh now", use_container_width=True):
    st.rerun()

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Risk bands**
    
    - Low: probability below 30%
    - Medium: probability from 30% to below 60%
    - High: probability of 60% or above
    """
)


# =========================================================
# 5. UPLOAD CSV OR JSON
# =========================================================

st.header("1. Upload new logistics data")

st.write(
    "Upload prepared order features as CSV or JSON. "
    "The columns must match the features used to train your model."
)

uploaded_file = st.file_uploader(
    "Choose a CSV or JSON file",
    type=["csv", "json"],
    key="logistics_upload",
)

if uploaded_file is not None:

    try:
        preview_df = read_uploaded_orders(uploaded_file)

        st.success(
            f"File loaded: {uploaded_file.name} "
            f"({len(preview_df):,} rows)"
        )

        with st.expander("Preview uploaded data", expanded=True):
            st.dataframe(
                preview_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            f"{len(preview_df.columns)} columns detected."
        )

        if st.button(
            "Score new orders and save",
            type="primary",
            use_container_width=True,
        ):

            try:
                with st.spinner(
                    "Validating orders and calculating risk scores..."
                ):
                    uploaded_file.seek(0)
                    orders_df = read_uploaded_orders(uploaded_file)

                    new_records, duplicate_count = score_and_store(
                        orders_df
                    )

                # Ensure subsequent reads use the newest saved files.
                st.cache_data.clear()

                if new_records.empty:
                    st.info(
                        "No new orders were added. "
                        "The uploaded order IDs may already exist."
                    )
                else:
                    st.success(
                        f"{len(new_records):,} new orders "
                        "were scored and saved successfully."
                    )

                    result_columns = [
                        "order_id",
                        "late_risk_probability",
                        "predicted_late",
                        "risk_band",
                    ]

                    st.dataframe(
                        new_records[result_columns],
                        use_container_width=True,
                        hide_index=True,
                    )

                if duplicate_count > 0:
                    st.warning(
                        f"{duplicate_count:,} duplicate order(s) "
                        "were skipped."
                    )

                # Rerun to update all dashboard charts and KPIs.
                st.rerun()

            except Exception as exc:
                st.error(f"Could not score the uploaded data: {exc}")

    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")


st.divider()


# =========================================================
# 6. LOAD HISTORICAL AND LIVE RECORDS
# =========================================================

st.header("2. Logistics monitoring")

historical_df = normalise_actuals(prepare_historical_data())
live_df = normalise_actuals(prepare_live_data())

available_frames = [
    frame
    for frame in [historical_df, live_df]
    if not frame.empty
]

if not available_frames:
    st.warning(
        "No scored records were found. Check that the original "
        "test predictions exist or upload a new CSV/JSON file."
    )
    st.stop()

all_orders = pd.concat(
    available_frames,
    ignore_index=True,
    sort=False,
)

required_output_columns = [
    "order_id",
    "late_risk_probability",
    "risk_band",
    "record_source",
]

missing_output_columns = [
    col for col in required_output_columns
    if col not in all_orders.columns
]

if missing_output_columns:
    st.error(
        "Required output columns are missing: "
        + ", ".join(missing_output_columns)
    )
    st.stop()

all_orders["order_id"] = all_orders["order_id"].astype("string")

all_orders["late_risk_probability"] = pd.to_numeric(
    all_orders["late_risk_probability"], errors="coerce"
)

all_orders = all_orders.dropna(
    subset=["late_risk_probability"]
).copy()

all_orders["risk_band"] = (
    all_orders["risk_band"].fillna("Unknown").astype(str)
)

all_orders = all_orders[
    all_orders["late_risk_probability"].between(0, 1)
].copy()


# =========================================================
# 7. FILTERS
# =========================================================

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    available_sources = sorted(
        all_orders["record_source"].dropna().unique().tolist()
    )

    selected_sources = st.multiselect(
        "Data source",
        options=available_sources,
        default=available_sources,
    )

with filter_col2:
    available_risks = [
        risk for risk in ["Low", "Medium", "High", "Unknown"]
        if risk in all_orders["risk_band"].unique()
    ]

    selected_risks = st.multiselect(
        "Risk level",
        options=available_risks,
        default=available_risks,
    )

with filter_col3:
    threshold = st.slider(
        "Predicted-late threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

filtered = all_orders[
    all_orders["record_source"].isin(selected_sources)
    & all_orders["risk_band"].isin(selected_risks)
].copy()

filtered["dashboard_predicted_late"] = (
    filtered["late_risk_probability"] >= threshold
).astype(int)


# =========================================================
# 8. KEY PERFORMANCE INDICATORS
# =========================================================

total_orders = len(filtered)

high_risk_count = int(
    (filtered["risk_band"] == "High").sum()
)

predicted_late_count = int(
    filtered["dashboard_predicted_late"].sum()
)

known_outcomes = filtered["actual_is_late"].dropna()

if len(known_outcomes) > 0:
    actual_late_rate = known_outcomes.mean() * 100
    actual_late_display = f"{actual_late_rate:.1f}%"
else:
    actual_late_display = "N/A"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Orders displayed", f"{total_orders:,}")
kpi2.metric("High-risk orders", f"{high_risk_count:,}")
kpi3.metric("Predicted late", f"{predicted_late_count:,}")
kpi4.metric("Actual late rate", actual_late_display)

if known_outcomes.empty:
    st.info(
        "Actual delivery outcomes are unavailable for the selected "
        "records. Predicted delays must not be treated as confirmed delays."
    )
else:
    st.caption(
        f"Actual late rate is based on {len(known_outcomes):,} "
        "records with known outcomes."
    )


# =========================================================
# 9. VISUAL ANALYTICS
# =========================================================

st.subheader("Risk and order analysis")

if filtered.empty:

    st.info("No records match the selected filters.")

else:

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        risk_counts = (
            filtered["risk_band"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High", "Unknown"],
                fill_value=0,
            )
            .rename_axis("Risk level")
            .reset_index(name="Orders")
        )

        risk_fig = px.bar(
            risk_counts,
            x="Risk level",
            y="Orders",
            title="Orders by risk level",
        )

        st.plotly_chart(
            risk_fig,
            use_container_width=True,
        )

    with chart_col2:
        source_counts = (
            filtered["record_source"]
            .value_counts()
            .rename_axis("Source")
            .reset_index(name="Orders")
        )

        source_fig = px.bar(
            source_counts,
            x="Source",
            y="Orders",
            title="Historical versus uploaded orders",
        )

        st.plotly_chart(
            source_fig,
            use_container_width=True,
        )

    # Actual versus predicted comparison only where outcomes are known.
    known = filtered.dropna(
        subset=["actual_is_late"]
    ).copy()

    if not known.empty:

        known["Actual outcome"] = known[
            "actual_is_late"
        ].map({
            0.0: "On time",
            1.0: "Late",
        })

        known["Predicted outcome"] = known[
            "dashboard_predicted_late"
        ].map({
            0: "Predicted on time",
            1: "Predicted late",
        })

        outcome_counts = (
            known.groupby(
                ["Actual outcome", "Predicted outcome"],
                observed=True,
            )
            .size()
            .reset_index(name="Orders")
        )

        outcome_fig = px.bar(
            outcome_counts,
            x="Actual outcome",
            y="Orders",
            color="Predicted outcome",
            barmode="group",
            title="Predictions versus known delivery outcomes",
        )

        st.plotly_chart(
            outcome_fig,
            use_container_width=True,
        )


# =========================================================
# 10. ORIGINAL MODEL EVALUATION
# =========================================================

st.header("3. Model performance")

comparison_df = read_optional_csv(COMPARISON_PATH)

if not comparison_df.empty:
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("The original model comparison file was not found.")

if METADATA_PATH.exists():
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        with st.expander("Model metadata"):
            st.json(metadata)

    except (json.JSONDecodeError, OSError):
        st.caption("Model metadata could not be read.")


# =========================================================
# 11. ORDER RISK REGISTER
# =========================================================

st.header("4. Order risk register")

display_columns = [
    column
    for column in [
        "order_id",
        "record_source",
        "customer_state",
        "seller_state",
        "product_category_name",
        "estimated_delivery_days",
        "late_risk_probability",
        "predicted_late",
        "dashboard_predicted_late",
        "risk_band",
        "actual_is_late",
        "scored_at_utc",
    ]
    if column in filtered.columns
]

if filtered.empty:

    st.info("No orders are available to display.")

else:

    register = filtered[display_columns].sort_values(
        "late_risk_probability",
        ascending=False,
    )

    st.dataframe(
        register,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download filtered order register",
        data=register.to_csv(index=False),
        file_name="logistics_risk_register.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# 12. DATA STATUS
# =========================================================

st.header("5. Data status")

status_col1, status_col2 = st.columns(2)

with status_col1:
    st.metric(
        "Historical test records",
        f"{len(historical_df):,}",
    )

with status_col2:
    st.metric(
        "Persisted uploaded records",
        f"{len(live_df):,}",
    )

if LIVE_PATH.exists():
    st.caption(
        f"Live scored records last modified: "
        f"{pd.Timestamp(LIVE_PATH.stat().st_mtime, unit='s')}"
    )
else:
    st.caption(
        "No live scored file exists yet. It will be created "
        "after the first successful upload."
    )


# =========================================================
# 13. DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "This project uses a saved machine-learning model to estimate "
    "delivery-delay risk. Auto-refresh reloads dashboard data from "
    "local files; it does not connect to a carrier API, automatically "
    "ingest external events, retrain the model, or guarantee that "
    "a predicted delay will occur."
)
