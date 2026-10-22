
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "delay_model.joblib"
LIVE_DATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "live_scored_orders.csv"
)

# These must match the features used during model training.
FEATURE_COLUMNS = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "is_weekend",
    "estimated_delivery_days",
    "item_count",
    "total_price",
    "total_freight",
    "seller_count",
    "same_state",
    "customer_state",
    "seller_state",
    "product_category_name",
]

REQUIRED_COLUMNS = ["order_id"] + FEATURE_COLUMNS


def validate_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardise an uploaded feature dataset."""

    if df.empty:
        raise ValueError("The uploaded CSV contains no records.")

    df = df.copy()
    df.columns = df.columns.str.strip()

    if df.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df["order_id"] = df["order_id"].astype("string").str.strip()

    if df["order_id"].isna().any() or df["order_id"].eq("").any():
        raise ValueError("Every record must have a valid order_id.")

    # Keep the first copy if an order is repeated in the same upload.
    df = df.drop_duplicates(subset=["order_id"], keep="first")

    # Preserve an optional actual outcome for later model evaluation.
    if "is_late" in df.columns:
        df["actual_is_late"] = pd.to_numeric(
            df["is_late"], errors="coerce"
        )
        df["actual_is_late"] = df["actual_is_late"].where(
            df["actual_is_late"].isin([0, 1])
        )
        df = df.drop(columns=["is_late"])
    elif "actual_is_late" not in df.columns:
        df["actual_is_late"] = pd.NA

    # Check numeric feature columns before sending them to the model.
    numeric_columns = [
        "purchase_month",
        "purchase_weekday",
        "purchase_hour",
        "is_weekend",
        "estimated_delivery_days",
        "item_count",
        "total_price",
        "total_freight",
        "seller_count",
        "same_state",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # The model pipeline handles missing feature values using its
    # fitted imputers. Completely invalid numeric columns are rejected.
    for column in numeric_columns:
        if df[column].notna().sum() == 0:
            raise ValueError(
                f"Column '{column}' has no usable numeric values."
            )

    return df


def score_and_store(
    uploaded_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Score new orders and append only previously unseen order IDs.

    Returns:
        newly_scored: records added during this upload
        duplicate_count: records skipped because their IDs already exist
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    df = validate_uploaded_data(uploaded_df)

    # Load the existing trained pipeline; do not retrain it here.
    model = joblib.load(MODEL_PATH)

    # Create the persistent data folder if it does not exist.
    LIVE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if LIVE_DATA_PATH.exists():
        existing = pd.read_csv(
            LIVE_DATA_PATH, dtype={"order_id": "string"}
        )
        existing_ids = set(
            existing["order_id"].dropna().astype(str)
        )
    else:
        existing = pd.DataFrame()
        existing_ids = set()

    df["order_id"] = df["order_id"].astype(str)

    already_seen = df["order_id"].isin(existing_ids)
    duplicate_count = int(already_seen.sum())
    new_df = df.loc[~already_seen].copy()

    if new_df.empty:
        return pd.DataFrame(), duplicate_count

    # The model receives only the exact features used in training.
    X = new_df[FEATURE_COLUMNS].copy()

    probabilities = model.predict_proba(X)[:, 1]

    new_df["late_risk_probability"] = probabilities
    new_df["predicted_late"] = (
        new_df["late_risk_probability"] >= 0.50
    ).astype(int)

    new_df["risk_band"] = pd.cut(
        new_df["late_risk_probability"],
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype("string")

    new_df["scored_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    # Append to the persistent live dataset.
    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat(
            [existing, new_df],
            ignore_index=True,
            sort=False,
        )

    combined.to_csv(LIVE_DATA_PATH, index=False)

    return new_df, duplicate_count
