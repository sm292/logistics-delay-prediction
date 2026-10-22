
from pathlib import Path

import numpy as np
import pandas as pd

# Project root: the folder containing data/, src/, models/, etc.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)


def load_csv(filename):
    """Load one CSV file and provide a useful error if it is missing."""
    path = RAW / filename

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file: {path}\n"
            "Check that the CSV files are directly inside data/raw/."
        )

    df = pd.read_csv(path)
    print(f"Loaded {filename}: {len(df):,} rows")
    return df


def prepare_data():
    print("\n--- STEP 1: Loading data ---")

    orders = load_csv("olist_orders_dataset.csv")
    customers = load_csv("olist_customers_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")
    sellers = load_csv("olist_sellers_dataset.csv")
    products = load_csv("olist_products_dataset.csv")

    print("\n--- STEP 2: Cleaning order dates ---")

    date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_columns:
        orders[col] = pd.to_datetime(
            orders[col], errors="coerce"
        )

    # Only completed orders with usable actual and estimated dates.
    valid_orders = orders[
        (orders["order_status"] == "delivered")
        & orders["order_purchase_timestamp"].notna()
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    # Keep one record per order.
    valid_orders = valid_orders.drop_duplicates(
        subset="order_id", keep="first"
    )

    print(f"Eligible delivered orders: {len(valid_orders):,}")

    print("\n--- STEP 3: Creating target and features ---")

    # Late means delivered on a calendar date after the estimated date.
    valid_orders["is_late"] = (
        valid_orders["order_delivered_customer_date"].dt.normalize()
        > valid_orders["order_estimated_delivery_date"].dt.normalize()
    ).astype("int8")

    purchase = valid_orders["order_purchase_timestamp"]

    valid_orders["purchase_month"] = purchase.dt.month
    valid_orders["purchase_weekday"] = purchase.dt.dayofweek
    valid_orders["purchase_hour"] = purchase.dt.hour
    valid_orders["is_weekend"] = (
        purchase.dt.dayofweek >= 5
    ).astype("int8")

    valid_orders["estimated_delivery_days"] = (
        valid_orders["order_estimated_delivery_date"] - purchase
    ).dt.total_seconds() / 86400

    print("\n--- STEP 4: Aggregating order items ---")

    items["price"] = pd.to_numeric(
        items["price"], errors="coerce"
    )
    items["freight_value"] = pd.to_numeric(
        items["freight_value"], errors="coerce"
    )

    # Deterministically select one primary seller/product for
    # categorical features while aggregating all items.
    items = items.sort_values(
        ["order_id", "order_item_id", "seller_id", "product_id"]
    )

    item_features = items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        seller_count=("seller_id", "nunique"),
        primary_seller_id=("seller_id", "first"),
        primary_product_id=("product_id", "first"),
    ).reset_index()

    seller_features = (
        sellers[["seller_id", "seller_state"]]
        .drop_duplicates("seller_id")
    )

    product_features = (
        products[["product_id", "product_category_name"]]
        .drop_duplicates("product_id")
    )

    item_features = item_features.merge(
        seller_features,
        left_on="primary_seller_id",
        right_on="seller_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["seller_id"])

    item_features = item_features.merge(
        product_features,
        left_on="primary_product_id",
        right_on="product_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["product_id"])

    customer_features = (
        customers[["customer_id", "customer_state"]]
        .drop_duplicates("customer_id")
    )

    print("\n--- STEP 5: Joining datasets ---")

    df = valid_orders.merge(
        customer_features,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    df["same_state"] = (
        df["customer_state"].notna()
        & df["seller_state"].notna()
        & (df["customer_state"] == df["seller_state"])
    ).astype("int8")

    # Only features intended to be available at order placement.
    feature_columns = [
        "purchase_month",
        "purchase_weekday",
        "purchase_hour",
        "is_weekend",
        "estimated_delivery_days",
        "item_count",
        "total_price",
        "total_freight",
        "seller_count",
        "customer_state",
        "seller_state",
        "product_category_name",
        "same_state",
    ]

    result = df[
        [
            "order_id",
            "order_purchase_timestamp",
            "is_late",
        ] + feature_columns
    ].copy()

    result = result.replace(
        [np.inf, -np.inf], np.nan
    )

    result = result.dropna(
        subset=["order_purchase_timestamp", "is_late"]
    )

    result = result.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    if result["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs found in output.")

    if result["is_late"].nunique() != 2:
        raise ValueError(
            "Target contains only one class. Check the dates and source data."
        )

    output_file = PROCESSED / "orders_features.csv"
    result.to_csv(output_file, index=False)

    print("\n--- DATA PREPARATION COMPLETE ---")
    print(f"Rows: {len(result):,}")
    print(f"Columns: {len(result.columns)}")
    print(f"Late deliveries: {result['is_late'].sum():,}")
    print(f"Late-delivery rate: {result['is_late'].mean():.2%}")
    print(f"Duplicate order IDs: {result['order_id'].duplicated().sum()}")
    print(f"Output file: {output_file}")

    print("\nTarget distribution:")
    print(result["is_late"].value_counts().rename({
        0: "On time",
        1: "Late",
    }))

    return result


if __name__ == "__main__":
    prepare_data()
