
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "orders_features.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_late"

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
    "product_category_name",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline(classifier):
    """Preprocess data and train a classifier in one pipeline."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def main():
    print("=" * 60)
    print("LOGISTICS DELAY PREDICTION - MODEL TRAINING")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_FILE}\n"
            "Run: python src/prepare_data.py"
        )

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["order_purchase_timestamp"],
    )

    required_columns = FEATURES + [
        TARGET, "order_id", "order_purchase_timestamp"
    ]

    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(
        subset=["order_purchase_timestamp", TARGET]
    ).sort_values("order_purchase_timestamp").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("Too few records for this training workflow.")

    if df[TARGET].nunique() != 2:
        raise ValueError("The target must contain both classes: 0 and 1.")

    # Chronological split: oldest 80% for training,
    # newest 20% for testing.
    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "Training and test periods must each contain both classes. "
            "Inspect the chronological target distribution."
        )

    print(f"\nTraining records: {len(train_df):,}")
    print(f"Testing records:  {len(test_df):,}")
    print(f"Training late rate: {y_train.mean():.2%}")
    print(f"Testing late rate:  {y_test.mean():.2%}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    trained_models = {}
    results = []

    for name, classifier in models.items():
        print(f"\n{'-' * 60}")
        print(f"Training {name}...")

        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)

        metrics = {
            "model": name,
            "average_precision": average_precision_score(
                y_test, probabilities
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
        }

        trained_models[name] = pipeline
        results.append(metrics)

        print("\nEvaluation metrics:")
        for metric, value in metrics.items():
            if metric != "model":
                print(f"{metric}: {value:.4f}")

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, predictions))

        print("\nClassification report:")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

    # Average precision is the primary comparison metric.
    comparison = pd.DataFrame(results).sort_values(
        "average_precision", ascending=False
    )

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    best_name = comparison.iloc[0]["model"]
    best_model = trained_models[best_name]

    # Save the entire pipeline, including preprocessing.
    model_path = MODEL_DIR / "delay_model.joblib"
    joblib.dump(best_model, model_path)

    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "classification_threshold": 0.50,
        "selection_metric": "average_precision",
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_start": str(
            test_df["order_purchase_timestamp"].min()
        ),
        "test_end": str(
            test_df["order_purchase_timestamp"].max()
        ),
    }

    with open(
        MODEL_DIR / "metadata.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    # Generate test-set predictions for the dashboard.
    predictions_df = test_df[
        ["order_id", "order_purchase_timestamp", TARGET]
    ].copy()

    predictions_df["late_risk_probability"] = (
        best_model.predict_proba(X_test)[:, 1]
    )

    predictions_df["predicted_late"] = (
        predictions_df["late_risk_probability"] >= 0.50
    ).astype(int)

    predictions_df["risk_band"] = pd.cut(
        predictions_df["late_risk_probability"],
        bins=[-0.001, 0.30, 0.60, 1.00],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    predictions_df.to_csv(
        OUTPUT_DIR / "scored_test_orders.csv",
        index=False,
    )

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.round(4).to_string(index=False))

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {model_path}")
    print(
        f"Saved comparison: {OUTPUT_DIR / 'model_comparison.csv'}"
    )
    print(
        f"Saved predictions: {OUTPUT_DIR / 'scored_test_orders.csv'}"
    )
    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
