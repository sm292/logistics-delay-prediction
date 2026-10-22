
# Logistics Delay Prediction & Supply Chain Intelligence

## Project Overview

This project uses machine learning to predict potential delivery delays and provide supply chain risk insights. It processes historical e-commerce order data, engineers relevant delivery-related features, compares classification models, and presents order-level risk predictions through an interactive Streamlit dashboard.

The project uses the Brazilian E-Commerce Public Dataset by Olist for model development and evaluation.

**Project type:** End-to-end Data Science and Machine Learning  
**Domain:** Logistics, E-commerce, Supply Chain Analytics  
**Interface:** Streamlit dashboard  
**Language:** Python

## Business Problem

Late deliveries can affect customer satisfaction, operational efficiency, seller performance, and logistics costs.

This project aims to help identify orders that may be at greater risk of arriving late, allowing logistics teams to investigate potentially problematic orders and prioritize operational attention.

## Project Objectives

- Prepare and validate historical e-commerce order data.
- Engineer features relevant to delivery performance.
- Define late delivery as delivery after the estimated delivery date.
- Train and compare Logistic Regression and Random Forest classifiers.
- Evaluate models using Average Precision, ROC-AUC, precision, recall, F1-score, and a confusion matrix.
- Generate order-level late-delivery risk probabilities.
- Categorize predictions into Low, Medium, and High risk.
- Display model results and risk indicators in a Streamlit dashboard.
- Support CSV and JSON uploads for scoring new orders.
- Store newly scored orders and identify previously processed order IDs.

## Technology Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web dashboard | Streamlit |
| Model persistence | Joblib |
| Data storage | CSV and JSON |
| Testing support | Pytest |
| Version control | Git and GitHub |

## Dataset

The project uses selected files from the Olist Brazilian E-Commerce Public Dataset.

Required source files:

- `olist_orders_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_products_dataset.csv`

These files provide information about orders, delivery timestamps, customers, sellers, products, prices, and freight.

The dataset is used to build a retrospective predictive model. It does not represent a live logistics feed.

## Project Structure

```text
logistics-delay-prediction/
│
├── app.py
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── olist_orders_dataset.csv
│   │   ├── olist_customers_dataset.csv
│   │   ├── olist_order_items_dataset.csv
│   │   ├── olist_sellers_dataset.csv
│   │   └── olist_products_dataset.csv
│   │
│   └── processed/
│       ├── orders_features.csv
│       ├── model_comparison.csv
│       ├── scored_test_orders.csv
│       └── live_scored_orders.csv
│
├── src/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── score_new_orders.py
│
├── models/
│   ├── delay_model.joblib
│   └── metadata.json
│
├── tests/
│
└── .venv/
```

Some generated files may not exist until the relevant pipeline stage has been executed.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd logistics-delay-prediction
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can temporarily allow activation for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit plotly pytest streamlit-autorefresh
```

## Execution Guide

Run commands from the project root directory, with the virtual environment activated.

### Step 1: Prepare the dataset

Place the five required Olist CSV files in `data/raw/`.

Run:

```bash
python src/prepare_data.py
```

Expected output:

```text
data/processed/orders_features.csv
```

This stage processes order timestamps, filters suitable delivered orders, creates the late-delivery target, engineers features, joins supporting datasets, and validates the resulting data.

### Step 2: Train and evaluate models

Run:

```bash
python src/train_model.py
```

Expected outputs:

```text
models/delay_model.joblib
models/metadata.json
data/processed/model_comparison.csv
data/processed/scored_test_orders.csv
```

The training pipeline uses a chronological train/test split to better reflect the order of events over time.

Models compared:

- Logistic Regression
- Random Forest Classifier

The selected model is saved for subsequent scoring.

### Step 3: Launch the dashboard

Run:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

### Step 4: Score uploaded orders

Prepare a CSV or JSON file containing the required feature columns and a unique `order_id` for each order.

Upload the file through the dashboard and trigger scoring.

The scoring pipeline generates:

- Late-delivery risk probability
- Predicted late-delivery label
- Risk band
- Scoring timestamp

Newly scored orders are stored in:

```text
data/processed/live_scored_orders.csv
```

The application checks existing IDs in the live scoring file to avoid scoring those IDs again. Duplicate IDs within an uploaded file are reduced to the first occurrence.

**Important:** Existing IDs with changed feature values are currently skipped rather than updated. The historical test-set file is separate from the live scoring file, so an ID present only in the historical file may be scored again.

## Feature Engineering

The model uses the following features:

| Feature | Description |
|---|---|
| `purchase_month` | Month of purchase |
| `purchase_weekday` | Day of the week of purchase |
| `purchase_hour` | Hour of purchase |
| `is_weekend` | Whether purchase occurred on a weekend |
| `estimated_delivery_days` | Estimated delivery duration |
| `item_count` | Number of items in the order |
| `total_price` | Combined item price |
| `total_freight` | Combined freight value |
| `seller_count` | Number of distinct sellers |
| `same_state` | Whether customer and seller are in the same state |
| `customer_state` | Customer state |
| `seller_state` | Seller state |
| `product_category_name` | Product category |

The preprocessing pipeline imputes missing values, scales numerical features, and one-hot encodes categorical features.

## Model Evaluation

The models are evaluated using:

- **Average Precision:** Useful for evaluating performance when late deliveries are less common.
- **ROC-AUC:** Measures the model's ability to rank late and on-time deliveries.
- **Precision:** Proportion of predicted late orders that were actually late.
- **Recall:** Proportion of actual late orders identified by the model.
- **F1-score:** Balance between precision and recall.
- **Confusion matrix:** Summarizes correct and incorrect classifications.

The selected model is chosen using Average Precision on the test set.

A classification threshold of 0.5 is used for the standard predicted label, while the dashboard allows the threshold to be adjusted.

Actual evaluation scores should be taken from the generated project output files rather than assumed in advance.

## Risk Classification

The scoring pipeline assigns risk bands based on predicted probability:

| Risk band | Probability |
|---|---|
| Low | Below 0.30 |
| Medium | 0.30 to below 0.60 |
| High | 0.60 or above |

These thresholds are configurable business rules, not guaranteed probabilities of actual operational failure.

## Dashboard Capabilities

The Streamlit application provides:

- Historical test-set and uploaded-order analysis
- CSV and JSON order uploads
- Order-level risk scoring
- Low, Medium, and High risk classification
- Adjustable classification threshold
- Risk distribution charts
- Model comparison results
- Actual-versus-predicted analysis where actual outcomes are available
- Order risk register and CSV export
- Periodic dashboard refresh and manual refresh
- Duplicate-order handling for live-scored records

Automatic refresh reloads available local data; it does not automatically retrain the model or connect to a live carrier tracking system.

## Data Privacy and GitHub

Do not commit the virtual environment, raw customer/order datasets, private credentials, or large generated files unless you have a clear reason and permission to share them.

For a public portfolio, prefer documented dataset download instructions and reproducible setup steps. Review the dataset's terms and repository contents before publishing.

## Limitations

- The model is trained on historical Olist data.
- Model performance may differ on new regions, sellers, products, or operational conditions.
- Predictions are estimates, not guarantees of delivery outcomes.
- The current pipeline requires prepared model features rather than arbitrary raw order records.
- Risk thresholds are configurable and should be validated against business costs.
- The dashboard does not independently retrieve real-time tracking updates.
- Estimated financial savings or avoided losses have not been established by this project.

## Future Improvements

- Add automated unit and integration tests.
- Improve duplicate detection across historical and live datasets.
- Add explicit handling for updated orders.
- Tune decision thresholds using business costs and validation data.
- Add model explainability with feature importance or SHAP.
- Add drift monitoring and periodic model retraining.
- Integrate authorized carrier tracking APIs.
- Add authentication, access control, and audit logging before production deployment.
- Package the application with Docker and set up CI/CD.

## Author

**Sumit Mondal**

Data Science | Machine Learning | Python | SQL | Business Intelligence

GitHub: https://github.com/sm292

## Disclaimer

This is a portfolio and learning project. It is not a production-validated logistics decision system. Predictions should be evaluated on representative operational data before being used for business decisions.
