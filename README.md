
# 🚚 Logistics Delay Prediction & Supply Chain Intelligence

An end-to-end Data Science and Machine Learning project that predicts late-delivery risk using historical e-commerce data. The solution combines data preprocessing, feature engineering, classification models, model evaluation, and an interactive Streamlit dashboard to support proactive shipment risk assessment.

🔗 **GitHub Repository:** https://github.com/sm292/logistics-delay-prediction

## 📌 Project Overview

Delivery delays can negatively affect customer satisfaction, logistics costs, inventory planning, and supply chain efficiency.

This project uses the Olist Brazilian E-Commerce Public Dataset to build a machine learning pipeline that estimates the risk of late delivery and presents actionable order-level insights through an interactive dashboard.

The system also supports scoring new orders from CSV or JSON files and maintaining a persistent record of scored orders.

## 🎯 Business Objectives

- Predict whether an order is likely to be delivered late.
- Identify patterns associated with delivery performance.
- Engineer meaningful features from order, seller, product, and delivery data.
- Compare classification models using appropriate evaluation metrics.
- Generate late-delivery probabilities and configurable risk categories.
- Enable batch prediction through CSV and JSON uploads.
- Track previously scored order IDs to reduce duplicate processing.
- Present results through an interactive analytics dashboard.

## 🗂️ Dataset

**Source:** Olist Brazilian E-Commerce Public Dataset

The project uses selected datasets covering:

- Customer and order information
- Order item quantities and prices
- Freight costs
- Seller and customer locations
- Product categories
- Purchase and estimated delivery dates

The target variable identifies whether an eligible order was delivered after its estimated delivery date.

The model uses historical data and is intended for retrospective evaluation and risk estimation, not as a live carrier-tracking system.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Classification | Logistic Regression, Random Forest |
| Visualization | Plotly, Matplotlib, Seaborn |
| Interactive Dashboard | Streamlit |
| Model Persistence | Joblib |
| Data Exchange | CSV, JSON |
| Testing Support | Pytest |
| Version Control | Git, GitHub |

## ⚙️ Methodology

### 1. Data Preparation

- Load and combine the required Olist datasets.
- Parse purchase and delivery timestamps.
- Validate records and handle missing values.
- Define the late-delivery target.
- Prepare a structured dataset for model development.

### 2. Feature Engineering

Create predictive features such as:

- Purchase month, weekday, and hour
- Weekend purchase indicator
- Estimated delivery duration
- Number of items and sellers per order
- Total product price and freight
- Customer and seller states
- Same-state shipping indicator
- Product category

### 3. Machine Learning

Train and compare:

- Logistic Regression
- Random Forest Classifier

The preprocessing pipeline handles numerical and categorical features through imputation, scaling, and one-hot encoding where appropriate.

A chronological train/test split is used to better reflect the order of events over time.

### 4. Model Evaluation

Evaluate model performance using:

- Average Precision
- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

The model selection process uses Average Precision on the test set. Actual performance figures should be taken from the generated evaluation outputs.

### 5. Shipment Risk Scoring

The prediction pipeline produces:

- Predicted late-delivery probability
- Predicted delivery-risk label
- Risk category
- Scoring timestamp

Configurable risk bands:

| Risk Category | Predicted Probability |
|---|---|
| Low | Below 30% |
| Medium | 30% to below 60% |
| High | 60% or above |

These thresholds are configurable business rules, not guarantees of actual delivery outcomes.

## 📊 Streamlit Dashboard

The interactive dashboard supports:

- Historical model evaluation and order analysis
- CSV and JSON file uploads
- Batch scoring of new orders
- Order-level risk assessment
- Adjustable classification thresholds
- Risk distribution visualizations
- Model comparison results
- Actual-versus-predicted analysis when outcomes are available
- Order risk register and CSV export
- Persistent storage of newly scored orders
- Duplicate order ID checks
- Manual and periodic dashboard refresh

The application refreshes available local data but does not automatically retrain the model or retrieve real-time carrier updates.

## 📁 Project Structure

```text
logistics-delay-prediction/
├── app.py
├── app_backup.py
├── README.md
├── .gitignore
├── models/
│   ├── delay_model.joblib
│   └── metadata.json
├── src/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── score_new_orders.py
├── data/
│   ├── raw/
│   └── processed/
└── tests/
```

Some directories and generated files are created during the execution of the pipeline.

## 🚀 Installation & Execution

### 1. Clone the repository

```bash
git clone https://github.com/sm292/logistics-delay-prediction.git
cd logistics-delay-prediction
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit plotly pytest streamlit-autorefresh
```

### 4. Prepare the dataset

Download the required Olist dataset files and place them in `data/raw/`.

Run:

```bash
python src/prepare_data.py
```

### 5. Train and evaluate the models

```bash
python src/train_model.py
```

This step generates the trained model, model metadata, model comparison results, and test-order predictions.

### 6. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 💡 Business Applications

- Prioritize orders with elevated predicted delay risk.
- Support logistics exception management.
- Identify delivery-performance patterns across product and location segments.
- Improve operational monitoring and investigation.
- Support data-driven discussions around shipment planning and customer experience.

These are potential applications; actual operational improvements require validation on representative business data.

## 🔮 Future Enhancements

- Explain predictions using SHAP and feature importance.
- Integrate authorized carrier tracking, traffic, and weather APIs.
- Add model drift monitoring and scheduled retraining.
- Improve duplicate detection and updated-order handling.
- Introduce automated testing and CI/CD.
- Deploy with Docker and cloud infrastructure.
- Add authentication, audit logs, and role-based access controls.

## 👨‍💻 Author

**Sumit Mondal**

Data Science | Machine Learning | Python | SQL | Business Intelligence

- GitHub: https://github.com/sm292
- Project: https://github.com/sm292/logistics-delay-prediction

## ⚠️ Disclaimer

This repository is a portfolio and learning project. Predictions are estimates based on historical data and should not be treated as guaranteed delivery outcomes. Evaluate model performance and validate the solution on representative operational data before using it for business decisions.
