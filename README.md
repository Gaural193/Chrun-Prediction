# Telco Customer Churn Prediction Dashboard

This project is an end-to-end machine learning system that predicts customer churn. It takes historical customer data, cleans it, trains a classification model, and serves predictions through an interactive Streamlit dashboard. 

The goal is to provide non-technical stakeholders with an easy-to-use tool to identify high-risk customers and understand the key factors driving churn.

## Project Structure
- `data/`: Contains the original and cleaned CSV datasets.
- `notebooks/`: Contains generated exploratory data analysis (EDA) plots and model evaluation charts (Confusion Matrix, ROC Curve, Feature Importance).
- `src/`: Python scripts for the data pipeline:
  - `load_data.py`: Loads the raw data.
  - `eda.py`: Cleans the data and generates exploratory visualizations.
  - `feature_engineering.py`: Preprocesses data, handles encoding/scaling, and splits into train/test sets.
  - `train_models.py`: Trains Logistic Regression and XGBoost models, evaluates them, and exports the best one.
  - `interpret_model.py`: Extracts and visualizes the top factors driving churn.
- `app/`: Contains the Streamlit dashboard (`app.py`) and the exported machine learning models (`model.joblib`, `scaler.joblib`, `features.joblib`).

## Screenshots

*(Add your screenshots below!)*

### 1. Single Customer Prediction
![Single Customer Prediction](screenshot_single_customer.png)

### 2. Batch Scoring & Filtering
![Batch Scoring](screenshot_batch_scoring.png)

### 3. Key Churn Drivers
![Churn Drivers](notebooks/feature_importance.png)

## The Tech Stack
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (Logistic Regression, Data Scaling), `xgboost`
- **Visualizations:** `matplotlib`, `seaborn`
- **Dashboard:** `streamlit`

## Key Insights
Based on our Logistic Regression model, the top factors that influence whether a customer will churn are:
1. **Two-Year Contracts & Longer Tenure:** These strongly decrease the likelihood of a customer leaving.
2. **Fiber Optic Internet:** Customers on the Fiber Optic plan are noticeably more likely to churn, indicating a potential issue with pricing or service reliability in that specific tier.

## How to Run Locally

1. **Clone the repository**
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app/app.py
   ```
