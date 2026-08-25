import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# Load assets
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join("app", "model.joblib"))
    scaler = joblib.load(os.path.join("app", "scaler.joblib"))
    features = joblib.load(os.path.join("app", "features.joblib"))
    return model, scaler, features

model, scaler, feature_cols = load_assets()

def preprocess_input(df):
    """Applies the same feature engineering steps as training."""
    # Derived Features
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ['0-12', '13-24', '25-48', '49-60', '60+']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
    
    df['avg_charge_per_month'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])
        
    # Get dummies for categoricals
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align columns with training features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    
    # Scale numerics
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_charge_per_month']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # Ensure booleans are int (required for some models like XGBoost, though LogisticRegression handles both)
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
        
    return df

st.title("Customer Churn Prediction Dashboard")

st.markdown("""
This dashboard predicts customer churn based on our trained machine learning model.
Use the sidebar to upload a batch CSV or score a single customer.
""")

st.sidebar.header("Data Input")
input_mode = st.sidebar.radio("Select Input Mode", ["Batch Upload", "Single Customer"])

if input_mode == "Batch Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        # Ensure TotalCharges is numeric
        if 'TotalCharges' in input_data.columns:
            input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'].replace(' ', np.nan))
            input_data['TotalCharges'] = input_data['TotalCharges'].fillna(0)
            
        st.subheader("Batch Predictions")
        
        # Preprocess and Predict
        processed_data = preprocess_input(input_data.copy())
        preds = model.predict(processed_data)
        probs = model.predict_proba(processed_data)[:, 1]
        
        results_df = input_data.copy()
        results_df['Churn_Probability'] = probs
        results_df['Risk_Level'] = pd.cut(probs, bins=[0, 0.33, 0.66, 1.0], labels=['Low', 'Medium', 'High'])
        results_df['Predicted_Churn'] = np.where(preds == 1, 'Yes', 'No')
        
        # Filtering
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
        if risk_filter != "All":
            display_df = results_df[results_df['Risk_Level'] == risk_filter]
        else:
            display_df = results_df
            
        st.dataframe(display_df)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers Scored", len(results_df))
        col2.metric("Predicted to Churn", sum(preds))
        col3.metric("Average Churn Risk", f"{probs.mean():.1%}")

else:
    st.sidebar.subheader("Customer Details")
    # Form for single customer
    with st.sidebar.form("customer_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=300.0, value=70.0)
        total = st.number_input("Total Charges", min_value=0.0, value=800.0)
        
        submit = st.form_submit_button("Predict Churn")
        
    if submit:
        data = {
            'gender': [gender], 'SeniorCitizen': [senior], 'Partner': [partner],
            'Dependents': [dependents], 'tenure': [tenure], 'PhoneService': [phone],
            'MultipleLines': [multiple], 'InternetService': [internet], 'OnlineSecurity': [security],
            'OnlineBackup': [backup], 'DeviceProtection': [protection], 'TechSupport': [support],
            'StreamingTV': [tv], 'StreamingMovies': [movies], 'Contract': [contract],
            'PaperlessBilling': [paperless], 'PaymentMethod': [payment],
            'MonthlyCharges': [monthly], 'TotalCharges': [total]
        }
        single_df = pd.DataFrame(data)
        processed = preprocess_input(single_df)
        prob = model.predict_proba(processed)[0, 1]
        
        st.subheader("Prediction Result")
        risk = "High" if prob > 0.66 else "Medium" if prob > 0.33 else "Low"
        st.metric("Churn Risk", f"{prob:.1%}", delta_color="inverse")
        st.write(f"This customer is categorized as **{risk} Risk**.")
        if prob > 0.5:
            st.error("This customer is likely to churn. Consider retention offers.")
        else:
            st.success("This customer is likely to stay.")

# Global Feature Importance Chart
st.markdown("---")
st.subheader("Global Insights: What Drives Churn?")
if type(model).__name__ == "LogisticRegression":
    coefs = model.coef_[0]
    importance_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': coefs})
    importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
    top_features = importance_df.sort_values(by='Abs_Coefficient', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_features, x='Coefficient', y='Feature', palette='coolwarm', hue='Feature', legend=False, ax=ax)
    ax.set_title("Top 10 Churn Drivers (Impact Direction)")
    st.pyplot(fig)
    st.markdown("""
    * **Negative Values (Blue):** Factors that decrease the likelihood of churn (e.g., Two-year contracts, longer tenure).
    * **Positive Values (Red):** Factors that increase the likelihood of churn (e.g., Fiber optic internet).
    """)
else:
    # XGBoost fallback
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    top_features = importance_df.sort_values(by='Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False, ax=ax)
    ax.set_title("Top 10 Churn Drivers")
    st.pyplot(fig)
