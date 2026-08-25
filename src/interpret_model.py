import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def interpret_model():
    print("Loading the best model and features...")
    model = joblib.load(os.path.join("app", "model.joblib"))
    features = joblib.load(os.path.join("app", "features.joblib"))
    
    # Check model type to determine how to get importances
    model_type = type(model).__name__
    
    if model_type == "XGBClassifier":
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        print("\n--- Top 10 Drivers of Churn (XGBoost Feature Importance) ---")
        print(importance_df.head(10))
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette='viridis')
        plt.title("Top 10 Churn Drivers")
        plt.tight_layout()
        plt.savefig('notebooks/feature_importance.png')
        plt.close()
        
    elif model_type == "LogisticRegression":
        # Coefficients
        coefs = model.coef_[0]
        importance_df = pd.DataFrame({'Feature': features, 'Coefficient': coefs})
        # Absolute magnitude determines importance
        importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
        importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False)
        
        print("\n--- Top 10 Drivers of Churn (Logistic Regression Coefficients) ---")
        print(importance_df.head(10))
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x='Coefficient', y='Feature', palette='coolwarm')
        plt.title("Top 10 Churn Drivers (Impact Direction)")
        plt.tight_layout()
        plt.savefig('notebooks/feature_importance.png')
        plt.close()
        
    else:
        print("Unknown model type for interpretation.")

if __name__ == "__main__":
    interpret_model()
