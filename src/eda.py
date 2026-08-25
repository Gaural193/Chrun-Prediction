import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clean_and_explore():
    data_path = os.path.join("data", "churn_dataset.csv")
    df = pd.read_csv(data_path)
    
    # 1. Clean TotalCharges
    # Replace blank strings with NaN, then convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
    
    # Fill NaN with 0 or drop them. The guide says 'handle blanks'. 
    # Usually, 0 tenure -> blank TotalCharges. Let's fill with 0.
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    print("--- Cleaned Data Types ---")
    print(df.dtypes)
    
    # Save cleaned data
    cleaned_path = os.path.join("data", "cleaned_churn_dataset.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataset saved to {cleaned_path}")
    
    # 2. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    # 3. Generate Plots
    # Make sure notebooks directory exists for output
    os.makedirs("notebooks", exist_ok=True)
    
    # Plot 1: Churn rate by contract type
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn')
    plt.title('Churn by Contract Type')
    plt.tight_layout()
    plt.savefig('notebooks/churn_by_contract.png')
    plt.close()
    
    # Plot 2: Churn rate by tenure (using a boxplot or histplot)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
    plt.title('Churn by Tenure')
    plt.tight_layout()
    plt.savefig('notebooks/churn_by_tenure.png')
    plt.close()
    
    # Plot 3: Churn rate by internet service
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='InternetService', hue='Churn')
    plt.title('Churn by Internet Service')
    plt.tight_layout()
    plt.savefig('notebooks/churn_by_internet.png')
    plt.close()
    
    print("EDA plots saved in notebooks/")

if __name__ == "__main__":
    clean_and_explore()
