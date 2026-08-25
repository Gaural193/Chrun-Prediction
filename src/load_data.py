import pandas as pd
import os

def load_and_explore_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print("\n--- Basic Information ---")
    print(f"Shape: {df.shape}")
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    return df

if __name__ == "__main__":
    # Assuming script is run from project root
    filepath = os.path.join("data", "churn_dataset.csv")
    load_and_explore_data(filepath)
