import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             RocCurveDisplay)
import joblib
import os

def evaluate_model(y_true, y_pred, y_prob, model_name):
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")
    return roc_auc_score(y_true, y_prob)

def train_and_evaluate():
    print("Loading prepared datasets...")
    X_train = pd.read_csv(os.path.join("data", "X_train.csv"))
    X_test = pd.read_csv(os.path.join("data", "X_test.csv"))
    y_train = pd.read_csv(os.path.join("data", "y_train.csv"))['Churn']
    y_test = pd.read_csv(os.path.join("data", "y_test.csv"))['Churn']
    
    # 1. Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_auc = evaluate_model(y_test, lr_preds, lr_probs, "Logistic Regression")
    
    # 2. Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = evaluate_model(y_test, xgb_preds, xgb_probs, "XGBoost")
    
    # 3. Determine best model
    best_model = xgb_model if xgb_auc > lr_auc else lr_model
    best_name = "XGBoost" if xgb_auc > lr_auc else "Logistic Regression"
    best_preds = xgb_preds if xgb_auc > lr_auc else lr_preds
    best_probs = xgb_probs if xgb_auc > lr_auc else lr_probs
    
    print(f"\nBest model based on ROC-AUC is {best_name}.")
    
    # Save best model
    joblib.dump(best_model, os.path.join("app", "model.joblib"))
    print("Best model saved to app/model.joblib")
    
    # Also save column names so we know feature order for the app
    joblib.dump(list(X_train.columns), os.path.join("app", "features.joblib"))
    
    # 4. Plots for best model
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{best_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('notebooks/best_model_confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(6,4))
    RocCurveDisplay.from_predictions(y_test, best_probs, name=best_name)
    plt.title(f'{best_name} - ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.tight_layout()
    plt.savefig('notebooks/best_model_roc_curve.png')
    plt.close()
    
    print("Evaluation plots saved in notebooks/")

if __name__ == "__main__":
    train_and_evaluate()
