import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import argparse
import os

def train_classifier(file_path):
    print(f"--- Training Supervised Baseline (3-Feature System): {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # 1. Target: 1 if close price increases in NEXT bar, else 0
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Fixed 6 Original + 3 HTF + 1 regime Features (Strictly non-leaking)
    features = [
        'feat_returns_5', 'feat_price_pos', 'feat_norm_vol', 
        'feat_hour_sin', 'feat_hour_cos', 'feat_vol_spike',
        'htf_1h_trend', 'htf_4h_trend', 'htf_4h_rsi',
        'feat_regime'
    ]
    
    df.dropna(inplace=True)
    
    X = df[features]
    y = df['target']
    
    # Time-series split (No Shuffling)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"Training on {len(X_train)} bars, testing on {len(X_test)}")
    
    # XGBoost Fallback to RandomForest
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        print("Using XGBClassifier")
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        print("Using RandomForestClassifier (XGBoost not found)")
        
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\nFinal Validation Accuracy: {acc:.4f}")
    
    # Most Frequent Baseline
    most_freq_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"Baseline (Dummy/Frequent): {most_freq_acc:.4f}")
    
    # Success threshold: 52.5%
    if acc > 0.525:
        print("SUCCESS: Signal strength exceeds 52.5% target.")
    elif acc > most_freq_acc:
        print("NOTE: Predicting better than chance, but below 52.5% target.")
    else:
        print("FAILURE: No predictive power found. Overfitting likely.")

    # Feature Importance
    print("\nFeature Importances:")
    importances = model.feature_importances_
    for f, imp in zip(features, importances):
        print(f"{f}: {imp:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/EURUSD_15m_cleaned.csv")
    args = parser.parse_args()
    
    train_classifier(args.file)
