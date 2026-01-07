import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_features(data_path):
    print(f"--- Feature Predictability Analysis: {data_path} ---")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 6 Original + 3 HTF + 1 regime Non-Leaking Features
    features = [
        'feat_returns_5', 'feat_price_pos', 'feat_norm_vol', 
        'feat_hour_sin', 'feat_hour_cos', 'feat_vol_spike',
        'htf_1h_trend', 'htf_4h_trend', 'htf_4h_rsi',
        'feat_regime'
    ]
    
    # Ensure features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Missing features: {missing}")
        return

    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # 1. Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[features + ['target']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation with Target (Next Price Direction)")
    plt.tight_layout()
    plt.savefig("logs/feature_correlation.png")
    
    # 2. Predictability by Hour
    if 'Datetime' in df.columns:
        df['hour'] = df['Datetime'].dt.hour
        hourly_acc = []
        for h in range(24):
            subset = df[df['hour'] == h]
            if len(subset) > 0:
                # Predict direction based on momentum (feat_returns_5)
                # If returns > 0, predict Up (1). This is a simple trend following proxy.
                pred = (subset['feat_returns_5'] > 0).astype(int)
                acc = (pred == subset['target']).mean()
                hourly_acc.append(acc)
            else:
                hourly_acc.append(0.5)
                
        plt.figure(figsize=(12, 5))
        sns.barplot(x=list(range(24)), y=hourly_acc, palette="viridis")
        plt.axhline(0.5, color='red', linestyle='--')
        plt.title("Simple Predictability Accuracy by Hour (UTC)")
        plt.ylabel("Accuracy")
        plt.xlabel("Hour of Day")
        plt.ylim(0.45, 0.55)
        plt.tight_layout()
        plt.savefig("logs/predictability_by_hour.png")
    
    print("Analysis plots saved to logs/ directory.")

def analyze_trade_patterns(trades_csv):
    """
    Analyzes trading logs after evaluation.
    """
    if not os.path.exists(trades_csv):
        print(f"Note: {trades_csv} not found. Run evaluation to generate trade logs.")
        return
        
    df = pd.read_csv(trades_csv)
    # logic for trade statistics...
    print(f"Trade Analysis summary for {len(df)} trades.")

if __name__ == "__main__":
    analyze_features("data/EURUSD_15m_cleaned.csv")
    analyze_trade_patterns("logs/evaluated_trades.csv")
