import pandas as pd
import numpy as np
import os
import argparse
from scipy.stats import spearmanr

def diagnose_data(file_path):
    print(f"--- Diagnosing: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # 1. Leakage Check: Returns vs Next Return
    df['returns'] = df['Close'].pct_change()
    df['next_return'] = df['returns'].shift(-1)
    
    leakage_corr = df['returns'].corr(df['next_return'])
    print(f"Target Leakage Check (Lagged Corr): {leakage_corr:.4f}")
    
    # Check for feature-price contemporaneous correlation
    feat_cols = [c for c in df.columns if 'feat_' in c or 'htf_' in c]
    for col in feat_cols:
        corr = df[col].corr(df['Close'])
        if abs(corr) > 0.8:
            print(f"CRITICAL LEAKAGE: {col} has {corr:.4f} corr with price.")
            
    if abs(leakage_corr) < 0.001:
        print("PASS: Target Leakage is < 0.1%.")
    else:
        print(f"WARNING: Target Leakage is {leakage_corr:.4f}. Aim for < 0.1%.")

    # 2. Benchmarks: Buy & Hold vs Moving Average Crossover
    # B&H
    total_bh_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
    
    # MA Crossover (50/200)
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    # 1 when 50 > 200 (Long), -1 when 50 < 200 (Short)
    signals = (sma_50 > sma_200).astype(int).diff() 
    
    # 3. Cleanliness: Check for NaNs/Zeros
    nan_count = df.isna().sum().sum()
    print(f"NaN Count: {nan_count}")
    zero_vol = (df['Volume'] == 0).sum()
    print(f"Zero Volume Bars: {zero_vol} (Expected on weekends/news gaps)")

    print(f"\nBaseline: Buy & Hold Total Return: {total_bh_return*100:.2f}%")
    print("Optimization Tip: If B&H > 5%, the market is trending heavily.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/EURUSD_15m_cleaned.csv")
    args = parser.parse_args()
    
    diagnose_data(args.file)
