import pandas as pd
import numpy as np
import os

def run_benchmarks(data_path):
    print(f"--- Running Baseline Benchmarks: {data_path} ---")
    if not os.path.exists(data_path):
        print("Clean data first.")
        return
        
    df = pd.read_csv(data_path)
    df['returns'] = df['Close'].pct_change()
    
    # 1. Buy & Hold
    bh_ret = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    
    # 2. SMA Crossover (50/200) - Realistic for trends
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    df['signal'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
    # Strategy returns
    df['strat_ret'] = df['signal'].shift(1) * df['returns']
    sma_ret = (1 + df['strat_ret'].fillna(0)).prod() - 1
    
    # 3. Random Balanced Strategy (Balanced coins)
    np.random.seed(42)
    random_signals = np.random.choice([-1, 0, 1], size=len(df))
    random_ret = (1 + random_signals * df['returns']).prod() - 1
    
    print(f"Buy & Hold Return: {bh_ret*100:.2f}%")
    print(f"SMA 50/200 Return: {sma_ret*100:.2f}%")
    print(f"Random Strategy Return: {random_ret*100:.2f}%")
    
    sharpe = df['strat_ret'].mean() / (df['strat_ret'].std() + 1e-9) * np.sqrt(96 * 252)
    print(f"SMA 50/200 Annualized Sharpe: {sharpe:.4f}")

if __name__ == "__main__":
    run_benchmarks("data/EURUSD_15m_cleaned.csv")
