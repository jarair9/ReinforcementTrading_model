import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env_advanced import AdversarialTradingEnv
from indicators import load_and_preprocess_data, add_correlation_features

def evaluate():
    print("--- AI AGENT REPORT CARD ---")
    
    # 1. Load Data (Same as Training)
    pairs = ["EURUSD", "GBPUSD", "XAUUSD"]
    data_1h = {}
    
    print("Loading 1H Data...")
    try:
        for p in pairs:
            data_1h[p] = load_and_preprocess_data(f"data/{p}_1h.csv")
            
        print("Computing Cross-Pair Correlations...")
        add_correlation_features(data_1h)
    except Exception as e:
        print(f"Data Load Error: {e}")
        return

    # 2. Setup Env (1H Only)
    env = AdversarialTradingEnv(data_dict=data_1h, 
                                data_dict_1h=None, 
                                data_dict_1d=None, 
                                window_size=30)
    vec_env = DummyVecEnv([lambda: env])
    
    # 3. Load Model
    model_name = "model_3pair_1h_sniper"
    try:
        model = SAC.load(model_name)
        print(f"Loaded Model: {model_name}")
    except:
        print(f"Could not load '{model_name}'.")
        return

    # 4. Run Backtest
    print("Running Backtest Simulation...")
    obs = vec_env.reset()
    done = False
    equity_curve = [env.initial_balance]
    
    total_trades = 0
    prev_positions = np.zeros(len(pairs))
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        # Trade Counting Logic
        current_positions = env.positions
        for i in range(len(pairs)):
            # Entry: 0 to non-zero
            if abs(current_positions[i]) > 0.1 and abs(prev_positions[i]) < 0.001:
                total_trades += 1
            # Flip: Sign change
            elif abs(current_positions[i]) > 0.1 and abs(prev_positions[i]) > 0.1 and np.sign(current_positions[i]) != np.sign(prev_positions[i]):
                total_trades += 2 # Close + Open
        
        prev_positions = current_positions.copy()
        
        # Extract equity from info (vectorized env returns list of infos)
        current_equity = info[0]['equity']
        equity_curve.append(current_equity)
        
    # 5. Calculate Metrics
    equity_curve = np.array(equity_curve)
    returns = pd.Series(equity_curve).pct_change().dropna()
    
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
    
    # Sharpe (Annualized assuming 1H candles -> 24 per day * 252 days)
    risk_free_rate = 0.0
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = 0.0
    if std_ret > 0:
        sharpe = (mean_ret * np.sqrt(24 * 252)) / std_ret
        
    # Win Rate (Percentage of periods with positive returns)
    win_rate = (returns > 0).mean() * 100
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    print("\n" + "="*30)
    print(f"FINAL RESULT: ${equity_curve[-1]:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print("="*30 + "\n")
    
    if total_return > 0 and sharpe > 1.0:
        print("VERDICT: PASS (Profitable & Stable)")
    elif total_return > 0:
        print("VERDICT: VOLATILE (Profitable but risky)")
    else:
        print("VERDICT: FAIL (Losing strategy)")

if __name__ == "__main__":
    evaluate()
