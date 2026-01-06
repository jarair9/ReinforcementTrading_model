import pandas as pd
import numpy as np
from trading_env_advanced import AdversarialTradingEnv

def create_dummy_data():
    dates = pd.date_range(start="2024-01-01", periods=100, freq='1h')
    data = {
        'Open': np.linspace(1.1000, 1.1100, 100),
        'High': np.linspace(1.1010, 1.1110, 100),
        'Low': np.linspace(1.0990, 1.1090, 100),
        'Close': np.linspace(1.1000, 1.1100, 100), # Constant trend up
        'Volume': np.random.rand(100) * 1000,
        'atr': np.full(100, 0.0020) # Constant volatility
    }
    df = pd.DataFrame(data, index=dates)
    
    # Add some indicators expected by Env
    df['log_ret'] = 0.0001
    df['rsi'] = 0.5
    
    return {'EURUSD': df}

def test_friction():
    print("--- Testing Friction (Spread & Slippage Costs) ---")
    data = create_dummy_data()
    
    env = AdversarialTradingEnv(data_dict=data, window_size=10, initial_balance=10000.0)
    
    obs = env.reset()
    start_equity = env.equity
    print(f"Start Equity: {start_equity:.2f}")
    
    # Step 1: Buy Max (100% position)
    # Price is approx 1.1000. 
    # Spread is default 2.0 pips (0.0002).
    # Slippage (Adversarial) might trigger if delta > 0.1.
    # We expect immediate loss due to spread/slippage.
    
    action = np.array([1.0]) # 100% Long EURUSD
    obs, reward, done, info = env.step(action)
    
    end_equity = env.equity
    cost = start_equity - end_equity
    
    print(f"Eq after Trade: {end_equity:.2f}")
    print(f"Cost Incurred: {cost:.4f}")
    print(f"Info Costs: {info['costs']:.6f}")
    
    if start_equity > end_equity:
        print("[PASS] Equity decreased immediately after trade (Friction exists).")
    else:
        print("[FAIL] Equity did not decrease! Frictionless trading detected.")
        
    # Step 2: Hold
    # We are 100% long. Price moves up in our dummy data (linspace).
    # We expect equity to rise.
    prev_eq = env.equity
    obs, reward, done, info = env.step(action) # Keep holding
    
    print(f"Eq after Hold: {env.equity:.2f}")
    if env.equity > prev_eq:
        print("[PASS] Equity increased as price moved up (PnL working).")
    else:
        print("[FAIL] Equity didn't track price move.")

if __name__ == "__main__":
    test_friction()
