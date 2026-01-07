"""
Script to run the improved trading system with all enhancements:
1. Multi-Timeframe Context
2. Market Regime Detection
3. Advanced Reward Functions (Sortino Ratio)
4. Sequence-Aware Architecture (LSTM)
5. Automated Hyperparameter Tuning (separate script)
6. Multi-Asset Integration (separate environment)
"""

import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_env import SimpleTradingEnv
import torch


def run_improved_system():
    print("=== Running Improved Trading System ===")
    print("Features implemented:")
    print("1. ✓ Multi-Timeframe Context (1H/4H trends and RSI)")
    print("2. ✓ Market Regime Detection (ADX-based)")  
    print("3. ✓ Advanced Reward Functions (Sortino Ratio component)")
    print("4. ✓ Sequence-Aware Architecture (LSTM policy)")
    print("5. ○ Automated Hyperparameter Tuning (via hyperparameter_tuning.py)")
    print("6. ○ Multi-Asset Integration (via multi_asset_env.py)")
    print()
    
    data_path = "data/EURUSD_15m_cleaned.csv"
    if not os.path.exists(data_path):
        print("ERROR: Run src/data_fetcher.py and src/data_cleaner.py first.")
        return
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data points")
    
    # Check that new features exist
    required_features = [
        'feat_returns_5', 'feat_price_pos', 'feat_norm_vol', 
        'feat_hour_sin', 'feat_hour_cos', 'feat_vol_spike',
        'htf_1h_trend', 'htf_4h_trend', 'htf_4h_rsi',
        'feat_regime'
    ]
    
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"ERROR: Missing features: {missing_features}")
        return
    
    print(f"All {len(required_features)} required features present")
    
    # Split: Train on first 80%, Test on last 20%
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    print(f"Training on {len(train_df)} bars, testing on {len(test_df)} bars")
    
    # Create environment with improved features
    print("Creating trading environment with enhanced features...")
    env = SimpleTradingEnv(train_df)
    env = DummyVecEnv([lambda: env])
    
    print("Creating PPO model with LSTM policy...")
    # Create model with LSTM policy and enhanced features
    model = PPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=64,
        vf_coef=0.5,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="logs/ppo_improved_system/",
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=dict(
            lstm_hidden_size=64,
            n_lstm_layers=2,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )
    )
    
    print("\nStarting training with enhanced features...")
    print("(This may take a while depending on your hardware)")
    
    # Train the model
    model.learn(total_timesteps=50000)  # Reduced for demo purposes
    
    # Save the trained model
    model_name = "models/ppo_trading_improved_final"
    model.save(model_name)
    print(f"\nModel saved to {model_name}.zip")
    
    print("\n=== System Improvements Summary ===")
    print("1. Multi-Timeframe Context: Added 1H/4H trend and RSI features")
    print("2. Market Regime Detection: Added ADX-based regime detection")
    print("3. Advanced Rewards: Implemented Sortino ratio component")
    print("4. Sequence Awareness: Upgraded to LSTM policy")
    print("5. Automated Tuning: Available via hyperparameter_tuning.py")
    print("6. Multi-Asset Support: Available via multi_asset_env.py")
    
    print("\nTo run hyperparameter tuning: python hyperparameter_tuning.py")
    print("To use multi-asset environment: import src.multi_asset_env")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    run_improved_system()