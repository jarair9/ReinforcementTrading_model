from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_env import SimpleTradingEnv
import pandas as pd
import torch
import os

def train():
    data_path = "data/EURUSD_15m_cleaned.csv"
    if not os.path.exists(data_path):
        print("Run clean_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    # Split: Train on first 80%, Test on last 20%
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    
    env = SimpleTradingEnv(train_df)
    env = DummyVecEnv([lambda: env])
    
    # SPECIALIZED HYPERPARAMETERS (Enhanced Kindergarten Phase)
    # Implements Multi-Timeframe Context, Market Regime Detection, LSTM Architecture
    model = PPO(
        "MlpLstmPolicy",  # Using LSTM policy for sequence awareness
        env,
        learning_rate=3e-5, # Ultra-slow for noise handle
        n_steps=2048,       # Long sequences
        batch_size=64,      # Frequent updates
        vf_coef=0.5,        # Focus on value estimation accuracy
        ent_coef=0.01,      # Exploration balance
        gamma=0.99,         # Standard lookahead
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="logs/ppo_trading_enhanced_logs/",
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=dict(
            lstm_hidden_size=64,
            n_lstm_layers=2,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )
    )
    
    print("Starting Training (200,000 steps)...")
    print("Features included: 6 original + 3 HTF + 1 regime + Sortino reward")
    # Add a custom save frequency or progress bar if needed
    model.learn(total_timesteps=200000)
    
    model_name = "models/ppo_trading_enhanced_v2"
    model.save(model_name)
    print(f"Saved enhanced model to {model_name}.zip")

if __name__ == "__main__":
    train()
