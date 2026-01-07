from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_env import SimpleTradingEnv
import pandas as pd
import torch
import os
import time

# Fix Gym warning - use Gymnasium instead
import gymnasium as gym
import gym  # Keep for backward compatibility


class TrainingProgressCallback:
    """Custom callback to track and display training progress"""
    def __init__(self, total_timesteps):
        self.total_timesteps = total_timesteps
        self.start_time = time.time()
        self.last_print_time = self.start_time
        
    def __call__(self, locals, globals):
        # Get current timestep from the model
        model = locals.get('self')
        if model is not None:
            current_timestep = getattr(model, 'num_timesteps', 0)
            
            # Print progress every 10 seconds or at key milestones
            current_time = time.time()
            if current_time - self.last_print_time >= 10 or current_timestep >= self.total_timesteps:
                progress_percent = (current_timestep / self.total_timesteps) * 100
                
                # Create a simple progress bar
                bar_length = 50
                filled_length = int(bar_length * progress_percent // 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                # Print progress bar on a new line
                print(f"\n[{bar}] {progress_percent:.1f}% ({current_timestep:,}/{self.total_timesteps:,})")
                
                self.last_print_time = current_time
                
        return True  # Continue training

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
    # Implements Multi-Timeframe Context, Market Regime Detection
    model = PPO(
        "MlpPolicy",  # Using standard MLP policy (LSTM not available in this SB3 version)
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
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )
    )
    
    print("Starting Training (200,000 steps)...")
    print("Features included: 6 original + 3 HTF + 1 regime + Sortino reward")
    
    # Create progress callback
    progress_callback = TrainingProgressCallback(200000)
    
    # Train with progress tracking
    print("Training Progress:")
    model.learn(total_timesteps=200000, callback=progress_callback)
    
    # Print completion message
    print(f"\nTraining completed! 100.0% (200,000/200,000)")
    
    model_name = "models/ppo_trading_enhanced_v2"
    model.save(model_name)
    print(f"Saved enhanced model to {model_name}.zip")

if __name__ == "__main__":
    train()
