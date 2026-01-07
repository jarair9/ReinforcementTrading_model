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
                elapsed_time = current_time - self.start_time
                progress_percent = (current_timestep / self.total_timesteps) * 100
                
                # Estimate remaining time
                if progress_percent > 0:
                    estimated_total_time = elapsed_time / (progress_percent / 100)
                    remaining_time = estimated_total_time - elapsed_time
                else:
                    remaining_time = 0
                
                # Format time nicely
                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.0f}s"
                    elif seconds < 3600:
                        return f"{seconds/60:.1f}m"
                    else:
                        return f"{seconds/3600:.1f}h"
                
                print(f"\rProgress: {progress_percent:.1f}% ({current_timestep:,}/{self.total_timesteps:,} steps) | "
                      f"Elapsed: {format_time(elapsed_time)} | "
                      f"Remaining: {format_time(remaining_time)} | "
                      f"Speed: {current_timestep/elapsed_time:.0f} steps/sec", end="", flush=True)
                
                self.last_print_time = current_time
                
                # Print newline at completion
                if current_timestep >= self.total_timesteps:
                    print()  # New line at the end
        
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
    
    model_name = "models/ppo_trading_enhanced_v2"
    model.save(model_name)
    print(f"Saved enhanced model to {model_name}.zip")

if __name__ == "__main__":
    train()
