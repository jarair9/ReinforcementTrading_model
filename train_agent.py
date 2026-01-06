import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback

from indicators import load_and_preprocess_data, add_correlation_features
from trading_env_advanced import AdversarialTradingEnv
from custom_policy import ResNetFeatureExtractor


def main():
    print("--- 3-PAIR HEDGE FUND AGENT TRAINING (1H ONLY) ---")
    
    # 1. DEFINE PAIRS
    pairs = ["EURUSD", "GBPUSD", "XAUUSD"]
    
    # 2. LOAD 1H DATA
    print("\n[INIT] Loading 1H Data Assets...")
    data_1h = {}
    
    for p in pairs:
        try:
            path = f"data/{p}_1h.csv"
            data_1h[p] = load_and_preprocess_data(path)
            print(f"Loaded {p}: 1h ({len(data_1h[p])})")
        except Exception as e:
            print(f"CRITICAL: Failed to load {p} from {path}. Error: {e}")

    if len(data_1h) < len(pairs):
        print(f"CRITICAL: Only {len(data_1h)}/{len(pairs)} pairs loaded. Cannot proceed.")
        return

    # Add Correlations
    print("Computing Cross-Pair Correlations...")
    add_correlation_features(data_1h)

    # --- TRAINING (1H ONLY) ---
    print("\n=================================================")
    print("TRAINING SESSION: 1H MULTI-ASSET PORTFOLIO")
    print("Pairs:", pairs)
    print("=================================================")
    
    # Initialize Environment with 1H data only
    # data_dict is the primary timeframe.
    # We pass None for secondary/tertiary to keep it 1H-only.
    env = AdversarialTradingEnv(data_dict=data_1h, 
                                data_dict_1h=None, 
                                data_dict_1d=None, 
                                window_size=30)
    
    vec_env = DummyVecEnv([lambda: env])
    
    # --- AUTOMATIC DEVICE DETECTION (GPU/CPU) ---
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected: {device.upper()}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Configure Custom Policy
    policy_kwargs = dict(
        features_extractor_class=ResNetFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512, window_size=30, n_pairs=len(pairs)),
        net_arch=[1024, 1024, 512, 512]
    )
    
    # Initialize SAC Model
    model = SAC("MlpPolicy", vec_env, verbose=1, 
                device=device,
                tensorboard_log="./tensorboard_log_3pair_1h/",
                learning_rate=0.0001, 
                buffer_size=100000, 
                batch_size=512,
                policy_kwargs=policy_kwargs)
    
    print("Starting Training (100,000 Steps)...")
    model.learn(total_timesteps=100000, callback=ProgressBarCallback())
    
    model_path = "model_3pair_1h_final"
    model.save(model_path)
    print(f"\nTraining Complete. Model saved to '{model_path}.zip'")
    print("Done.")

if __name__ == "__main__":
    main()