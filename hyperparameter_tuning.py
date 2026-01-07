import optuna
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_env import SimpleTradingEnv
import os


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization
    """
    data_path = "data/EURUSD_15m_cleaned.csv"
    if not os.path.exists(data_path):
        print("Run clean_data.py first.")
        return -100  # Return a bad score if data not found
    
    df = pd.read_csv(data_path)
    # Use first 70% for training, 20% for validation
    split = int(len(df) * 0.7)
    train_df = df.iloc[:split]
    val_df = df.iloc[split:split+int(len(df)*0.2)]
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-5, 1e-1)
    vf_coef = trial.suggest_uniform('vf_coef', 0.1, 0.9)
    
    # Create environment
    env = SimpleTradingEnv(train_df)
    env = DummyVecEnv([lambda: env])
    
    try:
        # Create model with suggested hyperparameters
        model = PPO(
            "MlpLstmPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=0,
            tensorboard_log="logs/optuna_tuning/",
            policy_kwargs=dict(
                lstm_hidden_size=64,
                n_lstm_layers=2,
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
            )
        )
        
        # Train for fewer steps for faster optimization
        model.learn(total_timesteps=min(10000, n_steps * 5))
        
        # Evaluate on validation set
        val_env = SimpleTradingEnv(val_df)
        val_env = DummyVecEnv([lambda: val_env])
        
        obs = val_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = val_env.step(action)
            total_reward += reward[0]
        
        # Return the total reward as the optimization target
        return total_reward
    
    except Exception as e:
        print(f"Error during trial: {e}")
        return -100  # Return a bad score in case of error


def run_hyperparameter_tuning():
    """
    Run the hyperparameter tuning process
    """
    print("Starting hyperparameter tuning with Optuna...")
    
    # Create study object
    study = optuna.create_study(direction='maximize')
    
    # Run optimization
    study.optimize(objective, n_trials=20)  # Reduced trials for initial implementation
    
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    
    # Save best parameters to a file
    with open("best_hyperparameters.txt", "w") as f:
        f.write(f"Best parameters: {study.best_params}\n")
        f.write(f"Best value: {study.best_value}\n")
    
    return study.best_params


if __name__ == "__main__":
    best_params = run_hyperparameter_tuning()
    print("Hyperparameter tuning completed!")
    print("Best parameters saved to best_hyperparameters.txt")