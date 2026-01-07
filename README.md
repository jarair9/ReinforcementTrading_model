# Hedge Fund AI Agent - Reinforcement Learning Trader

An advanced multi-asset trading agent powered by Reinforcement Learning (SAC) and a deep ResNet (Residual Network) architecture. This system is designed to manage a portfolio of three major pairs (**EURUSD**, **GBPUSD**, **XAUUSD**) using the **1-hour (1H)** timeframe.

## 🚀 Key Features

- **Multi-Asset Portfolio Management**: Simultaneously manages positions across three instruments to optimize risk and correlation.
- **Deep ResNet Architecture**: Uses a 6-block deep Residual Network to extract complex patterns from time-series data.
- **Adversarial Environment**: Features an `AdversarialTradingEnv` that simulates real-world slippage and spread widening to stress-test strategies.
- **Interactive Visualization**: Built-in Plotly-based visualizer to see exactly where the model buys and sells on a TradingView-style chart.
- **GPU Accelerated**: Automatically detects and utilizes CUDA for deep learning training.
- **Multi-Timeframe Context**: Incorporates 1H and 4H trend indicators and RSI for "Big Picture" context.
- **Market Regime Detection**: ADX-based regime detection to identify trending vs. ranging markets.
- **Advanced Reward Functions**: Sortino ratio component for optimizing smooth equity curves.
- **Sequence-Aware Architecture**: LSTM-based policy for memory and pattern recognition over time.
- **Automated Hyperparameter Tuning**: Optuna-based optimization for finding optimal parameters.
- **Multi-Asset Integration**: Support for correlated assets like DXY or GBPUSD for confirmation signals.

## 🛠️ Tech Stack

- **Brain**: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (Soft Actor-Critic algorithm)
- **Environment**: Custom Gymnasium-compatible environment
- **Deep Learning**: PyTorch (ResNet configuration)
- **Data Engineering**: Pandas, NumPy
- **Graphics**: Plotly (Interactive Dashboards)

# 🚀 EURUSD RL Trading Project (Refined Kindergarten Phase)

A simplified, robust reinforcement learning trading system for EURUSD 15m data, focusing on zero data leakage and extreme risk management.

## 📁 Project Structure

- **`data/`**: Raw and cleaned datasets.
- **`models/`**: Saved training checkpoints (.zip).
- **`logs/`**: Tensorboard logs for monitoring training performance.
- **`src/`**: Core logic and scripts.
  - `data_fetcher.py`: Downloads high-quality historical data from GitHub.
  - `data_cleaner.py`: Processes and encodes 9 non-leaking features (6 original + 3 HTF).
  - `data_diagnostics.py`: Checks for target/feature leakage.
  - `trading_env.py`: Gym-compatible trading environment with risk circuit breakers.
  - `multi_asset_env.py`: Multi-asset trading environment with correlated asset features.
  - `supervised_model.py`: Random Forest / XGBoost baseline (Target: >52% accuracy).
  - `rl_model.py`: PPO trainer with LSTM policy and enhanced features.
  - `benchmarks.py`: Standard trading benchmarks (SMA Crossover, B&H).
  - `analysis.py`: Predictability heatmaps and win/loss statistics.
- **`hyperparameter_tuning.py`**: Optuna-based automated hyperparameter optimization.
- **`run_improved_system.py`**: Script to run the complete improved system.

## 🛠️ Execution Sequence

1. **Setup**:
   ```bash
   pip install -r Requirements.txt
   ```

2. **Data Acquisition**:
   ```bash
   python src/data_fetcher.py
   python src/data_cleaner.py
   ```

3. **Validation**:
   ```bash
   python src/data_diagnostics.py
   python src/supervised_model.py
   ```

4. **Training**:
   ```bash
   python src/rl_model.py
   # Or run the complete improved system:
   python run_improved_system.py
   ```

5. **Hyperparameter Tuning** (Optional):
   ```bash
   python hyperparameter_tuning.py
   ```

6. **Analysis**:
   ```bash
   tensorboard --logdir logs/
   python src/analysis.py
   ```

---
**Note**: This project uses a strictly non-leaking feature engineering system to ensure real predictive power.

### 3. Interactive Analysis (TradingView Style)
After training, generate a visual report for a specific pair to see the model's performance:
```bash
python visualize_trades.py --pair EURUSD
```
This will open an interactive HTML chart (`trade_analysis_EURUSD.html`) in your browser.

### 4. Evaluation Metrics
Run the evaluation script to get a "Report Card" with Win Rate, Sharpe Ratio, and Max Drawdown:
```bash
python evaluate_model.py
```

## 🧠 Brain Architecture (ResNet)

The model's observation space includes:
- **30-hour Lookback**: Price action and volume.
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and ATR.
- **Cross-Pair Correlations**: Rolling correlation features between assets.
- **Cyclic Time Features**: Sine/Cosine encoding for Hour and Day.
- **Higher Timeframe Features**: 1H and 4H trend indicators and RSI.
- **Market Regime Features**: ADX-based regime detection (trending vs. ranging).

## ⚠️ Disclaimer
This project is for educational purposes only. Reinforcement learning in financial markets is highly experimental. Trading involves significant risk of loss.
