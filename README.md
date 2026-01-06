# Hedge Fund AI Agent - Reinforcement Learning Trader

An advanced multi-asset trading agent powered by Reinforcement Learning (SAC) and a deep ResNet (Residual Network) architecture. This system is designed to manage a portfolio of three major pairs (**EURUSD**, **GBPUSD**, **XAUUSD**) using the **1-hour (1H)** timeframe.

## 🚀 Key Features

- **Multi-Asset Portfolio Management**: Simultaneously manages positions across three instruments to optimize risk and correlation.
- **Deep ResNet Architecture**: Uses a 6-block deep Residual Network to extract complex patterns from time-series data.
- **Adversarial Environment**: Features an `AdversarialTradingEnv` that simulates real-world slippage and spread widening to stress-test strategies.
- **Interactive Visualization**: Built-in Plotly-based visualizer to see exactly where the model buys and sells on a TradingView-style chart.
- **GPU Accelerated**: Automatically detects and utilizes CUDA for deep learning training.

## 🛠️ Tech Stack

- **Brain**: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (Soft Actor-Critic algorithm)
- **Environment**: Custom Gymnasium-compatible environment
- **Deep Learning**: PyTorch (ResNet configuration)
- **Data Engineering**: Pandas, NumPy
- **Graphics**: Plotly (Interactive Dashboards)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jarair9/ReinforcementTrading_model.git
   cd ReinforcementTrading_model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage Workflow

### 1. Training the Model
Run the main training script. It will load 1H data for EURUSD, GBPUSD, and XAUUSD, initialize the deep ResNet policy, and start the SAC learning process.
```bash
python train_agent.py
```

### 2. Monitoring Progress
You can monitor the training progress in real-time using Tensorboard:
```bash
tensorboard --logdir ./tensorboard_log_3pair_1h/
```

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

## ⚠️ Disclaimer
This project is for educational purposes only. Reinforcement learning in financial markets is highly experimental. Trading involves significant risk of loss.
