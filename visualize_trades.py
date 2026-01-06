import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import os

from trading_env_advanced import AdversarialTradingEnv
from indicators import load_and_preprocess_data, add_correlation_features

def run_simulation(model_path, pairs):
    # 1. Load Data
    data_1h = {}
    print(f"Loading data for {pairs}...")
    for p in pairs:
        data_1h[p] = load_and_preprocess_data(f"data/{p}_1h.csv")
    
    add_correlation_features(data_1h)
    
    # 2. Setup Env
    env = AdversarialTradingEnv(data_dict=data_1h, window_size=30)
    vec_env = DummyVecEnv([lambda: env])
    
    # 3. Load Model
    print(f"Loading model {model_path}...")
    model = SAC.load(model_path)
    
    # 4. Run Step-by-Step and Log
    obs = vec_env.reset()
    done = False
    
    history = []
    equity_curve = []
    
    print("Simulating trades...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Action is weights for all pairs
        
        # Save state before step
        current_step = env.current_step
        prices = {p: env.price_arrays[p][current_step] for p in pairs}
        positions = env.positions.copy()
        
        obs, reward, done, info = vec_env.step(action)
        
        # Save state after step
        equity = info[0]['equity']
        new_positions = env.positions.copy()
        
        history.append({
            'step': current_step,
            'prices': prices,
            'prev_positions': positions,
            'new_positions': new_positions,
            'equity': equity
        })
        equity_curve.append(equity)
        
    return history, data_1h

def plot_pair(pair_name, history, data_dict):
    df = data_dict[pair_name].copy()
    
    # Extract trade signals from history
    steps = [h['step'] for h in history]
    # Map history to df indices. Steps in history are index in price_arrays.
    # Note: feature_matrices/price_arrays might have different length than original CSV due to dropna.
    # But env.current_step is absolute index into self.price_arrays.
    
    # Create Trade Markers
    buy_signals = []
    sell_signals = []
    exit_signals = []
    
    pair_idx = list(data_dict.keys()).index(pair_name)
    
    for i in range(1, len(history)):
        prev_pos = history[i-1]['new_positions'][pair_idx]
        curr_pos = history[i]['new_positions'][pair_idx]
        price = history[i]['prices'][pair_name]
        timestamp = df.index[history[i]['step']]
        
        if curr_pos > 0 and prev_pos <= 0:
            buy_signals.append((timestamp, price))
        elif curr_pos < 0 and prev_pos >= 0:
            sell_signals.append((timestamp, price))
        elif abs(curr_pos) < 0.001 and abs(prev_pos) > 0.001:
            exit_signals.append((timestamp, price))

    # --- PLOTLY DASHBOARD ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        subplot_titles=(f'{pair_name} Price & Trades', 'Indicators (RSI/MACD)', 'Portfolio Equity'),
                        row_heights=[0.6, 0.2, 0.2])

    # 1. Price & Trades
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    # MA
    if 'ma_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ma_20'], line=dict(color='rgba(255,165,0,0.5)', width=1), name='SMA 20'), row=1, col=1)

    # Buy Markers
    if buy_signals:
        b_x, b_y = zip(*buy_signals)
        fig.add_trace(go.Scatter(x=b_x, y=b_y, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='BUY 入场'), row=1, col=1)
    
    # Sell Markers
    if sell_signals:
        s_x, s_y = zip(*sell_signals)
        fig.add_trace(go.Scatter(x=s_x, y=s_y, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='SELL 入场'), row=1, col=1)
        
    # Exit Markers
    if exit_signals:
        e_x, e_y = zip(*exit_signals)
        fig.add_trace(go.Scatter(x=e_x, y=e_y, mode='markers', marker=dict(symbol='x', size=8, color='white', line=dict(width=1, color='black')), name='EXIT 出场'), row=1, col=1)

    # 2. Indicators
    if 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi']*100, line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # 3. Equity Curve
    equity_timestamps = [df.index[h['step']] for h in history]
    equity_values = [h['equity'] for h in history]
    fig.add_trace(go.Scatter(x=equity_timestamps, y=equity_values, line=dict(color='cyan'), fill='tozeroy', name='Equity $'), row=3, col=1)

    # Layout
    fig.update_layout(height=900, title_text=f"AI Trader Analysis: {pair_name}", template='plotly_dark', showlegend=True,
                      xaxis_rangeslider_visible=False)
    
    output_file = f"trade_analysis_{pair_name}.html"
    fig.write_html(output_file)
    print(f"Analysis saved to {output_file}")
    
    # Try to open
    try:
        os.startfile(output_file)
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="EURUSD", choices=["EURUSD", "GBPUSD", "XAUUSD"])
    parser.add_argument("--model", type=str, default="model_3pair_1h_final")
    args = parser.parse_args()
    
    pairs = ["EURUSD", "GBPUSD", "XAUUSD"]
    history, data_dict = run_simulation(args.model, pairs)
    plot_pair(args.pair, history, data_dict)
