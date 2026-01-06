import pandas as pd
import numpy as np

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Loads data and adds stationary/normalized features for RL.
    Calculates indicators manually to avoid dependencies like pandas_ta.
    """
    # Detect format and load
    try:
        # Check first line
        with open(csv_path, 'r') as f:
            first_line = f.readline()
        
        if "Price" in first_line:
            # yfinance multi-header format
            # Skip 3 lines (Headers), read data. 
            # Columns: Index, Open, High, Low, Close, Volume
            df = pd.read_csv(csv_path, skiprows=3, header=None)
            df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            df.set_index('Datetime', inplace=True)
        else:
            # Standard Format
            df = pd.read_csv(csv_path)
            if 'Datetime' in df.columns:
                df.set_index('Datetime', inplace=True)
            elif 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            elif 'Gmt time' in df.columns: # Dukascopy
                 df['Datetime'] = pd.to_datetime(df['Gmt time'], format="%d.%m.%Y %H:%M:%S.%f")
                 df.set_index('Datetime', inplace=True)
                 # Drop other cols if needed
                 
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        raise e

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    
    # Ensure numeric
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df.dropna(inplace=True)

    
    # --- Feature Engineering ---
    
    # 1. Log Returns (Stationary Price Movement)
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. Relative Indicators
    
    # Simple Moving Averages
    sma_20 = df['Close'].rolling(window=20).mean()
    df['ma_20'] = sma_20 # Assign to column
    sma_50 = df['Close'].rolling(window=50).mean()
    
    # Distance from MA (Trend strength relative to price)
    df['ma_dist_20'] = (df['Close'] - sma_20) / df['Close']
    df['ma_dist_50'] = (df['Close'] - sma_50) / df['Close']
    
    # Bollinger Bands Width
    std_20 = df['Close'].rolling(window=20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    # Width relative to Mid (SMA)
    df['bb_width'] = (bb_upper - bb_lower) / sma_20
    
    # MACD (12, 26, 9)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_diff = macd_line - signal_line
    
    # Normalize MACD by price
    df['macd_diff'] = macd_diff / df['Close']
    df['macd_line'] = macd_line / df['Close']

    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # Avoid zero division
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Normalize RSI to 0-1
    df['rsi'] = rsi / 100.0

    # ATR (14)
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Normalize ATR
    df['atr'] = atr / df['Close']

    # 3. Volume Features (NEW)
    # Log Volume Change (Stationary)
    # Avoid zero volume log error
    df['Volume'] = df['Volume'].replace(0, 1)
    df['log_vol_ret'] = np.log(df['Volume'] / df['Volume'].shift(1))
    
    # Relative Volume (Current Vol vs Moving Avg Vol)
    vol_sma_20 = df['Volume'].rolling(window=20).mean()
    df['rel_vol_20'] = df['Volume'] / vol_sma_20
    
    # VWAP-like feature (Volume Force)
    # Normed by price to keep scale small
    df['vol_force'] = (df['log_ret'] * df['Volume']) / vol_sma_20

    # 4. Time Features (Cyclic Encoding)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    df['ma_20_slope'] = df['ma_20'].diff()
    
    # Sentiment
    if 'sentiment' not in df.columns:
        df['sentiment'] = 0.0
    
    # Drop NaNs
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

def add_correlation_features(data_dict: dict, window=30):
    """
    Computes rolling correlations between all pairs in the dictionary.
    Modifies DataFrames in-place.
    
    Args:
        data_dict: { 'PAIR_NAME': pd.DataFrame, ... }
    """
    pairs = list(data_dict.keys())
    
    # Create a unified DataFrame of returns
    # We need to ensure alignment
    # Strategy: Merge all on index
    
    # 1. Extract Log Returns
    returns_df = pd.DataFrame()
    for pair, df in data_dict.items():
        if 'log_ret' in df.columns:
            returns_df[pair] = df['log_ret']
            
    # 2. Compute Rolling Correlations
    # For each pair, compute corr with every other pair
    for i, pair_a in enumerate(pairs):
        for pair_b in pairs[i+1:]:
            # Rolling Corr
            corr_series = returns_df[pair_a].rolling(window=window).corr(returns_df[pair_b])
            
            # Add to BOTH DataFrames
            col_name = f"corr_{pair_b}"
            data_dict[pair_a][col_name] = corr_series
            
            col_name_b = f"corr_{pair_a}"
            data_dict[pair_b][col_name_b] = corr_series
            
    # Drop NaNs generated by rolling
    for pair in pairs:
        data_dict[pair].dropna(inplace=True)
