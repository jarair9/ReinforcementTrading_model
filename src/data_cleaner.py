import pandas as pd
import numpy as np
import os
import warnings

# Suppress division by zero warnings in ADX calculation
warnings.filterwarnings('ignore', message='invalid value encountered in divide')

def clean_eurusd_data(input_path, output_path, htf_input_path=None):
    print(f"Cleaning data: {input_path}")
    if not os.path.exists(input_path):
        print("Error: Input file not found.")
        return

    # Load primary (15m) data
    df = pd.read_csv(input_path)
    
    # Load higher timeframe data if provided
    df_htf = None
    if htf_input_path and os.path.exists(htf_input_path):
        print(f"Loading higher timeframe data: {htf_input_path}")
        df_htf = pd.read_csv(htf_input_path)
        
        # Standardize HTF data
        if 'Date' in df_htf.columns:
            df_htf['Datetime'] = pd.to_datetime(df_htf['Date'])
            df_htf.set_index('Datetime', inplace=True)
        
        # Standardize columns
        mapping_htf = {'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Tick_volume': 'Volume', 'Volume': 'Volume'}
        df_htf = df_htf.rename(columns=mapping_htf)
        
        # Scaling check for HTF data
        if df_htf['Close'].mean() > 10:
            print("Scaling HTF prices down (detecting Pip values)...")
            for col in ['Open', 'High', 'Low', 'Close']:
                df_htf[col] = df_htf[col] / 100000.0
                
        # Standardize timezone
        df_htf.index = pd.to_datetime(df_htf.index, utc=True)
        df_htf.sort_index(inplace=True)
        
        print(f"Loaded HTF data: {len(df_htf)} rows")
    
    # Handle Date/Time (MT4 export style)
    if 'Date' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'])
        df.set_index('Datetime', inplace=True)
    
    # Standardize columns
    mapping = {'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Tick_volume': 'Volume', 'Volume': 'Volume'}
    df = df.rename(columns=mapping)
    
    # Scaling Check: If prices are > 10, they are likely scaled (Pip values)
    if df['Close'].mean() > 10:
        print("Scaling prices down (detecting Pip values)...")
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col] / 100000.0
            
    # Standardize Timezone
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    # 2. Handle Gaps
    # Reindex to a complete 15-minute frequency
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min', tz='UTC')
    df = df.reindex(full_range)
    
    df['Close'] = df['Close'].ffill()
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Volume'] = df['Volume'].fillna(0)

    # --- FIXED FEATURE ENGINEERING (No Data Leakage) ---
    # 1. Price momentum (5-period return)
    df['feat_returns_5'] = df['Close'].pct_change(5)
    
# Debug: Check for problematic data
    print(f"\n--- Feature Calculation Debug ---")
    print(f"Price range: {df['Close'].min():.6f} to {df['Close'].max():.6f}")
    print(f"Volume range: {df['Volume'].min():.0f} to {df['Volume'].max():.0f}")
    
    # 2. Price position relative to SHIFTED 20-period range
    rolling_low = df['Low'].rolling(20).min().shift(1)
    rolling_high = df['High'].rolling(20).max().shift(1)
    range_diff = rolling_high - rolling_low
    
    # Debug range calculation
    print(f"Range stats: min={range_diff.min():.6f}, max={range_diff.max():.6f}")
    zero_ranges = (range_diff <= 1e-10).sum()
    print(f"Zero/near-zero ranges: {zero_ranges} out of {len(df)}")
    
    # Safe division with proper handling
    df['feat_price_pos'] = np.where(range_diff > 1e-10, 
                                   (df['Close'] - rolling_low) / range_diff, 
                                   0.5)  # Default to middle when range is zero
    
    # 3. Normalized volatility (Current range / Past ATR)
    past_atr = (df['High'] - df['Low']).rolling(20).mean().shift(1)
    current_range = df['High'] - df['Low']
    
    # Debug ATR calculation
    print(f"ATR stats: min={past_atr.min():.6f}, max={past_atr.max():.6f}")
    zero_atr = (past_atr <= 1e-10).sum()
    print(f"Zero/near-zero ATR: {zero_atr} out of {len(df)}")
    
    df['feat_norm_vol'] = np.where(past_atr > 1e-10,
                                  current_range / past_atr,
                                  1.0)  # Default to neutral volatility
    
    # 4. Time features (Cyclic)
    df['feat_hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['feat_hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
# 5. Volume spike (Relative to past 20 periods)
    past_avg_volume = df['Volume'].rolling(20).mean().shift(1)
    
    # Debug volume calculation
    print(f"Avg volume stats: min={past_avg_volume.min():.2f}, max={past_avg_volume.max():.2f}")
    zero_volume = (past_avg_volume <= 1e-10).sum()
    print(f"Zero/near-zero avg volume: {zero_volume} out of {len(df)}")
    
    df['feat_vol_spike'] = np.where(past_avg_volume > 1e-10,
                                   df['Volume'] / past_avg_volume,
                                   1.0)  # Default to neutral volume
    
    # 5.5 Relative range normalization (for slippage cost calculation)
    current_range = df['High'] - df['Low']
    past_avg_range = current_range.rolling(20).mean().shift(1)
    df['feat_rel_range_norm'] = np.where(past_avg_range > 1e-10,
                                       current_range / past_avg_range,
                                       1.0)  # Default to neutral range
    
    # 6. Higher Timeframe (HTF) Context
    # Use actual HTF data if available, otherwise create synthetic features
    
    if df_htf is not None:
        print("Using actual 4H data for HTF features")
        # Use actual 4H data
        df_4h = df_htf.copy()
        # Create synthetic 1H data from 15m
        df_1h = df.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).ffill()
    else:
        print("Creating synthetic HTF features from 15m data")
        # Fallback to synthetic features
        df_1h = df.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).ffill()
        
        df_4h = df.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).ffill()
    
    # Calculate HTF Trends using moving averages
    # 20-period SMA on 1H (Known at 15m intervals by shifting)
    sma_1h_20 = df_1h['Close'].rolling(20).mean().shift(1) 
    # 20-period SMA on 4H (Known at 15m intervals by shifting)
    sma_4h_20 = df_4h['Close'].rolling(20).mean().shift(1)
    
    # Calculate HTF RSI on 4H timeframe
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    rsi_4h = calculate_rsi(df_4h['Close']).shift(1)  # Shift to prevent leakage
    
    # Debug: Check SMA values
    print(f"\n--- HTF Debug Info ---")
    print(f"1H SMA range: {sma_1h_20.min():.6f} to {sma_1h_20.max():.6f}")
    print(f"4H SMA range: {sma_4h_20.min():.6f} to {sma_4h_20.max():.6f}")
    print(f"4H RSI range: {rsi_4h.min():.2f} to {rsi_4h.max():.2f}")
    print(f"Close price range: {df['Close'].min():.6f} to {df['Close'].max():.6f}")
    
    # Map back to 15m index using merge_asof for proper alignment
    df_temp = df[['Close']].copy()
    df_temp['datetime'] = df_temp.index
    
    # Merge with HTF data
    df_1h_with_sma = pd.DataFrame({'datetime_1h': sma_1h_20.index, 'sma_1h_20': sma_1h_20})
    df_4h_with_sma = pd.DataFrame({'datetime_4h': sma_4h_20.index, 'sma_4h_20': sma_4h_20})
    df_4h_with_rsi = pd.DataFrame({'datetime_4h_rsi': rsi_4h.index, 'rsi_4h': rsi_4h})
    
    # Forward fill to align timestamps
    df_1h_with_sma = df_1h_with_sma.ffill()
    df_4h_with_sma = df_4h_with_sma.ffill()
    df_4h_with_rsi = df_4h_with_rsi.ffill()
    
    # Merge using forward search
    df_merged = pd.merge_asof(df_temp.sort_values('datetime'), 
                             df_1h_with_sma.sort_values('datetime_1h'), 
                             left_on='datetime', right_on='datetime_1h', 
                             direction='backward')
    
    df_merged = pd.merge_asof(df_merged.sort_values('datetime'), 
                             df_4h_with_sma.sort_values('datetime_4h'), 
                             left_on='datetime', right_on='datetime_4h', 
                             direction='backward')
    
    df_merged = pd.merge_asof(df_merged.sort_values('datetime'), 
                             df_4h_with_rsi.sort_values('datetime_4h_rsi'), 
                             left_on='datetime', right_on='datetime_4h_rsi', 
                             direction='backward')
    
    # Calculate HTF features
    df['htf_1h_trend'] = np.where(df_merged['Close'] > df_merged['sma_1h_20'], 1.0, -1.0)
    df['htf_4h_trend'] = np.where(df_merged['Close'] > df_merged['sma_4h_20'], 1.0, -1.0)
    df['htf_4h_rsi'] = np.where(pd.notna(df_merged['rsi_4h']), (df_merged['rsi_4h'] - 50) / 50, 0.0)
    
    # Debug: Show sample results
    print(f"\nSample HTF features:")
    print(df[['htf_1h_trend', 'htf_4h_trend', 'htf_4h_rsi']].head(10))
    print(f"\nHTF feature distributions:")
    print(f"htf_1h_trend: {df['htf_1h_trend'].value_counts().to_dict()}")
    print(f"htf_4h_trend: {df['htf_4h_trend'].value_counts().to_dict()}")
    print(f"htf_4h_rsi non-zero: {(df['htf_4h_rsi'] != 0).sum()} out of {len(df)}")
    
    # 7. Market Regime Detection (ADX-based)
    def calculate_adx(high, low, close, window=14):
        # Convert to pandas Series if needed
        if isinstance(high, np.ndarray):
            high = pd.Series(high, index=close.index)
        if isinstance(low, np.ndarray):
            low = pd.Series(low, index=close.index)
        if isinstance(close, np.ndarray):
            close = pd.Series(close, index=close.index)
            
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        true_range = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3)), index=close.index)
        
        # Calculate Directional Movements
        dm_plus = pd.Series(np.where((high - high.shift()) > (low.shift() - low), high - high.shift(), 0), index=close.index)
        dm_minus = pd.Series(np.where((low.shift() - low) > (high - high.shift()), low.shift() - low, 0), index=close.index)
        
        # Smooth using Wilder's smoothing (EMA-like)
        def wilders_smoothing(series, period):
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[period-1] = series.iloc[:period].mean()
            for i in range(period, len(series)):
                result.iloc[i] = (result.iloc[i-1] * (period-1) + series.iloc[i]) / period
            return result
        
        tr_smooth = wilders_smoothing(true_range, window)
        dm_plus_smooth = wilders_smoothing(dm_plus, window)
        dm_minus_smooth = wilders_smoothing(dm_minus, window)
        
        # Calculate DI (handle division by zero)
        di_plus = np.where(tr_smooth > 0, (dm_plus_smooth / tr_smooth) * 100, 0)
        di_minus = np.where(tr_smooth > 0, (dm_minus_smooth / tr_smooth) * 100, 0)
        
        # Calculate DX and ADX (handle division by zero)
        dx_numerator = np.abs(di_plus - di_minus)
        dx_denominator = di_plus + di_minus
        dx = np.where(dx_denominator > 0, (dx_numerator / dx_denominator) * 100, 0)
        
        # ADX using Wilder's smoothing
        adx = wilders_smoothing(pd.Series(dx, index=close.index), window)
        
        return adx.fillna(0)
    
    df['feat_adx'] = calculate_adx(df['High'], df['Low'], df['Close'], 14)
    
    # Debug: Check ADX distribution
    print(f"\n--- Regime Debug Info ---")
    print(f"ADX range: {df['feat_adx'].min():.2f} to {df['feat_adx'].max():.2f}")
    print(f"ADX mean: {df['feat_adx'].mean():.2f}")
    print(f"ADX > 25 count: {(df['feat_adx'] > 25).sum()} out of {len(df)}")
    print(f"ADX > 20 count: {(df['feat_adx'] > 20).sum()} out of {len(df)}")
    
    # Adjust threshold to get better regime distribution
    regime_threshold = min(25, df['feat_adx'].quantile(0.7))  # Use 70th percentile if 25 is too high
    print(f"Using regime threshold: {regime_threshold:.2f}")
    
    df['feat_regime'] = np.where(df['feat_adx'] > regime_threshold, 1.0, -1.0)  # 1 for trending, -1 for ranging
    
    # Debug: Show regime distribution
    print(f"Regime distribution: {df['feat_regime'].value_counts().to_dict()}")
    
    # 4. Cleanup & Clipping (+/- 3 standard deviations)
    df.dropna(inplace=True) 
    
    feature_cols = ['feat_returns_5', 'feat_price_pos', 'feat_norm_vol', 
                    'feat_hour_sin', 'feat_hour_cos', 'feat_vol_spike',
                    'htf_1h_trend', 'htf_4h_trend', 'htf_4h_rsi',
                    'feat_regime', 'feat_rel_range_norm']
    
    # Print feature statistics
    print("\n--- Feature Statistics ---")
    for col in feature_cols:
        if col in df.columns:
            print(f"{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, min={df[col].min():.4f}, max={df[col].max():.4f}")
        else:
            print(f"WARNING: {col} not found in dataframe")
    
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = np.clip(df[col], mean - 3*std, mean + 3*std)

    # Selected Columns for Output
    final_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + feature_cols
    df = df[final_cols]
    
    if len(df) == 0:
        print("CRITICAL: After cleaning, 0 rows remain. Check dataset size.")
    else:
        df.to_csv(output_path)
        print(f"Cleaned data saved to {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    clean_eurusd_data("data/EURUSD_15m_real.csv", "data/EURUSD_15m_cleaned.csv", "data/EURUSD_4h_real.csv")
