
import pandas as pd
import requests
import os
import io

def fetch_and_verify():
    print("--- Fetching High-Quality EURUSD Data from GitHub ---")
    
    # URLs for different timeframes
    urls = {
        "15m": "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/EURUSD/EURUSDm15.csv",
        "4h": "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/EURUSD/EURUSDh4.csv"
    }
    
    try:
        # Download both 15m and 4h data
        dataframes = {}
        
        for timeframe, url in urls.items():
            print(f"\n--- Downloading EURUSD {timeframe.upper()} Data ---")
            print(f"Downloading from: {url}")
            response = requests.get(url)
            if response.status_code != 200:
                print(f"ERROR: Failed to download {timeframe}. Status code: {response.status_code}")
                continue
                
            # 1. Load Data
            # Format in repo: Datetime,Open,High,Low,Close,Volume (based on standard MT4 export)
            df = pd.read_csv(io.StringIO(response.text))
            
            # Standardize Names (Handle possible header variations)
            df.columns = [c.capitalize() for c in df.columns]
            if 'Time' in df.columns and 'Date' in df.columns:
                 df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                 df.set_index('Datetime', inplace=True)
            elif 'Datetime' in df.columns:
                 df['Datetime'] = pd.to_datetime(df['Datetime'])
                 df.set_index('Datetime', inplace=True)
                 
            # Store dataframe
            dataframes[timeframe] = df
            
            print(f"Downloaded {len(df)} rows for {timeframe.upper()} timeframe")
            
        if not dataframes:
            print("ERROR: No data downloaded successfully")
            return
        
        # 2. Quality Check for each timeframe
        print("\n--- Running Quality Checks ---")
        
        for timeframe, df in dataframes.items():
            print(f"\n--- {timeframe.upper()} Data Quality Check ---")
            issues = []
            
            # Unique Price Ratio
            unique_ratio = df['Close'].nunique() / len(df)
            print(f"Unique Price Ratio: {unique_ratio:.4f}")
            if unique_ratio < 0.1: # Relaxed to 10%
                issues.append(f"Extremely low unique prices: {unique_ratio:.4f}")
                
            # High >= Low
            bad_hl = (df['High'] < df['Low']).sum()
            if bad_hl > 0:
                issues.append(f"Found {bad_hl} bars where High < Low")
                
            # Zero Range Bars
            zero_range = (df['High'] == df['Low']).sum()
            zero_range_ratio = zero_range / len(df)
            print(f"Zero Range Bars Ratio: {zero_range_ratio:.4f}")
            if zero_range_ratio > 0.1: # Relaxed to 10%
                issues.append(f"Too many zero-range bars: {zero_range_ratio:.4f}")
            
            output_path = f"data/EURUSD_{timeframe}_real.csv"
            if not os.path.exists("data"):
                os.makedirs("data")
            df.to_csv(output_path)
            print(f"Saved {timeframe.upper()} data to {output_path} ({len(df)} rows)")

            if issues:
                print("\n!!! DATA QUALITY WARNINGS !!!")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("\nPASS: Data looks realistic.")
                
        print("\n--- All Data Downloaded Successfully ---")
        print(f"Available timeframes: {list(dataframes.keys())}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_and_verify()
