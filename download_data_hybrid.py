import yfinance as yf
import pandas as pd
import os

def download_hybrid_data():
    if not os.path.exists("data"):
        os.makedirs("data")
        
    # Pairs (Yahoo Tickers)
    # Euro, GBP, JPY, Aussie, CAD, Swiss, Kiwi, Gold
    pairs = [
        "EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", 
        "CAD=X", "CHF=X", "NZDUSD=X", "GC=F"
    ]
    
    # Clean Names Mapping
    name_map = {
        "EURUSD=X": "EURUSD",
        "GBPUSD=X": "GBPUSD",
        "JPY=X": "USDJPY",
        "AUDUSD=X": "AUDUSD",
        "CAD=X": "USDCAD",
        "CHF=X": "USDCHF",
        "NZDUSD=X": "NZDUSD",
        "GC=F": "XAUUSD"
    }

    print("--- HYBRID DATA STRATEGY DOWNLOAD ---")
    
    # 1. Daily Data (5 Years)
    print("\n[1/3] Downloading 5 Years Daily Data...")
    for p in pairs:
        clean_name = name_map[p]
        try:
            df = yf.download(p, period="5y", interval="1d", progress=False)
            if not df.empty:
                # Cleanup
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                # Remove Timezone
                if df.index.tz is not None:
                     df.index = df.index.tz_localize(None)
                     
                filename = f"data/{clean_name}_1d.csv"
                df.to_csv(filename)
                print(f"Saved {filename} ({len(df)} rows)")
            else:
                print(f"Warning: No Daily data for {p}")
        except Exception as e:
            print(f"Error {p} 1d: {e}")

    # 2. Hourly Data (2 Years - Max)
    print("\n[2/3] Downloading 2 Years Hourly Data...")
    for p in pairs:
        clean_name = name_map[p]
        try:
            # 730d is max for 1h
            df = yf.download(p, period="730d", interval="1h", progress=False)
            if not df.empty:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                if df.index.tz is not None:
                     df.index = df.index.tz_localize(None)
                
                filename = f"data/{clean_name}_1h.csv"
                df.to_csv(filename)
                print(f"Saved {filename} ({len(df)} rows)")
            else:
                print(f"Warning: No Hourly data for {p}")
        except Exception as e:
            print(f"Error {p} 1h: {e}")

    # 3. 15-Minute Data (60 Days - Max)
    print("\n[3/3] Downloading 60 Days 15-Min Data...")
    for p in pairs:
        clean_name = name_map[p]
        try:
            # 60d is reliable max for 15m/5m often
            df = yf.download(p, period="60d", interval="15m", progress=False)
            if not df.empty:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                if df.index.tz is not None:
                     df.index = df.index.tz_localize(None)
                
                filename = f"data/{clean_name}_15m.csv"
                df.to_csv(filename)
                print(f"Saved {filename} ({len(df)} rows)")
            else:
                print(f"Warning: No 15m data for {p}")
        except Exception as e:
            print(f"Error {p} 15m: {e}")
            
    print("\nDownload Complete.")

if __name__ == "__main__":
    download_hybrid_data()
