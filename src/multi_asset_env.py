import gym
import numpy as np
from gym import spaces
import pandas as pd


class MultiAssetTradingEnv(gym.Env):
    """
    Multi-Asset Trading Environment:
    - Multiple Assets: EURUSD, GBPUSD, XAUUSD
    - Correlated Asset Features for confirmation signals
    - 3 Actions per asset: 0 (Close/Flat), 1 (Buy), 2 (Sell)
    - Risk Mgmt: Drawdown circuit breaker, daily loss limit.
    - Rewards: Spread + Commission, Asymmetric penalty, Time decay.
    """
    def __init__(self, df_dict, initial_balance=10000.0, spread=0.0001, commission=0.0001):
        super(MultiAssetTradingEnv, self).__init__()
        
        self.df_dict = df_dict  # Dictionary with asset names as keys and dataframes as values
        self.initial_balance = initial_balance
        self.base_cost = spread + commission
        self.assets = list(df_dict.keys())
        self.n_assets = len(self.assets)
        
        # Get the asset with the shortest length to avoid index errors
        self.max_steps = min([len(df) for df in df_dict.values()]) - 1
        
        # Calculate ATR for stop-loss (5x ATR) for each asset
        self.atr_dict = {}
        for asset, df in self.df_dict.items():
            tr = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low'] - df['Close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            self.df_dict[asset]['atr'] = tr.rolling(window=14).mean().fillna(tr.mean())
            self.atr_dict[asset] = self.df_dict[asset]['atr']
        
        # Features per asset: 6 original + 3 HTF + 1 regime = 10 features per asset
        self.feature_cols = [
            'feat_returns_5', 'feat_price_pos', 'feat_norm_vol', 
            'feat_hour_sin', 'feat_hour_cos', 'feat_vol_spike',
            'htf_1h_trend', 'htf_4h_trend', 'htf_4h_rsi',
            'feat_regime'
        ]
        
        # Define action space: 3 actions per asset
        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)  # [0,1,2] for each asset
        
        # Observation space: features per asset + position + current PnL for each asset + balance info
        obs_size = (len(self.feature_cols) * self.n_assets) + (2 * self.n_assets) + 1  # +1 for balance ratio
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,  # Features are clipped at +/- 3
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = {asset: 0 for asset in self.assets}  # Position per asset
        self.entry_prices = {asset: 0 for asset in self.assets}  # Entry price per asset
        self.current_step = 0
        
        # Stats
        self.peak_equity = self.initial_balance
        self.daily_start_equity = self.initial_balance
        self.steps_held = {asset: 0 for asset in self.assets}  # Steps held per asset
        self.returns_list = []  # For Sortino calculation
        
        return self._get_observation()

    def _get_observation(self):
        obs_features = []
        
        # Collect features for each asset
        for asset in self.assets:
            df = self.df_dict[asset]
            if self.current_step < len(df):
                feat = df[self.feature_cols].iloc[self.current_step].values
                obs_features.extend(feat)
                
                # Add position and PnL for this asset
                position = float(self.positions[asset])
                pnl = 0
                if self.positions[asset] != 0:
                    price = df['Close'].iloc[self.current_step]
                    if self.positions[asset] == 1:  # Long
                        pnl = (price - self.entry_prices[asset]) / self.entry_prices[asset]
                    else:  # Short
                        pnl = (self.entry_prices[asset] - price) / self.entry_prices[asset]
                
                obs_features.extend([position, pnl])
            else:
                # If we're past the data length for this asset, use zeros
                obs_features.extend([0.0] * len(self.feature_cols))  # Zero features
                obs_features.extend([0.0, 0.0])  # Zero position and PnL
        
        # Add overall balance ratio
        balance_ratio = (self.equity / self.initial_balance) - 1  # Deviation from initial balance
        obs_features.append(balance_ratio)
        
        return np.array(obs_features, dtype=np.float32)

    def _get_cost(self, asset):
        # Slippage modeled as % of volatility for specific asset
        df = self.df_dict[asset]
        vol_slippage = (df['feat_rel_range_norm'].iloc[self.current_step] * 0.0001) if 'feat_rel_range_norm' in df.columns else 0.0001
        return self.base_cost + max(0, vol_slippage)

    def step(self, actions):
        """
        actions: array of actions for each asset [action_asset1, action_asset2, ...]
        """
        prev_equity = self.equity
        step_reward = 0
        
        # Execute actions for each asset
        for i, asset in enumerate(self.assets):
            action = actions[i]
            df = self.df_dict[asset]
            if self.current_step >= len(df):
                continue  # Skip if past the data length for this asset
            
            price = df['Close'].iloc[self.current_step]
            current_cost = self._get_cost(asset)
            
            # 1. Action Execution
            if action == 0:  # FLAT
                if self.positions[asset] != 0:
                    self.balance = self.equity * (1 - current_cost)
                    self.positions[asset] = 0
                    self.entry_prices[asset] = 0
                    self.steps_held[asset] = 0
            
            elif action == 1:  # BUY
                if self.positions[asset] == 2:  # Close Short
                    self.balance = self.equity * (1 - current_cost)
                    self.positions[asset] = 0
                
                if self.positions[asset] == 0:  # Open Long
                    self.balance *= (1 - current_cost)
                    self.entry_prices[asset] = price
                    self.positions[asset] = 1
                    self.steps_held[asset] = 0
                    
            elif action == 2:  # SELL
                if self.positions[asset] == 1:  # Close Long
                    self.balance = self.equity * (1 - current_cost)
                    self.positions[asset] = 0
                    
                if self.positions[asset] == 0:  # Open Short
                    self.balance *= (1 - current_cost)
                    self.entry_prices[asset] = price
                    self.positions[asset] = 2
                    self.steps_held[asset] = 0

        # 2. Update Equity (calculate total P&L across all assets)
        total_asset_value = self.balance  # Start with cash balance
        
        for asset in self.assets:
            if self.current_step >= len(self.df_dict[asset]):
                continue  # Skip if past the data length for this asset
            
            price = self.df_dict[asset]['Close'].iloc[self.current_step]
            if self.positions[asset] == 1:  # Long
                asset_value = self.balance * (price / self.entry_prices[asset]) if self.entry_prices[asset] != 0 else self.balance
                total_asset_value += asset_value - self.balance  # Add profit/loss
                self.steps_held[asset] += 1
            elif self.positions[asset] == 2:  # Short
                asset_value = self.balance * (2 - (price / self.entry_prices[asset])) if self.entry_prices[asset] != 0 else self.balance
                total_asset_value += asset_value - self.balance  # Add profit/loss
                self.steps_held[asset] += 1
            # If flat (position == 0), no change to asset value
        
        self.equity = total_asset_value

        # 3. Calculate Reward
        ret = (self.equity - prev_equity) / (prev_equity + 1e-9)
        step_reward = ret * 100.0
        
        # Asymmetric Penalty (Balanced to -1.5x)
        if ret < 0:
            step_reward *= 1.5
            
        # Drawdown Penalty
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = (self.peak_equity - self.equity) / (self.peak_equity + 1e-9)
        step_reward -= dd * 0.1 # Reduced weight
        
        # Advanced Reward: Sortino Ratio Component
        # Track returns for Sortino calculation
        self.returns_list.append(ret)
        
        # Calculate Sortino ratio periodically and add to reward
        if len(self.returns_list) >= 10:
            returns_array = np.array(self.returns_list[-50:])  # Use last 50 returns
            if len(returns_array) > 1:
                mean_return = np.mean(returns_array)
                negative_returns = returns_array[returns_array < 0]
                if len(negative_returns) > 0:
                    downside_std = np.std(negative_returns)
                    sortino = mean_return / (downside_std + 1e-9) if downside_std > 0 else mean_return * 10
                else:
                    sortino = mean_return * 10  # High reward if no negative returns
                
                # Add Sortino component to reward (scaled down)
                step_reward += max(0, sortino) * 0.01  # Small positive contribution
        
        # Time Decay + Overtrading Penalty
        for asset in self.assets:
            if self.positions[asset] != 0:
                step_reward -= 0.001 * self.steps_held[asset]
        
        # 4. Risk Mgmt (Circuit Breakers)
        done = False
        
        # Hard Stop Loss (5x ATR) for each asset
        for asset in self.assets:
            if self.current_step >= len(self.df_dict[asset]):
                continue  # Skip if past the data length for this asset
            
            if self.positions[asset] != 0:
                atr = self.df_dict[asset]['atr'].iloc[self.current_step]
                if self.positions[asset] == 1:  # Long SL
                    if price < self.entry_prices[asset] - (5 * atr):
                        done = True
                        step_reward -= 2.0
                elif self.positions[asset] == 2:  # Short SL
                    if price > self.entry_prices[asset] + (5 * atr):
                        done = True
                        step_reward -= 2.0

        # Max Drawdown (10%)
        if dd > 0.10:
            done = True
            step_reward -= 5.0
            
        # Daily Loss (5% for learning)
        if self.current_step % 96 == 0:
            self.daily_start_equity = self.equity
        
        daily_ret = (self.equity - self.daily_start_equity) / (self.daily_start_equity + 1e-9)
        if daily_ret < -0.05:
            done = True
            step_reward -= 3.0
            
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            
        obs = self._get_observation()
        return obs, float(step_reward), done, {"equity": self.equity, "balance": self.balance, "positions": self.positions}


# Example usage function
def create_multi_asset_env(eurusd_df, gbpusd_df=None, xausd_df=None):
    """
    Helper function to create a multi-asset environment
    """
    df_dict = {"EURUSD": eurusd_df}
    if gbpusd_df is not None:
        df_dict["GBPUSD"] = gbpusd_df
    if xausd_df is not None:
        df_dict["XAUUSD"] = xausd_df
    
    return MultiAssetTradingEnv(df_dict)