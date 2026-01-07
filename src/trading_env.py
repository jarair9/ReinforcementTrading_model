import gym
import numpy as np
from gym import spaces
import pandas as pd

class SimpleTradingEnv(gym.Env):
    """
    Revised Kindergarten Environment:
    - Single Asset: EURUSD 15m
    - 3 Features: Log Ret (5), Range, Time (Sin)
    - 3 Actions: 0 (Close/Flat), 1 (Buy), 2 (Sell)
    - Risk Mgmt: Drawdown circuit breaker, daily loss limit.
    - Rewards: Spread + Commission, Asymmetric penalty, Time decay.
    """
    def __init__(self, df, initial_balance=10000.0, spread=0.0001, commission=0.0001):
        super(SimpleTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.base_cost = spread + commission
        
        # Calculate ATR for stop-loss (5x ATR)
        tr = pd.concat([
            self.df['High'] - self.df['Low'],
            (self.df['High'] - self.df['Close'].shift(1)).abs(),
            (self.df['Low'] - self.df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(window=14).mean().fillna(tr.mean())
        
        # Features: Strictly non-leaking 6-feature set + 3 HTF + 1 regime features
        self.feature_cols = [
            'feat_returns_5', 'feat_price_pos', 'feat_norm_vol', 
            'feat_hour_sin', 'feat_hour_cos', 'feat_vol_spike',
            'htf_1h_trend', 'htf_4h_trend', 'htf_4h_rsi',
            'feat_regime'
        ]
        
        self.action_space = spaces.Discrete(3) 
        
        # Obs: Features + Position + Current PnL
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, # Features are clipped at +/- 3
            shape=(len(self.feature_cols) + 2,), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0 
        self.entry_price = 0
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Stats
        self.peak_equity = self.initial_balance
        self.daily_start_equity = self.initial_balance
        self.steps_held = 0
        
        # Reset returns list for Sortino calculation
        self.returns_list = []
        
        return self._get_observation()

    def _get_observation(self):
        feat = self.df[self.feature_cols].iloc[self.current_step].values
        
        pnl = 0
        if self.position != 0:
            price = self.df['Close'].iloc[self.current_step]
            if self.position == 1: pnl = (price - self.entry_price) / self.entry_price
            else: pnl = (self.entry_price - price) / self.entry_price
            
        return np.concatenate([feat, [float(self.position), pnl]]).astype(np.float32)

    def _get_cost(self):
        # Slippage modeled as % of volatility
        vol_slippage = (self.df['feat_rel_range_norm'].iloc[self.current_step] * 0.0001)
        return self.base_cost + max(0, vol_slippage)

    def step(self, action):
        price = self.df['Close'].iloc[self.current_step]
        prev_equity = self.equity
        step_reward = 0
        current_cost = self._get_cost()
        
        # 1. Action Execution
        if action == 0: # FLAT
            if self.position != 0:
                self.balance = self.equity * (1 - current_cost)
                self.position = 0
                self.entry_price = 0
                self.steps_held = 0
        
        elif action == 1: # BUY
            if self.position == 2: # Close Short
                self.balance = self.equity * (1 - current_cost)
                self.position = 0
            
            if self.position == 0: # Open Long
                self.balance *= (1 - current_cost)
                self.entry_price = price
                self.position = 1
                self.steps_held = 0
                
        elif action == 2: # SELL
            if self.position == 1: # Close Long
                self.balance = self.equity * (1 - current_cost)
                self.position = 0
                
            if self.position == 0: # Open Short
                self.balance *= (1 - current_cost)
                self.entry_price = price
                self.position = 2
                self.steps_held = 0

        # 2. Update Equity
        if self.position == 1:
            self.equity = self.balance * (price / self.entry_price)
            self.steps_held += 1
        elif self.position == 2:
            self.equity = self.balance * (2 - (price / self.entry_price))
            self.steps_held += 1
        else:
            self.equity = self.balance

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
        if not hasattr(self, 'returns_list'):
            self.returns_list = []
        self.returns_list.append(ret)
        
        # Calculate Sortino ratio periodically (every 10 steps) and add to reward
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
        if self.position != 0:
            step_reward -= 0.001 * self.steps_held
        
        if prev_equity != self.equity and self.position == 0: # Just closed a trade
             step_reward -= 0.001 # Overtrading penalty
            
        # 4. Risk Mgmt (Circuit Breakers)
        done = False
        
        # Hard Stop Loss (5x ATR)
        atr = self.df['atr'].iloc[self.current_step]
        if self.position == 1: # Long SL
            if price < self.entry_price - (5 * atr):
                done = True
                step_reward -= 2.0
        elif self.position == 2: # Short SL
            if price > self.entry_price + (5 * atr):
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
        return obs, float(step_reward), done, {"equity": self.equity}
