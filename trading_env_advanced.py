import gym
import numpy as np
from gym import spaces
from collections import deque

class AdversarialTradingEnv(gym.Env):
    """
    AlphaZero-Style Adversarial Environment:
    - Multi-Asset: EURUSD, GBPUSD, JPY, AUDUSD, CAD
    - Adversarial: Environment applies slippage/spread widening against the potential trade.
    - Continuous Action: Portfolio weights for 5 pairs.
    """
    
    def __init__(self, data_dict, data_dict_1h=None, data_dict_1d=None, window_size=30, initial_balance=10000.0, base_spread=2.0, max_steps=5000):
        super(AdversarialTradingEnv, self).__init__()
        
        self.pairs = list(data_dict.keys())
        self.n_pairs = len(self.pairs)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.base_spread = base_spread
        self.max_steps = max_steps
        self.pip_value = 0.0001
        self.data_dict_1h = data_dict_1h  
        self.data_dict_1d = data_dict_1d  # Macro context
        
        # 1. Align Data
        first_key = self.pairs[0]
        self.n_steps = len(data_dict[first_key])
        
        # Convert to Feature Matrices Keyed by Pair
        self.feature_matrices = {}
        self.price_arrays = {}
        self.atr_arrays = {}
        self.timestamps = {} 
        
        # Secondary matrices
        self.feature_matrices_1h = {}
        self.timestamps_1h = {}
        
        self.feature_matrices_1d = {}
        self.timestamps_1d = {}
        
        for p in self.pairs:
            # Primary (Could be 15m or 1h depending on mode)
            df = data_dict[p]
            raw_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            feat_cols = [c for c in df.columns if c not in raw_cols]
            self.feature_matrices[p] = df[feat_cols].values.astype(np.float32)
            self.price_arrays[p] = df['Close'].values.astype(np.float32)
            self.timestamps[p] = df.index 
            
            # ATR for spread
            if 'atr' in feat_cols:
                atr_idx = feat_cols.index('atr')
                self.atr_arrays[p] = self.feature_matrices[p][:, atr_idx]
            else:
                self.atr_arrays[p] = np.zeros(self.n_steps)
                
            # Secondary (1h)
            if self.data_dict_1h and p in self.data_dict_1h:
                df_1h = self.data_dict_1h[p]
                feat_cols_1h = [c for c in df_1h.columns if c not in raw_cols]
                self.feature_matrices_1h[p] = df_1h[feat_cols_1h].values.astype(np.float32)
                self.timestamps_1h[p] = df_1h.index
                
            # Tertiary (1d)
            if self.data_dict_1d and p in self.data_dict_1d:
                df_1d = self.data_dict_1d[p]
                feat_cols_1d = [c for c in df_1d.columns if c not in raw_cols]
                self.feature_matrices_1d[p] = df_1d[feat_cols_1d].values.astype(np.float32)
                # Ensure 1d has access
                
        # Helper: Feature Dims
        self.n_features_per_pair = self.feature_matrices[first_key].shape[1]
        
        self.n_features_per_pair_1h = 0
        if self.data_dict_1h:
            self.n_features_per_pair_1h = self.feature_matrices_1h[first_key].shape[1]
            
        self.n_features_per_pair_1d = 0
        if self.data_dict_1d:
            self.n_features_per_pair_1d = self.feature_matrices_1d[first_key].shape[1]
        
        # Observation Space
        # Window*Feat (Primary) + Window*Feat (1h) + Window*Feat (1d) + Context
        self.obs_dim_primary = self.window_size * self.n_features_per_pair * self.n_pairs
        self.obs_dim_1h = self.window_size * self.n_features_per_pair_1h * self.n_pairs
        self.obs_dim_1d = self.window_size * self.n_features_per_pair_1d * self.n_pairs # Macro window also 30?
        self.context_size = 3 * self.n_pairs 
        
        self.total_features = self.obs_dim_primary + self.obs_dim_1h + self.obs_dim_1d + self.context_size

        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_features,), dtype=np.float32
        )
        
        # Action Space: 5 continuous actions (-1 to 1) for position sizing per pair
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_pairs,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        
        # State per pair
        self.positions = np.zeros(self.n_pairs) # % of equity
        self.entry_prices = np.zeros(self.n_pairs)
        
        # Random Start
        min_start = self.window_size
        max_start = self.n_steps - self.max_steps - 1
        if max_start > min_start:
            self.current_step = np.random.randint(min_start, max_start)
        else:
            self.current_step = min_start
            
        self.steps_taken = 0 
        self.done = False
        
        return self._get_observation()
    
    def _get_adversarial_spread_slippage(self, pair_idx, intended_action_delta):
        """
        Adversary Logic:
        If agent wants to BUY (delta > 0), Adversary pushes Ask price UP.
        If agent wants to SELL (delta < 0), Adversary pushes Bid price DOWN.
        Magnitude depends on Volatility (ATR).
        """
        p_name = self.pairs[pair_idx]
        current_atr = self.atr_arrays[p_name][self.current_step]
        
        # Base Spread
        spread = self.base_spread
        
        # Adversarial Slippage: 10% of ATR if trading aggressively
        # "AlphaZero" self-play pressure
        slippage = 0.0
        if abs(intended_action_delta) > 0.1: # Significant trade
            slippage = current_atr * 0.1 # Adversary moves price 10% of hourly range against you
            
        return spread, slippage

    def _get_observation(self):
        # Stack all pair windows
        obs_list = []
        
        end = self.current_step
        start = end - self.window_size
        
        # 1. Feature Windows (15m)
        if start < 0: start=0
            
        for p in self.pairs:
            mat = self.feature_matrices[p]
            window = mat[start:end].flatten()
            obs_list.append(window)
            
        # 2. Feature Windows (1H) - Alignment
        if self.data_dict_1h:
             # Find 1h Index corresponding to current_step (15m)
             # Heuristic: 1h index = 15m index // 4 (approx)
             # Proper way: Search timestamp
             # Optimization: Cached or Calculated
             # Since assumed synchronized start, approx is fine for RL speed.
             # or map: current_time = self.timestamps[p][end-1]
             # idx_1h = self.timestamps_1h[p].get_loc(current_time, method='nearest') or pad/backfill.
             # For speed, we will use index ratio (assuming both start at same date).
             # If downloaded together, they start roughly same date.
             
             idx_1h = end // 4
             start_1h = idx_1h - self.window_size
             if start_1h < 0: start_1h = 0
             
             for p in self.pairs:
                 mat_1h = self.feature_matrices_1h[p]
                 # Ensure bounds
                 if idx_1h >= len(mat_1h): idx_1h = len(mat_1h) - 1
                 # Adjust start
                 s_1h = max(0, idx_1h - self.window_size)
                 e_1h = idx_1h
                 
                 window_1h = mat_1h[s_1h:e_1h].flatten()
                 
                 # Pad if not enough history
                 expected_len = self.n_features_per_pair_1h * self.window_size
                 if len(window_1h) < expected_len:
                     padding = np.zeros(expected_len - len(window_1h))
                     window_1h = np.concatenate([padding, window_1h])
                     
                 obs_list.append(window_1h)
            
        # 3. Feature Windows (1D) - Alignment
        if self.data_dict_1d:
             # Heuristic: 1d index depends on primary.
             # If Primary=15m (96 steps/day), 1d = idx // 96
             # If Primary=1h (24 steps/day), 1d = idx // 24
             # Let's detect ratio or just assume 15m mode for now since that's the complex one?
             # Auto-detect scale ratio
             # Use timestamp estimation or simple constant.
             # 15m is 96 steps. 1h is 24 steps.
             # If len(1h data) ~ 1/4 len(primary), then primary is 15m.
             
             # Ratio Estimate
             # len(primary) / len(1d) ~ steps_per_day
             ratio = max(1, len(self.feature_matrices[self.pairs[0]]) / len(self.feature_matrices_1d[self.pairs[0]]))
             
             idx_1d = int(end // ratio)
             
             for p in self.pairs:
                 mat_1d = self.feature_matrices_1d[p]
                 if idx_1d >= len(mat_1d): idx_1d = len(mat_1d) - 1
                 
                 s_1d = max(0, idx_1d - self.window_size)
                 e_1d = idx_1d
                 
                 window_1d = mat_1d[s_1d:e_1d].flatten()
                 
                 expected_len_1d = self.n_features_per_pair_1d * self.window_size
                 if len(window_1d) < expected_len_1d:
                    padding = np.zeros(expected_len_1d - len(window_1d))
                    window_1d = np.concatenate([padding, window_1d])
                    
                 obs_list.append(window_1d)

        # 4. Context
        context_list = []
        for i, p in enumerate(self.pairs):
            pos = self.positions[i]
            
            # Unrealized PnL
            curr_price = self.price_arrays[p][self.current_step]
            entry = self.entry_prices[i]
            pnl = 0.0
            if abs(pos) > 0.001 and entry > 0:
                if pos > 0: pnl = (curr_price - entry) / entry
                else: pnl = (entry - curr_price) / entry
            
            context_list.extend([pos, pnl, 0.0]) # 0.0 is placeholder for Sentiment if not used
            
        full_obs = np.concatenate(obs_list + [np.array(context_list)])
        return full_obs.astype(np.float32)
    
    def step(self, actions):
        current_portfolio_return = 0.0
        total_cost_pnl = 0.0
        
        for i, p in enumerate(self.pairs):
            current_price = self.price_arrays[p][self.current_step]
            
            # Action for this pair with Sniper Neutral Zone
            raw_action = actions[i]
            if abs(raw_action) < 0.15:
                target_pct = 0.0
            else:
                target_pct = np.clip(raw_action, -1.0, 1.0)
                
            current_pct = self.positions[i]
            delta = target_pct - current_pct
            
            # --- ADVERSARY ---
            # Moves price against the trade
            spread, slippage_pct = self._get_adversarial_spread_slippage(i, delta)
            
            # Execution Price (Adversarial)
            # Buy -> Price goes UP (Slippage + Half Spread)
            # Sell -> Price goes DOWN
            # Slippage is % of ATR, which is % of Price. So slippage_pct is % of Price.
            slippage_val = slippage_pct * current_price
            
            exec_price = current_price
            if delta > 0: # Buy
                exec_price = current_price + slippage_val + (spread * self.pip_value)
            elif delta < 0: # Sell
                exec_price = current_price - slippage_val - (spread * self.pip_value)
                
            # Transaction Cost
            # Roughly: spread cost is baked into the exec_price difference from mid
            # But we track equity change
            
            # 1. Update Position
            if abs(delta) > 0.001:
                # Update Entry Price (Weighted Avg)
                if abs(target_pct) > abs(current_pct) and np.sign(target_pct) == np.sign(current_pct):
                    # Adding to pos
                    # New Entry = Weighted Avg of Old Entry and Exec Price
                    old_amt = abs(current_pct)
                    added_amt = abs(delta)
                    new_amt = abs(target_pct)
                    self.entry_prices[i] = (self.entry_prices[i] * old_amt + exec_price * added_amt) / new_amt
                elif np.sign(target_pct) != np.sign(current_pct):
                    # Flip or new
                    self.entry_prices[i] = exec_price
                    
                self.positions[i] = target_pct
                
            # 2. Calculate Return for this step (Change in Equity)
            # We use Next Price - Current Price for finding PnL of held position
            # Note: We just executed at 'exec_price' which might be worse than 'current_price'
            # The discrepancy is the cost.
            
            # Immediate hit from slippage/spread on the delta
            # delta is fraction of equity.
            # (exec_price - current_price) / current_price is the % loss per unit traded.
            # So cost_pnl is the return impact on the total equity.
            cost_pnl = -abs(delta) * (abs(exec_price - current_price) / current_price)
            total_cost_pnl += cost_pnl
            
        # Time Step
        prev_equity = self.equity
        self.current_step += 1
        self.steps_taken += 1
        
        # Calculate new Equity based on new prices and held positions
        step_pnl_sum = 0.0
        
        for i, p in enumerate(self.pairs):
            new_price = self.price_arrays[p][self.current_step]
            old_price = self.price_arrays[p][self.current_step - 1]
            
            price_ret = (new_price - old_price) / old_price
            pos_pnl = self.positions[i] * price_ret
            step_pnl_sum += pos_pnl
            
        # Update Equity
        # Apply Market Return + Transaction Costs
        self.equity = self.equity * (1.0 + step_pnl_sum + total_cost_pnl)
        
        # Reward (Sniper Mode Scaling)
        step_ret = (self.equity - prev_equity) / prev_equity
        reward = step_ret * 100.0
        
        if step_ret < 0:
            reward *= 3.5 # Aggressive penalty for accuracy focus
            
        # Holding friction (discourages staying in non-moving trades)
        holding_count = np.sum(np.abs(self.positions) > 0.001)
        reward -= (holding_count * 0.002) 
            
        # Error Handling: NaN/Inf Protection
        if not np.isfinite(reward):
            reward = -1.0 # Penalty for crash
            
        done = self.current_step >= self.n_steps - 1 or self.steps_taken >= self.max_steps
        if self.equity <= self.initial_balance * 0.1: # Bust
             done = True
             reward = -10.0
        
        obs = self._get_observation()
        # Nan protection for Obs
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        info = {'equity': self.equity, 'costs': total_cost_pnl}
        
        return obs, reward, done, info
