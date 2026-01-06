import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(ResidualBlock, self).__init__()
        # Padding = (Kernel-1) * Dilation / 2
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNetFeatureExtractor(BaseFeaturesExtractor):
    """
    AlphaZero-Style ResNet for Time Series.
    Deep stack of Residual Blocks to capture complex, multi-scale patterns.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, window_size=30, n_pairs=5):
        super(ResNetFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.window_size = window_size
        self.n_pairs = n_pairs
        
        # Calculate Input Dimensions
        # Observation is likely flattened: (Window * Features * Pairs) + Context
        
        # Context size: 3 features per pair (Position, PnL, Placeholder)
        self.context_size = 3 * n_pairs 
        self.flattened_window_size = observation_space.shape[0] - self.context_size
        
        # Derive N_Features (Channels)
        self.n_features = self.flattened_window_size // self.window_size
        
        # Network Architecture
        # Input: (Batch, N_Features, Window)
        
        self.entry_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=2),
            ResidualBlock(64, dilation=4),
            ResidualBlock(64, dilation=8), # Exponential dilation for long memory
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=1)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1) # Flatten time dimension
        
        # Output dim of ResNet body is 64
        self.cnn_output_dim = 64
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_output_dim + self.context_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. Split
        window_flat = observations[:, :-self.context_size]
        context = observations[:, -self.context_size:]
        
        batch_size = window_flat.shape[0]
        
        # 2. Reshape to (Batch, Features, Window)
        # Note: Env MUST flatten as [t0_all_feats, t1_all_feats...] for this view to work:
        # .view(Batch, Window, Features) -> permute(Batch, Features, Window)
        window_reshaped = window_flat.view(batch_size, self.window_size, self.n_features)
        window_permuted = window_reshaped.permute(0, 2, 1)
        
        # 3. ResNet Body
        out = self.entry_conv(window_permuted)
        out = self.res_blocks(out)
        
        # 4. Pooling (Global Average)
        # (Batch, 64, Window) -> (Batch, 64, 1)
        out = self.global_pool(out)
        out = out.view(batch_size, -1) # (Batch, 64)
        
        # 5. Fusion
        combined = torch.cat([out, context], dim=1)
        return self.fusion(combined)
