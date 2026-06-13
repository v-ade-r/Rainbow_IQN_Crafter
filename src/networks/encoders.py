"""Image encoders for Crafter observations.

Provides two encoders:
    - CNNEncoder: Nature-DQN-style 3-layer CNN (small, fast).
    - ImpalaEncoder: IMPALA ResNet (Espeholt et al. 2018), standard choice
      for Crafter / procgen. Better sample efficiency and representation
      quality at a modest parameter cost.
"""

import torch
import torch.nn as nn
from einops import rearrange


class CNNEncoder(nn.Module):
    """3-layer Nature-DQN-style convolutional encoder.

    Architecture: Conv(32,8,s4) -> Conv(64,4,s2) -> Conv(64,3,s1) -> FC.
    Fast, lightweight, but limited representational capacity.
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 512) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.SiLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self._conv_output_size(in_channels), feature_dim),
            nn.SiLU(),
            nn.LayerNorm(feature_dim),
        )

    def _conv_output_size(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 64, 64)
        with torch.no_grad():
            out = self.convs(dummy)
        return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs(x)
        h = rearrange(h, "b c h w -> b (c h w)")
        return self.fc(h)


class ResidualBlock(nn.Module):
    """IMPALA residual block: two 3x3 convs with a skip connection.

    Activation is SiLU (smoother gradients than ReLU, torch.compile friendly).
    Pre-activation layout: act -> conv -> act -> conv + skip.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(x)
        h = self.conv1(h)
        h = self.act2(h)
        h = self.conv2(h)
        return x + h


class ImpalaBlock(nn.Module):
    """One stage of the IMPALA encoder.

    Conv(3x3) -> MaxPool(3x3, stride=2) -> 2x ResidualBlock.
    Downsamples spatial resolution by 2.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.pool(h)
        h = self.res1(h)
        h = self.res2(h)
        return h


class ImpalaEncoder(nn.Module):
    """IMPALA-ResNet encoder (Espeholt et al. 2018).

    Architecture: 3 IMPALA blocks with channel widths (16, 32, 32),
    followed by SiLU, flatten, Linear -> LayerNorm. Total downsample 8x,
    so 64x64 -> 8x8 spatial at the output.

    Standard choice for Crafter / procgen baselines. ~0.6M params in the
    convolutional stack.
    """

    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 512,
        channel_widths: tuple[int, ...] = (16, 32, 32),
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        blocks = []
        prev_c = in_channels
        for c in channel_widths:
            blocks.append(ImpalaBlock(prev_c, c))
            prev_c = c
        self.blocks = nn.Sequential(*blocks)
        self.final_act = nn.SiLU()

        conv_out = self._conv_output_size(in_channels)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, feature_dim),
            nn.SiLU(),
            nn.LayerNorm(feature_dim),
        )

    def _conv_output_size(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 64, 64)
        with torch.no_grad():
            out = self.final_act(self.blocks(dummy))
        return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        h = self.final_act(h)
        h = rearrange(h, "b c h w -> b (c h w)")
        return self.fc(h)
