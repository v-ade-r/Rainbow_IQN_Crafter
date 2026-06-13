"""IQN quantile embedding and Dueling IQN heads.

Provides a dual-head IQN network: shared encoder and quantile embedding,
but two independent Dueling heads for extrinsic and intrinsic Q-values.
This follows Burda et al. (2018) "Exploration by Random Network
Distillation", where separate heads let the agent learn about extrinsic
(episodic) and intrinsic (non-episodic) returns without one drowning
out the other.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.networks.noisy_linear import NoisyLinear


class QuantileEmbedding(nn.Module):
    """Cosine basis embedding for quantile fractions tau in [0, 1].

    Maps each tau to a feature_dim vector using:
        phi(tau) = SiLU(Linear(cos(pi * i * tau) for i in 1..embedding_dim))
    """

    def __init__(self, embedding_dim: int = 64, feature_dim: int = 512) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, feature_dim)
        self.activation = nn.SiLU()

        indices = torch.arange(1, embedding_dim + 1, dtype=torch.float32)
        self.register_buffer("indices", indices)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau: (B, N) quantile fractions in [0, 1].
        Returns:
            (B, N, feature_dim) quantile embeddings.
        """
        tau_expanded = rearrange(tau, "b n -> b n 1")
        indices = rearrange(self.indices, "d -> 1 1 d")
        cos_features = torch.cos(math.pi * indices * tau_expanded)
        return self.activation(self.fc(cos_features))


class DuelingIQNHead(nn.Module):
    """Dueling architecture on top of IQN quantile features.

    Splits into Value and Advantage streams, each using NoisyLinear layers.
    Output: Q(s, a, tau) = V(s, tau) + A(s, a, tau) - mean_a(A(s, a, tau))
    """

    def __init__(self, feature_dim: int = 512, n_actions: int = 17) -> None:
        super().__init__()
        self.n_actions = n_actions

        self.value_stream = nn.Sequential(
            NoisyLinear(feature_dim, feature_dim),
            nn.SiLU(),
            NoisyLinear(feature_dim, 1),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, feature_dim),
            nn.SiLU(),
            NoisyLinear(feature_dim, n_actions),
        )

    def forward(
        self, encoder_features: torch.Tensor, quantile_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: (B, feature_dim) from the encoder.
            quantile_embeddings: (B, N, feature_dim) from QuantileEmbedding.
        Returns:
            (B, N, n_actions) quantile values for each action.
        """
        z = repeat(encoder_features, "b d -> b n d", n=quantile_embeddings.shape[1])
        h = z * quantile_embeddings

        value = self.value_stream(h)
        advantage = self.advantage_stream(h)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DualHeadIQNNetwork(nn.Module):
    """IQN network with shared encoder and two Dueling heads.

    Layout:
        encoder (CNN or IMPALA) ---> features (B, D)
        QuantileEmbedding(tau)  ---> tau_feats  (B, N, D)
        features * tau_feats    ---> fused      (B, N, D)
        fused -> head_ext -> Q_ext  (B, N, A)
        fused -> head_int -> Q_int  (B, N, A)

    The head_int path is used only when RND is enabled (rnd_beta > 0).
    When disabled, its outputs are still computed (tiny overhead) but
    the agent ignores them.
    """

    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 512,
        n_actions: int = 17,
        quantile_embedding_dim: int = 64,
        encoder_type: str = "impala",
    ) -> None:
        super().__init__()
        from src.networks.encoders import CNNEncoder, ImpalaEncoder

        if encoder_type == "impala":
            self.encoder = ImpalaEncoder(in_channels=in_channels, feature_dim=feature_dim)
        elif encoder_type == "nature":
            self.encoder = CNNEncoder(in_channels=in_channels, feature_dim=feature_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.quantile_embed = QuantileEmbedding(
            embedding_dim=quantile_embedding_dim, feature_dim=feature_dim
        )
        self.head_ext = DuelingIQNHead(feature_dim=feature_dim, n_actions=n_actions)
        self.head_int = DuelingIQNHead(feature_dim=feature_dim, n_actions=n_actions)

    def forward(
        self, obs: torch.Tensor, tau: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, C, H, W) observations in [0, 1].
            tau: (B, N) quantile fractions.
        Returns:
            q_ext: (B, N, n_actions) extrinsic quantile Q-values.
            q_int: (B, N, n_actions) intrinsic quantile Q-values.
        """
        features = self.encoder(obs)
        embeddings = self.quantile_embed(tau)
        q_ext = self.head_ext(features, embeddings)
        q_int = self.head_int(features, embeddings)
        return q_ext, q_int

    def reset_noise(self) -> None:
        self.head_ext.reset_noise()
        self.head_int.reset_noise()
