"""Toy Lottery test: verify IQN learns a bimodal step-function distribution.

A 50/50 lottery pays 0 or 100. With gamma=0, the true quantile function is:
    Q(tau) = 0   for tau < 0.5
    Q(tau) = 100 for tau >= 0.5

We train a minimal IQN head (no CNN, just a learned embedding) and verify
the predicted quantiles approximate this step function.
"""

import torch
import torch.optim as optim

from src.networks.heads import QuantileEmbedding
from src.utils.losses import quantile_huber_loss


class ToyIQNHead(torch.nn.Module):
    """Minimal IQN head for a 1-state, 1-action environment."""

    def __init__(self, feature_dim: int = 64, embedding_dim: int = 64) -> None:
        super().__init__()
        self.state_embed = torch.nn.Parameter(torch.randn(1, feature_dim) * 0.01)
        self.quantile_embed = QuantileEmbedding(
            embedding_dim=embedding_dim, feature_dim=feature_dim
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(feature_dim, 1),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau: (B, N) quantile fractions.
        Returns:
            (B, N) predicted quantile values.
        """
        B, N = tau.shape
        qe = self.quantile_embed(tau)
        state = self.state_embed.expand(B, -1).unsqueeze(1).expand(-1, N, -1)
        h = state * qe
        return self.fc(h).squeeze(-1)


class TestIQNDistribution:
    def test_toy_lottery_bimodal(self):
        """Train IQN on 50/50 lottery of 0 or 100, verify quantile step function."""
        torch.manual_seed(42)
        device = torch.device("cpu")

        model = ToyIQNHead(feature_dim=64, embedding_dim=64).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        n_steps = 5000
        batch_size = 64
        n_quantiles = 32

        for _step in range(n_steps):
            # Sample target returns: 50% chance of 0, 50% chance of 100
            targets_raw = (torch.rand(batch_size) > 0.5).float() * 100.0
            targets_raw = targets_raw.to(device)

            tau = torch.rand(batch_size, n_quantiles, device=device)
            predictions = model(tau)

            target_values = targets_raw.unsqueeze(1).expand(-1, n_quantiles)

            loss_per_sample, _ = quantile_huber_loss(predictions, target_values, tau, kappa=1.0)
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate: check quantiles at tau=0.1 (should be ~0) and tau=0.9 (should be ~100)
        model.eval()
        with torch.no_grad():
            test_taus = torch.tensor([[0.05, 0.1, 0.2, 0.3, 0.8, 0.9, 0.95]])
            predicted = model(test_taus).squeeze(0)

        low_quantiles = predicted[:4].numpy()
        high_quantiles = predicted[4:].numpy()

        for i, val in enumerate(low_quantiles):
            assert val < 30, (
                f"Low quantile {test_taus[0, i].item():.2f} predicted {val:.1f}, expected ~0"
            )

        for i, val in enumerate(high_quantiles):
            assert val > 70, (
                f"High quantile {test_taus[0, 4+i].item():.2f} predicted {val:.1f}, expected ~100"
            )
