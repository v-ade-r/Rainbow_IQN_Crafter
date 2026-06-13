"""Quantile Huber loss for IQN training."""

import torch


def quantile_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the quantile Huber loss for IQN.

    Args:
        predictions: (B, N) predicted quantile values for the chosen action.
        targets: (B, N') target quantile values.
        taus: (B, N) quantile fractions corresponding to predictions.
        kappa: Huber loss threshold.

    Returns:
        loss: scalar loss averaged over batch, summed over target quantiles,
              meaned over predicted quantiles.
        td_errors: (B,) mean absolute TD errors per sample (for PER).
    """
    # (B, N, 1) - (B, 1, N') -> (B, N, N')
    delta = targets.unsqueeze(1) - predictions.unsqueeze(2)

    abs_delta = delta.abs()
    huber = torch.where(
        abs_delta <= kappa,
        0.5 * delta.pow(2) / kappa,
        abs_delta - 0.5 * kappa,
    )

    # rho_tau weighting: |tau - I{delta < 0}|. The torch.where form below is
    # mathematically identical to (taus - (delta<0).float()).abs() but avoids
    # the explicit float cast and abs() call (cheaper on GPU).
    taus_b = taus.unsqueeze(2)  # (B, N, 1)
    tau_weight = torch.where(delta < 0, 1.0 - taus_b, taus_b)
    element_loss = tau_weight * huber

    # Sum over target quantiles N', mean over prediction quantiles N
    loss_per_sample = element_loss.sum(dim=2).mean(dim=1)

    # Mean absolute TD error for priority updates
    td_errors = delta.abs().mean(dim=(1, 2))

    return loss_per_sample, td_errors
