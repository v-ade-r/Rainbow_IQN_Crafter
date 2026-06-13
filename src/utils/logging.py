"""Training diagnostics: gradient norms, TD-error percentiles, Crafter score tracker."""

from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class RollingCrafterTracker:
    """Maintains a rolling window of recent episodes' achievement dicts and
    computes the official Crafter Score on the fly.

    The benchmark's "100-episode evaluation" convention is mirrored here as
    a rolling window over the most recent `window` episodes of training --
    this is the single most informative signal you can display during a run,
    because raw episode reward in Crafter is dominated by HP-delta noise.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.episodes: deque[dict[str, int]] = deque(maxlen=window)
        self.rewards: deque[float] = deque(maxlen=window)
        self.lengths: deque[int] = deque(maxlen=window)
        # Achievements that have EVER been unlocked across the whole run,
        # independent of the rolling window -- useful for tracking raw
        # progress of exploration over time.
        self.ever_unlocked: set[str] = set()

    def push(
        self,
        achievements: dict[str, int],
        episode_reward: float,
        episode_length: int,
    ) -> None:
        self.episodes.append(dict(achievements))
        self.rewards.append(float(episode_reward))
        self.lengths.append(int(episode_length))
        for name, count in achievements.items():
            if count > 0:
                self.ever_unlocked.add(name)

    def achievement_rates(self) -> dict[str, float]:
        """Fraction of episodes in the window where each achievement was unlocked."""
        if not self.episodes:
            return {}
        all_names: set[str] = set()
        for ep in self.episodes:
            all_names.update(ep.keys())
        n = len(self.episodes)
        return {
            name: sum(1 for ep in self.episodes if ep.get(name, 0) > 0) / n
            for name in all_names
        }

    def crafter_score(self) -> float:
        """Rolling Crafter Score on the current window (in percent, 0-100)."""
        rates = self.achievement_rates()
        if not rates:
            return 0.0
        rates_pct = np.array(list(rates.values())) * 100.0
        return float(np.exp(np.mean(np.log(1.0 + rates_pct))) - 1.0)

    def unique_unlocked_in_window(self) -> int:
        """Count of distinct achievements with non-zero rate in the window."""
        return sum(1 for r in self.achievement_rates().values() if r > 0)

    def total_unique_ever(self) -> int:
        return len(self.ever_unlocked)

    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    def mean_length(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else 0.0

    def mean_achievements_per_episode(self) -> float:
        if not self.episodes:
            return 0.0
        return float(
            np.mean(
                [sum(1 for c in ep.values() if c > 0) for ep in self.episodes]
            )
        )

    def n_episodes(self) -> int:
        return len(self.episodes)

    def summary_dict(self) -> dict[str, float]:
        """Flat dict suitable for W&B logging."""
        return {
            "rolling/crafter_score_pct": self.crafter_score(),
            "rolling/unique_achievements_window": self.unique_unlocked_in_window(),
            "rolling/unique_achievements_ever": self.total_unique_ever(),
            "rolling/mean_reward": self.mean_reward(),
            "rolling/mean_length": self.mean_length(),
            "rolling/mean_achievements_per_ep": self.mean_achievements_per_episode(),
            "rolling/window_size": self.n_episodes(),
        }


def compute_gradient_norms(model: nn.Module) -> dict[str, float]:
    """Compute per-layer gradient L2 norms.

    Returns a dict of {layer_name: grad_norm}. Flags vanishing (<1e-7)
    or exploding (>100) gradients in the key suffix.
    """
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            suffix = ""
            if norm < 1e-7:
                suffix = "_VANISHING"
            elif norm > 100:
                suffix = "_EXPLODING"
            norms[f"grad_norm/{name}{suffix}"] = norm
    return norms


def compute_td_error_percentiles(td_errors: np.ndarray) -> dict[str, float]:
    """Compute P10, P50, P90 of TD errors for stability monitoring.

    Widening spread (P90-P10) over time indicates unstable training.
    """
    return {
        "td_error/p10": float(np.percentile(td_errors, 10)),
        "td_error/p50": float(np.percentile(td_errors, 50)),
        "td_error/p90": float(np.percentile(td_errors, 90)),
        "td_error/spread": float(np.percentile(td_errors, 90) - np.percentile(td_errors, 10)),
    }


def compute_q_value_stats(
    q_values: torch.Tensor,
) -> dict[str, float]:
    """Compute summary statistics of Q-value quantile distributions.

    Args:
        q_values: (B, N, n_actions) quantile Q-values.

    Returns:
        Dict with mean, std, min, max across the batch.
    """
    mean_q = q_values.mean(dim=1)
    return {
        "q_values/mean": mean_q.mean().item(),
        "q_values/std": mean_q.std().item(),
        "q_values/min": mean_q.min().item(),
        "q_values/max": mean_q.max().item(),
    }


class DiagnosticLogger:
    """Aggregates and periodically flushes diagnostic metrics to W&B."""

    def __init__(self, log_freq: int = 1000) -> None:
        self.log_freq = log_freq
        self._td_errors: list[float] = []
        self._losses: list[float] = []
        self._grad_norms: list[dict[str, float]] = []
        self._rnd_rewards: list[float] = []
        self._rnd_losses: list[float] = []

    def record_learn_step(
        self,
        td_errors: np.ndarray,
        loss: float,
        grad_norms: dict[str, float],
    ) -> None:
        self._td_errors.extend(td_errors.tolist())
        self._losses.append(loss)
        self._grad_norms.append(grad_norms)

    def record_rnd(self, intrinsic_reward: float, predictor_loss: float | None = None) -> None:
        self._rnd_rewards.append(intrinsic_reward)
        if predictor_loss is not None:
            self._rnd_losses.append(predictor_loss)

    def flush(self, step: int) -> dict[str, Any]:
        """Compute and return aggregated metrics, clearing internal buffers."""
        metrics: dict[str, Any] = {}

        if self._td_errors:
            td_arr = np.array(self._td_errors)
            metrics.update(compute_td_error_percentiles(td_arr))
            self._td_errors.clear()

        if self._losses:
            metrics["train/loss_mean"] = float(np.mean(self._losses))
            self._losses.clear()

        if self._grad_norms:
            latest = self._grad_norms[-1]
            for k, v in latest.items():
                metrics[k] = v
            self._grad_norms.clear()

        if self._rnd_rewards:
            metrics["rnd/intrinsic_reward_mean"] = float(np.mean(self._rnd_rewards))
            metrics["rnd/intrinsic_reward_std"] = float(np.std(self._rnd_rewards))
            self._rnd_rewards.clear()

        if self._rnd_losses:
            metrics["rnd/predictor_loss"] = float(np.mean(self._rnd_losses))
            self._rnd_losses.clear()

        return metrics
