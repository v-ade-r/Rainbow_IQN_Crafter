"""Random Network Distillation + NovelD intrinsic motivation.

NovelD (Zhang et al., 2021) computes the intrinsic reward as the positive
difference between novelty at the next state and novelty at the current
state:
    r_int = alpha * max(RND(s_{t+1}) - beta_nov * RND(s_t), 0)

This rewards transitions INTO new territory rather than being in a novel
state, which empirically outperforms vanilla RND on procedurally
generated environments (NetHack, MiniGrid, and -- our case -- Crafter).

Two-stage warm-up:
  * Stage 1 (steps 0..warmup_steps/2): only obs_normalizer updates.
  * Stage 2 (steps warmup_steps/2..warmup_steps): predictor *training*
    starts (so it begins fitting the target on real obs statistics) but
    `compute_intrinsic_reward` still returns 0.0, keeping the replay
    buffer free of spurious novelty signals from a randomly-initialised
    predictor.
  * Post-warmup: full NovelD signal flows into the buffer alongside a
    predictor that has already been roughly calibrated.

Without stage 2, the first ~thousands of post-warmup transitions land in
PER with astronomical TD errors and dominate sampling for tens of
thousands of steps after.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RunningMeanStd:
    """Online Welford algorithm for computing running mean and variance."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-8) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class RNDNetwork(nn.Module):
    """Lightweight CNN for RND target/predictor.

    Per Burda et al. 2018, the target and predictor share a compact
    architecture (two conv layers + linear). A larger frozen target
    just wastes compute.
    """

    def __init__(self, in_channels: int, output_dim: int = 256, image_size: int = 64) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
        )
        conv_out_size = self._get_conv_output_size(in_channels, image_size)
        self.fc = nn.Linear(conv_out_size, output_dim)

    def _get_conv_output_size(self, in_channels: int, image_size: int) -> int:
        dummy = torch.zeros(1, in_channels, image_size, image_size)
        with torch.no_grad():
            out = self.convs(dummy)
        return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class RNDModule:
    """RND + NovelD intrinsic motivation.

    Exposes compute_intrinsic_reward(state, next_state) which implements
    the NovelD criterion. Warm-up phase runs only obs-normalizer updates
    to stabilise statistics before producing intrinsic signals.
    """

    def __init__(
        self,
        in_channels: int,
        output_dim: int = 256,
        image_size: int = 64,
        learning_rate: float = 1e-4,
        device: torch.device | str = "cuda",
        warmup_steps: int = 20_000,
        noveld_scale: float = 1.0,
        noveld_beta: float = 0.5,
        gamma_int: float = 0.9,
    ) -> None:
        self.device = torch.device(device)
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.noveld_scale = noveld_scale
        self.noveld_beta = noveld_beta
        self.gamma_int = gamma_int

        self.target = RNDNetwork(in_channels, output_dim, image_size).to(self.device)
        self.predictor = RNDNetwork(in_channels, output_dim, image_size).to(self.device)

        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)

        self.obs_normalizer = RunningMeanStd(shape=(in_channels, image_size, image_size))

        # Burda et al. 2018 §2.4: normalize intrinsic reward by the running std
        # of DISCOUNTED INTRINSIC RETURNS (not raw rewards), so the reward
        # magnitude that feeds into Q-targets is O(1) regardless of episode
        # length. Without this, Q_int grows unboundedly with the discount
        # horizon (we observed Q_int -> +25 with the raw-novelty normalizer).
        self.return_normalizer = RunningMeanStd(shape=())
        self.intrinsic_return = 0.0  # running discounted sum, never reset
        # Last computed contribution diagnostics (for monitoring).
        self.last_raw_noveld: float = 0.0
        self.last_normalized_reward: float = 0.0

    @property
    def is_warming_up(self) -> bool:
        """True until intrinsic rewards may be released into the buffer."""
        return self.step_count < self.warmup_steps

    @property
    def is_predictor_training_allowed(self) -> bool:
        """Train the predictor during the second half of warm-up so that by
        the time intrinsic rewards start flowing into the replay buffer,
        the predictor is no longer producing pure-noise novelty values.
        """
        return self.step_count >= self.warmup_steps // 2

    def _novelty(self, obs_norm: np.ndarray) -> float:
        """Compute scalar RND novelty for a single normalized obs."""
        obs_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            target_feat = self.target(obs_t)
            pred_feat = self.predictor(obs_t)
            return (target_feat - pred_feat).pow(2).sum(dim=1).item()

    def compute_intrinsic_reward(
        self, state: np.ndarray, next_state: np.ndarray
    ) -> float:
        """NovelD intrinsic reward: max(N(s') - beta * N(s), 0), normalized.

        During warm-up, only updates obs_normalizer and returns 0.0.

        Args:
            state: (C, H, W) uint8 current observation.
            next_state: (C, H, W) uint8 next observation.

        Returns:
            Normalized NovelD reward (0.0 during warm-up).
        """
        state_f = state.astype(np.float64) / 255.0
        next_state_f = next_state.astype(np.float64) / 255.0

        self.obs_normalizer.update(next_state_f[np.newaxis])
        self.step_count += 1

        if self.is_warming_up:
            return 0.0

        s_norm = np.clip(self.obs_normalizer.normalize(state_f), -5.0, 5.0)
        sn_norm = np.clip(self.obs_normalizer.normalize(next_state_f), -5.0, 5.0)

        n_s = self._novelty(s_norm)
        n_sn = self._novelty(sn_norm)

        raw_noveld = max(n_sn - self.noveld_beta * n_s, 0.0)

        # Update running estimate of discounted intrinsic returns and divide
        # by its std (Burda et al. 2018). The single-env discounted return is
        # never reset across episodes, mirroring the non-episodic intrinsic
        # bootstrap used by the IQN intrinsic head.
        self.intrinsic_return = self.gamma_int * self.intrinsic_return + raw_noveld
        self.return_normalizer.update(np.array([self.intrinsic_return]))
        return_std = np.sqrt(self.return_normalizer.var + 1e-8)
        normalized = raw_noveld / max(float(return_std), 1e-8)

        self.last_raw_noveld = float(raw_noveld)
        self.last_normalized_reward = float(self.noveld_scale * normalized)
        return self.noveld_scale * normalized

    def train_predictor(self, obs_batch: np.ndarray) -> float:
        """Train the predictor to match the frozen target on a batch."""
        obs_float = obs_batch.astype(np.float64) / 255.0
        obs_norm = self.obs_normalizer.normalize(obs_float)
        obs_norm = np.clip(obs_norm, -5.0, 5.0)

        obs_t = torch.from_numpy(obs_norm).float().to(self.device)

        with torch.no_grad():
            target_features = self.target(obs_t)
        predictor_features = self.predictor(obs_t)

        loss = (target_features - predictor_features).pow(2).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_state_dict(self) -> dict:
        return {
            "predictor": self.predictor.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_mean": self.obs_normalizer.mean,
            "obs_var": self.obs_normalizer.var,
            "obs_count": self.obs_normalizer.count,
            "return_mean": self.return_normalizer.mean,
            "return_var": self.return_normalizer.var,
            "return_count": self.return_normalizer.count,
            "intrinsic_return": self.intrinsic_return,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.predictor.load_state_dict(state["predictor"])
        if "target" in state:
            self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.obs_normalizer.mean = state["obs_mean"]
        self.obs_normalizer.var = state["obs_var"]
        self.obs_normalizer.count = state["obs_count"]
        if "return_mean" in state:
            self.return_normalizer.mean = state["return_mean"]
            self.return_normalizer.var = state["return_var"]
            self.return_normalizer.count = state["return_count"]
            self.intrinsic_return = state.get("intrinsic_return", 0.0)
        self.step_count = state["step_count"]
