"""Rainbow-IQN agent: Double-DQN, Dueling, NoisyNet, Prioritized Experience
 Replay, n-step returns, IQN + RND/NovelD.

Architectural choices:
  - Dual-head IQN (shared encoder/embedding, separate Dueling heads for
    extrinsic and intrinsic Q-values). Extrinsic uses Munchausen-augmented
    targets; intrinsic is non-episodic Double-DQN. The intrinsic Huber loss
    is down-weighted (0.25x) so the shared encoder is dominated by the
    extrinsic objective (NGU/Agent57 style).
  - Hard target sync every `target_update_freq` learn steps (Rainbow / IQN /
    M-DQN standard); soft Polyak was too unstable for distributional Q.
  - NoisyNet exploration: the network stays in train() mode during act()
    and re-samples noise each call.
  - NovelD (Zhang et al. 2021) for intrinsic rewards, stored separately in
    the replay buffer alongside extrinsic rewards. Munchausen log-policy
    bonus is computed from the TARGET network per Vieillard et al. 2020.
"""

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.components.replay_buffer import PrioritizedReplayBuffer
from src.components.rnd import RNDModule
from src.networks.heads import DualHeadIQNNetwork
from src.utils.logging import compute_gradient_norms
from src.utils.losses import quantile_huber_loss


class RainbowIQNAgent(BaseAgent):
    """Full Rainbow-IQN agent with dual-head Q, NovelD exploration, and
    Munchausen-augmented extrinsic targets."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        device: torch.device | str = "cuda",
        # Network
        in_channels: int = 12,
        feature_dim: int = 512,
        quantile_embedding_dim: int = 64,
        n_quantiles_train: int = 32,
        n_quantiles_eval: int = 64,
        huber_kappa: float = 1.0,
        encoder_type: str = "impala",
        # Optimizer
        learning_rate: float = 1e-4,
        adam_eps: float = 1.5e-4,
        grad_clip_norm: float = 10.0,
        # RL
        gamma: float = 0.99,
        gamma_int: float = 0.9,
        n_step: int = 3,
        # Target network: hard sync every `target_update_freq` LEARN steps.
        # With train_freq=4, 2000 learn steps = 8000 env steps (Rainbow / IQN /
        # Munchausen standard). Soft Polyak (tau~0.005) was empirically
        # unstable for distributional RL because the target distribution
        # tracks the online distribution too quickly to fix a learning signal.
        target_update_freq: int = 2000,
        # Replay buffer
        buffer_size: int = 1_000_000,
        batch_size: int = 64,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 1_000_000,
        # Intrinsic motivation (RND + NovelD)
        rnd_beta: float = 0.5,
        rnd_lr: float = 1e-4,
        rnd_output_dim: int = 256,
        rnd_warmup_steps: int = 20_000,
        noveld_beta: float = 0.5,
        # Loss-side weight on intrinsic head's quantile Huber loss. Encoder
        # is shared across heads, so an unweighted sum splits its gradient
        # ~50/50 between "predict reward" and "explore novelty". 0.25 keeps
        # intrinsic learning alive while letting Q_ext dominate the encoder
        # representation (NGU/Agent57 use a similar imbalance).
        loss_int_weight: float = 0.25,
        # Munchausen (extrinsic head only)
        munchausen_alpha: float = 0.9,
        munchausen_tau: float = 0.03,
        munchausen_clip: float = -1.0,
    ) -> None:
        self._device = torch.device(device)
        self.n_actions = n_actions
        self.n_quantiles_train = n_quantiles_train
        self.n_quantiles_eval = n_quantiles_eval
        self.huber_kappa = huber_kappa
        self.grad_clip_norm = grad_clip_norm
        self.gamma = gamma
        self.gamma_int = gamma_int
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.loss_int_weight = loss_int_weight

        self.rnd_beta = rnd_beta
        self.munchausen_alpha = munchausen_alpha
        self.munchausen_tau = munchausen_tau
        self.munchausen_clip = munchausen_clip

        self.online_net = DualHeadIQNNetwork(
            in_channels=in_channels,
            feature_dim=feature_dim,
            n_actions=n_actions,
            quantile_embedding_dim=quantile_embedding_dim,
            encoder_type=encoder_type,
        ).to(self._device)

        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=learning_rate, eps=adam_eps
        )

        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            obs_shape=obs_shape,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
            n_step=n_step,
            gamma=gamma,
        )

        self.rnd: RNDModule | None = None
        if rnd_beta > 0:
            self.rnd = RNDModule(
                in_channels=in_channels,
                output_dim=rnd_output_dim,
                image_size=64,
                learning_rate=rnd_lr,
                device=self._device,
                warmup_steps=rnd_warmup_steps,
                noveld_beta=noveld_beta,
                gamma_int=gamma_int,
            )

        self.learn_step_count = 0

    @property
    def device(self) -> torch.device:
        return self._device

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select action by argmax of (Q_ext + beta * Q_int) averaged over taus.

        In training mode (deterministic=False) the network stays in train()
        with fresh NoisyNet noise per call -- NoisyNet IS the exploration
        policy. In evaluation (deterministic=True), eval() disables noise.
        """
        with torch.no_grad():
            if deterministic:
                self.online_net.eval()
            else:
                self.online_net.train()
                self.online_net.reset_noise()

            obs_t = (
                torch.from_numpy(obs).float().unsqueeze(0).to(self._device) / 255.0
            )
            tau = torch.rand(1, self.n_quantiles_eval, device=self._device)
            q_ext, q_int = self.online_net(obs_t, tau)

            combined = q_ext.mean(dim=1)
            if self.rnd is not None and self.rnd_beta > 0:
                combined = combined + self.rnd_beta * q_int.mean(dim=1)
            action = combined.argmax(dim=1).item()
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Store transition. Computes NovelD intrinsic reward and records
        both streams separately in the replay buffer.

        Returns the intrinsic reward (0.0 if RND disabled or in warmup).
        """
        intrinsic_reward = 0.0
        if self.rnd is not None:
            intrinsic_reward = self.rnd.compute_intrinsic_reward(state, next_state)

        self.buffer.push(
            state, action, float(reward), float(intrinsic_reward), next_state, done
        )
        return intrinsic_reward

    def can_learn(self) -> bool:
        return len(self.buffer) >= self.batch_size

    def _compute_policy(
        self, q_mean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute softmax policy log-probs from mean Q-values (Munchausen).

        Args:
            q_mean: (B, A) mean-over-quantiles Q-values.
        Returns:
            pi: (B, A) policy probabilities.
            log_pi: (B, A) log probabilities.
        """
        logits = q_mean / self.munchausen_tau
        log_pi = F.log_softmax(logits, dim=-1)
        pi = log_pi.exp()
        return pi, log_pi

    def _compute_ext_target(
        self,
        states_t: torch.Tensor,
        actions_t: torch.Tensor,
        rewards_ext_t: torch.Tensor,
        next_states_t: torch.Tensor,
        dones_t: torch.Tensor,
        n_step_lengths_t: torch.Tensor,
    ) -> torch.Tensor:
        """Munchausen + distributional target for the extrinsic head.

        y(tau') = r_ext + alpha * tau_m * clip(log pi(a_t|s_t), l_0, 0)
                + gamma**k * (1-done) * sum_a pi(a|s') * (Q_tgt(s',a,tau') - tau_m * log pi(a|s'))

        where k is the per-sample effective n-step length (1..n_step). The
        (1-done) mask zeroes the bootstrap on tail transitions, but using the
        correct exponent is still necessary for the masked-out targets to be
        consistent across heads.

        Returns:
            (B, N_train) target quantile values.
        """
        with torch.no_grad():
            # --- Munchausen bonus: log pi(a_t | s_t) from TARGET mean Q ---
            # Per Vieillard et al. 2020 (M-DQN, eq. 5), the log-policy term
            # is computed against the *target* network. Using the online
            # network here creates a positive feedback loop where the agent
            # rewards itself for its own current preferences and amplifies
            # spurious early Q estimates.
            tau_policy = torch.rand(
                self.batch_size, self.n_quantiles_eval, device=self._device
            )
            q_ext_s, _ = self.target_net(states_t, tau_policy)
            mean_q_ext_s = q_ext_s.mean(dim=1)
            _, log_pi_s = self._compute_policy(mean_q_ext_s)
            log_pi_sa = log_pi_s.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            m_bonus = self.munchausen_alpha * self.munchausen_tau * log_pi_sa.clamp(
                min=self.munchausen_clip, max=0.0
            )

            # --- Target policy at s': use TARGET mean Q for soft expectation ---
            tau_next_policy = torch.rand(
                self.batch_size, self.n_quantiles_eval, device=self._device
            )
            q_ext_sn_target, _ = self.target_net(next_states_t, tau_next_policy)
            mean_q_ext_sn = q_ext_sn_target.mean(dim=1)
            pi_sn, log_pi_sn = self._compute_policy(mean_q_ext_sn)

            # --- Target distribution quantiles at s' ---
            tau_target = torch.rand(
                self.batch_size, self.n_quantiles_train, device=self._device
            )
            q_ext_target_quantiles, _ = self.target_net(next_states_t, tau_target)
            # q_ext_target_quantiles: (B, N, A)

            # Soft expectation: sum_a pi(a|s') * (Q(s',a,tau') - tau_m * log pi(a|s'))
            soft_correction = self.munchausen_tau * log_pi_sn  # (B, A)
            corrected = q_ext_target_quantiles - soft_correction.unsqueeze(1)
            weighted = pi_sn.unsqueeze(1) * corrected
            target_q = weighted.sum(dim=-1)  # (B, N)

            # Per-sample gamma**k, broadcast to (B, 1) for the (B, N) target.
            gamma_k = (self.gamma**n_step_lengths_t).unsqueeze(1)
            augmented_reward = rewards_ext_t + m_bonus  # (B,)
            target_quantiles = augmented_reward.unsqueeze(1) + gamma_k * (
                1.0 - dones_t.unsqueeze(1)
            ) * target_q

        return target_quantiles

    def _compute_int_target(
        self,
        rewards_int_t: torch.Tensor,
        next_states_t: torch.Tensor,
        n_step_lengths_t: torch.Tensor,
    ) -> torch.Tensor:
        """Double-DQN distributional target for the intrinsic head.

        Intrinsic is treated as NON-EPISODIC: done does not stop bootstrapping
        (Burda et al. 2018). We use a SHORTER discount (gamma_int < gamma) so
        that Q_int is bounded by r_max / (1 - gamma_int^n) and cannot dominate
        the action selection.

        IMPORTANT: because the intrinsic head bootstraps on every transition
        (including episode tails), the discount exponent MUST match the
        per-sample effective n-step length. Hardcoding gamma_int**n_step would
        over-discount tail transitions (length 1 or 2) by up to ~20% with
        gamma_int=0.9, biasing PER through inflated TD errors there.

        Returns:
            (B, N_train) target quantile values.
        """
        with torch.no_grad():
            # Online net selects best action (Double DQN)
            tau_select = torch.rand(
                self.batch_size, self.n_quantiles_eval, device=self._device
            )
            _, q_int_online = self.online_net(next_states_t, tau_select)
            best_actions = q_int_online.mean(dim=1).argmax(dim=1)

            # Target net evaluates quantiles
            tau_target = torch.rand(
                self.batch_size, self.n_quantiles_train, device=self._device
            )
            _, q_int_target = self.target_net(next_states_t, tau_target)
            best_actions_exp = best_actions.unsqueeze(1).unsqueeze(2).expand(
                -1, self.n_quantiles_train, 1
            )
            q_int_selected = q_int_target.gather(2, best_actions_exp).squeeze(2)

            # Per-sample gamma_int**k. Non-episodic: no (1 - done) factor.
            gamma_k = (self.gamma_int**n_step_lengths_t).unsqueeze(1)
            target_quantiles = rewards_int_t.unsqueeze(1) + gamma_k * q_int_selected

        return target_quantiles

    def learn(self) -> dict[str, float]:
        """One learning step: sample PER, compute dual-head IQN loss, update."""
        self.online_net.reset_noise()

        (
            states,
            actions,
            rewards_ext,
            rewards_int,
            next_states,
            dones,
            n_step_lengths,
            is_weights,
            tree_indices,
        ) = self.buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).float().to(self._device) / 255.0
        actions_t = torch.from_numpy(actions).long().to(self._device)
        rewards_ext_t = torch.from_numpy(rewards_ext).float().to(self._device)
        rewards_int_t = torch.from_numpy(rewards_int).float().to(self._device)
        next_states_t = torch.from_numpy(next_states).float().to(self._device) / 255.0
        dones_t = torch.from_numpy(dones).float().to(self._device)
        n_step_lengths_t = torch.from_numpy(n_step_lengths).float().to(self._device)
        is_weights_t = torch.from_numpy(is_weights).float().to(self._device)

        # --- Current quantile estimates for chosen actions (both heads) ---
        tau = torch.rand(self.batch_size, self.n_quantiles_train, device=self._device)
        q_ext_all, q_int_all = self.online_net(states_t, tau)

        actions_exp = actions_t.unsqueeze(1).unsqueeze(2).expand(
            -1, self.n_quantiles_train, 1
        )
        q_ext_pred = q_ext_all.gather(2, actions_exp).squeeze(2)
        q_int_pred = q_int_all.gather(2, actions_exp).squeeze(2)

        # --- Targets ---
        target_ext = self._compute_ext_target(
            states_t,
            actions_t,
            rewards_ext_t,
            next_states_t,
            dones_t,
            n_step_lengths_t,
        )
        target_int = self._compute_int_target(
            rewards_int_t, next_states_t, n_step_lengths_t
        )

        # --- Quantile Huber losses ---
        loss_ext_per_sample, td_ext = quantile_huber_loss(
            q_ext_pred, target_ext, tau, kappa=self.huber_kappa
        )
        loss_int_per_sample, td_int = quantile_huber_loss(
            q_int_pred, target_int, tau, kappa=self.huber_kappa
        )

        # Intrinsic loss only contributes when RND is enabled, and is
        # down-weighted because the encoder is shared: an unweighted sum
        # split its gradient ~50/50 between the extrinsic objective and the
        # novelty objective, observably slowing extrinsic learning.
        if self.rnd is not None:
            total_loss_per_sample = (
                loss_ext_per_sample + self.loss_int_weight * loss_int_per_sample
            )
        else:
            total_loss_per_sample = loss_ext_per_sample

        weighted_loss = (is_weights_t * total_loss_per_sample).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.grad_clip_norm
        )
        self.optimizer.step()

        # --- PER priority update from extrinsic TD error (primary signal) ---
        td_errors_np = td_ext.detach().cpu().numpy()
        self.buffer.update_priorities(tree_indices, td_errors_np)

        # --- Train RND predictor on next_states ---
        # Predictor training kicks in halfway through warm-up so that by the
        # time intrinsic rewards start flowing into the buffer (end of
        # warm-up) the predictor is no longer producing pure-noise novelty,
        # avoiding a catastrophically biased first wave of PER priorities.
        rnd_loss = None
        if self.rnd is not None and self.rnd.is_predictor_training_allowed:
            rnd_loss = self.rnd.train_predictor(next_states)

        # --- Hard target sync every target_update_freq learn steps ---
        self.learn_step_count += 1
        if self.learn_step_count % self.target_update_freq == 0:
            self._hard_update_target()

        grad_norms = compute_gradient_norms(self.online_net)

        # Action-selection contribution diagnostic: in healthy training Q_ext
        # should dominate (Q_int contribution = beta * |Q_int| / (|Q_ext| +
        # beta * |Q_int|) below ~20%). Higher means intrinsic motivation is
        # hijacking the policy.
        q_ext_abs = q_ext_pred.abs().mean().item()
        q_int_abs = q_int_pred.abs().mean().item()
        denom = q_ext_abs + self.rnd_beta * q_int_abs + 1e-8
        intrinsic_contribution_pct = 100.0 * self.rnd_beta * q_int_abs / denom

        metrics = {
            "loss": weighted_loss.item(),
            "loss_ext": loss_ext_per_sample.mean().item(),
            "loss_int": loss_int_per_sample.mean().item(),
            "td_error_mean": td_ext.mean().item(),
            "td_error_int_mean": td_int.mean().item(),
            "grad_norm": grad_norm.item(),
            "q_ext_mean": q_ext_pred.mean().item(),
            "q_int_mean": q_int_pred.mean().item(),
            "intrinsic_contribution_pct": intrinsic_contribution_pct,
            "td_errors": td_errors_np,
            "grad_norms_detailed": grad_norms,
        }
        if rnd_loss is not None:
            metrics["rnd_predictor_loss"] = rnd_loss

        return metrics

    def _hard_update_target(self) -> None:
        """Copy online -> target. Run periodically (every target_update_freq
        learn steps) so the target distribution is stationary between syncs,
        which is what distributional Q-learning needs for the quantile fit
        to converge before the regression target shifts again.
        """
        with torch.no_grad():
            for tp, op in zip(
                self.target_net.parameters(),
                self.online_net.parameters(),
                strict=True,
            ):
                tp.data.copy_(op.data)

    @staticmethod
    def _strip_compile_prefix(state_dict: dict) -> dict:
        """Remove '_orig_mod.' prefix added by torch.compile."""
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def _get_save_dict(self) -> dict:
        d = {
            "online_net": self._strip_compile_prefix(self.online_net.state_dict()),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learn_step_count": self.learn_step_count,
            "buffer_frame_count": self.buffer.frame_count,
            "buffer_max_priority": self.buffer.max_priority,
            "rng": self._collect_rng_state(),
        }
        if self.rnd is not None:
            d["rnd"] = self.rnd.get_state_dict()
        return d

    def _load_save_dict(self, state: dict) -> None:
        online_sd = state["online_net"]
        try:
            self.online_net.load_state_dict(online_sd)
        except RuntimeError:
            prefixed = {f"_orig_mod.{k}": v for k, v in online_sd.items()}
            self.online_net.load_state_dict(prefixed)
        self.target_net.load_state_dict(self._strip_compile_prefix(state["target_net"]))
        self.optimizer.load_state_dict(state["optimizer"])
        self.learn_step_count = state["learn_step_count"]
        if "buffer_frame_count" in state:
            self.buffer.frame_count = state["buffer_frame_count"]
        if "buffer_max_priority" in state:
            self.buffer.max_priority = state["buffer_max_priority"]
        if self.rnd is not None and "rnd" in state:
            self.rnd.load_state_dict(state["rnd"])
        if "rng" in state:
            self._restore_rng_state(state["rng"])

    # ------------------------------------------------------------------
    # RNG state preservation
    # ------------------------------------------------------------------
    # Without restoring RNG state, every restart produces a different IQN
    # tau sample (torch.rand), different NoisyNet noise (torch.randn in
    # reset_noise), and different PER stratified samples (np.random). The
    # policy weights are identical, but the immediate trajectory after a
    # restart diverges from what would have happened without a crash.
    # Note: this gives only "best effort" determinism -- cuDNN convolutions
    # and the environment's own RNG are still external sources of variance.

    @staticmethod
    def _collect_rng_state() -> dict:
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        return rng

    @staticmethod
    def _restore_rng_state(rng: dict) -> None:
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"].cpu().to(torch.uint8))
        if "torch_cuda_all" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [s.cpu().to(torch.uint8) for s in rng["torch_cuda_all"]]
            )

    # ------------------------------------------------------------------
    # Replay buffer persistence
    # ------------------------------------------------------------------
    # Network checkpoints (`save`/`load`) deliberately exclude the replay
    # buffer because torch.save serialization of ~12GB of uint8 image data
    # is slow and produces inflated files. The buffer is persisted to a
    # separate compressed .npz alongside the network checkpoint so a crash
    # never costs more than `checkpoint_freq` transitions.

    def save_buffer(self, path: str | Path) -> None:
        self.buffer.save_to_file(path)

    def load_buffer(self, path: str | Path) -> None:
        self.buffer.load_from_file(path)
