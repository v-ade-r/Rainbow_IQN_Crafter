"""Smoke test for the full learn() step.

Verifies that loss decreases after several gradient steps on deterministic
synthetic transitions -- a sanity check that the full pipeline (buffer ->
dual-head network -> Munchausen target -> quantile Huber loss -> backprop
-> hard target sync) converges on a simple task.
"""

import numpy as np
import torch

from src.agents.rainbow_iqn_agent import RainbowIQNAgent


class TestBellmanUpdate:
    def test_loss_decreases_after_updates(self):
        """A handful of learn() steps on synthetic data should reduce the loss."""
        torch.manual_seed(123)
        np.random.seed(123)
        device = torch.device("cpu")

        obs_shape = (4, 64, 64)
        n_actions = 5

        agent = RainbowIQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            device=device,
            in_channels=4,
            feature_dim=128,
            quantile_embedding_dim=32,
            n_quantiles_train=16,
            n_quantiles_eval=16,
            learning_rate=1e-3,
            adam_eps=1e-8,
            gamma=0.99,
            n_step=1,
            target_update_freq=10,
            buffer_size=100,
            batch_size=4,
            per_alpha=0.0,
            per_beta_start=1.0,
            per_beta_frames=1,
            rnd_beta=0.0,
            encoder_type="nature",
        )

        # Deterministic repeating transitions
        state = np.zeros(obs_shape, dtype=np.uint8)
        next_state = np.full(obs_shape, 128, dtype=np.uint8)
        for _ in range(20):
            agent.store_transition(state, 0, 1.0, next_state, False)
            agent.store_transition(next_state, 1, -0.5, state, False)
        agent.store_transition(next_state, 1, -0.5, state, True)

        initial_losses = [agent.learn()["loss"] for _ in range(5)]
        for _ in range(40):
            agent.learn()
        final_losses = [agent.learn()["loss"] for _ in range(5)]

        mean_initial = float(np.mean(initial_losses))
        mean_final = float(np.mean(final_losses))

        assert mean_final < mean_initial, (
            f"Loss should decrease with training: initial={mean_initial:.4f}, "
            f"final={mean_final:.4f}"
        )
