"""Unit tests for PrioritizedReplayBuffer: priority-proportional sampling and IS-weights."""

import numpy as np

from src.components.replay_buffer import NStepAccumulator, PrioritizedReplayBuffer


class TestNStepAccumulator:
    def test_nstep_return_computation(self):
        acc = NStepAccumulator(n=3, gamma=0.99)
        s = np.zeros((4, 4), dtype=np.uint8)

        r1 = acc.append(s, 0, 1.0, 0.0, s, False)
        assert r1 is None
        r2 = acc.append(s, 0, 2.0, 0.0, s, False)
        assert r2 is None
        r3 = acc.append(s, 0, 3.0, 0.0, s, False)
        assert r3 is not None

        _, _, R_ext, _, _, _, length = r3
        expected = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
        assert np.isclose(R_ext, expected, rtol=1e-5)
        assert length == 3

    def test_nstep_intrinsic_accumulation(self):
        """Intrinsic stream accumulates with the same gamma and n as extrinsic."""
        acc = NStepAccumulator(n=2, gamma=0.5)
        s = np.zeros((4, 4), dtype=np.uint8)

        assert acc.append(s, 0, 0.0, 1.0, s, False) is None
        result = acc.append(s, 0, 0.0, 3.0, s, False)
        assert result is not None
        _, _, R_ext, R_int, _, _, length = result
        assert np.isclose(R_ext, 0.0)
        assert np.isclose(R_int, 1.0 + 0.5 * 3.0)
        assert length == 2

    def test_done_does_not_autoflush(self):
        """append() on done=True must NOT emit or clear; caller uses flush_remaining."""
        acc = NStepAccumulator(n=3, gamma=0.99)
        s = np.zeros((4, 4), dtype=np.uint8)

        assert acc.append(s, 0, 1.0, 0.0, s, False) is None
        assert acc.append(s, 0, 2.0, 0.0, s, True) is None
        # All pending transitions must still be in the accumulator
        remaining = acc.flush_remaining()
        assert len(remaining) == 2
        _, _, R1, _, _, done1, len1 = remaining[0]
        _, _, R2, _, _, done2, len2 = remaining[1]
        # First flush sees buffer [t0(done=False), t1(done=True)] -> sums both,
        # breaks at t1 (k=1), effective length 2.
        assert np.isclose(R1, 1.0 + 0.99 * 2.0)
        assert len1 == 2
        # Second flush sees buffer [t1(done=True)] -> length 1, R = r1.
        assert np.isclose(R2, 2.0)
        assert len2 == 1
        assert done1 is True and done2 is True

    def test_nstep_length_with_full_window(self):
        """Full n=3 windows (no done) must report length=3."""
        acc = NStepAccumulator(n=3, gamma=0.9)
        s = np.zeros((4, 4), dtype=np.uint8)

        acc.append(s, 0, 1.0, 0.0, s, False)
        acc.append(s, 0, 2.0, 0.0, s, False)
        result = acc.append(s, 0, 3.0, 0.0, s, False)
        assert result is not None
        *_, length = result
        assert length == 3


class TestPERSampling:
    def _make_buffer(self, capacity=1000) -> PrioritizedReplayBuffer:
        return PrioritizedReplayBuffer(
            capacity=capacity,
            obs_shape=(4, 4),
            alpha=1.0,
            beta_start=1.0,
            beta_frames=1,
            n_step=1,
            gamma=0.99,
        )

    def test_high_priority_sampled_more(self):
        """Transition with error=100 should be sampled ~100x more than error=1."""
        buf = self._make_buffer()
        s = np.zeros((4, 4), dtype=np.uint8)

        for _ in range(3):
            buf.push(s, 0, 0.0, 0.0, s, True)

        td_errors = np.array([1.0, 10.0, 100.0])
        tree_indices = np.array([
            buf.tree.tree_idx_for_data(i) for i in range(3)
        ])
        buf.update_priorities(tree_indices, td_errors)

        n_samples = 30_000
        counts = np.zeros(3)
        for _ in range(n_samples):
            sample = buf.sample(1)
            ti = sample[-1]  # tree_indices is the last element
            data_idx = ti[0] - (buf.tree.capacity - 1)
            counts[data_idx] += 1

        ratio = counts[2] / max(counts[0], 1)
        assert ratio > 30, f"Expected high-priority item sampled much more, got ratio={ratio:.1f}"

    def test_uniform_weights_when_equal_priorities(self):
        """With alpha=1, beta=1 and equal priorities, IS-weights should be uniform (all 1.0)."""
        buf = self._make_buffer()
        s = np.zeros((4, 4), dtype=np.uint8)

        for _ in range(100):
            buf.push(s, 0, 0.0, 0.0, s, True)

        sample = buf.sample(32)
        is_weights = sample[-2]
        assert np.allclose(is_weights, 1.0, atol=0.05), (
            f"IS-weights should be ~1.0 for equal priorities, got {is_weights}"
        )

    def test_buffer_size_tracking(self):
        buf = self._make_buffer(capacity=50)
        s = np.zeros((4, 4), dtype=np.uint8)

        for i in range(100):
            buf.push(s, 0, float(i), 0.0, s, True)

        assert len(buf) == 50

    def test_nstep_integration(self):
        """Buffer with n_step=3 should compute compound returns and flush tail on done."""
        buf = PrioritizedReplayBuffer(
            capacity=100,
            obs_shape=(4, 4),
            alpha=0.6,
            beta_start=0.4,
            beta_frames=100_000,
            n_step=3,
            gamma=0.99,
        )
        s = np.zeros((4, 4), dtype=np.uint8)

        for _i in range(10):
            buf.push(s, 0, 1.0, 0.0, s, False)
        buf.push(s, 0, 1.0, 0.0, s, True)

        # 11 transitions, n=3: first 2 don't emit, next 8 each emit 1
        # (total 8), then terminal flush emits remaining 3 (n-1=2 + 1 final)
        # = 8 regular + 3 tail = 11 stored transitions. Verify at least
        # all final-episode transitions are preserved.
        assert len(buf) == 11
