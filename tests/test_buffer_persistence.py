"""Round-trip persistence tests for PrioritizedReplayBuffer and SumTree.

Validates that saving the full buffer state to disk and loading it back
into a fresh instance preserves all sampled transitions, priorities, and
internal counters bit-for-bit. This is critical for crash recovery in
long Crafter runs (24h+ at 1M steps).
"""

import numpy as np

from src.components.replay_buffer import PrioritizedReplayBuffer
from src.components.sum_tree import SumTree


def _fill_buffer(buf: PrioritizedReplayBuffer, n_episodes: int, ep_len: int) -> None:
    """Push n_episodes of length ep_len into buf using deterministic data."""
    rng = np.random.default_rng(0)
    for ep in range(n_episodes):
        for t in range(ep_len):
            done = t == ep_len - 1
            obs = rng.integers(0, 256, size=buf.states.shape[1:], dtype=np.uint8)
            next_obs = rng.integers(0, 256, size=buf.states.shape[1:], dtype=np.uint8)
            buf.push(
                state=obs,
                action=int(rng.integers(0, 4)),
                reward_ext=float(rng.normal()),
                reward_int=float(rng.uniform()),
                next_state=next_obs,
                done=done,
            )


class TestSumTreeStateDict:
    def test_round_trip_preserves_priorities(self):
        tree = SumTree(capacity=64)
        for p in np.linspace(0.1, 5.0, 50):
            tree.add(float(p))

        state = tree.state_dict()
        restored = SumTree(capacity=64)
        restored.load_state_dict(state)

        assert restored.n_entries == tree.n_entries
        assert restored.data_pointer == tree.data_pointer
        np.testing.assert_array_equal(restored.tree, tree.tree)
        assert np.isclose(restored.total(), tree.total())

    def test_capacity_mismatch_raises(self):
        tree = SumTree(capacity=64)
        tree.add(1.0)
        wrong = SumTree(capacity=128)
        try:
            wrong.load_state_dict(tree.state_dict())
        except ValueError as e:
            assert "capacity mismatch" in str(e)
        else:
            raise AssertionError("expected ValueError for mismatched capacity")


class TestReplayBufferPersistence:
    def test_round_trip_preserves_all_arrays(self, tmp_path):
        obs_shape = (3, 8, 8)
        original = PrioritizedReplayBuffer(
            capacity=200, obs_shape=obs_shape, n_step=3, gamma=0.99
        )
        _fill_buffer(original, n_episodes=4, ep_len=20)

        path = tmp_path / "buffer.npz"
        original.save_to_file(path)

        restored = PrioritizedReplayBuffer(
            capacity=200, obs_shape=obs_shape, n_step=3, gamma=0.99
        )
        restored.load_from_file(path)

        assert len(restored) == len(original)
        assert restored.max_priority == original.max_priority
        assert restored.frame_count == original.frame_count

        n = len(original)
        np.testing.assert_array_equal(restored.states[:n], original.states[:n])
        np.testing.assert_array_equal(
            restored.next_states[:n], original.next_states[:n]
        )
        np.testing.assert_array_equal(restored.actions[:n], original.actions[:n])
        np.testing.assert_array_equal(
            restored.rewards_ext[:n], original.rewards_ext[:n]
        )
        np.testing.assert_array_equal(
            restored.rewards_int[:n], original.rewards_int[:n]
        )
        np.testing.assert_array_equal(restored.dones[:n], original.dones[:n])
        np.testing.assert_array_equal(
            restored.n_step_lengths[:n], original.n_step_lengths[:n]
        )
        np.testing.assert_array_equal(restored.tree.tree, original.tree.tree)
        assert restored.tree.data_pointer == original.tree.data_pointer
        assert restored.tree.n_entries == original.tree.n_entries

    def test_sampling_is_deterministic_after_reload(self, tmp_path):
        """Same priorities + same RNG seed -> identical sampling indices."""
        obs_shape = (3, 8, 8)
        original = PrioritizedReplayBuffer(
            capacity=200, obs_shape=obs_shape, n_step=3, gamma=0.99
        )
        _fill_buffer(original, n_episodes=4, ep_len=20)

        # Mutate priorities so the tree is non-uniform
        sample = original.sample(8)
        original.update_priorities(
            sample[-1], np.linspace(0.1, 10.0, 8).astype(np.float32)
        )

        path = tmp_path / "buffer.npz"
        original.save_to_file(path)

        restored = PrioritizedReplayBuffer(
            capacity=200, obs_shape=obs_shape, n_step=3, gamma=0.99
        )
        restored.load_from_file(path)

        np.random.seed(123)
        s1 = original.sample(16)
        np.random.seed(123)
        s2 = restored.sample(16)

        for arr1, arr2 in zip(s1, s2, strict=True):
            np.testing.assert_array_equal(arr1, arr2)

    def test_capacity_mismatch_raises(self, tmp_path):
        obs_shape = (3, 8, 8)
        a = PrioritizedReplayBuffer(capacity=200, obs_shape=obs_shape)
        _fill_buffer(a, n_episodes=2, ep_len=20)

        path = tmp_path / "buffer.npz"
        a.save_to_file(path)

        b = PrioritizedReplayBuffer(capacity=400, obs_shape=obs_shape)
        try:
            b.load_from_file(path)
        except ValueError as e:
            assert "capacity mismatch" in str(e)
        else:
            raise AssertionError("expected ValueError for mismatched capacity")

    def test_partial_buffer_save_does_not_pad(self, tmp_path):
        """Saving an under-filled buffer should write only `size` entries."""
        obs_shape = (3, 8, 8)
        buf = PrioritizedReplayBuffer(capacity=10_000, obs_shape=obs_shape)
        _fill_buffer(buf, n_episodes=2, ep_len=20)

        path = tmp_path / "buffer.npz"
        buf.save_to_file(path)
        data = np.load(path)
        assert data["states"].shape[0] == buf.size
        assert data["states"].shape[0] < buf.capacity
