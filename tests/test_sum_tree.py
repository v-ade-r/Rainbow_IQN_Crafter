"""Unit tests for SumTree: correctness of totals and sampling distribution."""

import numpy as np
from scipy import stats

from src.components.sum_tree import SumTree


class TestSumTreeBasics:
    def test_total_matches_sum_of_priorities(self):
        tree = SumTree(capacity=1024)
        priorities = np.random.exponential(scale=5.0, size=1000)
        for p in priorities:
            tree.add(float(p))

        assert np.isclose(tree.total(), priorities.sum(), rtol=1e-9)

    def test_total_after_overwrap(self):
        """When capacity is exceeded, old entries are overwritten."""
        tree = SumTree(capacity=100)
        for _i in range(150):
            tree.add(1.0)
        assert tree.n_entries == 100
        assert np.isclose(tree.total(), 100.0, rtol=1e-9)

    def test_update_propagates_correctly(self):
        tree = SumTree(capacity=8)
        for _ in range(8):
            tree.add(1.0)
        assert np.isclose(tree.total(), 8.0)

        tree_idx = tree.tree_idx_for_data(3)
        tree.update(tree_idx, 10.0)
        assert np.isclose(tree.total(), 17.0)

    def test_min_priority(self):
        tree = SumTree(capacity=100)
        for p in [5.0, 3.0, 7.0, 1.0, 9.0]:
            tree.add(p)
        assert np.isclose(tree.min_priority(), 1.0)


class TestSumTreeSampling:
    def test_sampling_proportional_to_priority(self):
        """Chi-squared test: sampling frequency matches priority distribution."""
        tree = SumTree(capacity=5)
        priorities = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for p in priorities:
            tree.add(p)

        n_samples = 50_000
        counts = np.zeros(5)
        for _ in range(n_samples):
            _, data_indices = tree.sample(1)
            counts[data_indices[0]] += 1

        expected_freq = priorities / priorities.sum()
        expected_counts = expected_freq * n_samples

        chi2, p_value = stats.chisquare(counts, f_exp=expected_counts)
        assert p_value > 0.001, (
            f"Sampling distribution deviates from priorities: chi2={chi2:.1f}, p={p_value:.6f}"
        )

    def test_zero_priority_never_sampled(self):
        tree = SumTree(capacity=4)
        tree.add(0.0)
        tree.add(1.0)
        tree.add(0.0)
        tree.add(1.0)

        for _ in range(1000):
            _, data_indices = tree.sample(1)
            assert data_indices[0] in [1, 3]

    def test_single_item_always_sampled(self):
        tree = SumTree(capacity=10)
        tree.add(5.0)

        for _ in range(100):
            _, data_indices = tree.sample(1)
            assert data_indices[0] == 0
