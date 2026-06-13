"""SumTree data structure for O(log n) prioritized sampling."""

import numpy as np


class SumTree:
    """Binary tree where each parent is the sum of its children.

    Leaf nodes store priorities. Internal nodes store partial sums,
    enabling O(log n) sampling proportional to priority and O(log n) updates.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    def total(self) -> float:
        return float(self.tree[0])

    def min_priority(self) -> float:
        """Return the minimum priority among stored entries."""
        leaf_start = self.capacity - 1
        if self.n_entries == 0:
            return 0.0
        return float(np.min(self.tree[leaf_start : leaf_start + self.n_entries]))

    def add(self, priority: float) -> int:
        """Add a new priority and return the data index it was assigned to."""
        data_idx = self.data_pointer
        tree_idx = data_idx + self.capacity - 1

        self.update(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return data_idx

    def update(self, tree_idx: int, priority: float) -> None:
        """Update the priority at a given tree index and propagate up."""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def _retrieve(self, idx: int, value: float) -> int:
        """Walk down the tree to find the leaf for a given cumulative value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample indices proportional to stored priorities.

        Uses stratified sampling: divides the total priority into equal
        segments and samples uniformly within each segment.

        Returns:
            tree_indices: array of tree indices (for later update calls).
            data_indices: array of data indices (for buffer lookups).
        """
        segment = self.total() / batch_size
        tree_indices = np.empty(batch_size, dtype=np.int64)
        data_indices = np.empty(batch_size, dtype=np.int64)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            tree_idx = self._retrieve(0, value)
            data_idx = tree_idx - (self.capacity - 1)
            tree_indices[i] = tree_idx
            data_indices[i] = data_idx

        return tree_indices, data_indices

    def get_priority(self, tree_idx: int) -> float:
        return float(self.tree[tree_idx])

    def tree_idx_for_data(self, data_idx: int) -> int:
        return data_idx + self.capacity - 1

    def state_dict(self) -> dict:
        """Return all state needed to faithfully restore the tree."""
        return {
            "capacity": int(self.capacity),
            "tree": self.tree.copy(),
            "data_pointer": int(self.data_pointer),
            "n_entries": int(self.n_entries),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore tree from a state dict produced by state_dict().

        The capacity must match the current tree (we pre-allocate buffers
        elsewhere of identical size).
        """
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"SumTree capacity mismatch: have {self.capacity}, got {state['capacity']}"
            )
        tree = np.asarray(state["tree"], dtype=np.float64)
        if tree.shape != self.tree.shape:
            raise ValueError(
                f"SumTree array shape mismatch: have {self.tree.shape}, got {tree.shape}"
            )
        self.tree = tree
        self.data_pointer = int(state["data_pointer"])
        self.n_entries = int(state["n_entries"])
