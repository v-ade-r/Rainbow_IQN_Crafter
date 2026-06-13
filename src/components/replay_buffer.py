"""Prioritized Experience Replay with N-step accumulation and dual rewards.

Stores transitions with both extrinsic and intrinsic rewards separately,
accumulated through the same n-step window. The agent can then target
Q_ext and Q_int independently with matched discount structure.
"""

from collections import deque
from pathlib import Path

import numpy as np

from src.components.sum_tree import SumTree


class NStepAccumulator:
    """Accumulates transitions to compute n-step discounted returns for two
    reward streams (extrinsic and intrinsic).

    Holds up to `n` transitions and emits compound transitions:
        R_ext = sum_{k=0}^{n-1} gamma^k * r_ext_k
        R_int = sum_{k=0}^{n-1} gamma^k * r_int_k
        s_next = s_{t+n}
    """

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n)

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward_ext: float,
        reward_int: float,
        next_state: np.ndarray,
        done: bool,
    ) -> tuple[np.ndarray, int, float, float, np.ndarray, bool, int] | None:
        """Append a 1-step transition. Returns an n-step transition when ready.

        On episode end (done=True), does NOT flush here -- the caller must
        invoke flush_remaining() to drain ALL pending partial n-step
        transitions.

        Returned tuple: (s_0, a_0, R_ext, R_int, s_n, d_n, length).
        `length` is the EFFECTIVE number of one-step rewards aggregated into
        R_ext/R_int (1..n). Required so the agent can apply the correct
        gamma**length when bootstrapping, instead of always assuming n.
        """
        self.buffer.append((state, action, reward_ext, reward_int, next_state, done))

        if done:
            return None

        if len(self.buffer) == self.n:
            return self._make_nstep()

        return None

    def _make_nstep(
        self,
    ) -> tuple[np.ndarray, int, float, float, np.ndarray, bool, int]:
        """Construct a single n-step transition from the current buffer.

        Returns the bootstrap state taken from the LAST transition that
        contributed a reward (so a mid-window terminal does not leak features
        from the next episode), plus the effective length used to compose the
        return.
        """
        R_ext = 0.0
        R_int = 0.0
        last_idx = len(self.buffer) - 1
        for k in range(len(self.buffer)):
            _, _, r_ext_k, r_int_k, _, d_k = self.buffer[k]
            discount = self.gamma**k
            R_ext += discount * r_ext_k
            R_int += discount * r_int_k
            if d_k:
                last_idx = k
                break

        s_0, a_0, _, _, _, _ = self.buffer[0]
        _, _, _, _, s_n, d_n = self.buffer[last_idx]
        length = last_idx + 1

        return s_0, a_0, R_ext, R_int, s_n, d_n, length

    def flush_remaining(
        self,
    ) -> list[tuple[np.ndarray, int, float, float, np.ndarray, bool, int]]:
        """Flush all remaining partial sequences at episode boundary."""
        results = []
        while len(self.buffer) > 0:
            results.append(self._make_nstep())
            self.buffer.popleft()
        return results

    def reset(self) -> None:
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer backed by a SumTree.

    Stores (state, action, r_ext, r_int, next_state, done) tuples with
    priorities based on TD errors. Supports dual rewards for separate
    extrinsic/intrinsic value learning.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 1_000_000,
        n_step: int = 3,
        gamma: float = 0.99,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_count = 0

        self.tree = SumTree(capacity)
        self.n_step_acc = NStepAccumulator(n=n_step, gamma=gamma)

        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards_ext = np.zeros(capacity, dtype=np.float32)
        self.rewards_int = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        # Per-transition effective n-step length (1..n_step). Needed so the
        # agent applies gamma**actual_length on bootstrap, not always n.
        self.n_step_lengths = np.zeros(capacity, dtype=np.uint8)

        self.max_priority = 1.0
        self.size = 0

    @property
    def beta(self) -> float:
        fraction = min(self.frame_count / max(self.beta_frames, 1), 1.0)
        return self.beta_start + (1.0 - self.beta_start) * fraction

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward_ext: float,
        reward_int: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a 1-step transition through the n-step accumulator."""
        nstep = self.n_step_acc.append(
            state, action, reward_ext, reward_int, next_state, done
        )
        if nstep is not None:
            self._store(*nstep)

        if done:
            for remaining in self.n_step_acc.flush_remaining():
                self._store(*remaining)

    def _store(
        self,
        state: np.ndarray,
        action: int,
        reward_ext: float,
        reward_int: float,
        next_state: np.ndarray,
        done: bool,
        length: int,
    ) -> None:
        """Store an n-step transition with current max priority."""
        data_idx = self.tree.add(self.max_priority**self.alpha)

        self.states[data_idx] = state
        self.actions[data_idx] = action
        self.rewards_ext[data_idx] = reward_ext
        self.rewards_int[data_idx] = reward_int
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = done
        self.n_step_lengths[data_idx] = length

        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Sample a batch weighted by priority.

        Returns:
            states, actions, rewards_ext, rewards_int, next_states, dones,
            n_step_lengths, is_weights, tree_indices
        """
        tree_indices, data_indices = self.tree.sample(batch_size)

        priorities = np.array(
            [self.tree.get_priority(ti) for ti in tree_indices], dtype=np.float64
        )
        sampling_probs = priorities / self.tree.total()

        beta = self.beta
        is_weights = (self.size * sampling_probs) ** (-beta)
        is_weights /= is_weights.max()
        is_weights = is_weights.astype(np.float32)

        states = self.states[data_indices]
        actions = self.actions[data_indices]
        rewards_ext = self.rewards_ext[data_indices]
        rewards_int = self.rewards_int[data_indices]
        next_states = self.next_states[data_indices]
        dones = self.dones[data_indices]
        n_step_lengths = self.n_step_lengths[data_indices]

        self.frame_count += batch_size

        return (
            states,
            actions,
            rewards_ext,
            rewards_int,
            next_states,
            dones,
            n_step_lengths,
            is_weights,
            tree_indices,
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities from absolute n-step TD errors."""
        raw_priorities = np.abs(td_errors) + 1e-6
        priorities = raw_priorities**self.alpha if self.alpha > 0 else np.ones_like(raw_priorities)
        for tree_idx, raw_p, priority in zip(
            tree_indices, raw_priorities, priorities, strict=True
        ):
            self.tree.update(int(tree_idx), float(priority))
            self.max_priority = max(self.max_priority, float(raw_p))

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_to_file(self, path: str | Path) -> None:
        """Persist full buffer state to a single compressed .npz file.

        Crafter uses uint8 image observations with significant spatial
        redundancy, so np.savez_compressed gets ~50% reduction at the
        cost of ~1-2 minutes per save for a 250k buffer. We slice arrays
        to `self.size` to avoid writing zero-padding when the buffer is
        not yet full.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n = self.size
        tree_state = self.tree.state_dict()

        np.savez_compressed(
            path,
            states=self.states[:n],
            actions=self.actions[:n],
            rewards_ext=self.rewards_ext[:n],
            rewards_int=self.rewards_int[:n],
            next_states=self.next_states[:n],
            dones=self.dones[:n],
            n_step_lengths=self.n_step_lengths[:n],
            size=np.int64(self.size),
            capacity=np.int64(self.capacity),
            max_priority=np.float64(self.max_priority),
            frame_count=np.int64(self.frame_count),
            tree_array=tree_state["tree"],
            tree_data_pointer=np.int64(tree_state["data_pointer"]),
            tree_n_entries=np.int64(tree_state["n_entries"]),
        )

    def load_from_file(self, path: str | Path) -> None:
        """Restore buffer state from a file produced by save_to_file().

        Capacity must match the current buffer instance (we pre-allocated
        contiguous arrays in __init__). Per-step n-step accumulator is
        intentionally NOT serialized: at most n-1 = 2 transitions can be
        pending in the accumulator, dwarfed by the saved buffer size.
        """
        path = Path(path)
        data = np.load(path)

        cap_saved = int(data["capacity"])
        if cap_saved != self.capacity:
            raise ValueError(
                f"Replay buffer capacity mismatch: have {self.capacity}, saved {cap_saved}"
            )

        n = int(data["size"])
        self.states[:n] = data["states"]
        self.actions[:n] = data["actions"]
        self.rewards_ext[:n] = data["rewards_ext"]
        self.rewards_int[:n] = data["rewards_int"]
        self.next_states[:n] = data["next_states"]
        self.dones[:n] = data["dones"]
        self.n_step_lengths[:n] = data["n_step_lengths"]

        self.size = n
        self.max_priority = float(data["max_priority"])
        self.frame_count = int(data["frame_count"])

        self.tree.load_state_dict(
            {
                "capacity": self.capacity,
                "tree": data["tree_array"],
                "data_pointer": int(data["tree_data_pointer"]),
                "n_entries": int(data["tree_n_entries"]),
            }
        )
