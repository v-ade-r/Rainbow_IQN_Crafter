"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class BaseAgent(ABC):
    """Interface all agents must implement."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """Select an action given an observation."""

    @abstractmethod
    def learn(self) -> dict[str, float]:
        """Perform one learning step. Returns metrics dict."""

    def save(self, path: str | Path) -> None:
        """Save agent state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._get_save_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load agent state from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self._load_save_dict(state)

    @abstractmethod
    def _get_save_dict(self) -> dict:
        """Return a dict of everything to persist."""

    @abstractmethod
    def _load_save_dict(self, state: dict) -> None:
        """Restore agent from a persisted dict."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device this agent's tensors live on."""
