"""Environment wrappers for Crafter: preprocess, frame stacking, action repeat."""

from collections import deque

import cv2
import numpy as np


class ImagePreprocess:
    """Convert HWC uint8 observations to CHW, optionally grayscale/resize.

    Crafter natively yields (H, W, 3) uint8 frames at the requested size.
    This wrapper unifies the pipeline so downstream wrappers/networks can
    assume (C, H, W) uint8 tensors regardless of color mode.
    """

    def __init__(self, env, size: int = 64, grayscale: bool = False) -> None:
        self.env = env
        self.size = size
        self.grayscale = grayscale
        self.action_space = env.action_space
        c = 1 if grayscale else 3
        self.observation_shape = (c, size, size)

    def _process(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[:2] != (self.size, self.size):
            obs = cv2.resize(
                obs, (self.size, self.size), interpolation=cv2.INTER_AREA
            )
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            return obs[np.newaxis, :, :]
        return np.ascontiguousarray(obs.transpose(2, 0, 1))

    def reset(self):
        return self._process(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process(obs), reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class ActionRepeat:
    """Repeat each action k times, summing rewards."""

    def __init__(self, env, k: int = 2) -> None:
        self.env = env
        self.k = k
        self.action_space = env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        for _i in range(self.k):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class FrameStack:
    """Stack the last k frames along the channel dimension.

    Uses lazy deque-based stacking. Output shape: (k * C, H, W).
    """

    def __init__(self, env, k: int = 4) -> None:
        self.env = env
        self.k = k
        self.frames: deque = deque(maxlen=k)
        self.action_space = env.action_space

    @property
    def observation_shape(self) -> tuple[int, ...]:
        c, h, w = self.env.observation_shape
        return (c * self.k, h, w)

    def reset(self):
        obs = self.env.reset()
        for _i in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_crafter_env(
    frame_stack: int = 4,
    action_repeat: int = 1,
    image_size: int = 64,
    grayscale: bool = False,
) -> FrameStack:
    """Create a fully wrapped Crafter environment.

    Default pipeline: RGB 64x64, no action repeat, 4-frame stack.
    Matches the Hafner (2021) benchmark preprocessing.
    """
    import crafter

    env = crafter.Env()
    if action_repeat > 1:
        env = ActionRepeat(env, k=action_repeat)
    env = ImagePreprocess(env, size=image_size, grayscale=grayscale)
    env = FrameStack(env, k=frame_stack)
    return env
