"""Shared helpers for inference, demo, and GIF recording."""

import cv2
import numpy as np
import torch

from src.agents.rainbow_iqn_agent import RainbowIQNAgent
from src.envs.wrappers import make_crafter_env

# Display-only upscale for 64×64 Crafter frames. Integer nearest-neighbor keeps
# crisp pixel-art edges; bilinear/lanczos would blur the image without adding detail.
DEFAULT_DISPLAY_SCALE = 8


def obs_to_rgb(obs: np.ndarray) -> np.ndarray:
    """Latest RGB frame from a frame-stacked CHW observation."""
    c, _, _ = obs.shape
    if c >= 3 and c % 3 == 0:
        return obs[-3:].transpose(1, 2, 0)
    gray = obs[-1]
    return np.stack([gray, gray, gray], axis=-1)


def upscale_for_display(frame: np.ndarray, scale: int = DEFAULT_DISPLAY_SCALE) -> np.ndarray:
    """Upscale a pixel-art RGB frame for demo/GIF output (not used in training).

    Crafter renders at 64×64. Nearest-neighbor at an integer scale (default 8×
    → 512×512) preserves sharp tiles; smooth interpolation only adds blur.
    """
    if scale <= 1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(
        frame,
        (w * scale, h * scale),
        interpolation=cv2.INTER_NEAREST,
    )


def load_agent_for_inference(
    checkpoint_path: str,
    device: torch.device | str,
    frame_stack: int = 4,
    action_repeat: int = 1,
    grayscale: bool = False,
    encoder_type: str = "impala",
) -> tuple[RainbowIQNAgent, object]:
    """Build env + agent with the same defaults as training / evaluate.py."""
    dev = torch.device(device)
    env = make_crafter_env(
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        image_size=64,
        grayscale=grayscale,
    )
    obs_shape = env.observation_shape
    n_actions = env.action_space.n

    agent = RainbowIQNAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        device=dev,
        in_channels=obs_shape[0],
        rnd_beta=0.0,
        encoder_type=encoder_type,
        buffer_size=100,
    )
    agent.load(checkpoint_path)
    agent.online_net.eval()
    return agent, env
