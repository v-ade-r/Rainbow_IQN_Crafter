"""Record trained agent gameplay to a GIF (for README / demos).

Uses the same env settings and policy as evaluate.py (RGB 64×64, frame stack 4,
deterministic extrinsic-only action selection).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.inference import (
    DEFAULT_DISPLAY_SCALE,
    load_agent_for_inference,
    obs_to_rgb,
    upscale_for_display,
)

log = logging.getLogger(__name__)


def record_episode_gif(
    checkpoint_path: str,
    output_path: str | Path,
    device: str = "cuda",
    max_steps: int = 500,
    fps: int = 10,
    frame_stack: int = 4,
    action_repeat: int = 1,
    grayscale: bool = False,
    encoder_type: str = "impala",
    display_scale: int = DEFAULT_DISPLAY_SCALE,
) -> dict:
    agent, env = load_agent_for_inference(
        checkpoint_path,
        device,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        grayscale=grayscale,
        encoder_type=encoder_type,
    )

    obs = env.reset()
    frames: list[np.ndarray] = []
    ep_reward = 0.0
    done = False
    step = 0
    achievements: dict[str, int] = {}

    while not done and step < max_steps:
        frames.append(obs_to_rgb(obs))
        action = agent.act(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        step += 1

    frames.append(obs_to_rgb(obs))

    if "achievements" in info:
        achievements = {k: v for k, v in info["achievements"].items() if v > 0}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = int(1000 / max(fps, 1))
    pil_frames = [
        Image.fromarray(upscale_for_display(f.astype(np.uint8), scale=display_scale))
        for f in frames
    ]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )

    log.info(
        f"Saved {len(frames)} frames → {output_path} "
        f"({step} steps, reward={ep_reward:.1f})"
    )
    if achievements:
        log.info(f"Achievements: {', '.join(sorted(achievements))}")

    return {
        "output_path": str(output_path),
        "n_frames": len(frames),
        "steps": step,
        "episode_reward": ep_reward,
        "achievements": achievements,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Record agent episode to GIF")
    parser.add_argument("checkpoint", type=str, help="Path to agent checkpoint")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results/demo_episode.gif",
        help="Output GIF path (default: results/demo_episode.gif)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_DISPLAY_SCALE,
        help="Pixel-art upscale factor (default: 8 → 512×512 frames)",
    )
    args = parser.parse_args()

    record_episode_gif(
        args.checkpoint,
        args.output,
        device=args.device,
        max_steps=args.max_steps,
        fps=args.fps,
        display_scale=args.scale,
    )
