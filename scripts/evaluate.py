"""Evaluate a trained Rainbow-IQN agent on Crafter."""

import argparse
import logging

import numpy as np
import torch
from crafter.constants import achievements as CRAFTER_ACHIEVEMENTS

from src.agents.rainbow_iqn_agent import RainbowIQNAgent
from src.envs.wrappers import make_crafter_env
from src.utils.logging import RollingCrafterTracker

log = logging.getLogger(__name__)


def format_evaluation_report(tracker: RollingCrafterTracker) -> str:
    """DreamerV3-style Crafter evaluation summary."""
    rates = tracker.achievement_rates()
    n_episodes = tracker.n_episodes()
    lines = [
        f"Epizody: {n_episodes}",
        f"Crafter Score: {tracker.crafter_score():.2f}%",
        f"Mean reward:   {tracker.mean_reward():.2f}",
        f"Mean length:   {tracker.mean_length():.1f}",
        f"Achievements/ep: {tracker.mean_achievements_per_episode():.1f}",
        f"Unique unlocked: {tracker.unique_unlocked_in_window()}/{len(CRAFTER_ACHIEVEMENTS)}",
        "-" * 60,
        f"{'Achievement':<28}Unlock %",
    ]
    for name in CRAFTER_ACHIEVEMENTS:
        pct = rates.get(name, 0.0) * 100.0
        lines.append(f"{name:<28}{pct:6.1f}%")
    lines.append("=" * 60)
    return "\n".join(lines)


def evaluate(
    checkpoint_path: str,
    n_episodes: int = 100,
    device: str = "cuda",
    frame_stack: int = 4,
    action_repeat: int = 1,
    grayscale: bool = False,
    encoder_type: str = "impala",
) -> tuple[dict, RollingCrafterTracker]:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    env = make_crafter_env(
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        image_size=64,
        grayscale=grayscale,
    )

    obs_shape = env.observation_shape
    n_actions = env.action_space.n

    # rnd_beta=0 skips building the RND module -- we only load online_net
    # weights; the intrinsic head is still present in the network but unused
    # for action selection when rnd_beta=0. buffer_size is minimised because
    # the replay buffer is irrelevant for evaluation and a full-size RGB
    # buffer would require tens of GB of RAM.
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

    tracker = RollingCrafterTracker(window=n_episodes)

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_length = 0
        done = False

        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1

        achievements = info.get("achievements", {})
        tracker.push(achievements, ep_reward, ep_length)

        if (ep + 1) % 10 == 0:
            log.info(
                f"Ep {ep + 1}/{n_episodes} | "
                f"CS={tracker.crafter_score():5.2f}% "
                f"AchU={tracker.unique_unlocked_in_window():>2}/{len(CRAFTER_ACHIEVEMENTS)} "
                f"R={tracker.mean_reward():5.2f} L={tracker.mean_length():>4.0f}"
            )

    rates = tracker.achievement_rates()
    results = {
        "mean_reward": tracker.mean_reward(),
        "std_reward": float(np.std(list(tracker.rewards))) if tracker.rewards else 0.0,
        "mean_length": tracker.mean_length(),
        "mean_achievements_per_episode": tracker.mean_achievements_per_episode(),
        "n_episodes": n_episodes,
        "crafter_score": tracker.crafter_score(),
        "unique_achievements": tracker.unique_unlocked_in_window(),
        "achievement_rates": {
            name: rates.get(name, 0.0) for name in CRAFTER_ACHIEVEMENTS
        },
    }

    return results, tracker


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (use 429 for DreamerV3-comparable runs)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _results, tracker = evaluate(
        args.checkpoint, n_episodes=args.episodes, device=args.device
    )
    print()
    print(format_evaluation_report(tracker))
