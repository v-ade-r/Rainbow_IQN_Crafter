"""Interactive Gradio demo for showcasing the trained Rainbow-IQN agent."""

import argparse
import logging
import time
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.diagnostics import GradCAM
from src.utils.inference import (
    DEFAULT_DISPLAY_SCALE,
    load_agent_for_inference,
    obs_to_rgb,
    upscale_for_display,
)

log = logging.getLogger(__name__)

CRAFTER_ACTIONS = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]


def load_agent(checkpoint_path: str, device: str = "cpu"):
    """Load agent and environment (same settings as training / evaluate.py)."""
    dev = torch.device(device if torch.cuda.is_available() else device)
    return load_agent_for_inference(checkpoint_path, dev)


def make_quantile_plot(
    q_values: np.ndarray,
    action_names: list[str],
    figsize: tuple[float, float] = (5.0, 3.2),
) -> plt.Figure:
    """Bar chart of mean Q-values per action."""
    fig, ax = plt.subplots(figsize=figsize)
    n_actions = len(action_names)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_actions))
    bars = ax.bar(range(n_actions), q_values, color=colors)

    best_action = int(np.argmax(q_values))
    bars[best_action].set_edgecolor("red")
    bars[best_action].set_linewidth(3)

    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(action_names, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Mean Q-value")
    ax.set_title("Quantile Distribution (mean over taus)")
    fig.tight_layout()
    return fig


def build_demo(
    checkpoint_path: str,
    device: str = "cpu",
    display_scale: int = DEFAULT_DISPLAY_SCALE,
):
    """Build and return the Gradio interface."""
    agent, env = load_agent(checkpoint_path, device)
    dev = torch.device(device)
    display_size = 64 * display_scale

    grad_cam = GradCAM(agent.online_net.encoder)

    state = {
        "obs": None,
        "done": True,
        "episode_reward": 0.0,
        "step": 0,
        "achievements": {},
    }

    def _show(frame: np.ndarray) -> np.ndarray:
        return upscale_for_display(frame, scale=display_scale)

    def _blank_saliency() -> np.ndarray:
        """Placeholder when saliency is off."""
        return np.zeros((display_size, display_size, 3), dtype=np.uint8)

    def _saliency_overlay(obs: np.ndarray) -> np.ndarray:
        """Grad-CAM heatmap blended onto the current RGB frame."""
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev) / 255.0
        heatmap = grad_cam.compute(obs_t, agent)
        rgb = obs_to_rgb(obs)
        overlay = grad_cam.overlay_on_observation(rgb, heatmap, alpha=0.45)
        return _show(overlay)

    def _q_plot_for_obs(obs: np.ndarray) -> plt.Figure:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev) / 255.0
            tau = torch.rand(1, agent.n_quantiles_eval, device=dev)
            q_ext, _ = agent.online_net(obs_t, tau)
            mean_q = q_ext.mean(dim=1).cpu().numpy().squeeze()
        return make_quantile_plot(mean_q, CRAFTER_ACTIONS)

    def _render(
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        show_saliency: bool,
        status_prefix: str = "",
    ) -> tuple[np.ndarray, plt.Figure, np.ndarray, str]:
        achievements_str = ", ".join(
            f"{k}: {v}" for k, v in sorted(state["achievements"].items()) if v > 0
        )
        status = (
            f"{status_prefix}"
            f"Step: {state['step']} | Action: {CRAFTER_ACTIONS[action]} | "
            f"Reward: {reward:.1f} | Total: {state['episode_reward']:.1f} | "
            f"Done: {done}\n"
            f"Achievements: {achievements_str or 'none yet'}"
        )
        saliency = _saliency_overlay(obs) if show_saliency else _blank_saliency()
        return _show(obs_to_rgb(obs)), _q_plot_for_obs(obs), saliency, status

    def run_step(show_saliency: bool):
        if state["done"]:
            state["obs"] = env.reset()
            state["done"] = False
            state["episode_reward"] = 0.0
            state["step"] = 0
            state["achievements"] = {}

        obs = state["obs"]
        action = agent.act(obs, deterministic=True)
        _obs, reward, done, info = env.step(action)
        state["obs"] = _obs
        state["done"] = done
        state["episode_reward"] += reward
        state["step"] += 1

        if "achievements" in info:
            for name, count in info["achievements"].items():
                if count > 0:
                    state["achievements"][name] = state["achievements"].get(name, 0) + count

        return _render(obs, action, reward, done, show_saliency)

    def run_full_episode(show_saliency: bool, frame_delay_ms: int):
        """Stream one frame per env step so the UI updates live."""
        state["obs"] = env.reset()
        state["done"] = False
        state["episode_reward"] = 0.0
        state["step"] = 0
        state["achievements"] = {}

        delay_s = max(frame_delay_ms, 0) / 1000.0
        max_steps = 5000

        while not state["done"] and state["step"] < max_steps:
            obs = state["obs"]
            action = agent.act(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)

            state["step"] += 1
            state["episode_reward"] += reward
            state["done"] = done
            state["obs"] = next_obs

            if "achievements" in info:
                for name, count in info["achievements"].items():
                    if count > 0:
                        state["achievements"][name] = (
                            state["achievements"].get(name, 0) + count
                        )

            yield _render(obs, action, reward, done, show_saliency)

            if delay_s > 0:
                time.sleep(delay_s)

        yield _render(
            state["obs"],
            action,
            0.0,
            True,
            show_saliency,
            status_prefix=f"Episode complete! ({state['step']} steps)\n",
        )

    tsne_path = Path(checkpoint_path).parent / "tsne_latent.png"
    tsne_image = str(tsne_path) if tsne_path.exists() else None

    with gr.Blocks(title="Rainbow-IQN Crafter Agent") as demo:
        gr.Markdown("# Rainbow-IQN + RND Agent — Crafter Demo")
        gr.Markdown(
            "Step through episodes or **Run Full Episode** to stream gameplay live. "
            "**Saliency (Grad-CAM)** highlights image regions that most influence "
            "the agent's chosen action (warmer colors = stronger influence on Q-values)."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                show_saliency = gr.Checkbox(
                    label="Show Saliency (Grad-CAM)",
                    value=False,
                )
                frame_delay = gr.Slider(
                    minimum=0,
                    maximum=200,
                    value=40,
                    step=10,
                    label="Stream delay (ms per frame)",
                    info="Only affects Run Full Episode. 0 = as fast as possible.",
                )
                step_btn = gr.Button("Step", variant="primary")
                episode_btn = gr.Button("Run Full Episode (live stream)", variant="secondary")
                q_plot = gr.Plot(label="Q-Value Distribution")
                status_text = gr.Textbox(label="Status", lines=4, interactive=False)

            with gr.Column(scale=2):
                frame_display = gr.Image(
                    label="Agent View",
                    height=display_size,
                    width=display_size,
                    interactive=False,
                )
                saliency_display = gr.Image(
                    label="Saliency — what drives the chosen action",
                    height=display_size,
                    width=display_size,
                    interactive=False,
                )

        if tsne_image:
            gr.Markdown("### Latent Space (t-SNE)")
            gr.Image(value=tsne_image, label="Encoder Latent Space")

        step_btn.click(
            fn=run_step,
            inputs=[show_saliency],
            outputs=[frame_display, q_plot, saliency_display, status_text],
        )
        episode_btn.click(
            fn=run_full_episode,
            inputs=[show_saliency, frame_delay],
            outputs=[frame_display, q_plot, saliency_display, status_text],
        )

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to agent checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_DISPLAY_SCALE,
        help="Pixel-art upscale factor for the game view (default: 8 → 512×512)",
    )
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(args.checkpoint, device=args.device, display_scale=args.scale)
    demo.launch(server_port=args.port, share=args.share)
