"""Latent space visualization (t-SNE/UMAP) and saliency maps (Grad-CAM)."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_latent_features(
    agent: Any,
    env: Any,
    n_episodes: int = 50,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, list[dict]]:
    """Run agent for n_episodes, collect encoder outputs and Crafter state info.

    Returns:
        features: (N, feature_dim) array of encoder outputs.
        metadata: list of dicts with Crafter state labels per observation.
    """
    device = torch.device(device)
    encoder = agent.online_net.encoder
    encoder.eval()

    all_features = []
    all_metadata = []

    for _ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                feat = encoder(obs_t).cpu().numpy().squeeze(0)

            all_features.append(feat)

            action = agent.act(obs)
            obs, _reward, done, info = env.step(action)

            meta = {}
            if "inventory" in info:
                meta["has_pickaxe"] = info["inventory"].get("wood_pickaxe", 0) > 0
                meta["has_sword"] = info["inventory"].get("wood_sword", 0) > 0
            if "health" in info:
                meta["low_health"] = info["health"] < 5
            if "sleeping" in info:
                meta["night"] = info["sleeping"]
            all_metadata.append(meta)

    encoder.train()
    return np.array(all_features), all_metadata


def compute_tsne(
    features: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> np.ndarray:
    """Project features to 2D using t-SNE.

    Returns:
        (N, 2) array of 2D coordinates.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    return tsne.fit_transform(features)


def compute_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Project features to 2D using UMAP.

    Returns:
        (N, 2) array of 2D coordinates.
    """
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(features)


def plot_latent_space(
    coords_2d: np.ndarray,
    metadata: list[dict],
    label_key: str = "night",
    title: str = "Latent Space",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot 2D latent space colored by a semantic label.

    Args:
        coords_2d: (N, 2) from t-SNE or UMAP.
        metadata: list of dicts per observation.
        label_key: which metadata key to color by.
        title: plot title.
        save_path: if provided, save the figure to this path.
    """
    labels = np.array([m.get(label_key, False) for m in metadata], dtype=bool)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(
        coords_2d[~labels, 0],
        coords_2d[~labels, 1],
        c="steelblue",
        alpha=0.4,
        s=5,
        label=f"not {label_key}",
    )
    ax.scatter(
        coords_2d[labels, 0],
        coords_2d[labels, 1],
        c="tomato",
        alpha=0.6,
        s=8,
        label=label_key,
    )
    ax.set_title(title)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


class GradCAM:
    """Grad-CAM saliency for any convolutional encoder (CNN or IMPALA).

    Highlights image regions that most influence the agent's chosen action
    (gradient-weighted activation map on the last Conv2d layer).
    """

    def __init__(self, encoder: nn.Module) -> None:
        self.encoder = encoder
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        last_conv = None
        for module in encoder.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in encoder")

        last_conv.register_forward_hook(self._save_activation)
        last_conv.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def compute(
        self,
        obs: torch.Tensor,
        agent: Any,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for the best action's Q-value.

        Args:
            obs: (1, C, H, W) normalized observation tensor.
            agent: the agent (needs online_net and quantile params).

        Returns:
            (H, W) heatmap array in [0, 1].
        """
        self.encoder.eval()

        obs.requires_grad_(True)
        tau = torch.rand(1, agent.n_quantiles_eval, device=obs.device)
        q_ext, _ = agent.online_net(obs, tau)

        mean_q = q_ext.mean(dim=1)
        best_action = mean_q.argmax(dim=1)
        target_q = mean_q[0, best_action]

        agent.online_net.zero_grad()
        target_q.backward()

        gradients = self._gradients
        activations = self._activations
        if gradients is None or activations is None:
            return np.zeros(obs.shape[2:], dtype=np.float32)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=obs.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_on_observation(
        self,
        obs_raw: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay Grad-CAM heatmap on a raw observation image.

        Args:
            obs_raw: (H, W) or (H, W, 3) uint8 image.
            heatmap: (H, W) float in [0, 1].
            alpha: blend factor.

        Returns:
            (H, W, 3) uint8 blended image.
        """
        cmap = plt.cm.jet(heatmap)[:, :, :3]
        cmap = (cmap * 255).astype(np.uint8)

        obs_rgb = np.stack([obs_raw] * 3, axis=-1) if obs_raw.ndim == 2 else obs_raw

        blended = (alpha * cmap + (1 - alpha) * obs_rgb).astype(np.uint8)
        return blended
