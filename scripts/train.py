"""Main training loop for Rainbow-IQN + RND on Crafter.

Supports:
  - Resuming from a checkpoint: set resume_checkpoint=/path/to/agent.pt
  - Quick pipeline test: set test_run=true (5k steps, tiny buffer, no W&B)
"""

import logging
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from src.agents.rainbow_iqn_agent import RainbowIQNAgent
from src.envs.wrappers import make_crafter_env
from src.utils.logging import DiagnosticLogger, RollingCrafterTracker

log = logging.getLogger(__name__)

TEST_RUN_OVERRIDES = {
    "total_steps": 5_000,
    "training_starts": 500,
    "checkpoint_freq": 2_500,
    "log_freq": 500,
}
TEST_RUN_AGENT_OVERRIDES = {
    "buffer_size": 5_000,
    "batch_size": 8,
    "rnd_warmup_steps": 200,
}


def _apply_test_run_overrides(cfg: DictConfig) -> None:
    """Override config values for a quick pipeline smoke test."""
    with open_dict(cfg):
        for k, v in TEST_RUN_OVERRIDES.items():
            cfg[k] = v
        for k, v in TEST_RUN_AGENT_OVERRIDES.items():
            cfg.agent[k] = v
        cfg.logger.mode = "disabled"
    log.info("TEST RUN mode: using reduced settings for quick pipeline verification")


def _extract_step_from_checkpoint(path: str) -> int:
    """Try to extract the step number from a checkpoint filename like agent_step_150000.pt."""
    name = Path(path).stem
    parts = name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


def _buffer_path_for_checkpoint(ckpt_path: Path) -> Path:
    """Convention: agent_step_X.pt sits next to buffer_step_X.npz."""
    name = ckpt_path.name.replace("agent_", "buffer_").replace(".pt", ".npz")
    return ckpt_path.parent / name


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def train(cfg: DictConfig) -> None:
    if cfg.test_run:
        _apply_test_run_overrides(cfg)
 
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    project_root = Path(get_original_cwd())
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log.info(f"Device: {device}")
    log.info(f"Checkpoints: {ckpt_dir}")

    env = make_crafter_env(
        frame_stack=cfg.env.frame_stack,
        action_repeat=cfg.env.action_repeat,
        image_size=cfg.env.image_size,
        grayscale=cfg.env.grayscale,
    )

    obs_shape = env.observation_shape
    n_actions = env.action_space.n
    log.info(f"Obs shape: {obs_shape}, Actions: {n_actions}")

    agent = RainbowIQNAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        device=device,
        in_channels=obs_shape[0],
        feature_dim=cfg.agent.encoder_features,
        quantile_embedding_dim=cfg.agent.quantile_embedding_dim,
        n_quantiles_train=cfg.agent.n_quantiles_train,
        n_quantiles_eval=cfg.agent.n_quantiles_eval,
        huber_kappa=cfg.agent.huber_kappa,
        encoder_type=cfg.agent.encoder_type,
        learning_rate=cfg.agent.learning_rate,
        adam_eps=cfg.agent.adam_eps,
        grad_clip_norm=cfg.agent.grad_clip_norm,
        gamma=cfg.agent.gamma,
        gamma_int=cfg.agent.gamma_int,
        n_step=cfg.agent.n_step,
        target_update_freq=cfg.agent.target_update_freq,
        buffer_size=cfg.agent.buffer_size,
        batch_size=cfg.agent.batch_size,
        per_alpha=cfg.agent.per_alpha,
        per_beta_start=cfg.agent.per_beta_start,
        per_beta_frames=cfg.agent.per_beta_frames,
        rnd_beta=cfg.agent.rnd_beta,
        rnd_lr=cfg.agent.rnd_lr,
        rnd_output_dim=cfg.agent.rnd_output_dim,
        rnd_warmup_steps=cfg.agent.rnd_warmup_steps,
        noveld_beta=cfg.agent.noveld_beta,
        loss_int_weight=cfg.agent.loss_int_weight,
        munchausen_alpha=cfg.agent.munchausen_alpha,
        munchausen_tau=cfg.agent.munchausen_tau,
        munchausen_clip=cfg.agent.munchausen_clip,
    )

    # --- Resume from checkpoint ---
    start_step = 1
    if cfg.resume_checkpoint is not None:
        ckpt_path = Path(cfg.resume_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
        if ckpt_path.exists():
            agent.load(ckpt_path)
            start_step = _extract_step_from_checkpoint(str(ckpt_path)) + 1
            log.info(f"Resumed from {ckpt_path}, continuing from step {start_step:,}")

            buf_path = _buffer_path_for_checkpoint(ckpt_path)
            if buf_path.exists():
                t0 = time.time()
                agent.load_buffer(buf_path)
                log.info(
                    f"Resumed replay buffer from {buf_path.name} "
                    f"({len(agent.buffer):,} transitions, "
                    f"{time.time() - t0:.1f}s)"
                )
            else:
                log.warning(
                    f"No replay buffer file found alongside checkpoint "
                    f"({buf_path.name}); starting with an empty buffer. "
                    "Expect a transient performance dip until ~10k steps "
                    "have refilled the buffer."
                )
        else:
            log.error(f"Checkpoint not found: {ckpt_path}")
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    compiled_net = torch.compile(agent.online_net, mode="reduce-overhead")
    agent.online_net = compiled_net

    diag = DiagnosticLogger(log_freq=cfg.log_freq)
    tracker = RollingCrafterTracker(window=100)
    latest_metrics: dict = {}

    # --- W&B init ---
    wandb_run = None
    try:
        import wandb

        wandb_run = wandb.init(
            project=cfg.logger.project,
            entity=cfg.logger.entity,
            tags=list(cfg.logger.tags),
            config=dict(cfg),
            mode=cfg.logger.mode,
            resume="allow" if cfg.resume_checkpoint else None,
        )
        log.info(f"W&B run: {wandb_run.url}")
    except Exception as e:
        log.warning(f"W&B init failed, continuing without logging: {e}")

    # --- Training loop ---
    obs = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    start_time = time.time()
    last_buffer_path: Path | None = None
    if cfg.resume_checkpoint is not None:
        # If we resumed from buffer X, treat it as the "previous" file so
        # the next checkpoint cleans it up (we keep only the latest).
        candidate = _buffer_path_for_checkpoint(Path(cfg.resume_checkpoint))
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.exists():
            last_buffer_path = candidate

    for step in range(start_step, cfg.total_steps + 1):
        action = (
            np.random.randint(n_actions) if step < cfg.training_starts else agent.act(obs)
        )

        next_obs, reward, done, info = env.step(action)
        intrinsic_reward = agent.store_transition(obs, action, reward, next_obs, done)

        if intrinsic_reward > 0:
            diag.record_rnd(intrinsic_reward)

        # Pretrain RND predictor during the second half of its warm-up so it
        # is no longer producing pure-noise novelty values when intrinsic
        # rewards start flowing into the replay buffer at end of warm-up.
        # Without this, the first ~thousands of post-warmup transitions land
        # in PER with astronomical TD errors and dominate sampling for tens
        # of thousands of steps after. We sample next_states directly from
        # the buffer's arrays (uniform random over the populated region) to
        # avoid bumping the PER frame_count -- which would shift beta
        # annealing by ~16% before real training even begins.
        if (
            agent.rnd is not None
            and agent.rnd.is_predictor_training_allowed
            and agent.rnd.is_warming_up
            and len(agent.buffer) >= agent.batch_size
            and step % cfg.train_freq == 0
        ):
            idx = np.random.randint(0, len(agent.buffer), size=agent.batch_size)
            agent.rnd.train_predictor(agent.buffer.next_states[idx])

        episode_reward += reward
        episode_length += 1
        obs = next_obs

        if done:
            obs = env.reset()
            episode_count += 1

            achievements = info.get("achievements", {})
            tracker.push(achievements, episode_reward, episode_length)

            if wandb_run and step >= cfg.training_starts:
                ep_metrics = {
                    "episode/reward": episode_reward,
                    "episode/length": episode_length,
                    "episode/count": episode_count,
                }
                for name, count in achievements.items():
                    ep_metrics[f"achievements/{name}"] = count
                ep_metrics.update(tracker.summary_dict())
                wandb_run.log(ep_metrics, step=step)

            if episode_count % 10 == 0:
                elapsed = time.time() - start_time
                steps_done = step - start_step + 1
                sps = steps_done / elapsed

                loss = latest_metrics.get("loss", float("nan"))
                loss_ext = latest_metrics.get("loss_ext", float("nan"))
                loss_int = latest_metrics.get("loss_int", float("nan"))
                q_ext = latest_metrics.get("q_ext_mean", float("nan"))
                q_int = latest_metrics.get("q_int_mean", float("nan"))
                grad = latest_metrics.get("grad_norm", float("nan"))

                rnd_note = ""
                if agent.rnd is not None:
                    if agent.rnd.is_warming_up:
                        rnd_note = " | RND=warmup"
                    else:
                        int_pct = latest_metrics.get(
                            "intrinsic_contribution_pct", float("nan")
                        )
                        rnd_note = (
                            f" | RND_loss={latest_metrics.get('rnd_predictor_loss', float('nan')):.3f}"
                            f" IntPct={int_pct:4.1f}%"
                        )

                log.info(
                    f"Step {step:>7,}/{cfg.total_steps:,} | Ep {episode_count:>4} | "
                    f"CS={tracker.crafter_score():5.2f}% "
                    f"AchU={tracker.unique_unlocked_in_window():>2}/22 "
                    f"AchEver={tracker.total_unique_ever():>2}/22 "
                    f"AchMean={tracker.mean_achievements_per_episode():.2f} | "
                    f"R={tracker.mean_reward():5.2f} L={tracker.mean_length():>4.0f} | "
                    f"Loss={loss:.3f} (ext={loss_ext:.3f} int={loss_int:.3f}) "
                    f"Q=(ext={q_ext:+.2f} int={q_int:+.2f}) grad={grad:.2f}"
                    f"{rnd_note} | SPS={sps:.0f}"
                )

            episode_reward = 0.0
            episode_length = 0

        # --- Learn ---
        if step >= cfg.training_starts and step % cfg.train_freq == 0 and agent.can_learn():
            metrics = agent.learn()
            latest_metrics = metrics

            diag.record_learn_step(
                td_errors=metrics["td_errors"],
                loss=metrics["loss"],
                grad_norms=metrics["grad_norms_detailed"],
            )
            if "rnd_predictor_loss" in metrics:
                diag.record_rnd(0.0, predictor_loss=metrics["rnd_predictor_loss"])

            if wandb_run and step % cfg.log_freq == 0:
                log_metrics = diag.flush(step)
                log_metrics["train/loss"] = metrics["loss"]
                log_metrics["train/loss_ext"] = metrics["loss_ext"]
                log_metrics["train/loss_int"] = metrics["loss_int"]
                log_metrics["train/q_ext_mean"] = metrics["q_ext_mean"]
                log_metrics["train/q_int_mean"] = metrics["q_int_mean"]
                log_metrics["train/grad_norm"] = metrics["grad_norm"]
                wandb_run.log(log_metrics, step=step)

        # --- Checkpoint ---
        if step % cfg.checkpoint_freq == 0:
            save_path = ckpt_dir / f"agent_step_{step}.pt"
            agent.save(save_path)
            log.info(f"Checkpoint saved: {save_path}")

            # Persist replay buffer alongside the network checkpoint so a
            # crash never costs more than `checkpoint_freq` transitions.
            # Compressed .npz of a 250k uint8 image buffer is ~10 GB and
            # writing it takes ~1-2 minutes -- we accept that cost rather
            # than risk losing 24h of experience to a WSL reconnect.
            buffer_path = _buffer_path_for_checkpoint(save_path)
            t0 = time.time()
            agent.save_buffer(buffer_path)
            buf_size = buffer_path.stat().st_size / (1024**3)
            log.info(
                f"Buffer saved: {buffer_path.name} "
                f"({len(agent.buffer):,} transitions, "
                f"{buf_size:.1f} GB, {time.time() - t0:.1f}s)"
            )

            # Keep only the latest buffer file -- two 10 GB blobs is wasteful.
            if last_buffer_path is not None and last_buffer_path != buffer_path:
                try:
                    last_buffer_path.unlink()
                    log.info(f"Removed previous buffer: {last_buffer_path.name}")
                except OSError as e:
                    log.warning(f"Could not delete {last_buffer_path}: {e}")
            last_buffer_path = buffer_path

    # Final save
    agent.save(ckpt_dir / "agent_final.pt")
    log.info("Training complete.")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    train()
