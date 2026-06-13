"""Hyperparameter tuning with Optuna TPE sampler.

Short trials (500k steps each) to find good hyperparameters.
Final training uses the full 10M step budget.
"""

import logging

import numpy as np
import optuna
import torch

from src.agents.rainbow_iqn_agent import RainbowIQNAgent
from src.envs.wrappers import make_crafter_env

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TRIAL_STEPS = 500_000
TRAINING_STARTS = 20_000
TRAIN_FREQ = 4
EVAL_EPISODES = 10


def objective(trial: optuna.Trial) -> float:
    lr_encoder = trial.suggest_float("lr_encoder", 1e-5, 1e-3, log=True)
    trial.suggest_float("lr_iqn_head", 1e-4, 1e-3, log=True)  # logged for analysis
    rnd_beta = trial.suggest_float("rnd_beta", 0.01, 0.5, log=True)
    rnd_lr = trial.suggest_float("rnd_lr", 1e-4, 1e-3, log=True)
    per_alpha = trial.suggest_float("per_alpha", 0.4, 0.8)
    per_beta_start = trial.suggest_float("per_beta_start", 0.3, 0.6)
    n_quantiles = trial.suggest_categorical("n_quantiles", [32, 64])
    n_step = trial.suggest_categorical("n_step", [3, 5])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(trial.number)
    np.random.seed(trial.number)

    env = make_crafter_env(frame_stack=4, action_repeat=2, image_size=64, grayscale=True)
    obs_shape = env.observation_shape
    n_actions = env.action_space.n

    # Use a unified LR as a compromise -- Optuna explores the space
    agent = RainbowIQNAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        device=device,
        in_channels=obs_shape[0],
        learning_rate=lr_encoder,
        n_quantiles_train=n_quantiles,
        n_quantiles_eval=n_quantiles,
        gamma=0.99,
        n_step=n_step,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        rnd_beta=rnd_beta,
        rnd_lr=rnd_lr,
    )

    obs = env.reset()
    episode_rewards = []
    episode_reward = 0.0
    episode_achievements: dict[str, list[int]] = {}

    for step in range(1, TRIAL_STEPS + 1):
        action = (
            np.random.randint(n_actions) if step < TRAINING_STARTS else agent.act(obs)
        )

        next_obs, reward, done, info = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)
        episode_reward += reward
        obs = next_obs

        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0

            if "achievements" in info:
                for name, count in info["achievements"].items():
                    if name not in episode_achievements:
                        episode_achievements[name] = []
                    episode_achievements[name].append(count)

        if step >= TRAINING_STARTS and step % TRAIN_FREQ == 0 and agent.can_learn():
            agent.learn()

        # Pruning: report intermediate value every 100k steps
        if step % 100_000 == 0 and episode_rewards:
            trial.report(float(np.mean(episode_rewards[-50:])), step // 100_000)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Crafter score: geometric mean of achievement success rates
    if episode_achievements:
        rates = []
        for _name, counts in episode_achievements.items():
            rates.append(float(np.mean([c > 0 for c in counts])))
        if rates:
            score = float(np.exp(np.mean(np.log(np.array(rates) + 1e-8))))
            return score

    return float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0


def main() -> None:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name="rainbow_iqn_crafter",
    )

    study.optimize(objective, n_trials=25, show_progress_bar=True)

    log.info("--- Best Trial ---")
    log.info(f"  Value: {study.best_trial.value:.4f}")
    log.info(f"  Params: {study.best_trial.params}")

    # Save results
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)
    log.info("Results saved to optuna_results.csv")


if __name__ == "__main__":
    main()
