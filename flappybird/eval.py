"""Evaluation script for FlappyBird REINFORCE agent.

This script handles evaluation of trained agents, including video recording
and performance metrics logging.
"""

import mlflow
import numpy as np
import torch
import tyro

from agents import AgentType, make_env
from configs import EvalConfig
from utils import boxplot_episode_rewards, log_config_to_mlflow, set_seeds

# TODO: transitioning to wandb probably a good idea
# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")  # default mlflow server host:port


def eval(
    cfg: EvalConfig,
    model: AgentType = AgentType.VPG,
    run_id: str = "",
    no_record: bool = False,
):
    """Evaluate the policy."""
    set_seeds(cfg.seed)
    agent = model.agent_class(cfg, run_id, eval_mode=True)
    print(f"Evaluating {model.value.upper()} policy...")

    env = make_env(
        cfg.env,
        record_stats=True,
        video_folder="videos/eval" if not no_record else None,
        episode_trigger=lambda e: True,
        use_lidar=False,
    )

    mlflow.set_experiment(experiment_name=f"{model.default_model_name}_hparam_tuning")
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(nested=True):
            log_config_to_mlflow(cfg)
            with torch.no_grad():
                episode_rewards = []
                for episode in range(cfg.n_episodes):
                    # Each episode has predictable seed for reproducible evaluation
                    # making sure policy can cope with env stochasticity
                    seed = cfg.seed if cfg.seed_fixed else cfg.seed + episode
                    state, _ = env.reset(seed=seed)
                    done = False
                    while not done:
                        action = agent.act(state, deterministic=not cfg.stochastic)
                        state, reward, terminated, truncated, info = env.step(action.item())
                        done = terminated or truncated

                    # Extract episode statistics from info (available after episode ends)
                    if "episode" in info:
                        episode_reward = info["episode"]["r"]
                        episode_length = info["episode"]["l"]
                        episode_rewards.append(episode_reward)
                        print(
                            f"Episode {episode:> 3d} | Score: {info['score']:> 3d} | "
                            f"Reward: {episode_reward:> 6.2f} | Length: {episode_length:> 4d}"
                        )
                        mlflow.log_metric("episode/reward", episode_reward, step=episode)
                        mlflow.log_metric("episode/length", episode_length, step=episode)
                        # Log game score (# pipes crossed)
                        mlflow.log_metric("episode/score", info["score"], step=episode)

            if episode_rewards:
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                print(
                    f"\nMean reward over {len(episode_rewards)} episodes: {mean_reward:.2f} +/- {std_reward:.2f}"
                )
                mlflow.log_metric("reward/mean", mean_reward)
                mlflow.log_metric("reward/std", std_reward)

                # Copy training parameters from parent run to eval run for easy comparison
                parent_run = mlflow.get_run(run_id)
                parent_params = parent_run.data.params
                for param_name, param_value in parent_params.items():
                    mlflow.log_param(f"train/{param_name}", param_value)

                # Create and log boxplot of episode rewards
                fig = boxplot_episode_rewards(episode_rewards)
                mlflow.log_figure(fig, "episode_rewards_boxplot.png")

    env.close()

    # Hugging Face repo ID
    # model_id = f"Reinforce-{env_id}"
    # repo_id = f"jacobnzw/{model_id}"
    # eval_env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)
    # push_to_hub(repo_id, env_id, policy, hpars, eval_env)
    # record_video(eval_env, policy, "flappybird_reinforce.mp4", fps=30)


if __name__ == "__main__":
    tyro.cli(eval)
