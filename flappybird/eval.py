"""Evaluation script for FlappyBird REINFORCE agent.

This script handles evaluation of trained agents, including video recording
and performance metrics logging.
"""

import mlflow
import numpy as np
import torch
import tyro

from agents import ReinforceAgent, make_env
from configs import EvalConfig

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")  # default mlflow server host:port


def eval(
    cfg: EvalConfig = EvalConfig(),
    run_id: str = "",
    no_record: bool = False,
):
    """Evaluate the policy."""

    agent = ReinforceAgent(cfg, run_id, eval_mode=True)
    print("Evaluating policy...")

    env = make_env(
        cfg.env_id,
        record_stats=True,
        max_episode_steps=cfg.max_episode_steps,
        video_folder="videos/eval" if not no_record else None,
        episode_trigger=lambda e: True,
        use_lidar=False,
    )

    with mlflow.start_run(run_id=run_id):
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
                    episode_reward = info["episode"]["r"][0]
                    episode_length = info["episode"]["l"][0]
                    episode_rewards.append(episode_reward)
                    print(
                        f"Episode {episode:> 6d} | Reward: {episode_reward:> 6.2f} | Length: {episode_length:> 6d}"
                    )
                    mlflow.log_metric("eval/episode_reward", episode_reward, step=episode)
                    mlflow.log_metric("eval/episode_length", episode_length, step=episode)

        if episode_rewards:
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            print(
                f"\nMean reward over {len(episode_rewards)} episodes: {mean_reward:.2f} +/- {std_reward:.2f}"
            )
            # TODO: log a seaborn boxplot of episode rewards
            mlflow.log_metric("eval_mean_reward", mean_reward, step=0)
            mlflow.log_metric("eval_std_reward", std_reward, step=0)

    env.close()

    # Hugging Face repo ID
    # model_id = f"Reinforce-{env_id}"
    # repo_id = f"jacobnzw/{model_id}"
    # eval_env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)
    # push_to_hub(repo_id, env_id, policy, hpars, eval_env)
    # record_video(eval_env, policy, "flappybird_reinforce.mp4", fps=30)


if __name__ == "__main__":
    tyro.cli(eval)
