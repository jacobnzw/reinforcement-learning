"""Evaluation script for FlappyBird REINFORCE agent.

This script handles evaluation of trained agents, including video recording
and performance metrics logging.
"""

import numpy as np
import torch
import tyro
import wandb

from agents import AgentType, make_env
from configs import EvalConfig
from utils import boxplot_episode_rewards, log_config_to_wandb, set_seeds

# Wandb will be initialized in the eval function


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

    # Initialize wandb for evaluation
    wandb.init(
        project=f"{model.default_model_name}",
        name=f"eval_{model.value}",
        job_type="evaluation",
        config=cfg.__dict__,
    )

    log_config_to_wandb(cfg)

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
                wandb.log(
                    {
                        "episode/reward": episode_reward,
                        "episode/length": episode_length,
                        "episode/score": info["score"],
                    },
                    step=episode,
                )

        if episode_rewards:
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            print(
                f"\nMean reward over {len(episode_rewards)} episodes: {mean_reward:.2f} +/- {std_reward:.2f}"
            )
            wandb.log({"reward/mean": mean_reward, "reward/std": std_reward})

            # Create and log boxplot of episode rewards
            fig = boxplot_episode_rewards(episode_rewards)
            wandb.log({"episode_rewards_boxplot": wandb.Image(fig)})

    env.close()
    wandb.finish()


if __name__ == "__main__":
    tyro.cli(eval)
