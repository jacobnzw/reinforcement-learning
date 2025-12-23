"""Evaluation script for FlappyBird REINFORCE agent.

This script handles evaluation of trained agents, including video recording
and performance metrics logging.
"""

from dataclasses import asdict

import numpy as np
import torch
import tyro

import wandb
from agents import AgentType, make_env, ENTITY
from configs import EnvConfig, EvalConfig
from utils import boxplot_episode_rewards, set_seeds


def eval(
    cfg: EvalConfig,
    cfg_env: EnvConfig,
    run_id: str,
    model: AgentType = AgentType.VPG,
    project: str = "flappybird-v0-vpg",
    model_basename: str = "vpg_best",
    no_record: bool = False,
):
    """Evaluate the policy.

    Args:
        cfg: Evaluation configuration
        run_id: W&B run ID
        model: Agent type to evaluate
        no_record: If True, don't record videos
    """
    set_seeds(cfg.seed)

    # Get the training run using the API
    api = wandb.Api()
    try:
        # run_data / run-20251223_103822-x9ov235o / models
        train_run = api.run(f"{ENTITY}/{project}/{run_id}")
    except Exception as e:
        raise ValueError(
            f"Could not find run '{run_id}'. Make sure it's in format 'entity/project/run_id'. Error: {e}"
        )

    agent = model.agent_class(cfg, cfg_env, train_run, model_basename, eval_mode=True)
    print(f"Evaluating {model.value.upper()} policy...")

    env = make_env(
        cfg_env,
        record_stats=True,
        video_folder="videos/eval" if not no_record else None,
        episode_trigger=lambda e: True,
        use_lidar=False,
    )
    # --run-id 3ykbvncj --model-basename run_data/run-20251222_091906-3ykbvncj/models/vpg_best
    env.set_wrapper_attr("update_running_mean", False)

    # Initialize wandb for evaluation
    with wandb.init(
        project=f"{model.default_model_name}",
        name=f"eval_{model.value}",
        job_type="eval",
        config=asdict(cfg) | asdict(cfg_env),
    ) as run:
        print(f"Evaluating model from run: {run_id}")
        print(f"Evaluation run ID: {run.id}")
        print(f"Full evaluation run path: {run.entity}/{run.project}/{run.id}\n")

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
                    run.log(
                        {
                            "episode/reward": episode_reward,
                            "episode/length": episode_length,
                            "episode/score": info["score"],
                        },
                        step=episode,
                        commit=False,
                    )

            if episode_rewards:
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                print(
                    f"\nMean reward over {len(episode_rewards)} episodes: {mean_reward:.2f} +/- {std_reward:.2f}"
                )
                run.log({"reward/mean": mean_reward, "reward/std": std_reward})

                # Create and log boxplot of episode rewards
                fig = boxplot_episode_rewards(episode_rewards)
                run.log({"episode_rewards_boxplot": wandb.Image(fig)}, commit=True)

        env.close()


if __name__ == "__main__":
    tyro.cli(eval)
