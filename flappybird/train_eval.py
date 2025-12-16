"""Training script for FlappyBird REINFORCE agent.

This script handles the training loop, CLI interface, and MLflow logging
for training REINFORCE agents on FlappyBird.
"""

from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import tyro
from rich.pretty import pprint
from tqdm import tqdm

import wandb
from agents import AgentType, collect_episode, make_env, save_agent_with_wandb
from configs import EvalConfig, TrainEvalConfig
from utils import set_seeds


class AgentEvaluator:
    def __init__(self, cfg: EvalConfig, run: wandb.Run):
        self.cfg = cfg
        self.run = run
        self.best_mean_reward = -np.inf
        self.best_mean_score = -np.inf

    @staticmethod
    def _array_stats(array: list):
        return {
            "mean": np.mean(array),
            "std": np.std(array),
            "min": np.min(array),
            "max": np.max(array),
        }

    def _log_stats(self, reward_stats, length_stats, score_stats=None):
        def _stat_line(name, stats):
            return (
                f"{name:8s}: {stats['mean']: >6.2f} +/- {stats['std']:6.1e}"
                f"  ({stats['min']:6.2f} <-> {stats['max']:6.2f})"
            )

        tqdm.write(
            f"\nEvaluation: [mean +/- std (min <-> max)]\n"
            f"{_stat_line('Reward', reward_stats)}"
            f"{_stat_line('Length', length_stats)}"
            f"{_stat_line('Score', score_stats)}"
            f"{'Best':8s}: {self.best_mean_reward:6.2f}\n"
        )
        payload = {
            "eval/reward/mean": reward_stats["mean"],
            "eval/reward/std": reward_stats["std"],
            "eval/reward/min": reward_stats["min"],
            "eval/reward/max": reward_stats["max"],
            "eval/length/mean": length_stats["mean"],
            "eval/length/std": length_stats["std"],
            "eval/length/min": length_stats["min"],
            "eval/length/max": length_stats["max"],
        }
        if score_stats:
            payload.update(
                {
                    "eval/score/mean": score_stats["mean"],
                    "eval/score/std": score_stats["std"],
                    "eval/score/min": score_stats["min"],
                    "eval/score/max": score_stats["max"],
                }
            )
        self.run.log(payload)

    def __call__(self, agent, env: gym.Env, i_episode: int):
        with torch.no_grad():
            episode_rewards = []
            episode_lengths = []
            episode_scores = []
            for episode in range(self.cfg.n_episodes):
                # Each episode has predictable seed for reproducible evaluation
                # making sure policy can cope with env stochasticity
                seed = self.cfg.seed if self.cfg.seed_fixed else self.cfg.seed + episode
                state, _ = env.reset(seed=seed)
                done = False
                while not done:
                    action = agent.act(state, deterministic=not self.cfg.stochastic)
                    state, reward, terminated, truncated, info = env.step(action.item())
                    done = terminated or truncated

                # Extract episode statistics from info (available after episode ends)
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
                    if env.spec.id == "FlappyBird-v0":
                        episode_scores.append(info["score"])

            reward_stats = self._array_stats(episode_rewards)
            length_stats = self._array_stats(episode_lengths)
            score_stats = self._array_stats(episode_scores) if episode_scores else None

            self._log_stats(reward_stats, length_stats, score_stats)

            # Log model if current eval better than the last
            if reward_stats["mean"] > self.best_mean_reward:
                print(f"New best mean reward: {reward_stats['mean']:.2f}")
                self.best_mean_reward = reward_stats["mean"]
                save_agent_with_wandb(self.run, agent, model_name=f"{agent.type}_{i_episode}_best")

            return reward_stats["mean"]


def train(
    cfg: TrainEvalConfig,
    model: AgentType = AgentType.VPG,
):
    """Train using a basic policy gradient method.

    Args:
        cfg: Training configuration
        model: Type of agent to train ("vpg" or "reinforce")
    """

    batching = cfg.train.batch_size is not None and cfg.train.batch_size > 1

    env = make_env(
        cfg.env,
        record_stats=True,
        video_folder="videos/train",
        episode_trigger=lambda e: e % cfg.train.record_every == 0
        if cfg.train.record_every
        else None,
        use_lidar=False,
    )
    # TODO: need new config structure
    eval_env = make_env(
        cfg.env,
        record_stats=True,
        video_folder="videos/eval",
        use_lidar=False,
    )

    # Set seeds for reproducibility
    set_seeds(cfg.train.seed)

    # Set up the agent
    agent = model.agent_class(cfg.train, cfg.env)
    cfg_dict = asdict(cfg)
    cfg_dict.update({"optimizer": agent.policy_optimizer.__class__.__name__})
    with wandb.init(
        project=f"{model.default_model_name}",
        name=f"train_{model.value}",
        job_type="train_eval",
        config=cfg_dict,
    ) as run:
        print(f"\nTraining Flappy with {model.value.upper()}...\n")
        # TODO: convert dataclass to rich/richer table (dict); more readable in wandb logs
        pprint(cfg_dict)
        print(f"Wandb RUN ID: {run.id}")
        print(f"Full run path for loading: {run.entity}/{run.project}/{run.id}\n")

        evaluator = AgentEvaluator(cfg.eval, run)
        count_samples = 0
        for i_episode in tqdm(range(cfg.train.n_episodes), desc="Training", unit="ep"):
            # Collect experience over one episode
            seed = cfg.train.seed if cfg.train.seed_fixed else cfg.train.seed + i_episode
            info = collect_episode(agent, env, seed)
            count_samples += info["episode"]["l"]
            # Episode batching: average loss over several episodes and update only once
            if not batching or (i_episode + 1) % cfg.train.batch_size == 0:
                # Agent has collected enough experience to learn, do a policy update
                result = agent.update()

            # Assumes log_every > batch_size: log w/ lower frequency than update => the logged vars are available
            if (i_episode + 1) % cfg.train.log_every == 0:
                result.update({"count_samples": count_samples})
                agent.print_update_status(result, i_episode)
                if "episode" in info:
                    result.update(
                        {
                            "episode/reward": info["episode"]["r"],
                            "episode/length": info["episode"]["l"],
                        }
                    )
                run.log(result, step=i_episode, commit=True)

            if (i_episode + 1) % cfg.eval.eval_every == 0:
                # prints eval stats
                evaluator(agent, eval_env, i_episode)

        # Log summary metrics
        return_queue = env.get_wrapper_attr("return_queue")
        run.log({f"max_reward_last_{return_queue.maxlen}_episodes": max(return_queue)})
        save_agent_with_wandb(run, agent, model_name=f"{model.default_model_name}_last")

        env.close()
        eval_env.close()


if __name__ == "__main__":
    tyro.cli(train)
