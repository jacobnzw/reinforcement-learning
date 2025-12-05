"""Training script for FlappyBird REINFORCE agent.

This script handles the training loop, CLI interface, and MLflow logging
for training REINFORCE agents on FlappyBird.
"""

from typing import Optional

import mlflow
import tyro
from rich.pretty import pprint

from agents import ReinforceAgent, collect_episode, make_env
from configs import TrainConfig
from utils import log_config_to_mlflow, save_model_with_mlflow, set_seeds

# TODO: make upload to HF as another command
# TODO: add switch to turn off mlflow logging

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")  # default mlflow server host:port
mlflow.set_experiment(experiment_name="flappybird_reinforce_hparam_tuning")


def train(
    cfg: TrainConfig = TrainConfig(),
    run_id: Optional[str] = None,
):
    """Train using REINFORCE algorithm. A basic policy gradient method."""

    batching = cfg.batch_size is not None and cfg.batch_size > 1
    grad_clipping = cfg.max_grad_norm is not None and cfg.max_grad_norm > 0.0

    env = make_env(
        cfg.env_id,
        render_mode="rgb_array",
        max_episode_steps=cfg.max_episode_steps,
        record_stats=True,
        video_folder="videos/train",
        episode_trigger=lambda e: e % cfg.record_every == 0 if cfg.record_every else None,
        use_lidar=False,
    )

    # Set seeds for reproducibility
    # TODO: training should be repeated over several seeds and average the results
    # This needs however, training one model per seed, averaging the results across models and saving the best one
    # mlflow.start_run(nested=True) for child runs
    set_seeds(cfg.seed)

    # Set up the agent
    agent = ReinforceAgent(cfg, run_id)

    with mlflow.start_run() as run:
        print("\nTraining Flappy with REINFORCE...\n")
        pprint(cfg)
        print(f"MLflow RUN ID: {run.info.run_id}\n")

        log_config_to_mlflow(cfg)

        # TODO: extract into train_loop(policy, env, optimizer, scheduler, cfg)
        # for seed in cfg.seeds:
        #     train_loop(policy, env, optimizer, scheduler, cfg, seed)

        for i_episode in range(cfg.n_episodes):
            # Collect experience over one episode
            seed = cfg.seed if cfg.seed_fixed else cfg.seed + i_episode
            summed_reward, info = collect_episode(agent, env, seed)

            # Episode batching: average loss over several episodes and update only once
            if not batching or (i_episode + 1) % cfg.batch_size == 0:
                # Agent has collected enough experience to learn, do a policy update
                result = agent.update(grad_clipping)

            # Assumes log_every > batch_size: log w/ lower frequency than update => the logged vars are available
            if (i_episode + 1) % cfg.log_every == 0:
                print(
                    f"Episode {i_episode + 1:> 6d} | Reward Sum: {summed_reward:> 10.4f} | "
                    f"Loss: {result.loss:> 10.4f} | Entropy: {result.entropy_term:> .2e} | "
                    f"LR: {result.last_lr:> .4e}"
                )

                mlflow.log_metric("train/loss/total", result.loss, step=i_episode)
                mlflow.log_metric("train/loss/entropy_term", result.entropy_term, step=i_episode)
                # Policy stats
                mlflow.log_metric("train/policy/return_mean", result.returns_mean, step=i_episode)
                mlflow.log_metric("train/policy/return_std", result.returns_std, step=i_episode)
                mlflow.log_metric("train/policy/learning_rate", result.last_lr, step=i_episode)
                mlflow.log_metric("train/policy/gradient_norm", result.grad_norm, step=i_episode)
                if "episode" in info:
                    mlflow.log_metric("train/episode/reward", info["episode"]["r"], step=i_episode)
                    mlflow.log_metric("train/episode/length", info["episode"]["l"], step=i_episode)
                    mlflow.log_metric(
                        "train/episode/duration", info["episode"]["t"], step=i_episode
                    )

        # Log summary metrics
        return_queue = env.get_wrapper_attr("return_queue")
        mlflow.log_metric(f"max_reward_last_{return_queue.maxlen}_episodes", max(return_queue))
        mlflow.log_param("optimizer", agent.optimizer.__class__.__name__)

        save_model_with_mlflow(agent.policy_net)
        env.close()


if __name__ == "__main__":
    tyro.cli(train)
