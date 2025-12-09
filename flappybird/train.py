"""Training script for FlappyBird REINFORCE agent.

This script handles the training loop, CLI interface, and MLflow logging
for training REINFORCE agents on FlappyBird.
"""

from typing import Optional

import mlflow
import tyro
from rich.pretty import pprint
from tqdm import tqdm

from agents import (
    ReinforceAgent,  # noqa: F401
    VanillaPolicyGradientAgent,
    collect_episode,
    log_results_to_mlflow,
    make_env,
    save_agent_with_mlflow,
)
from configs import TrainConfig
from utils import log_config_to_mlflow, set_seeds

# TODO: make upload to HF as another command

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")  # default mlflow server host:port
mlflow.set_experiment(experiment_name="flappybird_reinforce_hparam_tuning")


def train(
    cfg: TrainConfig,
    run_id: Optional[str] = None,
):
    """Train using REINFORCE algorithm. A basic policy gradient method."""

    batching = cfg.batch_size is not None and cfg.batch_size > 1

    env = make_env(
        cfg.env,
        record_stats=True,
        video_folder="videos/train",
        episode_trigger=lambda e: e % cfg.record_every == 0 if cfg.record_every else None,
        use_lidar=False,
    )

    # Set seeds for reproducibility
    set_seeds(cfg.seed)

    # Set up the agent
    # agent = ReinforceAgent(cfg, run_id)
    agent = VanillaPolicyGradientAgent(cfg, run_id)

    with mlflow.start_run() as run:
        print("\nTraining Flappy with REINFORCE...\n")
        pprint(cfg)
        print(f"MLflow RUN ID: {run.info.run_id}\n")

        log_config_to_mlflow(cfg)

        tqdm.write(
            f"{'Episode':>8s} | {'Reward':>8s} | {'Loss (P)':>8s} | {'Loss (V)':>8s} | "
            f"{'Entropy':>9s} | {'LR':>8s}"
        )
        for i_episode in tqdm(range(cfg.n_episodes), desc="Training", unit="ep"):
            # Collect experience over one episode
            seed = cfg.seed if cfg.seed_fixed else cfg.seed + i_episode
            info = collect_episode(agent, env, seed)

            # Episode batching: average loss over several episodes and update only once
            if not batching or (i_episode + 1) % cfg.batch_size == 0:
                # Agent has collected enough experience to learn, do a policy update
                result = agent.update()

            # Assumes log_every > batch_size: log w/ lower frequency than update => the logged vars are available
            if (i_episode + 1) % cfg.log_every == 0:
                log_results_to_mlflow(result, info, i_episode)

        # Log summary metrics
        return_queue = env.get_wrapper_attr("return_queue")
        mlflow.log_metric(f"max_reward_last_{return_queue.maxlen}_episodes", max(return_queue))
        mlflow.log_param("policy_optimizer", agent.policy_optimizer.__class__.__name__)

        save_agent_with_mlflow(agent, model_name="flappybird_vpg")
        env.close()


if __name__ == "__main__":
    tyro.cli(train)
