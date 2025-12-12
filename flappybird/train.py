"""Training script for FlappyBird REINFORCE agent.

This script handles the training loop, CLI interface, and MLflow logging
for training REINFORCE agents on FlappyBird.
"""

from typing import Optional

import tyro
import wandb
from rich.pretty import pprint
from tqdm import tqdm

from agents import (
    AgentType,
    collect_episode,
    log_results_to_wandb,
    make_env,
    save_agent_with_wandb,
)
from configs import TrainConfig
from utils import log_config_to_wandb, set_seeds

# TODO: make upload to HF as another command

# Wandb will be initialized in the train function


def train(
    cfg: TrainConfig,
    model: AgentType = AgentType.VPG,
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
    agent = model.agent_class(cfg, run_id)

    # Initialize wandb
    wandb.init(
        project=f"{model.default_model_name}",
        name=f"train_{model.value}",
        config=cfg.__dict__,
    )

    print(f"\nTraining Flappy with {model.value.upper()}...\n")
    pprint(cfg)
    print(f"Wandb RUN ID: {wandb.run.id}\n")

    log_config_to_wandb(cfg)

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
            agent.print_update_status(result, i_episode)
            log_results_to_wandb(result, info, i_episode)

    # Log summary metrics
    return_queue = env.get_wrapper_attr("return_queue")
    wandb.log({f"max_reward_last_{return_queue.maxlen}_episodes": max(return_queue)})
    wandb.config.update({"policy_optimizer": agent.policy_optimizer.__class__.__name__})

    save_agent_with_wandb(agent, model_name=model.default_model_name)
    env.close()
    wandb.finish()


if __name__ == "__main__":
    tyro.cli(train)
