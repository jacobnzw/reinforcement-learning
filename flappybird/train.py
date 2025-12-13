"""Training script for FlappyBird REINFORCE agent.

This script handles the training loop, CLI interface, and MLflow logging
for training REINFORCE agents on FlappyBird.
"""

from dataclasses import asdict

import tyro
from rich.pretty import pprint
from tqdm import tqdm

import wandb
from agents import AgentType, collect_episode, make_env, save_agent_with_wandb
from configs import TrainConfig
from utils import set_seeds

# TODO: make upload to HF as another command


def train(
    cfg: TrainConfig,
    model: AgentType = AgentType.VPG,
):
    """Train using REINFORCE algorithm. A basic policy gradient method.

    Args:
        cfg: Training configuration
        model: Agent type to train
    """

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
    agent = model.agent_class(cfg)
    cfg_dict = asdict(cfg)
    cfg_dict.update({"optimizer": agent.policy_optimizer.__class__.__name__})
    with wandb.init(
        project=f"{model.default_model_name}",
        name=f"train_{model.value}",
        job_type="train",
        config=cfg_dict,
    ) as run:
        print(f"\nTraining Flappy with {model.value.upper()}...\n")
        # TODO: convert dataclass to rich/richer table (dict); more readable in wandb logs
        pprint(cfg)
        print(f"Wandb RUN ID: {run.id}")
        print(f"Full run path for loading: {run.entity}/{run.project}/{run.id}\n")

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
                if "episode" in info:
                    result.update(
                        {
                            "episode/reward": info["episode"]["r"],
                            "episode/length": info["episode"]["l"],
                            "episode/duration": info["episode"]["t"],
                        }
                    )
                run.log(result, step=i_episode, commit=True)

        # Log summary metrics
        return_queue = env.get_wrapper_attr("return_queue")
        run.log({f"max_reward_last_{return_queue.maxlen}_episodes": max(return_queue)})

        save_agent_with_wandb(run, agent, model_name=model.default_model_name)
        env.close()


if __name__ == "__main__":
    tyro.cli(train)
