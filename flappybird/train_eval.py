"""Training script for FlappyBird REINFORCE agent.

This script handles the training loop, CLI interface, and MLflow logging
for training REINFORCE agents on FlappyBird.
"""

from dataclasses import asdict

import tyro
from rich.pretty import pprint
from tqdm import tqdm

import wandb
from agents import AgentHandler, AgentType, collect_episode, make_env
from configs import TrainEvalConfig
from utils import set_seeds


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

    # Set seeds for reproducibility
    set_seeds(cfg.train.seed)

    # Set up the agent
    agent = model.agent_class(cfg.env, cfg.train)
    cfg_dict = asdict(cfg)
    cfg_dict["train"].update({"optimizer": agent.policy_optimizer.__class__.__name__})
    with wandb.init(
        project=f"{cfg.env.id.lower()}-{model.value}",
        name=f"train_{model.value}",
        job_type="train_eval",
        config=cfg_dict,
    ) as run:
        print(f"\nTraining {cfg.env.id} with {model.value.upper()}...\n")
        # TODO: convert dataclass to rich/richer table (dict); more readable in wandb logs
        pprint(cfg_dict)
        print(f"W&B RUN ID: {run.id}")
        print(f"Full W&B run path for loading: {run.entity}/{run.project}/{run.id}\n")

        handler = AgentHandler(run, cfg.eval)

        env = make_env(
            cfg.env,
            record_stats=True,
            video_folder=handler.work_dirs["videos_train"],
            episode_trigger=lambda e: e % cfg.train.record_every == 0
            if cfg.train.record_every
            else None,
        )
        eval_env = make_env(
            cfg.env,
            record_stats=True,
            video_folder=handler.work_dirs["videos_eval"],
            episode_trigger=lambda e: True,
        )

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
                # prints eval stats and logs to wandb
                handler.evaluate(agent, eval_env, i_episode)

        # Log summary metrics
        return_queue = env.get_wrapper_attr("return_queue")
        run.log({f"max_reward_last_{return_queue.maxlen}_episodes": max(return_queue)})
        handler.save_agent(agent)

        env.close()
        eval_env.close()


if __name__ == "__main__":
    tyro.cli(
        train,
        config=(  # wandb sweep friendly cli behavior
            tyro.conf.FlagConversionOff,
            tyro.conf.UsePythonSyntaxForLiteralCollections,
        ),
    )
