"""Evaluation script for FlappyBird REINFORCE agent.

This script handles evaluation of trained agents, including video recording
and performance metrics logging.
"""

from dataclasses import asdict

import tyro

import wandb
from agents import AgentHandler, AgentType, make_env
from configs import EnvConfig, EvalConfig
from utils import set_seeds


def eval(
    cfg: EvalConfig,
    cfg_env: EnvConfig,
    artifact_name: str,
    model: AgentType = AgentType.VPG,
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

    print(f"Evaluating {model.value.upper()} policy...")

    # Initialize wandb for evaluation
    with wandb.init(
        project=f"{cfg_env.id.lower()}-{model.value}",
        name=f"eval_{model.value}",
        job_type="eval",
        config=asdict(cfg) | asdict(cfg_env),
    ) as run:
        print(f"Evaluation run ID: {run.id}")
        print(f"Full evaluation run path: {run.entity}/{run.project}/{run.id}\n")

        # Load agent from artifact for "offline" evaluation
        handler = AgentHandler(run, cfg)
        agent = handler.load_agent(artifact_name, model, cfg_env)

        env = make_env(
            cfg_env,
            record_stats=True,
            video_folder=handler.work_dirs["videos_eval"] if not no_record else None,
            episode_trigger=lambda e: True,
        )
        env.set_wrapper_attr("update_running_mean", False)

        handler.evaluate(agent, env)

        # with torch.no_grad():
        #     episode_rewards = []
        #     for episode in range(cfg.n_episodes):
        #         # Each episode has predictable seed for reproducible evaluation
        #         # making sure policy can cope with env stochasticity
        #         seed = cfg.seed if cfg.seed_fixed else cfg.seed + episode
        #         state, _ = env.reset(seed=seed)
        #         done = False
        #         while not done:
        #             action = agent.act(state, deterministic=not cfg.stochastic)
        #             state, reward, terminated, truncated, info = env.step(action.item())
        #             done = terminated or truncated

        #         # Extract episode statistics from info (available after episode ends)
        #         if "episode" in info:
        #             episode_reward = info["episode"]["r"]
        #             episode_length = info["episode"]["l"]
        #             episode_rewards.append(episode_reward)
        #             print(
        #                 f"Episode {episode:> 3d} | Score: {info['score']:> 3d} | "
        #                 f"Reward: {episode_reward:> 6.2f} | Length: {episode_length:> 4d}"
        #             )
        #             run.log(
        #                 {
        #                     "episode/reward": episode_reward,
        #                     "episode/length": episode_length,
        #                     "episode/score": info["score"],
        #                 },
        #                 step=episode,
        #                 commit=False,
        #             )

        # if episode_rewards:
        #     mean_reward = np.mean(episode_rewards)
        #     std_reward = np.std(episode_rewards)
        #     print(
        #         f"\nMean reward over {len(episode_rewards)} episodes: {mean_reward:.2f} +/- {std_reward:.2f}"
        #     )
        #     run.log({"reward/mean": mean_reward, "reward/std": std_reward})

        #     # TODO: DECIDE Create and log boxplot of episode rewards
        #     fig = boxplot_episode_rewards(episode_rewards)
        #     run.log({"episode_rewards_boxplot": wandb.Image(fig)}, commit=True)

        env.close()


if __name__ == "__main__":
    tyro.cli(eval)
