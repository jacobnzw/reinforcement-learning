"""Training script for FlappyBird PPO agent using Stable Baselines 3.

This script trains a PPO agent on FlappyBird and logs results to MLflow.
"""

from dataclasses import dataclass

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym  # noqa: F401
import tyro
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder

from utils import set_seeds


@dataclass
class PPOConfig:
    """PPO training configuration."""

    # Environment
    env_id: str = "FlappyBird-v0"
    n_envs: int = 8  # Parallel envs
    max_episode_steps: int = 2_000

    # Training
    total_timesteps: int = 500_000  # Start here; scale to 1M+
    learning_rate: float = 1e-4  # Linear decay to 3e-5
    n_steps: int = 128  # Shorter for sparse rewards
    batch_size: int = 32
    n_epochs: int = 8
    gamma: float = 0.95
    gae_lambda: float = 0.98  # Higher for long horizons
    clip_range: float = 0.2
    ent_coef: float = 0.02  # Boost exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture
    net_arch: str = "128,128"  # Wider for states

    # Evaluation
    eval_freq: int = 10_000  # More frequent checks
    n_eval_episodes: int = 20

    # Logging
    checkpoint_freq: int = 50_000
    video_freq: int = 50_000  # Record more often

    # Reproducibility
    seed: int = 42


# Wandb will be initialized in the train_ppo function


def train_ppo(cfg: PPOConfig):
    """Train PPO agent on FlappyBird."""

    set_seeds(cfg.seed)

    # Parse network architecture
    net_arch = [int(x) for x in cfg.net_arch.split(",")]

    # Initialize wandb
    wandb.init(
        project="flappybird_ppo",
        name="ppo_train",
        config={
            "algorithm": "PPO",
            "env_id": cfg.env_id,
            "n_envs": cfg.n_envs,
            "total_timesteps": cfg.total_timesteps,
            "learning_rate": cfg.learning_rate,
            "n_steps": cfg.n_steps,
            "batch_size": cfg.batch_size,
            "n_epochs": cfg.n_epochs,
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "clip_range": cfg.clip_range,
            "ent_coef": cfg.ent_coef,
            "vf_coef": cfg.vf_coef,
            "max_grad_norm": cfg.max_grad_norm,
            "net_arch": str(net_arch),
            "seed": cfg.seed,
        },
    )

    print("\nTraining FlappyBird with PPO...")
    print(f"Wandb RUN ID: {wandb.run.id}\n")

    # Create vectorized training environment
    train_env = make_vec_env(
        cfg.env_id,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        env_kwargs={"max_episode_steps": cfg.max_episode_steps},
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)  # Key for sparse rewards

    # Create evaluation environment
    eval_env = make_vec_env(
        cfg.env_id,
        n_envs=1,
        seed=cfg.seed + 1000,
        env_kwargs={"max_episode_steps": cfg.max_episode_steps},
    )
    eval_env = VecNormalize(
        eval_env, training=False, norm_obs=True, norm_reward=True
    )  # Load train stats if needed

    # Wrap for video recording
    if cfg.video_freq > 0:
        train_env = VecVideoRecorder(
            train_env,
            "videos/ppo_train",
            record_video_trigger=lambda x: x % cfg.video_freq == 0,
            video_length=1000,
        )
        eval_env = VecVideoRecorder(
            eval_env,
            video_folder="videos/ppo_eval",
            record_video_trigger=lambda step: True,  # Record from step 0
            video_length=2000,  # Max steps (covers even long episodes)
        )

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs={"net_arch": net_arch},
        verbose=1,
        seed=cfg.seed,
        tensorboard_log="./tensorboard_logs/",
    )

    # Set up callbacks
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/ppo_flappybird_best/",
        log_path="./logs/",
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path="./models/ppo_flappybird_checkpoints/",
        name_prefix="ppo_flappybird",
    )
    callbacks.append(checkpoint_callback)

    # Train the agent
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    model.save("./models/ppo_flappybird_final")

    # Save model as wandb artifact
    model_artifact = wandb.Artifact("ppo_policy", type="model")
    model_artifact.add_file("./models/ppo_flappybird_final.zip")
    wandb.log_artifact(model_artifact)

    # Final evaluation
    print("\nFinal evaluation...")
    mean_reward, std_reward = sb3_evaluate(model, eval_env, n_eval_episodes=20)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Log final metrics
    wandb.log(
        {
            "final_mean_reward": mean_reward,
            "final_std_reward": std_reward,
        }
    )

    train_env.close()
    eval_env.close()

    print("\nTraining completed! Model saved and logged to wandb.")
    print(f"Run ID: {wandb.run.id}")
    wandb.finish()


if __name__ == "__main__":
    tyro.cli(train_ppo)
