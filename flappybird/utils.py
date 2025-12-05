"""Utility functions for FlappyBird RL training and evaluation.

This module contains shared utility functions for configuration loading,
MLflow integration, model saving/loading, mathematical computations, and
Hugging Face Hub integration.
"""

import datetime
import json
import random
import tempfile
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from huggingface_hub import HfApi, login
from huggingface_hub.repocard import metadata_eval_result, metadata_save


@dataclass
class UpdateResult:
    """Result of a policy update."""

    loss: float
    entropy_term: float
    returns_mean: float
    returns_std: float
    grad_norm: float
    last_lr: float


# Hugging Face Hub utilities
def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info, _ = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(env, policy, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    done = False
    state, _ = env.reset()
    img = env.render()
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action, _ = policy.act(state)
        state, reward, done, info, _ = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


def push_to_hub(repo_id, env_id, model, hyperparameters, eval_env, video_fps=30):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the Hub

    :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
    :param model: the pytorch model we want to save
    :param hyperparameters: training hyperparameters
    :param eval_env: evaluation environment
    :param video_fps: how many frame per seconds to record our video replay
    """

    login()

    _, repo_name = repo_id.split("/")
    api = HfApi()

    # Step 1: Create the repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_directory = Path(tmpdirname)

        # Step 2: Save the model
        torch.save(model, local_directory / "model.pt")

        # Step 3: Save the hyperparameters to JSON
        with open(local_directory / "hyperparameters.json", "w") as outfile:
            json.dump(hyperparameters, outfile)

        # Step 4: Evaluate the model and build JSON
        mean_reward, std_reward = evaluate_agent(
            eval_env,
            hyperparameters["max_t"],
            hyperparameters["n_evaluation_episodes"],
            model,
        )
        # Get datetime
        eval_datetime = datetime.datetime.now()
        eval_form_datetime = eval_datetime.isoformat()

        evaluate_data = {
            "env_id": hyperparameters["env_id"],
            "mean_reward": mean_reward,
            "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
            "eval_datetime": eval_form_datetime,
        }

        # Write a JSON file
        with open(local_directory / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)

        # Step 5: Create the model card
        env_name = hyperparameters["env_id"]

        metadata = {}
        metadata["tags"] = [
            env_name,
            "reinforce",
            "reinforcement-learning",
            "custom-implementation",
            "deep-rl-class",
        ]

        # Add metrics
        eval = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_name,
            dataset_id=env_name,
        )

        # Merges both dictionaries
        metadata = {**metadata, **eval}

        model_card = f"""
  # **Reinforce** Agent playing **{env_id}**
  This is a trained model of a **Reinforce** agent playing **{env_id}** .
  To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
  """

        readme_path = local_directory / "README.md"
        readme = ""
        if readme_path.exists():
            with readme_path.open("r", encoding="utf8") as f:
                readme = f.read()
        else:
            readme = model_card

        with readme_path.open("w", encoding="utf-8") as f:
            f.write(readme)

        # Save our metrics to Readme metadata
        metadata_save(readme_path, metadata)

        # Step 6: Record a video
        video_path = local_directory / "replay.mp4"
        record_video(eval_env, model, video_path, video_fps)

        # Step 7. Push everything to the Hub
        api.upload_folder(
            repo_id=repo_id,
            folder_path=local_directory,
            path_in_repo=".",
        )

        print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


# MLflow utilities
def log_config_to_mlflow(config):
    """Log config to MLflow."""

    IGNORE_KEYS = ["env", "seed_fixed", "log_every", "record_every"]

    # Log the full set of hyperparameters
    for field in fields(config):
        if field.name not in IGNORE_KEYS:
            mlflow.log_param(field.name, getattr(config, field.name))
    mlflow.log_param("frame_stack", config.env.frame_stack)

    # Log the rest as descriptive tags
    mlflow.set_tag("seed_fixed", config.seed_fixed)
    mlflow.set_tag("env_id", config.env.env_id)


def save_model_with_mlflow(model, model_name="flappybird_reinforce"):
    """Saves model using MLflow.

    Model move to CPU prior to saving for compatibility.
    """
    try:
        from .agents import FlappyBirdStatePolicy
    except ImportError:
        from agents import FlappyBirdStatePolicy

    # Log the model (Model Artifact)
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model.cpu(),
        input_example=np.random.rand(1, FlappyBirdStatePolicy.state_dim).astype(
            np.float32
        ),  # infers model signature automatically
        name=model_name,
        registered_model_name=model_name,
    )

    print(f"\nModel saved at URI: {model_info.model_uri}")
    print(f"RUN ID: {model_info.run_id}")


def load_model_with_mlflow(run_id, model_name="flappybird_reinforce", device=None):
    """Load model from MLflow."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from MLflow
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)

    print(f"Loaded model from MLflow URI: {model_uri}")
    return model


def log_results_to_mlflow(result: UpdateResult, info: dict, i_episode: int):
    print(
        f"Episode {i_episode + 1:> 6d} | Reward Sum: {info['summed_reward']:> 10.4f} | "
        f"Loss: {result.loss:> 10.4f} | Entropy: {result.entropy_term:> .2e} | "
        f"LR: {result.last_lr:> .4e}"
    )

    metrics = {
        "loss/total": result.loss,
        "loss/entropy_term": result.entropy_term,
        "policy/return_mean": result.returns_mean,
        "policy/return_std": result.returns_std,
        "policy/learning_rate": result.last_lr,
        "policy/gradient_norm": result.grad_norm,
    }

    if "episode" in info:
        metrics.update(
            {
                "episode/reward": info["episode"]["r"],
                "episode/length": info["episode"]["l"],
                "episode/duration": info["episode"]["t"],
            }
        )

    mlflow.log_metrics(metrics, step=i_episode)


# Mathematical utilities
def set_seeds(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def compute_returns(rewards, gamma, normalize=False, device="cuda"):
    """Compute the returns from the rewards.

    The discounted returns at each timestep are calculated as:
        G_t = r_(t+1) + gamma*G_(t+1)

    This follows a dynamic programming approach, computing from the last timestep to the first
    to avoid redundant computations.

    Args:
        rewards (list): List of rewards
        gamma (float): Discount factor
        normalize (bool): Whether to normalize the returns
        device (str): Device to use for computation

    Returns:
        returns (torch.Tensor): Tensor of returns
        mean (float): Mean of the returns (if normalize=True)
        std (float): Standard deviation of the returns (if normalize=True)
    """

    if gamma <= 0.0 or gamma > 1.0:
        raise ValueError(f"Invalid gamma: {gamma}. Should be in range (0, 1].")

    returns = deque(maxlen=len(rewards))
    discounted_return = 0
    for r in reversed(rewards):
        discounted_return = r + gamma * discounted_return
        # Using deque to prepend in O(1) to keep the returns in chronological order.
        returns.appendleft(discounted_return)

    # Standardize the returns to make the training more stable
    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    if normalize:
        eps = np.finfo(np.float32).eps.item()  # Guard against division by zero (std=0)
        mean, std = returns.mean(), returns.std()
        returns = (returns - mean) / (std + eps)
        return returns, mean, std

    return returns


def gradient_norm(model) -> float:
    """Compute the gradient L2 norm of the model."""
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


# Plotting utilities
def boxplot_episode_rewards(episode_rewards):
    """Create a boxplot of episode rewards."""
    fig, ax = plt.subplots(figsize=(4, 6))
    sns.boxplot(y=episode_rewards, ax=ax)
    ax.set_ylabel("Episode Reward")
    ax.set_title("Distribution of Episode Rewards")
    return fig
