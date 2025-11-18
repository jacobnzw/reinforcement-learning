import os
from collections import deque
from pathlib import Path
from typing import Callable

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# from huggingface_hub import login
# from huggingface_sb3 import package_to_hub
from utils import push_to_hub, record_video  # noqa: F401

device = torch.accelerator.current_accelerator()
app = typer.Typer()

# TODO: add tensorboard logging


class FlappyBirdImagePolicy(nn.Module):
    """Network represeting the agent's policy $\pi_\theta(a | s)$."""

    INPUT_HW = (84, 84)  # (800, 576)

    def __init__(self, frame_stack, action_dim):
        super(FlappyBirdImagePolicy, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3 * frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Figure out the state dimensionality given sample input
        with torch.no_grad():
            state_dim = self.conv_stack(
                torch.zeros(1, 3 * frame_stack, *self.INPUT_HW)
            ).numel()
        hidden_dim = 512

        self.fc_stack = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.fc_stack(self.conv_stack(x))

    def act(self, state):
        """Select an action from the state."""

        print(f"state.ptp(): {state.ptp()}")

        state = state.transpose(0, 3, 1, 2).reshape(-1, state.shape[1], state.shape[2])
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # interpolate expects (B, C, H, W)
        state = F.interpolate(state, size=self.INPUT_HW, mode="nearest")

        probs = self.forward(state).cpu()

        # convert probs to a categorical distribution and sample the action from it
        dist = Categorical(probs)
        action = dist.sample()
        # return the action and its log probability under categorical distribution
        return action.item(), dist.log_prob(action)


class FlappyBirdStatePolicy(nn.Module):
    """Network represeting Flappy's policy $\pi_\theta(a | s)$ that takes in the state.

    The state consists of the following:
      - the last pipe's horizontal position
      - the last top pipe's vertical position
      - the last bottom pipe's vertical position
      - the next pipe's horizontal position
      - the next top pipe's vertical position
      - the next bottom pipe's vertical position
      - the next next pipe's horizontal position
      - the next next top pipe's vertical position
      - the next next bottom pipe's vertical position
      - player's vertical position
      - player's vertical velocity
      - player's rotation
    """

    # TODO: FrameStack could be optional
    def __init__(self, state_dim=12, hidden_dim=64, action_dim=2):
        super(FlappyBirdStatePolicy, self).__init__()

        self.fc_stack = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.fc_stack(x)

    def act(self, state, deterministic=False):
        """Select an action given the state."""

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)

        # convert probs to a categorical distribution and sample the action from it
        dist = Categorical(probs)
        action = dist.sample() if not deterministic else dist.probs.argmax()
        # return the action and its log probability under categorical distribution
        return action, dist.log_prob(action), dist.logits


def load_config(config_path: Path) -> DictConfig:
    """Load Hydra config from file."""
    config_path = Path(config_path).resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return cfg


def save_model_with_metadata(model, filepath, **metadata):
    """Save model with metadata safely."""
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "metadata": metadata,
    }
    torch.save(save_dict, filepath)


def load_model_with_metadata(filepath, model_class, device=None):
    """Load model with metadata safely."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load with weights_only=True for security
    checkpoint = torch.load(filepath, weights_only=True, map_location=device)

    # Create new model instance
    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint["metadata"]


def make_env(
    env_id,
    render_mode="rgb_array",
    record_stats=False,
    max_episode_steps: int | None = 10_000,
    stack_size: int | None = None,
    video_folder: str | None = None,
    episode_trigger: Callable[[int], bool] | None = None,
    **kwargs,
):
    """Make the environment."""
    env = gym.make(
        env_id, render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs
    )
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    if video_folder:
        if episode_trigger is None:
            print("No episode trigger provided.")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=episode_trigger,
            disable_logger=True,
        )
    if stack_size:
        env = gym.wrappers.FrameStack(env, num_stack=stack_size)
    return env


def compute_returns(rewards, gamma, normalize=True, device="cuda"):
    """Compute the returns from the rewards."""
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
        returns = (returns - returns.mean()) / (returns.std() + eps)

    return returns


def reinforce_episode(
    policy,
    env,
    gamma,
    entropy_coef: float | None = None,
    writer: SummaryWriter | None = None,
):
    """Unroll the policy in the environment for one episode and compute the loss.

    The discounted returns at each timestep, are calculated as:

        G_t = r_(t+1) + gamma*G_(t+1)

    This follows a dynamic programming approach, with which we memorize solutions in order to avoid computing
    them multiple times. We compute this starting from the last timestep to the first, in order to employ the formula
    presented above and avoid redundant computations that would be needed if we were to do it from first to last.

    Args:
        policy (FlappyBirdStatePolicy): Policy network
        env (gymnasium.Env): Environment
        gamma (float): Discount factor
        entropy_coef (float | None): Entropy coefficient. If None, no entropy term is added to the loss.
    """
    if env.spec.max_episode_steps is None:
        raise ValueError(
            "Env must have a finite max episode length. Check if env is wrapped in TimeLimit."
        )

    log_probs = []
    logits = []
    rewards = []

    # Collect trajectory: run the whole episode
    state, _ = env.reset()
    episode_over = False
    while not episode_over:  # expecting env.spec.max_episode_steps is not None
        action, log_prob, dist_logit = policy.act(state)
        state, reward, terminated, truncated, info = env.step(action.item())

        log_probs.append(log_prob)
        logits.append(dist_logit)
        rewards.append(reward)

        episode_over = terminated or truncated

    # Calculate the (discounted) returns for each time step
    returns = compute_returns(rewards, gamma, normalize=True, device=device)

    # Calculate the policy loss
    # torch.stack and @ preserve gradients (doesn't break computation graph as opposed to torch.tensor and .dot())
    loss = -torch.stack(log_probs).T @ returns

    # Add the entropy term if specified
    entropy_term = (
        entropy_coef * Categorical(logits=torch.stack(logits)).entropy().sum()
        if entropy_coef
        else 0.0
    )
    loss += entropy_term

    if writer:
        i_episode = (
            env.episode_count
        )  # present if env is wrapped in RecordEpisodeStatistics
        writer.add_scalar("Loss", loss.item(), i_episode)
        if entropy_coef:
            writer.add_scalar("Entropy Term", entropy_term.item(), i_episode)
        # Policy stats
        writer.add_scalar("Policy/Return Mean", returns.mean().item(), i_episode)
        writer.add_scalar("Policy/Return STD", returns.std().item(), i_episode)
        if "episode" in info:
            writer.add_scalar("Episode Reward", info["episode"]["r"], i_episode)
            writer.add_scalar("Episode Length", info["episode"]["l"], i_episode)
            writer.add_scalar("Episode Duration", info["episode"]["t"], i_episode)

    return loss, sum(rewards), entropy_term


@app.command()
def train(config_path: str = "train_config.yaml"):
    """Train using REINFORCE algorithm. A basic policy gradient method."""

    cfg = load_config(config_path)

    env = make_env(
        cfg.env_id,
        render_mode="rgb_array",
        max_episode_steps=cfg.max_episode_steps,
        video_folder="videos/",
        episode_trigger=lambda e: e % 5000 == 0,  # Record every 1000th episode
        use_lidar=False,
    )

    # Set seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env.reset(seed=cfg.seed)

    # Set up policy model
    if cfg.load_model_path and os.path.exists(cfg.load_model_path):
        policy = load_model_with_metadata(
            cfg.load_model_path, FlappyBirdStatePolicy, device
        )[0]
        print(f"Loaded model from {cfg.load_model_path}")
    else:
        policy = FlappyBirdStatePolicy().to(device)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    # scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.1)

    writer = SummaryWriter(f"logs/run_lr{cfg.learning_rate}_ec{cfg.entropy_coeff}")
    print("Training Flappy with REINFORCE...")
    for i_episode in range(cfg.n_episodes):
        loss, summed_reward, entropy_term = reinforce_episode(
            policy, env, cfg.gamma, cfg.entropy_coeff, writer
        )

        # Update the policy parameters w/ optimizer based on gradients computed during backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if i_episode % cfg.print_every == cfg.print_every - 1:
            print(
                f"Episode {i_episode + 1:> 6d} | Reward Sum: {summed_reward:> 10.4f} | "
                f"Loss: {loss.item():> 10.4f} | Entropy: {entropy_term.item():> g}"
            )

    # Log hyperparameters and summary metrics
    buffer_size = env.return_queue.maxlen
    writer.add_hparams(
        hparam_dict=cfg,
        metric_dict={f"max_reward_last_{buffer_size}_episodes": max(env.return_queue)},
    )

    writer.close()
    env.close()

    save_model_with_metadata(policy, cfg.save_model_path, **cfg)


@app.command()
def eval(
    config_path: str = typer.Option(
        "eval_config.yaml", help="Evaluation config file path"
    ),
    stochastic: bool = typer.Option(
        False, "--stochastic", "-s", help="Whether to use stochastic policy"
    ),
):
    """Evaluate the policy."""
    cfg = load_config(config_path)

    policy, meta = load_model_with_metadata(
        cfg.load_model_path, FlappyBirdStatePolicy, device
    )
    print(
        f"Evaluating policy loaded from {cfg.load_model_path} with metadata:\n {meta}"
    )

    env = make_env(
        cfg.env_id,
        record_stats=True,
        max_episode_steps=cfg.max_episode_steps,
        video_folder="eval_videos/",
        # episode_trigger=lambda e: e in (1, 3, 7, 9),
        use_lidar=False,
    )

    episode_rewards = []
    for episode in range(cfg.n_episodes):
        # Each episode has predictable seed for reproducible evaluation
        # making sure policy can cope with env stochasticity
        state, _ = env.reset(seed=cfg.seed + episode)
        done = False
        while not done:
            action, _, _ = policy.act(state, deterministic=not stochastic)
            state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

        # Extract episode statistics from info (available after episode ends)
        if "episode" in info:
            episode_reward = info["episode"]["r"][0]
            episode_length = info["episode"]["l"][0]
            episode_rewards.append(episode_reward)
            print(
                f"Episode {episode + 1} | Reward: {episode_reward:.2f} | Length: {episode_length}"
            )

    if episode_rewards:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(
            f"\nMean reward over {len(episode_rewards)} episodes: {mean_reward:.2f} +/- {std_reward:.2f}"
        )

    env.close()

    # Hugging Face repo ID
    # model_id = f"Reinforce-{env_id}"
    # repo_id = f"jacobnzw/{model_id}"
    # eval_env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)
    # push_to_hub(repo_id, env_id, policy, hpars, eval_env)
    # record_video(eval_env, policy, "flappybird_reinforce.mp4", fps=30)


if __name__ == "__main__":
    app()
