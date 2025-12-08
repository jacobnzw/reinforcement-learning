"""Reinforcement Learning Agents for FlappyBird.

This module contains all RL agents and policy networks for training
agents to play FlappyBird.
"""

from collections import deque
from typing import Callable

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from configs import EnvConfig
from utils import UpdateResult, load_model_with_mlflow

device = torch.accelerator.current_accelerator()


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
            state_dim = self.conv_stack(torch.zeros(1, 3 * frame_stack, *self.INPUT_HW)).numel()
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

    state_dim = 12
    action_dim = 2

    def __init__(self, hidden_dim=64, frame_stack=1):
        super(FlappyBirdStatePolicy, self).__init__()

        input_dim = self.state_dim * frame_stack
        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.fc_stack(x)


def make_env(
    env_cfg: EnvConfig,
    record_stats=False,
    video_folder: str | None = None,
    episode_trigger: Callable[[int], bool] | None = None,
    **kwargs,
):
    """Make the environment."""
    env = gym.make(
        env_cfg.env_id,
        render_mode="rgb_array",
        max_episode_steps=env_cfg.max_episode_steps,
        **kwargs,
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
    if env_cfg.frame_stack > 1:
        env = gym.wrappers.FrameStack(env, num_stack=env_cfg.frame_stack)
    return env


class ReinforceAgent:
    def __init__(self, cfg, run_id=None, eval_mode=False):
        if eval_mode:
            if run_id is None:
                raise ValueError("Run ID must be specified in eval mode")
            self.cfg = cfg
            self.policy_net = load_model_with_mlflow(run_id, device=device)
        else:
            self.cfg = cfg
            self.batching = cfg.batch_size is not None and cfg.batch_size > 1
            self.grad_clipping = cfg.max_grad_norm is not None and cfg.max_grad_norm > 0.0

            self.policy_net = prepare_policy_model(cfg, run_id)
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=cfg.learning_rate)
            # Set up LR scheduler to decay from initial to target learning rate by the end of training
            n_scheduler_steps = (
                cfg.n_episodes // cfg.batch_size if self.batching else cfg.n_episodes
            )
            # Ensure we have at least 1 scheduler step to avoid division by zero
            n_scheduler_steps = max(1, n_scheduler_steps)
            gamma = (cfg.target_learning_rate / cfg.learning_rate) ** (1 / n_scheduler_steps)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

            # batch history buffers
            self.batch_log_probs = []
            self.batch_logits = []
            self.batch_returns = []

        # episode history buffers
        self.log_probs = []
        self.logits = []
        self.rewards = []

    def act(self, state, deterministic=False):
        """Select an action given the state."""

        # when frame stacking, state[0] stores game states as a LazyFrame instance
        if self.cfg.env.frame_stack > 1:
            state = state[0][:].reshape(-1)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits = self.policy_net.forward(state)

        # convert probs to a categorical distribution and sample the action from it
        dist = Categorical(logits=logits)
        action = dist.sample() if not deterministic else dist.mode

        if not deterministic:  # store only during training (=stochastic policy)
            self.log_probs.append(dist.log_prob(action))
            self.logits.append(logits)

        return action

    def observe(self, reward, episode_end=False):
        self.rewards.append(reward)
        summed_reward = None
        if episode_end:
            # Compute and store the returns
            returns = compute_returns(self.rewards, self.cfg.gamma, normalize=False, device=device)
            self.batch_returns.append(returns)
            self.batch_log_probs.append(torch.cat(self.log_probs))
            self.batch_logits.append(torch.cat(self.logits))

            summed_reward = sum(self.rewards)

            # Clear the episode buffers
            self.rewards.clear()
            self.log_probs.clear()
            self.logits.clear()

        return summed_reward

    def update(self, grad_clipping=False) -> UpdateResult:
        """Update the policy network's weights based on episode history."""

        # Clear gradients
        self.optimizer.zero_grad()

        # Stack the batch returns and normalize them
        returns = torch.cat(self.batch_returns)
        returns_mean, returns_std = returns.mean(), returns.std()
        returns = (returns - returns_mean) / (returns_std + 1e-8)
        # Calculate the policy loss
        # torch.cat and @ preserve gradients and computation graph unlike torch.tensor and .dot()
        loss = -torch.cat(self.batch_log_probs) @ returns

        # Add the entropy term if specified
        if self.cfg.entropy_coeff:
            entropy_term = Categorical(logits=torch.cat(self.batch_logits)).entropy().sum()
            loss -= self.cfg.entropy_coeff * entropy_term

        # average loss over the episode batch; divide by 1 if not batching
        loss /= len(self.batch_returns)
        loss.backward()

        if grad_clipping:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), max_norm=self.cfg.max_grad_norm
            )
        grad_norm = gradient_norm(self.policy_net)

        # Update parameters and clear gradients
        self.optimizer.step()
        self.scheduler.step()

        # Clear history buffers
        self.batch_returns.clear()
        self.batch_log_probs.clear()
        self.batch_logits.clear()

        return UpdateResult(
            loss=loss.item(),
            entropy_term=entropy_term.item()
            if isinstance(entropy_term, torch.Tensor)
            else entropy_term,
            returns_mean=returns_mean.item(),
            returns_std=returns_std.item(),
            grad_norm=grad_norm,
            last_lr=self.scheduler.get_last_lr()[0],
        )


def prepare_policy_model(cfg, run_id=None):
    if run_id:
        return load_model_with_mlflow(run_id, FlappyBirdStatePolicy, device)
    else:
        print("No run ID provided. Creating new model.")
        return FlappyBirdStatePolicy(hidden_dim=cfg.hidden_dim).to(device)


def compute_returns(rewards, gamma, normalize=False, device="cuda") -> torch.Tensor:
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


def collect_episode(agent, env, seed):
    """Collect experience from one episode by running the agent in the environment.

    The agent selects actions using its policy and observes rewards. At the end of the episode,
    the discounted returns are computed and stored for later policy updates.

    Args:
        agent (ReinforceAgent): The REINFORCE agent
        env (gymnasium.Env): The environment
        seed (int): Random seed for episode initialization

    Returns:
        tuple: (summed_reward, info) where summed_reward is the total episode reward
               and info contains episode statistics
    """
    if env.spec.max_episode_steps is None:
        raise ValueError(
            "Env must have a finite max episode length. Check if env is wrapped in TimeLimit."
        )

    # Collect trajectory: run the whole episode
    state, _ = env.reset(seed=seed)
    episode_over = False
    while not episode_over:  # expecting env.spec.max_episode_steps is not None
        action = agent.act(state)

        state, reward, terminated, truncated, info = env.step(action.item())
        episode_over = terminated or truncated

        summed_reward = agent.observe(reward, episode_over)

    info["summed_reward"] = summed_reward
    return info
