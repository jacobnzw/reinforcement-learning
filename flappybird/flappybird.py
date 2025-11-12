import os
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

# from huggingface_hub import login
# from huggingface_sb3 import package_to_hub
from utils import push_to_hub, record_video  # noqa: F401

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
        return action.item(), dist.log_prob(action)


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
    render_mode="rbg_array",
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
            env, video_folder, episode_trigger=episode_trigger
        )
    if stack_size:
        env = gym.wrappers.FrameStack(env, stack_size)
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


def reinforce_episode(policy, env, gamma, entropy_coef: float | None = None):
    if env.spec.max_episode_steps is None:
        raise ValueError(
            "Env must have a finite max episode length. Check if env is wrapped in TimeLimit."
        )

    log_probs = []
    rewards = []
    state, _ = env.reset()

    # Collect trajectory: run the whole episode
    episode_over = False
    while not episode_over:  # expecting gymnasium.wrappers.TimeLimit applied on env
        action, log_prob = policy.act(state)
        state, reward, terminated, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        episode_over = terminated or truncated

    # Calculate the return
    returns = compute_returns(rewards, gamma, normalize=True, device=device)

    # Calculate the policy loss
    # torch.stack and @ preserve gradients (doesn't break computation graph as opposed to torch.tensor and .dot())
    loss = -torch.stack(log_probs).T @ returns

    # Add the entropy term if specified
    entropy_term = (
        entropy_coef
        * sum([Categorical(logits=logits).entropy() for logits in log_probs])
        if entropy_coef
        else 0.0
    )
    loss += entropy_term

    return loss, sum(rewards), entropy_term


def reinforce_train_loop(
    policy,
    env,
    learning_rate,
    n_training_episodes,
    gamma,
    print_every=100,
    entropy_coef: float | None = None,
):
    """REINFORCE algorithm. A basic policy gradient method.

    The discounted returns at each timestep, are calculated as:
        G_t = r_(t+1) + gamma*G_(t+1)
    This follows a dynamic programming approach, with which we memorize solutions in order to avoid computing
    them multiple times.

    We compute this starting from the last timestep to the first, in order to employ the formula presented above
    and avoid redundant computations that would be needed if we were to do it from first to last.

    Args:
        policy (Policy): Policy network
        optimizer (optim.Optimizer): Optimizer
        env (gymmasium.Env): Environment
        n_training_episodes (int): Number of training episodes
        gamma (float): Discount factor
        print_every (int): Print after this many episodes
    """

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    # scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.1)

    # TODO: wrap env in RecordVideo and RecordEpisodeStatistics

    for i_episode in range(1, n_training_episodes + 1):
        loss, summed_reward, entropy_term = reinforce_episode(
            policy, env, gamma, entropy_coef
        )

        # Update the policy parameters w/ optimizer based on gradients computed during backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if i_episode % print_every == 0:
            print(
                f"Episode {i_episode:6d} | Reward Sum: {summed_reward: 2.4f} | "
                f"Loss: {loss.item(): 2.4f} | Entropy: {entropy_term: 2.4f}"
            )


if __name__ == "__main__":
    env_id = "FlappyBird-v0"
    # TODO: FrameStack can still work even for state observations
    env = make_env(
        env_id,
        render_mode="rgb_array",
        max_episode_steps=10_000,
        video_folder="videos/",
        episode_trigger=lambda e: e % 1000 == 0,  # Record every 1000th episode
        use_lidar=False,
    )

    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

    hpars = {
        "env_id": env_id,
        "seed": seed,
        "n_training_episodes": 1000,
        "n_evaluation_episodes": 10,
        "max_episode_steps": 10_000,
        "gamma": 0.99,
        "lr": 1e-4,
        "entropy_coeff": None,
    }

    policy_file = "flappybird_policy.pt"
    exists = os.path.exists(policy_file)
    reload = True
    policy = (
        load_model_with_metadata(policy_file, FlappyBirdStatePolicy, device)[0]
        if exists and reload
        else FlappyBirdStatePolicy().to(device)
    )

    print("Training a bird how to flap with Reinforce...")
    reinforce_train_loop(
        policy,
        env,
        hpars["lr"],
        hpars["n_training_episodes"],
        hpars["gamma"],
        entropy_coef=hpars["entropy_coeff"],
    )
    env.close()
    save_model_with_metadata(policy, "flappybird_policy.pt", **hpars)

    # Hugging Face repo ID
    # model_id = f"Reinforce-{env_id}"
    # repo_id = f"jacobnzw/{model_id}"
    # eval_env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)
    # push_to_hub(repo_id, env_id, policy, hpars, eval_env)
    # TODO: better recording with gymnasium RecordVideo and RecordEpisodeStatistics
    # TODO: better reproducible eval per Grok
    # record_video(eval_env, policy, "flappybird_reinforce.mp4", fps=30)
