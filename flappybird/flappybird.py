import os
from collections import deque

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


def compute_returns(rewards, gamma):
    """Compute the returns from the rewards."""
    returns = deque(maxlen=len(rewards))
    discounted_return = 0
    for r in reversed(rewards):
        discounted_return = r + gamma * discounted_return
        # Using deque to prepend in O(1) to keep the returns in chronological order.
        returns.appendleft(discounted_return)
    return returns


def reinforce_episode(policy, env, max_t, gamma, entropy_coef: float | None = None):
    log_probs = []
    rewards = []
    state, _ = env.reset()  # TODO: seed?

    # Collect trajectory: run the whole episode
    for t in range(max_t):  # could be handled via gymnasium.wrappers.TimeLimit
        action, log_prob = policy.act(state)
        state, reward, terminated, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        if terminated or truncated:
            break

    # Calculate the return
    returns = compute_returns(rewards, gamma)
    # Standardize the returns to make the training more stable
    returns = torch.tensor(returns)
    eps = np.finfo(np.float32).eps.item()  # Guard against division by zero (std=0)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # Calculate the policy loss
    loss = -torch.tensor(log_probs).dot(returns)

    # Add the entropy term if specified
    if entropy_coef:
        loss += entropy_coef * sum(
            [Categorical(logits=logits).entropy() for logits in log_probs]
        )

    return loss, sum(rewards)


def reinforce_train_loop(
    policy,
    env,
    learning_rate,
    n_training_episodes,
    max_t,
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
        max_t (int): Maximum number of timesteps per episode
        gamma (float): Discount factor
        print_every (int): Print after this many episodes
    """

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    # scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.1)

    # TODO: wrap env in RecordVideo and RecordEpisodeStatistics

    for i_episode in range(1, n_training_episodes + 1):
        loss, summed_reward = reinforce_episode(policy, env, max_t, gamma, entropy_coef)

        # Update the policy parameters w/ optimizer based on gradients computed during backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if i_episode % print_every == 0:
            print(
                f"Episode {i_episode:6d} | Loss: {loss.item(): 2.4f} | Summed Reward: {np.mean(summed_reward): 2.4f}"
            )


if __name__ == "__main__":
    env_id = "FlappyBird-v0"
    # TODO: FrameStack can still work even for state observations
    env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)

    hpars = {
        "n_training_episodes": 10_000,
        "n_evaluation_episodes": 10,
        "max_t": 10_000,
        "gamma": 0.99,
        "lr": 1e-4,
        "env_id": env_id,
        "entropy_coeff": None,
        # TODO: Include seeds for reproducibility
    }

    torch.manual_seed(50)
    policy_file = "model_from_hf.pt"
    exists = os.path.exists(policy_file)
    reload = True
    policy = (
        torch.load(policy_file, weights_only=False, map_location=device)
        if exists and reload
        else FlappyBirdStatePolicy().to(device)
    )

    print("Training a bird how to flap with Reinforce...")
    reinforce_train_loop(
        policy,
        env,
        hpars["lr"],
        hpars["n_training_episodes"],
        hpars["max_t"],
        hpars["gamma"],
        entropy_coef=hpars["entropy_coeff"],
    )

    torch.save(policy, "flappybird_policy.pt")

    # Hugging Face repo ID
    model_id = f"Reinforce-{env_id}"
    repo_id = f"jacobnzw/{model_id}"
    eval_env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)
    # push_to_hub(repo_id, env_id, policy, hpars, eval_env)
    # TODO: better recording with gymnasium RecordVideo and RecordEpisodeStatistics
    # TODO: better reproducible eval per Grok
    record_video(eval_env, policy, "flappybird_reinforce.mp4", fps=30)
