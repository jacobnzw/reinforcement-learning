from collections import deque

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import FrameStack
from torch.distributions import Categorical

device = torch.accelerator.current_accelerator()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# TODO: class FlappyBirdStatePolicy(nn.Module)


def reinforce(
    policy, optimizer, env, n_training_episodes, max_t, gamma, print_every=100
):
    """REINFORCE algorithm.



    Args:
        policy (Policy): Policy network
        optimizer (optim.Optimizer): Optimizer
        env (gymmasium.Env): Environment
        n_training_episodes (int): Number of training episodes
        max_t (int): Maximum number of timesteps per episode
        gamma (float): Discount factor
        print_every (int): Print after this many episodes
    """

    scores_deque = deque(maxlen=100)
    scores = []

    # TODO: move the loop out, have the func be just one episode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        for t in range(max_t):
            state = np.array(state)
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # G_t = r_(t+1) + gamma*G_(t+1)
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        # We compute this starting from the last timestep to the first, in order
        # to employ the formula presented above and avoid redundant computations that would be needed
        # if we were to do it from first to last.

        # Using deque here to prepend in O(1) to keep the returns in chronological order
        # TODO range(n_steps)[::-1] is equivalent to reversed(range(n_steps))
        for t in reversed(range(n_steps)):
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        # Standardize the returns to make the training more stable
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        # Guard against division by zero (std=0)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Zero out the gradients, perform a backward pass, and update the policy parameters w/ optimizer
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(
                "Episode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )

    return scores


env_id = "FlappyBird-v0"
frame_stack = 4
env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)
env = FrameStack(env, num_stack=frame_stack)
# TODO: GrayscaleObservation
hpars = {
    "n_training_episodes": 10,
    "n_evaluation_episodes": 10,
    "max_t": 10_000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": env_id,
    "frame_stack": frame_stack,
    "action_dim": int(env.action_space.n),
}


# Create policy and place it to the device
# torch.manual_seed(50)
policy = FlappyBirdImagePolicy(
    frame_stack=hpars["frame_stack"],
    action_dim=hpars["action_dim"],
).to(device)
optimizer = optim.Adam(policy.parameters(), lr=hpars["lr"])

print("Training a bird how to flap with Reinforce...")
print(f"{hpars=}")
print(f"{optimizer=}")
scores = reinforce(
    policy,
    optimizer,
    env,
    hpars["n_training_episodes"],
    hpars["max_t"],
    hpars["gamma"],
)

torch.save(policy, "flappybird_policy.pt")


# Hugging Face repo ID
# model_id = f"Reinforce-{env_id}"
# repo_id = f"jacobnzw/{model_id}"
# eval_env = gym.make(env_id)

# User prompt: Login to Hugging Face
# login()
# package_to_hub(
#     model=policy,
#     model_name=model_id,
#     repo_id=repo_id,
#     end_id=env_id,
#     eval_env=eval_env,
#     video_fps=30,
# )
