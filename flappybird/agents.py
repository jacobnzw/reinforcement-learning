"""Reinforcement Learning Agents for FlappyBird.

This module contains all RL agents and policy networks for training
agents to play FlappyBird.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Callable

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

import wandb
from configs import EnvConfig, EvalConfig, TrainConfig

device = torch.accelerator.current_accelerator()


def make_mlp(input_dim, hidden_dims, output_dim):
    """Stack layers for a feedforward network."""
    stack_list = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
    for h in range(1, len(hidden_dims)):
        stack_list.extend([nn.Linear(hidden_dims[h - 1], hidden_dims[h]), nn.ReLU()])
    stack_list.extend([nn.Linear(hidden_dims[-1], hidden_dims[-1]), nn.ReLU()])
    stack_list.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*stack_list)


@dataclass
class UpdateResult:
    """Result of a policy update."""

    loss: float
    entropy_term: float
    grad_norm: float
    last_lr: float
    returns_mean: float | None = None
    returns_std: float | None = None
    value_loss: float | None = None


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
        hidden_dims = hidden_dim if isinstance(hidden_dim, tuple) else tuple(hidden_dim)
        self.fc_stack = make_mlp(input_dim, hidden_dims, self.action_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dims

    def forward(self, x):
        return self.fc_stack(x)


class FlappyBirdStateValue(nn.Module):
    """Network represeting Flappy's policy $V_{\theta}(s)$ that takes in the state.

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
    value_dim = 1

    def __init__(self, hidden_dim=32, frame_stack=1):
        super(FlappyBirdStateValue, self).__init__()

        input_dim = self.state_dim * frame_stack
        hidden_dims = hidden_dim if isinstance(hidden_dim, tuple) else tuple(hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dims

        self.fc_stack = make_mlp(input_dim, hidden_dims, self.value_dim)

    def forward(self, x):
        return self.fc_stack(x)


class AgentType(StrEnum):
    """Available RL agents for training."""

    VPG = "vpg"
    REINFORCE = "reinforce"

    @property
    def agent_class(self):
        """Get the corresponding agent class."""
        return {
            AgentType.VPG: VanillaPolicyGradientAgent,
            AgentType.REINFORCE: ReinforceAgent,
        }[self]

    @property
    def description(self) -> str:
        """Get agent description."""
        return {
            AgentType.VPG: "Vanilla Policy Gradient with value function baseline",
            AgentType.REINFORCE: "Basic REINFORCE policy gradient",
        }[self]


class ReinforceAgent:
    def __init__(
        self,
        cfg_env: EnvConfig,
        cfg: TrainConfig = None,
        policy_net: nn.Module = None,
        eval_mode=False,
    ):
        self.type = AgentType.REINFORCE
        self.cfg = cfg
        self.cfg_env = cfg_env
        if eval_mode:
            self.policy_net = policy_net
        else:
            self.batching = cfg.batch_size is not None and cfg.batch_size > 1
            self.grad_clipping = cfg.max_grad_norm is not None and cfg.max_grad_norm > 0.0

            self.policy_net = FlappyBirdStatePolicy(hidden_dim=cfg.hidden_dim).to(device)
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
        if self.cfg_env.frame_stack > 1:
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
            entropy_term=self.cfg.entropy_coeff * entropy_term.item()
            if self.cfg.entropy_coeff
            else 0.0,
            returns_mean=returns_mean.item(),
            returns_std=returns_std.item(),
            grad_norm=grad_norm,
            last_lr=self.scheduler.get_last_lr()[0],
        )


class VanillaPolicyGradientAgent:
    """Vanilla Policy Gradient Agent.

    REINFORCE with learned value function baseline.
    """

    def __init__(
        self,
        cfg_env: EnvConfig,
        cfg: TrainConfig = None,
        policy_net: nn.Module = None,
        value_net: nn.Module = None,
        eval_mode=False,
    ):
        self.type = AgentType.VPG
        self.cfg = cfg
        self.cfg_env = cfg_env
        if eval_mode:
            if policy_net is None or value_net is None:
                raise ValueError("Policy and value networks must be specified in eval mode")
            self.policy_net = policy_net
            self.value_net = value_net
        else:
            if cfg is None:
                raise ValueError("TrainConfig must be specified in train mode")
            self.batching = cfg.batch_size is not None and cfg.batch_size > 1
            self.grad_clipping = cfg.max_grad_norm is not None and cfg.max_grad_norm > 0.0

            # Set up policy network
            self.policy_net = FlappyBirdStatePolicy(hidden_dim=cfg.hidden_dim).to(device)
            self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=cfg.learning_rate)
            # Set up LR scheduler to decay from initial to target learning rate by the end of training
            n_scheduler_steps = (
                cfg.n_episodes // cfg.batch_size if self.batching else cfg.n_episodes
            )
            # Ensure we have at least 1 scheduler step to avoid division by zero
            n_scheduler_steps = max(1, n_scheduler_steps)
            gamma = (cfg.target_learning_rate / cfg.learning_rate) ** (1 / n_scheduler_steps)
            self.policy_scheduler = optim.lr_scheduler.ExponentialLR(
                self.policy_optimizer, gamma=gamma
            )

            # Set up value network
            self.value_net = FlappyBirdStateValue(hidden_dim=cfg.vf_hidden_dim).to(device)
            self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=cfg.vf_learning_rate)

            # batch history buffers
            self.batch_log_probs = []
            self.batch_logits = []
            self.batch_returns = []
            self.batch_values = []

            self.summed_reward = None
            self.is_header_printed = False

        # episode history buffers
        self.log_probs = []
        self.logits = []
        self.rewards = []
        self.values = []

    def act(self, state, deterministic=False):
        """Select an action given the state."""

        # when frame stacking, state[0] stores game states as a LazyFrame instance
        if self.cfg_env.frame_stack > 1:
            state = state[0][:].reshape(-1)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits = self.policy_net.forward(state)
        value = self.value_net.forward(state)

        # convert probs to a categorical distribution and sample the action from it
        dist = Categorical(logits=logits)
        action = dist.sample() if not deterministic else dist.mode

        if not deterministic:  # store only during training (=stochastic policy)
            self.log_probs.append(dist.log_prob(action))
            self.logits.append(logits)
            self.values.append(value)

        return action

    def observe(self, reward, episode_end=False):
        self.rewards.append(reward)
        if episode_end:
            # Compute and store the returns
            returns = compute_returns(self.rewards, self.cfg.gamma, normalize=False, device=device)
            self.batch_returns.append(returns)
            self.batch_log_probs.append(torch.cat(self.log_probs))
            self.batch_logits.append(torch.cat(self.logits))
            self.batch_values.append(torch.cat(self.values))

            self.summed_reward = sum(self.rewards)

            # Clear the episode buffers
            self.rewards.clear()
            self.log_probs.clear()
            self.logits.clear()
            self.values.clear()

    def _update_policy_net(self, advantages: torch.Tensor):
        # Calculate the policy loss
        adv_mean, adv_std = advantages.mean(), advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        loss = -torch.cat(self.batch_log_probs) @ advantages

        # Add the entropy term if specified
        if self.cfg.entropy_coeff:
            entropy_term = Categorical(logits=torch.cat(self.batch_logits)).entropy().sum()
            loss -= self.cfg.entropy_coeff * entropy_term

        # Average loss over the episode batch; divide by 1 if not batching
        loss /= len(self.batch_returns)
        self.policy_optimizer.zero_grad()
        loss.backward()

        if self.grad_clipping:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), max_norm=self.cfg.max_grad_norm
            )
        grad_norm = gradient_norm(self.policy_net)

        # Update parameters and clear gradients
        self.policy_optimizer.step()
        self.policy_scheduler.step()

        entropy = self.cfg.entropy_coeff * entropy_term.item() if self.cfg.entropy_coeff else 0.0
        return loss.item(), entropy, grad_norm

    def _update_value_net(self, returns: torch.Tensor):
        # MSE as value loss
        values = torch.cat(self.batch_values).squeeze()
        # Approximate: should really sum episode MSEs and divide by batch size
        advantages = returns - values
        loss = torch.mean(advantages**2) / len(self.batch_values)

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return advantages.detach(), loss.item()

    def print_update_status(self, result: dict, i_episode: int):
        if not self.is_header_printed:
            tqdm.write(
                f"{'Episode':>8s} | {'Samples':>8s} | {'Reward':>8s} | {'Loss (P)':>8s} | {'Loss (V)':>8s} | "
                f"{'Entropy':>9s} | {'LR':>8s}"
            )
            self.is_header_printed = True
        tqdm.write(
            f"{i_episode + 1:> 8d} | {result['count_samples']:> 8d} | {self.summed_reward:> 8.1f} | "
            f"{result['policy/loss']:> 8.2f} | {result['value/loss']:> 8.2f} | "
            f"{result['policy/entropy']:> .2e} | {result['policy/learning_rate']:> .2e}"
        )

    def update(self) -> dict:
        """Update the policy network's weights based on episode history."""

        # Stack the batch returns
        returns = torch.cat(self.batch_returns)

        advantages, value_loss = self._update_value_net(returns)
        loss, entropy, grad_norm = self._update_policy_net(advantages)

        # Clear history buffers
        self.batch_returns.clear()
        self.batch_log_probs.clear()
        self.batch_logits.clear()
        self.batch_values.clear()

        return {
            "value/loss": value_loss,
            "policy/loss": loss,
            "policy/entropy": entropy,
            "policy/learning_rate": self.policy_scheduler.get_last_lr()[0],
            "policy/gradient_norm": grad_norm,
        }


def make_env(
    env_cfg: EnvConfig,
    record_stats=False,
    video_folder: str | None = None,
    episode_trigger: Callable[[int], bool] | None = None,
):
    """Make the environment."""
    env = gym.make(
        env_cfg.id,
        render_mode="rgb_array",
        max_episode_steps=env_cfg.max_episode_steps,
        use_lidar=False,
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
    if env_cfg.norm_reward_gamma:  # apparently crucial for sparse rewards!
        env = gym.wrappers.NormalizeReward(env, gamma=env_cfg.norm_reward_gamma)
    if env_cfg.norm_observations:
        env = gym.wrappers.NormalizeObservation(env)
    if env_cfg.frame_stack > 1:
        env = gym.wrappers.FrameStack(env, num_stack=env_cfg.frame_stack)
    return env


def prepare_policy_model(cfg, train_run=None):
    if train_run:
        return load_model_with_wandb(train_run, "flappybird_reinforce_policy", device)
    else:
        print("No train run provided. Creating new model.")
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

        agent.observe(reward, episode_over)

    return info


class AgentHandler:
    ENTITY = "jacobnzw-n-a"  # W&B entity
    POLICY_NAME = "policy"
    VALUE_NAME = "value"

    def __init__(self, run: wandb.Run, cfg: EvalConfig):
        self.run = run
        self.cfg = cfg
        self.best_mean_reward = -np.inf
        self.best_mean_score = -np.inf
        self.work_dirs = self.create_run_folder_structure(run.id)

    @staticmethod
    def create_run_folder_structure(run_id: str) -> dict[str, str]:
        """Create folder structure for run data.

        Args:
            run: W&B run object

        Returns:
            Dictionary with paths to created folders
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f"run_data/run-{timestamp}-{run_id}")

        folders = {
            "root": run_dir,
            "videos_train": run_dir / "videos" / "train",
            "videos_eval": run_dir / "videos" / "eval",
            "models": run_dir / "models",
        }

        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)

        return {k: str(v) for k, v in folders.items()}

    @staticmethod
    def _array_stats(array: list):
        return {
            "mean": np.mean(array),
            "std": np.std(array),
            "min": np.min(array),
            "max": np.max(array),
        }

    def _log_stats(self, reward_stats, length_stats, score_stats=None):
        def _stat_line(name, stats):
            return (
                f"{name:8s}: {stats['mean']: >6.2f} +/- {stats['std']:6.1e}"
                f"  ({stats['min']:6.2f} <-> {stats['max']:6.2f})"
            )

        tqdm.write(
            f"\nEvaluation: [mean +/- std (min <-> max)]\n"
            f"{_stat_line('Reward', reward_stats)}\n"
            f"{_stat_line('Length', length_stats)}\n"
            f"{_stat_line('Score', score_stats)}\n"
            f"{'Best':8s}: {self.best_mean_reward:6.2f}\n"
        )
        payload = {
            "eval/reward/mean": reward_stats["mean"],
            "eval/reward/std": reward_stats["std"],
            "eval/reward/min": reward_stats["min"],
            "eval/reward/max": reward_stats["max"],
            "eval/length/mean": length_stats["mean"],
            "eval/length/std": length_stats["std"],
            "eval/length/min": length_stats["min"],
            "eval/length/max": length_stats["max"],
        }
        if score_stats:
            payload.update(
                {
                    "eval/score/mean": score_stats["mean"],
                    "eval/score/std": score_stats["std"],
                    "eval/score/min": score_stats["min"],
                    "eval/score/max": score_stats["max"],
                }
            )
        self.run.log(payload)

    def evaluate(self, agent, env: gym.Env, train_episode: int = None):
        """Evaluate the agent.

        Can be used to evaluate agent in "online" setting (train_episode is not None) during training,
        or "offline" setting (train_episode is None) after training.

        Args:
            agent: Agent to evaluate
            env: Environment to evaluate in
            train_episode: Current training episode number. If None, assume "offline" evaluation of trained model.
        """
        # Create a new eval video folder for each evaluation
        eval_video_folder = Path(self.work_dirs["videos_eval"])
        if train_episode is not None:
            eval_video_folder = eval_video_folder / str(train_episode)
        eval_video_folder.mkdir(parents=True, exist_ok=True)
        env.set_wrapper_attr("video_folder", eval_video_folder.as_posix())

        with torch.no_grad():
            episode_rewards = []
            episode_lengths = []
            episode_scores = []
            for episode in range(self.cfg.n_episodes):
                # Each episode has predictable seed for reproducible evaluation
                # making sure policy can cope with env stochasticity
                seed = self.cfg.seed if self.cfg.seed_fixed else self.cfg.seed + episode
                state, _ = env.reset(seed=seed)
                done = False
                while not done:
                    action = agent.act(state, deterministic=not self.cfg.stochastic)
                    state, reward, terminated, truncated, info = env.step(action.item())
                    done = terminated or truncated

                # Extract episode statistics from info (available after episode ends)
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
                    if env.spec.id == "FlappyBird-v0":
                        episode_scores.append(info["score"])

            reward_stats = self._array_stats(episode_rewards)
            length_stats = self._array_stats(episode_lengths)
            score_stats = self._array_stats(episode_scores) if episode_scores else None

            self._log_stats(reward_stats, length_stats, score_stats)

            # Log model if current eval better than the last
            if train_episode is not None and reward_stats["mean"] > self.best_mean_reward:
                print(f"New best mean reward: {reward_stats['mean']:.2f}")
                self.best_mean_reward = reward_stats["mean"]
                # TODO: should we save the env w/ running stats as well?
                self.save_agent(agent)

            # TODO: Offline eval: episode rewards for boxplot?
            return reward_stats["mean"]

    def save_agent(self, agent, aliases: list[str] = None):
        def save_model_to_artifact(artifact: wandb.Artifact, model: nn.Module, net_name: str):
            model_filename = f"{agent.type}_{net_name}.pth"
            model_path = Path(self.work_dirs["models"]) / model_filename
            torch.save(model.state_dict(), model_path)
            artifact.add_file(model_path, name=model_filename)

        # Create model artifact, save policy and value networks locally, add to artifact, and log
        model_artifact = wandb.Artifact(
            f"{agent.type}_agent", type="model", metadata=self.run.config.as_dict()
        )
        save_model_to_artifact(model_artifact, agent.policy_net, self.POLICY_NAME)
        if hasattr(agent, "value_net"):
            save_model_to_artifact(model_artifact, agent.value_net, self.VALUE_NAME)
        art = self.run.log_artifact(model_artifact, aliases=aliases)

        print(
            f"\nAgent saved locally at: {self.work_dirs['models']} "
            f"logged to artifact: {model_artifact.name} ({art.state})"
        )

    def load_agent(self, artifact_name: str, agent_type: AgentType, device=None):
        """Load agent from wandb artifact.

        Loads policy and value nets from W&B artifact and returns an agent instance.

        Args:
            run (wandb.Run or wandb.Api.Run): W&B run object from which to load the model.
            artifact_name (str): Name of the artifact to load (e.g., "vpg_agent:v0")
            agent_type (AgentType): Type of agent to load
            device: PyTorch device to load the model on

        Returns:
            torch.nn.Module: The loaded model with weights
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_artifact = self.run.use_artifact(artifact_name, type="model")
        model_dir = model_artifact.download(root=self.work_dirs["models"])
        print(f"Agent artifact {artifact_name} downloaded to: {model_dir}")

        # Load policy network
        hidden_dim = model_artifact.metadata["train"]["hidden_dim"]
        policy_net = FlappyBirdStatePolicy(hidden_dim=hidden_dim).to(device)
        policy_path = Path(model_dir) / f"{agent_type.value.lower()}_{self.POLICY_NAME}.pth"
        policy_state = torch.load(policy_path, map_location=device)
        policy_net.load_state_dict(policy_state)

        # Instantiate agent
        if agent_type == AgentType.REINFORCE:
            agent = agent_type.agent_class(self.cfg_env, policy_net=policy_net, eval_mode=True)
        else:  # Load value network, if VPG
            vf_hidden_dim = model_artifact.metadata["train"]["vf_hidden_dim"]
            value_net = FlappyBirdStateValue(hidden_dim=vf_hidden_dim).to(device)
            value_path = Path(model_dir) / f"{agent_type.value.lower()}_{self.VALUE_NAME}.pth"
            value_state = torch.load(value_path, map_location=device)
            value_net.load_state_dict(value_state)

            agent = agent_type.agent_class(
                self.cfg_env, policy_net=policy_net, value_net=value_net, eval_mode=True
            )

        return agent
