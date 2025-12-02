from collections import deque
from pathlib import Path
from typing import Callable

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typer  # TODO: check out tyro?
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Categorical

# from huggingface_hub import login
# from huggingface_sb3 import package_to_hub
from utils import push_to_hub, record_video  # noqa: F401

# TODO: refactor according to https://gymnasium.farama.org/tutorials/training_agents/mujoco_reinforce/
# TODO: move helper funcs to utils; or delete utils completely
# TODO: make upload to HF as another command

device = torch.accelerator.current_accelerator()
app = typer.Typer()

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")  # default mlflow server host:port
mlflow.set_experiment(experiment_name="flappybird_reinforce_hparam_tuning")


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

    state_dim = 12

    # TODO: FrameStack could be optional
    def __init__(self, hidden_dim=64, action_dim=2):
        super(FlappyBirdStatePolicy, self).__init__()

        self.fc_stack = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.fc_stack(x)

    def act(self, state, deterministic=False):
        """Select an action given the state."""

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits = self.forward(state)

        # convert probs to a categorical distribution and sample the action from it
        dist = Categorical(logits=logits)
        action = dist.sample() if not deterministic else dist.mode
        # return the action and its log probability under categorical distribution
        return action, dist.log_prob(action), logits


def load_config(config_path: Path) -> DictConfig:
    """Load Hydra config from file."""
    # TODO: simplify hydra may be overkill here
    config_path = Path(config_path).resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return cfg


def log_config_to_mlflow(config):
    """Log config to MLflow."""

    HPARAM_KEYS = [
        "gamma",
        "entropy_coeff",
        "learning_rate",
        "target_learning_rate",
        "batch_size",
        "max_grad_norm",
        "hidden_dim",
    ]
    TAG_KEYS = ["env_id", "n_episodes", "seed"]

    # Log the full set of hyperparameters
    for key in HPARAM_KEYS:
        mlflow.log_param(key, config.get(key))

    # Log the rest as descriptive tags (Context)
    for key in TAG_KEYS:
        # Tags must be strings, and are great for searching/filtering
        mlflow.set_tag(key, str(config.get(key)))


def save_model_with_mlflow(model, model_name="flappybird_reinforce"):
    """Saves model using MLflow.

    Model move to CPU prior to saving for compatibility.
    """

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


def make_run_name(cfg):
    """Make a run name from the config."""

    def human_format(num: int) -> str:
        """Convert number to human readable format (e.g., 1.5k, 2.3M)."""
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0

        # Format with appropriate precision
        if magnitude == 0:
            return f"{int(num)}"
        elif num == int(num):
            return f"{int(num)}{['', 'k', 'M', 'B', 'T'][magnitude]}"
        else:
            return f"{num:.1f}{['', 'k', 'M', 'B', 'T'][magnitude]}"

    name = f"e{human_format(cfg.n_episodes)}_g{cfg.gamma:.2f}_llr{cfg.target_learning_rate:.2e}"
    if cfg.batch_size is not None and cfg.batch_size > 1:
        name += f"_b{cfg.batch_size:d}"
    return name


def prepare_policy_model(cfg, run_id=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if run_id:
        return load_model_with_mlflow(run_id, FlappyBirdStatePolicy, device)
    else:
        print("No run ID provided. Creating new model.")
        return FlappyBirdStatePolicy(hidden_dim=cfg.hidden_dim).to(device)


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
        mean, std = returns.mean(), returns.std()
        returns = (returns - mean) / (std + eps)
        return returns, mean, std

    return returns


def gradient_norm(model):
    """Compute the gradient L2 norm of the model."""
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def reinforce_episode(
    policy,
    env,
    gamma,
    entropy_coef: float | None = None,
    log_metrics: bool = True,
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
        log_metrics (bool): Whether to log metrics to MLflow
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
    returns, ret_mean, ret_std = compute_returns(
        rewards, gamma, normalize=True, device=device
    )

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

    if log_metrics:
        # present if env is wrapped in RecordEpisodeStatistics
        i_episode = env.get_wrapper_attr("episode_count")

        mlflow.log_metric("loss/total", loss.item(), step=i_episode)
        if entropy_coef:
            mlflow.log_metric("loss/entropy_term", entropy_term.item(), step=i_episode)
        # Policy stats
        mlflow.log_metric("policy/return_mean", ret_mean.item(), step=i_episode)
        mlflow.log_metric("policy/return_std", ret_std.item(), step=i_episode)
        if "episode" in info:
            mlflow.log_metric("episode/reward", info["episode"]["r"], step=i_episode)
            mlflow.log_metric("episode/length", info["episode"]["l"], step=i_episode)
            mlflow.log_metric("episode/duration", info["episode"]["t"], step=i_episode)

    return loss, sum(rewards), entropy_term


def train_loop(policy, env, optimizer, scheduler, cfg, seed):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env.reset(seed=cfg.seed)

    batching = cfg.batch_size is not None and cfg.batch_size > 1
    grad_clipping = cfg.max_grad_norm is not None and cfg.max_grad_norm > 0.0

    for i_episode in range(cfg.n_episodes):
        log_episode = i_episode % cfg.log_every == cfg.log_every - 1
        loss, summed_reward, entropy_term = reinforce_episode(
            policy, env, cfg.gamma, cfg.entropy_coeff, log_metrics=log_episode
        )

        # Scale loss for gradient accumulation if batching
        if batching:
            loss /= cfg.batch_size
            entropy_term /= cfg.batch_size

        # Compute gradients: accumulates if batching
        loss.backward()

        if log_episode:
            print(
                f"Episode {i_episode + 1:> 6d} | Reward Sum: {summed_reward:> 10.4f} | "
                f"Loss: {loss.item():> 10.4f} | Entropy: {entropy_term.item():> .2e} | "
                f"LR: {scheduler.get_last_lr()[0]:> .4e}"
            )

        # Episode batching: average loss over several episodes and update only once
        if not batching or (i_episode + 1) % cfg.batch_size == 0:
            # Gradient clipping
            if grad_clipping:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=cfg.max_grad_norm
                )

            mlflow.log_metric(
                "policy/gradient_norm", gradient_norm(policy), step=i_episode
            )

            # Update parameters and clear gradients
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


@app.command()
def train(
    config_path: str = typer.Option(
        "train_config.yaml", help="Path to training config file"
    ),
    run_id: str = typer.Option(None, "-r", "--run-id", help="MLflow run ID"),
):
    """Train using REINFORCE algorithm. A basic policy gradient method."""

    cfg = load_config(config_path)
    batching = cfg.batch_size is not None and cfg.batch_size > 1
    grad_clipping = cfg.max_grad_norm is not None and cfg.max_grad_norm > 0.0

    env = make_env(
        cfg.env_id,
        render_mode="rgb_array",
        max_episode_steps=cfg.max_episode_steps,
        record_stats=True,
        video_folder="videos/train",
        episode_trigger=lambda e: e % cfg.record_every == 0,
        use_lidar=False,
    )

    # Set seeds for reproducibility
    # TODO: training should be repeated over several seeds and average the results
    # This needs however, training one model per seed, averaging the results across models and saving the best one
    # mlflow.start_run(nested=True) for child runs
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env.reset(seed=cfg.seed)

    # Set up policy model
    policy = prepare_policy_model(cfg, run_id, device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    # Set up LR scheduler to decay from initial to target learning rate by the end of training
    gamma = (cfg.target_learning_rate / cfg.learning_rate) ** (1 / cfg.n_episodes)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    with mlflow.start_run(run_name=make_run_name(cfg)) as run:
        print("\nTraining Flappy with REINFORCE...\n")
        print(OmegaConf.to_yaml(cfg))
        print(f"MLflow RUN ID: {run.info.run_id}\n")

        log_config_to_mlflow(cfg)

        # TODO: extract into train_loop(policy, env, optimizer, scheduler, cfg)
        # for seed in cfg.seeds:
        #     train_loop(policy, env, optimizer, scheduler, cfg, seed)

        for i_episode in range(cfg.n_episodes):
            log_episode = i_episode % cfg.log_every == cfg.log_every - 1
            loss, summed_reward, entropy_term = reinforce_episode(
                policy, env, cfg.gamma, cfg.entropy_coeff, log_metrics=log_episode
            )

            # Scale loss for gradient accumulation if batching
            if batching:
                loss /= cfg.batch_size
                entropy_term /= cfg.batch_size

            # Compute gradients: accumulates if batching
            loss.backward()

            if log_episode:
                print(
                    f"Episode {i_episode + 1:> 6d} | Reward Sum: {summed_reward:> 10.4f} | "
                    f"Loss: {loss.item():> 10.4f} | Entropy: {entropy_term.item():> .2e} | "
                    f"LR: {scheduler.get_last_lr()[0]:> .4e}"
                )

            # Episode batching: average loss over several episodes and update only once
            if not batching or (i_episode + 1) % cfg.batch_size == 0:
                # Gradient clipping
                if grad_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), max_norm=cfg.max_grad_norm
                    )

                mlflow.log_metric(
                    "policy/gradient_norm", gradient_norm(policy), step=i_episode
                )

                # Update parameters and clear gradients
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Log hyperparameters and summary metrics
        return_queue = env.get_wrapper_attr("return_queue")
        buffer_size = return_queue.maxlen
        mlflow.log_metric(f"max_reward_last_{buffer_size}_episodes", max(return_queue))
        save_model_with_mlflow(policy)

        env.close()


@app.command()
def eval(
    config_path: str = typer.Option(
        "eval_config.yaml", help="Evaluation config file path"
    ),
    run_id: str = typer.Option(..., "-r", "--run-id", help="MLflow run ID"),
    no_record: bool = typer.Option(
        False, "-n", "--no-record", help="Don't record videos"
    ),
):
    """Evaluate the policy."""
    cfg = load_config(config_path)

    policy = load_model_with_mlflow(run_id, device=device)
    print("Evaluating policy...")

    env = make_env(
        cfg.env_id,
        record_stats=True,
        max_episode_steps=cfg.max_episode_steps,
        video_folder="videos/eval" if not no_record else None,
        episode_trigger=lambda e: True,
        use_lidar=False,
    )

    with torch.no_grad():
        episode_rewards = []
        for episode in range(cfg.n_episodes):
            # Each episode has predictable seed for reproducible evaluation
            # making sure policy can cope with env stochasticity
            state, _ = env.reset(seed=cfg.seed + episode)
            done = False
            while not done:
                action, _, _ = policy.act(state, deterministic=not cfg.stochastic)
                state, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

            # Extract episode statistics from info (available after episode ends)
            if "episode" in info:
                episode_reward = info["episode"]["r"][0]
                episode_length = info["episode"]["l"][0]
                episode_rewards.append(episode_reward)
                print(
                    f"Episode {episode:> 6d} | Reward: {episode_reward:> 6.2f} | Length: {episode_length:> 6d}"
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
