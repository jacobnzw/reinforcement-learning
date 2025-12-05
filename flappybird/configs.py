"""Configuration dataclasses for FlappyBird training and evaluation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Training configuration for REINFORCE agent."""
    
    # Environment settings
    env_id: str = "FlappyBird-v0"
    max_episode_steps: int = 10_000
    
    # Training settings
    n_episodes: int = 5_000
    hidden_dim: int = 128
    gamma: float = 0.99
    entropy_coeff: float = 0.01
    learning_rate: float = 1e-2  # initial learning rate decays to target_learning_rate
    target_learning_rate: float = 1e-4
    batch_size: Optional[int] = 25  # None or 0 to disable gradient accumulation
    max_grad_norm: Optional[float] = 5.0  # None or 0.0 to disable gradient clipping
    
    # Reproducibility
    seed: int = 42
    seed_fixed: bool = False  # whether to fix the seed for each episode
    
    # Logging and recording
    log_every: int = 250  # log/print after this many episodes
    record_every: Optional[int] = 5000  # record video after this many episodes, None to disable


@dataclass
class EvalConfig:
    """Evaluation configuration for trained agents."""
    
    # Environment settings
    env_id: str = "FlappyBird-v0"
    max_episode_steps: int = 10_000
    
    # Evaluation settings
    n_episodes: int = 20  # eval on this many episodes
    seed: int = 0  # different from training seed
    seed_fixed: bool = False  # whether to fix the seed for each episode
    stochastic: bool = False  # whether to use stochastic policy
