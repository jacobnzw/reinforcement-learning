# Unit 4: Policy Gradient Methods with REINFORCE
This unit covers policy gradient methods, specifically the REINFORCE algorithm, for reinforcement learning.


## Overview
Policy gradient methods directly optimize the policy by computing gradients of the expected return with respect to 
policy parameters. Unlike value-based methods (Q-learning, DQN), policy gradients can handle:

- Continuous action spaces
- Stochastic policies
- High-dimensional action spaces

## Contents
- **REINFORCE Algorithm**: Monte Carlo policy gradient implementation
- **PyGame Learning Environment**: Training agents on classic games
- **Video Recording**: Utilities for recording and visualizing agent performance
- **Hugging Face Integration**: Push trained models to the Hugging Face Hub


## Setup

### System Dependencies
```bash
sudo apt install -y xvfb python3-opengl ffmpeg
```

### Python Environment
This project uses Python 3.8 and uv for package management, so by running
```bash
uv run python pixelcopter.py
```
should "just work", dependencies should be installed automatically by `uv` accoring to the `pyproject.toml` file.


## Resources
- [Hugging Face Deep RL Course - Unit 4](https://huggingface.co/deep-rl-course/unit4/introduction)
- [Spinning Up - Policy Gradient Methods](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [REINFORCE Algorithm Paper](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)