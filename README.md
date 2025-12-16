Reinforcement Learning Experiments
====================================

This repository contains my experiments with reinforcement learning algorithms. The main focus is on training agents to play Flappy Bird using outdated policy gradient algorithms and PPO.


## Algorithms
Implementations are just for practicing PyTorch and learning about basic concepts of RL.

- *REINFORCE*
  - Episode-batched normalized returns (i.e. mean as baseline)
  - Optional entropy loss term to encourage exploration
- *Vanilla Policy Gradient with Value Function Baseline (VPG)*
  - Learned value function as baseline 
- *Proximal Policy Optimization (PPO)*

In all cases, a simple feedforward network with 2 hidden layers of 128 units each is used for 
 - the policy netwok
 - the value network (for VPG and PPO)


## Environment
The environment used for training is [Flappy Bird Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium/tree/main). It's a simple yet effective environment for testing RL algorithms.

I used **12-dimensional environment states** (via `use_lidar=False`) consisting of the following:
 - the last pipe's horizontal position
 - the last top pipe's vertical position
 - the last bottom pipe's vertical position
 - the next pipe's horizontal position
 - the next top pipe's vertical position
 - the next bottom pipe's vertical position
 - the next next pipe's horizontal position
 - the next next top pipe's vertical position
 - the next next bottom pipe's vertical position
 - bird's vertical position
 - bird's vertical velocity
 - bird's rotation

The **action** space is discrete, with 2 actions:
 - `0`: flap
 - `1`: do nothing

The **reward** structure is sparse:
 - `+1.0` for passing a pipe
 - `-1.0` for death
 - `+0.1` for each frame alive (to encourage longer lives)
 - `-0.5` for reaching top of screen (to penalize flying too high)

## Results
REINFORCE is fucking hopeless for sparse rewards. Adding a value function baseline in VPG doesn't help much at all. PPO works with increased exploration via entropy coefficient.

## Possible Improvements
RL is CPU-bound due to the environment interactions. Therefore, the following improvements are possible:
- Use vectorized environments for training
- Use frame-stacking
- Use a more sophisticated reward shaping

## Misc
The RL theory takes some time setting up. [Gonkee's The FASTEST Introduction to Reinforcement Learning](https://youtu.be/VnpRp7ZglfA?si=02KyDUDj2Gi4Qeds) is a great introductory bird's eye view perspective on the whole field, which my [quick notes](RL_NOTES.md) are based on.
Otherwise, [Open AI Spinning Up](https://spinningup.openai.com/en/latest/) is incredibly good intro to RL, that's more formal and detailed.

### Links
 - [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)