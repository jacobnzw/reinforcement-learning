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

I used **12-dimensional environment states** (via `use_lidar=False`) consisting of bird's position, velocity, and rotation, as well as the positions of the 3 pipes in front of the bird.

The **action** space is discrete, with 2 actions:
 - `0`: flap
 - `1`: do nothing

The **reward** structure is sparse:
 - `+1.0` for passing a pipe
 - `-1.0` for death
 - `+0.1` for each frame alive (to encourage longer lives)
 - `-0.5` for reaching top of screen (to penalize flying too high)


## Experiments
REINFORCE is fucking hopeless for sparse rewards. Adding a value function baseline in VPG doesn't help much at all. PPO works with increased exploration via entropy coefficient.

### VPG: Sparse vs Dense Rewards
VPG works on InvertedPendulum with dense rewards, but fails on FlappyBird-v0 with sparse rewards.

### VPG: Normalized vs Un-Normalized Sparse Rewards
Compare normalized vs un-normalized rewards on FlappyBird-v0.

### Sample Efficiency: PPO vs VPG
PPO is way more sample efficient than VPG.

| Algo | Reward / 1K Samples | 
|------|---------------------|
| PPO  | Fill                |
| VPG  | Fill                |


## Possible Improvements
RL is CPU-bound due to the environment interactions. Therefore, the following improvements are possible:
- Use vectorized environments for training
- Use [Generalized Advantage Estimation (GAE)](https://shivang-ahd.medium.com/generalized-advantage-estimation-a-deep-dive-into-bias-variance-and-policy-gradients-a5e0b3454dad)

## Misc
The RL theory takes some time setting up. [Gonkee's The FASTEST Introduction to Reinforcement Learning](https://youtu.be/VnpRp7ZglfA?si=02KyDUDj2Gi4Qeds) is a great introductory bird's eye view perspective on the whole field, which my [quick notes](RL_NOTES.md) are based on.
Otherwise, [Open AI Spinning Up](https://spinningup.openai.com/en/latest/) is incredibly good intro to RL, that's more formal and detailed.

### Links
 - [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)