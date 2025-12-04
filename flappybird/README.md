# Flappy Bird with REINFORCE

My attempt at training an agent to play Flappy Bird using REINFORCE. 
REINFORCE unsuitable due to sparse rewards, as shown by the flat-supported reward curve.

## Dev notes
  - Grok: Adding the `entropy_coef` term to the loss function helps exploration. 
  - I'm observing faster improvement in first few hundreds of episodes, but after that the loss gets starts to 
  oscillate. This did not happen when not using the entropy term.
  - Higher initial learning rates give earlier spike in rewards, but don't help the overall flat trend.
  - Fixed seed for training: rewards much less noisy and plateau early
  - Random seed for training: rewards can spike wildly, more noisy and higher eval rewards
  - Variable seeds for training: crucial for training and reaching higher rewards
  - The better baseline makes a huge difference, training shows longer periods of higher reward, what were previously 
  just reward spikes are now sustained rewards. Drops still occur though.


[Gymnasium: Recording Agents](https://gymnasium.farama.org/introduction/record_agent/)