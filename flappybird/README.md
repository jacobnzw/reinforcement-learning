# Flappy Bird with REINFORCE

My attempt at training an agent to play Flappy Bird using REINFORCE. 
REINFORCE unsuitable due to sparse rewards, as shown by the flat-supported reward curve.

## Adding Entropy Term to the Loss
Grok: Adding the `entropy_coef` term to the loss function helps exploration. 
  - I'm observing faster improvement in first few hundreds of episodes, but after that the loss gets starts to 
  oscillate. This did not happen when not using the entropy term.
  - Higher initial learning rates give earlier spike in rewards, but don't help the overall flat trend.


[Gymnasium: Recording Agents](https://gymnasium.farama.org/introduction/record_agent/)