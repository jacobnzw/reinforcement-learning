# Reinforcement Learning Notes

**Policy**
> How should I act in any given state
Could be just a deterministic function mapping states to actions

$$
    \pi(s): S \to A
$$

or stochastic, returning a conditional probability distribution over actions given a state

$$\pi(a | s)$$


**Reward**

$$
r_t = r(s, a, s')
$$

  - **Shaped reward** gives increasing reward in states that are closer to the end goal. They are easier to learn, because they provide positive feedback even when the policy hasn’t figured out a full solution to the problem. 
  - **Sparse reward** gives reward only at the goal state.



**Return**

$$
  G_t = \sum_{k=t+1}^T \gamma^{k-1-t} r_k
$$

where $ \gamma \in [0, 1] $, $T$ is the terminal time step for episodic tasks. For continuing tasks, $T=\infty$, in which case the sum would diverge if $\gamma = 1$, thus we require $\gamma < 1$ to ensure convergence.


**Trajectory**

$$
    \tau = (s_0, a_0, r_0, s_1, a_1, r_1, s_2, \dots)
$$

In state $s_t$ we take action $a_t$, receive reward $r_t$ and end up in state $s_{t+1}$. Sometimes the reward index is aligned with the state index, sometimes it's not, in which case it means that the reward comes after the action and at the time of transition to the next state $s_{t+1}$. So then the trajectory would be $\tau = (\dots, s_t, a_t, r_{t+1}, \dots)$. Both conventions are used.


**World model** 
> I believe (know) that if I'm in state $s$ and I take action $a$, I will end up in state $s'$ and receive reward $r$.

Expressible with the probabilistic transition dynamics

$$
    p(s', r | s, a)
$$

but a world model can also be expressed with a deterministic function.

My claim: RL is doomed without a world model. World model needs to "understand" physicallity in general, which probably implies it shouldn't suffer from fractured entangled representations (FER).


**Model-based** and **model-free** methods differ depending on whether they have access to the world model.


**Goal of RL**
Learn a policy that maximizes the expected return

$$
    \pi^* = \arg\max_\pi \mathbb{E}_{\pi} [G_t]
$$

The expectation is over *trajectories*, which includes uncertainty in the environment dynamics and the policy itself (in case it is stochastic). If the environment and the policy are deterministic, the expectation collapses, because there can ever be only one trajectory.


**State-value function**
> What is the expected return if I'm in state $s$ and I follow policy $\pi$?

$$
  V_\pi(s) = \mathbb{E}_{\pi} [G_t | s_t = s]
$$


**Action-value function**
> What is the expected return if I'm in state $s$, take action $a$ and then follow policy $\pi$?

The expected return if you start in state $s$, take an arbitrary action $a$ (which may not have come from the policy $\pi$), and then forever after act according to policy $\pi$.

$$
    Q_\pi(s, a) = \mathbb{E}_{\pi} [G_{t} | s_t = s, a_t = a]
$$

*Hint: The policy $\pi$ determines the rewards $r_t$, because it determines the actions which determine next states and rewards (cf. trajectory definition), these then determine the expected return $G_t$, per the definition above. In conclusion, $\pi$ determines $G_t$*.




### Learning Value Fuction

These are **model-free methods** as they don't require the world model. They are also **value-based methods** because they depend on action-value function.

The arrow $ \to $ indicates the update direction, "moving towards the target value". The right-hand side is the **target** value. The shorthand arrow notation merely foregrounds the essential nature of the update, and should not be interpreted as implying equality or prescribing how to update $Q(s, a)$.

*Hint: Think of these updates as operating point-wise: the notation $ Q(s, a) $ refers to a single point on the surface of the $Q$ function, rather then a mathematical object of type function.*

$$
\begin{align*}
  & \text{Monte Carlo:}    & Q(s, a) & & \to & & G_t \\
  & \text{SARSA:}          & Q(s, a) & & \to & & r + \gamma Q(s', a') \\
  & \text{Expected SARSA:} & Q(s, a) & & \to & & r + \gamma \sum_{a} \pi(a | s')  Q(s', a) \\
  & \text{Q-Learning:}     & Q(s, a) & & \to & & r + \gamma \max_a Q(s', a)
\end{align*}
$$

Notice here, we're as it were updating the surface of the $Q$ function point by point (s, a) based on other points (s', a') on its surface.


#### Example Value-Function Update
In a typical gradient ascent fashion we can just write for Monte Carlo

$$
  Q(s_t, a_t) = Q(s_t, a_t) + \alpha[G_t - Q(s_t, a_t)]
$$

where $\alpha$ is the learning rate and $G_t$ is the target (cf. target network in Q-learning). So we're just adding a corrective term to the old value $Q(s_t, a_t)$, where the corrective term is just a $\alpha$-scaled departure of the old $Q(s_t, a_t)$ from the target $G_t$.

The updates for the SARSA, expected SARSA and Q-learning above are analogous.


#### Temporal Difference Learning
The problem with Monte Carlo method is that we have to wait until the episode terminates, the trajectory is complete and the sum for $G_t$ can be evaluated. 

Now suppose we approximate the return $G_t \approx r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$, we plug back into the Monte Carlo update and get

$$
  Q(s_t, a_t) = Q(s_t, a_t) + \alpha[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)].
$$

Same iteration, just with a different target $r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) $. It's called *temporal difference* learning, because the temporal difference is $\gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$.

*Hint: Best to imagine this in discrete state and discrete action space, then Q is a 2D array (table). In that case, remember that $ Q(s_{t+1}, a_{t+1}) $ is just an element in a table representing the current estimate of the expected return at coordinates $ (s_{t+1}, a_{t+1}) $. Initially this estimate could be set to zero (cf. notes in [Bellman optimality equation](#bellman-optimality-equation)).*


#### Greedy Policy
Once $Q^*$ is known, the optimal policy is trivially recovered by local argmax

$$
  a^{*} = \pi^{*}(s) = \arg\max_a Q^{*}(s, a)
$$

where $s$ is some *fixed* state before the $\arg\max$ takes place. (Otherwise, we would be dealing with a variational optimization problem, where the result of $ \arg\max_a $ would be a function of $s$.)
The policy algorithm, that is the way of recovering an optimal action $a^*$ given a state $s_0$, can be stated in two steps:
 1. In the two-argument function $Q^*(s, a)$, fix $s$ to some value $s_0$, which results in a one-argument function $Q^*(s_0, a)$.
 2. Take argmax over actions in the one-argument function: $a^* = \arg\max_a Q^*(s_0, a)$

This is one way to convert a value function into a policy.

#### $\epsilon$-Greedy Policy
- with probability $\epsilon$, take a random action
- with probability $1 - \epsilon$, take the action according to policy

The $ \epsilon $ gets tapered off over time, thus ensuring that the agent explores the environment initially, and then exploits the knowledge it has gained. This is a way of balancing exploration and exploitation.

#### On-Policy vs Off-Policy
The equation above demonstrating the temporal difference property is a SARSA update.

$$
  Q(s_t, a_t) = Q(s_t, a_t) + \alpha[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)],
$$

where $a_{t+1} \sim \pi(a | s_{t+1})$, that is the next action is sampled *from the policy $\pi$* (derived from $Q$ by $\epsilon$-greedy policy rule) - the same policy that chose the action $a_t$. 
In other words, the **behavior policy** that chose the action $ a_{t+1} $ during update is the same as the **target policy** that chose action $ a_t $ and that is being improved.
This is why SARSA is *on-policy*.

If we were to replace the term $Q(s_{t+1}, a_{t+1})$ with $\max_a Q(s_{t+1}, a)$, we get the Q-learning update.

$$
  Q(s_t, a_t) = Q(s_t, a_t) + \alpha[r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)],
$$

where the next action is effectively $a_{t+1} = \arg\max Q(s_{t+1}, a)$ which is a greedy policy, which is different from the main target $\epsilon$-greedy policy used to complete the task. Therefore, Q-learning is an *off-policy* method.

SARSA (and its variants) is on-policy, because the behavior policy, the policy used to gather experiences, is the same as the target policy (the one that's being improved/evaluated). Q-Learning (and its variants) is off-policy, because the behavior policy is different from the target policy. The off-policy methods are more sample efficient.

**Sample Efficiency** is number of episodes/time steps that it takes to get good at a task. Less samples $\implies$ higher efficiency. Monte Carlo is least sample efficient. The SARSA, expected SARSA and Q-Learning are increasingly more sample efficient, because they take advantage of temporal differences.

**Generalized Policy Iteration**
Interdependece between policy $\pi$ and action-value function $Q_\pi$. We can see this in greedy policy rule to derive $\pi$ from $ Q $ and in Bellman optimality equation to derive $Q$ from $\pi$.


### Bellman Optimality Equation
for the state-value function

$$
\begin{align*}
  V^{*}(s) 
  &= \mathbb{E}_\pi[r + \gamma V^{*}(s')] \\
\end{align*}
$$

>But what's the point of that if I have to do the same thing for $V(s_{t+1})$, why is this even such a big deal of equation if it just kicks the can down the road (shift evaluation from $V(s_t)$ to $V(s_{t+1})$)?

The key insight: it’s a *recursive consistency condition*. The Bellman equation doesn’t just define $V_\pi$; it gives a local *self-consistency constraint* that $V_\pi$  must satisfy for every state $s$. That means if you can find a function $V$ that satisfies

$$
  V(s) = \mathbb{E_{\pi}}[r + \gamma V(s')]
$$

for all $s$, then $V$ is the true value function of policy $\pi$. So you’ve turned an infinite-horizon expectation (intractable to compute directly) into a system of equations that can, in principle, be solved locally or iteratively (fixed-point iteration).

It transforms prediction into fixed-point computation.
The Bellman operator

$$
  T^\pi V(s) = \mathbb{E_{\pi}}[r + \gamma V(s')]
$$

is a *contraction mapping*. So repeatedly applying it converges to the unique fixed point $ V_\pi $ (in function space).
By the *Banach fixed-point theorem*, any γ-contraction has a unique fixed point, and repeated application converges to it from *any* starting point:

$$
  V_{k+1} = T^\pi V_k  \implies V_{k+1} \to V^* \;\textnormal{ as }\; k \to \infty
$$

So yes — this property guarantees both convergence and uniqueness of $V_\pi$. The choice of the initial estimate $V_0$ thus doesn't matter, but can speed up convergence. Proofs built on this insight must account for the fact that when V is approximated by deep neural networks, the Banach fixed-point theorem may no longer apply. With linear function approximation some results still hold, but with nonlinear function approximators (deep nets) the contraction arguments no longer guarantee stability — divergence can occur (e.g., Baird’s counterexample and other instability issues). That’s why practical deep RL requires tricks (experience replay, target networks, etc.).

The SARSA, expected SARSA and Q-Learning iterations above define a contraction operators on the Q-space.

Without the Bellman equation, you could only learn $V(s)$ by waiting until an episode ends and summing all rewards.
With the Bellman equation, you can learn $V(s)$ incrementally using immediate feedback and your current guess for $V(s')$.
That’s what enables *temporal difference learning*, *Q-learning*, *dynamic programming*, and *actor-critic* methods.
It’s the structural backbone of all value-based RL.

[ChatGPT thread on details of convergence](https://chatgpt.com/share/68db9219-64b4-8012-a905-5f4ecf11c32f)


### Policy Based Methods

We no longer have a clear target as in the case of value-based methods.
Learn to approximate the optimal policy $\pi^*$ directly without having to learn a value function.
Policy can be parameterized by a neural network with parameters $ \theta $, we can thus write for a stochastic policy

$$
\pi_\theta(s) = P(A \mid S = s; \theta)
$$.

The goal is to increase the policy performance $J(\theta)$ by gradient ascent.

Performance objective: maximize expected return

$$
  J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [G_t]
$$

where $\tau$ is a trajectory drawn from the policy $ \pi_\theta $ parameterized by $ \theta $.
If we break this down further, we get

$$
  J(\theta) = \sum_{\tau} p(\tau; \theta) G_t(\tau)
$$

assuming the world model is known we can write

$$
  p(\tau; \theta) = \prod_{t} p(s_{t+1} | s_t, a_t) \pi_\theta(a_t | s_t)
$$

This evaluates the whole policy (function), not just a single state-action pair as the $Q$-value methods do.
Doesn't give us info on how to improve, but if we just maximize its value we will know we have a good policy.

We don't necessarily wanna evaluate it, we just need to define it so we can derive gradients with respect to $ \theta $, and then use gradient ascent to maximize it.


#### Pros and Cons of Policy-based Methods
  - ✅ We can estimate the policy directly without storing additional data (action values)
  - ✅ Can learn stochastic policy
    - ➡️ We don't need to implement exploration/exploitation strategies
    - ➡️ No perceptual aliasing: two states that look the same to the agent might require different actions.
  - ✅ More effective in high-dimensional continuous action spaces
  - ✅ Better convergence properties

  - ❌ Sample inefficient; can take longer to train
  - ❌ Converge to local optima
  - ❌ Policy-gradient can have high variance (actor-critic can addresss this)


#### The difference between policy-based and policy-gradient methods
Policy-gradient methods, what we’re going to study in this unit, is a subclass of policy-based methods. In policy-based methods, the optimization is most of the time on-policy since for each update, we only use data (trajectories) collected by our most recent version of $\pi_\theta$ .

The difference between these two methods lies on how we optimize the parameter $\theta$:
  - In policy-based methods, we search directly for the optimal policy. We can optimize the parameter $\theta$ indirectly by maximizing the local approximation of the objective function with techniques like hill climbing, simulated annealing, or evolution strategies.
  - In policy-gradient methods, because it is a subclass of the policy-based methods, we search directly for the optimal policy. But we optimize the parameter $\theta$ directly by performing the gradient ascent on the performance of the objective function $J(\theta)$.


#### Training Loop
- Collect an episode with policy $\pi_\theta$
- Calculate the objective function $J(\pi_\theta)$ (return = sum of rewards)
- Compute the gradient of the objective function $\nabla_\theta J(\pi_\theta)$
- Update the weights of the policy network $\theta \leftarrow \theta + \alpha \nabla_\theta J(\pi_\theta)$ 
  - (If positive/negative return: increase/decrease probability of each (state, action) pairs in the episode trajectory)


#### Policy Gradient Theorem
Gradient of $J(\theta)$ with respect to $\theta$ is hard to compute because:
  - $J(\theta)$ is an expectation over the trajectory distribution, which is intractable to evaluate
  - We might not have access to the world model, in which case we can only evaluate the expectation empirically (by sampling trajectories)

Policy gradient theorem provides a way to estimate the gradient of $J(\theta)$ using only the data collected by the policy itself. 
For any differentiable policy $\pi_\theta$ and for any policy objective function $J(\pi_\theta)$, the policy gradient theorem states that

$$
  \nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau\sim\pi_\theta} \left[\sum_t \nabla_\theta \log \pi_\theta (a_t | s_t) \Psi_t \right].
$$

The expectation is over all the trajectories (state-action) pairs that the agent might encounter.
The term $\nabla_\theta \log \pi_\theta (a_t | s_t)$ is the **policy gradient**, which is effectively gradient of log-likelihood of taking action $a_t$ in state $s_t$ under policy $\pi_\theta$.

The term $\Psi_t$ is the **advantage function**, which is the difference between the expected return and the value function.
> How good is the action a_t?

There are several choices for the advantage function. If we set $\Psi_t = G_t$ (total return that occured after taking action $a_t$), we get the **Monte Carlo** policy gradient, which has high variance, because it suffers from the problem of different states having higher or lower returns on average. If we set $\Psi_t = G_t - V_\pi(s_t)$, thus using the value function as a baseline to normalize the returns, we get the **REINFORCE** algorithm. 

A problem with computing the return $G_t$, is that we need to wait until the end of the episode, which is inefficient. We can use bootstrapping to estimate the return more efficiently. If we set $\Psi_t = r_t + \gamma V_\pi(s_{t+1}) - V_\pi(s_t)$, we get the **actor-critic** method.




### Actor-Critic methods
- combination of value-based and policy-based methods
- use a separate model to estimate the value function $V_\pi(s)$. This model is called the **critic**, who critiques the actor in order to help it improve. The policy is called the **actor**.
- don't use return value $G_t$
- relies on temporal differences to estimate the advantage function

Generalize advantage estimation (GAE) is a technique for reducing the variance of the advantage estimate. It uses a moving average of the advantage estimates to get a more stable estimate.

Modelling the policy $\pi_\theta(a | s)$:
- **Categorical distribution** for discrete action spaces. This might be implemented as a softmax layer on top of a linear layer, where the softmax is given by

$$
  \pi_\theta(a | s) = \frac{\exp({W a + b})}{\sum_{a'} \exp({W a' + b})}
$$

- **Gaussian distribution** for continuous action spaces, where we tune the mean of the Gaussian distribution. The variance can be fixed or learned.

### Advanced actor-critic RL algorithms 
don't even use the above.

**Trust Region Policy Optimization (TRPO)** uses the KL divergence as a constraint to ensure that the new policy is not too different from the old policy. The surrogate objective function is given by

$$
  \max_{\theta} \mathbb{E}_{\pi_\theta} \left[ \frac{\pi_\theta(a | s)}{\pi_{\theta_{old}}(a | s)} A(s, a) \right] \quad \text{subject to} \quad D_{KL}(\pi_\theta || \pi_{\theta_{old}}) < \epsilon
$$

where $A(s, a)$ is the advantage function.

**Proximal Policy Optimization (PPO)** uses the KL divergence as a penalty term in the objective function. The surrogate objective function is given by

$$
  \max_{\theta} \mathbb{E}_{\pi_\theta} \left[ \min \left( \frac{\pi_\theta(a | s)}{\pi_{\theta_{old}}(a | s)} A(s, a), \text{clip} \left( \frac{\pi_\theta(a | s)}{\pi_{\theta_{old}}(a | s)}, 1 - \epsilon, 1 + \epsilon \right) A(s, a) \right) \right]
$$




### Research areas


#### Model-based RL

World model helps us to see what states are reachable from the current state, and what rewards we can expect to receive.

Two ways to *obtain* the world model
- **Simulator**. We can create a simulator for the environment. This is the most accurate way, but it's not always possible. For example, it's hard to simulate the physics of a real-world robot.
- **Learning from experience**. We can learn the world model from experience. This is the most general way, but it's also the most sample inefficient.

Two ways to *use* the world model
- **Planning**. We can use the world model to plan ahead and decide what actions to take. This is the most sample efficient way, but it's also the most computationally expensive. 
- **Imagination**. We can use the world model to imagine the consequences of our actions before we take them. This is the most computationally efficient way, but it's also the most sample inefficient.

**Monte Carlo tree search (MCTS)** is a popular algorithm for planning used in AlphaGo from DeepMind.

**Dreamer model series** learns a world model that's used only for training with imagined experience/data.



#### Imitation learning / Inverse RL

**Behavior cloning:** Learn from expert trajectories (demonstrations) without a reward signal. Works well if the agent sticks to the given trajectories, but if it veers off, it has no idea what to do. To solve that there is an improvement on behavior cloning, still within the imitation learning family, called **Dataset aggregation (DAgger)**, which generates mistakes on purpose and humans corrects them. It adds noise to trajectories (eg. when it veered off course) and asks human to correct them. 

Used in Kober, Peters, Learning Motor Primitives for Robotics, where robot learns to swing a ball on a string to a cup attached to it. Robot dogs learning to walk from real dogs.

**Inverse RL** is trying to learn the reward function from expert trajectories of state-action pairs and then learn policy that optimizes that reward function. In other words, it's trying to answer the question: "What is the reward function that the expert is optimizing?". Examples include: inverse Q-learning, apprenticeship learning, maximum entropy inverse reinforcement learning (MaxEnt IRL).



### Practicalities & Implementation Techniques

[How to do good experiments in RL? [Paper] ](https://arxiv.org/pdf/2304.01315)

[Gymnasium](https://gymnasium.farama.org/) library with evironments for the agents and [StableBaselines](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) with SOTA learning algorithms are the shit these days.

#### Vectorized Environments


#### Frame Stacking
#### Frame Skipping



#### Evaluation 
Because most algorithms use exploration noise during training, you need a separate test environment to evaluate the performance of your agent at a given time. It is recommended to periodically evaluate your agent for n test episodes (n is usually between 5 and 20) and average the reward per episode to have a good estimate.


# References

[Open AI Spining Up](https://spinningup.openai.com/en/latest/) Best for grasping the terminology, basic concepts and intuitions. For example, has best explained definition of the quality function Q and what it means to "... thereafter follow policy $\pi$", which I had trouble with initially.

[Yandex Practical RL](https://github.com/yandexdataschool/Practical_RL/tree/master) Didn't do it, but looks like it covers more advanced and cutting-endge topics. Is more self-led, requires more initiative and dedication (eg. reading a chapter in Sutton's book).

[HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) Low-maintainance, notebooks outdated with old dependencies, had to continously struggle with finding the right versions of libraries that work together.