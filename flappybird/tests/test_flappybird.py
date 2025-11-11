import numpy as np

from flappybird import compute_returns


def test_compute_returns():
    """Test compute_returns function."""
    rewards = [1, 1, 1]
    gamma = 0.99
    returns = compute_returns(rewards, gamma)

    assert len(returns) == len(rewards)
    assert np.allclose(returns, [1 + gamma * (1 + gamma * 1), 1 + gamma * 1, 1])
