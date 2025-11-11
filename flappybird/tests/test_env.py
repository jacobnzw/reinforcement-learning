import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def test_flappybird_env():
    """Test FlappyBird-v0 environment and visualize observations."""

    # Create environment
    env_id = "FlappyBird-v0"
    env = gym.make(env_id, render_mode="rgb_array")
    # frame_stack = 4
    # env = FrameStack(env, num_stack=frame_stack)

    print(f"Environment: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")

    # Take a few actions and collect observations
    observations = []
    actions_taken = []

    for step in range(10):
        # Alternate between actions (0: do nothing, 1: flap)
        action = step % 2
        actions_taken.append(action)

        obs, reward, done, truncated, info = env.step(action)
        observations.append(obs.copy())

        print(
            f"Step {step}: action={action}, reward={reward:.2f}, "
            f"obs_range=[{obs.min():.2f}, {obs.max():.2f}]"
        )

        if done or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()

    # Visualize some observations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("FlappyBird Observations (Frame Stack Visualization)")

    for i, (obs, action) in enumerate(zip(observations[:6], actions_taken[:6])):
        row, col = i // 3, i % 3

        # obs shape is (4, H, W, 3) for frame stack
        # Take the most recent frame (index -1)
        current_frame = obs[-1]  # Shape: (H, W, 3)

        axes[row, col].imshow(current_frame.astype(np.uint8))
        axes[row, col].set_title(f"Step {i}, Action: {action}")
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("flappybird_observations.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Check if observations are meaningful
    print("\n=== Analysis ===")
    print(f"Observations collected: {len(observations)}")
    # print(f"Frame stack size: {frame_stack}")
    print(f"Individual frame shape: {observations[0][-1].shape}")

    # Check if frames are different (indicating movement)
    if len(observations) >= 2:
        diff = np.abs(
            observations[1][-1].astype(float) - observations[0][-1].astype(float)
        )
        print(
            f"Frame difference (step 0 vs 1): mean={diff.mean():.2f}, max={diff.max():.2f}"
        )

        if diff.mean() > 1.0:
            print("✅ Frames are changing - environment is working!")
        else:
            print("⚠️  Frames seem static - check environment setup")

    env.close()


if __name__ == "__main__":
    test_flappybird_env()
