from enum import Enum

import gymnasium as gym
import panda_gym  # noqa: F401, I001
import typer
from huggingface_hub import login
from huggingface_sb3 import package_to_hub
from rich.console import Console
from stable_baselines3 import A2C, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

app = typer.Typer(help="PandaReach-v3 RL Training and Evaluation CLI")
console = Console()

env_id = "PandaReachDense-v3"


class Policy(str, Enum):
    A2C = "A2C"
    SAC = "SAC"
    TD3 = "TD3"


POLICY_MAP = {
    Policy.A2C: A2C,
    Policy.SAC: SAC,
    Policy.TD3: TD3,
}


def make_experiment_id(policy: Policy, env_id: str):
    return f"{policy.lower()}-{env_id}"


@app.command(help="Train the agent using A2C and save it along with the environment.")
def train(
    policy_choice: Policy = typer.Option(
        Policy.A2C,
        "-a",
        "--algorithm",
        help="RL algorithm to use (A2C, SAC, TD3)",
    ),
    total_time_steps: int = typer.Option(
        1_000_000, "-t", "--time-steps", help="Total time steps to train for"
    ),
    n_envs: int = typer.Option(
        4, "-n", "--n-envs", help="Number of parallel environments"
    ),
):
    # Create the env
    env = gym.make(env_id)

    # Get the state space and action space
    s_size = env.observation_space.shape
    a_size = env.action_space

    console.print("_____OBSERVATION SPACE_____ \n")
    console.print("State Space Dims: ", s_size)

    console.print("\n _____ACTION SPACE_____ \n")
    console.print("Action Space Dims: ", a_size)

    env = make_vec_env(env_id, n_envs=n_envs)

    # Adding this wrapper to normalize the observation and the reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # Create the A2C model and try to find the best parameters
    policy_class = POLICY_MAP[policy_choice]
    model = policy_class("MultiInputPolicy", env, device="auto", verbose=1)
    # Train the agent on GPU if available per device="auto"
    model.learn(total_time_steps, progress_bar=True)

    # Save the model and  VecNormalize statistics when saving the agent
    experiment_id = make_experiment_id(policy_choice, env_id)
    model.save(f"{experiment_id}_model")
    env.save(f"{experiment_id}_env.pkl")


@app.command(
    help="Evaluate the agent and optionally upload it to the Hugging Face Hub."
)
def eval(
    policy_choice: Policy = typer.Option(
        Policy.A2C,
        "-a",
        "--algorithm",
        help="RL algorithm to use (A2C, SAC, TD3)",
    ),
    hf_user_id: str = typer.Option(
        None,
        "-p",
        "--package-to-hub",
        help="Package the model to the hub. Provide your Hugging Face username.",
    ),
    commit_message: str = typer.Option(
        "No message.",
        "-m",
        "--commit-message",
        help="Commit message for Hugging Face Hub",
    ),
):
    experiment_id = make_experiment_id(policy_choice, env_id)
    # EVALUATE the agent
    # Load the saved statistics
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id))])
    eval_env = VecNormalize.load(f"{experiment_id}_env.pkl", eval_env)

    # We need to override the render_mode
    eval_env.render_mode = "rgb_array"

    #  do not update them at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False

    # Load the agent
    model = POLICY_MAP[policy_choice].load(f"{experiment_id}_model")

    mean_reward, std_reward = evaluate_policy(model, eval_env)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # PUBLISH TO HF
    if hf_user_id:
        login()

        package_to_hub(
            model=model,
            model_name=f"{experiment_id}",
            model_architecture=f"{policy_choice.upper()}",
            env_id=env_id,
            eval_env=eval_env,
            repo_id=f"{hf_user_id}/{experiment_id}",
            commit_message=commit_message,
        )


if __name__ == "__main__":
    app()
