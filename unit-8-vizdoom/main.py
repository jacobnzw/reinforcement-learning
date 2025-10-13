import functools
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import (
    add_doom_env_args,
    doom_override_defaults,
)
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec

# Initialize typer app and rich console
app = typer.Typer(help="ViZDoom RL Training and Evaluation CLI")
console = Console()


# Registers all the ViZDoom environments
def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


# Sample Factory allows the registration of a custom Neural Network architecture
# See https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/vizdoom/doom/doom_model.py for more details
def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()


# parse the command line args and create a config
def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


@app.command()
def train(
    env: str = typer.Option(
        "doom_health_gathering_supreme",
        "--env",
        "-e",
        help="ViZDoom environment to train on",
    ),
    steps: int = typer.Option(
        4000000,
        "--steps",
        "-s",
        help="Number of environment steps to train for",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of worker processes",
    ),
    envs_per_worker: int = typer.Option(
        4,
        "--envs-per-worker",
        help="Number of environments per worker",
    ),
    device: str = typer.Option(
        "gpu",
        "--device",
        "-d",
        help="Device to use for training (cpu/gpu)",
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-x",
        help="Name for the experiment (optional)",
    ),
):
    """Train a RL agent on ViZDoom environment."""
    console.print(
        f"🎮 Starting training on [bold cyan]{env}[/bold cyan]", style="green"
    )

    # Register ViZDoom components
    register_vizdoom_components()

    # Build training arguments
    train_args = [
        f"--env={env}",
        f"--num_workers={workers}",
        f"--num_envs_per_worker={envs_per_worker}",
        f"--train_for_env_steps={steps}",
        "--async_rl=False",
        "--env_gpu_observations=False",
        "--serial_mode=True",
        f"--device={device}",
    ]

    if experiment_name:
        train_args.append(f"--experiment={experiment_name}")

    # Parse configuration
    cfg = parse_vizdoom_cfg(argv=train_args)

    console.print("\n📋 Training Configuration:")
    console.print(f"  Environment: {cfg.env}")
    console.print(f"  Training steps: {cfg.train_for_env_steps:,}")
    console.print(f"  Workers: {cfg.num_workers}")
    console.print(f"  Envs per worker: {cfg.num_envs_per_worker}")
    console.print(f"  Device: {cfg.device}")
    console.print()

    # Start training
    console.print("🚀 Starting training...", style="bold green")
    status = run_rl(cfg)

    if status == 0:
        console.print("✅ Training completed successfully!", style="bold green")
    else:
        console.print(f"❌ Training failed with status: {status}", style="bold red")

    return status


@app.command()
def evaluate(
    env: str = typer.Option(
        "doom_health_gathering_supreme",
        "--env",
        "-e",
        help="ViZDoom environment to evaluate on",
    ),
    checkpoint_kind: str = typer.Option(
        "best",
        "--load_checkpoint_kind",
        "-l",
        help="Load checkpoint kind (latest/best)",
    ),
    episodes: int = typer.Option(
        10,
        "--episodes",
        "-n",
        help="Number of episodes to evaluate",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of workers to use for evaluation. Should be <= number of available GPUs.",
    ),
    render: bool = typer.Option(
        False,
        "--render/--no_render",
        help="Whether to render the environment during evaluation",
    ),
    push_to_hub: bool = typer.Option(
        False,
        "--push_to_hub",
        help="Whether to push the evaluation results to the Hugging Face Hub",
    ),
    hf_repository: Optional[str] = typer.Option(
        None,
        "--hf_repository",
        help="Hugging Face repository to push the evaluation results to. Must be of the form <username>/<repo_name>",
    ),
):
    """Evaluate a trained RL agent on ViZDoom environment."""
    console.print(
        f"🎯 Starting evaluation on [bold cyan]{env}[/bold cyan]", style="blue"
    )

    # Register ViZDoom components
    register_vizdoom_components()

    # Build evaluation arguments
    eval_args = [
        f"--env={env}",
        f"--load_checkpoint_kind={checkpoint_kind}",
        f"--max_num_episodes={episodes}",
        f"--num_workers={workers}",
        "--no_render" if not render else "--render",
        "--save_video",
        "--push_to_hub" if push_to_hub else "",
        f"--hf_repository={hf_repository}" if push_to_hub else "",
    ]

    # Remove empty arguments
    eval_args = [arg for arg in eval_args if arg]

    console.print("\n📋 Evaluation Configuration:")
    console.print(f"  Environment: {env}")
    console.print(f"  Checkpoint kind: {checkpoint_kind}")
    console.print(f"  Episodes: {episodes}")
    console.print(f"  Render: {render}")
    console.print()

    # Parse configuration for evaluation
    cfg = parse_vizdoom_cfg(argv=eval_args, evaluation=True)

    console.print("🎮 Starting evaluation...", style="bold blue")
    status = enjoy(cfg)

    if status == 0:
        console.print("✅ Evaluation completed successfully!", style="bold green")
    else:
        console.print(f"❌ Evaluation failed with status: {status}", style="bold red")

    return status


@app.command()
def list_envs():
    """List all available ViZDoom environments."""
    console.print("🎮 Available ViZDoom Environments:", style="bold cyan")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Environment Name", style="cyan")
    table.add_column("Description", style="green")

    # Add available environments
    env_descriptions = {
        "doom_basic": "Basic ViZDoom scenario",
        "doom_two_colors_easy": "Two colors easy scenario",
        "doom_health_gathering_supreme": "Health gathering supreme scenario",
        "doom_dm": "Deathmatch scenario",
        "doom_dwango5": "Dwango5 scenario",
        "doom_my_way_home": "My way home scenario",
        "doom_deadly_corridor": "Deadly corridor scenario",
        "doom_defend_the_center": "Defend the center scenario",
        "doom_defend_the_line": "Defend the line scenario",
    }

    for env_name, description in env_descriptions.items():
        table.add_row(env_name, description)

    console.print(table)


if __name__ == "__main__":
    app()
