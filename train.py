# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom self-launching script to train the AutoNav agent with skrl.
- Uses custom log and snapshot directories based on a user-defined tag.
- Snapshots important code files for reproducibility.
- Based on the official Isaac Lab skrl training script structure.
"""

import argparse
import pathlib
import shutil
import sys
from datetime import datetime

# -- 1. BOILERPLATE: ARGUMENT PARSING & APP LAUNCHER --

from isaaclab.app import AppLauncher

# Create the argument parser
parser = argparse.ArgumentParser(description="Custom trainer for the AutoNav RL agent.")

# Add task-specific and common arguments
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-AutoNav-Direct-v8", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and RL agent.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--terrain", type=str, default="ELW_v0",
                    help="Base name of the terrain USD file (e.g., 'hobbs_v0').")
parser.add_argument("--tag", type=str, default="", help="Custom tag for the training run name.")
parser.add_argument("--destination", type=str, nargs='+', default=None,
                    help="One or more destination region names (e.g., --destination r_26 r_41).")
# MODIFIED: Replaced --focus-node with --force-path flag.
parser.add_argument("--force_path", action="store_true",
                    help="If set, forces the agent down a single random path from its start node each episode.")

# Append AppLauncher cli args and parse them
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear out sys.argv for Hydra and relaunch the app
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- 2. POST-LAUNCH IMPORTS AND SETUP --

import gymnasium as gym
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# -- 3. CUSTOM CONFIGURATION: SNAPSHOTS --

# A list of important code files to save with the training run for reproducibility.
CODE_FILES_TO_SNAPSHOT = [
    "source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/train.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/autonav_env.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/robots/autonav.py",
]

# The entry point for the skrl agent configuration in the task's config file.
agent_cfg_entry_point = "skrl_cfg_entry_point"


# -- 4. MAIN TRAINING FUNCTION --

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Main function to configure and train the AutoNav agent."""

    # -- START: CONFIG OVERRIDES FROM CLI --
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed

    # Set terrain properties from the command-line argument.
    if args_cli.terrain:
        env_cfg.terrain_name = args_cli.terrain
        terrain_folder = env_cfg.terrain_name.split('_')[0]
        env_cfg.terrain_usd_path = (f"source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/terrains/"
                                    f"{terrain_folder}/{env_cfg.terrain_name}.usd")

    # Pass the destination list to the environment config.
    if args_cli.destination:
        env_cfg.destination = args_cli.destination

    # MODIFIED: Pass the force_path flag to the environment config.
    if args_cli.force_path:
        env_cfg.force_path = True
    # -- END: CONFIG OVERRIDES FROM CLI --

    # MODIFIED: Custom Path and Snapshot Logic using the new format
    name_parts = [env_cfg.terrain_name]
    if args_cli.force_path:
        name_parts.append("force-path")
    if args_cli.destination:
        # Join multiple destinations with a hyphen for a clean filename
        dest_str = "-".join(args_cli.destination)
        name_parts.append(dest_str)
    if args_cli.tag:
        name_parts.append(args_cli.tag)

    base_name = "_".join(name_parts)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{base_name}_{timestamp}"

    log_root = pathlib.Path("source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/logs")
    snapshot_root = pathlib.Path("source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/snapshots")

    log_dir = log_root / run_name
    snapshot_dir = snapshot_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Configure skrl agent's experiment directory
    agent_cfg["agent"]["experiment"]["directory"] = str(log_dir)
    agent_cfg["agent"]["experiment"]["experiment_name"] = ""

    print(f"[INFO] Logging experiment in: {log_dir}")
    print(f"[INFO] Saving snapshots in: {snapshot_dir}")

    # Save a copy of the important code files and configs
    for file_path_str in CODE_FILES_TO_SNAPSHOT:
        source_path = pathlib.Path(file_path_str)
        if source_path.is_file():
            shutil.copy(source_path, snapshot_dir / source_path.name)
        else:
            print(f"[Warning] Could not find file to snapshot: {source_path}")

    dump_yaml(str(snapshot_dir / "env.yaml"), env_cfg)
    dump_yaml(str(snapshot_dir / "agent.yaml"), agent_cfg)

    # Standard Training Setup
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(env, agent_cfg)

    # Resume from checkpoint if provided
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # Train the agent
    runner.run()
    env.close()


# -- 5. SCRIPT ENTRY POINT --

if __name__ == "__main__":
    main()
    simulation_app.close()
