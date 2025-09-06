# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pathlib
import shutil
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# ── 1. ARGUMENTS AND APP LAUNCH ────────────────────────────────────────

parser = argparse.ArgumentParser(description="Custom trainer for the AutoNav RL agent.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--tag", type=str, default="", help="Optional tag to label the training run.")
parser.add_argument("--task", type=str, default="Isaac-AutoNav-Direct-v11")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO"])

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. POST-LAUNCH IMPORTS ─────────────────────────────────────────────

import gymnasium as gym
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# ── 3. MAIN TRAINING FUNCTION ──────────────────────────────────────────

CODE_FILES_TO_SNAPSHOT = [
    "source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v11/train.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v11/autonav_env.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v11/robots/autonav.py",
]


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg["seed"] = args_cli.seed

    run_name = (
        f"{args_cli.algorithm.upper()}_{args_cli.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}".strip("_")
    )

    log_dir = pathlib.Path("source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v11/logs") / run_name
    snapshot_dir = pathlib.Path("source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v11/snapshots") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg["agent"]["experiment"]["directory"] = str(log_dir)
    agent_cfg["agent"]["experiment"]["experiment_name"] = ""

    print(f"[INFO] Logging experiment to: {log_dir}")
    print(f"[INFO] Saving snapshots to: {snapshot_dir}")

    for path_str in CODE_FILES_TO_SNAPSHOT:
        src = pathlib.Path(path_str)
        if src.is_file():
            shutil.copy(src, snapshot_dir / src.name)
        else:
            print(f"[Warning] Missing snapshot file: {src}")

    dump_yaml(str(snapshot_dir / "env.yaml"), env_cfg)
    dump_yaml(str(snapshot_dir / "agent.yaml"), agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(env, agent_cfg)

    if args_cli.checkpoint:
        from isaaclab.utils.assets import retrieve_file_path
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    runner.run()
    env.close()


# ── 4. ENTRY POINT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
    simulation_app.close()
