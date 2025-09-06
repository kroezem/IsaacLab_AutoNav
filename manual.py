#!/usr/bin/env python3
# manual_play.py
# Drive the Prerunner using WASD, or let a trained PPO checkpoint do it.

from __future__ import annotations
import argparse
import time
import torch
import gymnasium as gym
import keyboard  # <- Python keyboard module (pip install keyboard)

from isaaclab.app import AppLauncher

# ------------------------------------------------ CLI
parser = argparse.ArgumentParser(description="Play-test PrerunnerEnv.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to skrl checkpoint (omit for manual driving)")
parser.add_argument("--headless", action="store_true",
                    help="Run Isaac-Sim without viewer")
args = parser.parse_args()

# ------------------------------------------- Isaac-Sim boot
simulation_app = AppLauncher(headless=args.headless).app

# Register env
import isaaclab_tasks.direct.autonav_v11
from isaaclab_tasks.direct.autonav_v11.autonav_env import AutoNavEnvCfg

cfg = AutoNavEnvCfg()
cfg.scene.num_envs = 1

env = gym.make("Isaac-AutoNav-Direct-v11", cfg=cfg, render_mode=None if args.headless else "human")
device = env.unwrapped.device


# ------------------------------------------- Action helper
def wasd_action() -> torch.Tensor:
    t = 0.0
    s = 0.0
    if keyboard.is_pressed("w"): t += 1.0
    if keyboard.is_pressed("s"): t -= 1.0
    if keyboard.is_pressed("d"): s -= 1.0
    if keyboard.is_pressed("a"): s += 1.0
    return torch.tensor([[t, s]], dtype=torch.float32, device=device)


# ------------------------------------------- Load agent (optional)
agent = None
if args.checkpoint:
    from skrl.utils.model_instantiator import SKRLRunnerLoader

    agent = SKRLRunnerLoader.load_agent(args.checkpoint, device=device)
    agent.eval()
    print(f"[INFO] Loaded checkpoint from: {args.checkpoint}")

print("[INFO]  W/S throttle  |  A/D steer  |  R reset  |  Esc quit")
obs, _ = env.reset()

# ------------------------------------------------ Main loop
while simulation_app.is_running():
    if keyboard.is_pressed("esc"):
        break

    if keyboard.is_pressed("r"):
        obs, _ = env.reset()
        continue

    if agent is None:
        action = wasd_action()
    else:
        with torch.no_grad():
            action, *_ = agent.act(obs)

    obs, *_ = env.step(action)

    # -- Decode and Display Heading --
    # Extract the policy observation tensor
    observations = obs["policy"]

    # The last 2 elements are sin(yaw) and cos(yaw)
    sin_yaw = observations[0, -2]
    cos_yaw = observations[0, -1]

    # Convert back to a human-readable angle in degrees
    current_heading_deg = torch.atan2(sin_yaw, cos_yaw).rad2deg()

    # Print the live heading to the console
    print(f"\rCurrent Heading: {current_heading_deg:+.1f}°", end="")

    time.sleep(0.01)

# ------------------------------------------------ Shutdown
env.close()
simulation_app.close()
