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
# Add task-specific arguments that might be in the config
parser.add_argument("--destination", type=str, nargs='+', default=None,
                    help="Optional fixed destination for the test.")
parser.add_argument("--start", type=str, default=None,
                    help="Optional fixed start for the test.")
args = parser.parse_args()

# ------------------------------------------- Isaac-Sim boot
simulation_app = AppLauncher(headless=args.headless).app

# Register env
# Assuming the user has updated their local files to v8 to match the script
import isaaclab_tasks.direct.autonav_v8
from isaaclab_tasks.direct.autonav_v8.autonav_env import AutoNavEnvCfg

# Create a config and override with any CLI args
cfg = AutoNavEnvCfg()
cfg.scene.num_envs = 1
if args.destination:
    cfg.destination = args.destination
if args.start:
    cfg.start = args.start


env = gym.make("Isaac-AutoNav-Direct-v8", cfg=cfg, render_mode=None if args.headless else "human")
device = env.unwrapped.device


# ------------------------------------------- Action helper
def wasd_action() -> torch.Tensor:
    t = 0.0
    s = 0.0
    if keyboard.is_pressed("w"): t += 1.0
    if keyboard.is_pressed("s"): t -= 1.0
    if keyboard.is_pressed("d"): s -= 1.0
    if keyboard.is_pressed("a"): s += 1.0
    # The action space is just throttle and steer
    return torch.tensor([[t, s]], dtype=torch.float32, device=device)


# ------------------------------------------- Load agent (optional)
agent = None
if args.checkpoint:
    from skrl.utils.model_instantiator import SKRLRunnerLoader

    # Note: SKRLRunnerLoader is deprecated. For newer skrl versions, you might need to load differently.
    # This assumes it works with your current version.
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
            # The agent expects a dictionary of observations
            action, *_ = agent.act(obs)

    obs, *_ = env.step(action)

    # -- MODIFIED: Decode and Display Heading and Goal Compass Direction --
    # Extract the policy observation tensor
    obs_tensor = obs["policy"]

    # Heading is at indices -4 and -3 in the 36-element observation vector
    sin_yaw = obs_tensor[0, -4].item()
    cos_yaw = obs_tensor[0, -3].item()
    current_heading_deg = torch.atan2(torch.tensor(sin_yaw), torch.tensor(cos_yaw)).rad2deg()

    # Goal quadrant is at indices -2 and -1 (world-frame)
    # Based on the env's coordinate system (+X=N, +Y=W):
    # obs[-2] is the West(+1)/East(-1) component
    # obs[-1] is the North(+1)/South(-1) component
    goal_we = obs_tensor[0, -2].item()
    goal_ns = obs_tensor[0, -1].item()

    # Create a mapping from the vector to a compass direction string
    ns_str = ""
    if goal_ns > 0: ns_str = "N"
    elif goal_ns < 0: ns_str = "S"

    we_str = ""
    if goal_we > 0: we_str = "W"
    elif goal_we < 0: we_str = "E"

    compass_display = ns_str + we_str
    if not compass_display:
        compass_display = "On Target"


    # Print the live data to the console
    print(f"\rHeading: {current_heading_deg: >+6.1f}° (sin: {sin_yaw: >+6.3f}, cos: {cos_yaw: >+6.3f}) | Goal Bearing: {compass_display: <10}", end="")

    time.sleep(0.01)

# ------------------------------------------------ Shutdown
env.close()
simulation_app.close()
