from __future__ import annotations

import json
import pathlib
import random
from collections.abc import Sequence

import networkx as nx
import torch
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
import omni.kit.actions.core
import omni.usd
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg, spawn_from_usd
from isaaclab.sim.utils import find_matching_prims
from isaaclab.utils import configclass, math

from .robots.autonav import LIDAR_CFG, ROBOT_CFG

action_registry = omni.kit.actions.core.get_action_registry()


@configclass
class AutoNavEnvCfg(DirectRLEnvCfg):
    # Simulation settings
    dt = 1 / 100
    decimation = 10
    episode_length_s = 10000.0

    # RL settings
    action_space = 2
    observation_space = 32
    state_space = 0

    # Task-specific settings
    collision_kill_dist = 0.30
    collision_penalty_dist = 1.0
    obs_noise = True

    # Scene settings
    sim = sim_utils.SimulationCfg(dt=dt, render_interval=decimation)
    terrain_usd_path: str = "source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v11/terrains/ELW/ELW_v0.usd"
    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    lidar_cfg: RayCasterCfg = LIDAR_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/Rigid_Bodies/Chassis/Electronics_Mounting/Lidar"
    )

    # Robot DOF names
    throttle_dof_name = ["Wheel__Knuckle__Front_Left", "Wheel__Knuckle__Front_Right", "Wheel__Upright__Rear_Right",
                         "Wheel__Upright__Rear_Left"]
    steering_dof_name = ["Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"]

    # Viewer settings
    viewer: ViewerCfg = ViewerCfg(origin_type="world", env_index=0, eye=(10.0, 0, 100.0), lookat=(10.0, 0.0, 0.0))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(env_spacing=0, replicate_physics=True)


class AutoNavEnv(DirectRLEnv):
    cfg: AutoNavEnvCfg

    def __init__(self, cfg: AutoNavEnvCfg, render_mode: str | None = None, **kw):
        super().__init__(cfg, render_mode, **kw)

        # Robot joint indices
        self._throttle_dof_idx, _ = self.prerunner.find_joints(cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.prerunner.find_joints(cfg.steering_dof_name)

        self.task_failed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self._region_pos, self._region_scale = self._get_regions()
        self._num_regions = self._region_pos.shape[0]
        self._visited = torch.zeros((self.num_envs, self._num_regions), device=self.device, dtype=torch.bool)
        self._last_new = torch.zeros((self.num_envs,), device=self.device, dtype=self.episode_length_buf.dtype)

        self._prev_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)

        self._action_stack = torch.zeros((2, self.num_envs, cfg.action_space), device=self.device)
        self._obs_stack = torch.zeros((2, self.num_envs, cfg.observation_space), device=self.device)

    def _setup_scene(self):
        spawn_from_usd("/World/Terrain", UsdFileCfg(usd_path=self.cfg.terrain_usd_path))
        self.prerunner = Articulation(self.cfg.robot_cfg)
        self.lidar = RayCaster(self.cfg.lidar_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["prerunner"] = self.prerunner
        self.scene.sensors["lidar"] = self.lidar

        self._region_pos, self._region_scale = self._get_regions()

        action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera").execute()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.prerunner._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # --- random spawn location ------------------------------------------------
        reg_idx = torch.randint(0, self._num_regions, (n,), device=self.device)
        spawn_pos = self._region_pos[reg_idx]  # (n, 3)

        # --- random orientation ---------------------------------------------------
        yaw = torch.rand(n, device=self.device) * (2 * torch.pi)
        pad_q = torch.stack(
            (torch.cos(yaw * 0.5),
             torch.zeros_like(yaw),
             torch.zeros_like(yaw),
             torch.sin(yaw * 0.5)),
            dim=1,
        )

        # --- write state to sim ---------------------------------------------------
        default_state = self.prerunner.data.default_root_state[env_ids]
        pose = default_state[:, :7].clone()
        pose[:, :3] = spawn_pos + self.scene.env_origins[env_ids, :3]  # keep grid spacing
        pose[:, 3:7] = pad_q

        self.prerunner.write_root_pose_to_sim(pose, env_ids)
        self.prerunner.write_root_velocity_to_sim(default_state[:, 7:].clone(), env_ids)
        self.prerunner.write_joint_state_to_sim(
            self.prerunner.data.default_joint_pos[env_ids],
            self.prerunner.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )

        self._visited[env_ids] = False
        self.task_failed[env_ids] = False
        self._action_stack[:, env_ids] = 0
        self._obs_stack[:, env_ids] = 0
        self._prev_lin_vel[env_ids] = 0
        self._last_new[env_ids] = self.episode_length_buf[env_ids]

    def _pre_physics_step(self, actions: torch.Tensor):
        self._action_stack = torch.roll(self._action_stack, shifts=1, dims=0)
        self._action_stack[0] = actions

        delayed_action = self._action_stack[-1]
        esc = torch.clamp(delayed_action[:, 0], 0.0, 1.0)
        steer = torch.clamp(delayed_action[:, 1], -1.0, 1.0)

        self._throttle_action = (esc * 3.0).repeat_interleave(4).reshape(-1, 4)
        self._steering_action = (steer * 0.7).repeat_interleave(2).reshape(-1, 2)

    def _apply_action(self):
        self.prerunner.set_joint_effort_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.prerunner.set_joint_position_target(self._steering_action, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        # Lidar data
        hits_vec = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
        lidar_obs = torch.clamp(torch.norm(hits_vec, dim=-1), 0.0, 12.0)
        if self.cfg.obs_noise:
            lidar_obs = torch.clamp(lidar_obs + torch.randn_like(lidar_obs) * (0.03 * lidar_obs + 0.01), 0.0, 12.0)
        self._latest_lidar = lidar_obs.clone()
        lidar_obs = 1 / (0.1 + lidar_obs)

        # IMU data
        quat_w = self.prerunner.data.root_quat_w
        ang_vel_b = math.quat_rotate_inverse(quat_w, self.prerunner.data.root_ang_vel_w)
        gyro_z = ang_vel_b[:, 2:3]  # radians/sec

        world_accel = (self.prerunner.data.root_lin_vel_w - self._prev_lin_vel) / self.cfg.sim.dt
        proper_accel_w = world_accel - torch.tensor([0.0, 0.0, -9.81], device=self.device)
        proper_accel_b = math.quat_rotate_inverse(quat_w, proper_accel_w)
        accel_x = proper_accel_b[:, 0:1] / 9.81  # g

        self._prev_lin_vel = self.prerunner.data.root_lin_vel_w.clone()

        last_action = self._action_stack[0]
        raw_obs = torch.cat([last_action, lidar_obs, gyro_z, accel_x], dim=1)

        # if self.num_envs == 1:
        #     print(raw_obs)

        self._obs_stack = torch.roll(self._obs_stack, shifts=1, dims=0)
        self._obs_stack[0] = raw_obs

        return {"policy": self._obs_stack[-1].reshape(self.num_envs, -1)}

    def _get_rewards(self) -> torch.Tensor:
        velocity = self.prerunner.data.root_lin_vel_w[:, 0]  # m/s

        d_min = self._latest_lidar.min(dim=1).values  # closest LiDAR hit

        clearance_bonus = torch.clamp(d_min, 0.0, 2.0)

        proximity_penalty = 1 / d_min

        # Exploration bonus for visiting a new region for the first time.
        reg_id, inside = self._current_region()
        first_visit = inside & (~self._visited[torch.arange(self.num_envs), reg_id])
        self._visited[torch.arange(self.num_envs), reg_id] |= inside  # mark visited
        self._last_new[first_visit] = self.episode_length_buf[first_visit]

        self.task_failed = d_min < self.cfg.collision_kill_dist

        steer_jerk = torch.abs(self._action_stack[0][:, 1] - self._action_stack[1][:, 1])

        reward = (
                15.0 * velocity  # Encourage forward progress
                + 20.0 * first_visit
                # + 1 * clearance_bonus
                - 1 * proximity_penalty  # Penalize getting too close to obstacles
                - 0.05 * steer_jerk
                - 50 * self.task_failed
        )
        reward[torch.isnan(reward)] = 0.0
        return reward

    def _get_dones(self):
        stalled = (self.episode_length_buf - self._last_new) > 100
        truncated = self.episode_length_buf > 600
        dones = self.task_failed | stalled
        self.task_failed[:] = False

        return dones, truncated

    def _get_regions(self):
        pos, scale = [], []
        for r_prim in find_matching_prims("/World/Terrain/Regions/r_.*"):
            xf = UsdGeom.Xformable(r_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = xf.ExtractTranslation()
            s = Gf.Vec3d(1.0)
            for op in UsdGeom.Xformable(r_prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    s = op.Get()
                    break
            pos.append([t[0], t[1], t[2]])
            scale.append(min(s) * 0.5)

        # tensors shared by *all* envs
        return (
            torch.tensor(pos, dtype=torch.float32, device=self.device),
            torch.tensor(scale, dtype=torch.float32, device=self.device),
        )

    def _current_region(self):
        pos = self.prerunner.data.root_pos_w[:, :3].unsqueeze(1)  # (E,1,3)
        dxyz = pos - self._region_pos.unsqueeze(0)  # (E,R,3)
        dist = torch.linalg.norm(dxyz, dim=-1)  # (E,R)
        idx = dist.argmin(dim=1)  # (E,)
        inside = dist[torch.arange(self.num_envs), idx] < self._region_scale[idx]
        return idx, inside
