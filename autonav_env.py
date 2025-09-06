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
    dt = 1 / 60
    decimation = 2
    episode_length_s = 10000.0

    # RL settings
    action_space = 2
    observation_space = 36
    state_space = 0

    # Task-specific settings
    collision_kill_dist = 0.2
    obs_noise = True

    # Scene settings
    sim = sim_utils.SimulationCfg(dt=dt, render_interval=decimation)
    terrain_name: str = "ELW_v0"
    terrain_usd_path: str = "source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/terrains/ELW/ELW_v0.usd"
    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    lidar_cfg: RayCasterCfg = LIDAR_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/Rigid_Bodies/Chassis/Electronics_Mounting/Lidar"
    )
    # Optional list of fixed destination node names. If None, goal is randomized.
    destination: list[str] | None = None
    # Optional fixed start node name. If None, start is randomized.
    start: str | None = None
    # If True, forces the agent down a single random path from its start node each episode.
    force_path: bool = False

    # Robot DOF names
    throttle_dof_name = ["Wheel__Knuckle__Front_Left", "Wheel__Knuckle__Front_Right", "Wheel__Upright__Rear_Right",
                         "Wheel__Upright__Rear_Left"]
    steering_dof_name = ["Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"]

    # Viewer settings
    viewer: ViewerCfg = ViewerCfg(origin_type="world", env_index=0, eye=(0.0, 0, 25.0), lookat=(0.0, 0.0, 0.0))
    # viewer: ViewerCfg = ViewerCfg(origin_type="asset_root", asset_name="prerunner", env_index=0, eye=(0.0, 0, 25.0),lookat=(0.0, 0.0, 0.0))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(env_spacing=0, replicate_physics=True)


class AutoNavEnv(DirectRLEnv):
    cfg: AutoNavEnvCfg

    """
    ================
    Core Class Methods
    ================
    """

    def __init__(self, cfg: AutoNavEnvCfg, render_mode: str | None = None, **kw):
        # Initialize graph attributes before super call
        self.region_network = nx.Graph()
        self.region_node_names: list[str] = []
        self._region_centers: torch.Tensor | None = None
        self._region_radii: torch.Tensor | None = None
        self.network_json_path = pathlib.Path(cfg.terrain_usd_path).parent / "region_network.json"

        super().__init__(cfg, render_mode, **kw)

        if self.cfg.destination:
            print(f"[INFO] Using fixed destination list: {self.cfg.destination}")
        else:
            print("[INFO] No fixed destination provided. Goal will be randomized each episode.")

        if self.cfg.start:
            print(f"[INFO] Using fixed start node: {self.cfg.start}")
        else:
            print("[INFO] No fixed start provided. Start will be randomized each episode.")

        if self.cfg.force_path:
            print("[INFO] --force-path enabled: A single random path will be chosen from the start node each episode.")

        # Robot joint indices
        self._throttle_dof_idx, _ = self.prerunner.find_joints(cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.prerunner.find_joints(cfg.steering_dof_name)

        # State buffers
        self.task_failed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.success = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self.last_goal = torch.zeros((self.num_envs,), device=self.device, dtype=self.episode_length_buf.dtype)
        self.max_route_len = 50
        self._route_pos = torch.zeros((self.num_envs, self.max_route_len, 3), device=self.device)
        self._route_scale = torch.zeros((self.num_envs, self.max_route_len, 3), device=self.device)
        self._route_len = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._target_index = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._position_error = torch.zeros((self.num_envs,), device=self.device)
        self._prev_position_error = torch.zeros((self.num_envs,), device=self.device)
        self._prev_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)

        self._throttle_mod = torch.ones(self.num_envs, device=self.device)
        self._steering_mod = torch.ones(self.num_envs, device=self.device)
        self._lidar_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self._lidar_scale = torch.ones(self.num_envs, 1, device=self.device)

        # Observation and action history stacks
        self._action_stack = torch.zeros((4, self.num_envs, cfg.action_space), device=self.device)

        self._lidar_stack = torch.zeros((4, self.num_envs, 28), device=self.device)
        # self._obs_stack = torch.zeros((4, self.num_envs, cfg.observation_space), device=self.device)

        # Buffer to signal final destination was reached
        self.reached_final_destination = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    """
    ==========================================
    Isaac Lab Simulation Lifecycle Overrides
    ==========================================
    """

    def _setup_scene(self):
        # Spawn assets
        spawn_from_usd("/World/Terrain", UsdFileCfg(usd_path=self.cfg.terrain_usd_path))
        self.prerunner = Articulation(self.cfg.robot_cfg)
        self.lidar = RayCaster(self.cfg.lidar_cfg)

        # Finalize scene
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["prerunner"] = self.prerunner
        self.scene.sensors["lidar"] = self.lidar

        # Generate region graph from USD
        self._get_region_network()
        self.region_node_names = sorted(list(self.region_network.nodes))

        # Populate region tensors
        if self.region_node_names:
            self._region_centers = torch.stack([self.region_network.nodes[n]['pos'] for n in self.region_node_names])
            self._region_radii = torch.tensor([self.region_network.nodes[n]['radius'] for n in self.region_node_names],
                                              device=self.device)

        # Set viewer lighting
        action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera").execute()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.prerunner._ALL_INDICES
        super()._reset_idx(env_ids)

        # Create and set a new route
        self._update_routes(env_ids)

        # Set robot pose at the start of the new route
        pad_pos = self._route_pos[env_ids, 0]

        # Always face the next waypoint on reset
        vec = self._route_pos[env_ids, 1] - pad_pos
        # yaw = torch.atan2(vec[:, 1], vec[:, 0])
        yaw = torch.atan2(vec[:, 1], vec[:, 0]) + (torch.rand_like(vec[:, 0]) * 2 - 1) * (torch.pi / 3)

        pad_quat = torch.stack(
            [torch.cos(yaw / 2),
             torch.zeros_like(yaw),
             torch.zeros_like(yaw),
             torch.sin(yaw / 2)],
            dim=1)

        # Write new pose and default velocities to sim
        default_state = self.prerunner.data.default_root_state[env_ids]
        pose = default_state[:, :7].clone()
        pose[:, :3] = pad_pos + self.scene.env_origins[env_ids, :3]
        pose[:, 3:7] = pad_quat
        self.prerunner.write_root_pose_to_sim(pose, env_ids)
        self.prerunner.write_root_velocity_to_sim(default_state[:, 7:].clone(), env_ids)
        self.prerunner.write_joint_state_to_sim(self.prerunner.data.default_joint_pos[env_ids],
                                                self.prerunner.data.default_joint_vel[env_ids], None, env_ids)

        self._throttle_mod[env_ids] = torch.rand(len(env_ids), device=self.device) * 0.2 + 0.9
        self._steering_mod[env_ids] = torch.rand(len(env_ids), device=self.device) * 0.2 + 0.9
        self._lidar_bias[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) - 0.5) * 0.06
        self._lidar_scale[env_ids] = torch.rand(len(env_ids), 1, device=self.device) * 0.04 + 0.98

        # Reset state buffers
        self._target_index[env_ids] = 1
        self.last_goal[env_ids] = self.episode_length_buf[env_ids]
        self._position_error[env_ids] = 0.0
        self._prev_position_error[env_ids] = 0.0
        self.task_failed[env_ids] = False
        self._action_stack[:, env_ids] = 0
        self._lidar_stack[:, env_ids] = 0
        self._prev_lin_vel[env_ids] = 0
        self.reached_final_destination[env_ids] = False

    def _pre_physics_step(self, actions: torch.Tensor):
        self._action_stack = torch.roll(self._action_stack, shifts=1, dims=0)
        self._action_stack[0] = actions

        delayed_action = self._action_stack[-1]
        esc = torch.clamp(delayed_action[:, 0], 0.0, 1.0) * self._throttle_mod
        steer = torch.clamp(delayed_action[:, 1], -1.0, 1.0) * self._steering_mod

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
            # (1) Per-episode bias / scale  (±3 cm, ±2 %)
            lidar_obs = lidar_obs * self._lidar_scale + self._lidar_bias

            # (2) Additive Gaussian noise  (σ = 0.03·d + 0.03)
            sigma = 0.03 * lidar_obs + 0.03
            lidar_obs = lidar_obs + torch.randn_like(lidar_obs) * sigma

            # (3) Random drop-outs  (p = 3 %)
            drop_mask = torch.rand_like(lidar_obs) < 0.03
            lidar_obs[drop_mask] = 12.0  # sensor returns “no hit”

            # (4) Sporadic outliers  (p = 0.5 %)
            outlier_mask = (torch.rand_like(lidar_obs) < 0.005) & ~drop_mask
            lidar_obs[outlier_mask] = torch.rand_like(lidar_obs[outlier_mask]) * 11.9 + 0.1

            # (5) Hard limits
            lidar_obs = torch.clamp(lidar_obs, 0.0, 12.0)

        self._latest_lidar = lidar_obs.clone()
        lidar_obs = 1 / (0.1 + lidar_obs)

        # IMU data
        quat_w = self.prerunner.data.root_quat_w
        ang_vel_b = math.quat_rotate_inverse(quat_w, self.prerunner.data.root_ang_vel_w)
        gyro_z = ang_vel_b[:, 2:3]
        world_accel = (self.prerunner.data.root_lin_vel_w - self._prev_lin_vel) / self.cfg.sim.dt
        proper_accel_b = math.quat_rotate_inverse(quat_w,
                                                  world_accel - torch.tensor([0.0, 0.0, -9.81], device=self.device))
        accel_x = proper_accel_b[:, 0:1] / 9.81
        self._prev_lin_vel = self.prerunner.data.root_lin_vel_w.clone()

        # Robot's own heading
        # MODIFIED: Correct the simulator's yaw to match the real-world sensor's frame.
        # The standard sim yaw is 90 degrees offset from the sensor's output.
        yaw_sim = self.prerunner.data.heading_w
        yaw = yaw_sim + (torch.pi / 2.0)
        heading_obs = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)

        # Vague goal direction vector from the current region's center (world-frame).
        tgt_idx = self._target_index
        goal_pos_w = self._route_pos[self.prerunner._ALL_INDICES, tgt_idx]

        # Initialize the goal quadrant observation to zeros.
        goal_cardinal_obs = torch.zeros((self.num_envs, 2), device=self.device)

        # Find which environments are in a valid region.
        current_region_ids = self._get_current_region_ids().squeeze(-1).long()
        valid_region_mask = current_region_ids != -1

        # Calculate the goal quadrant ONLY for those in a valid region.
        if valid_region_mask.any():
            # Get data for valid agents
            valid_goal_pos_w = goal_pos_w[valid_region_mask]
            valid_region_ids = current_region_ids[valid_region_mask]

            # Use the region center as the origin for the vector
            origin_pos_w = self._region_centers[valid_region_ids]
            vec_to_goal_w = valid_goal_pos_w - origin_pos_w

            # Normalize the vector to get direction components
            norm = torch.norm(vec_to_goal_w, dim=-1, keepdim=True)
            norm_vec = torch.zeros_like(vec_to_goal_w)
            non_zero_mask = norm.squeeze() > 1e-6
            if non_zero_mask.any():
                norm_vec[non_zero_mask] = vec_to_goal_w[non_zero_mask] / norm[non_zero_mask]

            # Get the East/West and North/South components based on the standard world frame (+X=E, +Y=N)
            ew_comp = -norm_vec[:, 0]
            ns_comp = norm_vec[:, 1]

            # Define the threshold for pure cardinal directions (cone of 45 degrees)
            threshold = 0.92388  # cos(22.5 degrees)

            # Initialize output vector with intercardinal directions
            cardinal_vec = torch.stack([ew_comp.sign(), ns_comp.sign()], dim=-1)

            # Override with pure cardinal directions if the vector is within the deadzone
            # East
            cardinal_vec[ew_comp > threshold] = torch.tensor([1.0, 0.0], device=self.device)
            # West
            cardinal_vec[ew_comp < -threshold] = torch.tensor([-1.0, 0.0], device=self.device)
            # North
            cardinal_vec[ns_comp > threshold] = torch.tensor([0.0, 1.0], device=self.device)
            # South
            cardinal_vec[ns_comp < -threshold] = torch.tensor([0.0, -1.0], device=self.device)

            goal_cardinal_obs[valid_region_mask] = cardinal_vec

        dropout_mask = torch.rand((self.num_envs,), device=self.device) < 0.25
        goal_cardinal_obs[dropout_mask] = 0

        # Assemble the full observation vector

        self._lidar_stack = torch.roll(self._lidar_stack, shifts=1, dims=0)
        self._lidar_stack[0] = lidar_obs

        raw_obs = torch.cat([
            self._action_stack[0],
            self._lidar_stack[-1],
            gyro_z, accel_x,
            heading_obs,
            goal_cardinal_obs
        ], dim=1)

        return {"policy": raw_obs.reshape(self.num_envs, -1)}

    def _get_rewards(self) -> torch.Tensor:
        env_ids = self.prerunner._ALL_INDICES
        tgt_idx = self._target_index[env_ids]
        wp_pos = self._route_pos[env_ids, tgt_idx]

        # -- Distance / heading --
        vec_to_wp_w = wp_pos - self.prerunner.data.root_pos_w[env_ids, :3]
        dist_now = torch.norm(vec_to_wp_w, dim=-1)

        yaw = self.prerunner.data.heading_w
        robot_fwd_vec_w = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        vec_to_wp_w_2d_norm = vec_to_wp_w[:, :2] / (torch.norm(vec_to_wp_w[:, :2], dim=-1, keepdim=True) + 1e-6)

        # -- Base terms --
        progress_reward = self._prev_position_error - dist_now
        self._prev_position_error = dist_now.clone()

        lin_vel_b = math.quat_rotate_inverse(self.prerunner.data.root_quat_w,
                                             self.prerunner.data.root_lin_vel_w)
        velocity_reward = torch.clamp(lin_vel_b[:, 0], min=0.0, max=5.0)

        dmin = self._latest_lidar.min(dim=1).values
        clearance_reward = torch.clamp(dmin, 0.0, 5.0)

        steer_jerk_penalty = torch.abs(self._action_stack[0][:, 1] - self._action_stack[1][:, 1])

        crashed = dmin < self.cfg.collision_kill_dist

        alignment_reward = torch.sum(robot_fwd_vec_w * vec_to_wp_w_2d_norm, dim=-1).clamp(min=0.0)
        spaciousness_reward = self._latest_lidar.mean(dim=1)

        # -- Proximity & spin penalties --
        proximity_penalty = torch.zeros_like(dmin)
        mask = dmin < 1.0
        proximity_penalty[mask] = (1.0 - dmin[mask]).square()

        ang_vel_b = math.quat_rotate_inverse(self.prerunner.data.root_quat_w,
                                             self.prerunner.data.root_ang_vel_w)
        spin_penalty = ang_vel_b[:, 2].square()

        # -- Stationary penalty --
        fwd_speed = lin_vel_b[:, 0]
        stall_penalty = torch.clamp(0.25 - fwd_speed, 0.0).square()

        # -- Waypoint logic --
        radius_wp = self._route_scale[env_ids, tgt_idx, 0] * 0.5
        reached_wp = dist_now < radius_wp

        is_final_wp = tgt_idx >= (self._route_len[env_ids] - 1)
        self.reached_final_destination = reached_wp & is_final_wp

        advance = reached_wp & ~is_final_wp
        if advance.any():
            self._target_index[advance] += 1
            next_wp_pos = self._route_pos[advance, self._target_index[advance]]
            self._prev_position_error[advance] = torch.norm(
                next_wp_pos - self.prerunner.data.root_pos_w[advance, :3], dim=-1
            )
            self.last_goal[advance] = self.episode_length_buf[advance]

        reward = (
                2.5 * progress_reward
                + 1.0 * velocity_reward
                + 1.0 * alignment_reward
                + 0.1 * spaciousness_reward
                + 20.0 * reached_wp
                + 100.0 * self.reached_final_destination
                - 10.0 * proximity_penalty
                - 2.0 * spin_penalty
                - 0.5 * steer_jerk_penalty
                - 5.0 * stall_penalty
                - 50.0 * crashed
                - 5.0 * self.task_failed
        )

        reward[torch.isnan(reward)] = 0.0
        return reward

    def _get_dones(self):
        stalled = (self.episode_length_buf - self.last_goal) > 300
        too_close = self._latest_lidar.min(dim=1).values < self.cfg.collision_kill_dist

        # Termination: these are failure conditions that end the episode.
        dones = stalled | self.task_failed | too_close
        self.task_failed[:] = False

        # Truncation: these are success or time-limit conditions that also end the episode.
        truncations = self.reached_final_destination

        return dones, truncations

    """
    =========================
    Internal Helper Methods
    =========================
    """

    def _update_routes(self, env_ids: Sequence[int]):
        """Generates paths for each agent from a start node to an end node."""
        if not self.region_network or not self.region_node_names:
            raise RuntimeError("Region network has not been initialized.")
        if len(self.region_node_names) < 2:
            raise ValueError("Region network needs at least two regions to create a route.")

        for env_id in env_ids:
            # Set destination node
            if self.cfg.destination:
                end_node = random.choice(self.cfg.destination)
            else:
                end_node = random.choice(self.region_node_names)

            if self.cfg.force_path:
                self._generate_forced_path(env_id, end_node)
            else:
                self._generate_standard_path(env_id, end_node)

    def _generate_standard_path(self, env_id: int, end_node: str):
        """Generates a standard shortest path with a random or fixed start and end node."""
        path_found = False
        while not path_found:
            # Set start node from config if provided, otherwise randomize.
            if self.cfg.start:
                start_node = self.cfg.start
            else:
                start_node = random.choice(self.region_node_names)

            if start_node == end_node:
                # If start is not fixed, just pick a new one.
                if not self.cfg.start:
                    continue
                # If start is fixed, this is a configuration error.
                else:
                    raise ValueError(
                        f"Fixed start node '{start_node}' and destination node '{end_node}' cannot be the same.")

            try:
                path_nodes = nx.shortest_path(self.region_network, source=start_node, target=end_node, weight="weight")
                path_len = len(path_nodes)

                if path_len < 2 or path_len > self.max_route_len:
                    if self.cfg.start:
                        raise ValueError(f"Path between fixed start '{start_node}' and '{end_node}' is invalid length.")
                    continue

                self._store_path(env_id, path_nodes)
                path_found = True
            except nx.NetworkXNoPath:
                if self.cfg.start:
                    raise ValueError(f"No path found between fixed start '{start_node}' and '{end_node}'.")
                continue

    def _generate_forced_path(self, env_id: int, end_node: str):
        """Generates a path by forcing the agent down one of the start node's neighbors."""
        path_found = False
        while not path_found:
            temp_graph = self.region_network.copy()
            start_node = random.choice(self.region_node_names)

            if start_node == end_node:
                continue

            neighbors = list(temp_graph.neighbors(start_node))
            if not neighbors:
                continue  # Isolated start node, retry

            chosen_neighbor = random.choice(neighbors)
            edges_to_remove = [(start_node, n) for n in neighbors if n != chosen_neighbor]
            temp_graph.remove_edges_from(edges_to_remove)

            try:
                path_nodes = nx.shortest_path(temp_graph, source=start_node, target=end_node, weight="weight")
                path_len = len(path_nodes)

                if path_len < 2 or path_len > self.max_route_len:
                    continue  # Path is invalid, retry

                self._store_path(env_id, path_nodes)
                path_found = True
            except nx.NetworkXNoPath:
                # Chosen path was a dead end, retry
                continue

    def _store_path(self, env_id: int, path_nodes: list[str]):
        """Helper function to store a valid path in the instance buffers."""
        path_len = len(path_nodes)
        path_pos = torch.stack([self.region_network.nodes[n]['pos'] for n in path_nodes])
        path_radii = torch.tensor([self.region_network.nodes[n]['radius'] for n in path_nodes], device=self.device)
        path_scale = (path_radii * 2.0).unsqueeze(1).repeat(1, 3)

        self._route_len[env_id] = path_len
        self._route_pos[env_id, :path_len] = path_pos
        self._route_scale[env_id, :path_len] = path_scale
        if path_len < self.max_route_len:
            self._route_pos[env_id, path_len:] = 0.0
            self._route_scale[env_id, path_len:] = 0.0

    def _get_current_region_ids(self) -> torch.Tensor:
        if self._region_centers is None:
            return torch.full((self.num_envs, 1), -1.0, device=self.device)

        robot_pos_2d = self.prerunner.data.root_pos_w[:, :2].unsqueeze(1)
        region_centers_2d = self._region_centers[:, :2].unsqueeze(0)
        distances_to_centers = torch.norm(robot_pos_2d - region_centers_2d, dim=-1)
        is_inside = distances_to_centers <= self._region_radii.unsqueeze(0)

        distances_to_centers[~is_inside] = float('inf')
        region_ids = torch.argmin(distances_to_centers, dim=1)
        not_in_any_region = ~torch.any(is_inside, dim=1)
        region_ids[not_in_any_region] = -1

        return region_ids.unsqueeze(-1).float()

    def _get_region_network(self):
        """
        Generates the region network graph from primitives in the USD scene
        and saves it to a JSON file, overwriting any existing file.
        """
        print("[INFO] Generating region network from USD and saving to cache...")
        self._generate_network_from_usd()
        self._save_network_to_json()

    def _generate_network_from_usd(self):
        """Builds the network graph by parsing region primitives from the USD stage."""
        self.region_network.clear()
        nodes = []
        for r_prim in find_matching_prims("/World/Terrain/Regions/r_.*"):
            xformable = UsdGeom.Xformable(r_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = world_transform.ExtractTranslation()
            scale_vec = Gf.Vec3d(1.0)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_vec = op.Get()
                    break
            center = torch.tensor([translation[0], translation[1], translation[2]], device=self.device)
            # Radius is half of the smallest scale dimension
            radius = min(scale_vec) * 0.5
            name = r_prim.GetName()
            self.region_network.add_node(name, pos=center, radius=radius)
            nodes.append((name, center, radius))

        # Create edges between overlapping regions
        for i in range(len(nodes)):
            name_i, center_i, radius_i = nodes[i]
            for j in range(i + 1, len(nodes)):
                name_j, center_j, radius_j = nodes[j]
                dist = torch.linalg.norm(center_i - center_j)
                # Add a small tolerance for floating point errors
                if dist <= radius_i + radius_j + 1e-3:
                    self.region_network.add_edge(name_i, name_j, weight=dist.item())

    def _save_network_to_json(self):
        """Saves the current region network graph to a JSON file."""
        if not self.region_network:
            print("[WARNING] Region network is empty. Skipping save.")
            return

        output_data = {"regions": {}}
        for node_name, node_data in self.region_network.nodes(data=True):
            neighbors = list(self.region_network.neighbors(node_name))
            output_data["regions"][node_name] = {
                "position": node_data["pos"].cpu().tolist(),
                "radius": node_data["radius"],
                "neighbors": sorted(neighbors),
            }

        with open(self.network_json_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"[INFO] Saved region network to cache: {self.network_json_path}")
