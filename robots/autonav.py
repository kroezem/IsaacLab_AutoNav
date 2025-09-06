import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns

# These are the prim paths to the collision meshes of the wheels, relative to the robot's root prim.
# We need these to programmatically change the friction properties for dynamics randomization.
WHEEL_COLLISION_PRIM_PATHS = [
    "Rigid_Bodies/Wheel__Knuckle__Front_Left/Collision",
    "Rigid_Bodies/Wheel__Knuckle__Front_Right/Collision",
    "Rigid_Bodies/Wheel__Upright__Rear_Left/Collision",
    "Rigid_Bodies/Wheel__Upright__Rear_Right/Collision",
]

ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v5/robots/autonav.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            "Wheel__Knuckle__Front_Left": 0.0,
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
        },
    ),
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            # effort_limit=1,
            velocity_limit=15.0,
            stiffness=0.0,
            damping=0.12,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=2000.0,
            damping=0.2,
        ),
    },
    collision_group=0
)

LIDAR_CFG = RayCasterCfg(
    prim_path="",
    mesh_prim_paths=["/World/Terrain/Structure/Geom"],
    max_distance=12.0,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0, 0),
        # vertical_fov_range=(-5.0, 5.0),
        horizontal_fov_range=(-135, 136),
        horizontal_res=10.0,
    ),
    attach_yaw_only=False,
    # debug_vis=True,
    debug_vis=False,
)
