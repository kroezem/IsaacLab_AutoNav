# AutoNav Training Environment for Isaac Lab

Self-contained training task for indoor region-to-region navigation in NVIDIA Isaac Lab.  
This repo provides the RL environment and a reproducible trainer, not the hardware runtime.

## Features
- Shortest-path routing over a region graph parsed from the USD scene
- LiDAR-only observations with realistic noise, dropouts, and outliers
- PPO-ready 2D action space with a small control delay via action stacking
- Waypoint curriculum with success bonuses and clear failure modes
- Deterministic logging and code snapshots for reproducibility

## Requirements
- Isaac Sim and Isaac Lab installed and working
- Isaac Lab Python deps
- torch, gymnasium, networkx, pxr bindings


## Quick start
Train with 256 parallel envs on the ELW_v0 terrain:
```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/train.py \
  --num_envs 256 \
  --terrain ELW_v0 \
  --tag test0 \
````

Target specific destination regions and force a random outgoing paths from each start:

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/train.py \
  --num_envs 256 \
  --terrain ELW_v0 \
  --destination r_26 r_41 \
  --force_path \
  --tag focus \
  --headless
```

Resume from a checkpoint:

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/autonav_v8/train.py \
  --checkpoint path/to/checkpoint.pt \
  --headless
```

All runs write:

* logs to `.../logs/<terrain>[_force-path][_destinations][_tag]_YYYYMMDD_HHMMSS/`
* code and config snapshots to `.../snapshots/<same_run_name>/`

## Environment API summary

### Action space

* 2 values: `[throttle, steering]`
* Throttle clamped to \[0, 1], steering to \[-1, 1]
* One-step control latency via a 4-frame action buffer; last entry is applied

### Observation vector (size 36)

Concatenation:

* 2: current action `[throttle, steering]`
* 28: LiDAR beams (range in meters transformed as `1 / (0.1 + d)`)
* 1: gyro z rate
* 1: body-frame x proper acceleration
* 2: heading `[sin(yaw), cos(yaw)]` with sim-to-sensor +90 deg fix
* 2: coarse goal direction in cardinal plane `[east-west, north-south]` with dropout

LiDAR noise model per step:

* per-episode bias and scale, Gaussian noise `sigma = 0.03*d + 0.03`
* 3% dropouts set to max range, 0.5% outliers, clamped to \[0, 12 m]

### Rewards

Positive terms

* progress toward next waypoint
* forward velocity clamp \[0, 5]
* heading alignment to waypoint direction
* mean LiDAR distance (spaciousness)
* +20 on waypoint reach
* +100 on final destination reach

Penalties

* proximity penalty when nearest LiDAR < 1.0 m
* spin penalty from body yaw rate squared
* steering jerk between consecutive actions
* stall penalty when forward speed < 0.25 m/s
* -50 on collision threshold breach
* -5 on task\_failed flag

### Termination and truncation

* Done when stalled for 300 steps since last waypoint, collision proximity below threshold, or `task_failed`
* Truncated when final destination is reached

## Routing and regions

* Regions are read from USD prims matching `/World/Terrain/Regions/r_*`
* Each region is a sphere: center from the prim transform, radius is half of the smallest scale axis
* Undirected edges are created between overlapping regions with edge weight equal to center distance
* Shortest path is computed by `networkx.shortest_path` with `weight="weight"`
* Cache of the parsed network is written to `<terrain_folder>/region_network.json`
* Start and destination:

  * By default both are randomized per env
  * CLI `--destination` limits possible goal nodes
  * Config fields `start` and `destination` exist in `AutoNavEnvCfg` if you prefer fixed nodes
  * `--force_path` restricts the first move to one randomly chosen neighbor to create simpler paths

## Important config knobs

Defined in `AutoNavEnvCfg`:

* `dt = 1/60`, `decimation = 2` so RL step is 1/30 s
* `collision_kill_dist = 0.2` meters
* `obs_noise = True`
* `terrain_usd_path` is auto-built from `--terrain` name
* `action_space = 2`, `observation_space = 36`

## Reproducibility and logging

* Trainer builds a run name from terrain, optional `force-path`, optional destination list, and `--tag`
* Snapshots include:

  * `autonav_env.py`, `train.py`, `robots/autonav.py`
  * `env.yaml` and `agent.yaml`
* Pass `--seed` to set both env and agent seeds

## Tips

* If you add a new USD, put it under `terrains/<name>/<name>_<version>.usd` and call with `--terrain <name>`
* To lock starting point, set `env_cfg.start = "r_xx"` in a task config or code
* If LiDAR range or density changes, update `LIDAR_CFG` in `robots/autonav.py` and the observation size if needed



If you want, I can wire in a minimal task config example to set `env_cfg.start`, or add a short troubleshooting section for common Isaac Lab issues.
::contentReference[oaicite:0]{index=0}
```
