"""Write b2w_nav2_params.yaml: Nav2's Jazzy defaults (nav2_params_default.yaml, copied from
nav2_bringup) with the B2W's footprint, speeds and our sensor setup. Re-run after editing
(run_nav2.sh does it at every start).

    NAV2_SENSOR=os0|scan|both  NAV2_CONTROLLER=mppi|rpp  python3 nav2/make_params.py

Defaults: os0 (the robot's own 3D lidar, PointCloud2 /points) and rpp (Regulated Pure Pursuit to the
docking staging pose; the docking server does the last 2.1 m to the pick point). mppi = holonomic
MPPI (forward, backward, sideways): it crept at 0.02-0.1 m/s and wandered (9.8 m for a 1.5 m path) --
this robot's dead zones and the CPU's 2 s horizon suit it badly; 4/4 docked with rpp in 7-12 s.
"""

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
P = yaml.safe_load(open(os.path.join(HERE, "nav2_params_default.yaml")))
SENSOR = os.environ.get("NAV2_SENSOR", "os0")
CONTROLLER = os.environ.get("NAV2_CONTROLLER", "rpp")

# B2W + Z1 (stowed) from above, base_link at the body centre. Width: the hip housings are the widest
# part at table height (|y| 0.287 m) -> 0.30. At the pick point the base centre is 0.337 m from the
# table edge, so a wider footprint (0.32 + 0.02 padding, the first setup) puts the goal itself in
# collision. Length: wheels and head, +-0.55 m.
FOOTPRINT = "[[0.55, 0.30], [0.55, -0.30], [-0.55, -0.30], [-0.55, 0.30]]"
# ETH controller (fine-tuned): forward 0.1-1.0 m/s and turning at 90-105 %; sideways works with the
# fetch script's stall kick (the docking trims), which nav2_link applies to Nav2's commands too.
VX_MAX, VX_MIN, VY_MAX, WZ_MAX = 0.6, -0.35, 0.35, 0.8
if CONTROLLER == "rpp":
    VX_MAX, VX_MIN, VY_MAX = 0.8, -0.2, 0.0

for node, sect in P.items():  # every node runs on the simulator's clock
    if isinstance(sect, dict):
        inner = sect.get("ros__parameters")
        if inner is None and node in sect:  # costmaps nest one level deeper
            inner = sect[node]["ros__parameters"]
        if inner is not None:
            inner["use_sim_time"] = True

# the behaviour tree waits this long (ms) for an action server to accept a goal; the default 20 ms
# aborted a navigation when the planner was busy with the point clouds ("Timed out while waiting for
# action server to acknowledge goal request for compute_path_to_pose")
P["bt_navigator"]["ros__parameters"].update(default_server_timeout=200, wait_for_service_timeout=2000)

c = P["controller_server"]["ros__parameters"]
# PoseProgressChecker: turning in place counts as progress (the default SimpleProgressChecker aborted
# the final turn to the goal heading: "Failed to make progress").
c["progress_checker"] = {"plugin": "nav2_controller::PoseProgressChecker", "required_movement_radius": 0.3,
                         "required_movement_angle": 0.3, "movement_time_allowance": 15.0}
if CONTROLLER == "rpp":
    # E only roughly right; the scripted E -> P -> D drive corrects the rest. At 0.10 m, turning in place
    # crept the robot out of the goal circle and Nav2 looped.
    c["general_goal_checker"].update(xy_goal_tolerance=0.25, yaw_goal_tolerance=0.15)
    c["FollowPath"] = {
        "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
        "desired_linear_vel": 0.8, "lookahead_dist": 0.6, "min_lookahead_dist": 0.3, "max_lookahead_dist": 0.9,
        "lookahead_time": 1.5, "use_velocity_scaled_lookahead_dist": True, "transform_tolerance": 0.2,
        "use_rotate_to_heading": True, "rotate_to_heading_min_angle": 0.6, "rotate_to_heading_angular_vel": 0.8,
        "max_angular_accel": 1.5, "approach_velocity_scaling_dist": 0.6, "min_approach_linear_velocity": 0.12,
        "use_collision_detection": True, "max_allowed_time_to_collision_up_to_carrot": 1.0,
        "use_regulated_linear_velocity_scaling": True, "regulated_linear_scaling_min_radius": 0.9,
        "regulated_linear_scaling_min_speed": 0.15, "use_cost_regulated_linear_velocity_scaling": False,
        "allow_reversing": False, "max_robot_pose_search_dist": 10.0,
        "interpolate_curvature_after_goal": False, "use_fixed_curvature_lookahead": False,
    }
else:
    # NavigateToPose ends at the docking server's staging pose S (2.1 m before the pick point, on its
    # line, facing along the table); DockRobot does the precise part. Loose here: at 0.05 m / 0.07 rad
    # on the pick point itself, MPPI hovered 20 cm off the table line (the inflation cost pushes it
    # out) and overshot 1.4 m along the table.
    c["general_goal_checker"].update(xy_goal_tolerance=0.20, yaw_goal_tolerance=0.15)
    f = c["FollowPath"]  # MPPI from the defaults, made holonomic and light enough for this CPU
    f.update(motion_model="Omni", time_steps=40, model_dt=0.05, batch_size=1000,  # 2 s horizon (default 56 x 2000 ran at 12 Hz)
             vx_max=VX_MAX, vx_min=VX_MIN, vy_max=VY_MAX, wz_max=WZ_MAX,
             vx_std=0.2, vy_std=0.15, wz_std=0.4, ax_max=1.5, ax_min=-1.5, ay_max=1.5, ay_min=-1.5, az_max=2.0,
             prune_distance=1.5, transform_tolerance=0.2, visualize=True)
    f["critics"] = ["ConstraintCritic", "CostCritic", "GoalCritic", "GoalAngleCritic", "PathAlignCritic",
                    "PathFollowCritic", "PathAngleCritic", "PreferForwardCritic", "TwirlingCritic"]
    f["CostCritic"].update(consider_footprint=True, near_goal_distance=0.5)  # the pick point is 3-4 cm from the table
    f["PreferForwardCritic"].update(cost_weight=2.0)  # backing up is allowed, forward still preferred
    f["PathAngleCritic"].update(mode=1)  # follow the path forwards or backwards
    f["GoalAngleCritic"].update(cost_weight=5.0, threshold_to_consider=0.8)
    f["TwirlingCritic"] = {"enabled": True, "cost_power": 1, "cost_weight": 10.0}

# ---- what the costmaps see
scan_src = {  # the ray-cast scan is a 2D projection published in base_link at z = 0: keep every point
    "topic": "/scan", "data_type": "LaserScan", "clearing": True, "marking": True,
    "min_obstacle_height": -1.0, "max_obstacle_height": 2.0,
    "raytrace_min_range": 0.0, "raytrace_max_range": 7.0, "obstacle_min_range": 0.0, "obstacle_max_range": 6.0,
}
# the OS0 cloud (frame os0, base_link on the floor): obstacles are points 5 cm - 1.35 m above the floor
# (the robot is ~1.2 m tall); it cannot see below ~45 deg, so nothing within ~0.9 m at floor level --
# the static map keeps the table and shelves in both costmaps
points_src = {
    "topic": "/points", "data_type": "PointCloud2", "clearing": True, "marking": True,
    # 0.15 m: the RTX lidar's floor returns scatter +-0.1-0.2 m beyond ~4 m (elevation noise x range)
    "min_obstacle_height": 0.15, "max_obstacle_height": 1.35,
    "raytrace_min_range": 0.0, "raytrace_max_range": 6.0, "obstacle_min_range": 0.0, "obstacle_max_range": 5.0,
}
sources = {"scan": {"scan": scan_src}, "os0": {"points": points_src}, "both": {"scan": scan_src, "points": points_src}}[SENSOR]
for name in ("local_costmap", "global_costmap"):
    cm = P[name][name]["ros__parameters"]
    cm.pop("robot_radius", None)
    cm["footprint"] = FOOTPRINT
    cm["footprint_padding"] = 0.01
    cm["inflation_layer"].update(inflation_radius=0.75, cost_scaling_factor=4.0)  # >= circumscribed radius 0.63
    layer = {"plugin": "nav2_costmap_2d::ObstacleLayer", "enabled": True,
             "observation_sources": " ".join(sources), **{k: dict(v) for k, v in sources.items()}}
    if name == "local_costmap":
        cm.update(width=5, height=5)
        cm.pop("voxel_layer", None)
        if SENSOR == "scan":
            cm["plugins"] = ["obstacle_layer", "inflation_layer"]
            cm["obstacle_layer"] = layer
        else:  # 3D: a voxel layer clears along the real rays from the lidar, the static map keeps the blind zone
            cm["plugins"] = ["static_layer", "voxel_layer", "inflation_layer"]
            cm["static_layer"] = {"plugin": "nav2_costmap_2d::StaticLayer", "map_subscribe_transient_local": True}
            cm["voxel_layer"] = dict(layer, plugin="nav2_costmap_2d::VoxelLayer", publish_voxel_map=True, origin_z=0.0,
                                     z_resolution=0.1, z_voxels=14, max_obstacle_height=1.35, mark_threshold=0)
    else:
        cm["obstacle_layer"] = layer

# The static map already says what is free. With track_unknown_space the global costmap kept every
# cell the lidar had not seen yet as UNKNOWN (its blind zone round the robot, the shadow behind its
# body), the obstacle layer's unknown won over the map's free, and plans failed ("Failed to create
# plan") or detoured -- the 360 deg ray-cast scan never left such gaps.
P["global_costmap"]["global_costmap"]["ros__parameters"]["track_unknown_space"] = False
P["planner_server"]["ros__parameters"]["GridBased"].update(tolerance=0.25, use_astar=True, allow_unknown=True)

b = P["behavior_server"]["ros__parameters"]
b.update(max_rotational_vel=WZ_MAX, min_rotational_vel=0.3)

v = P["velocity_smoother"]["ros__parameters"]
v.update(max_velocity=[VX_MAX, VY_MAX, WZ_MAX], min_velocity=[VX_MIN, -VY_MAX, -WZ_MAX],
         max_accel=[1.5, 1.5, 2.0], max_decel=[-1.5, -1.5, -2.0])

cmon = P["collision_monitor"]["ros__parameters"]
cmon["base_frame_id"] = "base_link"
if SENSOR == "scan":
    cmon["scan"].update(min_height=-1.0, max_height=2.0)
else:
    cmon.pop("scan", None)
    cmon["observation_sources"] = ["points"]
    cmon["points"] = {"type": "pointcloud", "topic": "/points", "min_height": 0.15, "max_height": 1.35, "enabled": True}

# ---- the final approach to the pick point: Nav2's docking server, dock pose given by the caller
d = P["docking_server"]["ros__parameters"]
d.update(controller_frequency=20.0, initial_perception_timeout=2.0, dock_approach_timeout=60.0, max_retries=2,
         base_frame="base_link", fixed_frame="odom", dock_backwards=False, dock_prestaging_tolerance=0.5)
d["dock_plugins"] = ["pick_point"]
d.pop("simple_charging_dock", None)
d["pick_point"] = {
    "plugin": "opennav_docking::SimpleNonChargingDock",
    "docking_threshold": 0.04,          # m from the pick point: docked
    "staging_x_offset": -2.1,           # staging pose S: 2.1 m behind it along its heading (P 1.3 + entry 0.8)
    "staging_yaw_offset": 0.0,
    "use_external_detection_pose": False,  # the dock pose comes with the request (the sim knows the box)
    "use_stall_detection": False,
}
d["controller"].update(k_phi=3.0, k_delta=2.0, v_linear_min=0.12, v_linear_max=0.30, v_angular_max=0.6,
                       # collisions are ignored this close to the dock: the whole 2.1 m approach along the table.
                       # At 0.40 every approach failed ("Failed to get control", 905): the line runs inside the
                       # table's inscribed band (footprint half-width 0.30, the line 0.39 m from the edge). The
                       # line itself is fixed by the table's geometry; the collision monitor still watches the lidar.
                       use_collision_detection=True, dock_collision_threshold=2.3,
                       projection_time=3.0, simulation_step=0.1, transform_tolerance=0.2)

MAP = os.environ.get("NAV2_MAP", "warehouse")  # warehouse = one table; warehouse3 = the three-table scene
P["map_server"] = {"ros__parameters": {"use_sim_time": True, "yaml_filename": f"/maps/{MAP}.yaml",
                                       "topic_name": "map", "frame_id": "map"}}

out = os.path.join(HERE, "b2w_nav2_params.yaml")
with open(out, "w") as fh:
    fh.write(f"# GENERATED by make_params.py (sensor {SENSOR}, controller {CONTROLLER}, map {MAP}) from nav2_params_default.yaml"
             " -- edit make_params.py instead\n")
    yaml.safe_dump(P, fh, sort_keys=False, default_flow_style=None)
print("wrote", out, f"(sensor {SENSOR}, controller {CONTROLLER}, map {MAP})")
