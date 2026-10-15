"""Print what a policy actually sees and commands in an Isaac Lab task.

Builds the task with a single environment and reports, prefixed TASK|:
  * the robot's joints in articulation (simulator) order
  * observation group sizes and, for every joint-based observation term, the
    joints it reads IN THE ORDER IT READS THEM
  * the action size and the joints each action term drives

Use it to confirm a trained checkpoint still fits after changing the robot:
the checkpoint only knows sizes, so a same-size observation with its joints in
a different order loads without complaint and simply feeds the policy wrong
numbers.

    /isaac-sim/python.sh /workspace/training/inspect_task.py --task <gym id>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401  (registers the robot_lab tasks)


def say(*a):
    print("TASK|", *a, flush=True)


cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]

say(f"task: {args.task}")
say(f"robot: {robot.num_joints} joints, {robot.num_bodies} bodies")
say(f"articulation order: {robot.joint_names}")

om = u.observation_manager
for group, dim in om.group_obs_dim.items():
    say(f"obs group '{group}': {dim}")
for group in om.group_obs_dim:
    for name, term in zip(om._group_obs_term_names[group], om._group_obs_term_cfgs[group]):
        ac = term.params.get("asset_cfg") if term.params else None
        if ac is None or ac.joint_ids is None:
            continue
        ids = range(robot.num_joints) if isinstance(ac.joint_ids, slice) else ac.joint_ids
        say(f"  {group}.{name} reads joints: {[robot.joint_names[i] for i in ids]}")

am = u.action_manager
say(f"action dim: {am.total_action_dim}")
for name, term in am._terms.items():
    say(f"  action.{name} drives: {[robot.joint_names[i] for i in term._joint_ids]}")

env.close()
app.close()
