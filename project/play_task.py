"""Play a trained RSL-RL checkpoint on any Isaac Lab / robot_lab task.

robot_lab's own play.py is written for its locomotion tasks: it unconditionally
edits env_cfg.scene.terrain and velocity-command curricula, so it crashes on a
task without them (e.g. RobotLab-Isaac-Reach-B2W-Z1-v0). This one only touches
what every task has.

    PLAY_SCRIPT=/workspace/project/play_task.py ./run_train.sh --play \
        --task <gym id> --checkpoint <path> --stream

Every few seconds it prints the metrics the task's commands report (for reach:
the hand's position error), so a headless run doubles as a quick evaluation.
Observation noise is switched off, as in any play.
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--real-time", action="store_true", help="run no faster than real time")
parser.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until closed)")
parser.add_argument("--report_every", type=float, default=3.0, help="seconds between metric prints")
parser.add_argument("--follow", action="store_true",
                    help="START with the camera locked to the robot. Press V in the viewport to "
                         "release it, C to snap it back. Without this the camera is free from the "
                         "start, placed next to the robot.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401  (registers the robot_lab tasks)

from free_camera import CameraToggle  # noqa: E402


def say(*a):
    print("PLAY|", *a, flush=True)


cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
if hasattr(cfg.observations, "policy"):
    cfg.observations.policy.enable_corruption = False
# No training disturbances in a viewer: the random shove (body velocity set to up
# to +-0.5 m/s every 10-15 s) and the random constant force/torque on the body
# (up to 30 N / 10 N m, held all episode). They make the policy robust in training;
# here they only add jolts and lean that are not the policy's own behaviour.
for _name in ("randomize_push_robot", "randomize_apply_external_force_torque"):
    if hasattr(cfg.events, _name):
        setattr(cfg.events, _name, None)
# The camera mode is set AFTER the env exists (see free_camera.py): a follow
# camera re-points itself every rendering step, which silently undoes any mouse
# movement, so it has to be switchable at runtime rather than fixed here.
cfg.viewer.eye = (2.5, 2.5, 1.8)
cfg.viewer.lookat = (0.0, 0.0, 0.3)

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
dt = u.step_dt
cam = CameraToggle(u, following=args.follow)
say(f"playing {args.checkpoint} on {args.task} with {args.num_envs} envs")

obs = env.get_observations()
step, every = 0, max(1, int(round(args.report_every / dt)))
while app.is_running():
    t0 = time.time()
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    step += 1
    if step % every == 0:
        for name in u.command_manager.active_terms:
            metrics = u.command_manager.get_term(name).metrics
            say(f"t={step * dt:6.1f}s  {name}: " + "  ".join(
                f"{k} {float(v.mean()):.4f}" for k, v in metrics.items()))
    if args.seconds and step * dt >= args.seconds:
        break
    if args.real_time:
        time.sleep(max(0.0, dt - (time.time() - t0)))

say("done")
cam.close()          # release the camera before teardown
env.close()
app.close()
