"""Fine-tune ETH's SRU B2W locomotion policy for a normal stance width (Option B).

Task RobotLab-Isaac-EthFinetune-Flat-Unitree-B2W-Z1-v0 (eth_finetune_env_cfg.py) reproduces
ETH's controller contract exactly; the actor (LSTM 256 -> 256-256-128-64 -> 16) starts from
ETH's deployment weights (Isaac/eth/policy_force_new_rslrl.npz, converted from the ONNX and
verified to 1e-6 against onnxruntime). The critic is new, so it is trained alone first
(--warmup iterations, actor frozen, fixed learning rate) before the whole policy is.

    ./run_script.sh train_eth_ft.py --headless --num_envs 2048 --max_iterations 1500
    ./run_script.sh train_eth_ft.py --headless --eval_only [--load <model_XXX.pt>]   # ETH itself without --load

--eval_only: deterministic actions, commands resampled every 5 s, and reports per 1000
steps the front / rear wheel track, velocity tracking and falls (0.9-tracked quantities).
"""

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotLab-Isaac-EthFinetune-Flat-Unitree-B2W-Z1-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--warmup", type=int, default=100, help="critic-only iterations first")
parser.add_argument("--init", default="/workspace/eth/policy_force_new_rslrl.npz")
parser.add_argument("--load", default="", help="an rsl_rl checkpoint to start from / evaluate instead of ETH's weights")
parser.add_argument("--tag", default="")
parser.add_argument("--desired_kl", type=float, default=0.0, help="override PPO's adaptive-LR KL target")
parser.add_argument("--rewarm", action="store_true", help="with --load: critic warm-up too (the rewards changed)")
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--eval_steps", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_cmd", default="", help="eval_only: fixed command(s) vx,vy,wz for every env, several separated by ; (default: resampled)")
parser.add_argument("--no_rand", action="store_true", help="eval_only: no pushes / randomised masses, gains or friction")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w import (  # noqa: E402
    eth_finetune_env_cfg as EF,
)


def say(*a):
    print("ETHFT|", *a, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg.max_iterations = args.max_iterations
agent_cfg.seed = args.seed
if args.desired_kl:
    agent_cfg.algorithm.desired_kl = args.desired_kl
if args.eval_only:
    env_cfg.commands.base_velocity.resampling_time_range = (5.0, 5.0)
    env_cfg.observations.policy.enable_corruption = False
    if args.eval_cmd:
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    if args.no_rand:
        for name in list(vars(env_cfg.events)):
            if name.startswith(("randomize_", "push_")) and name != "randomize_reset_base":
                setattr(env_cfg.events, name, None)
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
u = env.unwrapped

stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("/workspace/repos/robot_lab/logs/rsl_rl", agent_cfg.experiment_name,
                       f"{stamp}{'_' + args.tag if args.tag else ''}") if not args.eval_only else None
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
pol = runner.alg.policy
if args.load:
    runner.load(args.load)
    say(f"loaded {args.load}")
else:
    W = np.load(args.init)
    missing, unexpected = torch.nn.Module.load_state_dict(pol, {k: torch.tensor(v) for k, v in W.items()}, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith(("memory_c", "critic", "std", "log_std", "critic_obs_normalizer", "actor_obs_normalizer"))
               for k in missing), missing
    say(f"ETH weights loaded into the actor ({len(W)} tensors); new: {sorted({k.split('.')[0] for k in missing})}")


def evaluate(steps, cmd=""):
    """Deterministic rollout: wheel tracks, velocity tracking error, falls."""
    policy = runner.get_inference_policy(device=u.device)
    robot = u.scene["robot"]
    feet = robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)[0]
    cfg = type("C", (), {"name": "robot", "body_ids": feet})
    fixed = torch.tensor([float(v) for v in cmd.split(",")], device=u.device) if cmd else None
    # several fixed commands run back to back (the env cannot be reset from here); the first
    # 100 steps of each are not scored
    obs = env.get_observations()
    vel_term = u.command_manager.get_term("base_velocity")
    tr, vm, ev, ew, falls, upright, tr_st, tr_st_min = [], [], [], [], 0, [], [], []
    for k in range(steps):
        with torch.inference_mode():
            if fixed is not None:
                vel_term.vel_command_b[:] = fixed
                obs = env.get_observations()
            obs, _, dones, _ = env.step(policy(obs))
            if k >= 100:
                t = EF.wheel_tracks(u, cfg)
                tr.append(t.mean(0).cpu().numpy())
                c = u.command_manager.get_command("base_velocity")
                st = (c[:, 1].abs() < 0.05) & (c[:, 2].abs() < 0.1)  # straight driving / standing
                if st.any():
                    tr_st.append(t[st].mean(0).cpu().numpy())
                    tr_st_min.append(float(t[st, 1].min()))
                vm.append(torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], 1).mean(0).cpu().numpy())
                ev.append(float(torch.norm(robot.data.root_lin_vel_b[:, :2] - c[:, :2], dim=1).mean()))
                ew.append(float((robot.data.root_ang_vel_b[:, 2] - c[:, 2]).abs().mean()))
                g = robot.data.projected_gravity_b[:, 2]
                upright.append(float((g < -0.7).float().mean()))
                falls += int(dones.sum())
                if (k + 1) % 1000 == 0:
                    trk = np.mean(tr[-900:], 0)
                    say(f"step {k + 1}: track front {trk[0]:.3f} m rear {trk[1]:.3f} m | |v err| {np.mean(ev[-900:]):.3f} m/s "
                        f"|w err| {np.mean(ew[-900:]):.3f} rad/s | upright {np.mean(upright[-900:]) * 100:.1f}% | "
                        f"episode ends {falls}")
    trk = np.array(tr)
    rear_min = float(np.percentile(trk[:, 1], 5))
    say(f"EVAL track front {trk[:, 0].mean():.3f} rear {trk[:, 1].mean():.3f} (5th pct rear {rear_min:.3f}) m | "
        f"|v err| {np.mean(ev):.3f} |w err| {np.mean(ew):.3f} | upright {np.mean(upright) * 100:.1f}% | ends {falls}"
        + (f" | STRAIGHT track front {np.mean(tr_st, 0)[0]:.3f} rear {np.mean(tr_st, 0)[1]:.3f} "
           f"(worst env rear, mean over steps {np.mean(tr_st_min):.3f})" if tr_st and not cmd else "")
        + (f" | cmd {cmd} measured vx,vy,wz {np.round(np.mean(vm, 0), 3).tolist()}" if fixed is not None else ""))


if args.eval_only:
    for c in (args.eval_cmd.split(";") if args.eval_cmd else [""]):
        evaluate(args.eval_steps, c)
else:
    os.makedirs(log_dir, exist_ok=True)
    say(f"logging to {log_dir}")
    actor_params = list(pol.memory_a.parameters()) + list(pol.actor.parameters()) + \
        [p for n, p in pol.named_parameters() if n in ("std", "log_std")]
    if args.warmup > 0 and (not args.load or args.rewarm):
        for p in actor_params:
            p.requires_grad_(False)
        schedule, lr = runner.alg.schedule, runner.alg.learning_rate
        runner.alg.schedule = "fixed"  # KL is 0 with the actor frozen: adaptive would ramp the rate up
        say(f"critic warm-up: {args.warmup} iterations, actor frozen")
        runner.learn(num_learning_iterations=args.warmup, init_at_random_ep_len=True)
        for p in actor_params:
            p.requires_grad_(True)
        runner.alg.schedule, runner.alg.learning_rate = schedule, lr
        for g in runner.alg.optimizer.param_groups:
            g["lr"] = lr
        say("warm-up done: training the whole policy")
    runner.learn(num_learning_iterations=args.max_iterations - (args.warmup if (not args.load or args.rewarm) else 0),
                 init_at_random_ep_len=args.warmup <= 0)
    say(f"done -> {log_dir}")
env.close()
app.close()
