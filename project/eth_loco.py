"""ETH's SRU B2W locomotion policy as the base controller of our Isaac Lab tasks.

The policy is policy_force_new.onnx from leggedrobotics/sru-robot-deployment (b2w_sim), the
one deployed on the real B2W (copy, SHA256 verified, in Isaac/eth/; reference adapter in
leggedrobotics/sru-robot-deployment). Contract, from b2w_controllers.cpp:
  obs (60)  base lin vel (3), base ang vel (3), projected gravity (3), [vx, vy, wz] (3),
            joint pos - default wrapped to [-2pi, 2pi) (16), joint vel (16), last action (16)
            -- joints grouped hip/thigh/calf/foot x FL/FR/RL/RR, values unscaled
  LSTM      hidden and cell (1, 1, 256), carried between steps, zeroed on reset
  actions   leg target = default + 0.5 a[0:12], wheel velocity = 5 a[12:16]; 50 Hz
and as trained it needs ETH's drives (implicit PD 100 / 3.5, calf effort 320, wheels
damping 3.0) and stance (thigh 0.4, calf -1.3: the body stands ~0.75 m, 15 cm above
model_2600's). patch_cfg() applies all of that to a task config; the env's own action
terms then carry the ETH mapping exactly (scale 0.5 / 5.0, offset = the ETH stance).

Import after the AppLauncher. A/B against model_2600: project/eth_compare.py.
"""

import math

import numpy as np
import torch

from isaaclab.actuators import ImplicitActuatorCfg

import os

ONNX = "/workspace/eth/policy_force_new.onnx"
# Default (9 Oct 2026): ETH's policy fine-tuned for a wider rear stance (training/train_eth_ft.py,
# v5 iteration 600; PHASE5_ETH_STANCE_FINETUNE.md) -- rear track 0.31 m driving straight vs
# ETH's 0.20, same velocity tracking, fetch 24/24; weaker sideways from a standstill.
# ETH_POLICY overrides it: ETH_POLICY=/workspace/eth/policy_force_new.onnx for the original.
FINETUNED = "/workspace/eth/policy_stance_v5_600.pt"
POLICY = os.environ.get("ETH_POLICY", FINETUNED)
JOINTS = [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR")]
DEFAULT = np.asarray([0.0] * 4 + [0.4] * 4 + [-1.3] * 4 + [0.0] * 4, dtype=np.float32)
STANCE = {"hip": 0.0, "thigh": 0.4, "calf": -1.3, "foot": 0.0}
# ETH's "crouched" deployment profile (b2w_controllers.yaml environment.profiles.crouched):
# thigh 0.6, calf -1.6 -- the body ~6 cm lower. Its feet entry (-0.15) is left out: in their
# controller it also biases every wheel velocity target by -0.15 rad/s.
CROUCH = DEFAULT.copy()
CROUCH[4:8], CROUCH[8:12] = 0.6, -1.6
SPAWN_Z = 0.75
# command floors for scripted controllers: ETH follows 0.1 m/s forward (104 %) and 0.2 m/s
# sideways (68-89 %) from rest, where model_2600 needs 0.25 m/s and ignores 0.2 sideways
MIN_SPEED, MIN_TURN = 0.12, 0.25


def patch_cfg(cfg):
    """ETH drives, stance and action scaling on a robot_lab B2W(+Z1) task config, in place."""
    r = cfg.scene.robot
    acts = dict(r.actuators)
    for k in ("hip", "thigh", "calf", "wheel"):
        assert k in acts, f"actuator group {k} missing: {list(acts)}"
    acts["hip"] = ImplicitActuatorCfg(joint_names_expr=[".*_hip_joint"], effort_limit_sim=200.0, velocity_limit_sim=23.0,
                                      stiffness=100.0, damping=3.5, friction=0.0)
    acts["thigh"] = ImplicitActuatorCfg(joint_names_expr=[".*_thigh_joint"], effort_limit_sim=200.0,
                                        velocity_limit_sim=23.0, stiffness=100.0, damping=3.5, friction=0.0)
    acts["calf"] = ImplicitActuatorCfg(joint_names_expr=[".*_calf_joint"], effort_limit_sim=320.0,
                                       velocity_limit_sim=14.0, stiffness=100.0, damping=3.5, friction=0.0)
    acts["wheel"] = ImplicitActuatorCfg(joint_names_expr=[".*_foot_joint"], effort_limit_sim=20.0,
                                        velocity_limit_sim=50.0, stiffness=0.0, damping=3.0, friction=0.0)
    jp = {k: v for k, v in r.init_state.joint_pos.items()
          if not any(k.endswith(f"_{part}_joint") for part in STANCE)}
    jp.update({f".*_{part}_joint": v for part, v in STANCE.items()})
    cfg.scene.robot = r.replace(actuators=acts, soft_joint_pos_limit_factor=0.95,
                                init_state=r.init_state.replace(pos=(r.init_state.pos[0], r.init_state.pos[1], SPAWN_Z),
                                                                joint_pos=jp))
    cfg.actions.joint_pos.scale = 0.5
    cfg.actions.joint_vel.scale = 5.0


class Driver:
    """Runs the policy on one robot and writes its 16 actions into the env's action vector."""

    def __init__(self, env, onnx_path=None):
        path = onnx_path or POLICY
        self.torch_net = None
        if path.endswith(".onnx"):
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            ins = {i.name: i.shape for i in self.sess.get_inputs()}
            assert ins == {"obs": [1, 60], "h_in": [1, 1, 256], "c_in": [1, 1, 256]}, ins
        else:  # an rsl_rl ActorCriticRecurrent checkpoint from train_eth_ft.py: the actor only
            sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
            lstm = torch.nn.LSTM(60, 256, 1)
            lstm.load_state_dict({k.replace("memory_a.rnn.", ""): v for k, v in sd.items() if k.startswith("memory_a.rnn.")})
            dims, layers = [256, 256, 256, 128, 64, 16], []
            for i in range(5):
                layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
                if i < 4:
                    layers.append(torch.nn.ELU())
            mlp = torch.nn.Sequential(*layers)
            mlp.load_state_dict({k.replace("actor.", ""): v for k, v in sd.items() if k.startswith("actor.")})
            self.torch_net = (lstm.eval(), mlp.eval())
        print(f"[eth_loco] policy {path}", flush=True)
        self.env = env
        self.robot = env.scene["robot"]
        am = env.action_manager
        slices, start = {}, 0
        for name, width in zip(am.active_terms, am.action_term_dim):
            slices[name] = slice(start, start + width)
            start += width
        legs, wheels = am.get_term("joint_pos")._joint_names, am.get_term("joint_vel")._joint_names
        self.ids = self.robot.find_joints(JOINTS, preserve_order=True)[0]
        self.leg_term = am.get_term("joint_pos")
        self.leg_pos = [legs.index(n) for n in JOINTS[:12]]  # ETH leg index -> position in the leg action term
        self.default = DEFAULT.copy()
        self.crouch = 0.0
        self.to_act = [slices["joint_pos"].start + legs.index(n) if n in legs else slices["joint_vel"].start + wheels.index(n)
                       for n in JOINTS]
        dflt = self.robot.data.default_joint_pos[0, self.ids].cpu().numpy()
        assert np.allclose(dflt, DEFAULT, atol=1e-4), f"not the ETH stance (patch_cfg?): {dflt}"
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 256), np.float32)
        self.c = np.zeros((1, 1, 256), np.float32)
        self.last = np.zeros(16, np.float32)

    def obs(self, cmd):
        d = self.robot.data
        q = d.joint_pos[0, self.ids].cpu().numpy() - self.default
        q = np.mod(q + 2 * np.pi, 4 * np.pi) - 2 * np.pi
        return np.concatenate([d.root_lin_vel_b[0].cpu().numpy(), d.root_ang_vel_b[0].cpu().numpy(),
                               d.projected_gravity_b[0].cpu().numpy(), np.asarray(cmd, np.float32), q,
                               d.joint_vel[0, self.ids].cpu().numpy(), self.last]).astype(np.float32)

    def set_rear_hip_splay(self, rad):
        """(experimental) Shift the rear hips' reference outward by `rad` -- in the policy's
        joint offsets AND the leg action term's offset, like a stance profile. The policy
        commands RL ~ -0.3 / RR ~ +0.3 rad (inward) all the time, on our model and on ETH's
        own b2w_rsl.usd alike (probe_track.py): the rear track shrinks to 0.1-0.2 m."""
        self.splay = float(rad)
        self.set_crouch(self.crouch)

    def set_crouch(self, frac):
        """Blend the stance between ETH's default (0) and crouched (1) profiles: the policy's
        joint offsets AND the leg action term's offset move together, as a profile switch does
        in ETH's controller (there at start-up; here ramped by the caller)."""
        self.crouch = float(frac)
        self.default = DEFAULT + self.crouch * (CROUCH - DEFAULT)
        self.default[2] += getattr(self, "splay", 0.0)   # RL hip: + is outward on the left
        self.default[3] -= getattr(self, "splay", 0.0)   # RR hip: - is outward on the right
        off = self.leg_term._offset
        for i, pos in enumerate(self.leg_pos):
            off[:, pos] = float(self.default[i])

    def fill(self, a, cmd):
        """One policy step for base command cmd = (vx, vy, wz); writes into action tensor a (1, N).
        Resets the LSTM by itself at the first step of an episode."""
        if int(self.env.episode_length_buf[0]) == 0:
            self.reset()
            if self.crouch:
                self.set_crouch(0.0)
        if self.torch_net is None:
            out, self.h, self.c = self.sess.run(["actions", "h_out", "c_out"],
                                                {"obs": self.obs(cmd)[None], "h_in": self.h, "c_in": self.c})
        else:
            lstm, mlp = self.torch_net
            with torch.no_grad():
                y, (h, c) = lstm(torch.from_numpy(self.obs(cmd))[None, None], (torch.from_numpy(self.h), torch.from_numpy(self.c)))
                out = mlp(y[0]).numpy()
            self.h, self.c = h.numpy(), c.numpy()
        self.last = out[0].astype(np.float32)
        a[0, self.to_act] = torch.as_tensor(self.last, device=a.device)
        return a


def tilt_deg(robot):
    g = robot.data.projected_gravity_b[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(-g[2])))))
