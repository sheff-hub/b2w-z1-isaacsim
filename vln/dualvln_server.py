"""DualVLN (InternVLA-N1) inference server for the Isaac Sim robot.

Adapted from InternNav scripts/realworld/http_internvla_server.py (InternRobotics). Changes:
  * the instruction comes from each request (upstream hard-codes one), as do the
    camera intrinsics, so the simulator decides both;
  * the model loads with PyTorch SDPA attention (no flash-attn build needed) and,
    with --quant 8bit, the language model's 28 transformer layers in 8-bit
    (bitsandbytes). The vision tower, the trajectory diffusion model (System 1),
    the RGB-D encoder and the output head stay bf16;
  * plan_step_gap is set (the upstream agent reads it; its server never set it);
  * System 2 runs on its own thread and System 1 answers every request from its latest
    goal (class DualSystem); --s2_blocking restores upstream's wait-for-System-2.

POST /eval_dual   files: image (RGB PNG), depth (16-bit PNG, 0.1 mm units),
                         optional image_down, depth_down: the same camera pitched 30 deg down,
                         used when the model asks to look down (as in its Habitat training,
                         where LOOKDOWN tilts the camera 2 x 15 deg); without them the level
                         image is reused, as the upstream real-world server does
                  form json: {"reset": bool, "instruction": str, "intrinsic": 4x4 list, "rid": int,
                             "s2_blocking": bool (wait for System 2 before answering),
                             "session": str (one per client; default "default")}
  -> {"trajectory": [[x, y, ...], ...]}        robot frame at THIS request's image, m
  or {"discrete_action": [...]}                0 STOP, 1 fwd, 2 left, 3 right (about the s2_rid image)
  or {"wait": true}                            no System 2 answer yet (start of an episode)
  plus, when System 2 has just finished: "s2_rid" (the request whose image it planned on),
  "s2_s" (its time), "looked_down", "pixel_goal": [row, col] in that request's look-down image.

Run inside the dualvln-server image (see Isaac/vln/run_dualvln_server.sh).
"""

import argparse
import json
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import BitsAndBytesConfig

import importlib
import os
import sys
import threading
import types


def _bare_package(name):
    """Register an InternNav sub-package WITHOUT running its __init__: internnav/agent's
    imports every agent, including old ones needing gym/habitat that DualVLN does not use."""
    root = importlib.import_module("internnav")
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.join(os.path.dirname(root.__file__), *name.split(".")[1:])]
    sys.modules[name] = pkg


_bare_package("internnav.agent")
_bare_package("internnav.model.encoder")  # its __init__ imports a LongCLIP encoder that is not in the repo
agent_mod = importlib.import_module("internnav.agent.internvla_n1_agent_realworld")
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM  # noqa: E402
from internnav.model.basemodel.internvla_n1 import nextdit_traj  # noqa: E402

# System 1's DiT turns on gradient checkpointing (a training-time memory saver) through
# the old diffusers hook signature; diffusers in this image calls it with new keywords.
# Inference never uses it, so make the call a no-op.
nextdit_traj.LuminaNextDiT2DModel.enable_gradient_checkpointing = lambda self, *a, **k: None


class _LuminaFeedForward23(nextdit_traj.LuminaFeedForward):
    """System 1's DiT feed-forward, sized as the checkpoint was trained: older diffusers
    scaled the hidden size by 2/3 first (SwiGLU convention), 4*384 -> 1024; diffusers
    0.33.1 (InternNav's own pin) does not, building 1536 and failing to load
    traj_dit...feed_forward.linear_{1,2,3} ([1024, 384] in the checkpoint)."""

    def __init__(self, dim, inner_dim, multiple_of=256, ffn_dim_multiplier=None):
        super().__init__(dim, int(2 * inner_dim / 3), multiple_of, ffn_dim_multiplier)


nextdit_traj.LuminaFeedForward = _LuminaFeedForward23

p = argparse.ArgumentParser()
p.add_argument("--model_path", default="InternRobotics/InternVLA-N1-DualVLN")
p.add_argument("--device", default="cuda:0")
p.add_argument("--quant", choices=("8bit", "none"), default="8bit")
p.add_argument("--resize_w", type=int, default=384)
p.add_argument("--resize_h", type=int, default=384)
p.add_argument("--num_history", type=int, default=8)
p.add_argument("--plan_step_gap", type=int, default=4, help="System 2 re-plans every N requests (System 1 runs every one)")
p.add_argument("--port", type=int, default=5801)
p.add_argument("--s2_blocking", action="store_true",
               help="default for requests without \"s2_blocking\": answer only after System 2 finishes "
                    "(upstream behaviour) instead of in the background")
args = p.parse_args()

# kept in bf16 under 8-bit: everything but the language model's transformer layers
SKIP = ["lm_head", "visual", "traj_dit", "rgb_model", "rgb_resampler", "memory_encoder",
        "action_encoder", "action_decoder", "cond_projector", "latent_queries"]
_orig_from_pretrained = InternVLAN1ForCausalLM.from_pretrained.__func__


def _from_pretrained(cls, path, *a, **kw):
    kw["attn_implementation"] = "sdpa"
    if args.quant == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP)
    model = _orig_from_pretrained(cls, path, *a, **kw)
    if args.quant != "none":
        model.to = lambda *x, **y: model  # the agent calls .to(device); quantized models refuse it
    return model


InternVLAN1ForCausalLM.from_pretrained = classmethod(_from_pretrained)

t0 = time.time()
agent = agent_mod.InternVLAN1AsyncAgent(args)
agent.reset()
print(f"[dualvln] loaded {args.model_path} ({args.quant}) in {time.time() - t0:.0f} s, "
      f"GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)



class DualSystem:
    """DualVLN with the two systems decoupled, as the model is meant to run: System 2
    (the VLM: pixel goal, or turns / STOP; ~2-3 s) works on a background thread while
    System 1 (the trajectory DiT; ~0.1 s) answers every request from the latest goal.
    Upstream's server runs both in the request, so the robot gets no new path for the
    2-3 s System 2 takes and runs off the end of its old one.

    System 1 compares the pixel-goal frame with the CURRENT frame of the same camera: a
    goal picked in the look-down view is tracked in the look-down view (as the Habitat
    evaluator does), else System 1 sees no match and returns ~0 m paths.

    The agent's history (rgb_list / episode_idx) is only touched under `lock`, and not
    while System 2 runs: frames that arrive meanwhile wait in `pending` and are appended
    after it, in order (step_s2 reads the history when it starts)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.reset()

    def reset(self):
        if self.thread is not None:
            self.thread.join()  # never reset under a running System 2
        agent.reset()
        torch.cuda.empty_cache()  # hand cached blocks back: the GPU is shared, and Isaac renders on it too
        self.pending, self.goal_down, self.fresh, self.s2_rid, self.s2_s = [], False, None, None, None

    def _s2(self, rid, frames, instruction, intrinsic):
        (rgb, depth), (rgb_down, depth_down) = frames
        t0 = time.time()
        try:
            with torch.no_grad():
                action, latent, pixel = agent.step_s2(rgb, depth, np.eye(4), instruction, intrinsic, False)
                look_down = action == [5]  # the model asked to look down: ask again with that view
                if look_down:
                    action, latent, pixel = agent.step_s2(rgb_down, depth_down, np.eye(4), instruction, intrinsic, True)
        except Exception as e:  # noqa: BLE001
            print(f"[dualvln] System 2 failed: {e!r}", flush=True)
            action, latent, pixel, look_down = None, None, None, False
        with self.lock:
            agent.last_s2_idx = agent.episode_idx
            for img in self.pending:  # frames that came in meanwhile, in order
                agent.rgb_list.append(img)
                agent.episode_idx += 1
            self.pending = []
            if action is not None or latent is not None:
                agent.output_action, agent.output_latent, agent.output_pixel = action, latent, pixel
                if latent is not None:
                    src = (rgb_down, depth_down) if look_down else (rgb, depth)
                    agent.pixel_goal_rgb, agent.pixel_goal_depth = src[0].copy(), src[1].copy()
                    self.goal_down = look_down
            self.fresh = {"s2_rid": rid, "looked_down": look_down, "s2_s": round(time.time() - t0, 2)}
            if pixel is not None:
                self.fresh["pixel_goal"] = [int(v) for v in pixel]
            self.thread = None
        print(f"[dualvln] S2 rid {rid}: {action if action is not None else 'pixel ' + str(pixel)}"
              f"{' (looked down)' if look_down else ''} in {time.time() - t0:.2f} s", flush=True)

    def step(self, rid, frames, instruction, intrinsic, blocking):
        (rgb, depth), (rgb_down, depth_down) = frames
        launch = None
        with self.lock:
            busy = self.thread is not None
            due = (agent.episode_idx - agent.last_s2_idx > agent.PLAN_STEP_GAP) or \
                  (agent.output_action is None and agent.output_latent is None)
            if busy:
                self.pending.append(Image.fromarray(rgb).convert("RGB").resize((agent.resize_w, agent.resize_h)))
            elif due:  # step_s2 appends this frame to the history itself
                launch = threading.Thread(target=self._s2, args=(rid, frames, instruction, intrinsic), daemon=True)
                self.thread = launch
            else:
                agent.rgb_list.append(Image.fromarray(rgb).convert("RGB").resize((agent.resize_w, agent.resize_h)))
                agent.episode_idx += 1
        if launch is not None:
            launch.start()
            if blocking:
                launch.join()
        reply = {}
        with self.lock:
            if self.fresh is not None:
                reply.update(self.fresh)
                self.fresh = None
            if agent.output_action is not None:  # turns / STOP: handed out once
                reply["discrete_action"] = [int(a) for a in agent.output_action]
                agent.output_action = None
                return reply
            if agent.output_latent is None:
                reply["wait"] = True  # the first System 2 answer is not there yet
                return reply
            latent, goal_rgb, goal_depth, down = (agent.output_latent, agent.pixel_goal_rgb,
                                                  agent.pixel_goal_depth, self.goal_down)
        cur_rgb, cur_depth = (rgb_down, depth_down) if down else (rgb, depth)

        def prep(a, scale):
            return torch.from_numpy(np.array(Image.fromarray(a).resize((224, 224))) / scale)

        with torch.no_grad():  # the agent runs System 1 outside one; its output then can't go to numpy
            rgbs = torch.stack([prep(goal_rgb, 255), prep(cur_rgb, 255)]).unsqueeze(0).to(agent.device)
            depths = torch.stack([prep(goal_depth, 1), prep(cur_depth, 1)]).unsqueeze(0).unsqueeze(-1).to(agent.device)
            traj = agent_mod.traj_to_actions(agent.step_s1(latent, rgbs, depths), use_discrate_action=False)
        reply["trajectory"] = np.asarray(traj).tolist()
        return reply


# Per-client sessions. The agent keeps ONE episode's memory (image history, conversation, the
# current goal); several simulations sharing this server (parallel data recording) reset and
# mixed each other's history. Each request names its session; its state is swapped in for the
# request. Safe because requests are served one at a time and, with s2_blocking (the client
# default), System 2 runs inside the request.
AGENT_STATE = ["rgb_list", "depth_list", "pose_list", "episode_idx", "conversation_history", "llm_output",
               "past_key_values", "last_s2_idx", "output_action", "output_latent", "output_pixel",
               "pixel_goal_rgb", "pixel_goal_depth", "save_dir", "input_images"]
DUAL_STATE = ["pending", "goal_down", "fresh", "s2_rid", "s2_s"]
sessions = {"current": None, "store": {}}


def activate(sid):
    cur = sessions["current"]
    if cur == sid:
        return
    if dual.thread is not None:
        dual.thread.join()  # never swap under a running System 2
    if cur is not None:
        sessions["store"][cur] = ({k: getattr(agent, k, None) for k in AGENT_STATE},
                                  {k: getattr(dual, k) for k in DUAL_STATE}, dict(state))
    if sid in sessions["store"]:
        a_st, d_st, s_st = sessions["store"].pop(sid)
        for k, v in a_st.items():
            setattr(agent, k, v)
        for k, v in d_st.items():
            setattr(dual, k, v)
        state.clear()
        state.update(s_st)
    else:
        dual.reset()
        state.clear()
        state["n"] = 0
    sessions["current"] = sid


dual = DualSystem()
app = Flask(__name__)
state = {"n": 0}


@app.route("/eval_dual", methods=["POST"])
def eval_dual():
    t_req = time.time()

    def load(key):
        rgb_ = np.asarray(Image.open(request.files[key].stream).convert("RGB"))
        depth_ = np.asarray(Image.open(request.files[key.replace("image", "depth")].stream).convert("I"))
        return rgb_, depth_.astype(np.float32) / 10000.0

    frames = load("image"), (load("image_down") if "image_down" in request.files else None)
    frames = (frames[0], frames[1] or frames[0])
    data = json.loads(request.form["json"])
    activate(str(data.get("session", "default")))
    intrinsic = np.array(data["intrinsic"], dtype=np.float64)
    instruction = data["instruction"]
    if data.get("reset"):
        dual.reset()
        state["n"] = 0
        print(f"[dualvln] reset ({data.get('session', 'default')}), instruction: {instruction!r}, "
              f"{len(sessions['store']) + 1} session(s)", flush=True)
    state["n"] += 1
    reply = dual.step(data.get("rid", state["n"]), frames, instruction, intrinsic,
                      blocking=data.get("s2_blocking", args.s2_blocking))
    reply["infer_s"] = round(time.time() - t_req, 3)
    print(f"[dualvln] #{state['n']} {reply.get('discrete_action', 'wait' if reply.get('wait') else 'traj')} "
          f"pixel {reply.get('pixel_goal')} in {reply['infer_s']} s", flush=True)
    return jsonify(reply)


@app.route("/health")
def health():
    return jsonify({"ok": True, "quant": args.quant})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, threaded=False)  # one request at a time; System 2 has its own thread
