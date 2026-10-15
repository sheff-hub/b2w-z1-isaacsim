"""Starting checkpoint for rfm15 (the calm arm): rfm14's weights with the ARM's
action std capped at 0.3 (it had grown to ~2.0).

The std is a learned parameter, and over 33,800 iterations at a constant entropy
bonus it had grown to ~2.0 on the arm (1.0 at the start) -- with an action scale
of 0.5, every arm joint target was jittered by ~1 rad per step during training.
The mean policy is untouched, so the policy behaves exactly as before at play
time; only the exploration around it narrows. The optimizer state is kept.

Any Python with torch; the logs are owned by the container user, so normally
    ./run_script.sh calm_init.py <in.pt> <out.pt> [--arm_std 0.3]
"""
import argparse
import os
import shutil

import torch

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--arm_std", type=float, default=0.3)
p.add_argument("--arm", default="16:22", help="slice of the 22-wide action that is the arm")
p.add_argument("--zero_arm_head", action="store_true",
               help="zero the actor's arm outputs (rfm19: incremental arm commands, so a = 0 holds still)")
p.add_argument("--arm_std_set", type=float, default=None, help="SET the arm std to this instead of capping")
a = p.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
lo, hi = (int(v) for v in a.arm.split(":"))
before = sd["std"][lo:hi].clone()
# a cap, not a reset: a joint the policy already holds tightly (joint6, ~0.08) keeps it
sd["std"][lo:hi] = torch.clamp(before, max=a.arm_std)
if a.arm_std_set is not None:
    sd["std"][lo:hi] = a.arm_std_set
if a.zero_arm_head:
    # last actor layer: its rows lo:hi produce the arm's 6 outputs
    last = max(int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight"))
    sd[f"actor.{last}.weight"][lo:hi] = 0.0
    sd[f"actor.{last}.bias"][lo:hi] = 0.0
    print(f"zeroed actor.{last} rows {lo}:{hi}")
print("arm std", [round(x, 3) for x in before.tolist()], "->", [round(x, 3) for x in sd["std"][lo:hi].tolist()])
os.makedirs(os.path.dirname(a.dst), exist_ok=True)
# keep the source run's params/ next to it, as a record of what it was trained on
src_params = os.path.join(os.path.dirname(a.src), "params")
if os.path.isdir(src_params) and not os.path.exists(os.path.join(os.path.dirname(a.dst), "params")):
    shutil.copytree(src_params, os.path.join(os.path.dirname(a.dst), "params"))
torch.save(ck, a.dst)
print("wrote", a.dst)
