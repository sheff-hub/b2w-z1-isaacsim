"""Extend a trained RSL-RL actor-critic with new observation inputs, appended at
the end, initialised to ZERO -- so the extended network computes exactly what the
original did until training gives the new inputs a weight.

Used for rfm17: rfm15d's calm arm + 6 numbers of integral hand error. The Adam
moments of the first layers are extended with zeros too, so the optimizer state
still loads and training resumes where it was.

    ./run_script.sh add_obs_columns.py <in.pt> <out.pt> --actor_new 6 --critic_new 6
"""
import argparse
import os
import shutil

import torch

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--actor_new", type=int, required=True)
p.add_argument("--critic_new", type=int, required=True)
a = p.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
old = {}
for name, extra in (("actor.0.weight", a.actor_new), ("critic.0.weight", a.critic_new)):
    w = sd[name]
    old[tuple(w.shape)] = extra
    sd[name] = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype)], dim=1)
    print(f"{name}: {tuple(w.shape)} -> {tuple(sd[name].shape)}")
assert len(old) == 2, "actor and critic first layers have the same shape; match by name instead"
for idx, st in ck["optimizer_state_dict"]["state"].items():
    for key in ("exp_avg", "exp_avg_sq"):
        t = st.get(key)
        if t is not None and tuple(t.shape) in old:
            st[key] = torch.cat([t, torch.zeros(t.shape[0], old[tuple(t.shape)], dtype=t.dtype)], dim=1)
            print(f"optimizer state {idx}.{key} -> {tuple(st[key].shape)}")
os.makedirs(os.path.dirname(a.dst), exist_ok=True)
src_params = os.path.join(os.path.dirname(a.src), "params")
if os.path.isdir(src_params) and not os.path.exists(os.path.join(os.path.dirname(a.dst), "params")):
    shutil.copytree(src_params, os.path.join(os.path.dirname(a.dst), "params"))
torch.save(ck, a.dst)
print("wrote", a.dst)
