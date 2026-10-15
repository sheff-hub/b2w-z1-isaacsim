"""Stand-in for dualvln_server.py with the same API, to test the simulator side
(camera capture, depth encoding, trajectory following, videos) without the model.

Answers each request with a 2 m path curving gently left (robot frame), and STOP
after --stop_after requests. Also checks what it receives (image sizes, depth range).

    docker run --rm --network=host -v $PWD/vln:/v dualvln-server python /v/mock_server.py
"""

import argparse
import json

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--stop_after", type=int, default=25)
p.add_argument("--port", type=int, default=5801)
args = p.parse_args()
app = Flask(__name__)
n = {"i": 0}


@app.route("/eval_dual", methods=["POST"])
def eval_dual():
    rgb = np.asarray(Image.open(request.files["image"].stream).convert("RGB"))
    depth = np.asarray(Image.open(request.files["depth"].stream).convert("I")).astype(np.float32) / 10000.0
    data = json.loads(request.form["json"])
    if data.get("reset"):
        n["i"] = 0
    n["i"] += 1
    valid = depth[depth > 0]
    print(f"[mock] #{n['i']} rgb {rgb.shape} mean {rgb.mean():.0f} | depth {depth.shape} "
          f"{valid.min() if valid.size else 0:.2f}-{valid.max() if valid.size else 0:.2f} m | "
          f"fx {data['intrinsic'][0][0]:.1f} | {data['instruction'][:40]!r}", flush=True)
    if n["i"] > args.stop_after:
        return jsonify({"discrete_action": [0], "infer_s": 0.0})
    s = np.linspace(0, 2.0, 12)
    traj = np.c_[s, 0.15 * s ** 2, np.zeros_like(s)]  # forward, curving left
    return jsonify({"trajectory": traj.tolist(), "pixel_goal": [200, 192], "infer_s": 0.0})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
