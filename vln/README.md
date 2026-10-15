# vln/: the DualVLN server

DualVLN ([InternVLA-N1](https://github.com/InternRobotics/InternNav), Wei et al.) runs as an HTTP server in its own
container, next to the simulator; `project/vln_nav.py` sends it camera images and gets short paths back.

    ./setup.sh --vln              # InternNav at the pinned commit, the image, the model weights
    python3 run.py vln start      # or: ./vln/run_dualvln_server.sh [--quant 8bit|none]
    python3 run.py vln stop

| File | What |
|---|---|
| `Dockerfile.dualvln` | the server image (PyTorch 25.10 + transformers 4.51; no flash-attn: SDPA instead) |
| `dualvln_server.py` | the server: loads `InternRobotics/InternVLA-N1-DualVLN` (bf16 ~21 GB or 8-bit ~10 GB of GPU memory) |
| `run_dualvln_server.sh` | starts it on the GPU, port, and Hugging Face cache of `config.yaml` |
| `mock_server.py` | answers like the server without a model, for testing the client |

InternNav also loads the Depth-Anything-V2 weights from the relative path `checkpoints/...`; `run_dualvln_server.sh`
links them from the Hugging Face cache into `out/checkpoints/` (its working folder) before it starts. Logs: `out/`.
