# Isaac Sim launchers (`isaac/`)

How the containers are started: plain Isaac Sim (GUI or headless with a WebRTC stream), scripts of this repository
inside the Isaac Lab container, and RL training. Machine settings come from `config.yaml` in the repository root
(see `config.example.yaml`); `isaac_env.sh` turns them into the docker arguments every launcher shares.
[SETUP_NOTES.md](SETUP_NOTES.md) records why the scripts look the way they do (written on the lab workstation `ProArt`).

---

## Quick start

```bash
./isaac/launch_gui.sh            # from the repository root: Isaac Sim 5.1.0 in a window
./isaac/launch_gui_6.sh          # Isaac Sim 6.0.1 (photoreal NuRec scenes)
```

Wait about a minute. The Isaac Sim window opens on your desktop. Close the window, or
press `Ctrl-C` in the terminal, to stop it.

That's it for normal use. The rest of this file covers the other modes and what to do
when something goes wrong.

---

## ⚠ One rule on shared machines: never use `--gpus all`

Set `gpu.uuid` in `config.yaml` to your card. Example, the lab workstation this was built on, which has two GPUs
and **only the smaller one is ours**:

```
01:00.0   RTX PRO 6000 Blackwell, 96 GB   ← Slurm's. Other people's jobs run here.
03:00.0   RTX 6000 Ada Generation, 48 GB  ← ours
```

The host confines your account to the Ada card, but **that confinement does not reach
Docker containers**. `--gpus all` would silently take Slurm's GPU out from under whoever
is using it.

The launchers pin the GPU from `config.yaml`. If you ever write a new `docker run` by hand, copy the GPU flag:

```bash
--gpus device=<gpu.uuid from config.yaml>
```

---

## Fresh container vs. resuming one

```bash
./launch_gui.sh                    # fresh container -- deletes the previous one first
docker start -i isaac-sim-gui      # resume the existing one, keeps everything inside it
```

Both work, exactly as on the office workstation. Your simulation work is safe either
way: the repository, the third-party checkouts and the cache are mounted from the host and
survive both. The difference is only what lives *inside* the container — packages you
`pip install` or `apt install` while it's running.

**Default to `./launch_gui.sh`.** A fresh container each time means nothing accumulates
invisibly, so the setup stays reproducible and portable across machines. If you need an
extra package permanently, add it to a Dockerfile rather than installing it by hand.

**Use `docker start -i`** when you have installed something inside the container and want
it back. Careful: running `./launch_gui.sh` afterwards **deletes** that container and
everything you added to it.

Note that `docker ps` shows only *running* containers. To see stopped ones:
```bash
docker ps -a
```

### A note on `xhost`

`launch_gui.sh` runs `xhost +local:` so the container (which runs as uid 1234, not as
you) can reach your X server. It does **not** revoke it on exit — deliberately, because
revoking breaks `docker start -i`: the container would load Isaac Sim fully and then have
nowhere to draw, printing `Isaac Sim Full App is loaded.` with no window and no error.

The grant lasts until you log out. While it is active, any user logged into this machine
can connect to your display, and colleagues do have sessions here. To close it when you
finish for the day:

```bash
xhost -local:
```

Set `XHOST_REVOKE=1 ./launch_gui.sh` if you want revoke-on-exit back — but then
`docker start -i` will not show a window.

## The two launch modes

### GUI mode — `./launch_gui.sh`

Opens the Isaac Sim window directly on this machine's screen. Use this when you're
sitting at the workstation.

Requires a graphical session (`DISPLAY` must be set). The script handles the X server
permission grant for you and revokes it when you exit.

### Headless mode — `./launch_headless.sh`

Runs with no window and streams the view over the network. Use this when working from
another machine. **Verified working from the laptop over ZeroTier.**

```bash
./launch_headless.sh                                 # default: ZeroTier 10.147.19.106
LIVESTREAM_ADDR=10.162.45.149 ./launch_headless.sh   # TUM LAN instead
```

Then open the **Isaac Sim WebRTC Streaming Client** on the viewing machine, enter
`10.147.19.106`, and click Connect.

#### You need the standalone client — there is no browser page

Isaac Sim 5.1.0 does **not** serve a web client. Port `8211` is not listening and
`omni.services.streamclient.webrtc` is not loaded; only the streaming server
(`omni.kit.livestream.webrtc`) runs. Any tutorial pointing you at
`http://<ip>:8211/streaming/webrtc-demo/` is for an older release and will show nothing.

Download it on the machine you view *from*:

```bash
wget https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
chmod +x isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
./isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
```

AppImages need `libfuse2` on Ubuntu 22.04+ (`sudo apt install libfuse2`).

#### Ports

| Port | Protocol | Behaviour |
|---|---|---|
| `49100` | TCP | Signaling. Listens from startup. |
| `47998` | UDP | Media. **Only binds once a client connects** — its absence beforehand is normal, not a fault. |

Don't go hunting for a bug because `ss -tuln` doesn't show 47998 before you connect.

#### Addresses, and the routing gotcha

| Address | Interface | Use when |
|---|---|---|
| `10.147.19.106` | ZeroTier `ztc3q54tec` | **Default.** Laptop is on this network (`10.147.19.133`). |
| `10.147.20.106` | ZeroTier `zt6ovyltm4` | Other ZeroTier network |
| `10.162.45.149` | `enp12s0`, TUM LAN | Client on the same LAN |

The address must be on a network the **viewing machine shares with this one**. The
script warns if you pick one that isn't on this host at all, but it cannot tell whether
your client can route to it.

> **The gotcha that cost us an afternoon:** a machine can belong to *several* ZeroTier
> networks at once, and the address you happen to know may not be the one that matters.
> The laptop is on both `172.23.136.182` and `10.147.19.133`; testing against the
> `172.23.x` address showed no route and produced a wrong "no common network" conclusion,
> when in fact `10.147.19.x` was shared all along.
>
> Check **both ends** before concluding anything:
> ```bash
> sudo zerotier-cli listnetworks
> ```
> This box is genuinely not on the office workstation's `172.23.x` network. The old
> hardcoded `172.23.59.19` was the office box's own address — meaningless here.

---

## Changing where things live

Edit `config.yaml` (template: `config.example.yaml`). Every value can also be overridden for one command:

```bash
ISAAC_GPU_UUID=GPU-... ./isaac/launch_gui.sh
CACHE_ROOT=/fast/disk/isaac-cache ./isaac/launch_gui.sh
LIVESTREAM_ADDR=10.162.45.149 ./isaac/launch_headless.sh
CONTAINER_NAME=isaac-experiment ./isaac/launch_gui.sh
```

| config.yaml | Variable | What it does |
|---|---|---|
| `gpu.uuid` | `ISAAC_GPU_UUID` | The GPU the containers get (`all` only on a machine you own) |
| `paths.isaaclab`, `paths.robot_lab` | `ISAACLAB_DIR`, `ROBOT_LAB_DIR` | Third-party checkouts (setup.sh clones them) |
| `paths.isaac_cache` | `CACHE_ROOT` | Shader/asset cache (several GB) |
| `paths.nurec` | `NUREC_ROOT` | NuRec scene assets |
| `livestream.address` / `.port` | `LIVESTREAM_ADDR` / `LIVESTREAM_PORT` | Headless streaming endpoint |
| `images.*` | `ISAAC_IMAGE` | Container images |
| — | `CONTAINER_NAME` | `isaac-sim-gui` / `-headless` by default; useful for running two at once |

---

## Everyday commands

```bash
docker ps                              # is Isaac Sim running?
docker stop isaac-sim-gui              # stop it
docker logs -f isaac-sim-gui           # watch its output
docker exec -it isaac-sim-gui bash     # open a shell inside the running container
```

To get a shell in the container *instead of* starting the simulator — handy for running
ROS 2 commands or poking at files:

```bash
./isaac/run_script.sh <script.py>      # a repository script, with every mount and the GPU pin
docker run --rm -it --entrypoint bash --gpus device=<gpu.uuid> -v "$PWD:/workspace" \
  --network=host --ipc=host isaac-lab:2.3.2     # a bare shell (no third-party mounts)
```

---

## Where your files go

| Inside the container | On this machine |
|---|---|
| `/workspace` | this repository |
| `/workspace/repos/IsaacLab`, `/workspace/repos/robot_lab` | `paths.isaaclab`, `paths.robot_lab` |
| `/workspace/scenes/nurec` | `paths.nurec` |
| `/isaac-sim/.cache` etc. | `paths.isaac_cache` |

Anything you want to keep must be saved into `/workspace/...`. **Files written elsewhere
in the container are lost when it stops.**

### Scenes

```
<paths.nurec>/nvidia/   NVIDIA's reference scenes -- read-only inside the container
<paths.nurec>/mine/     your own reconstructions (reconstruction/) -- writable
```

On the lab workstation: `zh_fourth_floor_iphone` (1.1 G) and `hand_hold-endeavor-livingroom` (3.7 G), from the
`nvidia/PhysicalAI-Robotics-NuRec` dataset on Hugging Face (ten scenes in total; download them at tag `v0.0`).

To add another, download it on the **host** — not inside the container:

```bash
hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset \
  --local-dir <paths.nurec>/nvidia \
  --include "<scene-name>/*"
```

Use `hf`, not the deprecated `huggingface-cli`. Keep `--local-dir` — without it the files
go into the Hugging Face cache in a blob/symlink layout instead of a plain tree.

> **Two ways this bites.** Running `hf download` with no `--include` downloads the *whole*
> dataset, which is much larger than you want. And `--include A B C` silently drops the
> **first** pattern — it gets absorbed into the command's filename argument, the download
> still appears to succeed, and one file is quietly missing. Check afterwards that you got
> what you asked for. Details in [SETUP_NOTES.md](SETUP_NOTES.md) §9.

Writing into `nurec/nvidia/` from inside the container fails with `Permission denied`. That
is deliberate — NVIDIA's reference assets are not meant to be modified. Save your own work
to `nurec/mine/`, which is writable.

---

## Troubleshooting

**`cannot talk to the Docker daemon`**
Your shell doesn't have the `docker` group yet. Log out and back in, or run
`newgrp docker`. Check with `id -nG` — you should see `docker` in the list.

**`DISPLAY is not set`**
You're on SSH or a text console. Use `./launch_headless.sh` instead.

**The window doesn't appear, but the container is running**
The X permission grant may have failed. Try `xhost +local:` manually, then relaunch.

**`a container named 'isaac-sim-gui' is already running`**
One is already up. Attach with `docker exec -it isaac-sim-gui bash`, or stop it with
`docker stop isaac-sim-gui`. (A *stopped* leftover is cleaned up automatically.)

**First launch is slow**
Expected — Isaac Sim compiles shaders on first run. That's cached to
`paths.isaac_cache`, so later launches are much faster. If it's slow *every* time, the
cache isn't persisting: check that the cache folder is filling up.

**Streaming client connects but the view is blank / it disconnects**
Check three things in order: (1) you're using the **standalone** Isaac Sim WebRTC
Streaming Client, not a browser — there is no browser page in 5.1.0; (2) the
`LIVESTREAM_ADDR` you launched with is on a network your client machine actually shares
(`sudo zerotier-cli listnetworks` on **both** ends); (3) `docker logs isaac-sim-headless`
shows `Streaming server started.`
A `NVST_CCE_DISCONNECTED` warning in the log means a client reached the signaling port
and then failed to negotiate — usually the wrong client type or an unroutable address.

**Is my hardware still OK?**
Run NVIDIA's checker inside the container:
```
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```
Last run: **PASSED** — driver 570.207 (minimum 535.161), RTX 6000 Ada, 51.53 GB VRAM.

**Did I accidentally grab Slurm's GPU?**
Inside a running container, `nvidia-smi` should show exactly **one** GPU, the RTX 6000
Ada. If you see the RTX PRO 6000 Blackwell, stop immediately — the GPU pin isn't working.

---

## ROS 2

The container ships ROS 2 **Jazzy**; there is no native ROS 2 on this host (it's Ubuntu
22.04, and Jazzy targets 24.04). Run ROS commands inside the container.

The `fastdds.xml` profile here is the Phase 1 networking fix — it disables shared memory,
which Docker's PID/IPC namespaces block, and forces plain UDPv4 so topics cross the
container/host boundary. It's wired up automatically via
`FASTRTPS_DEFAULT_PROFILES_FILE`. Don't remove it; ROS 2 topics will silently stop
crossing the boundary.

Verified working: `rclpy` loads and the `isaacsim.ros2.bridge` extension starts.

---

## Files here

| File | What it is |
|---|---|
| `README.md` | This guide |
| `config.py` | Reads `config.yaml` for the shell scripts (`eval "$(python3 isaac/config.py --shell)"`) |
| `scene_info.py` | Reads a scene file (`scenes/*.yaml`) for the shell scripts |
| `isaac_env.sh`, `isaac_env_6.sh` | Shared docker arguments (mounts, GPU pin, ROS 2), for 5.1.0 and 6.0.1. Not run directly. |
| `launch_gui.sh`, `launch_gui_6.sh` | Isaac Sim in a window, fresh container |
| `launch_headless.sh`, `launch_headless_6.sh` | Headless with a WebRTC livestream |
| `run_script.sh` | Any repository script in the Isaac Lab container (`run.py script` calls it) |
| `run_train.sh` | RL training / play with robot_lab's scripts (`run.py train` calls it) |
| `run_stage_check.sh`, `open_stage_check.py`, `open_stage_stream.py` | Open a NuRec stage headlessly / at startup |
| `gpu_who.sh` | Who uses the GPU, and how much |
| `fastdds.xml` | ROS 2 networking profile (UDP only) |
| `docker/Dockerfile.isaaclab` | The `isaac-lab:2.3.2` image (Isaac Lab + robot_lab on Isaac Sim 5.1.0), built by `setup.sh` |
| `SETUP_NOTES.md` | Why the scripts look the way they do |
