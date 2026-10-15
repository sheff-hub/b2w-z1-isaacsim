> **Note (repository version).** These notes were written on the lab workstation before the code became this
> repository. Read `~/Firdavs/Isaac` as the repository's `isaac/` and `project/` folders, `~/Firdavs/repos` as
> `repos/`, and the `WORKSPACE_ROOT` / `SCENES_ROOT` variables as `config.yaml` (`paths.*`).

# Setup notes — what changed from the office-workstation scripts, and why

Technical record of the adaptation for the lab workstation (`ProArt`), 6–7 August 2026.
The originals in `../claude project/` are left untouched as the reference copy.

**For instructions on actually launching Isaac Sim, see [README.md](README.md).**

## What changed, and why

### 1. GPU is pinned — do not revert this to `--gpus all`

This workstation has two GPUs:

```
00000000:01:00.0   RTX PRO 6000 Blackwell, 96 GB   ← Slurm's
00000000:03:00.0   RTX 6000 Ada Generation, 48 GB  ← ours
```

`/etc/profile.d/gpu-firvads_ruiyang.sh` confines this account to the Ada card via
`CUDA_VISIBLE_DEVICES`. **That variable does not reach containers** — Docker does not
inherit the host environment, so the original `--gpus all` would have handed Isaac Sim
the 96 GB card that Slurm allocates to other people's jobs. Verified: a container
started with `--gpus all` sees both cards.

Both scripts now pass `--gpus device=GPU-5be167c7-...`, selected by UUID rather than
index so it survives PCI reordering. Confirmed to expose only the Ada card.

Agreed with the admin: Ada only, for now.

### 2. Livestream address is no longer hardcoded

`launch_headless.sh` previously hardcoded `172.23.59.19` — the **office** workstation's
ZeroTier address, which does not exist on this machine. It now defaults to this host's
LAN address and warns if the configured address isn't actually present locally.

Addresses available here:

```
10.162.45.149   enp12s0, TUM LAN
10.147.20.106   ZeroTier zt6ovyltm4
10.147.19.106   ZeroTier ztc3q54tec
```

Note this machine is **not** on the office's `172.23.x` ZeroTier network. Streaming
between the two requires joining a common network first.

### 3. Paths parameterized

`WORKSPACE_ROOT` (default `$HOME/Firdavs`) replaces the hardcoded `~/Firdavs` mounts.
`CACHE_ROOT` is separate so it can be repointed at `/bulk` independently — which is the
intended destination once a directory exists there.

### 4. Cache volumes added

The originals mounted no cache directories, so every container start redid shader
compilation from scratch. Cache paths are now mounted under `CACHE_ROOT`
(default `~/docker/isaac-sim`). This matters most for iterative Phase 3 scenario work.

**These use the Isaac Sim 5.1.0 layout, where the container's home is `/isaac-sim`.**
Isaac Sim 4.x used `/root/...` instead. Do not copy mount paths from older tutorials —
wrong paths fail silently, the container simply doesn't persist its cache. Source:
[install_container.html](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html).

### 5. Container name collisions handled

A fixed `--name` fails with "name already in use" once a container is left behind. The
scripts now remove a stopped container of the same name and refuse to clobber a running
one, printing the `docker exec` command to attach instead.

### 6. Preflight checks

Before starting, both scripts verify the Docker daemon is reachable, the `nvidia`
runtime is registered, and `fastdds.xml` exists where
`FASTRTPS_DEFAULT_PROFILES_FILE` points. `launch_gui.sh` additionally requires `DISPLAY`.

### 6b. `xhost` grant corrected, and TTY handling added

Two bugs found while running the first real launch:

**`xhost +local:root` granted the wrong identity.** The container runs as uid 1234, not
root (see §7), so that grant authorized nobody relevant and the X connection would fail.
Changed to `xhost +local:`, which allows any client on the local unix socket.

The original script had **no `xhost` handling at all** — it neither granted nor revoked,
relying on the X session already being open. Adding a revoke-on-exit was a mistake: it
broke the established `docker start -i isaac-sim-gui` resume workflow, where the container
loads Isaac Sim fully and then silently has nowhere to draw. The grant is now left in
place for the session, matching the original behaviour. `XHOST_REVOKE=1` restores
revoke-on-exit for anyone who wants it.

**`docker run -it` fails without a terminal** ("the input device is not a TTY"), which
breaks any non-interactive use — cron, CI, scripts, agent shells. `set_run_mode()` now
detects this: `-it` when stdin is a terminal, `-d` (detached) otherwise, with the
container name and `docker logs -f` instructions printed.

### 7. `--allow-root` is a no-op — the container does not run as root

Worth knowing, because it was assumed otherwise. Inspecting the 5.1.0 image:

```
User: isaac-sim          uid=1234(isaac-sim) gid=1234(isaac-sim)
HOME: /isaac-sim         (from /etc/passwd; no HOME set in the image env)
```

Neither the original scripts nor these pass `-u`, so the container runs as **uid 1234**,
not root. `--allow-root` only has an effect when the process is genuinely uid 0, so it
has been doing nothing all along — on the office workstation too. It is harmless and is
left in place to keep these scripts diff-comparable with the originals.

This means the setup already matches NVIDIA's documented user model, and the
`/isaac-sim/...` cache paths above are the correct ones.

**Permissions consequence:** the host cache directories are owned by the host user
(1005) but written by container uid 1234. Without a fix the container cannot write, and
the failure is silent — shaders just recompile every start. NVIDIA's docs prescribe
`sudo chown -R 1234:1234 ~/docker/isaac-sim`. `ensure_dirs()` instead applies
`chmod -R a+rwX`, which needs no root; safe because `$HOME` is mode 0750, so nothing
outside this account can traverse in. Verified: container writes succeed to `.cache`,
`.nvidia-omniverse/logs`, and `.local/share/ov/data`.

### 8. `PRIVACY_CONSENT=Y` added

Prevents Isaac Sim blocking on an interactive telemetry prompt on first run. This is a
new environment variable not present in the originals — flagged here so it isn't a
surprise.

### 9. Scene assets mounted at `/workspace/scenes` (7 Aug 2026)

`SCENES_ROOT` (default `$HOME/Firdavs/scenes`) follows the same pattern as `CACHE_ROOT`:
a separate variable rather than a path derived from `WORKSPACE_ROOT`, because these are
large downloaded assets rather than working files and may need to move independently.
It is mounted at `/workspace/scenes`.

```
scenes/nurec/nvidia/   NVIDIA's reference scenes (PhysicalAI-Robotics-NuRec) -- reference only
scenes/nurec/mine/     our own reconstructions
```

Backup of the pre-change file: `isaac_env.sh.bak-2026-08-07`.

**`ensure_dirs()` creates the tree, and that is not optional.** Docker creates a missing
bind-mount source directory **as root**. Without the `mkdir -p`, a first launch on a fresh
machine would silently leave a root-owned `scenes/` that this account cannot write and
cannot remove without the admin.

#### The permission split is deliberate

Same uid-1234 problem as the cache (§7), but the answer differs per directory:

| Directory | Mode | Container (uid 1234) can |
|---|---|---|
| `nurec/nvidia/` | `775` / `664` | **read only** — "other" gets `r-x`/`r--` |
| `nurec/mine/` | `a+rwX` via `ensure_dirs()` | **read and write** |

Reference assets stay unwritable because the container has no business modifying them;
our own reconstructions must be writable or Isaac Sim cannot save them back out. Both
verified inside a running container: `touch` succeeds in `mine/`, fails with
`Permission denied` in `nvidia/`.

Note the mount itself is read-write in both cases — the restriction is filesystem
permissions on the host, not the mount flag. Don't "fix" a write failure in `nvidia/` by
loosening it; that is the intended behaviour.

#### Downloading scenes

Use `hf`, not the deprecated `huggingface-cli`. `--local-dir` gives a plain directory tree
rather than the hub's blob/symlink cache layout, which is what Isaac Sim wants:

```bash
hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset \
  --local-dir ~/Firdavs/scenes/nurec/nvidia \
  --include "<scene>/*"
```

Downloaded so far — `zh_fourth_floor_iphone` (1.1 G) and `hand_hold-endeavor-livingroom`
(3.7 G), 4.8 G total, plus `README.md` and `USAGE.md`. The dataset holds ten scenes;
the rest were deliberately not fetched.

**Two traps, both hit on the first attempt:**

`hf download <repo> --include A B C D` silently drops the **first** pattern. Argparse
claims it for the positional `filenames` slot, which is what produces
`Ignoring --include since filenames have been explicitly set` — yet the remaining patterns
still apply, so the download looks successful and one file is simply absent. Pass filenames
positionally, or verify afterwards against `HfApi().list_repo_files()`.

`hf download` with no `--include` **and** no `--local-dir` pulls the entire repository into
`HF_HUB_CACHE`. On this dataset that is far more than intended. There is no dry-run flag;
check the command before running it.

Also note the `.cache/` directory that appears inside `--local-dir`: that is per-file
download metadata that makes re-runs resumable, not the hub cache. Leave it.

## Host setup done (6 Aug 2026)

- Docker Engine 29.7.2 + containerd 2.3.3, official `jammy` repo, enabled at boot
- NVIDIA Container Toolkit 1.18.1 (from Lambda's repo, which is pinned at apt priority
  1001 and would otherwise downgrade a hand-installed NVIDIA build)
- `nvidia` runtime registered in `/etc/docker/daemon.json`
- `firvads_ruiyang` added to the `docker` group
- Isaac Sim 5.1.0 image pulled — 22.9 GB on disk, 7.6 GB compressed, 17 layers.
  **No NGC login or API key required**; the 5.1.0 image is public, verified by reading
  its manifest anonymously. (Older releases did require `docker login nvcr.io` with an
  NGC API key — obsolete for 5.1.0.)
- Verified: `docker run --rm --gpus all ubuntu:22.04 nvidia-smi`
- Verified: full GUI launch, window on display `:1`, ROS 2 Jazzy bridge + `rclpy` loaded,
  cache persisting to host, GPU pin holding under active rendering

## NGC CLI installed (7 Aug 2026)

NGC CLI **4.34.10** (published 4 Aug 2026), installed and configured on the host.

| | |
|---|---|
| Binary | `~/.local/ngc-cli/`, symlinked to `~/.local/bin/ngc` |
| Config | `~/.ngc/config`, mode `600` |
| Org | `1021907888348323` — "Firdavs Shukurilloev", role `ORG_OWNER`, no teams |
| Format | `ascii` |

Both integrity checks from NVIDIA's installer page passed: SHA256
`58bc4d7b6901551a095ce5a43a0065b1c22d5eaf1a164d40029ee3fda11ccd47` on the zip, and the
per-file `ngc-cli.md5` manifest. Verified afterwards with `ngc user who` and
`ngc registry image info nvidia/isaac-sim:5.1.0`.

### This does not change the "no NGC login needed" finding above

Docker still needs no credentials for Isaac Sim — the 5.1.0 image is public, and the CLI
confirms it. NGC CLI is here for a **different** job: pulling assets, models and resources
from the registry (`ngc registry resource download-version ...`), which is what Phase 3
world-building will want from NVIDIA's libraries. Don't read this section as a reversal.

### Installed to `~/.local`, *not* via NVIDIA's `~/.bash_profile` step

The official instructions end with:

```bash
echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile && source ~/.bash_profile
```

**Do not run that on this account.** There is no `~/.bash_profile` here, only `~/.profile`.
Bash reads *one* of the two for login shells and prefers `~/.bash_profile`, so creating it
shadows `~/.profile` entirely — and `~/.profile` is what puts `~/.local/bin` on `PATH` in
the first place. Following the doc literally would have quietly broken `PATH` at next
login, with nothing to connect the breakage to the NGC install.

Installing under `~/.local/ngc-cli` and symlinking into the already-on-`PATH`
`~/.local/bin` avoids touching any shell startup file. Uninstall is
`rm -rf ~/.local/ngc-cli ~/.local/bin/ngc`.

### Configuring

`ngc config set` prompts for API key, output format, org and team. In 4.34.10 there is
**no `ace` prompt** — older docs list one. Accepting the defaults is correct here; the org
is the only real choice and this account belongs to exactly one.

Chicken-and-egg worth knowing: before an org is configured, *every* authenticated command
— including `ngc org list` — fails with `Missing org - If Authenticated, org is also
required.` That is not a bad key. The org ID can be recovered without the CLI by
exchanging the key for a token at `authn.nvidia.com` and calling
`https://api.ngc.nvidia.com/v2/users/me`.

Cosmetic quirk: `ngc config current` reports the apikey's `source` as
`environment variable` even when `NGC_CLI_API_KEY` is unset and the key lives in
`~/.ngc/config`. The value is genuinely coming from the file — ignore the label.

## Resolved

**Driver — no upgrade needed.** Isaac Sim's own compatibility check passed on
6 Aug 2026 with driver 570.207:

   ```
   NVIDIA GPU(s)
     |-- Driver version [supported]
     |     |-- installed: 570.207
     |     |-- minimum: 535.161
     |-- GPU 0 [supported]     name: NVIDIA RTX 6000 Ada Generation
     |-- GPU 0: VRAM [excellent]   total: 51.53 GB   (minimum: 10 GB)
   System checking result: PASSED
   ```

The **580.65.06** figure on NVIDIA's requirements page is the *tested* driver, not the
minimum. The actual floor is **535.161**, so 570.207 is comfortably supported. No
partition drain, no reboot, no admin request.

Re-run the check any time with:
```
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

## Still outstanding

1. **`/bulk` access** — 7.3 TB, currently `root:root` and not writable by this account.
   The house convention is `/bulk/users/<name>` owned by the user, group `lab`, setgid,
   with quotas on. Needs `lab` group membership. Once available, repoint `CACHE_ROOT`
   and put datasets and rollout logs there rather than on `/`.
2. **CPU governor is `Powersave`** — flagged by the compatibility check as a performance
   drag. Affects the whole shared machine including Slurm jobs, and needs root. Raise
   with the admin alongside the `/bulk` request.
3. **Phase 2 open item** — physics rigging audit (joint limits, masses, inertia tensors,
   friction) against the Unitree datasheets, before Phase 4 WBC work.
