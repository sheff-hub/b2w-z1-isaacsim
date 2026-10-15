#!/bin/bash
# One-time setup of this repository on a new machine. Safe to run again: every step checks first.
#
#   ./setup.sh                     the simulator part: config, Isaac Lab + robot_lab + our tasks, images, robot assets
#   ./setup.sh --vln               also language navigation: InternNav, the DualVLN image and model (~16 GB download)
#   ./setup.sh --assets-from DIR   take the robot assets (robots_usd/) from a local folder instead of the download
#   ./setup.sh --check             only report what is there and what is missing
#
# Needs: Linux with an NVIDIA GPU and driver, Docker with the NVIDIA container toolkit, git, python3 with PyYAML.
# Reconstruction from phone photos (reconstruction/) has its own setup: see reconstruction/README.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# pinned third-party versions (the ones every result in docs/ was produced with)
ISAACLAB_URL=https://github.com/isaac-sim/IsaacLab.git;         ISAACLAB_REF=v2.3.2
ROBOT_LAB_URL=https://github.com/fan-ziqi/robot_lab.git;        ROBOT_LAB_REF=v2.3.2    # 09f6a9d
INTERNNAV_URL=https://github.com/InternRobotics/InternNav.git;  INTERNNAV_REF=7a5c62400ac45b313d9b709c740b64191556a242
DUALVLN_MODEL=InternRobotics/InternVLA-N1-DualVLN
DEPTH_MODEL=depth-anything/Depth-Anything-V2-Metric-Hypersim-Small   # InternNav loads its .pth as well
# robot assets (robots_usd/): too big for git, attached to a GitHub release of this repository
ASSETS_URL="${ASSETS_URL:-https://github.com/sheff-hub/b2w-z1-isaacsim/releases/download/v1.0/robots_usd.tar.gz}"   # e.g. https://github.com/<owner>/<repo>/releases/download/v1.0/robots_usd.tar.gz

WITH_VLN=0; CHECK_ONLY=0; ASSETS_FROM=""
while [ $# -gt 0 ]; do
    case "$1" in
        --vln) WITH_VLN=1 ;;
        --check) CHECK_ONLY=1 ;;
        --assets-from) ASSETS_FROM="$2"; shift ;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1 (see ./setup.sh --help)" >&2; exit 2 ;;
    esac
    shift
done

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
todo() { printf '  \033[33mtodo\033[0m  %s\n' "$*"; MISSING=$((MISSING + 1)); }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
MISSING=0

# ------------------------------------------------------------------------------------------------ 1. prerequisites
step "1. prerequisites"
command -v git >/dev/null || fail "git is not installed"
command -v docker >/dev/null || fail "docker is not installed"
docker info >/dev/null 2>&1 || fail "cannot talk to Docker (are you in the 'docker' group? log out and in again)"
docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -qw nvidia \
    || fail "the 'nvidia' Docker runtime is missing: install the NVIDIA container toolkit"
python3 -c "import yaml" 2>/dev/null || fail "python3 needs PyYAML:  python3 -m pip install --user pyyaml"
nvidia-smi >/dev/null 2>&1 && ok "NVIDIA driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
    || fail "nvidia-smi does not work: install the NVIDIA driver"
ok "git, docker + nvidia runtime, python3 + PyYAML"

# ------------------------------------------------------------------------------------------------ 2. config.yaml
step "2. machine config (config.yaml)"
if [ ! -f config.yaml ]; then
    if [ "${CHECK_ONLY}" = 1 ]; then todo "config.yaml missing (./setup.sh creates it from config.example.yaml)"
    else cp config.example.yaml config.yaml; ok "created config.yaml from config.example.yaml"; fi
fi
eval "$(python3 isaac/config.py --shell)"
if [ "${ISAAC_GPU_UUID}" = "all" ]; then
    echo "  note  gpu.uuid is 'all'. On a machine with GPUs that are not yours (shared server, Slurm), set it to"
    echo "        your card's UUID:  nvidia-smi --query-gpu=uuid,name --format=csv"
else
    nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -qx "${ISAAC_GPU_UUID}" \
        && ok "GPU ${ISAAC_GPU_UUID}" || fail "gpu.uuid ${ISAAC_GPU_UUID} is not a GPU of this machine (nvidia-smi -L)"
fi

# ------------------------------------------------------------------------------------------------ 3. third-party code
clone_at() {  # dir url ref
    local dir="$1" url="$2" ref="$3"
    if [ -d "${dir}/.git" ]; then
        local have want
        have="$(git -C "${dir}" rev-parse HEAD)"
        want="$(git -C "${dir}" rev-parse "${ref}^{commit}" 2>/dev/null || echo "${ref}")"
        [ "${have}" = "${want}" ] && ok "$(basename "${dir}") at ${ref}" \
            || echo "  note  $(basename "${dir}") is at ${have:0:9}, not ${ref}: results may differ"
    elif [ "${CHECK_ONLY}" = 1 ]; then
        todo "$(basename "${dir}") missing (${dir})"
    else
        mkdir -p "$(dirname "${dir}")"
        git clone --quiet "${url}" "${dir}"
        git -C "${dir}" -c advice.detachedHead=false checkout --quiet "${ref}"
        ok "cloned $(basename "${dir}") at ${ref}"
    fi
}
step "3. third-party code (pinned versions)"
clone_at "${ISAACLAB_DIR}" "${ISAACLAB_URL}" "${ISAACLAB_REF}"
clone_at "${ROBOT_LAB_DIR}" "${ROBOT_LAB_URL}" "${ROBOT_LAB_REF}"
# the isaac-lab image runs Isaac Lab's scripts against /isaac-sim
[ -d "${ISAACLAB_DIR}" ] && [ ! -e "${ISAACLAB_DIR}/_isaac_sim" ] && [ "${CHECK_ONLY}" = 0 ] && ln -sfn /isaac-sim "${ISAACLAB_DIR}/_isaac_sim"

# ------------------------------------------------------------------------------------------------ 4. our tasks
step "4. our Isaac Lab tasks into robot_lab (training/robot_lab_overlay)"
OVERLAY="${ROOT}/training/robot_lab_overlay"
inside_repo=0; case "${ROBOT_LAB_DIR}/" in "${ROOT}/"*) inside_repo=1 ;; esac
n_ok=0; n_new=0
[ -d "${ROBOT_LAB_DIR}/source/robot_lab" ] || { todo "robot_lab missing, so are our tasks"; OVERLAY_SKIP=1; }
[ "${OVERLAY_SKIP:-0}" = 1 ] || while IFS= read -r -d '' src; do
    rel="${src#"${OVERLAY}"/}"
    dst="${ROBOT_LAB_DIR}/${rel}"
    if [ "${inside_repo}" = 1 ]; then
        # a relative symlink: edits in robot_lab land in training/ (tracked), and it resolves in the container too
        target="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], os.path.dirname(sys.argv[2])))' "${src}" "${dst}")"
        if [ -L "${dst}" ] && [ "$(readlink "${dst}")" = "${target}" ]; then n_ok=$((n_ok + 1)); continue; fi
        [ "${CHECK_ONLY}" = 1 ] && { todo "${rel} not linked"; continue; }
        mkdir -p "$(dirname "${dst}")"; rm -rf "${dst}"; ln -s "${target}" "${dst}"; n_new=$((n_new + 1))
    else
        if [ -f "${dst}" ] && cmp -s "${src}" "${dst}"; then n_ok=$((n_ok + 1)); continue; fi
        [ "${CHECK_ONLY}" = 1 ] && { todo "${rel} not installed"; continue; }
        mkdir -p "$(dirname "${dst}")"; cp "${src}" "${dst}"; n_new=$((n_new + 1))
    fi
done < <(find "${OVERLAY}" -type f -not -name '*.pyc' -not -path '*/__pycache__/*' -print0)
[ -d "${ROBOT_LAB_DIR}" ] && ok "${n_ok} task files in place, ${n_new} installed now$([ "${inside_repo}" = 1 ] && echo ' (symlinks)' || echo ' (copies: re-run after editing training/)')"

# ------------------------------------------------------------------------------------------------ 5. images
have_image() { docker image inspect "$1" >/dev/null 2>&1; }
step "5. container images"
if have_image "${IMAGE_ISAAC_LAB}"; then ok "${IMAGE_ISAAC_LAB}"
elif [ "${CHECK_ONLY}" = 1 ]; then todo "${IMAGE_ISAAC_LAB} not built"
else
    [ "$(dirname "${ISAACLAB_DIR}")" = "$(dirname "${ROBOT_LAB_DIR}")" ] && [ "$(basename "${ISAACLAB_DIR}")" = IsaacLab ] \
        && [ "$(basename "${ROBOT_LAB_DIR}")" = robot_lab ] \
        || fail "to build ${IMAGE_ISAAC_LAB}, paths.isaaclab and paths.robot_lab must be IsaacLab/ and robot_lab/ in one folder"
    echo "  building ${IMAGE_ISAAC_LAB} (pulls ${IMAGE_ISAAC_SIM}, ~30 GB; 20-40 min) ..."
    # build from clean checkouts: the overlay's symlinks would dangle inside the build context
    ctx="$(mktemp -d)"; trap 'rm -rf "${ctx}"' EXIT
    for d in "${ISAACLAB_DIR}" "${ROBOT_LAB_DIR}"; do
        git clone --quiet "${d}" "${ctx}/$(basename "${d}")"
        git -C "${ctx}/$(basename "${d}")" -c advice.detachedHead=false checkout --quiet "$(git -C "${d}" rev-parse HEAD)"
    done
    docker build -f isaac/docker/Dockerfile.isaaclab -t "${IMAGE_ISAAC_LAB}" "${ctx}"
    ok "built ${IMAGE_ISAAC_LAB}"
fi
if have_image "${IMAGE_NAV2}"; then ok "${IMAGE_NAV2}"
elif [ "${CHECK_ONLY}" = 1 ]; then todo "${IMAGE_NAV2} not built"
else docker build -t "${IMAGE_NAV2}" nav2/ && ok "built ${IMAGE_NAV2}"; fi

# ------------------------------------------------------------------------------------------------ 6. robot assets
step "6. robot assets (robots_usd/)"
if [ -f robots_usd/b2w_z1_gripper.usd ] && [ -d robots_usd/Unitree_B2W_rl ]; then ok "robots_usd/ present"
elif [ "${CHECK_ONLY}" = 1 ]; then todo "robots_usd/ missing"
elif [ -n "${ASSETS_FROM}" ]; then
    cp -r "${ASSETS_FROM}"/. robots_usd/ && ok "copied robots_usd/ from ${ASSETS_FROM}"
elif [ -n "${ASSETS_URL}" ]; then
    curl -fL "${ASSETS_URL}" | tar -xz -C robots_usd && ok "downloaded robots_usd/"
else
    todo "robots_usd/ missing: ./setup.sh --assets-from <folder>, or ASSETS_URL=<release tarball> ./setup.sh"
fi

# ------------------------------------------------------------------------------------------------ 7. language navigation
step "7. language navigation (DualVLN)$([ "${WITH_VLN}" = 1 ] || echo ' -- skipped, add --vln')"
if [ "${WITH_VLN}" = 1 ]; then
    clone_at "${INTERNNAV_DIR}" "${INTERNNAV_URL}" "${INTERNNAV_REF}"
    if have_image "${IMAGE_DUALVLN}"; then ok "${IMAGE_DUALVLN}"
    elif [ "${CHECK_ONLY}" = 1 ]; then todo "${IMAGE_DUALVLN} not built"
    else docker build -f vln/Dockerfile.dualvln -t "${IMAGE_DUALVLN}" vln/ && ok "built ${IMAGE_DUALVLN}"; fi
    for m in "${DUALVLN_MODEL}" "${DEPTH_MODEL}"; do
        if [ -d "${HF_CACHE}/models--${m//\//--}/snapshots" ]; then ok "${m} in ${HF_CACHE}"
        elif [ "${CHECK_ONLY}" = 1 ]; then todo "${m} not downloaded"
        else
            mkdir -p "${HF_CACHE}"
            pattern=""; [ "${m}" = "${DEPTH_MODEL}" ] && pattern=", allow_patterns=['depth_anything_v2_metric_hypersim_vits.pth']"
            docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "${HF_CACHE}:/hf" "${IMAGE_DUALVLN}" \
                python -c "from huggingface_hub import snapshot_download; snapshot_download('${m}', cache_dir='/hf'${pattern})"
            ok "downloaded ${m}"
        fi
    done
fi

# ------------------------------------------------------------------------------------------------ done
step "summary"
if [ "${MISSING}" -gt 0 ]; then
    echo "  ${MISSING} item(s) still missing (see 'todo' above)."
else
    echo "  Ready. Try:  python3 run.py fetch --episodes 1        (README.md: all commands)"
fi
