#!/bin/bash
# Start the DualVLN server in its own container, on the GPU of config.yaml (gpu.uuid; never `--gpus all` on a
# shared machine: it bypasses the host's confinement and can reach GPUs that belong to other people's jobs).
#
#   ./vln/run_dualvln_server.sh [--quant 8bit|none] [--port 5801]
#
# Listens on the host network (config dualvln.port); the Isaac container reaches it at 127.0.0.1:<port>.
# Needs: the image (./setup.sh --vln builds it), InternNav (paths.internnav) and the model weights in the
# Hugging Face hub cache (paths.hf_cache; ./setup.sh --vln downloads InternRobotics/InternVLA-N1-DualVLN).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(python3 "${HERE}/../isaac/config.py" --shell)"
if [ "${ISAAC_GPU_UUID}" = "all" ]; then GPUS="all"; else GPUS="device=${ISAAC_GPU_UUID}"; fi
NAME="${CONTAINER_NAME:-${DUALVLN_CONTAINER}}"
[ -d "${INTERNNAV_DIR}/internnav" ] || { echo "ERROR: InternNav not found at ${INTERNNAV_DIR}: run ./setup.sh --vln" >&2; exit 1; }
mkdir -p "${HERE}/out/checkpoints"
# InternNav loads the Depth-Anything-V2 weights from the RELATIVE path checkpoints/... (internvla_n1_arch.py,
# MODEL_PATH_TO): link them from the Hugging Face cache into the server's working folder, as a container path
DEPTH="$(ls "${HF_CACHE}"/models--depth-anything--Depth-Anything-V2-Metric-Hypersim-Small/snapshots/*/depth_anything_v2_metric_hypersim_vits.pth 2>/dev/null | head -1)"
[ -n "${DEPTH}" ] || { echo "ERROR: Depth-Anything-V2-Metric-Hypersim-Small is not in ${HF_CACHE}: run ./setup.sh --vln" >&2; exit 1; }
ln -sfn "/workspace/huggingface/${DEPTH#"${HF_CACHE}"/}" "${HERE}/out/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
docker rm -f "${NAME}" >/dev/null 2>&1 || true
exec docker run --rm --name "${NAME}" \
    --gpus "${GPUS}" --network=host --ipc=host \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
    -v "${INTERNNAV_DIR}:/workspace/InternNav:ro" \
    -v "${HF_CACHE}:/workspace/huggingface" \
    -v "${HERE}:/workspace/vln" \
    -w /workspace/vln/out \
    "${IMAGE_DUALVLN}" \
    python -u /workspace/vln/dualvln_server.py --port "${DUALVLN_PORT}" "$@"
