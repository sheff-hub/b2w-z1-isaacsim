#!/bin/bash
# Evaluate an RFM checkpoint on the three things that actually matter for it.
#
#   ./eval_rfm.sh <checkpoint-inside-container> <tag>
#   ./eval_rfm.sh latest <tag>          # newest checkpoint of the newest rfm run
#
#   approach   can it go to a pose anywhere around it, and arrive upright
#              (this is the real test: targets are fixed in the world)
#   reachmap   stationary accuracy over a grid, with --fixed_targets so a
#              DRIVING policy is not scored against a target that rides along
#              with it -- without that flag this test reads ~10% for a policy
#              that is actually fine
#   The reach map also prints arm-vs-base activity and lidar-mast intrusion.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${TASK:-RobotLab-Isaac-RFM-Unitree-B2W-Z1-v0}"
eval "$(python3 "${SCRIPT_DIR}/../isaac/config.py" --shell)"   # config.yaml: where robot_lab (and its logs/) is
LOGS_HOST="${ROBOT_LAB_DIR}/logs/rsl_rl/b2w_z1_rfm"
OUT=/workspace/repos/robot_lab/logs/eval
CK="${1:?checkpoint path inside the container, or 'latest'}"
TAG="${2:?short tag}"

if [ "${CK}" = "latest" ]; then
    RUN=$(ls -t "${LOGS_HOST}" | head -1)
    MODEL=$(ls "${LOGS_HOST}/${RUN}"/model_*.pt | xargs -n1 basename | sort -V | tail -1)
    CK="/workspace/repos/robot_lab/logs/rsl_rl/b2w_z1_rfm/${RUN}/${MODEL}"
    echo "latest = ${RUN}/${MODEL}"
fi

LOG_DIR="${ROBOT_LAB_DIR}/logs/eval"
mkdir -p "${LOG_DIR}"

echo "=============== approach ==============="
ISAAC_IMAGE_DEFAULT=lab CONTAINER_NAME="isaac-ev-appr-${TAG}" \
    "${SCRIPT_DIR}/../isaac/run_script.sh" approach_test.py --task "${TASK}" --checkpoint "${CK}" \
    --seconds 18 --out "${OUT}/approach_${TAG}.json" > "${LOG_DIR}/${TAG}_approach.log" 2>&1
grep -E "^APPR\|" "${LOG_DIR}/${TAG}_approach.log" | tail -18


# The sit-back was reported on HIGH targets, where it cannot be a reach strategy,
# so they get a run of their own.
echo
echo "=============== approach, high targets (1.0 m) ==============="
ISAAC_IMAGE_DEFAULT=lab CONTAINER_NAME="isaac-ev-high-${TAG}" \
    "${SCRIPT_DIR}/../isaac/run_script.sh" approach_test.py --task "${TASK}" --checkpoint "${CK}" \
    --seconds 18 --target_height 1.0 --out "${OUT}/approach_high_${TAG}.json" > "${LOG_DIR}/${TAG}_high.log" 2>&1
grep -E "^APPR\|" "${LOG_DIR}/${TAG}_high.log" | tail -18

echo
echo "=============== reach map =============="
ISAAC_IMAGE_DEFAULT=lab CONTAINER_NAME="isaac-ev-map-${TAG}" \
    "${SCRIPT_DIR}/../isaac/run_script.sh" reach_map.py --task "${TASK}" --checkpoint "${CK}" \
    --fixed_targets --out "${OUT}/reach_map_${TAG}.json" > "${LOG_DIR}/${TAG}_map.log" 2>&1
grep -E "^MAP\|" "${LOG_DIR}/${TAG}_map.log" | grep -E "reached |ARM vs BASE|INTO THE|into the|median|base height used|span"
echo
echo "logs in ${LOG_DIR}/${TAG}_*.log"
