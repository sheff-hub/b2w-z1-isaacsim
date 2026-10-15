#!/bin/bash
# Run the whole measurement battery on one checkpoint and drop the results in
# logs/eval/<tag>_*.json.
#
#   ./eval_all.sh <task-id> <checkpoint-inside-container> <tag>
#
# Covers the four questions we ask of every policy:
#   reach_map       which world targets it can actually reach (+ base height, pitch)
#   crouch_test     does the body help the arm (the wb9 comparison)
#   eval_locomotion does it still drive (tracking, tilt, falls)
#   waypoint        can it still go from A to B
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:?task id}"
CKPT="${2:?checkpoint path inside the container}"
TAG="${3:?short tag, e.g. wb12}"
OUT="/workspace/repos/robot_lab/logs/eval"
eval "$(python3 "${SCRIPT_DIR}/../isaac/config.py" --shell)"   # config.yaml: where robot_lab (and its logs/) is
LOG_DIR="${ROBOT_LAB_DIR}/logs/eval"
mkdir -p "${LOG_DIR}"

run() {
    local name="$1"; shift
    echo "=============== ${name} ==============="
    ISAAC_IMAGE_DEFAULT=lab CONTAINER_NAME="isaac-eval-${name}" \
        "${SCRIPT_DIR}/../isaac/run_script.sh" "$@" > "${LOG_DIR}/${TAG}_${name}.log" 2>&1
    grep -E "^(MAP|CROUCH|EVAL|WP)\|" "${LOG_DIR}/${TAG}_${name}.log" || \
        tail -5 "${LOG_DIR}/${TAG}_${name}.log"
}

run reach_map  reach_map.py  --task "${TASK}" --checkpoint "${CKPT}" --out "${OUT}/reach_map_${TAG}.json"
run crouch     crouch_test.py --task "${TASK}" --checkpoint "${CKPT}" --out "${OUT}/crouch_${TAG}.json"
run locomotion eval_locomotion.py --task "${TASK}" --checkpoint "${CKPT}" --out "${OUT}/loco_${TAG}.json"
run waypoint   waypoint.py --task "${TASK}" --checkpoint "${CKPT}" --vmax 0.3 --out "${OUT}/waypoint_${TAG}.json"
echo "all logs in ${LOG_DIR}/${TAG}_*.log"
