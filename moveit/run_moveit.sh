#!/bin/bash
# Start MoveIt (move_group + the HTTP bridge moveit_server.py) for the Z1 arm, detached.
#   moveit/run_moveit.sh          start   (python3 run.py moveit start does the same)
#   moveit/run_moveit.sh stop     stop it
#   docker logs -f <containers.moveit>         the bridge; move_group's own log: docker exec <name> cat /tmp/move_group.log
# CPU only, no GPU. Listens on 127.0.0.1:<moveit.port> (host network); its own ROS domain (moveit.domain_id), local
# to this machine, so it never mixes with Nav2's.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(python3 "${HERE}/../isaac/config.py" --shell)"
if [ "${1:-}" = "stop" ]; then docker stop "${MOVEIT_CONTAINER}" >/dev/null 2>&1 && echo "moveit stopped" || echo "moveit not running"; exit 0; fi
[ -f "${HERE}/config/z1.urdf" ] || python3 "${HERE}/make_config.py"
MESHES="${HERE}/../robots_usd/z1_gripper_description/meshes"
[ -d "${MESHES}" ] || { echo "ERROR: ${MESHES} missing: run ./setup.sh (robot assets)" >&2; exit 1; }
docker rm -f "${MOVEIT_CONTAINER}" >/dev/null 2>&1 || true
docker run -d --rm --name "${MOVEIT_CONTAINER}" --network host \
    -e ROS_DOMAIN_ID="${MOVEIT_DOMAIN_ID}" -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    -e MOVEIT_PORT="${MOVEIT_PORT}" \
    -v "${HERE}:/moveit" -v "${MESHES}:/moveit/meshes:ro" \
    "${IMAGE_MOVEIT}" bash /moveit/entrypoint.sh >/dev/null
for _ in $(seq 1 60); do
    curl -sf -m 2 "http://127.0.0.1:${MOVEIT_PORT}/health" >/dev/null 2>&1 && { echo "moveit ready on :${MOVEIT_PORT} (docker logs -f ${MOVEIT_CONTAINER})"; exit 0; }
    sleep 1
done
echo "ERROR: MoveIt did not come up; see: docker logs ${MOVEIT_CONTAINER}" >&2; exit 1
