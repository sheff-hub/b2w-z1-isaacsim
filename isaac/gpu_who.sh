#!/bin/bash
# Who is using the Ada GPU (the one our containers are pinned to), and how much.
#
#   ./gpu_who.sh
#
# Lists every process on the card with its owner, GPU memory, run time and command,
# marks ours, and prints the total. Other users' processes cannot be stopped from this
# account (no sudo); ask the owner, or an admin with the PIDs listed here.
set -euo pipefail
GPU_UUID="${ISAAC_GPU_UUID:-GPU-5be167c7-2d06-56dc-cacb-d299d6860070}"
ME="$(id -un)"

nvidia-smi --id="${GPU_UUID}" --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader | awk -F', ' '{printf "%s: %s of %s used, %s busy\n\n", $1, $2, $3, $4}'

printf "%-8s %-16s %9s %12s  %s\n" PID USER GPU_MEM RUNNING COMMAND
nvidia-smi --id="${GPU_UUID}" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
while IFS=', ' read -r pid mem; do
    [ -n "${pid}" ] || continue
    user="$(ps -o user= -p "${pid}" 2>/dev/null || echo '?')"
    etime="$(ps -o etime= -p "${pid}" 2>/dev/null | tr -d ' ' || echo '?')"
    cmd="$(ps -o args= -p "${pid}" 2>/dev/null | cut -c1-90 || echo '?')"
    # container processes show uid 1234 (Isaac) or our own uid (DualVLN server)
    tag=""
    { [ "${user}" = "${ME}" ] || [ "${user}" = "1234" ]; } && tag="  <- ours"
    printf "%-8s %-16s %7s MB %12s  %s%s\n" "${pid}" "${user}" "${mem}" "${etime}" "${cmd}" "${tag}"
done

echo
echo "Logged-in users (to ask them):"
who | awk '{print "  " $1 "  since " $3 " " $4}' | sort -u
