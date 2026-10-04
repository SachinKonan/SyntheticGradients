#!/usr/bin/env bash
# Runs on every host of the TPU slice (from the SkyPilot task's `run`).
# Usage: bash tpu/run.sh <python module> [module args...]
# Needs SKYPILOT_NODE_IPS (one line per TPU VM, rank order) and SKYPILOT_NODE_RANK.
set -euo pipefail
cd "$(dirname "$0")/.."

# Full-slice run: drop any sub-slice settings left by earlier jobs on this worker.
unset TPU_PROCESS_BOUNDS TPU_CHIPS_PER_PROCESS_BOUNDS TPU_PROCESS_ADDRESSES TPU_PROCESS_PORT \
  CLOUD_TPU_TASK_ID TPU_VISIBLE_CHIPS
export JAX_PLATFORMS=tpu
# Sliced gcloud downloads gave repeatable hash mismatches on this fleet.
export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0
export PATH="$HOME/.local/bin:$PATH"

# Idle pool workers can still hold the TPU from an earlier job.
holders=$(sudo lsof -t /dev/accel* 2>/dev/null | sort -u || true)
if [ -n "$holders" ]; then
  echo "[rank $SKYPILOT_NODE_RANK] TPU held by leftover processes: $(ps -o pid=,comm= -p ${holders//$'\n'/,} | tr '\n' ' ')"
  echo "$holders" | xargs -r sudo kill -9
  sleep 5
fi
sudo rm -f /tmp/libtpu_lockfile

venv="$HOME/.venvs/synthgrad"
[ -x "$venv/bin/python" ] || UV_NO_CONFIG=1 uv venv "$venv" --python 3.12
UV_NO_CONFIG=1 uv pip install -q --python "$venv/bin/python" \
  'jax==0.11.1' 'jaxlib==0.11.1' 'libtpu==0.0.46' 'requests==2.32.5' numpy pillow array-record safetensors

ips=$(printf '%s\n' "$SKYPILOT_NODE_IPS" | awk 'NF')
coordinator="$(echo "$ips" | head -n1):8476"
module="$1"; shift
exec "$venv/bin/python" -m "$module" \
  --coordinator "$coordinator" --num-processes "$(echo "$ips" | wc -l)" --process-id "$SKYPILOT_NODE_RANK" "$@"
