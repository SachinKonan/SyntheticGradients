#!/usr/bin/env bash
# SkyPilot client for the tpuswarm pools. The API server runs on $SKY_HOST at
# 127.0.0.1:46580. If it is not reachable here, this starts one shared SSH
# tunnel to $SKY_HOST that later calls reuse and that is killed after one hour
# (a later call starts a new one), so concurrent calls never cut each other off
# and nothing is left running for long.
# Never start a local API server: it would share (and clobber) the server's state.
export CLOUDSDK_CONFIG=/home/sk7524/.config/gcloud-tpuswarm-compute-sa-v6e32
export GOOGLE_APPLICATION_CREDENTIALS=/home/sk7524/.config/gcloud/vision-mix-compute-sa-key.json
export HOME=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32
export SKYPILOT_CONFIG=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/skypilot-config.yaml
export SKY_API_SERVER_URL=http://127.0.0.1:46580
export SKY_API_SERVER_ENDPOINT=$SKY_API_SERVER_URL
export SKYPILOT_API_SERVER_ENDPOINT=$SKY_API_SERVER_URL
export SKYPILOT_DISABLE_LOCAL_API_SERVER=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SKY=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky
SKY_HOST=della-milan  # was della-vis2 until 2026-10-08
CONTROL=/tmp/synthgrad-sky-tunnel-$(id -u)-$SKY_HOST.sock

healthy() { curl -sf -m 10 "$SKY_API_SERVER_URL/api/health" >/dev/null; }

if ! healthy; then
  # Start the shared tunnel unless one is already up (a concurrent call may be starting it).
  if ! ssh -S "$CONTROL" -O check "$SKY_HOST" >/dev/null 2>&1; then
    # Only the tunnel goes to the background, fully detached, so callers never wait on it.
    setsid nohup timeout 3600 ssh -N -M -S "$CONTROL" -o BatchMode=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -L 127.0.0.1:46580:127.0.0.1:46580 "$SKY_HOST" >/dev/null 2>&1 < /dev/null &
  fi
  for _ in $(seq 30); do healthy && break; sleep 1; done
  if ! healthy; then
    echo "SkyPilot API not reachable at $SKY_API_SERVER_URL, even through a $SKY_HOST tunnel" >&2
    exit 1
  fi
fi
exec "$SKY" "$@"
