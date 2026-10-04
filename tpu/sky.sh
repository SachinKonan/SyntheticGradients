#!/usr/bin/env bash
# SkyPilot client for the tpuswarm pools. The API server runs on della-vis2 at
# 127.0.0.1:46580; from della9, open a tunnel first:
#   ssh -N -L 127.0.0.1:46580:127.0.0.1:46580 della-vis2
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

if ! curl -sf -m 10 "$SKY_API_SERVER_URL/api/health" >/dev/null; then
  echo "SkyPilot API not reachable at $SKY_API_SERVER_URL (is the della-vis2 tunnel up?)" >&2
  exit 1
fi
exec /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky "$@"
