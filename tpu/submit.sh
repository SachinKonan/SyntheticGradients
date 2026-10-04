#!/usr/bin/env bash
# Submit a module to the v4-64 pool at the current commit.
# Usage: bash tpu/submit.sh <job name> <python module> [module args...]
# The code bundle is `git archive HEAD`, so the tree must be clean and pushed.
set -euo pipefail
cd "$(dirname "$0")/.."
POOL=${POOL:-tpuswarm-v4-64-central2-qwen35-erdos}
BUCKET=gs://sk7524-tinker-tpu-us-central2/synthgrad
export CLOUDSDK_CONFIG=/home/sk7524/.config/gcloud-tpuswarm-compute-sa-v6e32
GCLOUD=/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud

name=$1; module=$2; shift 2
[ -z "$(git status --porcelain)" ] || { echo "commit your changes first" >&2; exit 1; }
sha=$(git rev-parse --short=12 HEAD)
code_url=$BUCKET/code/synthgrad-$sha.tar.gz
if ! $GCLOUD storage ls "$code_url" >/dev/null 2>&1; then
  tmp=$(mktemp --suffix=.tar.gz)
  git archive --format=tar.gz -o "$tmp" HEAD
  $GCLOUD storage cp "$tmp" "$code_url"
  rm -f "$tmp"
fi
echo "submitting $name ($module @ $sha) to $POOL"
bash tpu/sky.sh jobs launch -y -d -p "$POOL" -n "$name" \
  --env CODE_URL="$code_url" --env MODULE="$module" --env ARGS="$*" tpu/v4-64.yaml
