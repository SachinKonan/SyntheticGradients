#!/usr/bin/env bash
# Submit a module to the v4-64 pool (or, with TPU=v6e8, the us-east5 v6e-8 pool) at the current commit.
# Usage: [TPU=v6e8] bash tpu/submit.sh <job name> <python module> [module args...]
# The code bundle is `git archive HEAD`, so the tree must be clean and pushed.
set -euo pipefail
cd "$(dirname "$0")/.."
case ${TPU:-v4-64} in
  v4-64) POOL=tpuswarm-v4-64-central2-qwen35-erdos; BUCKET=gs://sk7524-tinker-tpu-us-central2/synthgrad; YAML=tpu/v4-64.yaml ;;
  v6e8) POOL=tpuswarm-v6e8-east5b; BUCKET=gs://sk7524-tinker-tpu-us-east5/synthgrad; YAML=tpu/v6e-8.yaml ;;
  *) echo "unknown TPU=$TPU" >&2; exit 1 ;;
esac
export CLOUDSDK_CONFIG=/home/sk7524/.config/gcloud-tpuswarm-compute-sa-v6e32
GCLOUD=/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud

name=$1; module=$2; shift 2
# Cross-region GCS traffic is expensive: every gs:// argument must be in this pool's bucket.
for a in "$@"; do
  for url in $(grep -oE 'gs://[^,= ]+' <<<"$a" || true); do
    [[ $url == $BUCKET/* ]] || { echo "$url is outside $BUCKET" >&2; exit 1; }
  done
done
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
  --env CODE_URL="$code_url" --env MODULE="$module" --env ARGS="$*" "$YAML"
