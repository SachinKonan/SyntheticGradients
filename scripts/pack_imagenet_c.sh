#!/bin/bash
# Pack all downloaded ImageNet-C tars in parallel, then check every shard.
# Run on della after download + md5 check. Usage: bash scripts/pack_imagenet_c.sh
set -euo pipefail
DATA=/scratch/gpfs/ZHUANGL/sk7524/data/imagenet-c
export UV_CACHE_DIR=/scratch/gpfs/ZHUANGL/sk7524/.cache/uv
cd "$(dirname "$0")/.."

for f in noise blur weather digital extra; do
  uv run --python 3.11 --group prep scripts/pack_imagenet.py imagenet-c "$DATA/raw/$f.tar" \
    --out "$DATA/packed" > "$DATA/logs/pack_$f.log" 2>&1 &
done
wait
uv run --python 3.11 --group prep scripts/check_packed.py "$DATA/packed"
