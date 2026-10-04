"""Reproduce source / BN-adapt / Tent error on ImageNet-C (severity 5), ResNet-50.

Streams are independent and stepped together, one or more per device:
15 corruptions x 2 orders + clean val x 2 orders = 32 streams (one per v4-64 chip).
Each stream follows Tent's protocol: batch 64, SGD lr 2.5e-4 momentum 0.9, reset
per corruption, every batch scored before the update on it. The final partial
batch is dropped (49,984 of 50,000 images).

Per step and stream we score three models on the same batch:
  source   frozen model, running BN statistics
  bn_adapt frozen model, test-batch BN statistics
  tent     Tent-adapted BN affine params, test-batch statistics

Run on every host of the slice (tpu/run.sh), or locally on CPU for a smoke test:
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
    python -m sg.experiments.tent_imagenet_c --data-root <packed> --weights <st> \\
      --streams 4 --batch 4 --steps 2 --out /tmp/sg-smoke
"""

import argparse
import concurrent.futures as cf
import functools
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sg import tent
from sg.data import imagenet
from sg.data.records import decode
from sg.models import resnet

METHODS = ("source", "bn_adapt", "tent")


# ----------------------------------------------------------------------------- data

def stream_specs():
    """(group, order) for every stream. Order 0 is the stored (pre-shuffled) order."""
    groups = [f"imagenet_c/{c}/5" for c in imagenet.TEST_CORRUPTIONS] + ["imagenet_val"]
    return [(g, order) for order in (0, 1) for g in groups]


def fetch(root: str, rel: str, cache: Path) -> Path:
    """Local path of root/rel. A gs:// root is copied into the cache once."""
    if not root.startswith("gs://"):
        return Path(root, rel)
    dst = cache / rel
    if not (dst / ".complete").exists():
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", "-r", f"{root}/{rel}/*", str(dst)], check=True)
        (dst / ".complete").touch()
    return dst


class Stream:
    def __init__(self, group_dir: Path, order: int, batch: int):
        self.records = []
        for path in imagenet.shard_paths(group_dir, ""):
            reader = ArrayRecordReader(str(path))
            self.records += reader.read_all()
            reader.close()
        n = len(self.records)
        self.perm = np.arange(n) if order == 0 else np.random.default_rng(order).permutation(n)
        self.batch = batch
        self.num_steps = n // batch

    def batch_records(self, step):
        return [self.records[i] for i in self.perm[step * self.batch:(step + 1) * self.batch]]


def prefetch_batches(streams, num_steps, workers, depth=3):
    """Background thread that decodes (local_streams, B, 224, 224, 3) uint8 batches."""
    q = queue.Queue(maxsize=depth)
    pool = cf.ThreadPoolExecutor(workers)

    def decode_one(rec):
        label, _, jpeg = decode(rec)
        return imagenet.load_uint8(jpeg), label

    def run():
        for step in range(num_steps):
            recs = [r for s in streams for r in s.batch_records(step)]
            out = list(pool.map(decode_one, recs))
            x = np.stack([o[0] for o in out]).reshape(len(streams), -1, 224, 224, 3)
            y = np.array([o[1] for o in out], np.int32).reshape(len(streams), -1)
            q.put((x, y))
        q.put(None)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


# ----------------------------------------------------------------------------- model

def stream_step(bn, velocity, totals, x_uint8, y, params, stats, bn0, *, lr, momentum):
    """One batch of one stream. Returns updated (bn, velocity, totals)."""
    x = imagenet.normalize(x_uint8)
    frozen = {**params, **bn0}
    src_logits, _ = resnet.apply(frozen, stats, x, batch_stats=False)
    bna_logits, _ = resnet.apply(frozen, stats, x, batch_stats=True)
    new_params, velocity, tent_logits, _ = tent.tent_step(
        {**params, **bn}, velocity, stats, x, lr=lr, momentum=momentum)
    correct = jnp.stack([jnp.sum(l.argmax(-1) == y) for l in (src_logits, bna_logits, tent_logits)])
    return tent.bn_params(new_params), velocity, totals + correct


def global_from_local(local, sharding, global_shape, local_index):
    """Assemble a stream-sharded global array from this host's stream slices."""
    arrays = [jax.device_put(local[local_index[d]], d) for d in sharding.addressable_devices]
    return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="gs://sk7524-tinker-tpu-us-central2/synthgrad/data/imagenet_v1")
    p.add_argument("--weights", default="gs://sk7524-tinker-tpu-us-central2/synthgrad/weights/resnet50_tv_in1k.safetensors")
    p.add_argument("--local-cache", default=os.path.expanduser("~/synthgrad/cache"))
    p.add_argument("--out", required=True, help="results dir (local or gs://)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--steps", type=int, default=None, help="cap steps (smoke tests)")
    p.add_argument("--streams", type=int, default=None, help="use the first N streams (smoke tests)")
    p.add_argument("--precision", default="highest", choices=["default", "high", "highest"])
    p.add_argument("--decode-workers", type=int, default=min(64, os.cpu_count()))
    p.add_argument("--coordinator", default=None)
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--process-id", type=int, default=0)
    args = p.parse_args()

    if args.num_processes > 1:
        jax.distributed.initialize(args.coordinator, args.num_processes, args.process_id)
    jax.config.update("jax_default_matmul_precision", args.precision)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    specs = stream_specs()[: args.streams]
    devices = np.array(jax.devices())
    assert len(specs) % len(devices) == 0, f"{len(specs)} streams for {len(devices)} devices"
    mesh = Mesh(devices, ("streams",))
    shard = NamedSharding(mesh, P("streams"))
    repl = NamedSharding(mesh, P())
    num_streams = len(specs)
    log(f"{jax.process_count()} hosts, {len(devices)} devices, {num_streams} streams")

    # Which global streams live on this host; each local device holds a contiguous range.
    ranges = {d: range(*sl[0].indices(num_streams))
              for d, sl in shard.addressable_devices_indices_map((num_streams,)).items()}
    local_ids = sorted(i for r in ranges.values() for i in r)
    pos = {s: k for k, s in enumerate(local_ids)}
    local_index = {d: slice(pos[r[0]], pos[r[-1]] + 1) for d, r in ranges.items()}

    cache = Path(args.local_cache)
    t0 = time.time()
    streams = [Stream(fetch(args.data_root, specs[s][0], cache), specs[s][1], args.batch) for s in local_ids]
    weights = args.weights
    if weights.startswith("gs://"):
        cache.mkdir(parents=True, exist_ok=True)
        local_w = cache / Path(weights).name
        if not local_w.exists():
            subprocess.run(["gcloud", "storage", "cp", weights, str(local_w)], check=True)
        weights = str(local_w)
    log(f"data + weights ready in {time.time() - t0:.0f}s")

    num_steps = min(s.num_steps for s in streams)
    if jax.process_count() > 1:
        num_steps = int(np.min(multihost_utils.process_allgather(np.array([num_steps]))))
    if args.steps:
        num_steps = min(num_steps, args.steps)

    params_np, stats_np = resnet.load_torchvision(weights)
    put_repl = lambda t: jax.tree.map(lambda a: jax.make_array_from_callback(a.shape, repl, lambda i: a[i]), t)
    params, stats = put_repl(params_np), put_repl(stats_np)
    bn0 = tent.bn_params(params)

    @functools.partial(jax.jit, out_shardings=shard)
    def init_state(bn0):
        bcast = lambda a: jnp.broadcast_to(a, (num_streams,) + a.shape)
        bn = jax.tree.map(bcast, bn0)
        return bn, jax.tree.map(jnp.zeros_like, bn), jnp.zeros((num_streams, len(METHODS)), jnp.int32)

    step_fn = jax.jit(
        jax.vmap(functools.partial(stream_step, lr=args.lr, momentum=args.momentum),
                 in_axes=(0, 0, 0, 0, 0, None, None, None)),
        out_shardings=(shard, shard, shard), donate_argnums=(0, 1, 2))

    bn, velocity, totals = init_state(bn0)
    x_shape = (num_streams, args.batch, 224, 224, 3)
    t0 = time.time()
    for step, (x, y) in enumerate(prefetch_batches(streams, num_steps, args.decode_workers)):
        xg = global_from_local(x, shard, x_shape, local_index)
        yg = global_from_local(y, shard, x_shape[:2], local_index)
        bn, velocity, totals = step_fn(bn, velocity, totals, xg, yg, params, stats, bn0)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == num_steps:
            tot = np.asarray(multihost_utils.process_allgather(totals, tiled=True))
            seen = (step + 1) * args.batch
            err = 100 * (1 - tot.sum(0) / (seen * num_streams))
            log(f"step {step + 1}/{num_steps}  {(time.time() - t0) / (step + 1):.3f}s/step  "
                + "  ".join(f"{m} {e:.1f}" for m, e in zip(METHODS, err)))

    tot = np.asarray(multihost_utils.process_allgather(totals, tiled=True))
    if lead:
        write_results(args, specs, tot, num_steps * args.batch)


def write_results(args, specs, totals, images_per_stream):
    per_stream = [
        {"group": g, "order": o, **{m: 100 * (1 - int(c) / images_per_stream) for m, c in zip(METHODS, row)}}
        for (g, o), row in zip(specs, totals)
    ]
    by_group = {}
    for r in per_stream:
        by_group.setdefault(r["group"], []).append(r)
    summary = {g: {m: float(np.mean([r[m] for r in rs])) for m in METHODS} for g, rs in by_group.items()}
    corrupt = [g for g in summary if g.startswith("imagenet_c/")]
    mean_c = {m: float(np.mean([summary[g][m] for g in corrupt])) for m in METHODS} if corrupt else {}
    result = {
        "config": vars(args), "images_per_stream": images_per_stream,
        "error_pct": {"imagenet_c_mean": mean_c, "per_group": summary}, "per_stream": per_stream,
    }
    text = json.dumps(result, indent=1)
    print("RESULT " + json.dumps(result["error_pct"]), flush=True)
    if args.out.startswith("gs://"):
        local = Path("/tmp/synthgrad-results.json")
        local.write_text(text)
        subprocess.run(["gcloud", "storage", "cp", str(local), f"{args.out}/results.json"], check=True)
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        Path(args.out, "results.json").write_text(text)


if __name__ == "__main__":
    main()
