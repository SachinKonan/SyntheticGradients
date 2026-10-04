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
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sg import tent
from sg.data import imagenet
from sg.experiments import streams as st
from sg.models import resnet

METHODS = ("source", "bn_adapt", "tent")


def stream_specs():
    """(group, order) for every stream. Order 0 is the stored (pre-shuffled) order."""
    groups = [f"imagenet_c/{c}/5" for c in imagenet.TEST_CORRUPTIONS] + ["imagenet_val"]
    return [(g, order) for order in (0, 1) for g in groups]


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


def main():
    p = argparse.ArgumentParser()
    st.add_launch_args(p)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--steps", type=int, default=None, help="cap steps (smoke tests)")
    p.add_argument("--streams", type=int, default=None, help="use the first N streams (smoke tests)")
    args = p.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    specs = stream_specs()[: args.streams]
    sm = st.StreamMesh(len(specs))
    log(f"{jax.process_count()} hosts, {len(jax.devices())} devices, {len(specs)} streams")

    cache = Path(args.local_cache)
    t0 = time.time()
    streams = [st.Stream(st.fetch(args.data_root, specs[s][0], cache), specs[s][1], args.batch)
               for s in sm.local_ids]
    weights = st.fetch_file(args.weights, cache)
    log(f"data + weights ready in {time.time() - t0:.0f}s")

    num_steps = st.common_steps(streams)
    if args.steps:
        num_steps = min(num_steps, args.steps)

    params, stats = sm.replicate(resnet.load_torchvision(weights))
    bn0 = tent.bn_params(params)

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def init_state(bn0):
        bn = jax.tree.map(lambda a: jnp.broadcast_to(a, (len(specs),) + a.shape), bn0)
        return bn, jax.tree.map(jnp.zeros_like, bn), jnp.zeros((len(specs), len(METHODS)), jnp.int32)

    step_fn = jax.jit(
        jax.vmap(functools.partial(stream_step, lr=args.lr, momentum=args.momentum),
                 in_axes=(0, 0, 0, 0, 0, None, None, None)),
        out_shardings=(sm.shard, sm.shard, sm.shard), donate_argnums=(0, 1, 2))

    bn, velocity, totals = init_state(bn0)
    t0 = time.time()
    for step, (x, y) in enumerate(st.prefetch_batches(streams, num_steps, args.decode_workers)):
        bn, velocity, totals = step_fn(bn, velocity, totals, sm.put(x), sm.put(y), params, stats, bn0)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == num_steps:
            tot = sm.gather(totals)
            err = 100 * (1 - tot.sum(0) / ((step + 1) * args.batch * len(specs)))
            log(f"step {step + 1}/{num_steps}  {(time.time() - t0) / (step + 1):.3f}s/step  "
                + "  ".join(f"{m} {e:.1f}" for m, e in zip(METHODS, err)))

    tot = sm.gather(totals)
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
    print("RESULT " + json.dumps(result["error_pct"]), flush=True)
    st.write_output(args.out, "results.json", json.dumps(result, indent=1).encode())


if __name__ == "__main__":
    main()
