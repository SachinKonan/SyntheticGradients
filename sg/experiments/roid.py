"""ROID baseline (Marsden, Döbler, Yang, WACV 2024): universal test-time adaptation.

Follows the authors' code (mariodoebler/test-time-adaptation: methods/roid.py,
cfgs/imagenet_c/roid.yaml, conf.py defaults), ResNet-50 on ImageNet-C:
  - only the BN (scale, bias) adapt, BN with test-batch statistics;
  - SGD, lr 2.5e-4, momentum 0.9, Nesterov, batch 64, one step per batch;
  - loss: soft likelihood ratio, weighted per example by exp(diversity x certainty / (1/3)),
    where diversity = 1 - cos(running mean prediction, prediction) and certainty = -entropy
    (each min-max normalized over the batch); examples with below-mean diversity get weight
    0; plus a symmetric cross-entropy consistency term between the predictions on an
    augmented copy of the kept examples and on the originals (both with gradients), weighted
    the same; both sums divided by the batch size;
  - the running mean prediction is an EMA (0.9) of the batch-mean softmax;
  - after every step, weight ensembling: BN params <- 0.99 BN params + 0.01 source;
  - prediction (scored before the update): the logits times a smoothed batch prior.
The augmented copy holds only the kept examples, so its BN statistics are over those
(resnet.apply batch_mask). Augmentations: sg.augment.tta_augment (strong ranges, reflect
padding, no blur or noise), as their get_tta_transforms(padding_mode="reflect",
cotta_augs=False).

Same streams, scoring and output as gate3_adapt (step = --lr x each of --mults).
"""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from sg.augment import tta_augment
from sg.data import imagenet
from sg.experiments import streams as st
from sg.experiments.cotta import symmetric_ce
from sg.experiments.gate3_adapt import add_set_args, deployment_streams
from sg.models import resnet


def soft_likelihood_ratio(logits, clip=0.99, eps=1e-5):
    probs = jnp.clip(jax.nn.softmax(logits), 0.0, clip)
    return -jnp.sum(probs * jnp.log(probs / (1.0 - probs) + eps), -1)


def minmax(w):
    return (w - w.min()) / (w.max() - w.min() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    st.add_launch_args(ap)
    add_set_args(ap)
    ap.add_argument("--lr", type=float, default=2.5e-4, help="ROID's ImageNet-C lr; scaled by each multiplier")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--mults", default="0.3,1,3,10")
    ap.add_argument("--momentum-src", type=float, default=0.99, help="weight ensembling with the source")
    ap.add_argument("--momentum-probs", type=float, default=0.9, help="EMA of the mean prediction")
    ap.add_argument("--temperature", type=float, default=1 / 3)
    args = ap.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)
    cache = Path(args.local_cache)
    specs, sm, streams, n_seg, orders, mask_logits = deployment_streams(args, cache, log)
    S = len(specs)
    mults = np.array([float(m) for m in args.mults.split(",")], np.float32)
    traj_shard = NamedSharding(sm.shard.mesh, P(None, "streams"))  # (multipliers, streams, ...)
    log(f"ROID on set {args.set}: {S} streams, lr {args.lr} x {mults.tolist()}")
    params, stats = sm.replicate(resnet.load_torchvision(st.fetch_file(args.weights, cache)))
    bn0 = {n: params[n] for n in resnet.bn_names()}
    num_steps = st.common_steps(streams)
    if args.steps:
        num_steps = min(num_steps, args.steps)

    def fwd(bn, x01, mask=None):
        logits, _ = resnet.apply({**params, **bn}, stats, (x01 - imagenet.IMAGENET_MEAN) / imagenet.IMAGENET_STD,
                                 batch_stats=True, batch_mask=mask)
        return mask_logits(logits)

    def per_stream(bn, vel, probs_ema, key, x, y, lr):
        x01 = x / 255.0
        key, k_aug = jax.random.split(key)
        B = x.shape[0]

        def loss_fn(bn):
            outputs = fwd(bn, x01)
            probs = jax.lax.stop_gradient(jax.nn.softmax(outputs))
            cos = probs @ probs_ema / (jnp.linalg.norm(probs, axis=-1) * jnp.linalg.norm(probs_ema) + 1e-12)
            w_div = minmax(1.0 - cos)
            keep = w_div >= jnp.mean(w_div)
            w_cert = minmax(jnp.sum(probs * jax.lax.stop_gradient(jax.nn.log_softmax(outputs)), -1))  # -entropy
            w = jnp.where(keep, jnp.exp(w_div * w_cert / args.temperature), 0.0)
            loss = jnp.sum(soft_likelihood_ratio(outputs) * w) / B
            out_aug = fwd(bn, tta_augment(k_aug, x01, pad_mode="reflect", blur_noise=False), mask=keep)
            loss += jnp.sum(symmetric_ce(out_aug, outputs) * w) / B
            return loss, (outputs, probs)

        (_, (outputs, probs)), g = jax.value_and_grad(loss_fn, has_aux=True)(bn)
        # prediction before the update, with prior correction
        prior = jnp.mean(jax.nn.softmax(outputs), 0)
        C = outputs.shape[1]
        smooth = max(1 / B, 1 / C) / jnp.max(prior)
        correct = jnp.sum(jnp.argmax(outputs * (prior + smooth) / (1 + smooth * C), -1) == y)
        new_probs = args.momentum_probs * probs_ema + (1 - args.momentum_probs) * jnp.mean(probs, 0)
        # SGD with Nesterov momentum (torch: buf = mu buf + g; p -= lr (g + mu buf)), then ensembling
        new_vel = jax.tree.map(lambda v, gg: args.momentum * v + gg, vel, g)
        new_bn = jax.tree.map(lambda w_, gg, v: w_ - lr * (gg + args.momentum * v), bn, g, new_vel)
        new_bn = jax.tree.map(lambda w_, w0: args.momentum_src * w_ + (1 - args.momentum_src) * w0, new_bn, bn0)
        ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(new_bn)]))
        keep_ = lambda a, b: jax.tree.map(lambda u, v: jnp.where(ok, u, v), a, b)
        return (keep_(new_bn, bn), keep_(new_vel, vel), jnp.where(ok, new_probs, probs_ema), key,
                jnp.stack([correct, 1 - ok.astype(jnp.int32)]))

    @functools.partial(jax.jit, donate_argnums=(0, 1, 2), out_shardings=(traj_shard,) * 5)
    def step(bn, vel, probs_ema, keys, x, y):
        def per_mult(a):
            b_m, v_m, p_m, k_m, mult = a
            return jax.vmap(per_stream, in_axes=(0, 0, 0, 0, 0, 0, None))(b_m, v_m, p_m, k_m, x, y, args.lr * mult)
        return jax.lax.map(per_mult, (bn, vel, probs_ema, keys, jnp.asarray(mults)))

    @functools.partial(jax.jit, out_shardings=(traj_shard,) * 4)
    def init():
        bcast = lambda t: jax.tree.map(lambda a: jnp.broadcast_to(a, (len(mults), S) + a.shape), t)
        keys = jax.random.split(jax.random.key(0), len(mults) * S).reshape(len(mults), S)
        return bcast(bn0), jax.tree.map(jnp.zeros_like, bcast(bn0)), jnp.full((len(mults), S, 1000), 1e-3), keys

    bn, vel, probs_ema, keys = init()
    totals = np.zeros((len(mults), S, 2), np.int64)
    seg_correct = np.zeros((len(mults), S, n_seg), np.int64)
    steps_per_seg = max(1, num_steps // n_seg)
    t0 = time.time()
    for i, (x, y) in enumerate(st.prefetch_batches(streams, num_steps, args.decode_workers)):
        bn, vel, probs_ema, keys, counts = step(bn, vel, probs_ema, keys, sm.put(x), sm.put(y))
        c = sm.gather(counts)
        totals += c
        seg_correct[:, :, min(i // steps_per_seg, n_seg - 1)] += c[:, :, 0]
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == num_steps:
            n = (i + 1) * args.batch
            corrupt = np.array([not g.startswith("imagenet_val") for g, _ in specs])
            err = 100 * (1 - totals[:, corrupt, 0].sum(-1) / (n * corrupt.sum()))
            log(f"step {i + 1}/{num_steps} {(time.time() - t0) / (i + 1):.1f}s/step  error by multiplier "
                f"{np.round(err, 1).tolist()}")

    if lead:
        images = num_steps * args.batch
        out = {"config": vars(args), "set": args.set, "images_per_stream": images,
               "streams": [{"group": g, "order": o} for g, o in specs], "multipliers": mults.tolist(),
               "error": {"roid": (100 * (1 - totals[:, :, 0] / images)).tolist()},
               "error_by_position": {"roid": (100 * (1 - seg_correct / (steps_per_seg * args.batch))).mean(1).tolist()}
               if n_seg > 1 else None,
               "continual_orders": [o.tolist() for o in orders] if orders is not None else None,
               "skipped_updates": {"roid": totals[:, :, 1].tolist()}}
        st.write_output(args.out, "results.json", json.dumps(out).encode())
        log("done")


if __name__ == "__main__":
    main()
