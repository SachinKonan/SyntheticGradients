"""CoTTA baseline (Wang et al., CVPR 2022): continual test-time adaptation of every weight.

Follows the authors' ImageNet code (qinenergy/cotta: imagenet/cotta.py, cfgs/.../cotta*.yaml):
  - student: every weight trained (BN with test-batch statistics) by SGD (lr 0.01,
    momentum 0.9, no weight decay, batch 64) on the symmetric cross-entropy between the
    student's and the teacher's predictions;
  - teacher: EMA of the student (0.999), updated after the student's step;
  - when the source model's mean confidence (max softmax) on the batch is below 0.1, the
    teacher's prediction is the mean of its logits over 32 augmented copies of the batch;
  - after every step each weight is reset to its source value with probability 0.001;
  - the scored prediction is the teacher's, before this batch's update.
Augmentations follow their get_tta_transforms (soft): color jitter (brightness 0.8-1.2,
contrast 0.85-1.15, saturation 0.75-1.25, hue +-0.03, gamma 0.85-1.15), edge padding then
a random affine (+-8 deg, translation up to 1/16 of the padded size, scale 0.95-1.05) and
center crop, blur (kernel 5, sigma 0.001-0.25), horizontal flip, noise (std 0.005), each
clipped to [0, 1]; one random draw per augmented copy of the batch, as their batched
transform does. Approximations: color ops in a fixed order (theirs is random), hue as a
YIQ rotation, edge-replicated borders for the blur.

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

from sg.data import imagenet
from sg.experiments import streams as st
from sg.experiments.gate3_adapt import add_set_args, deployment_streams
from sg.models import resnet

GRAY = jnp.array([0.299, 0.587, 0.114])
YIQ = jnp.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def augment(key, x):
    """One CoTTA augmentation of a whole batch x (B, H, W, 3) in [0, 1]."""
    k = jax.random.split(key, 12)
    u = lambda i, lo, hi: jax.random.uniform(k[i], (), minval=lo, maxval=hi)
    clip = lambda z: jnp.clip(z, 0.0, 1.0)
    # color jitter (each op clamps, as torchvision's do)
    x = clip(x * u(0, 0.8, 1.2))
    m = jnp.mean(x @ GRAY, axis=(1, 2))[:, None, None, None]
    x = clip((x - m) * u(1, 0.85, 1.15) + m)
    g = (x @ GRAY)[..., None]
    x = clip((x - g) * u(2, 0.75, 1.25) + g)
    a = 2 * jnp.pi * u(3, -0.03, 0.03)
    rot = jnp.array([[1, 0, 0], [0, jnp.cos(a), -jnp.sin(a)], [0, jnp.sin(a), jnp.cos(a)]])
    x = clip(jnp.einsum("bhwc,dc->bhwd", x, jnp.linalg.inv(YIQ) @ rot @ YIQ))
    x = clip(x ** u(4, 0.85, 1.15))
    # affine on edge padding (pad = H/2, translation up to 1/16 of the padded size), then crop
    n = x.shape[1]
    ang = jnp.deg2rad(u(5, -8.0, 8.0))
    tx, ty = jnp.round(u(6, -2 * n / 16, 2 * n / 16)), jnp.round(u(7, -2 * n / 16, 2 * n / 16))
    sc = u(8, 0.95, 1.05)
    c = (n - 1) / 2
    yy, xx = jnp.meshgrid(jnp.arange(n, dtype=jnp.float32), jnp.arange(n, dtype=jnp.float32), indexing="ij")
    qx, qy = xx - c - tx, yy - c - ty
    sx = (jnp.cos(ang) * qx + jnp.sin(ang) * qy) / sc + c
    sy = (-jnp.sin(ang) * qx + jnp.cos(ang) * qy) / sc + c
    x0, y0 = jnp.floor(sx), jnp.floor(sy)
    wx, wy = (sx - x0)[None, ..., None], (sy - y0)[None, ..., None]
    at = lambda yi, xi: x[:, jnp.clip(yi, 0, n - 1).astype(jnp.int32), jnp.clip(xi, 0, n - 1).astype(jnp.int32), :]
    x = ((1 - wy) * ((1 - wx) * at(y0, x0) + wx * at(y0, x0 + 1))
         + wy * ((1 - wx) * at(y0 + 1, x0) + wx * at(y0 + 1, x0 + 1)))
    # blur: separable 5-tap gaussian
    sigma = u(9, 0.001, 0.25)
    w = jnp.exp(-jnp.arange(-2, 3, dtype=jnp.float32) ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    pad = jnp.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="edge")
    x = sum(w[i] * pad[:, i:i + n, 2:2 + n] for i in range(5))
    pad = jnp.pad(x, ((0, 0), (0, 0), (2, 2), (0, 0)), mode="edge")
    x = sum(w[i] * pad[:, :, i:i + n] for i in range(5))
    # flip, noise
    x = jnp.where(jax.random.uniform(k[10]) < 0.5, x[:, :, ::-1], x)
    return clip(x + 0.005 * jax.random.normal(k[11], x.shape))


def symmetric_ce(s, t):
    """CoTTA's loss: 0.5 CE(teacher -> student) + 0.5 CE(student -> teacher), per example."""
    return (-0.5 * jnp.sum(jax.nn.softmax(t) * jax.nn.log_softmax(s), -1)
            - 0.5 * jnp.sum(jax.nn.softmax(s) * jax.nn.log_softmax(t), -1))


def main():
    ap = argparse.ArgumentParser()
    st.add_launch_args(ap)
    add_set_args(ap)
    ap.add_argument("--lr", type=float, default=0.01, help="CoTTA's ImageNet lr; scaled by each multiplier")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--mults", default="0.1,0.3,1,3")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--restore", type=float, default=0.001, help="per-weight reset probability per step")
    ap.add_argument("--confidence", type=float, default=0.1, help="augment when the source's mean confidence is below")
    ap.add_argument("--augs", type=int, default=32)
    args = ap.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)
    cache = Path(args.local_cache)
    specs, sm, streams, n_seg, orders, mask_logits = deployment_streams(args, cache, log)
    S = len(specs)
    mults = np.array([float(m) for m in args.mults.split(",")], np.float32)
    traj_shard = NamedSharding(sm.shard.mesh, P(None, "streams"))  # (multipliers, streams, ...)
    log(f"CoTTA on set {args.set}: {S} streams, lr {args.lr} x {mults.tolist()}")
    source, stats = sm.replicate(resnet.load_torchvision(st.fetch_file(args.weights, cache)))
    num_steps = st.common_steps(streams)
    if args.steps:
        num_steps = min(num_steps, args.steps)

    def fwd(p, x01):
        logits, _ = resnet.apply(p, stats, (x01 - imagenet.IMAGENET_MEAN) / imagenet.IMAGENET_STD, batch_stats=True)
        return mask_logits(logits)

    def per_stream(student, teacher, vel, key, x, y, lr):
        x01 = x / 255.0
        key, k_aug, k_restore = jax.random.split(key, 3)
        anchor_conf = jnp.mean(jnp.max(jax.nn.softmax(fwd(source, x01)), -1))
        t_std = fwd(teacher, x01)
        t_aug = jnp.mean(jax.lax.map(lambda kk: fwd(teacher, augment(kk, x01)),
                                     jax.random.split(k_aug, args.augs)), 0)
        t_out = jax.lax.stop_gradient(jnp.where(anchor_conf < args.confidence, t_aug, t_std))
        correct = jnp.sum(t_out.argmax(-1) == y)  # the teacher's prediction, before this update
        g = jax.grad(lambda p: jnp.mean(symmetric_ce(fwd(p, x01), t_out)))(student)
        new_vel = jax.tree.map(lambda v, gg: args.momentum * v + gg, vel, g)
        new_student = jax.tree.map(lambda w, v: w - lr * v, student, new_vel)
        new_teacher = jax.tree.map(lambda t, w: args.ema * t + (1 - args.ema) * w, teacher, new_student)
        leaves, treedef = jax.tree.flatten(new_student)
        ks = jax.random.split(k_restore, len(leaves))
        restored = [jnp.where(jax.random.uniform(kk, w.shape) < args.restore, w0, w)
                    for kk, w, w0 in zip(ks, leaves, jax.tree.leaves(source))]
        new_student = jax.tree.unflatten(treedef, restored)
        ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(new_student)]))
        keep = lambda a, b: jax.tree.map(lambda u, w: jnp.where(ok, u, w), a, b)
        return (keep(new_student, student), keep(new_teacher, teacher), keep(new_vel, vel), key,
                jnp.stack([correct, 1 - ok.astype(jnp.int32), (anchor_conf < args.confidence).astype(jnp.int32)]))

    @functools.partial(jax.jit, donate_argnums=(0, 1, 2), out_shardings=(traj_shard,) * 5)
    def step(student, teacher, vel, keys, x, y):
        def per_mult(a):
            s_m, t_m, v_m, k_m, mult = a
            return jax.vmap(per_stream, in_axes=(0, 0, 0, 0, 0, 0, None))(s_m, t_m, v_m, k_m, x, y, args.lr * mult)
        return jax.lax.map(per_mult, (student, teacher, vel, keys, jnp.asarray(mults)))

    @functools.partial(jax.jit, out_shardings=(traj_shard,) * 4)
    def init():
        bcast = lambda t: jax.tree.map(lambda a: jnp.broadcast_to(a, (len(mults), S) + a.shape), t)
        keys = jax.random.split(jax.random.key(0), len(mults) * S).reshape(len(mults), S)
        return bcast(source), bcast(source), jax.tree.map(jnp.zeros_like, bcast(source)), keys

    student, teacher, vel, keys = init()
    totals = np.zeros((len(mults), S, 3), np.int64)
    seg_correct = np.zeros((len(mults), S, n_seg), np.int64)
    steps_per_seg = max(1, num_steps // n_seg)
    t0 = time.time()
    for i, (x, y) in enumerate(st.prefetch_batches(streams, num_steps, args.decode_workers)):
        student, teacher, vel, keys, counts = step(student, teacher, vel, keys, sm.put(x), sm.put(y))
        c = sm.gather(counts)
        totals += c
        seg_correct[:, :, min(i // steps_per_seg, n_seg - 1)] += c[:, :, 0]
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == num_steps:
            n = (i + 1) * args.batch
            corrupt = np.array([not g.startswith("imagenet_val") for g, _ in specs])
            err = 100 * (1 - totals[:, corrupt, 0].sum(-1) / (n * corrupt.sum()))
            log(f"step {i + 1}/{num_steps} {(time.time() - t0) / (i + 1):.1f}s/step  error by multiplier "
                f"{np.round(err, 1).tolist()}  augmented {100 * totals[0, :, 2].sum() / ((i + 1) * S):.0f}% of batches")

    if lead:
        images = num_steps * args.batch
        out = {"config": vars(args), "set": args.set, "images_per_stream": images,
               "streams": [{"group": g, "order": o} for g, o in specs], "multipliers": mults.tolist(),
               "error": {"cotta": (100 * (1 - totals[:, :, 0] / images)).tolist()},
               "error_by_position": {"cotta": (100 * (1 - seg_correct / (steps_per_seg * args.batch))).mean(1).tolist()}
               if n_seg > 1 else None,
               "continual_orders": [o.tolist() for o in orders] if orders is not None else None,
               "skipped_updates": {"cotta": totals[:, :, 1].tolist()},
               "augmented_batches": {"cotta": totals[:, :, 2].tolist()}}
        st.write_output(args.out, "results.json", json.dumps(out).encode())
        log("done")


if __name__ == "__main__":
    main()
