"""Supervised fine-tuning baseline: use the labeled training data on the network itself.

Our predictor is trained with labels on the 4 extra corruptions (even image ids). This
baseline spends the same labeled images on fine-tuning the network (the BN affine params,
the 1x1 convs, or every weight) with cross-entropy, to check whether our gain just comes from having
seen labeled corrupted data. The fine-tuned network is then deployed like the original
(no adaptation, BN-adapt, Tent): gate3_adapt --weights <out>/weights.npz.

Every batch holds 64 images of one corruption/severity, as at deployment, so BN batch
statistics see one shift at a time; batches are drawn at random over all 20 fit groups.
Streams act as data-parallel workers (gradients averaged over them). SGD with momentum
and cosine decay; BN uses batch statistics throughout. After every epoch the network is
scored on the fit corruptions on odd ids (val) with batch statistics, no adaptation.
"""

import argparse
import functools
import io
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sg import feedback
from sg.data import imagenet
from sg.experiments import streams as st
from sg.experiments.gate2_predictability import fit_ids, fit_specs, test_ids
from sg.models import resnet


class PlanStream:
    """This stream's share of an epoch plan: batch i is (group, image indices)."""

    def __init__(self, loaded, plan):
        self.loaded, self.plan = loaded, plan
        self.decoded = True

    def batch_arrays(self, step):
        g, idx = self.plan[step]
        return self.loaded[g].images[idx], self.loaded[g].labels[idx]


def main():
    ap = argparse.ArgumentParser()
    st.add_launch_args(ap)
    ap.add_argument("--params", default="all", choices=["bn", "conv1x1", "all"], help="what is fine-tuned")
    ap.add_argument("--lr", type=float, required=True, help="peak SGD lr (cosine decay to 0)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--streams", type=int, default=None, help="data-parallel workers (default: one per device)")
    ap.add_argument("--max-images", type=int, default=None, help="per group (smoke tests)")
    args = ap.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    S = args.streams or len(jax.devices())
    sm = st.StreamMesh(S)
    cache = Path(args.local_cache)
    groups = sorted({g for g, _ in fit_specs()})
    specs = (fit_specs() * S)[:S]
    loaded = {g: st.Stream(st.fetch(args.data_root, g, cache), 0, args.batch, fit_ids) for g in groups}
    val = [st.Stream(st.fetch(args.data_root, specs[s][0], cache), specs[s][1], args.batch, test_ids)
           for s in sm.local_ids]
    if args.max_images:
        for s in list(loaded.values()) + val:
            s.records = s.records[:args.max_images]
            s.perm, s.num_steps = s.perm[s.perm < len(s.records)], len(s.records) // args.batch
    t = time.time()
    st.decode_in_ram(list(loaded.values()) + val, max(args.decode_workers, 2 * (os.cpu_count() or 1) // 3))
    log(f"decoded into RAM in {time.time() - t:.0f}s; {sum(len(s.records) for s in loaded.values())} training images")

    def epoch_streams(epoch):
        rng = np.random.default_rng([epoch, 99])  # the same plan on every host
        plan = [(g, perm[b * args.batch:(b + 1) * args.batch])
                for g in groups for perm in [rng.permutation(len(loaded[g].records))]
                for b in range(len(loaded[g].records) // args.batch)]
        plan = [plan[i] for i in rng.permutation(len(plan))]
        n = len(plan) // S
        return [PlanStream(loaded, plan[s * n:(s + 1) * n]) for s in sm.local_ids], n

    params, stats = sm.replicate(resnet.load_torchvision(st.fetch_file(args.weights, cache)))
    names = {"bn": resnet.bn_names(), "conv1x1": feedback.conv1x1_names(), "all": sorted(params)}[args.params]
    theta = {n: params[n] for n in names}
    vel = jax.tree.map(jnp.zeros_like, theta)
    steps_per_epoch = epoch_streams(0)[1]
    total = steps_per_epoch * args.epochs
    log(f"fine-tuning {args.params}: {sum(int(np.prod(a.shape)) for a in jax.tree.leaves(theta)):,} weights, "
        f"{S} workers x {args.batch}, {steps_per_epoch} steps/epoch, {args.epochs} epochs, lr {args.lr}")

    def ce_and_correct(theta, x, y):
        logits, _ = resnet.apply({**params, **theta}, stats, imagenet.normalize(x), batch_stats=True)
        ce = jnp.mean(jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, y[:, None], -1)[:, 0])
        return ce, jnp.sum(logits.argmax(-1) == y)

    @functools.partial(jax.jit, donate_argnums=(0, 1), out_shardings=(sm.repl, sm.repl, sm.repl, sm.repl))
    def train_step(theta, vel, x, y, t):
        (ce, correct), g = jax.vmap(jax.value_and_grad(ce_and_correct, has_aux=True),
                                    in_axes=(None, 0, 0))(theta, x, y)
        g = jax.tree.map(lambda a: a.mean(0), g)
        lr = args.lr * 0.5 * (1 + jnp.cos(jnp.pi * t / total))
        vel = jax.tree.map(lambda v, gg: args.momentum * v + gg, vel, g)
        theta = jax.tree.map(lambda w, v: w - lr * v, theta, vel)
        return theta, vel, ce.mean(), correct.sum()

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def eval_step(theta, x, y):
        return jax.vmap(lambda x_s, y_s: ce_and_correct(theta, x_s, y_s)[1])(x, y)

    def evaluate(tag):
        correct = 0
        for x, y in st.prefetch_batches(val, args.eval_steps, args.decode_workers):
            correct += sm.gather(eval_step(theta, sm.put(x), sm.put(y))).sum()
        err = 100 * (1 - correct / (S * args.eval_steps * args.batch))
        history.append({"tag": tag, "val_error": float(err)})
        log(f"EVAL {tag}: val error (batch statistics, no adaptation) {err:.2f}%")

    history = []
    evaluate("start")
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        streams, n = epoch_streams(epoch)
        for i, (x, y) in enumerate(st.prefetch_batches(streams, n, args.decode_workers)):
            theta, vel, ce, correct = train_step(theta, vel, sm.put(x), sm.put(y), jnp.float32(step))
            step += 1
            if i % 50 == 0 or i + 1 == n:
                log(f"epoch {epoch} step {i + 1}/{n} {(time.time() - t0) / step:.2f}s/step  ce {float(ce):.3f}  "
                    f"train err {100 * (1 - float(correct) / (S * args.batch)):.1f}")
        evaluate(f"epoch {epoch + 1}")

    if lead:
        host_params, host_stats = jax.device_get(({**params, **theta}, stats))
        buf = io.BytesIO()
        resnet.save_npz(buf, host_params, host_stats)
        st.write_output(args.out, "weights.npz", buf.getvalue())
        st.write_output(args.out, "results.json", json.dumps({"config": vars(args), "history": history}).encode())
        log("done")


if __name__ == "__main__":
    main()
