"""Gate 3: test-time adaptation (Tent) driven by predicted gradients.

Every method runs Tent's protocol on the same streams: batch 64, entropy loss,
all 53 BN (scale, bias) updated by SGD with momentum 0.9, BN statistics from the
test batch, reset per stream, and every batch scored before the update on it.
Methods differ only in where the BN gradient comes from:

  tent          exact backprop
  shortcut@k    exact top k blocks, then the shortcut path only
  dfa           one fitted projection of the logit error per BN
  <name>@k      exact top k blocks, then a trained predictor (--predictors)

plus two references without gradients: no_adapt (running BN statistics) and
bn_adapt (test-batch statistics, no update).

The size of a predicted gradient is not calibrated, so each method runs at
every step-size multiplier in --mults (step = Tent's lr x multiplier). Pick the
multiplier on --set val (the fit corruptions on held-out images), then read the
test error at that multiplier from --set test; scripts/gate3_report.py does it.

Streams:
  val   the 4 fit corruptions (all severities) on odd val ids
  test  the 15 test corruptions at severity 5 + clean, on odd val ids
(no predictor was trained on odd ids or on test corruptions).
"""

import argparse
import functools
import io
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from sg import feedback, tent
from sg.data import imagenet
from sg.experiments import streams as st
from sg.experiments.gate2_grid import unflatten
from sg.experiments.gate2_predictability import fit_specs, test_ids, test_specs
from sg.models import resnet


def load_npz(url, cache):
    return dict(np.load(st.fetch_file(url, cache)))


def method_deltas(method, rec, dfa_w, p, stats, sig):
    """BN error signals for one method from the exact signals `sig` (only what it pays for is used)."""
    logits, true, d_stream, x_hats, block_io, stem, e, _ = sig
    if method == "tent":
        return true
    if method == "dfa":
        return feedback.dfa_predict(dfa_w, e, feedback.tap_masks(p, x_hats, block_io, stem))
    name, k = method.split("@")
    back = functools.partial(feedback.backward_over_depth, p, stats, x_hats, block_io, stem, d_stream, true,
                             exact_top=int(k))
    return back() if name == "shortcut" else back(rec=rec)


def main():
    ap = argparse.ArgumentParser()
    st.add_launch_args(ap)
    ap.add_argument("--set", required=True, choices=["val", "test"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.5e-4, help="Tent's step size; scaled by each multiplier")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--mults", default="0.3,1,3,10,30,100")
    ap.add_argument("--predictors", default="A=gs://sk7524-tinker-tpu-us-central2/synthgrad/predictors/rec_k0best.npz,"
                                            "B=gs://sk7524-tinker-tpu-us-central2/synthgrad/predictors/rec_k1best.npz")
    ap.add_argument("--dfa", default="gs://sk7524-tinker-tpu-us-central2/synthgrad/predictors/dfa_fit.npz")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--methods", default="tent,shortcut@0,shortcut@1,shortcut@2,shortcut@4,dfa,"
                                         "A@0,B@0,B@1,B@2,B@4")
    ap.add_argument("--steps", type=int, default=None, help="cap steps (smoke tests)")
    ap.add_argument("--streams", type=int, default=None, help="first N streams (smoke tests)")
    args = ap.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    specs = (fit_specs() if args.set == "val" else test_specs())[: args.streams]
    S = len(specs)
    sm = st.StreamMesh(S)
    mults = np.array([float(m) for m in args.mults.split(",")], np.float32)
    methods = args.methods.split(",")
    traj_shard = NamedSharding(sm.shard.mesh, P(None, "streams"))  # (multipliers, streams, ...)
    log(f"set {args.set}: {S} streams, methods {methods}, multipliers {mults.tolist()}")

    cache = Path(args.local_cache)
    streams = [st.Stream(st.fetch(args.data_root, specs[s][0], cache), specs[s][1], args.batch, test_ids,
                         resize=imagenet.needs_resize(specs[s][0])) for s in sm.local_ids]
    params, stats = sm.replicate(resnet.load_torchvision(st.fetch_file(args.weights, cache)))
    bn0 = tent.bn_params(params)

    template = feedback.init_recurrence(jax.random.key(0), args.rank)
    recs = {}
    for item in args.predictors.split(","):
        name, url = item.split("=")
        recs[name] = sm.replicate(unflatten(template, load_npz(url, cache)))
    dfa_w = sm.replicate(load_npz(args.dfa, cache))

    num_steps = st.common_steps(streams)
    if args.steps:
        num_steps = min(num_steps, args.steps)

    # ------------------------------------------------------------ references without gradients
    @functools.partial(jax.jit, out_shardings=sm.shard)
    def reference_correct(x, y):
        def one(x_s, y_s):
            x_s = imagenet.normalize(x_s)
            src, _ = resnet.apply(params, stats, x_s, batch_stats=False)
            bna, _ = resnet.apply(params, stats, x_s, batch_stats=True)
            return jnp.stack([jnp.sum(src.argmax(-1) == y_s), jnp.sum(bna.argmax(-1) == y_s)])
        return jax.vmap(one)(x, y)

    # ------------------------------------------------------------ one jitted step per method
    def make_step(method):
        rec = recs.get(method.split("@")[0])

        @functools.partial(jax.jit, donate_argnums=(0, 1), out_shardings=(traj_shard, traj_shard, traj_shard))
        def step(bn, velocity, x, y):
            def per_mult(args_):
                bn_m, vel_m, mult = args_

                def per_stream(bn_s, vel_s, x_s, y_s):
                    p = {**params, **bn_s}
                    sig = feedback.exact_signals(p, stats, imagenet.normalize(x_s), tent.entropy)
                    correct = jnp.sum(sig[0].argmax(-1) == y_s)  # scored before the update
                    deltas = method_deltas(method, rec, dfa_w, p, stats, sig)
                    grads = resnet.bn_grads(deltas, sig[3])
                    new_bn, new_vel = tent.sgd_momentum(bn_s, grads, vel_s, args.lr * mult, args.momentum)
                    ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(new_bn)]))
                    keep = lambda a, b: jax.tree.map(lambda u, v: jnp.where(ok, u, v), a, b)
                    return keep(new_bn, bn_s), keep(new_vel, vel_s), jnp.stack([correct, 1 - ok.astype(jnp.int32)])

                return jax.vmap(per_stream)(bn_m, vel_m, x, y)

            bn, velocity, counts = jax.lax.map(per_mult, (bn, velocity, jnp.asarray(mults)))
            return bn, velocity, counts

        return step

    steps = {m: make_step(m) for m in methods}
    fresh = jax.jit(lambda bn0: jax.tree.map(lambda a: jnp.broadcast_to(a, (len(mults), S) + a.shape), bn0),
                    out_shardings=traj_shard)
    state = {m: (fresh(bn0), jax.tree.map(jnp.zeros_like, fresh(bn0))) for m in methods}
    totals = {m: np.zeros((len(mults), S, 2), np.int64) for m in methods}
    ref_total = np.zeros((S, 2), np.int64)

    t0 = time.time()
    for step_i, (x, y) in enumerate(st.prefetch_batches(streams, num_steps, args.decode_workers)):
        xg, yg = sm.put(x), sm.put(y)
        ref_total += sm.gather(reference_correct(xg, yg))
        for m in methods:
            bn, vel = state[m]
            bn, vel, counts = steps[m](bn, vel, xg, yg)
            state[m] = (bn, vel)
            totals[m] += np.asarray(multihost_utils.process_allgather(counts, tiled=True))
        if step_i == 0 or (step_i + 1) % 50 == 0 or step_i + 1 == num_steps:
            n = (step_i + 1) * args.batch
            corrupt = np.array([g.startswith("imagenet_c/") for g, _ in specs])
            err = lambda c: 100 * (1 - c[..., corrupt].sum(-1) / (n * corrupt.sum()))
            ref = 100 * (1 - ref_total[corrupt].sum(0) / (n * corrupt.sum()))
            best = {m: f"{err(totals[m][:, :, 0]).min():.1f}" for m in methods}
            log(f"step {step_i + 1}/{num_steps} {(time.time() - t0) / (step_i + 1):.1f}s/step  "
                f"no_adapt {ref[0]:.1f}  bn_adapt {ref[1]:.1f}  best-multiplier error: {best}")

    if lead:
        images = num_steps * args.batch
        out = {"config": vars(args), "set": args.set, "images_per_stream": images,
               "streams": [{"group": g, "order": o} for g, o in specs], "multipliers": mults.tolist(),
               "reference_error": {"no_adapt": (100 * (1 - ref_total[:, 0] / images)).tolist(),
                                   "bn_adapt": (100 * (1 - ref_total[:, 1] / images)).tolist()},
               "error": {m: (100 * (1 - totals[m][:, :, 0] / images)).tolist() for m in methods},
               "skipped_updates": {m: totals[m][:, :, 1].tolist() for m in methods}}
        st.write_output(args.out, "results.json", json.dumps(out).encode())
        log("done")


if __name__ == "__main__":
    main()
