"""Step 1 of the architecture study: the low-rank branch backward, untrained.

Each branch conv's transpose is replaced by a rank-r factorization taken from
the SVD of the real weights (r = frac x the block's middle width); ReLU masks
and BN backward stay exact, and the shortcut path stays exact. No training.

Scored like the Gate 2 learning-curve evals: fixed batches, fresh exact Tent,
full-gradient cosine / norm ratio over all 53 BNs, per stage as well, on
  new_images  the fit corruptions on held-out (odd id) images
  test        the 15 test corruptions on held-out images (clean reported apart)
next to shortcut-only and a trained recurrence predictor at the same k.
"""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sg import feedback, tent
from sg.data import imagenet
from sg.experiments import streams as st
from sg.experiments.gate2_grid import unflatten
from sg.experiments.gate2_predictability import fit_specs, test_ids, test_specs
from sg.models import resnet

STAGES = ("bn1", "layer1", "layer2", "layer3", "layer4")


def main():
    ap = argparse.ArgumentParser()
    st.add_launch_args(ap)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.5e-4, help="Tent")
    ap.add_argument("--momentum", type=float, default=0.9, help="Tent")
    ap.add_argument("--fracs", default="0.03125,0.0625,0.125,0.25,0.5")
    ap.add_argument("--exact-tops", default="0,1")
    ap.add_argument("--rec", default="gs://sk7524-tinker-tpu-us-central2/synthgrad/predictors/rec_k1best.npz")
    ap.add_argument("--rank", type=int, default=64, help="rank of --rec")
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--streams", type=int, default=None, help="first N streams (smoke tests)")
    args = ap.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    fracs = [float(f) for f in args.fracs.split(",")]
    tops = [int(k) for k in args.exact_tops.split(",")]
    methods = ([f"lowrank{f:g}@{k}" for f in fracs for k in tops] + [f"shortcut@{k}" for k in tops]
               + [f"recurrence@{k}" for k in tops])
    taps = feedback.all_taps()
    stage_of = np.array([STAGES.index(t if t == "bn1" else t.split(".")[0]) for t in taps])

    fspecs, tspecs = fit_specs()[: args.streams], test_specs()[: args.streams]
    S = len(tspecs)
    sm = st.StreamMesh(S)
    cache = Path(args.local_cache)

    def stream(spec):
        return st.Stream(st.fetch(args.data_root, spec[0], cache), spec[1], args.batch, test_ids,
                         resize=imagenet.needs_resize(spec[0]))

    sets = {"new_images": ([stream(fspecs[s]) for s in sm.local_ids], [g for g, _ in fspecs]),
            "test": ([stream(tspecs[s]) for s in sm.local_ids], [g for g, _ in tspecs])}
    params_np, stats_np = resnet.load_torchvision(st.fetch_file(args.weights, cache))
    lowranks = {f: sm.replicate(feedback.init_lowrank(params_np, f)) for f in fracs}
    params, stats = sm.replicate((params_np, stats_np))
    rec = sm.replicate(unflatten(feedback.init_recurrence(jax.random.key(0), args.rank),
                                 dict(np.load(st.fetch_file(args.rec, cache)))))
    bn0 = tent.bn_params(params)
    log(f"{S} streams; methods {methods}; low-rank cost (fraction of exact branch backward): "
        + ", ".join(f"{f:g}: {feedback.lowrank_cost(f):.3f}" for f in fracs))

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def fresh_tent(bn0):
        bn = jax.tree.map(lambda a: jnp.broadcast_to(a, (S,) + a.shape), bn0)
        return bn, jax.tree.map(jnp.zeros_like, bn)

    @functools.partial(jax.jit, donate_argnums=(0, 1, 2, 3), out_shardings=(sm.shard,) * 4)
    def step(bn, velocity, acc_full, acc_tap, x, lowranks, rec):
        def one(bn_s, vel_s, x_s):
            p = {**params, **bn_s}
            sig = feedback.exact_signals(p, stats, imagenet.normalize(x_s), tent.entropy)
            back = functools.partial(feedback.backward_over_depth, p, stats, sig.x_hats, sig.block_io, sig.stem,
                                     sig.d_stream, sig.deltas, inv_stds=sig.inv_stds)
            preds = ([back(exact_top=k, lowrank=lowranks[f]) for f in fracs for k in tops]
                     + [back(exact_top=k) for k in tops] + [back(exact_top=k, rec=rec) for k in tops])
            per_tap, full = zip(*[feedback.compare(pr, sig.deltas, sig.x_hats) for pr in preds])
            new_bn, new_vel = tent.sgd_momentum(bn_s, resnet.bn_grads(sig.deltas, sig.x_hats), vel_s,
                                                args.lr, args.momentum)
            return new_bn, new_vel, jnp.stack(full), jnp.stack(per_tap)[..., 2]  # per-tap gradient cosine

        new_bn, new_vel, full, per_tap = jax.vmap(one)(bn, velocity, x)
        return new_bn, new_vel, acc_full + full, acc_tap + per_tap

    zeros = jax.jit(lambda: (jnp.zeros((S, len(methods), 2)), jnp.zeros((S, len(methods), len(taps)))),
                    out_shardings=(sm.shard, sm.shard))
    results = {"methods": methods, "fracs": fracs, "exact_tops": tops,
               "lowrank_cost": {f"{f:g}": feedback.lowrank_cost(f) for f in fracs}, "sets": {}}
    t0 = time.time()
    for name, (streams, groups) in sets.items():
        bn, vel = fresh_tent(bn0)
        acc_full, acc_tap = zeros()
        for x, _ in st.prefetch_batches(streams, args.eval_steps, args.decode_workers):
            bn, vel, acc_full, acc_tap = step(bn, vel, acc_full, acc_tap, sm.put(x), lowranks, rec)
        full, tap = sm.gather(acc_full) / args.eval_steps, sm.gather(acc_tap) / args.eval_steps
        corrupt = np.array([g.startswith("imagenet_c/") for g in groups])
        out = {}
        for subset, mask in (("corrupt", corrupt), ("clean", ~corrupt)):
            if not mask.any():
                continue
            f, t = full[mask].mean(0), tap[mask].mean(0)
            out[subset] = {m: {"cos": float(f[i, 0]), "ratio": float(f[i, 1]),
                               "stage_cos": {s: float(t[i][stage_of == j].mean()) for j, s in enumerate(STAGES)}}
                           for i, m in enumerate(methods)}
        results["sets"][name] = out
        log(f"{name} done ({time.time() - t0:.0f}s): " + "  ".join(
            f"{m} {out['corrupt'][m]['cos']:.3f}" for m in methods))

    if lead:
        st.write_output(args.out, "results.json", json.dumps(results, indent=1).encode())
        c = results["sets"]["test"]["corrupt"]
        print("TEST (corrupt) full-gradient cosine [size vs exact]; low-rank cost = fraction of exact branch backward")
        for k in tops:
            print(f"  k={k}: shortcut {c[f'shortcut@{k}']['cos']:.3f}   recurrence {c[f'recurrence@{k}']['cos']:.3f}")
            for f in fracs:
                m = c[f"lowrank{f:g}@{k}"]
                print(f"        low-rank {f:<8g} (cost {feedback.lowrank_cost(f):.3f}): {m['cos']:.3f} [{m['ratio']:.2f}]  "
                      + " ".join(f"{s} {v:.2f}" for s, v in m["stage_cos"].items()))
        log("done")


if __name__ == "__main__":
    main()
