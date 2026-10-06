"""Gate 3: test-time adaptation (Tent) driven by predicted gradients.

Every method runs Tent's protocol on the same streams: batch 64, entropy loss,
all 53 BN (scale, bias) updated by SGD with momentum 0.9, BN statistics from the
test batch, reset per stream, and every batch scored before the update on it.
Methods differ only in where the BN gradient comes from:

  tent          exact backprop
  shortcut@k    exact top k blocks, then the shortcut path only
  dfa           one fitted projection of the logit error per BN
  lowrank{f}@k  exact top k blocks, then the untrained low-rank backward (rank f x width)
  <name>@k      exact top k blocks, then a trained predictor (--predictors; recurrence,
                GRU or low-rank, read from the .json saved next to each .npz)

plus two references without gradients: no_adapt (running BN statistics) and
bn_adapt (test-batch statistics, no update).

The size of a predicted gradient is not calibrated, so each method runs at
every step-size multiplier in --mults (step = Tent's lr x multiplier). Pick the
multiplier on --set val (the fit corruptions on held-out images), then read the
test error at that multiplier from --set test; scripts/gate3_report.py does it.

Streams:
  val         the 4 fit corruptions (all severities) on odd val ids
  test        the 15 test corruptions at severity 5 + clean, on odd val ids
  continual   the 15 test corruptions back to back with no reset (as in CoTTA),
              --continual-images per corruption, a different random order per stream
  r / sketch / v2   ImageNet-R / -Sketch / -V2 (natural shifts never seen in
              training), all images, a different random order per stream; for
              ImageNet-R, logits (predictions and the entropy loss) are restricted
              to its 200 classes
(no predictor was trained on odd ids or on test corruptions). For continual, error
is also reported per position in the sequence.
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

from sg import feedback, tent, timerule
from sg.data import imagenet
from sg.experiments import streams as st
from sg.experiments.gate2_grid import unflatten
from sg.experiments.gate2_predictability import fit_specs, test_ids, test_specs
from sg.models import resnet


def load_npz(url, cache):
    return dict(np.load(st.fetch_file(url, cache)))


def method_deltas(method, rec, dfa_w, p, stats, sig):
    """BN error signals for one method from the exact signals `sig` (only what it pays for is used).
    rec: the method's predictor (recurrence, GRU or low-rank factors), or None."""
    if method == "tent" or (rec is not None and feedback.is_precond(rec)):
        return sig.deltas
    if method == "dfa":
        return feedback.dfa_predict(dfa_w, sig.e, feedback.tap_masks(p, sig.x_hats, sig.block_io, sig.stem))
    k = int(method.split("@")[1])
    back = functools.partial(feedback.backward_over_depth, p, stats, sig.x_hats, sig.block_io, sig.stem,
                             sig.d_stream, sig.deltas, exact_top=k)
    return back() if rec is None else back(**feedback.predictor_kwargs(rec, sig.inv_stds))


def main():
    ap = argparse.ArgumentParser()
    st.add_launch_args(ap)
    ap.add_argument("--set", required=True, choices=["val", "test", "continual", "r", "sketch", "v2"])
    ap.add_argument("--adapt", default="bn", choices=["bn", "conv1x1"],
                    help="what adapts: the 53 BN (scale, bias), or every 1x1 conv with BN frozen "
                         "(then 'tent' is exact full-backprop fine-tuning of those convs)")
    ap.add_argument("--continual-images", type=int, default=5000, help="images per corruption (continual)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.5e-4, help="Tent's step size; scaled by each multiplier")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--mults", default="0.3,1,3,10,30,100")
    ap.add_argument("--predictors", default="A=gs://sk7524-tinker-tpu-us-central2/synthgrad/predictors/rec_k0best.npz,"
                                            "B=gs://sk7524-tinker-tpu-us-central2/synthgrad/predictors/rec_k1best.npz",
                    help="name=url,...; 'none' for no trained predictors")
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

    shift_group = {"r": "imagenet_r", "sketch": "imagenet_sketch", "v2": "imagenet_v2"}.get(args.set)
    if shift_group:
        specs = [(shift_group, i) for i in range(args.streams or 32)]
    elif args.set == "continual":
        n_streams = args.streams or 32
        orders = [np.random.default_rng([7, i]).permutation(len(imagenet.TEST_CORRUPTIONS)) for i in range(n_streams)]
        specs = [(f"continual/{i}", 0) for i in range(n_streams)]
    else:
        specs = (fit_specs() if args.set == "val" else test_specs())[: args.streams]
    S = len(specs)
    sm = st.StreamMesh(S)
    mults = np.array([float(m) for m in args.mults.split(",")], np.float32)
    methods = args.methods.split(",")
    traj_shard = NamedSharding(sm.shard.mesh, P(None, "streams"))  # (multipliers, streams, ...)
    log(f"set {args.set}: {S} streams, methods {methods}, multipliers {mults.tolist()}")

    cache = Path(args.local_cache)
    n_seg = 1
    if args.set == "continual":
        groups = [f"imagenet_c/{c}/5" for c in imagenet.TEST_CORRUPTIONS]
        steps_each = args.continual_images // args.batch
        # One copy of each corruption's records per host; every stream draws its own order.
        loaded = {g: st.Stream(st.fetch(args.data_root, g, cache), 0, args.batch, test_ids) for g in groups}

        def view(g, seed):
            v = st.Stream.__new__(st.Stream)
            v.records, v.batch, v.resize = loaded[g].records, args.batch, False
            v.perm = np.random.default_rng(seed).permutation(len(v.records))
            v.num_steps = len(v.records) // args.batch
            return v

        streams = [st.ConcatStream([view(groups[c], [s, j]) for j, c in enumerate(orders[s])], steps_each)
                   for s in sm.local_ids]
        n_seg = len(groups)
    elif shift_group:
        shared = st.Stream(st.fetch(args.data_root, shift_group, cache), 0, args.batch, None, resize=True)

        def view(order):
            v = st.Stream.__new__(st.Stream)
            v.records, v.batch, v.resize, v.num_steps = shared.records, args.batch, True, shared.num_steps
            v.perm = np.arange(len(v.records)) if order == 0 else np.random.default_rng(order).permutation(len(v.records))
            return v

        streams = [view(specs[s][1]) for s in sm.local_ids]
    else:
        streams = [st.Stream(st.fetch(args.data_root, specs[s][0], cache), specs[s][1], args.batch, test_ids,
                             resize=imagenet.needs_resize(specs[s][0])) for s in sm.local_ids]

    # ImageNet-R: restrict logits to its 200 classes (predictions and the entropy loss).
    subset_file = (st.fetch(args.data_root, shift_group, cache) / "class_subset.json") if shift_group else None
    if subset_file is not None and subset_file.exists():
        class_mask = np.zeros(1000, bool)
        class_mask[json.loads(subset_file.read_text())] = True
        mask_logits = lambda l: jnp.where(class_mask, l, -1e9)
        log(f"restricting logits to {class_mask.sum()} classes")
    else:
        mask_logits = lambda l: l
    loss_fn = lambda logits: tent.entropy(mask_logits(logits))
    params, stats = sm.replicate(resnet.load_torchvision(st.fetch_file(args.weights, cache)))
    # The adapted parameters ("bn0" below): BN affine params, or the 1x1 conv weights (BN frozen;
    # BN still uses test-batch statistics, as for every method).
    adapted_names = resnet.bn_names() if args.adapt == "bn" else feedback.conv1x1_names()
    bn0 = {n: params[n] for n in adapted_names}

    params_np = resnet.load_torchvision(st.fetch_file(args.weights, cache))[0]
    recs, rules = {}, {}  # rules: name -> (time-rule kind, knobs); momentum otherwise
    for item in [i for i in args.predictors.split(",") if i and i != "none"]:
        name, url = item.split("=")
        meta = json.loads(Path(st.fetch_file(url.removesuffix(".npz") + ".json", cache)).read_text())
        arch = meta.get("arch", "recurrence")
        template = {"recurrence": lambda: feedback.init_recurrence(jax.random.key(0), meta.get("rank", args.rank)),
                    "gru": lambda: feedback.init_gru(jax.random.key(0), meta.get("rank", args.rank)),
                    "lowrank": lambda: feedback.init_lowrank(params_np, meta["lowrank_frac"]),
                    "precond": lambda: feedback.init_precond(params_np)}[arch]()
        recs[name] = sm.replicate(unflatten(template, load_npz(url, cache)))
        kind = meta.get("time_rule", "momentum")
        if kind != "momentum":  # learned time rule saved next to the predictor
            knob_template = timerule.init_knobs(kind, eta={n: 1.0 for n in adapted_names}, names=adapted_names)
            rules[name] = (kind, sm.replicate(unflatten(knob_template, load_npz(
                url.removesuffix("predictor.npz") + "time_rule.npz", cache))))
        log(f"predictor {name}: {arch} {meta.get('config')}, time rule {kind}")
    for m in methods:  # untrained low-rank backward, from the SVD of the real convs
        name = m.split("@")[0]
        if name.startswith("lowrank") and name not in recs:
            recs[name] = sm.replicate(feedback.init_lowrank(params_np, float(name.removeprefix("lowrank"))))
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
            src, bna = mask_logits(src), mask_logits(bna)
            return jnp.stack([jnp.sum(src.argmax(-1) == y_s), jnp.sum(bna.argmax(-1) == y_s)])
        return jax.vmap(one)(x, y)

    # ------------------------------------------------------------ one jitted step per method
    def make_step(method):
        rec = recs.get(method.split("@")[0])
        kind, knobs = rules.get(method.split("@")[0], ("momentum", {}))

        @functools.partial(jax.jit, donate_argnums=(0, 1), out_shardings=(traj_shard, traj_shard, traj_shard))
        def step(bn, velocity, x, y):
            def per_mult(args_):
                bn_m, vel_m, mult = args_

                def per_stream(bn_s, vel_s, x_s, y_s):
                    p = {**params, **bn_s}
                    sig = feedback.exact_signals(p, stats, imagenet.normalize(x_s), loss_fn)
                    correct = jnp.sum(mask_logits(sig[0]).argmax(-1) == y_s)  # scored before the update
                    deltas = method_deltas(method, rec, dfa_w, p, stats, sig)
                    if args.adapt == "bn":
                        grads = resnet.bn_grads(deltas, sig.x_hats)
                    else:
                        grads = feedback.conv1x1_grads(p, sig.x_hats, sig.inv_stds, sig.block_io, deltas)
                    if rec is not None and feedback.is_precond(rec):
                        grads = feedback.apply_precond(rec, grads)
                    new_bn, new_vel = timerule.apply(kind, knobs, vel_s, bn_s, grads, bn0, args.lr, mult)
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
    def fresh_rule(m):
        kind = rules.get(m.split("@")[0], ("momentum", {}))[0]
        init = jax.vmap(jax.vmap(lambda b: timerule.init_state(kind, b)))
        return jax.jit(init, out_shardings=traj_shard)(fresh(bn0))

    state = {m: (fresh(bn0), fresh_rule(m)) for m in methods}
    totals = {m: np.zeros((len(mults), S, 2), np.int64) for m in methods}
    seg_correct = {m: np.zeros((len(mults), S, n_seg), np.int64) for m in methods}  # by position in sequence
    ref_total = np.zeros((S, 2), np.int64)
    steps_per_seg = max(1, num_steps // n_seg)

    t0 = time.time()
    for step_i, (x, y) in enumerate(st.prefetch_batches(streams, num_steps, args.decode_workers)):
        xg, yg = sm.put(x), sm.put(y)
        ref_total += sm.gather(reference_correct(xg, yg))
        for m in methods:
            bn, vel = state[m]
            bn, vel, counts = steps[m](bn, vel, xg, yg)
            state[m] = (bn, vel)
            c = np.asarray(multihost_utils.process_allgather(counts, tiled=True))
            totals[m] += c
            seg_correct[m][:, :, min(step_i // steps_per_seg, n_seg - 1)] += c[:, :, 0]
        if step_i == 0 or (step_i + 1) % 50 == 0 or step_i + 1 == num_steps:
            n = (step_i + 1) * args.batch
            corrupt = np.array([not g.startswith("imagenet_val") for g, _ in specs])
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
               "error_by_position": {m: (100 * (1 - seg_correct[m] / (steps_per_seg * args.batch))).mean(1).tolist()
                                     for m in methods} if n_seg > 1 else None,
               "continual_orders": [o.tolist() for o in orders] if args.set == "continual" else None,
               "skipped_updates": {m: totals[m][:, :, 1].tolist() for m in methods}}
        st.write_output(args.out, "results.json", json.dumps(out).encode())
        log("done")


if __name__ == "__main__":
    main()
