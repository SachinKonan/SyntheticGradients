"""Gate 2: can cheap feedback predict Tent's error signals through depth?

Fit phase. The 4 held-out ImageNet-C corruptions (severities 1-5) on even val
image ids. Exact Tent runs on every stream; meanwhile DFA is fit by ridge
regression and the depth recurrence is trained onto the true signals (--target),
always against the exact gradient of all 53 BNs. Each step draws the number of
exactly backpropagated top blocks k; with --k-schedule anneal, early steps only
use large k (few blocks to predict) and the floor falls to 0 over the first
75% of training. One recurrence serves every hybrid setting.

Test phase. The 15 test corruptions at severity 5, and clean val, on odd val
image ids (no image is shared with the fit phase). Exact Tent runs again, and at
every step each predictor is scored against the true signals at all 53 BNs
that Tent updates:

  per tap   cos / norm ratio of delta, and of the BN (scale, bias) gradient
  full      cos / norm ratio of all 53 BN gradients joined into one vector

Scores are averaged per quarter of each stream, to show drift as Tent moves the
weights. The states are those reached by exact Tent; running Tent on predicted
signals is the next gate.

Predictors:
  dfa_random, dfa_fit   one projection of the logit error per tap
  shortcut@k            exact top k blocks, then the shortcut path only (branch BNs get nothing)
  recurrence@k          exact top k blocks, then shortcut + learned correction
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

from sg import feedback, tent
from sg.data import imagenet
from sg.experiments import streams as st
from sg.models import resnet

METRICS = ("cos_delta", "ratio_delta", "cos_grad", "ratio_grad")
QUARTERS = 4


def fit_specs():
    first = [(f"imagenet_c/{c}/{s}", 0) for c in imagenet.EXTRA_CORRUPTIONS for s in range(1, 6)]
    second = [(f"imagenet_c/{c}/{s}", 1) for c in imagenet.EXTRA_CORRUPTIONS for s in (3, 4, 5)]
    return first + second


def test_specs():
    corrupt = [(f"imagenet_c/{c}/5", o) for o in (0, 1) for c in imagenet.TEST_CORRUPTIONS]
    return corrupt + [("imagenet_val", 0), ("imagenet_val", 1)]


def fit_ids(image_id):
    return image_id % 2 == 0


def test_ids(image_id):
    return image_id % 2 == 1


def method_names(exact_tops):
    return (["dfa_random", "dfa_fit"] + [f"shortcut@{k}" for k in exact_tops]
            + [f"recurrence@{k}" for k in exact_tops])


# ----------------------------------------------------------------------------- losses

def rec_loss(rec, params, stats, fwd, d_stream, exact_top, true, target):
    """Loss of the predicted signals against the exact ones, over all 53 BNs."""
    x_hats, block_io, stem = fwd
    pred = feedback.backward_over_depth(params, stats, x_hats, block_io, stem, d_stream, true,
                                        exact_top=exact_top, rec=rec)
    taps = feedback.all_taps()
    if target == "delta":
        per_tap = [jnp.sum(jnp.square(pred[t] - true[t])) / (jnp.sum(jnp.square(true[t])) + 1e-30) for t in taps]
        return jnp.mean(jnp.stack(per_tap)), pred
    gp, gt = feedback.tap_grads(pred, x_hats), feedback.tap_grads(true, x_hats)
    # jnp.linalg.norm has a NaN gradient at 0 (heads start at zero), so use a safe norm.
    norm = lambda a: jnp.sqrt(jnp.sum(jnp.square(a)) + 1e-36)
    cos = lambda a, b: jnp.dot(a, b) / (norm(a) * norm(b))
    if target == "tap_cos":
        return jnp.mean(jnp.stack([1 - cos(gp[t], gt[t]) for t in taps])), pred
    gp, gt = jnp.concatenate([gp[t] for t in taps]), jnp.concatenate([gt[t] for t in taps])
    if target == "full_cos":
        return 1 - cos(gp, gt), pred
    return jnp.sum(jnp.square(gp - gt)) / (jnp.sum(jnp.square(gt)) + 1e-30), pred  # full_mse


# ----------------------------------------------------------------------------- per-stream steps

def fit_stream(bn, velocity, dfa_acc, x_uint8, y, exact_top, params, stats, rec, *, lr, momentum, target):
    p = {**params, **bn}
    logits, true, d_stream, x_hats, block_io, stem, e, _ = feedback.exact_signals(
        p, stats, imagenet.normalize(x_uint8), tent.entropy)
    (loss, pred), grad = jax.value_and_grad(rec_loss, has_aux=True)(
        rec, p, stats, (x_hats, block_io, stem), d_stream, exact_top, true, target)
    masks = feedback.tap_masks(p, x_hats, block_io, stem)
    dfa_acc = jax.tree.map(jnp.add, dfa_acc, feedback.dfa_stats(e, true, masks))
    _, full = feedback.compare(pred, true, x_hats)
    bn, velocity = tent.sgd_momentum(bn, resnet.bn_grads(true, x_hats), velocity, lr, momentum)
    return bn, velocity, dfa_acc, loss, grad, full, jnp.sum(logits.argmax(-1) == y)


def test_stream(bn, velocity, acc, x_uint8, y, quarter, params, stats, rec, w_fit, w_rand,
                *, lr, momentum, exact_tops):
    p = {**params, **bn}
    logits, true, d_stream, x_hats, block_io, stem, e, _ = feedback.exact_signals(
        p, stats, imagenet.normalize(x_uint8), tent.entropy)
    masks = feedback.tap_masks(p, x_hats, block_io, stem)
    back = functools.partial(feedback.backward_over_depth, p, stats, x_hats, block_io, stem, d_stream, true)
    preds = [feedback.dfa_predict(w_rand, e, masks), feedback.dfa_predict(w_fit, e, masks)]
    preds += [back(exact_top=k) for k in exact_tops]
    preds += [back(exact_top=k, rec=rec) for k in exact_tops]
    per_tap, full = zip(*[feedback.compare(pr, true, x_hats) for pr in preds])
    grads = resnet.bn_grads(true, x_hats)
    gnorm2 = jnp.stack([jnp.sum(jnp.square(grads[n]["scale"])) + jnp.sum(jnp.square(grads[n]["bias"]))
                        for n in resnet.bn_names()])
    acc = {"scores": acc["scores"].at[quarter].add(jnp.stack(per_tap)),
           "full": acc["full"].at[quarter].add(jnp.stack(full)),
           "count": acc["count"].at[quarter].add(1),
           "gnorm2": acc["gnorm2"] + gnorm2,
           "correct": acc["correct"] + jnp.sum(logits.argmax(-1) == y)}
    bn, velocity = tent.sgd_momentum(bn, grads, velocity, lr, momentum)
    return bn, velocity, acc


# ----------------------------------------------------------------------------- optimizer

def adam(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    # A non-finite gradient would poison every parameter through the global clip; skip it.
    ok = jnp.isfinite(norm)
    grads = jax.tree.map(lambda g: jnp.where(ok, g * jnp.minimum(1.0, clip / (norm + 1e-12)), 0.0), grads)
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    step = lr * jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t)
    params = jax.tree.map(lambda p, m, v: p - step * m / (jnp.sqrt(v) + eps), params, m, v)
    return params, {"t": t, "m": m, "v": v}, norm


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    st.add_launch_args(p)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.5e-4, help="Tent")
    p.add_argument("--momentum", type=float, default=0.9, help="Tent")
    p.add_argument("--target", default="full_cos", choices=["delta", "tap_cos", "full_cos", "full_mse"])
    p.add_argument("--k-schedule", default="anneal", choices=["anneal", "uniform", "slide"],
                   help="training draws of k (exact top blocks): uniform over 0..15; anneal: uniform over "
                        "[floor, 15] with the floor falling 15 -> 0; slide: a window of 4 values sliding "
                        "from [12, 15] to [0, 3], then k = 0 for the last 25%% of training")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--no-mix", action="store_true")
    p.add_argument("--rec-lr", type=float, default=1e-3)
    p.add_argument("--fit-passes", type=int, default=4)
    p.add_argument("--exact-tops", default="0,1,2,4,8,12")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=None, help="cap steps per phase (smoke tests)")
    p.add_argument("--streams", type=int, default=None, help="first N streams per phase (smoke tests)")
    args = p.parse_args()
    exact_tops = [int(k) for k in args.exact_tops.split(",")]
    methods = method_names(exact_tops)

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    fspecs, tspecs = fit_specs()[: args.streams], test_specs()[: args.streams]
    assert len(fspecs) == len(tspecs)
    S = len(fspecs)
    sm = st.StreamMesh(S)
    log(f"{jax.process_count()} hosts, {len(jax.devices())} devices, {S} streams per phase; "
        f"target {args.target}, rank {args.rank}, k-schedule {args.k_schedule}, exact_tops {exact_tops}")

    cache = Path(args.local_cache)
    t0 = time.time()
    def stream(spec, keep):
        group, order = spec
        return st.Stream(st.fetch(args.data_root, group, cache), order, args.batch, keep,
                         resize=imagenet.needs_resize(group))

    fit = [stream(fspecs[s], fit_ids) for s in sm.local_ids]
    test = [stream(tspecs[s], test_ids) for s in sm.local_ids]
    weights = st.fetch_file(args.weights, cache)
    log(f"data + weights ready in {time.time() - t0:.0f}s")

    params, stats = sm.replicate(resnet.load_torchvision(weights))
    bn0 = tent.bn_params(params)
    rec = sm.replicate(feedback.init_recurrence(jax.random.key(0), args.rank, mix=not args.no_mix))
    opt = sm.replicate({"t": np.zeros((), np.int32), "m": jax.tree.map(np.zeros_like, rec),
                        "v": jax.tree.map(np.zeros_like, rec)})
    w_rand = sm.replicate(feedback.dfa_random(jax.random.key(1)))
    taps = feedback.all_taps()
    tent_kw = dict(lr=args.lr, momentum=args.momentum)

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def fresh_tent(bn0):
        bn = jax.tree.map(lambda a: jnp.broadcast_to(a, (S,) + a.shape), bn0)
        return bn, jax.tree.map(jnp.zeros_like, bn)

    # ---------------------------------------------------------------- fit phase
    @functools.partial(jax.jit, donate_argnums=(0, 1, 2, 6, 7),
                       out_shardings=(sm.shard, sm.shard, sm.shard, sm.repl, sm.repl, sm.repl))
    def fit_step(bn, velocity, dfa_acc, x, y, exact_top, rec, opt):
        bn, velocity, dfa_acc, loss, grad, full, correct = jax.vmap(
            functools.partial(fit_stream, target=args.target, **tent_kw),
            in_axes=(0, 0, 0, 0, 0, None, None, None, None),
        )(bn, velocity, dfa_acc, x, y, exact_top, params, stats, rec)
        rec, opt, gnorm = adam(rec, jax.tree.map(lambda g: g.mean(0), grad), opt, args.rec_lr)
        info = {"loss": loss.mean(), "grad_norm": gnorm, "full_cos": full[:, 0].mean(), "correct": correct.sum()}
        return bn, velocity, dfa_acc, rec, opt, info

    tap_shapes = _tap_shapes(args.batch)
    dfa_shapes = jax.eval_shape(feedback.dfa_stats, jax.ShapeDtypeStruct((args.batch, 1000), jnp.float32),
                                tap_shapes, {t: jax.ShapeDtypeStruct(sd.shape, bool) for t, sd in tap_shapes.items()})
    zero_dfa_acc = jax.jit(lambda: jax.tree.map(lambda sd: jnp.zeros((S,) + sd.shape, sd.dtype), dfa_shapes),
                           out_shardings=sm.shard)
    dfa_acc = zero_dfa_acc()
    curve = []
    fit_steps = st.common_steps(fit)
    if args.steps:
        fit_steps = min(fit_steps, args.steps)
    k_rng = np.random.default_rng(0)  # same draws on every host
    k_max = len(resnet.blocks()) - 1
    total_steps = args.fit_passes * fit_steps

    def draw_k(t):
        return np.int32(draw_k_schedule(args.k_schedule, t, total_steps, k_max, k_rng))

    t0 = time.time()
    for pass_ in range(args.fit_passes):
        for s, spec_id in zip(fit, sm.local_ids):  # a new order each pass
            if pass_ > 0:
                s.perm = np.random.default_rng(1000 * pass_ + spec_id).permutation(len(s.records))
        bn, velocity = fresh_tent(bn0)
        for step, (x, y) in enumerate(st.prefetch_batches(fit, fit_steps, args.decode_workers)):
            k = draw_k(pass_ * fit_steps + step)
            bn, velocity, dfa_acc, rec, opt, info = fit_step(bn, velocity, dfa_acc, sm.put(x), sm.put(y), k, rec, opt)
            if step % 50 == 0 or step + 1 == fit_steps:
                info = jax.device_get(info)
                row = {"pass": pass_, "step": step, "exact_top": int(k), "loss": float(info["loss"]),
                       "grad_norm": float(info["grad_norm"]), "full_cos": float(info["full_cos"]),
                       "tent_err": 100 * (1 - float(info["correct"]) / (S * args.batch))}
                curve.append(row)
                log(f"fit pass {pass_} step {step + 1}/{fit_steps} "
                    f"{(time.time() - t0) / (pass_ * fit_steps + step + 1):.2f}s/step  k={int(k)}  "
                    f"loss {row['loss']:.4f}  full_cos {row['full_cos']:.3f}  tent_err(batch) {row['tent_err']:.1f}")

    w_fit = jax.jit(lambda acc: feedback.dfa_solve(jax.tree.map(lambda a: a.sum(0), acc), args.ridge),
                    out_shardings=sm.repl)(dfa_acc)
    del dfa_acc

    # ---------------------------------------------------------------- test phase
    @functools.partial(jax.jit, donate_argnums=(0, 1, 2), out_shardings=(sm.shard, sm.shard, sm.shard))
    def test_step(bn, velocity, acc, x, y, quarter, rec, w_fit, w_rand):
        return jax.vmap(functools.partial(test_stream, exact_tops=exact_tops, **tent_kw),
                        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None)
                        )(bn, velocity, acc, x, y, quarter, params, stats, rec, w_fit, w_rand)

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def zero_acc():
        M = len(methods)
        return {"scores": jnp.zeros((S, QUARTERS, M, len(taps), len(METRICS))),
                "full": jnp.zeros((S, QUARTERS, M, 2)), "count": jnp.zeros((S, QUARTERS)),
                "gnorm2": jnp.zeros((S, len(resnet.bn_names()))), "correct": jnp.zeros((S,), jnp.int32)}

    test_steps = st.common_steps(test)
    if args.steps:
        test_steps = min(test_steps, args.steps)
    bn, velocity = fresh_tent(bn0)
    acc = zero_acc()
    t0 = time.time()
    for step, (x, y) in enumerate(st.prefetch_batches(test, test_steps, args.decode_workers)):
        quarter = np.int32(step * QUARTERS // test_steps)
        bn, velocity, acc = test_step(bn, velocity, acc, sm.put(x), sm.put(y), quarter, rec, w_fit, w_rand)
        if step % 100 == 0 or step + 1 == test_steps:
            log(f"test step {step + 1}/{test_steps}  {(time.time() - t0) / (step + 1):.2f}s/step")

    acc = {k: sm.gather(v) for k, v in acc.items()}
    rec_np, w_fit_np = jax.device_get(rec), jax.device_get(w_fit)
    if lead:
        write_results(args, tspecs, taps, methods, exact_tops, acc, curve, test_steps)
        buf = io.BytesIO()
        flat, _ = jax.tree_util.tree_flatten_with_path({"rec": rec_np, "dfa_fit": w_fit_np})
        np.savez(buf, **{jax.tree_util.keystr(k): np.asarray(v) for k, v in flat})
        st.write_output(args.out, "predictors.npz", buf.getvalue())


def draw_k_schedule(schedule, t, total_steps, k_max, rng):
    """Number of exactly backpropagated top blocks for training step t."""
    progress = max(0.0, 1 - t / (0.75 * total_steps))  # 1 -> 0 over the first 75% of training
    if schedule == "uniform":
        return rng.integers(0, k_max + 1)
    if schedule == "anneal":
        return rng.integers(round(k_max * progress), k_max + 1)
    top = round(k_max * progress)  # slide
    return rng.integers(max(0, top - 3), top + 1)


def _tap_shapes(batch):
    """Shapes of every BN's delta for a batch of 224x224 images."""
    x = jax.ShapeDtypeStruct((batch, 224, 224, 3), jnp.float32)
    params, stats = jax.eval_shape(lambda: resnet_shapes())
    _, x_hats = jax.eval_shape(lambda p, s, x: resnet.apply(p, s, x, batch_stats=True), params, stats, x)
    return x_hats


def resnet_shapes():
    """Zero ResNet-50 params/stats with the right shapes (for shape inference only)."""
    widths, params, stats = feedback.tap_widths(), {}, {}
    for name, c in widths.items():
        params[name] = {"scale": jnp.zeros(c), "bias": jnp.zeros(c)}
        stats[name] = {"mean": jnp.zeros(c), "var": jnp.ones(c)}
    params["conv1"] = {"w": jnp.zeros((7, 7, 3, 64))}
    for (pre, _, projection), c_in, c_out in zip(resnet.blocks(), feedback.block_in_widths(),
                                                 feedback.block_out_widths()):
        mid = c_out // resnet.EXPANSION
        params[f"{pre}.conv1"] = {"w": jnp.zeros((1, 1, c_in, mid))}
        params[f"{pre}.conv2"] = {"w": jnp.zeros((3, 3, mid, mid))}
        params[f"{pre}.conv3"] = {"w": jnp.zeros((1, 1, mid, c_out))}
        if projection:
            params[f"{pre}.downsample.0"] = {"w": jnp.zeros((1, 1, c_in, c_out))}
    params["fc"] = {"w": jnp.zeros((1000, 2048)), "b": jnp.zeros(1000)}
    return params, stats


def write_results(args, specs, taps, methods, exact_tops, acc, curve, steps):
    corrupt = np.array([g.startswith("imagenet_c/") for g, _ in specs])
    clean = ~corrupt

    def mean_over(arr, mask, q=None):
        """Mean over steps within each stream, then over the selected streams."""
        qs = slice(None) if q is None else slice(q, q + 1)
        cnt = acc["count"][mask][:, qs].sum(1)
        x = arr[mask][:, qs].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (x / cnt.reshape((-1,) + (1,) * (x.ndim - 1))).mean(0)  # NaN where no steps

    def per_tap_table(mask):
        x = mean_over(acc["scores"], mask)                           # (M, T, K)
        return {m: {t: dict(zip(METRICS, map(float, x[i, j]))) for j, t in enumerate(taps)}
                for i, m in enumerate(methods)}

    def full_table(mask, q=None):
        x = mean_over(acc["full"], mask, q)                           # (M, 2)
        return {m: {"cos": float(x[i, 0]), "ratio": float(x[i, 1])} for i, m in enumerate(methods)}

    mix = not args.no_mix
    cost = {"dfa": feedback.method_cost("dfa", 0),
            **{f"shortcut@{k}": feedback.method_cost("shortcut", k) for k in exact_tops},
            **{f"recurrence@{k}": feedback.method_cost("recurrence", k, args.rank, mix) for k in exact_tops}}

    gnorm2 = acc["gnorm2"][corrupt].sum(0)
    share = gnorm2 / gnorm2.sum()
    grad_share = {name: float(v) for name, v in zip(resnet.bn_names(), share)}
    residual_share = float(sum(grad_share[t] for t in feedback.residual_taps()))

    by_group = {}
    for i, (g, _) in enumerate(specs):
        by_group.setdefault(g, []).append(i)
    group_full = {g: full_table(np.isin(np.arange(len(specs)), idx)) for g, idx in by_group.items()}
    tent_err = {g: 100 * (1 - acc["correct"][idx].sum() / (len(idx) * steps * args.batch))
                for g, idx in by_group.items()}

    result = {
        "config": vars(args), "taps": taps, "methods": methods, "metrics": METRICS,
        "cost_fraction_of_exact_backward": cost,
        "full_corrupt": full_table(corrupt), "full_clean": full_table(clean) if clean.any() else None,
        "full_corrupt_by_quarter": [full_table(corrupt, q) for q in range(QUARTERS)],
        "per_tap_corrupt": per_tap_table(corrupt),
        "grad_norm_share": grad_share, "grad_norm_share_residual_taps": residual_share,
        "full_by_group": group_full, "tent_err_pct_test_half": tent_err, "fit_curve": curve,
    }
    fc = result["full_corrupt"]
    print(f"FULL-GRADIENT COSINE (corrupt), target={args.target} rank={args.rank}", flush=True)
    print(f"  dfa_random {fc['dfa_random']['cos']:.3f}   dfa_fit {fc['dfa_fit']['cos']:.3f}"
          f"   (cost {cost['dfa']:.3f} of the exact backward)", flush=True)
    print(f"  {'exact top k':>12s} {'shortcut':>9s} {'cost':>6s} {'recurrence':>11s} {'cost':>6s}", flush=True)
    for k in exact_tops:
        print(f"  {k:12d} {fc[f'shortcut@{k}']['cos']:9.3f} {cost[f'shortcut@{k}']:6.2f}"
              f" {fc[f'recurrence@{k}']['cos']:11.3f} {cost[f'recurrence@{k}']:6.2f}", flush=True)
    print(f"residual taps hold {100 * residual_share:.1f}% of the true BN gradient norm^2", flush=True)
    k0 = min(exact_tops)
    print(f"per-stage cos_grad (recurrence@{k0} | shortcut@{k0}), mean over taps:", flush=True)
    pt = result["per_tap_corrupt"]
    for stage in ("bn1", "layer1", "layer2", "layer3", "layer4"):
        ts = [t for t in taps if t == stage or t.startswith(stage + ".")]
        r = np.mean([pt[f"recurrence@{k0}"][t]["cos_grad"] for t in ts])
        sc = np.mean([pt[f"shortcut@{k0}"][t]["cos_grad"] for t in ts])
        print(f"  {stage:7s} {r:.3f} | {sc:.3f}", flush=True)
    st.write_output(args.out, "results.json", json.dumps(result, indent=1).encode())


if __name__ == "__main__":
    main()
