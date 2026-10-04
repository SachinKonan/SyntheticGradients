"""Gate 2: can cheap feedback predict Tent's error signals through depth?

Fit phase. The 4 held-out ImageNet-C corruptions (severities 1-5) on even val
image ids. Exact Tent runs on every stream; meanwhile DFA is fit by ridge
regression and the depth recurrence is trained by regression onto the true
signals, free-running from the exact seed, loss normalized per tap.

Test phase. The 15 test corruptions at severity 5, and clean val, on odd val
image ids (no image is shared with the fit phase). Exact Tent runs again, and at
every step each predictor is scored against the true signals at the 20
residual taps (bn3 of every block, downsample BN of projection blocks):

  cos_delta, ratio_delta   cosine and norm ratio of delta itself
  cos_grad,  ratio_grad    the same for the BN (scale, bias) gradient it induces

Scores are averaged per quarter of each stream, to show drift as Tent moves the
weights. The states are those reached by exact Tent; running Tent on predicted
signals is the next gate.

Predictors:
  shortcut    exact shortcut path from the exact seed, branch term dropped
  dfa_random  one fixed random projection of the logit error per tap
  dfa_fit     the same projection, ridge-fit on the fit phase
  recurrence  shortcut path + learned low-rank correction (sg.feedback)
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

METHODS = ("shortcut", "dfa_random", "dfa_fit", "recurrence")
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


# ----------------------------------------------------------------------------- per-stream steps

def exact(params, stats, x):
    """Forward + one backward: true deltas at every BN, plus what predictors may read."""
    probes = jax.tree.map(lambda sd: jnp.zeros(sd.shape, sd.dtype), resnet.probe_shapes(params, stats, x))

    def loss(probes):
        logits, x_hats, block_io = resnet.apply(
            params, stats, x, batch_stats=True, probes=probes, return_blocks=True)
        return tent.entropy(logits), (logits, x_hats, block_io)

    deltas, (logits, x_hats, block_io) = jax.grad(loss, has_aux=True)(probes)
    d_top, e = feedback.seed(params, block_io[-1][1], tent.entropy)
    return logits, deltas, x_hats, block_io, d_top, e


def rec_loss(rec, params, stats, block_io, d_top, true):
    pred = feedback.backward_over_depth(params, stats, block_io, d_top, rec=rec)
    per_tap = [jnp.sum(jnp.square(pred[t] - true[t])) / (jnp.sum(jnp.square(true[t])) + 1e-30)
               for t in feedback.residual_taps()]
    return jnp.mean(jnp.stack(per_tap)), pred


def fit_stream(bn, velocity, dfa_acc, x_uint8, y, params, stats, rec, *, lr, momentum):
    p = {**params, **bn}
    logits, deltas, x_hats, block_io, d_top, e = exact(p, stats, imagenet.normalize(x_uint8))
    true = {t: deltas[t] for t in feedback.residual_taps()}
    (loss, pred), grad = jax.value_and_grad(rec_loss, has_aux=True)(rec, p, stats, block_io, d_top, true)
    dfa_acc = jax.tree.map(jnp.add, dfa_acc, feedback.dfa_stats(e, true))
    scores = feedback.compare(pred, true, x_hats)
    bn, velocity = tent.sgd_momentum(bn, resnet.bn_grads(deltas, x_hats), velocity, lr, momentum)
    return bn, velocity, dfa_acc, loss, grad, scores, jnp.sum(logits.argmax(-1) == y)


def test_stream(bn, velocity, acc, x_uint8, y, quarter, params, stats, rec, w_fit, w_rand, *, lr, momentum):
    p = {**params, **bn}
    logits, deltas, x_hats, block_io, d_top, e = exact(p, stats, imagenet.normalize(x_uint8))
    true = {t: deltas[t] for t in feedback.residual_taps()}
    preds = (
        feedback.backward_over_depth(p, stats, block_io, d_top, branch_term="none"),
        feedback.dfa_predict(w_rand, e, block_io),
        feedback.dfa_predict(w_fit, e, block_io),
        feedback.backward_over_depth(p, stats, block_io, d_top, rec=rec),
    )
    scores = jnp.stack([feedback.compare(pr, true, x_hats) for pr in preds])  # (methods, taps, metrics)
    acc = {"scores": acc["scores"].at[quarter].add(scores), "count": acc["count"].at[quarter].add(1),
           "correct": acc["correct"] + jnp.sum(logits.argmax(-1) == y)}
    bn, velocity = tent.sgd_momentum(bn, resnet.bn_grads(deltas, x_hats), velocity, lr, momentum)
    return bn, velocity, acc


# ----------------------------------------------------------------------------- optimizer

def adam(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, clip / (norm + 1e-12)), grads)
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
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--no-mix", action="store_true")
    p.add_argument("--rec-lr", type=float, default=1e-3)
    p.add_argument("--fit-passes", type=int, default=2)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=None, help="cap steps per phase (smoke tests)")
    p.add_argument("--streams", type=int, default=None, help="first N streams per phase (smoke tests)")
    args = p.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    fspecs, tspecs = fit_specs()[: args.streams], test_specs()[: args.streams]
    assert len(fspecs) == len(tspecs)
    S = len(fspecs)
    sm = st.StreamMesh(S)
    log(f"{jax.process_count()} hosts, {len(jax.devices())} devices, {S} streams per phase")

    cache = Path(args.local_cache)
    t0 = time.time()
    fit = [st.Stream(st.fetch(args.data_root, fspecs[s][0], cache), fspecs[s][1], args.batch, fit_ids)
           for s in sm.local_ids]
    test = [st.Stream(st.fetch(args.data_root, tspecs[s][0], cache), tspecs[s][1], args.batch, test_ids)
            for s in sm.local_ids]
    weights = st.fetch_file(args.weights, cache)
    log(f"data + weights ready in {time.time() - t0:.0f}s")

    params, stats = sm.replicate(resnet.load_torchvision(weights))
    bn0 = tent.bn_params(params)
    rec = sm.replicate(feedback.init_recurrence(jax.random.key(0), args.rank, mix=not args.no_mix))
    opt = sm.replicate({"t": np.zeros((), np.int32), "m": jax.tree.map(np.zeros_like, rec),
                        "v": jax.tree.map(np.zeros_like, rec)})
    w_rand = sm.replicate(feedback.dfa_random(jax.random.key(1)))
    taps = feedback.residual_taps()
    tent_kw = dict(lr=args.lr, momentum=args.momentum)

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def fresh_tent(bn0):
        bn = jax.tree.map(lambda a: jnp.broadcast_to(a, (S,) + a.shape), bn0)
        return bn, jax.tree.map(jnp.zeros_like, bn)

    # ---------------------------------------------------------------- fit phase
    @functools.partial(jax.jit, donate_argnums=(0, 1, 2, 5, 6),
                       out_shardings=(sm.shard, sm.shard, sm.shard, sm.repl, sm.repl, sm.repl))
    def fit_step(bn, velocity, dfa_acc, x, y, rec, opt):
        bn, velocity, dfa_acc, loss, grad, scores, correct = jax.vmap(
            functools.partial(fit_stream, **tent_kw), in_axes=(0, 0, 0, 0, 0, None, None, None)
        )(bn, velocity, dfa_acc, x, y, params, stats, rec)
        rec, opt, gnorm = adam(rec, jax.tree.map(lambda g: g.mean(0), grad), opt, args.rec_lr)
        info = {"loss": loss.mean(), "grad_norm": gnorm, "scores": scores.mean(0),
                "correct": correct.sum()}
        return bn, velocity, dfa_acc, rec, opt, info

    dfa_shapes = jax.eval_shape(feedback.dfa_stats, jax.ShapeDtypeStruct((args.batch, 1000), jnp.float32),
                                _tap_shapes(args.batch))
    zero_dfa_acc = jax.jit(lambda: jax.tree.map(lambda sd: jnp.zeros((S,) + sd.shape, sd.dtype), dfa_shapes),
                           out_shardings=sm.shard)
    dfa_acc = zero_dfa_acc()
    curve = []
    fit_steps = st.common_steps(fit)
    if args.steps:
        fit_steps = min(fit_steps, args.steps)
    t0 = time.time()
    for pass_ in range(args.fit_passes):
        for s, spec_id in zip(fit, sm.local_ids):  # a new order each pass
            if pass_ > 0:
                s.perm = np.random.default_rng(1000 * pass_ + spec_id).permutation(len(s.records))
        bn, velocity = fresh_tent(bn0)
        for step, (x, y) in enumerate(st.prefetch_batches(fit, fit_steps, args.decode_workers)):
            bn, velocity, dfa_acc, rec, opt, info = fit_step(bn, velocity, dfa_acc, sm.put(x), sm.put(y), rec, opt)
            if step % 50 == 0 or step + 1 == fit_steps:
                info = jax.device_get(info)
                sc = np.asarray(info["scores"])
                row = {"pass": pass_, "step": step, "loss": float(info["loss"]),
                       "grad_norm": float(info["grad_norm"]),
                       "cos_delta_mean": float(sc[:, 0].mean()), "cos_grad_mean": float(sc[:, 2].mean()),
                       "tent_err": 100 * (1 - float(info["correct"]) / (S * args.batch))}
                curve.append(row)
                log(f"fit pass {pass_} step {step + 1}/{fit_steps} "
                    f"{(time.time() - t0) / (pass_ * fit_steps + step + 1):.2f}s/step  "
                    f"loss {row['loss']:.4f}  cos_delta {row['cos_delta_mean']:.3f}  "
                    f"cos_grad {row['cos_grad_mean']:.3f}  tent_err(batch) {row['tent_err']:.1f}")

    w_fit = jax.jit(lambda acc: feedback.dfa_solve(jax.tree.map(lambda a: a.sum(0), acc), args.ridge),
                    out_shardings=sm.repl)(dfa_acc)
    del dfa_acc

    # ---------------------------------------------------------------- test phase
    @functools.partial(jax.jit, donate_argnums=(0, 1, 2), out_shardings=(sm.shard, sm.shard, sm.shard))
    def test_step(bn, velocity, acc, x, y, quarter, rec, w_fit, w_rand):
        return jax.vmap(functools.partial(test_stream, **tent_kw),
                        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None)
                        )(bn, velocity, acc, x, y, quarter, params, stats, rec, w_fit, w_rand)

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def zero_acc():
        return {"scores": jnp.zeros((S, QUARTERS, len(METHODS), len(taps), len(METRICS))),
                "count": jnp.zeros((S, QUARTERS)), "correct": jnp.zeros((S,), jnp.int32)}

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
        write_results(args, tspecs, taps, acc, curve, test_steps)
        buf = io.BytesIO()
        flat, _ = jax.tree_util.tree_flatten_with_path({"rec": rec_np, "dfa_fit": w_fit_np})
        np.savez(buf, **{jax.tree_util.keystr(k): np.asarray(v) for k, v in flat})
        st.write_output(args.out, "predictors.npz", buf.getvalue())


def _tap_shapes(batch):
    """Shapes of the residual-tap deltas for a batch of 224x224 images."""
    sizes = {"layer1": (56, 256), "layer2": (28, 512), "layer3": (14, 1024), "layer4": (7, 2048)}
    out = {}
    for tap in feedback.residual_taps():
        hw, c = sizes[tap.split(".")[0]]
        out[tap] = jax.ShapeDtypeStruct((batch, hw, hw, c), jnp.float32)
    return out


def write_results(args, specs, taps, acc, curve, steps):
    scores, count = acc["scores"], acc["count"]                     # (S, Q, M, T, K), (S, Q)
    corrupt = np.array([g.startswith("imagenet_c/") for g, _ in specs])
    clean = ~corrupt

    def table(mask, q=None):
        """Mean over steps within each stream, then over the selected streams."""
        if not mask.any():
            return None
        qs = slice(None) if q is None else slice(q, q + 1)
        x = scores[mask][:, qs].sum(1) / np.maximum(count[mask][:, qs].sum(1), 1)[:, None, None, None]
        x = x.mean(0)                                               # (M, T, K)
        return {m: {t: dict(zip(METRICS, map(float, x[i, j]))) for j, t in enumerate(taps)}
                for i, m in enumerate(METHODS)}

    def tap_mean(tbl):
        # Mean over taps below the top block (the top block's taps are exact for every method).
        if tbl is None:
            return None
        below = [t for t in taps if not t.startswith("layer4.2")]
        return {m: {k: float(np.mean([tbl[m][t][k] for t in below])) for k in METRICS} for m in METHODS}

    by_group = {}
    for i, (g, _) in enumerate(specs):
        by_group.setdefault(g, []).append(i)
    group_tables = {g: tap_mean(table(np.isin(np.arange(len(specs)), idx))) for g, idx in by_group.items()}
    tent_err = {g: 100 * (1 - acc["correct"][idx].sum() / (len(idx) * steps * args.batch))
                for g, idx in by_group.items()}

    result = {
        "config": vars(args), "taps": taps, "methods": METHODS, "metrics": METRICS,
        "test_corrupt": table(corrupt), "test_clean": table(clean),
        "test_corrupt_by_quarter": [tap_mean(table(corrupt, q)) for q in range(QUARTERS)],
        "summary_corrupt": tap_mean(table(corrupt)), "summary_clean": tap_mean(table(clean)),
        "by_group": group_tables, "tent_err_pct_test_half": tent_err, "fit_curve": curve,
    }
    print("SUMMARY corrupt (mean over taps below layer4.2): " + json.dumps(result["summary_corrupt"]), flush=True)
    tbl = result["test_corrupt"]
    print("cos_grad by tap (corrupt):", flush=True)
    print(f"{'tap':24s}" + "".join(f"{m:>12s}" for m in METHODS), flush=True)
    for t in taps:
        print(f"{t:24s}" + "".join(f"{tbl[m][t]['cos_grad']:12.3f}" for m in METHODS), flush=True)
    st.write_output(args.out, "results.json", json.dumps(result, indent=1).encode())


if __name__ == "__main__":
    main()
