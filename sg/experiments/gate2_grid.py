"""Gate 2 grid: k schedule x predictor batch x learning rate x training steps.

Many recurrence predictors are trained side by side on the same data. The
expensive part of a step, exact Tent signals for every stream, is computed once
and shared; each predictor then takes its own loss and update. Data and Tent
protocol are those of gate2_predictability (fit: 4 held-out corruptions on even
val ids; target: full-gradient cosine over all 53 BNs).

Per predictor:
  schedule  k (exact top blocks shown to the predictor) at training step t:
              k0         always 0
              uniform    uniform over 0..15
              cos{K}_{D} K * (1 + cos(pi * min(t / D passes, 1))) / 2, randomly
                         rounded; D is in absolute passes, so one long run
                         gives every training budget from its curve
            predictors with the same schedule share the same k draws
  batch     Tent batches per update: 32 streams x accumulation (1, 2, 4 steps)
  lr        Adam learning rate

Learning curve: after 0, 50, 100, 200 steps and every pass, every frozen
predictor is scored on three fixed sets (fixed batches, fresh exact Tent):
  train       the fit corruptions on the training (even id) images
  new_images  the fit corruptions on held-out (odd id) images
  test        the 15 test corruptions on held-out images (clean reported apart)
as full-gradient cosine / norm ratio at k = --eval-exact-tops, next to
shortcut-only at the same k.

Checkpoints go to <out>/ckpt after every pass; a restarted job resumes there.
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
from sg.experiments.gate2_predictability import fit_ids, fit_specs, rec_loss, test_ids, test_specs
from sg.models import resnet

K_MAX = len(resnet.blocks()) - 1


# ----------------------------------------------------------------------------- schedules

def k_value(schedule: str, t: int, steps_per_pass: int, rng) -> int:
    if schedule == "k0":
        return 0
    if schedule == "uniform":
        return int(rng.integers(0, K_MAX + 1))
    k_start, passes = schedule.removeprefix("cos").split("_")
    frac = min(t / (float(passes) * steps_per_pass), 1.0)
    x = int(k_start) * 0.5 * (1 + np.cos(np.pi * frac))
    return int(np.floor(x) + (rng.random() < x - np.floor(x)))  # random rounding


def draw_ks(schedules, t, steps_per_pass, seed=0):
    """k for every schedule at step t; a pure function of (seed, t), so resumes reproduce it."""
    return [k_value(s, t, steps_per_pass, np.random.default_rng([seed, t, i])) for i, s in enumerate(schedules)]


# ----------------------------------------------------------------------------- optimizer

def masked_adam(params, grads, state, lr, apply, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    """Adam on one predictor; leaves params/state unchanged unless `apply`."""
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    apply = apply & jnp.isfinite(norm)
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, clip / (norm + 1e-12)), grads)
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    step = lr * jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t)
    new = jax.tree.map(lambda p, m, v: p - step * m / (jnp.sqrt(v) + eps), params, m, v)
    keep = lambda a, b: jax.tree.map(lambda x, y: jnp.where(apply, x, y), a, b)
    return keep(new, params), {"t": jnp.where(apply, t, state["t"]), "m": keep(m, state["m"]),
                               "v": keep(v, state["v"])}, norm


# ----------------------------------------------------------------------------- checkpoints

def flatten(tree):
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(k): np.asarray(v) for k, v in flat}


def unflatten(template, arrays):
    flat, treedef = jax.tree_util.tree_flatten_with_path(template)
    return jax.tree_util.tree_unflatten(treedef, [arrays[jax.tree_util.keystr(k)] for k, _ in flat])


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    st.add_launch_args(p)
    p.add_argument("--batch", type=int, default=64, help="Tent batch")
    p.add_argument("--lr", type=float, default=2.5e-4, help="Tent")
    p.add_argument("--momentum", type=float, default=0.9, help="Tent")
    p.add_argument("--schedules", default="k0,uniform")
    p.add_argument("--accums", default="1,2,4", help="predictor batch = devices x accum Tent batches")
    p.add_argument("--rec-lrs", default="3e-4,1e-3,3e-3")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--target", default="full_cos", choices=["delta", "tap_cos", "full_cos", "full_mse"])
    p.add_argument("--passes", type=int, default=16)
    p.add_argument("--eval-steps", type=int, default=25)
    p.add_argument("--eval-exact-tops", default="0,1,2,4")
    p.add_argument("--steps", type=int, default=None, help="cap steps per pass (smoke tests)")
    p.add_argument("--streams", type=int, default=None, help="first N streams (smoke tests)")
    args = p.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    schedules = args.schedules.split(",")
    accums = [int(a) for a in args.accums.split(",")]
    rec_lrs = [float(x) for x in args.rec_lrs.split(",")]
    configs = [{"schedule": s, "accum": a, "rec_lr": lr} for s in schedules for a in accums for lr in rec_lrs]
    NP = len(configs)
    sched_idx = np.array([schedules.index(c["schedule"]) for c in configs])
    accum = np.array([c["accum"] for c in configs])
    lrs = np.array([c["rec_lr"] for c in configs], np.float32)
    eval_tops = [int(k) for k in args.eval_exact_tops.split(",")]

    fspecs, tspecs = fit_specs()[: args.streams], test_specs()[: args.streams]
    S = len(fspecs)
    sm = st.StreamMesh(S)
    pred_shard = NamedSharding(sm.shard.mesh, P(None, "streams"))  # (predictors, streams, ...)
    for c in configs:
        c["predictor_batch"] = S * c["accum"]
    log(f"{jax.process_count()} hosts, {len(jax.devices())} devices, {S} streams, {NP} predictors: "
        f"schedules {schedules} x accum {accums} x lr {rec_lrs}; rank {args.rank}, target {args.target}")

    cache = Path(args.local_cache)
    t0 = time.time()

    def stream(spec, keep):
        group, order = spec
        return st.Stream(st.fetch(args.data_root, group, cache), order, args.batch, keep,
                         resize=imagenet.needs_resize(group))

    fit = [stream(fspecs[s], fit_ids) for s in sm.local_ids]
    eval_sets = {
        "train": [stream(fspecs[s], fit_ids) for s in sm.local_ids],
        "new_images": [stream(fspecs[s], test_ids) for s in sm.local_ids],
        "test": [stream(tspecs[s], test_ids) for s in sm.local_ids],
    }
    eval_groups = {"train": [g for g, _ in fspecs], "new_images": [g for g, _ in fspecs],
                   "test": [g for g, _ in tspecs]}
    weights = st.fetch_file(args.weights, cache)
    log(f"data + weights ready in {time.time() - t0:.0f}s")

    params, stats = sm.replicate(resnet.load_torchvision(weights))
    bn0 = tent.bn_params(params)
    tent_kw = dict(lr=args.lr, momentum=args.momentum)

    rec_init = feedback.init_recurrence(jax.random.key(0), args.rank)  # same init for every predictor
    stack = lambda t: jax.tree.map(lambda a: np.broadcast_to(np.asarray(a), (NP,) + np.shape(a)).copy(), t)
    state = {"recs": stack(rec_init), "opt": {"t": np.zeros((NP,), np.int32), "m": stack(jax.tree.map(np.zeros_like, rec_init)),
                                               "v": stack(jax.tree.map(np.zeros_like, rec_init))},
             "gacc": stack(jax.tree.map(np.zeros_like, rec_init))}

    steps_per_pass = st.common_steps(fit)
    if args.steps:
        steps_per_pass = min(steps_per_pass, args.steps)
    total = args.passes * steps_per_pass
    eval_at = {50, 100, 200} | {(q + 1) * steps_per_pass for q in range(args.passes)}  # plus 0, before training

    # ---------------------------------------------------------------- resume
    curve, start_pass = [], 0
    meta = st.read_output(args.out, "ckpt/meta.json")
    if meta is not None:
        meta = json.loads(meta)
        arrays = dict(np.load(io.BytesIO(st.read_output(args.out, "ckpt/state.npz"))))
        state = unflatten(state, arrays)
        curve, start_pass = meta["curve"], meta["passes_done"]
        log(f"resumed from checkpoint after pass {start_pass}")
    state = sm.replicate(state)

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def fresh_tent(bn0):
        bn = jax.tree.map(lambda a: jnp.broadcast_to(a, (S,) + a.shape), bn0)
        return bn, jax.tree.map(jnp.zeros_like, bn)

    def exact_all(bn, x_uint8):
        """Exact signals for every stream (vmapped), and the exact Tent update."""
        def one(bn_s, x_s):
            ps = {**params, **bn_s}
            _, true, d_stream, x_hats, block_io, stem, _ = feedback.exact_signals(
                ps, stats, imagenet.normalize(x_s), tent.entropy)
            return ps, true, d_stream, x_hats, block_io, stem
        return jax.vmap(one)(bn, x_uint8)

    def tent_update(bn, velocity, true, x_hats):
        return jax.vmap(lambda b, v, t, xh: tent.sgd_momentum(b, resnet.bn_grads(t, xh), v, **tent_kw))(
            bn, velocity, true, x_hats)

    # ---------------------------------------------------------------- train step
    @functools.partial(jax.jit, donate_argnums=(0, 1, 4), out_shardings=(sm.shard, sm.shard, sm.repl, sm.repl))
    def train_step(bn, velocity, x, ks, state, apply):
        ps, true, d_stream, x_hats, block_io, stem = exact_all(bn, x)
        sg_ = jax.lax.stop_gradient
        ps, true, d_stream, x_hats, block_io, stem = map(sg_, (ps, true, d_stream, x_hats, block_io, stem))

        def per_predictor(args_):
            rec, k = args_
            def per_stream(ps_s, true_s, d_s, xh_s, io_s, stem_s):
                (loss, _), g = jax.value_and_grad(rec_loss, has_aux=True)(
                    rec, ps_s, stats, (xh_s, io_s, stem_s), d_s, k, true_s, args.target)
                return loss, g
            loss, g = jax.vmap(per_stream)(ps, true, d_stream, x_hats, block_io, stem)
            return loss.mean(), jax.tree.map(lambda a: a.mean(0), g)

        losses, grads = jax.lax.map(per_predictor, (state["recs"], ks))
        gacc = jax.tree.map(jnp.add, state["gacc"], grads)
        g_use = jax.tree.map(lambda a: a / jnp.asarray(accum, a.dtype).reshape((-1,) + (1,) * (a.ndim - 1)), gacc)
        recs, opt, gnorm = jax.vmap(masked_adam)(state["recs"], g_use, state["opt"], jnp.asarray(lrs), apply)
        gacc = jax.tree.map(lambda a: jnp.where(apply.reshape((-1,) + (1,) * (a.ndim - 1)), 0.0, a), gacc)
        bn, velocity = tent_update(bn, velocity, true, x_hats)
        return bn, velocity, {"recs": recs, "opt": opt, "gacc": gacc}, {"loss": losses, "grad_norm": gnorm}

    # ---------------------------------------------------------------- eval step
    @functools.partial(jax.jit, donate_argnums=(0, 1, 2, 3), out_shardings=(sm.shard, sm.shard, pred_shard, sm.shard))
    def eval_step(bn, velocity, acc, acc_sc, x, recs):
        ps, true, d_stream, x_hats, block_io, stem = exact_all(bn, x)

        def full(rec_or_none, ps_s, true_s, d_s, xh_s, io_s, stem_s):
            back = functools.partial(feedback.backward_over_depth, ps_s, stats, xh_s, io_s, stem_s, d_s, true_s)
            return jnp.stack([feedback.compare(back(exact_top=k, rec=rec_or_none), true_s, xh_s)[1]
                              for k in eval_tops])  # (K, 2)

        per_pred = jax.lax.map(lambda rec: jax.vmap(functools.partial(full, rec))(
            ps, true, d_stream, x_hats, block_io, stem), recs)      # (NP, S, K, 2)
        shortcut = jax.vmap(functools.partial(full, None))(ps, true, d_stream, x_hats, block_io, stem)
        bn, velocity = tent_update(bn, velocity, true, x_hats)
        return bn, velocity, acc + per_pred, acc_sc + shortcut

    zero_eval = jax.jit(lambda: (jnp.zeros((NP, S, len(eval_tops), 2)), jnp.zeros((S, len(eval_tops), 2))),
                        out_shardings=(pred_shard, sm.shard))

    def run_evals(step_done, n_steps, tag="curve"):
        rows = []
        for name, streams in eval_sets.items():
            bn_e, vel_e = fresh_tent(bn0)
            acc, acc_sc = zero_eval()
            for x, _ in st.prefetch_batches(streams, n_steps, args.decode_workers):
                bn_e, vel_e, acc, acc_sc = eval_step(bn_e, vel_e, acc, acc_sc, sm.put(x), state["recs"])
            acc = np.asarray(multihost_utils.process_allgather(acc, tiled=True)) / n_steps
            acc_sc = sm.gather(acc_sc) / n_steps
            corrupt = np.array([g.startswith("imagenet_c/") for g in eval_groups[name]])
            for subset, mask in (("", corrupt), ("_clean", ~corrupt)):
                if not mask.any():
                    continue
                for j in range(NP):
                    m = acc[j][mask].mean(0)
                    rows += [{"kind": tag, "fit_step": step_done, "predictor": j, "set": name + subset,
                              "exact_top": k, "cos": float(m[i, 0]), "ratio": float(m[i, 1])}
                             for i, k in enumerate(eval_tops)]
                m = acc_sc[mask].mean(0)
                rows += [{"kind": tag, "fit_step": step_done, "predictor": "shortcut", "set": name + subset,
                          "exact_top": k, "cos": float(m[i, 0]), "ratio": float(m[i, 1])}
                         for i, k in enumerate(eval_tops)]
        curve.extend(rows)
        k0 = eval_tops[0]
        summary = {s: np.array([r["cos"] for r in rows if r["set"] == s and r["exact_top"] == k0
                                and r["predictor"] != "shortcut"]) for s in ("train", "new_images", "test")}
        log(f"EVAL {tag} after {step_done} steps, cos@k={k0} over predictors (min/median/max) | "
            + " | ".join(f"{s} {v.min():.3f}/{np.median(v):.3f}/{v.max():.3f}" for s, v in summary.items()))

    def save_checkpoint(passes_done):
        host = jax.device_get(state)
        if lead:
            buf = io.BytesIO()
            np.savez(buf, **flatten(host))
            st.write_output(args.out, "ckpt/state.npz", buf.getvalue())
            st.write_output(args.out, "ckpt/meta.json", json.dumps(
                {"passes_done": passes_done, "curve": curve, "configs": configs}).encode())

    # ---------------------------------------------------------------- train
    if start_pass == 0:
        run_evals(0, args.eval_steps)
    t0, done_since = time.time(), 0
    for pass_ in range(start_pass, args.passes):
        for s, spec_id in zip(fit, sm.local_ids):  # a new order each pass, same on resume
            s.perm = (np.arange(len(s.records)) if pass_ == 0 and fspecs[spec_id][1] == 0 else
                      np.random.default_rng([pass_, fspecs[spec_id][1], spec_id]).permutation(len(s.records)))
        bn, velocity = fresh_tent(bn0)
        for step, (x, _) in enumerate(st.prefetch_batches(fit, steps_per_pass, args.decode_workers)):
            t = pass_ * steps_per_pass + step
            ks = np.array(draw_ks(schedules, t, steps_per_pass), np.int32)[sched_idx]
            apply = (t + 1) % accum == 0
            bn, velocity, state, info = train_step(bn, velocity, sm.put(x), ks, state, apply)
            done_since += 1
            if step % 50 == 0 or step + 1 == steps_per_pass:
                loss = np.asarray(info["loss"])
                log(f"pass {pass_} step {step + 1}/{steps_per_pass} "
                    f"{(time.time() - t0) / done_since:.2f}s/step  loss min/median/max "
                    f"{loss.min():.3f}/{np.median(loss):.3f}/{loss.max():.3f}")
            if t + 1 in eval_at:
                run_evals(t + 1, args.eval_steps)
        save_checkpoint(pass_ + 1)

    # ---------------------------------------------------------------- final
    run_evals(total, 4 * args.eval_steps, tag="final")
    if lead:
        result = {"config": vars(args), "predictors": configs, "eval_exact_tops": eval_tops,
                  "steps_per_pass": steps_per_pass, "curve": curve}
        st.write_output(args.out, "results.json", json.dumps(result).encode())
        buf = io.BytesIO()
        np.savez(buf, **flatten(jax.device_get(state["recs"])))
        st.write_output(args.out, "predictors.npz", buf.getvalue())
        log("done")


if __name__ == "__main__":
    main()
