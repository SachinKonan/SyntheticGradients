"""Phase C: train a gradient predictor on the result of adapting with it (first order).

Start from a pretrained predictor. On the training streams, the predictor drives
Tent-style adaptation of the BN parameters exactly as at deployment (entropy
loss, SGD with momentum, step = Tent's lr x a learned multiplier). Every K
adaptation steps, the label cross-entropy of the adapted model on the next batch
is backpropagated into the predictor and the step size only:

  - ResNet's weights and starting BN values are never trained.
  - Everything the predictor reads from ResNet (activations, ReLU patterns, BN
    statistics, the exact error seed and any exact top blocks) is stop-gradient,
    so no second derivatives appear. The outer gradient is
        dL/dphi = sum_t (d update_t / d phi)^T  dL/dBN_final
    where dL/dBN_final is one ordinary backprop of the final batch.
  - A small imitation term (1 - cosine to the exact BN gradient) keeps the
    predictor near its starting point.

Streams run continuously (the adapted BN state is carried from one unroll to the
next, stop-gradient), and reset at the start of every epoch, so the predictor is
trained on the states it reaches during deployment. --fresh instead restarts every
unroll from the original model. --unroll-schedule grows K over training (K for
epoch e is the entry at e x len / epochs). --loss-on all averages the label loss
over every batch after an update (each scored before its own update), as in
learned-optimizer training, instead of only the batch after the last update.

Across batches, --time-rule momentum is Tent's SGD with momentum; filter is a
learned diagonal filter with memory (sg.timerule: running gradient and gradient
size, anchor to the source model, four learned knobs per BN layer, trained here
with the same outer gradient). --freeze-predictor trains only the time rule.
--switching-segment N makes every training stream switch to another fit
corruption/severity every N batches, so the rule sees shifts while it learns.

Data: the 4 fit corruptions on even val ids (as for imitation training). After
every epoch the predictor is scored by deployment on the fit corruptions on odd
ids: online error over --eval-steps batches per stream, at the learned step.
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
from jax.sharding import NamedSharding, PartitionSpec as P

from sg import feedback, tent, timerule
from sg.data import imagenet
from sg.experiments import streams as st
from sg.experiments.gate2_grid import flatten, unflatten
from sg.experiments.gate2_predictability import adam, fit_ids, fit_specs, test_ids
from sg.models import resnet

sg_ = jax.lax.stop_gradient


def load_predictor(url, params_np, cache):
    """A saved predictor, 'svd:<frac>' for the untrained low-rank backward (no pretraining),
    or 'precond' for the exact gradient with a learned per-parameter scale (the Gate 4 control)."""
    if url == "precond":
        return feedback.init_precond(params_np), {"arch": "precond", "config": "exact gradient, learned scale"}
    if url.startswith("svd:"):
        frac = float(url.removeprefix("svd:"))
        return feedback.init_lowrank(params_np, frac), {"arch": "lowrank", "lowrank_frac": frac, "rank": 64,
                                                         "config": "untrained (SVD of the real convs)"}
    meta = json.loads(Path(st.fetch_file(url.removesuffix(".npz") + ".json", cache)).read_text())
    arch = meta.get("arch", "recurrence")
    template = {"recurrence": lambda: feedback.init_recurrence(jax.random.key(0), meta.get("rank", 64)),
                "gru": lambda: feedback.init_gru(jax.random.key(0), meta.get("rank", 64)),
                "lowrank": lambda: feedback.init_lowrank(params_np, meta["lowrank_frac"])}[arch]()
    return unflatten(template, dict(np.load(st.fetch_file(url, cache)))), meta


def adapted_grads(adapt, p, sig, deltas):
    """Gradients of the adapted parameters from error signals at the BNs."""
    if adapt == "bn":
        return resnet.bn_grads(deltas, sig.x_hats)
    return feedback.conv1x1_grads(p, sig.x_hats, sig.inv_stds, sig.block_io, deltas)


def flatten_grads(g, names):
    return jnp.concatenate([jnp.ravel(leaf) for n in names for leaf in jax.tree.leaves(g[n])])


def calibrate(params, stats, bn0, phi, train, sm, args):
    """Per-BN-layer initial filter steps that match momentum SGD at the starting multiplier,
    from the predictor's gradients on the first training batch of every stream."""
    def mean_sq(x_s):
        p = {**params, **bn0}
        sig = feedback.exact_signals(p, stats, imagenet.normalize(x_s), tent.entropy)
        deltas = feedback.backward_over_depth(p, stats, sig.x_hats, sig.block_io, sig.stem, sig.d_stream,
                                              sig.deltas, exact_top=args.exact_top,
                                              **feedback.predictor_kwargs(phi, sig.inv_stds))
        g = adapted_grads(args.adapt, p, sig, deltas)
        return jnp.stack([jnp.mean(jnp.square(flatten_grads(g, [n]))) for n in sorted(bn0)])

    x, _ = next(st.prefetch_batches(train, 1, args.decode_workers))
    ms = sm.gather(jax.jit(jax.vmap(mean_sq, spmd_axis_name="streams"), out_shardings=sm.shard)(sm.put(x, 1))).mean(0)  # (BNs,)
    return {n: args.lr * args.init_mult * float(np.sqrt(v)) / (1 - timerule.MOMENTUM) + 1e-12
            for n, v in zip(sorted(bn0), ms)}


def cosine(a, b):
    return jnp.dot(a, b) / (jnp.sqrt(jnp.sum(a * a) + 1e-36) * jnp.sqrt(jnp.sum(b * b) + 1e-36))


def main():
    ap = argparse.ArgumentParser()
    st.add_launch_args(ap)
    ap.add_argument("--predictor", default="gs://sk7524-tinker-tpu-us-central2/synthgrad/predictors/lr4.npz",
                    help="saved predictor .npz (with .json), or svd:<frac> to start without pretraining")
    ap.add_argument("--exact-top", type=int, default=1, help="exact top blocks, at training and deployment")
    ap.add_argument("--init-mult", type=float, default=3.0, help="starting step multiplier (from deployment tuning)")
    ap.add_argument("--unroll", type=int, default=4, help="K adaptation steps per outer step")
    ap.add_argument("--loss-on", default="last", choices=["last", "all"],
                    help="label loss on the batch after the K-th update, or the mean over all K batches")
    ap.add_argument("--remat", action="store_true", help="recompute steps in the outer backward (on for K >= 8)")
    ap.add_argument("--unroll-schedule", default="", help="e.g. 1,2,4,8,16: K grows over the epochs")
    ap.add_argument("--fresh", action="store_true", help="every unroll starts from the original model")
    ap.add_argument("--time-rule", default="momentum", choices=["momentum", "filter"])
    ap.add_argument("--adapt", default="bn", choices=["bn", "conv1x1"],
                    help="what adapts: the BN affine params, or every 1x1 conv with BN frozen")
    ap.add_argument("--knob-lr", type=float, default=1e-2, help="Adam lr of the time-rule knobs")
    ap.add_argument("--freeze-predictor", action="store_true", help="train only the time rule (and step)")
    ap.add_argument("--switching-segment", type=int, default=0,
                    help="training streams switch corruption/severity every N batches (0: one per stream)")
    ap.add_argument("--meta-lr", type=float, default=3e-5)
    ap.add_argument("--step-lr", type=float, default=1e-2, help="Adam lr of the log step multiplier")
    ap.add_argument("--imitation", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--updates", type=int, default=None,
                    help="stop after this many predictor updates (overrides --epochs; may end mid-epoch)")
    ap.add_argument("--freeze-eta", action="store_true",
                    help="filter: keep each layer's step at its calibrated start; learn only b1, b2, anchor")
    ap.add_argument("--seed", type=int, default=0, help="training stream order (0 reproduces earlier runs)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.5e-4, help="Tent's step size")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--steps", type=int, default=None, help="cap batches per epoch (smoke tests)")
    ap.add_argument("--streams", type=int, default=None, help="first N streams (smoke tests)")
    ap.add_argument("--batch-split", type=int, default=1,
                    help="devices per stream (each splits the batch), for steps that do not fit on one; "
                         "keeps every b-th training stream so the same devices suffice")
    args = ap.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    all_specs = fit_specs()[: args.streams]
    specs = all_specs[:: args.batch_split]  # every corruption stays in
    S = len(specs)
    k_schedule = [int(k) for k in args.unroll_schedule.split(",")] if args.unroll_schedule else [args.unroll]
    k_for_epoch = lambda e: k_schedule[min(e * len(k_schedule) // args.epochs, len(k_schedule) - 1)]
    sm = st.StreamMesh(S, args.batch_split)
    if args.batch_split > 1:  # keep every activation split over the stream's devices
        resnet.shard_batch(NamedSharding(sm.mesh, P("batch")))
    cache = Path(args.local_cache)

    def stream(spec, keep):
        return st.Stream(st.fetch(args.data_root, spec[0], cache), spec[1], args.batch, keep,
                         resize=imagenet.needs_resize(spec[0]))

    train = [stream(specs[s], fit_ids) for s in sm.local_ids]
    groups = sorted({g for g, _ in all_specs})
    if args.switching_segment:
        loaded = {g: stream((g, 0), fit_ids) for g in groups}

        def switching_streams(epoch):
            """Each stream: a random sequence of fit groups, N batches each, no reset in between."""
            out = []
            for spec_id in sm.local_ids:
                rng = np.random.default_rng([epoch, spec_id, 17] + ([args.seed] if args.seed else []))
                segs = []
                for j, gi in enumerate(rng.permutation(len(groups))):
                    v = st.Stream.__new__(st.Stream)
                    v.records, v.batch, v.resize = loaded[groups[gi]].records, args.batch, False
                    v.perm = np.random.default_rng([epoch, spec_id, j]).permutation(len(v.records))
                    v.num_steps = len(v.records) // args.batch
                    segs.append(v)
                out.append(st.ConcatStream(segs, args.switching_segment))
            return out
    val = [stream(specs[s], test_ids) for s in sm.local_ids]
    params_np, stats_np = resnet.load_torchvision(st.fetch_file(args.weights, cache))
    params, stats = sm.replicate((params_np, stats_np))
    # The adapted parameters ("bn0" below): BN affine params, or every 1x1 conv with BN frozen.
    adapted_names = resnet.bn_names() if args.adapt == "bn" else feedback.conv1x1_names()
    bn0 = {n: params[n] for n in adapted_names}
    phi0, meta = load_predictor(args.predictor, params_np, cache)
    log(f"{S} streams, predictor {meta.get('arch')} {meta.get('config')}, exact top {args.exact_top}, "
        f"K={k_schedule}, loss on {args.loss_on}, {'fresh' if args.fresh else 'carried'} start, "
        f"meta-lr {args.meta_lr}, imitation {args.imitation}, start step x{args.init_mult}, "
        f"time rule {args.time_rule}{' (predictor frozen)' if args.freeze_predictor else ''}, "
        f"switching every {args.switching_segment or 'never'}")

    zeros_like = lambda t: jax.tree.map(np.zeros_like, t)
    knobs0 = timerule.init_knobs("momentum")
    if args.time_rule == "filter":  # initial steps matched to momentum SGD at the starting multiplier
        knobs0 = timerule.init_knobs("filter", eta=calibrate(params, stats, bn0, phi0, train, sm, args),
                                     names=adapted_names)
    state = {"phi": phi0, "log_mult": np.float32(np.log(args.init_mult)),
             "opt_phi": {"t": np.zeros((), np.int32), "m": zeros_like(phi0), "v": zeros_like(phi0)},
             "opt_step": {"t": np.zeros((), np.int32), "m": np.float32(0), "v": np.float32(0)},
             "knobs": knobs0,
             "opt_knobs": {"t": np.zeros((), np.int32), "m": zeros_like(knobs0), "v": zeros_like(knobs0)}}
    history, start_epoch, updates_done = [], 0, 0
    meta_ckpt = st.read_output(args.out, "ckpt/meta.json")
    if meta_ckpt is not None:
        m = json.loads(meta_ckpt)
        state = unflatten(state, dict(np.load(io.BytesIO(st.read_output(args.out, "ckpt/state.npz")))))
        history, start_epoch, updates_done = m["history"], m["epochs_done"], m.get("updates_done", 0)
        log(f"resumed after epoch {start_epoch}")
    state = sm.replicate(state)

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def fresh(bn0):
        bn = jax.tree.map(lambda a: jnp.broadcast_to(a, (S,) + a.shape), bn0)
        return bn, jax.vmap(lambda b: timerule.init_state(args.time_rule, b))(bn)

    def predicted_grads(phi, bn_s, x_s):
        """Predictor's BN gradient at the current state; every ResNet quantity is stop-gradient."""
        p = sg_({**params, **bn_s})
        sig = sg_(feedback.exact_signals(p, stats, imagenet.normalize(x_s), tent.entropy))
        g_true = adapted_grads(args.adapt, p, sig, sig.deltas)
        if feedback.is_precond(phi):  # exact gradient, learned scale
            g = feedback.apply_precond(phi, g_true)
        else:
            deltas = feedback.backward_over_depth(p, stats, sig.x_hats, sig.block_io, sig.stem, sig.d_stream,
                                                  sig.deltas, exact_top=args.exact_top,
                                                  **feedback.predictor_kwargs(phi, sig.inv_stds))
            g = adapted_grads(args.adapt, p, sig, deltas)
        flat = lambda t: flatten_grads(t, adapted_names)
        return g, 1 - cosine(flat(g), flat(g_true)), sig.logits

    def label_ce(bn_s, x, y):
        logits, _ = resnet.apply({**params, **bn_s}, stats, imagenet.normalize(x), batch_stats=True)
        return jnp.mean(jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, y[:, None], -1)[:, 0]), logits

    def one_update(phi, knobs, mult, bn_s, rule_s, x):
        g, mis, logits = predicted_grads(phi, bn_s, x)
        bn_s, rule_s = timerule.apply(args.time_rule, knobs, rule_s, bn_s, g, bn0, args.lr, mult)
        return bn_s, rule_s, mis, logits

    def make_outer_step(K):
        """The jitted outer step for unrolls of length K."""
        remat = args.remat or K >= 8  # long unrolls: recompute each step in the outer backward
        update_fn = jax.checkpoint(one_update) if remat else one_update
        ce_fn = jax.checkpoint(label_ce) if remat else label_ce

        def unroll_loss(trainable, bn_s, vel_s, xs, ys):
            """K predictor-driven updates; label loss on the batch after the last update (--loss-on last)
            or the mean over every batch after an update, each scored before its own update (all)."""
            phi, knobs = trainable["phi"], trainable["knobs"]
            if args.freeze_eta:
                knobs = {n: {**k, "log_eta": sg_(k["log_eta"])} for n, k in knobs.items()}
            if args.freeze_predictor:
                phi = sg_(phi)
            mult = jnp.exp(trainable["log_mult"]) if args.time_rule == "momentum" else 1.0  # filter: eta is a knob

            def step(carry, batch):  # one update, then (--loss-on all) the label loss of the next batch
                bn_s, vel_s = carry
                x, y, x_next, y_next = batch
                bn_s, vel_s, mis, logits = update_fn(phi, knobs, mult, bn_s, vel_s, x)
                ce_t = ce_fn(bn_s, x_next, y_next)[0] if args.loss_on == "all" else 0.0
                return (bn_s, vel_s), (mis, jnp.sum(logits.argmax(-1) == y), ce_t)

            # The first K-1 updates as a scan (one compiled step, whatever K), then the last one, whose
            # next-batch loss counts under either --loss-on.
            (bn_s, vel_s), (mis, correct, ce_t) = jax.lax.scan(
                step, (bn_s, vel_s), (xs[:K - 1], ys[:K - 1], xs[1:K], ys[1:K]))
            bn_s, vel_s, mis_last, logits = update_fn(phi, knobs, mult, bn_s, vel_s, xs[K - 1])
            ce_last, logits_next = ce_fn(bn_s, xs[K], ys[K])
            imit = (jnp.sum(mis) + mis_last) / K
            correct = (jnp.sum(correct) + jnp.sum(logits.argmax(-1) == ys[K - 1])
                       + jnp.sum(logits_next.argmax(-1) == ys[K]))
            ce = (jnp.sum(ce_t) + ce_last) / K if args.loss_on == "all" else ce_last
            return ce + args.imitation * imit, (sg_(bn_s), sg_(vel_s), ce, imit, correct)

        @functools.partial(jax.jit, donate_argnums=(0, 1, 2), out_shardings=(sm.shard, sm.shard, sm.repl, sm.repl))
        def outer_step(bn, vel, state, xs, ys):
            trainable = {"phi": state["phi"], "log_mult": state["log_mult"], "knobs": state["knobs"]}
            grad_fn = jax.value_and_grad(unroll_loss, has_aux=True)
            (_, (bn, vel, ce, imit, correct)), grads = jax.vmap(
                grad_fn, in_axes=(None, 0, 0, 0, 0), spmd_axis_name="streams")(trainable, bn, vel, xs, ys)  # xs: (streams, K+1, B, ...)
            grads = jax.tree.map(lambda a: a.mean(0), grads)
            phi, opt_phi, gnorm = adam(state["phi"], grads["phi"], state["opt_phi"], args.meta_lr)
            log_mult, opt_step, _ = adam(state["log_mult"], grads["log_mult"], state["opt_step"], args.step_lr)
            knobs, opt_knobs, knob_norm = adam(state["knobs"], grads["knobs"], state["opt_knobs"], args.knob_lr)
            if args.freeze_predictor:
                phi, opt_phi = state["phi"], state["opt_phi"]
            new = {"phi": phi, "log_mult": log_mult, "opt_phi": opt_phi, "opt_step": opt_step,
                   "knobs": knobs, "opt_knobs": opt_knobs}
            return bn, vel, new, {"ce": ce.mean(), "imitation": imit.mean(), "correct": correct.sum(),
                                  "grad_norm": knob_norm if args.freeze_predictor else gnorm,
                                  "mult": jnp.exp(log_mult)}

        return outer_step

    outer_steps = {}  # one compiled outer step per K

    @functools.partial(jax.jit, donate_argnums=(0, 1), out_shardings=(sm.shard, sm.shard, sm.shard))
    def deploy_step(bn, vel, x, y, phi, log_mult, knobs):
        mult = jnp.exp(log_mult) if args.time_rule == "momentum" else 1.0

        def one(bn_s, rule_s, x_s, y_s):
            bn_s, rule_s, _, logits = one_update(phi, knobs, mult, bn_s, rule_s, x_s)
            return bn_s, rule_s, jnp.sum(logits.argmax(-1) == y_s)
        return jax.vmap(one, spmd_axis_name="streams")(bn, vel, x, y)

    def evaluate(tag):
        bn, vel = fresh(bn0)
        correct = np.zeros(S)
        for x, y in st.prefetch_batches(val, args.eval_steps, args.decode_workers):
            bn, vel, c = deploy_step(bn, vel, sm.put(x, 1), sm.put(y, 1), state["phi"], state["log_mult"], state["knobs"])
            correct += sm.gather(c)
        err = 100 * (1 - correct.sum() / (S * args.eval_steps * args.batch))
        mult = float(np.exp(jax.device_get(state["log_mult"])))
        history.append({"tag": tag, "val_error": float(err), "mult": mult})
        log(f"EVAL {tag}: deployment error on held-out images {err:.2f}% at step x{mult:.2f}")

    def checkpoint(epochs_done):
        host = jax.device_get(state)
        if lead:
            buf = io.BytesIO()
            np.savez(buf, **flatten(host))
            st.write_output(args.out, "ckpt/state.npz", buf.getvalue())
            st.write_output(args.out, "ckpt/meta.json", json.dumps(
                {"epochs_done": epochs_done, "updates_done": updates_done, "history": history}).encode())

    steps_per_epoch = st.common_steps(train)
    if args.steps:
        steps_per_epoch = min(steps_per_epoch, args.steps)
    if start_epoch == 0:
        evaluate("start")
    t0, n, data_wait = time.time(), 0, 0.0  # data_wait: seconds blocked on image loading
    assert not (args.updates and args.unroll_schedule), "--updates needs a fixed K"
    for epoch in range(start_epoch, 10 ** 6 if args.updates else args.epochs):
        if args.updates and updates_done >= args.updates:
            break
        for s, spec_id in zip(train, sm.local_ids):
            key = [epoch, specs[spec_id][1], spec_id] + ([args.seed] if args.seed else [])
            s.perm = np.random.default_rng(key).permutation(len(s.records))
        epoch_streams = switching_streams(epoch) if args.switching_segment else train
        K = k_for_epoch(epoch)
        if K not in outer_steps:
            outer_steps[K] = make_outer_step(K)
        outer_per_epoch = (steps_per_epoch - 1) // K
        if args.updates:
            outer_per_epoch = min(outer_per_epoch, args.updates - updates_done)
        bn, vel = fresh(bn0)
        batches = st.prefetch_batches(epoch_streams, outer_per_epoch * K + 1, args.decode_workers)
        x_prev, y_prev = next(batches)
        for i in range(outer_per_epoch):
            tw = time.time()
            window = [(x_prev, y_prev)] + [next(batches) for _ in range(K)]
            data_wait += time.time() - tw
            xs = np.stack([w[0] for w in window], axis=1)  # (local streams, K+1, B, ...)
            ys = np.stack([w[1] for w in window], axis=1)
            if args.fresh:
                bn, vel = fresh(bn0)
            bn, vel, state, info = outer_steps[K](bn, vel, state, sm.put(xs, 2), sm.put(ys, 2))
            x_prev, y_prev = window[-1]  # the scored batch is the next unroll's first batch
            n += 1
            updates_done += 1
            if i % 20 == 0 or i + 1 == outer_per_epoch:
                info = jax.device_get(info)
                err = 100 * (1 - info["correct"] / (S * (K + 1) * args.batch))
                log(f"epoch {epoch} K={K} outer {i + 1}/{outer_per_epoch} {(time.time() - t0) / n:.2f}s/step ({100 * data_wait / (time.time() - t0):.0f}% waiting on data)  "
                    f"ce {info['ce']:.3f}  imitation {info['imitation']:.3f}  online err {err:.1f}  "
                    f"step x{info['mult']:.2f}  |grad| {info['grad_norm']:.2e}")
        evaluate(f"epoch {epoch + 1}")
        checkpoint(epoch + 1)

    if lead:
        host = jax.device_get(state)
        buf = io.BytesIO()
        np.savez(buf, **flatten(host["phi"]))
        st.write_output(args.out, "predictor.npz", buf.getvalue())
        if args.time_rule != "momentum":
            buf = io.BytesIO()
            np.savez(buf, **flatten(host["knobs"]))
            st.write_output(args.out, "time_rule.npz", buf.getvalue())
        mult = float(np.exp(host["log_mult"]))
        st.write_output(args.out, "predictor.json", json.dumps(
            {**meta, "kind": "predictor", "phase_c": vars(args), "learned_mult": mult,
             "time_rule": args.time_rule, "adapt": args.adapt, "source_predictor": args.predictor}).encode())
        st.write_output(args.out, "results.json", json.dumps({"config": vars(args), "history": history,
                                                              "learned_mult": mult}).encode())
        log(f"done; learned step x{mult:.3f}")


if __name__ == "__main__":
    main()
