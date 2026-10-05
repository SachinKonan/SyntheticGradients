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
trained on the states it reaches during deployment.

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

from sg import feedback, tent
from sg.data import imagenet
from sg.experiments import streams as st
from sg.experiments.gate2_grid import flatten, unflatten
from sg.experiments.gate2_predictability import adam, fit_ids, fit_specs, test_ids
from sg.models import resnet

sg_ = jax.lax.stop_gradient


def load_predictor(url, params_np, cache):
    """A saved predictor, or 'svd:<frac>' for the untrained low-rank backward (no pretraining)."""
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
    ap.add_argument("--meta-lr", type=float, default=3e-5)
    ap.add_argument("--step-lr", type=float, default=1e-2, help="Adam lr of the log step multiplier")
    ap.add_argument("--imitation", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.5e-4, help="Tent's step size")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--steps", type=int, default=None, help="cap batches per epoch (smoke tests)")
    ap.add_argument("--streams", type=int, default=None, help="first N streams (smoke tests)")
    args = ap.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    specs = fit_specs()[: args.streams]
    S, K = len(specs), args.unroll
    sm = st.StreamMesh(S)
    cache = Path(args.local_cache)

    def stream(spec, keep):
        return st.Stream(st.fetch(args.data_root, spec[0], cache), spec[1], args.batch, keep,
                         resize=imagenet.needs_resize(spec[0]))

    train = [stream(specs[s], fit_ids) for s in sm.local_ids]
    val = [stream(specs[s], test_ids) for s in sm.local_ids]
    params_np, stats_np = resnet.load_torchvision(st.fetch_file(args.weights, cache))
    params, stats = sm.replicate((params_np, stats_np))
    bn0 = tent.bn_params(params)
    phi0, meta = load_predictor(args.predictor, params_np, cache)
    log(f"{S} streams, predictor {meta.get('arch')} {meta.get('config')}, exact top {args.exact_top}, "
        f"K={K}, meta-lr {args.meta_lr}, imitation {args.imitation}, start step x{args.init_mult}")

    zeros_like = lambda t: jax.tree.map(np.zeros_like, t)
    state = {"phi": phi0, "log_mult": np.float32(np.log(args.init_mult)),
             "opt_phi": {"t": np.zeros((), np.int32), "m": zeros_like(phi0), "v": zeros_like(phi0)},
             "opt_step": {"t": np.zeros((), np.int32), "m": np.float32(0), "v": np.float32(0)}}
    history, start_epoch = [], 0
    meta_ckpt = st.read_output(args.out, "ckpt/meta.json")
    if meta_ckpt is not None:
        m = json.loads(meta_ckpt)
        state = unflatten(state, dict(np.load(io.BytesIO(st.read_output(args.out, "ckpt/state.npz")))))
        history, start_epoch = m["history"], m["epochs_done"]
        log(f"resumed after epoch {start_epoch}")
    state = sm.replicate(state)

    @functools.partial(jax.jit, out_shardings=sm.shard)
    def fresh(bn0):
        bn = jax.tree.map(lambda a: jnp.broadcast_to(a, (S,) + a.shape), bn0)
        return bn, jax.tree.map(jnp.zeros_like, bn)

    def predicted_grads(phi, bn_s, x_s):
        """Predictor's BN gradient at the current state; every ResNet quantity is stop-gradient."""
        p = sg_({**params, **bn_s})
        sig = sg_(feedback.exact_signals(p, stats, imagenet.normalize(x_s), tent.entropy))
        deltas = feedback.backward_over_depth(p, stats, sig.x_hats, sig.block_io, sig.stem, sig.d_stream,
                                              sig.deltas, exact_top=args.exact_top,
                                              **feedback.predictor_kwargs(phi, sig.inv_stds))
        g = resnet.bn_grads(deltas, sig.x_hats)
        g_true = resnet.bn_grads(sig.deltas, sig.x_hats)
        flat = lambda t: jnp.concatenate([jnp.concatenate([t[n]["scale"], t[n]["bias"]]) for n in resnet.bn_names()])
        return g, 1 - cosine(flat(g), flat(g_true)), sig.logits

    def unroll_loss(trainable, bn_s, vel_s, xs, ys):
        """K predictor-driven updates, then the label loss on batch K (and online accuracy before each update)."""
        phi, log_mult = trainable["phi"], trainable["log_mult"]
        step = args.lr * jnp.exp(log_mult)
        imit, correct = 0.0, 0
        for t in range(K):
            g, mis, logits = predicted_grads(phi, bn_s, xs[t])
            correct += jnp.sum(logits.argmax(-1) == ys[t])
            imit += mis / K
            vel_s = jax.tree.map(lambda v, gg: args.momentum * v + gg, vel_s, g)
            bn_s = jax.tree.map(lambda b, v: b - step * v, bn_s, vel_s)
        logits, _ = resnet.apply({**params, **bn_s}, stats, imagenet.normalize(xs[K]), batch_stats=True)
        ce = jnp.mean(jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, ys[K][:, None], -1)[:, 0])
        correct += jnp.sum(logits.argmax(-1) == ys[K])
        return ce + args.imitation * imit, (sg_(bn_s), sg_(vel_s), ce, imit, correct)

    @functools.partial(jax.jit, donate_argnums=(0, 1, 2), out_shardings=(sm.shard, sm.shard, sm.repl, sm.repl))
    def outer_step(bn, vel, state, xs, ys):
        trainable = {"phi": state["phi"], "log_mult": state["log_mult"]}
        grad_fn = jax.value_and_grad(unroll_loss, has_aux=True)
        (_, (bn, vel, ce, imit, correct)), grads = jax.vmap(
            grad_fn, in_axes=(None, 0, 0, 0, 0))(trainable, bn, vel, xs, ys)  # xs: (streams, K+1, B, ...)
        grads = jax.tree.map(lambda a: a.mean(0), grads)
        phi, opt_phi, gnorm = adam(state["phi"], grads["phi"], state["opt_phi"], args.meta_lr)
        log_mult, opt_step, _ = adam(state["log_mult"], grads["log_mult"], state["opt_step"], args.step_lr)
        new = {"phi": phi, "log_mult": log_mult, "opt_phi": opt_phi, "opt_step": opt_step}
        return bn, vel, new, {"ce": ce.mean(), "imitation": imit.mean(), "correct": correct.sum(),
                              "grad_norm": gnorm, "mult": jnp.exp(log_mult)}

    @functools.partial(jax.jit, donate_argnums=(0, 1), out_shardings=(sm.shard, sm.shard, sm.shard))
    def deploy_step(bn, vel, x, y, phi, log_mult):
        def one(bn_s, vel_s, x_s, y_s):
            g, _, logits = predicted_grads(phi, bn_s, x_s)
            vel_s = jax.tree.map(lambda v, gg: args.momentum * v + gg, vel_s, g)
            bn_s = jax.tree.map(lambda b, v: b - args.lr * jnp.exp(log_mult) * v, bn_s, vel_s)
            return bn_s, vel_s, jnp.sum(logits.argmax(-1) == y_s)
        return jax.vmap(one)(bn, vel, x, y)

    def evaluate(tag):
        bn, vel = fresh(bn0)
        correct = np.zeros(S)
        for x, y in st.prefetch_batches(val, args.eval_steps, args.decode_workers):
            bn, vel, c = deploy_step(bn, vel, sm.put(x), sm.put(y), state["phi"], state["log_mult"])
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
                {"epochs_done": epochs_done, "history": history}).encode())

    steps_per_epoch = st.common_steps(train)
    if args.steps:
        steps_per_epoch = min(steps_per_epoch, args.steps)
    outer_per_epoch = (steps_per_epoch - 1) // K
    if start_epoch == 0:
        evaluate("start")
    t0, n = time.time(), 0
    for epoch in range(start_epoch, args.epochs):
        for s, spec_id in zip(train, sm.local_ids):
            s.perm = np.random.default_rng([epoch, specs[spec_id][1], spec_id]).permutation(len(s.records))
        bn, vel = fresh(bn0)
        batches = st.prefetch_batches(train, outer_per_epoch * K + 1, args.decode_workers)
        x_prev, y_prev = next(batches)
        for i in range(outer_per_epoch):
            window = [(x_prev, y_prev)] + [next(batches) for _ in range(K)]
            xs = np.stack([w[0] for w in window], axis=1)  # (local streams, K+1, B, ...)
            ys = np.stack([w[1] for w in window], axis=1)
            bn, vel, state, info = outer_step(bn, vel, state, sm.put(xs), sm.put(ys))
            x_prev, y_prev = window[-1]  # the scored batch is the next unroll's first batch
            n += 1
            if i % 20 == 0 or i + 1 == outer_per_epoch:
                info = jax.device_get(info)
                err = 100 * (1 - info["correct"] / (S * (K + 1) * args.batch))
                log(f"epoch {epoch} outer {i + 1}/{outer_per_epoch} {(time.time() - t0) / n:.2f}s/step  "
                    f"ce {info['ce']:.3f}  imitation {info['imitation']:.3f}  online err {err:.1f}  "
                    f"step x{info['mult']:.2f}  |grad| {info['grad_norm']:.2e}")
        evaluate(f"epoch {epoch + 1}")
        checkpoint(epoch + 1)

    if lead:
        host = jax.device_get(state)
        buf = io.BytesIO()
        np.savez(buf, **flatten(host["phi"]))
        st.write_output(args.out, "predictor.npz", buf.getvalue())
        mult = float(np.exp(host["log_mult"]))
        st.write_output(args.out, "predictor.json", json.dumps(
            {**meta, "kind": "predictor", "phase_c": vars(args), "learned_mult": mult,
             "source_predictor": args.predictor}).encode())
        st.write_output(args.out, "results.json", json.dumps({"config": vars(args), "history": history,
                                                              "learned_mult": mult}).encode())
        log(f"done; learned step x{mult:.3f}")


if __name__ == "__main__":
    main()
