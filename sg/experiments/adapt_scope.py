"""Does adapting more than BN help Tent, with exact gradients?

Tent's protocol (batch 64, entropy loss, SGD with momentum 0.9, test-batch BN
statistics, reset per stream, every batch scored before its update), but the
adapted parameters are one of these scopes:

  bn            all 53 BN (scale, bias)                         (Tent)
  bn+c3:L       BN, plus conv3 (the 1x1 output conv) of the last L blocks
  bn+c13:L      BN, plus conv1 and conv3 (both 1x1 convs) of the last L blocks

The 1x1 convs are ResNet's per-pixel "MLP" (the analog of a Transformer MLP);
updating them in the last quarter of blocks mirrors TTT-E2E. BN keeps Tent's
step (--bn-mult x Tent's lr); conv weights get their own step, swept over
--conv-mults (x Tent's lr). Gradients are exact (jax.grad); this checks whether
the larger scope is worth predicting at all.

Sets as in gate3_adapt: val (fit corruptions, odd ids) to choose the conv step,
test (15 corruptions + clean, odd ids) to report.
"""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from sg import tent
from sg.data import imagenet
from sg.experiments import streams as st
from sg.experiments.gate2_predictability import fit_specs, test_ids, test_specs
from sg.models import resnet


def conv_names(scope: str) -> list[str]:
    """The conv weights a scope adapts, besides BN."""
    if scope == "bn":
        return []
    kind, last = scope.split("+")[1].split(":")
    blocks = [pre for pre, _, _ in resnet.blocks()][-int(last):]
    which = {"c3": ("conv3",), "c13": ("conv1", "conv3")}[kind]
    return [f"{pre}.{c}" for pre in blocks for c in which]


def main():
    ap = argparse.ArgumentParser()
    st.add_launch_args(ap)
    ap.add_argument("--set", required=True, choices=["val", "test"])
    ap.add_argument("--scopes", default="bn,bn+c3:4,bn+c13:4,bn+c13:8")
    ap.add_argument("--bn-mult", type=float, default=3.0, help="BN step, x Tent's lr (3 is Tent's best)")
    ap.add_argument("--conv-mults", default="0.03,0.1,0.3,1,3")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.5e-4, help="Tent's lr")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--streams", type=int, default=None)
    args = ap.parse_args()

    st.init_distributed(args)
    lead = jax.process_index() == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)) if lead else (lambda *a: None)

    specs = (fit_specs() if args.set == "val" else test_specs())[: args.streams]
    S = len(specs)
    sm = st.StreamMesh(S)
    traj_shard = NamedSharding(sm.shard.mesh, P(None, "streams"))
    scopes = args.scopes.split(",")
    cmults = np.array([float(m) for m in args.conv_mults.split(",")], np.float32)
    cache = Path(args.local_cache)
    streams = [st.Stream(st.fetch(args.data_root, specs[s][0], cache), specs[s][1], args.batch, test_ids,
                         resize=imagenet.needs_resize(specs[s][0])) for s in sm.local_ids]
    params, stats = sm.replicate(resnet.load_torchvision(st.fetch_file(args.weights, cache)))
    num_steps = st.common_steps(streams)
    if args.steps:
        num_steps = min(num_steps, args.steps)
    for sc in scopes:
        n = sum(int(np.prod(params[c]["w"].shape)) for c in conv_names(sc))
        log(f"scope {sc}: {len(conv_names(sc))} convs, {n:,} conv weights (+53,120 BN)")

    def adapted(sc):
        return {**tent.bn_params(params), **{c: params[c] for c in conv_names(sc)}}

    def make_step(sc):
        convs = set(conv_names(sc))

        @functools.partial(jax.jit, donate_argnums=(0, 1), out_shardings=(traj_shard, traj_shard, traj_shard))
        def step(theta, vel, x, y):
            def per_mult(args_):
                th_m, v_m, cmult = args_

                def per_stream(th, v, x_s, y_s):
                    def loss(th):
                        logits, _ = resnet.apply({**params, **th}, stats, imagenet.normalize(x_s), batch_stats=True)
                        return tent.entropy(logits), logits
                    (_, logits), g = jax.value_and_grad(loss, has_aux=True)(th)
                    correct = jnp.sum(logits.argmax(-1) == y_s)  # scored before the update
                    v = jax.tree.map(lambda vv, gg: args.momentum * vv + gg, v, g)
                    lr = {n: args.lr * (cmult if n in convs else args.bn_mult) for n in th}
                    new = {n: jax.tree.map(lambda w, vv: w - lr[n] * vv, th[n], v[n]) for n in th}
                    ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(new)]))
                    keep = lambda a, b: jax.tree.map(lambda u, w: jnp.where(ok, u, w), a, b)
                    return keep(new, th), keep(v, v), correct

                return jax.vmap(per_stream)(th_m, v_m, x, y)

            return jax.lax.map(per_mult, (theta, vel, jnp.asarray(mults_for[sc])))

        return step

    mults_for = {sc: (cmults if conv_names(sc) else np.array([1.0], np.float32)) for sc in scopes}
    steps = {sc: make_step(sc) for sc in scopes}

    def fresh(sc):
        m = len(mults_for[sc])
        th = jax.jit(lambda t: jax.tree.map(lambda a: jnp.broadcast_to(a, (m, S) + a.shape), t),
                     out_shardings=traj_shard)(adapted(sc))
        return th, jax.tree.map(jnp.zeros_like, th)

    state = {sc: fresh(sc) for sc in scopes}
    correct = {sc: np.zeros((len(mults_for[sc]), S), np.int64) for sc in scopes}
    t0 = time.time()
    for i, (x, y) in enumerate(st.prefetch_batches(streams, num_steps, args.decode_workers)):
        xg, yg = sm.put(x), sm.put(y)
        for sc in scopes:
            th, v = state[sc]
            th, v, c = steps[sc](th, v, xg, yg)
            state[sc] = (th, v)
            correct[sc] += np.asarray(multihost_utils.process_allgather(c, tiled=True))
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == num_steps:
            n = (i + 1) * args.batch
            corrupt = np.array([g.startswith("imagenet_c/") for g, _ in specs])
            best = {sc: f"{100 * (1 - correct[sc][:, corrupt].sum(1) / (n * corrupt.sum())).min():.1f}" for sc in scopes}
            log(f"step {i + 1}/{num_steps} {(time.time() - t0) / (i + 1):.1f}s/step  best error by scope: {best}")

    if lead:
        images = num_steps * args.batch
        out = {"config": vars(args), "set": args.set, "images_per_stream": images,
               "streams": [{"group": g, "order": o} for g, o in specs],
               "conv_mults": {sc: mults_for[sc].tolist() for sc in scopes},
               "error": {sc: (100 * (1 - correct[sc] / images)).tolist() for sc in scopes}}
        st.write_output(args.out, "results.json", json.dumps(out).encode())
        log("done")


if __name__ == "__main__":
    main()
