"""Error signals at every BatchNorm of ResNet-50, exact or predicted.

Block l maps h_l -> h_{l+1} = relu(P_l h_l + F_l(h_l)). With d_l = dL/dh_l and
m_l = [h_{l+1} > 0], backpropagation over depth is

    d_l = P_l^T (m_l * d_{l+1}) + J_F^T (m_l * d_{l+1}).

Taps (the 53 BNs Tent updates):
  residual  bn3 and downsample.1 write into the sum; their signal is
            m_l * d_{l+1}, so it follows from d_{l+1} for free.
  branch    bn1 and bn2 sit inside F_l; exactly, they need the branch VJP.
  stem      bn1 before the max pool; a cheap exact step from d_0.

`backward_over_depth` runs the recursion from an exact error, keeps the cheap
shortcut term exactly, and replaces the branch with one of:

  "exact"       the true VJP, including the branch taps (a correctness check)
  "none"        nothing: shortcut-only feedback, branch taps get no signal
  a recurrence  a low-rank state s carried down the depth: a learned
                correction U_l s_l for d_l, and heads V s for the branch taps

Hybrid: with exact_top = k the top k blocks are backpropagated exactly (their
branch taps are taken from the exact signals). That makes the error at the
output of block n-1-k exact, so that block's residual taps are exact for free;
its branch taps would need its own branch VJP, so they are predicted.

DFA predicts each tap directly from the output error instead.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sg.models import resnet


def residual_taps() -> list[str]:
    taps = []
    for pre, _, projection in resnet.blocks():
        taps.append(f"{pre}.bn3")
        if projection:
            taps.append(f"{pre}.downsample.1")
    return taps


def all_taps() -> list[str]:
    return resnet.bn_names()


def tap_widths() -> dict:
    widths = {"bn1": 64}
    for (pre, _, projection), c_out in zip(resnet.blocks(), block_out_widths()):
        mid = c_out // resnet.EXPANSION
        widths |= {f"{pre}.bn1": mid, f"{pre}.bn2": mid, f"{pre}.bn3": c_out}
        if projection:
            widths[f"{pre}.downsample.1"] = c_out
    return widths


def exact_signals(params, stats, x, loss_from_logits):
    """One forward + one backward with probes on every BN output and every block output.

    Returns logits, deltas {bn: dL/dy}, d_stream [dL/dh_out per block], x_hats,
    block_io, stem, and e = dL/dlogits.
    """
    fwd = lambda p, s, x: resnet.apply(p, s, x, batch_stats=True, return_blocks=True)
    _, x_hat_shapes, io_shapes, _ = jax.eval_shape(fwd, params, stats, x)
    zeros = lambda sd: jnp.zeros(sd.shape, sd.dtype)
    probes = jax.tree.map(zeros, x_hat_shapes)
    stream = [zeros(h_out) for _, h_out in io_shapes]

    def loss(probes, stream):
        logits, x_hats, block_io, stem = resnet.apply(
            params, stats, x, batch_stats=True, probes=probes, stream_probes=stream, return_blocks=True)
        return loss_from_logits(logits), (logits, x_hats, block_io, stem)

    (deltas, d_stream), (logits, x_hats, block_io, stem) = jax.grad(
        loss, argnums=(0, 1), has_aux=True)(probes, stream)
    return logits, deltas, d_stream, x_hats, block_io, stem, jax.grad(loss_from_logits)(logits)


def relu_mask(params, x_hats, name):
    """Where the ReLU after BN `name` is active."""
    return params[name]["scale"] * x_hats[name] + params[name]["bias"] > 0


def tap_masks(params, x_hats, block_io, stem):
    """The ReLU derivative each tap's error passes through."""
    masks = {"bn1": stem > 0}
    for k, (pre, _, projection) in enumerate(resnet.blocks()):
        masks[f"{pre}.bn1"] = relu_mask(params, x_hats, f"{pre}.bn1")
        masks[f"{pre}.bn2"] = relu_mask(params, x_hats, f"{pre}.bn2")
        masks[f"{pre}.bn3"] = block_io[k][1] > 0
        if projection:
            masks[f"{pre}.downsample.1"] = block_io[k][1] > 0
    return masks


# ----------------------------------------------------------------------------- recurrence

def init_recurrence(key, rank=64, mix=True):
    """Parameters of the depth recurrence. U and V start at zero: shortcut-only feedback."""
    blocks = resnet.blocks()
    keys = iter(jax.random.split(key, 2 + 3 * len(blocks)))
    normal = lambda k, shape, fan_in: jax.random.normal(k, shape, jnp.float32) / np.sqrt(fan_in)
    return {
        "A": normal(next(keys), (rank, rank), rank),
        "B": 0.1 * normal(next(keys), (rank, rank), rank),  # small: s grows ~1.1x per block at init
        "blocks": [
            {
                "R": normal(next(keys), (c_out, rank), c_out),  # seeds s from an exact error here
                "C": normal(next(keys), (c_in, rank), c_in),
                "e": jnp.zeros((rank,)),
                "U": jnp.zeros((rank, c_in)),
                "V1": jnp.zeros((rank, c_out // resnet.EXPANSION)),
                "V2": jnp.zeros((rank, c_out // resnet.EXPANSION)),
                **({"mix": 0.1 * normal(next(keys), (3, 3, rank), 9)} if mix else {}),
            }
            for c_in, c_out in zip(block_in_widths(), block_out_widths())
        ],
    }


def block_in_widths():
    return [_block_in_width(pre) for pre, _, _ in resnet.blocks()]


def block_out_widths():
    return [resnet.STAGES[int(pre.split(".")[0][-1]) - 1][2] * resnet.EXPANSION for pre, _, _ in resnet.blocks()]


def branch_flops(image=224):
    """Forward multiply-adds of each block's residual branch; the exact VJP of a
    branch (input gradient only) costs about the same."""
    flops, h = [], image // 4
    for (pre, stride, _), c_in, c_out in zip(resnet.blocks(), block_in_widths(), block_out_widths()):
        mid, ho = c_out // resnet.EXPANSION, h // stride
        flops.append(h * h * c_in * mid + ho * ho * mid * mid * 9 + ho * ho * mid * c_out)
        h = ho
    return np.array(flops, np.float64)


def _block_in_width(pre):
    stage, b = pre.split(".")
    stage_idx = int(stage[-1]) - 1
    if b != "0":
        return resnet.STAGES[stage_idx][2] * resnet.EXPANSION
    return 64 if stage_idx == 0 else resnet.STAGES[stage_idx - 1][2] * resnet.EXPANSION


def _upsample(s, factor):
    return s if factor == 1 else jnp.repeat(jnp.repeat(s, factor, axis=1), factor, axis=2)


def _depthwise3x3(s, w):
    return jax.lax.conv_general_dilated(
        s, w[:, :, None, :], (1, 1), [(1, 1), (1, 1)],
        dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=s.shape[-1])


# ----------------------------------------------------------------------------- backward

def backward_over_depth(params, stats, x_hats, block_io, stem, d_exact, exact_deltas=None, *,
                        exact_top=0, branch_term="none", rec=None):
    """Error signals at all 53 BN taps.

    d_exact: exact dL/dh_out of every block (d_exact[-1] is the seed at the top
        of the residual stream). Only d_exact[n-1-exact_top] is used.
    exact_deltas: exact signals at every tap; supplies the branch taps of the
        exactly backpropagated blocks (indices > n-1-exact_top) only.
    exact_top: number of top blocks backpropagated exactly (may be traced).
    branch_term: "exact" | "none". rec: recurrence params.
    Returns {tap: delta}.
    """
    n = len(block_io)
    start = n - 1 - exact_top
    bn = resnet.plain_bn(params, stats, batch_stats=True)
    d, s, scale, out = None, None, None, {}
    for k in reversed(range(n)):
        pre, stride, projection = resnet.blocks()[k]
        h_in, h_out = block_io[k]
        exact_here = k >= start      # the error at this block's output is exact
        branch_paid = k > start      # this block's branch VJP was computed exactly
        d = d_exact[k] if d is None else jnp.where(exact_here, d_exact[k], d)
        g = jnp.where(h_out > 0, d, 0.0)
        out[f"{pre}.bn3"] = g
        if projection:
            out[f"{pre}.downsample.1"] = g
        _, sc_vjp = jax.vjp(lambda h: resnet.shortcut(params, pre, h, stride, bn, projection), h_in)
        d_next = sc_vjp(g)[0]
        b1, b2 = f"{pre}.bn1", f"{pre}.bn2"

        if branch_term == "exact":
            d_branch, out[b1], out[b2] = _branch_vjp(params, bn, pre, h_in, stride, g, x_hats)
            d_next = d_next + d_branch
        else:
            delta1, delta2 = jnp.zeros_like(x_hats[b1]), jnp.zeros_like(x_hats[b2])
            if rec is not None:
                r = rec["blocks"][k]
                # (Re)seed the state from the exact error wherever this block is exact.
                rms = jnp.sqrt(jnp.mean(jnp.square(d_exact[k]))) + 1e-30  # the map is linear in the seed
                seed = jnp.einsum("nhwc,cr->nhwr", d_exact[k] / rms, r["R"])
                s = seed if s is None else jnp.where(exact_here, seed, s)
                scale = rms if scale is None else jnp.where(exact_here, rms, scale)
                delta2 = scale * jnp.where(relu_mask(params, x_hats, b2), jnp.einsum("nhwr,rc->nhwc", s, r["V2"]), 0.0)
                s = _upsample(s, stride)
                a = h_in * jax.lax.rsqrt(jnp.mean(jnp.square(h_in), axis=-1, keepdims=True) + 1e-6)
                gate = jax.nn.silu(jnp.einsum("nhwc,cr->nhwr", a, r["C"]) + r["e"])
                s = s + jnp.einsum("nhwr,rq->nhwq", gate * jnp.einsum("nhwr,rq->nhwq", s, rec["A"]), rec["B"])
                if "mix" in r:
                    s = s + _depthwise3x3(s, r["mix"])
                delta1 = scale * jnp.where(relu_mask(params, x_hats, b1), jnp.einsum("nhwr,rc->nhwc", s, r["V1"]), 0.0)
                d_next = d_next + scale * jnp.einsum("nhwr,rc->nhwc", s, r["U"])
            if exact_deltas is not None:
                delta1 = jnp.where(branch_paid, exact_deltas[b1], delta1)
                delta2 = jnp.where(branch_paid, exact_deltas[b2], delta2)
            out[b1], out[b2] = delta1, delta2
        d = d_next

    # Stem: through the max pool and the stem ReLU (cheap and exact given d_0).
    _, pool_vjp = jax.vjp(resnet.maxpool, stem)
    out["bn1"] = jnp.where(stem > 0, pool_vjp(d)[0], 0.0)
    return out


def _branch_vjp(params, bn, pre, h_in, stride, g, x_hats):
    """Exact VJP of the branch: (dL/dh_in through F, delta at bn1, delta at bn2)."""
    def branch(h, p1, p2):
        def bn_probed(name, x):
            y = bn(name, x)
            return y + p1 if name.endswith("bn1") else y + p2 if name.endswith("bn2") else y
        return resnet.branch(params, pre, h, stride, bn_probed)

    zeros = lambda name: jnp.zeros_like(x_hats[name])
    _, vjp = jax.vjp(branch, h_in, zeros(f"{pre}.bn1"), zeros(f"{pre}.bn2"))
    return vjp(g)


# ----------------------------------------------------------------------------- DFA

def dfa_predict(W, e, masks):
    """delta_tap = mask * broadcast(e @ W_tap): one projection of the output error per tap."""
    return {tap: jnp.where(masks[tap], jnp.einsum("nk,kc->nc", e, W[tap])[:, None, None, :], 0.0)
            for tap in all_taps()}


def dfa_stats(e, deltas):
    """Sufficient statistics for a ridge fit of mean_hw(delta_tap) on e."""
    return {
        "G": jnp.einsum("nk,nj->kj", e, e),
        "H": {tap: jnp.einsum("nk,nhwc->kc", e, deltas[tap]) / (deltas[tap].shape[1] * deltas[tap].shape[2])
              for tap in all_taps()},
    }


def dfa_solve(stats, ridge=1e-3):
    G = stats["G"]
    lam = ridge * jnp.trace(G) / G.shape[0]
    chol = jax.scipy.linalg.cho_factor(G + lam * jnp.eye(G.shape[0]))
    return {tap: jax.scipy.linalg.cho_solve(chol, H) for tap, H in stats["H"].items()}


def dfa_random(key):
    widths = tap_widths()
    keys = jax.random.split(key, len(all_taps()))
    return {tap: jax.random.normal(k, (1000, widths[tap])) for k, tap in zip(keys, all_taps())}


# ----------------------------------------------------------------------------- metrics

def _cos_ratio(a, b):
    a, b = a.ravel(), b.ravel()
    na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    return jnp.dot(a, b) / (na * nb + 1e-30), na / (nb + 1e-30)


def tap_grads(deltas, x_hats):
    """{tap: concat(dL/dscale, dL/dbias)} at every BN."""
    g = resnet.bn_grads({t: deltas[t] for t in all_taps()}, {t: x_hats[t] for t in all_taps()})
    return {t: jnp.concatenate([g[t]["scale"], g[t]["bias"]]) for t in all_taps()}


def compare(pred, true, x_hats):
    """Scores of predicted against true error signals.

    Returns (per_tap, full):
      per_tap (taps, 4): cos and norm ratio of delta, then of the BN (scale, bias)
        gradient that delta induces at that tap.
      full (2,): cos and norm ratio of all BN gradients joined into one vector,
        i.e. Tent's update direction.
    """
    gp, gt = tap_grads(pred, x_hats), tap_grads(true, x_hats)
    rows = [jnp.stack([*_cos_ratio(pred[t], true[t]), *_cos_ratio(gp[t], gt[t])]) for t in all_taps()]
    full = jnp.stack(_cos_ratio(jnp.concatenate(list(gp.values())), jnp.concatenate(list(gt.values()))))
    return jnp.stack(rows), full
