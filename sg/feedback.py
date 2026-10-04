"""Error signals at the residual-stream taps of ResNet-50, exact or predicted.

Block l maps h_l -> h_{l+1} = relu(P_l h_l + F_l(h_l)). With d_l = dL/dh_l and
m_l = [h_{l+1} > 0], backpropagation over depth is

    d_l = P_l^T (m_l * d_{l+1}) + J_F^T (m_l * d_{l+1}).

The residual taps are the BNs that write into the sum (bn3, and downsample.1
on projection blocks). Their error signal is m_l * d_{l+1}, so it follows from
d_{l+1} for free. `backward_over_depth` runs this recursion from an exact
error, keeps the cheap shortcut term exactly, and replaces the branch term
J_F^T with one of:

  "exact"       the true vector-Jacobian product (a correctness check)
  "none"        nothing: shortcut-only feedback
  a recurrence  a small learned correction U_l s_l (proposal Eq. cell/pred)

Hybrid: with exact_top = k, the top k blocks are backpropagated exactly, so the
error at the output of block n-1-k is exact; prediction starts there.

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


def exact_signals(params, stats, x, loss_from_logits):
    """One forward + one backward with probes on every BN output and every block output.

    Returns logits, deltas {bn: dL/dy}, d_stream [dL/dh_out per block], x_hats,
    block_io, and e = dL/dlogits.
    """
    fwd = lambda p, s, x: resnet.apply(p, s, x, batch_stats=True, return_blocks=True)
    _, x_hat_shapes, io_shapes = jax.eval_shape(fwd, params, stats, x)
    zeros = lambda sd: jnp.zeros(sd.shape, sd.dtype)
    probes = jax.tree.map(zeros, x_hat_shapes)
    stream = [zeros(h_out) for _, h_out in io_shapes]

    def loss(probes, stream):
        logits, x_hats, block_io = resnet.apply(
            params, stats, x, batch_stats=True, probes=probes, stream_probes=stream, return_blocks=True)
        return loss_from_logits(logits), (logits, x_hats, block_io)

    (deltas, d_stream), (logits, x_hats, block_io) = jax.grad(loss, argnums=(0, 1), has_aux=True)(probes, stream)
    return logits, deltas, d_stream, x_hats, block_io, jax.grad(loss_from_logits)(logits)


# ----------------------------------------------------------------------------- recurrence

def init_recurrence(key, rank=64, mix=True):
    """Parameters of the depth recurrence. U starts at zero: shortcut-only feedback."""
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

def backward_over_depth(params, stats, block_io, d_exact, *, exact_top=0, branch_term="none", rec=None):
    """Error signals at every residual tap.

    d_exact: exact dL/dh_out of every block (a list; d_exact[-1] is the seed at
        the top of the residual stream). Only d_exact[n-1-exact_top] is read.
    exact_top: number of top blocks backpropagated exactly (may be traced).
        Taps at and above the start block are exact.
    branch_term: "exact" | "none". rec: recurrence params (adds U_l s_l).
    Returns {tap: delta (N, H, W, C)}.
    """
    n = len(block_io)
    start = n - 1 - exact_top
    bn = resnet.plain_bn(params, stats, batch_stats=True)
    d, s, scale, out = None, None, None, {}
    for k in reversed(range(n)):
        pre, stride, projection = resnet.blocks()[k]
        h_in, h_out = block_io[k]
        exact_here = k >= start
        d = d_exact[k] if d is None else jnp.where(exact_here, d_exact[k], d)
        g = jnp.where(h_out > 0, d, 0.0)
        out[f"{pre}.bn3"] = g
        if projection:
            out[f"{pre}.downsample.1"] = g
        _, sc_vjp = jax.vjp(lambda h: resnet.shortcut(params, pre, h, stride, bn, projection), h_in)
        d_next = sc_vjp(g)[0]
        if branch_term == "exact":
            _, br_vjp = jax.vjp(lambda h: resnet.branch(params, pre, h, stride, bn), h_in)
            d_next = d_next + br_vjp(g)[0]
        if rec is not None:
            r = rec["blocks"][k]
            # (Re)seed the state from the exact error wherever this block is exact.
            rms = jnp.sqrt(jnp.mean(jnp.square(d_exact[k]))) + 1e-30  # the map is linear in the seed
            seed = jnp.einsum("nhwc,cr->nhwr", d_exact[k] / rms, r["R"])
            s = seed if s is None else jnp.where(exact_here, seed, s)
            scale = rms if scale is None else jnp.where(exact_here, rms, scale)
            s = _upsample(s, stride)
            a = h_in * jax.lax.rsqrt(jnp.mean(jnp.square(h_in), axis=-1, keepdims=True) + 1e-6)
            gate = jax.nn.silu(jnp.einsum("nhwc,cr->nhwr", a, r["C"]) + r["e"])
            s = s + jnp.einsum("nhwr,rq->nhwq", gate * jnp.einsum("nhwr,rq->nhwq", s, rec["A"]), rec["B"])
            if "mix" in r:
                s = s + _depthwise3x3(s, r["mix"])
            d_next = d_next + scale * jnp.einsum("nhwr,rc->nhwc", s, r["U"])
        d = d_next
    return out


# ----------------------------------------------------------------------------- DFA

def dfa_predict(W, e, block_io):
    """delta_tap = m * broadcast(e @ W_tap): one projection of the output error per tap."""
    out = {}
    for k, (pre, _, projection) in enumerate(resnet.blocks()):
        mask = block_io[k][1] > 0
        for tap in [f"{pre}.bn3"] + ([f"{pre}.downsample.1"] if projection else []):
            out[tap] = jnp.where(mask, jnp.einsum("nk,kc->nc", e, W[tap])[:, None, None, :], 0.0)
    return out


def dfa_stats(e, deltas):
    """Sufficient statistics for a ridge fit of mean_hw(delta_tap) on e."""
    return {
        "G": jnp.einsum("nk,nj->kj", e, e),
        "H": {tap: jnp.einsum("nk,nhwc->kc", e, deltas[tap]) / (deltas[tap].shape[1] * deltas[tap].shape[2])
              for tap in residual_taps()},
    }


def dfa_solve(stats, ridge=1e-3):
    G = stats["G"]
    lam = ridge * jnp.trace(G) / G.shape[0]
    chol = jax.scipy.linalg.cho_factor(G + lam * jnp.eye(G.shape[0]))
    return {tap: jax.scipy.linalg.cho_solve(chol, H) for tap, H in stats["H"].items()}


def dfa_random(key):
    keys = jax.random.split(key, len(residual_taps()))
    widths = {}
    for pre, _, projection in resnet.blocks():
        c = int(pre.split(".")[0][-1]) - 1
        widths[f"{pre}.bn3"] = resnet.STAGES[c][2] * resnet.EXPANSION
        if projection:
            widths[f"{pre}.downsample.1"] = resnet.STAGES[c][2] * resnet.EXPANSION
    return {tap: jax.random.normal(k, (1000, widths[tap])) for k, tap in zip(keys, residual_taps())}


# ----------------------------------------------------------------------------- metrics

def _cos_ratio(a, b):
    a, b = a.ravel(), b.ravel()
    na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    return jnp.dot(a, b) / (na * nb + 1e-30), na / (nb + 1e-30)


def tap_grads(deltas, x_hats):
    """{tap: concat(dL/dscale, dL/dbias)} at the residual taps."""
    taps = residual_taps()
    g = resnet.bn_grads({t: deltas[t] for t in taps}, {t: x_hats[t] for t in taps})
    return {t: jnp.concatenate([g[t]["scale"], g[t]["bias"]]) for t in taps}


def compare(pred, true, x_hats):
    """Scores of predicted against true error signals.

    Returns (per_tap, full):
      per_tap (taps, 4): cos and norm ratio of delta, then of the BN (scale, bias)
        gradient that delta induces at that tap.
      full (2,): cos and norm ratio of all residual-tap BN gradients joined into
        one vector, i.e. the update direction for those parameters.
    """
    gp, gt = tap_grads(pred, x_hats), tap_grads(true, x_hats)
    rows = [jnp.stack([*_cos_ratio(pred[t], true[t]), *_cos_ratio(gp[t], gt[t])]) for t in residual_taps()]
    full = jnp.stack(_cos_ratio(jnp.concatenate(list(gp.values())), jnp.concatenate(list(gt.values()))))
    return jnp.stack(rows), full
