"""Error signals at the residual-stream taps of ResNet-50, exact or predicted.

Block l maps h_l -> h_{l+1} = relu(P_l h_l + F_l(h_l)). With d_l = dL/dh_l and
m_l = [h_{l+1} > 0], backpropagation over depth is

    d_l = P_l^T (m_l * d_{l+1}) + J_F^T (m_l * d_{l+1}).

The residual taps are the BNs that write into the sum (bn3, and downsample.1
on projection blocks). Their error signal is m_l * d_{l+1}, so it follows from
d_{l+1} for free. `backward_over_depth` runs this recursion from an exact seed
d_top = dL/dh_final, keeps the cheap shortcut term exactly, and replaces the
branch term J_F^T with one of:

  "exact"       the true vector-Jacobian product (a correctness check)
  "none"        nothing: shortcut-only feedback
  a recurrence  a small learned correction U_l s_l (proposal Eq. cell/pred)

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


def seed(params, h_top, loss_from_logits):
    """Exact error at the top of the residual stream and at the logits."""
    logits, head_vjp = jax.vjp(lambda h: resnet.head(params, h), h_top)
    e = jax.grad(loss_from_logits)(logits)
    return head_vjp(e)[0], e


# ----------------------------------------------------------------------------- recurrence

def init_recurrence(key, rank=64, mix=True):
    """Parameters of the depth recurrence. U starts at zero: shortcut-only feedback."""
    blocks = resnet.blocks()
    keys = iter(jax.random.split(key, 3 + 2 * len(blocks)))
    widths = [_block_in_width(pre) for pre, _, _ in blocks]
    normal = lambda k, shape, fan_in: jax.random.normal(k, shape, jnp.float32) / np.sqrt(fan_in)
    return {
        "E": normal(next(keys), (2048, rank), 2048),
        "A": normal(next(keys), (rank, rank), rank),
        "B": 0.1 * normal(next(keys), (rank, rank), rank),  # small: s grows ~1.1x per block at init
        "blocks": [
            {
                "C": normal(next(keys), (c, rank), c),
                "e": jnp.zeros((rank,)),
                "U": jnp.zeros((rank, c)),
                **({"mix": 0.1 * normal(next(keys), (3, 3, rank), 9)} if mix else {}),
            }
            for c in widths
        ],
    }


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

def backward_over_depth(params, stats, block_io, d_top, branch_term="none", rec=None):
    """Error signals at every residual tap, from the exact seed d_top.

    branch_term: "exact" | "none". rec: recurrence params (adds U_l s_l).
    Returns {tap: delta (N, H, W, C)}.
    """
    bn = resnet.plain_bn(params, stats, batch_stats=True)
    scale = jnp.sqrt(jnp.mean(jnp.square(d_top))) + 1e-30  # the map is linear in d_top
    s = jnp.einsum("nhwc,cr->nhwr", d_top / scale, rec["E"]) if rec is not None else None
    d, out = d_top, {}
    for k in reversed(range(len(block_io))):
        pre, stride, projection = resnet.blocks()[k]
        h_in, h_out = block_io[k]
        g = jnp.where(h_out > 0, d, 0.0)
        out[f"{pre}.bn3"] = g
        if projection:
            out[f"{pre}.downsample.1"] = g
        _, sc_vjp = jax.vjp(lambda h: resnet.shortcut(params, pre, h, stride, bn, projection), h_in)
        d = sc_vjp(g)[0]
        if branch_term == "exact":
            _, br_vjp = jax.vjp(lambda h: resnet.branch(params, pre, h, stride, bn), h_in)
            d = d + br_vjp(g)[0]
        if rec is not None:
            r = rec["blocks"][k]
            s = _upsample(s, stride)
            a = h_in * jax.lax.rsqrt(jnp.mean(jnp.square(h_in), axis=-1, keepdims=True) + 1e-6)
            gate = jax.nn.silu(jnp.einsum("nhwc,cr->nhwr", a, r["C"]) + r["e"])
            s = s + jnp.einsum("nhwr,rq->nhwq", gate * jnp.einsum("nhwr,rq->nhwq", s, rec["A"]), rec["B"])
            if "mix" in r:
                s = s + _depthwise3x3(s, r["mix"])
            d = d + scale * jnp.einsum("nhwr,rc->nhwc", s, r["U"])
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

def compare(pred, true, x_hats):
    """Per tap: [cos(delta), |pred|/|true| (delta), cos(grad), ratio (grad)].

    grad = the BN (scale, bias) gradient each delta induces at that tap.
    Returns an array (taps, 4).
    """
    def cos_ratio(a, b):
        a, b = a.ravel(), b.ravel()
        na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
        return jnp.dot(a, b) / (na * nb + 1e-30), na / (nb + 1e-30)

    rows = []
    for tap in residual_taps():
        gp = resnet.bn_grads({tap: pred[tap]}, {tap: x_hats[tap]})[tap]
        gt = resnet.bn_grads({tap: true[tap]}, {tap: x_hats[tap]})[tap]
        flat = lambda g: jnp.concatenate([g["scale"], g["bias"]])
        rows.append(jnp.stack([*cos_ratio(pred[tap], true[tap]), *cos_ratio(flat(gp), flat(gt))]))
    return jnp.stack(rows)
