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
                correction U_l s_l for d_l, and heads V s for the branch taps;
                the state update is either a gated linear cell (init_recurrence)
                or a convolution-free GRU cell (init_gru), shared across depth
  low-rank      the true branch backward (exact ReLU masks and BN backward)
                with each conv replaced by a rank-r factorization, initialized
                from the SVD of the real weights; full rank is exact

Hybrid: with exact_top = k the top k blocks are backpropagated exactly (their
branch taps are taken from the exact signals). That makes the error at the
output of block n-1-k exact, so that block's residual taps are exact for free;
its branch taps would need its own branch VJP, so they are predicted.

DFA predicts each tap directly from the output error instead.
"""

from typing import NamedTuple

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


class Signals(NamedTuple):
    logits: jax.Array
    deltas: dict      # {bn: dL/dy}, exact
    d_stream: list    # [dL/dh_out per block], exact
    x_hats: dict      # {bn: normalized input}
    block_io: list    # [(h_in, h_out) per block]
    stem: jax.Array   # input to the max pool
    e: jax.Array      # dL/dlogits
    inv_stds: dict    # {bn: 1 / batch std}


def exact_signals(params, stats, x, loss_from_logits) -> Signals:
    """One forward + one backward with probes on every BN output and every block output."""
    fwd = lambda p, s, x: resnet.apply(p, s, x, batch_stats=True, return_blocks=True)
    _, x_hat_shapes, io_shapes, _, _ = jax.eval_shape(fwd, params, stats, x)
    zeros = lambda sd: jnp.zeros(sd.shape, sd.dtype)
    probes = jax.tree.map(zeros, x_hat_shapes)
    stream = [zeros(h_out) for _, h_out in io_shapes]

    def loss(probes, stream):
        out = resnet.apply(params, stats, x, batch_stats=True, probes=probes, stream_probes=stream,
                           return_blocks=True)
        return loss_from_logits(out[0]), out

    (deltas, d_stream), (logits, x_hats, block_io, stem, inv_stds) = jax.grad(
        loss, argnums=(0, 1), has_aux=True)(probes, stream)
    return Signals(logits, deltas, d_stream, x_hats, block_io, stem, jax.grad(loss_from_logits)(logits), inv_stds)


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


def init_gru(key, rank=64, mix=True):
    """Like init_recurrence, but the shared state update is a GRU cell on [s, x_l],
    x_l = C_l a_l + e_l. Same per-block inputs and outputs, so only the cell differs."""
    rec = init_recurrence(key, rank, mix)
    k_z, k_r, k_h = jax.random.split(jax.random.fold_in(key, 1), 3)
    normal = lambda k: jax.random.normal(k, (2 * rank, rank), jnp.float32) / np.sqrt(2 * rank)
    del rec["A"], rec["B"]
    rec["gru"] = {"Wz": normal(k_z), "bz": jnp.full((rank,), -2.0),  # start mostly keeping s
                  "Wr": normal(k_r), "br": jnp.zeros((rank,)),
                  "Wh": normal(k_h), "bh": jnp.zeros((rank,))}
    return rec


def _cell(rec, s, x_in):
    """One step of the state update down the depth."""
    if "gru" in rec:
        g = rec["gru"]
        sx = jnp.concatenate([s, x_in], axis=-1)
        z = jax.nn.sigmoid(jnp.einsum("nhwi,ir->nhwr", sx, g["Wz"]) + g["bz"])
        r = jax.nn.sigmoid(jnp.einsum("nhwi,ir->nhwr", sx, g["Wr"]) + g["br"])
        h = jnp.tanh(jnp.einsum("nhwi,ir->nhwr", jnp.concatenate([r * s, x_in], axis=-1), g["Wh"]) + g["bh"])
        return (1 - z) * s + z * h
    gate = jax.nn.silu(x_in)
    return s + jnp.einsum("nhwr,rq->nhwq", gate * jnp.einsum("nhwr,rq->nhwq", s, rec["A"]), rec["B"])


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


def projection_flops(image=224):
    """Multiply-adds of each block's shortcut-projection VJP (0 for identity shortcuts).
    Every predictor runs these: the shortcut path is kept exact."""
    flops, h = [], image // 4
    for (pre, stride, projection), c_in, c_out in zip(resnet.blocks(), block_in_widths(), block_out_widths()):
        ho = h // stride
        flops.append(ho * ho * c_in * c_out if projection else 0)
        h = ho
    return np.array(flops, np.float64)


def recurrence_flops(rank, mix=True, image=224):
    """Multiply-adds of the recurrence step in each block (excluding the one-off seed)."""
    flops, h = [], image // 4
    for (pre, stride, _), c_in, c_out in zip(resnet.blocks(), block_in_widths(), block_out_widths()):
        mid, ho = c_out // resnet.EXPANSION, h // stride
        per_in = c_in * rank + 2 * rank * rank + rank * c_in + rank * mid + (9 * rank if mix else 0)  # C, A, B, U, V1, mix
        flops.append(h * h * per_in + ho * ho * rank * mid)                                          # + V2
        h = ho
    return np.array(flops, np.float64)


def seed_flops(rank, image=224):
    """Multiply-adds to seed the state from the exact error at each block's output (R)."""
    flops, h = [], image // 4
    for (pre, stride, _), c_out in zip(resnet.blocks(), block_out_widths()):
        ho = h // stride
        flops.append(ho * ho * c_out * rank)
        h = ho
    return np.array(flops, np.float64)


def method_cost(method, exact_top, rank=None, mix=True):
    """Error-propagation cost relative to the exact backward (all branch VJPs and
    projection VJPs). method: 'shortcut' | 'recurrence' | 'dfa'."""
    branch, proj = branch_flops(), projection_flops()
    total = branch.sum() + proj.sum()
    n = len(branch)
    start = n - 1 - exact_top
    if method == "dfa":  # one (1000 x C) projection of the logit error per tap, per image
        return float(sum(tap_widths().values()) * 1000 / total)
    cost = branch[start + 1:].sum() + proj.sum()
    if method == "recurrence":
        cost += recurrence_flops(rank, mix)[: start + 1].sum() + seed_flops(rank)[start]
    return float(cost / total)


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


# ----------------------------------------------------------------------------- low-rank backward

def init_lowrank(params, frac):
    """Rank-r factors of every branch conv, r = max(1, round(frac * mid)), from the SVD.

    1x1 convs: W (c_in, c_out) ~ P (c_in, r) @ Q (r, c_out).
    3x3 conv:  W as (9 c_in, c_out) ~ P @ Q, P reshaped to a (3, 3, c_in, r) kernel.
    """
    def factor(w, r):
        u, s, vt = np.linalg.svd(np.asarray(w, np.float64), full_matrices=False)
        return (u[:, :r] * s[:r]).astype(w.dtype), vt[:r].astype(w.dtype)

    blocks = []
    for pre, _, _ in resnet.blocks():
        w1, w2, w3 = (np.asarray(params[f"{pre}.conv{i}"]["w"]) for i in (1, 2, 3))
        mid = w1.shape[-1]
        r = max(1, round(frac * mid))
        p1, q1 = factor(w1.reshape(w1.shape[2], mid), r)
        p2, q2 = factor(w2.reshape(9 * mid, mid), r)
        p3, q3 = factor(w3.reshape(mid, w3.shape[-1]), r)
        blocks.append({"c1": {"P": p1, "Q": q1}, "c2": {"P": p2.reshape(3, 3, mid, r), "Q": q2},
                       "c3": {"P": p3, "Q": q3}})
    return {"blocks": blocks}


def init_precond(params):
    """A learned per-parameter scale on the EXACT BN gradient (a diagonal preconditioner).
    The control for 'is the gain from the learned update rule or from the predicted signal?'"""
    return {"log_scale": {n: {"scale": np.zeros_like(np.asarray(params[n]["scale"])),
                              "bias": np.zeros_like(np.asarray(params[n]["bias"]))} for n in resnet.bn_names()}}


def is_precond(pred) -> bool:
    return "log_scale" in pred


def apply_precond(pred, grads):
    return {n: {k: grads[n][k] * jnp.exp(pred["log_scale"][n][k]) for k in ("scale", "bias")} for n in grads}


def is_lowrank(pred) -> bool:
    return "c1" in pred["blocks"][0]


def predictor_kwargs(pred, inv_stds):
    """backward_over_depth keywords for a trained predictor of any kind."""
    return {"lowrank": pred, "inv_stds": inv_stds} if is_lowrank(pred) else {"rec": pred}


def _bn_backward(params, x_hats, inv_stds, name, dy):
    """dL/dx of a batch-statistics BN given dL/dy (exact, cheap)."""
    x_hat, mean = x_hats[name], lambda a: jnp.mean(a, axis=(0, 1, 2))
    return resnet.pin(params[name]["scale"] * inv_stds[name] * (dy - mean(dy) - x_hat * mean(dy * x_hat)))


def _lowrank_branch(params, x_hats, inv_stds, lr, pre, stride, g):
    """Branch backward with low-rank convs. g: error at bn3's output.
    Returns (dL/dh_in through the branch, delta at bn1, delta at bn2)."""
    mask = lambda name: relu_mask(params, x_hats, name)
    t1x1 = lambda dy, f: jnp.einsum("nhwc,rc,kr->nhwk", dy, f["Q"], f["P"])  # transpose of x @ P @ Q

    def conv2_lowrank(a):
        z = jax.lax.conv_general_dilated(a, lr["c2"]["P"], (stride, stride), [(1, 1), (1, 1)],
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
        return jnp.einsum("nhwr,rc->nhwc", z, lr["c2"]["Q"])

    dz3 = _bn_backward(params, x_hats, inv_stds, f"{pre}.bn3", g)
    delta2 = jnp.where(mask(f"{pre}.bn2"), t1x1(dz3, lr["c3"]), 0.0)
    dz2 = _bn_backward(params, x_hats, inv_stds, f"{pre}.bn2", delta2)
    _, vjp2 = jax.vjp(conv2_lowrank, jnp.zeros_like(x_hats[f"{pre}.bn1"]))
    delta1 = jnp.where(mask(f"{pre}.bn1"), vjp2(dz2)[0], 0.0)
    dz1 = _bn_backward(params, x_hats, inv_stds, f"{pre}.bn1", delta1)
    return t1x1(dz1, lr["c1"]), delta1, delta2


def lowrank_cost(frac):
    """Multiply-adds of the low-rank branch backward relative to the exact one (convs only)."""
    num = den = 0.0
    h = 224 // 4
    for (pre, stride, _), c_in, c_out in zip(resnet.blocks(), block_in_widths(), block_out_widths()):
        mid, ho = c_out // resnet.EXPANSION, h // stride
        r = max(1, round(frac * mid))
        den += h * h * c_in * mid + ho * ho * mid * mid * 9 + ho * ho * mid * c_out
        num += h * h * r * (c_in + mid) + ho * ho * r * (9 * mid + mid) + ho * ho * r * (mid + c_out)
        h = ho
    return num / den


# ----------------------------------------------------------------------------- backward

def backward_over_depth(params, stats, x_hats, block_io, stem, d_exact, exact_deltas=None, *,
                        exact_top=0, branch_term="none", rec=None, lowrank=None, inv_stds=None):
    """Error signals at all 53 BN taps.

    d_exact: exact dL/dh_out of every block (d_exact[-1] is the seed at the top
        of the residual stream). Only d_exact[n-1-exact_top] is used.
    exact_deltas: exact signals at every tap; supplies the branch taps of the
        exactly backpropagated blocks (indices > n-1-exact_top) only.
    exact_top: number of top blocks backpropagated exactly (may be traced).
    branch_term: "exact" | "none". rec: recurrence params.
    lowrank: low-rank factors (init_lowrank); needs inv_stds from the forward pass.
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
            if lowrank is not None:
                d_branch, delta1, delta2 = _lowrank_branch(
                    params, x_hats, inv_stds, lowrank["blocks"][k], pre, stride, g)
                d_next = d_next + d_branch
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
                s = _cell(rec, s, jnp.einsum("nhwc,cr->nhwr", a, r["C"]) + r["e"])
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


# ----------------------------------------------------------------------------- 1x1 conv gradients

def conv1x1_names() -> list[str]:
    """Every 1x1 conv: conv1 and conv3 of each block, and the projection shortcuts."""
    names = []
    for pre, _, projection in resnet.blocks():
        names += [f"{pre}.conv1", f"{pre}.conv3"] + ([f"{pre}.downsample.0"] if projection else [])
    return names


def conv1x1_grads(params, x_hats, inv_stds, block_io, deltas, names=None):
    """Weight gradients of 1x1 convs from the error signals at the BNs that follow them.

    A 1x1 conv z = x @ W feeds a batch-statistics BN; the error at z is that BN's
    exact backward of its output error (cheap), and dL/dW = sum over pixels of x^T dz.
    Inputs: conv1 reads the block input, conv3 reads relu(bn2), the projection
    reads the (strided) block input. Exact deltas give the exact gradient.
    """
    names = set(names or conv1x1_names())
    out = {}
    for k, (pre, stride, projection) in enumerate(resnet.blocks()):
        h_in = block_io[k][0]
        sources = {f"{pre}.conv1": (h_in, f"{pre}.bn1"),
                   f"{pre}.conv3": (jnp.maximum(params[f"{pre}.bn2"]["scale"] * x_hats[f"{pre}.bn2"]
                                                + params[f"{pre}.bn2"]["bias"], 0.0), f"{pre}.bn3")}
        if projection:
            sources[f"{pre}.downsample.0"] = (h_in[:, ::stride, ::stride, :], f"{pre}.downsample.1")
        for conv, (x_in, bn_name) in sources.items():
            if conv in names:
                dz = _bn_backward(params, x_hats, inv_stds, bn_name, deltas[bn_name])
                out[conv] = {"w": jnp.einsum("nhwc,nhwd->cd", x_in, dz)[None, None]}
    return out


# ----------------------------------------------------------------------------- DFA

def dfa_predict(W, e, masks):
    """delta_tap = mask * broadcast(e @ W_tap): one projection of the output error per tap."""
    return {tap: jnp.where(masks[tap], jnp.einsum("nk,kc->nc", e, W[tap])[:, None, None, :], 0.0)
            for tap in all_taps()}


def dfa_stats(e, deltas, masks):
    """Sufficient statistics for a ridge fit of e -> the per-channel mean of delta
    over active (unmasked) positions, the value the masked prediction should take."""
    def active_mean(tap):
        m = jnp.broadcast_to(masks[tap], deltas[tap].shape)
        return jnp.sum(deltas[tap], axis=(1, 2)) / jnp.maximum(jnp.sum(m, axis=(1, 2)), 1)

    return {"G": jnp.einsum("nk,nj->kj", e, e),
            "H": {tap: jnp.einsum("nk,nc->kc", e, active_mean(tap)) for tap in all_taps()}}


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
