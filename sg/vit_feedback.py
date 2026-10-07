"""Error signals at every LayerNorm of ViT-B/16, in closed form from the forward pass.

Block l (pre-LN) maps h_in -> h_mid -> h_out,

    h_mid = h_in  + proj(attn(qkv(norm1(h_in))))
    h_out = h_mid + fc2(gelu(fc1(norm2(h_mid)))),

so with g = dL/dh_out the backward over depth is

    d_fc1  = gelu'(u) * (g @ W_fc2^T)                      error at the fc1 output
    delta2 = d_fc1 @ W_fc1^T                               error at the norm2 output
    d_mid  = g + LN2^T(delta2)
    d_o    = d_mid @ W_proj^T                              error at the attention output
    d_qkv  = attention^T(d_o; A, q, k, v)                  exact, per head
    delta1 = d_qkv @ W_qkv^T                               error at the norm1 output
    d_in   = d_mid + LN1^T(delta1).

The residual stream is the identity; LN^T is the exact LayerNorm backward from (x_hat, 1/std,
scale). The attention backward is exact, from the saved probabilities A and q, k, v:

    dv = A^T d_o,  dA = d_o v^T,  dS = A * (dA - rowsum(dA * A)),
    dq = dS k / sqrt(d),  dk = dS^T q / sqrt(d).

Nothing here differentiates the network: every step is an einsum on saved forward quantities.
The only autodiff is dL/dlogits.

Low-rank feedback: each weight transpose (qkv, proj, fc1, fc2) can be replaced by low-rank
factors W ~ P Q (P (in, r), Q (r, out)) from the SVD of the real weight, r = round(frac *
min(in, out)) (= frac * 768 for every layer of ViT-B). frac = 1 is exact. The attention mixing
(value path and softmax backward) always stays exact. With exact_top = k the top k blocks use
the real weights, so the error at the output of block n-1-k is exact.

Cost per block, input gradients only (T tokens, width D, multiply-adds):
    forward:            12 D^2 T (qkv 3, proj 1, MLP 8)   + 2 T^2 D (scores, A v)
    exact backward:     12 D^2 T                          + 4 T^2 D (dv, dA, dq, dk)
    low-rank backward:  16 r D T (r (in + out) per layer) + 4 T^2 D
The exact attention backward is twice the forward attention mixing; at T = 197, D = 768 the
T^2 D terms are 4% (forward) and 8% (backward) of a block, so the backward costs about as much
as the forward. Low-rank beats exact transposes only below frac = 0.75. The largest tensors are
dA and dS, each the size of A (N, heads, T, T), as in the forward.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from sg.models import vit
from sg.models.resnet import pin

LINEARS = ("qkv", "proj", "fc1", "fc2")


def _linear_name(i, lin):
    return f"blocks.{i}.{'attn' if lin in ('qkv', 'proj') else 'mlp'}.{lin}"


class Signals(NamedTuple):
    logits: jax.Array
    deltas: dict      # {ln: dL/dy}, exact
    d_out: list       # [dL/dh_out per block], exact
    x_hats: dict      # {ln: normalized input}
    inv_stds: dict    # {ln: 1 / std}
    blocks: list      # per block forward quantities (see vit.apply)
    e: jax.Array      # dL/dlogits


class Backward(NamedTuple):
    deltas: dict      # {ln: error at the LN output}
    d_out: list       # [error at h_out, = at the fc2 output, per block]
    d_mid: list       # [error at h_mid, = at the proj output]
    d_fc1: list       # [error at the fc1 output (pre-GELU)]
    d_qkv: list       # [error at the fused qkv output (N, T, 3D)]
    d_embed: jax.Array  # error at the embedding output (input to block 0)


def exact_signals(params, x, loss_from_logits) -> Signals:
    """One forward + one backward with probes on every LN output and every block output."""
    fwd = lambda p, x: vit.apply(p, x, return_blocks=True)
    _, x_hat_shapes, _, block_shapes = jax.eval_shape(fwd, params, x)
    zeros = lambda sd: jnp.zeros(sd.shape, sd.dtype)
    probes = jax.tree.map(zeros, x_hat_shapes)
    stream = [zeros(b["h_out"]) for b in block_shapes]

    def loss(probes, stream):
        out = vit.apply(params, x, probes=probes, stream_probes=stream, return_blocks=True)
        return loss_from_logits(out[0]), out

    (deltas, d_out), (logits, x_hats, inv_stds, blocks) = jax.grad(loss, argnums=(0, 1), has_aux=True)(probes, stream)
    return Signals(logits, deltas, d_out, x_hats, inv_stds, blocks, jax.grad(loss_from_logits)(logits))


# ----------------------------------------------------------------------------- pieces

def ln_backward(scale, x_hat, inv_std, dy):
    """dL/dx of a LayerNorm (over the last axis) given dL/dy (exact)."""
    g = dy * scale
    mean = lambda a: jnp.mean(a, axis=-1, keepdims=True)
    return pin(inv_std[..., None] * (g - mean(g) - x_hat * mean(g * x_hat)))


def gelu_grad(u):
    """Derivative of the exact GELU u * Phi(u): Phi(u) + u * phi(u)."""
    return 0.5 * (1.0 + jax.lax.erf(u / math.sqrt(2.0))) + u * jnp.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)


def attention_backward(a, q, k, v, d_o):
    """Exact VJP of softmax attention with respect to q, k, v, all heads at once.
    a: (N, heads, T, T); q, k, v, d_o: (N, T, heads, hd). Returns d_qkv (N, T, 3 * heads * hd)
    in the fused-qkv layout."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    dv = jnp.einsum("nhqk,nqhd->nkhd", a, d_o)
    da = jnp.einsum("nqhd,nkhd->nhqk", d_o, v)
    ds = a * (da - jnp.sum(da * a, axis=-1, keepdims=True))
    dq = jnp.einsum("nhqk,nkhd->nqhd", ds, k) * scale
    dk = jnp.einsum("nhqk,nqhd->nkhd", ds, q) * scale
    n, t, h, d = q.shape
    return pin(jnp.stack([dq, dk, dv], axis=2).reshape(n, t, 3 * h * d))


def exact_transposes(params, i):
    """{layer: dy -> dy @ W^T} with the real weights of block i."""
    return {lin: (lambda dy, w=params[_linear_name(i, lin)]["w"]: pin(jnp.einsum("nto,io->nti", dy, w)))
            for lin in LINEARS}


def lowrank_transposes(factors):
    """{layer: dy -> dy @ Q^T @ P^T}, rank r: contracts with Q first (r x out), then P."""
    def t(f):
        return lambda dy: pin(jnp.einsum("ntr,ir->nti", jnp.einsum("nto,ro->ntr", dy, f["Q"]), f["P"]))
    return {lin: t(factors[lin]) for lin in LINEARS}


def block_backward(params, x_hats, inv_stds, blk, i, t, g):
    """Backward through block i from g = dL/dh_out, with weight transposes t.
    Returns (dL/dh_in, (delta at norm1, delta at norm2, d_mid, d_fc1, d_qkv))."""
    pre = f"blocks.{i}"
    ln = lambda name, dy: ln_backward(params[name]["scale"], x_hats[name], inv_stds[name], dy)
    d_fc1 = t["fc2"](g) * gelu_grad(blk["u"])
    delta2 = t["fc1"](d_fc1)
    d_mid = g + ln(f"{pre}.norm2", delta2)
    d_o = t["proj"](d_mid).reshape(blk["q"].shape)
    d_qkv = attention_backward(blk["A"], blk["q"], blk["k"], blk["v"], d_o)
    delta1 = t["qkv"](d_qkv)
    d_in = d_mid + ln(f"{pre}.norm1", delta1)
    return d_in, (delta1, delta2, d_mid, d_fc1, d_qkv)


# ----------------------------------------------------------------------------- low-rank feedback

def init_lowrank(params, frac):
    """Rank-r factors of every block's qkv, proj, fc1, fc2 from the SVD of the real weight:
    W (in, out) ~ P (in, r) @ Q (r, out), r = max(1, round(frac * min(in, out)))."""
    def factor(w):
        w = np.asarray(w)
        r = max(1, round(frac * min(w.shape)))
        u, s, vt = np.linalg.svd(np.asarray(w, np.float64), full_matrices=False)
        return {"P": (u[:, :r] * s[:r]).astype(w.dtype), "Q": vt[:r].astype(w.dtype)}

    return {"blocks": [{lin: factor(params[_linear_name(i, lin)]["w"]) for lin in LINEARS}
                       for i in range(vit.depth(params))]}


def lowrank_cost(frac, exact_top=0, tokens=197, width=vit.WIDTH, n_blocks=vit.DEPTH):
    """Multiply-adds of the low-rank backward (attention mixing exact) relative to the exact one."""
    r = max(1, round(frac * width))
    mix = 4 * tokens * tokens * width
    exact = 12 * width * width * tokens + mix
    low = 16 * r * width * tokens + mix
    return (exact_top * exact + (n_blocks - exact_top) * low) / (n_blocks * exact)


# ----------------------------------------------------------------------------- backward

def backward_over_depth(params, x_hats, inv_stds, blocks, e, *, lowrank=None, exact_top=0) -> Backward:
    """Errors at every LN output from e = dL/dlogits, top to bottom, in closed form.

    lowrank: factors from init_lowrank; None uses the real weights everywhere (exact).
    exact_top: number of top blocks that use the real weights anyway (may be traced: then
        each block picks its transposes with lax.cond, which stays a branch under vmap as
        long as exact_top is not batched).
    """
    n = len(blocks)
    head = params["head"]["w"]
    delta_top = jnp.zeros_like(x_hats["norm"]).at[:, 0].set(jnp.einsum("nk,ck->nc", e, head))
    g = ln_backward(params["norm"]["scale"], x_hats["norm"], inv_stds["norm"], delta_top)
    deltas, d_out, d_mid, d_fc1, d_qkv = {"norm": delta_top}, [None] * n, [None] * n, [None] * n, [None] * n
    for i in reversed(range(n)):
        d_out[i] = g
        step = lambda t, g: block_backward(params, x_hats, inv_stds, blocks[i], i, t, g)
        exact = lambda g: step(exact_transposes(params, i), g)
        if lowrank is None:
            g, out = exact(g)
        else:
            low = lambda g: step(lowrank_transposes(lowrank["blocks"][i]), g)
            paid = i >= n - exact_top
            if isinstance(paid, (bool, np.bool_)):
                g, out = (exact if paid else low)(g)
            else:
                g, out = jax.lax.cond(paid, exact, low, g)
        deltas[f"blocks.{i}.norm1"], deltas[f"blocks.{i}.norm2"], d_mid[i], d_fc1[i], d_qkv[i] = out
    return Backward(deltas, d_out, d_mid, d_fc1, d_qkv, g)


# ----------------------------------------------------------------------------- gradients

def mlp_grads(params, x_hats, blocks, bwd: Backward) -> dict:
    """fc1/fc2 weight and bias gradients from the errors at their outputs:
    dL/dW = sum over tokens and batch of input^T error, dL/db = sum of error.
    fc1 reads norm2's output, fc2 reads gelu(u)."""
    out = {}
    for i, blk in enumerate(blocks):
        ln = params[f"blocks.{i}.norm2"]
        sources = {"fc1": (x_hats[f"blocks.{i}.norm2"] * ln["scale"] + ln["bias"], bwd.d_fc1[i]),
                   "fc2": (jax.nn.gelu(blk["u"], approximate=False), bwd.d_out[i])}
        for lin, (x_in, dz) in sources.items():
            out[_linear_name(i, lin)] = {"w": jnp.einsum("ntc,ntd->cd", x_in, dz), "b": jnp.einsum("ntd->d", dz)}
    return out


def attn_grads(params, x_hats, blocks, bwd: Backward) -> dict:
    """qkv/proj weight and bias gradients, formed like mlp_grads.
    qkv reads norm1's output, proj reads the attention output o."""
    out = {}
    for i, blk in enumerate(blocks):
        ln = params[f"blocks.{i}.norm1"]
        sources = {"qkv": (x_hats[f"blocks.{i}.norm1"] * ln["scale"] + ln["bias"], bwd.d_qkv[i]),
                   "proj": (blk["o"], bwd.d_mid[i])}
        for lin, (x_in, dz) in sources.items():
            out[_linear_name(i, lin)] = {"w": jnp.einsum("ntc,ntd->cd", x_in, dz), "b": jnp.einsum("ntd->d", dz)}
    return out


def grads(params, x, loss_from_logits, *, lowrank=None, exact_top=0):
    """Forward, closed-form backward, then the LN and MLP gradients.
    Returns (logits, {name: grads}) for every LN and every fc1/fc2."""
    logits, x_hats, inv_stds, blocks = vit.apply(params, x, return_blocks=True)
    e = jax.grad(loss_from_logits)(logits)
    bwd = backward_over_depth(params, x_hats, inv_stds, blocks, e, lowrank=lowrank, exact_top=exact_top)
    return logits, vit.ln_grads(bwd.deltas, x_hats) | mlp_grads(params, x_hats, blocks, bwd)
