"""ViT-B/16 (timm vit_base_patch16_224.augreg2_in21k_ft_in1k) in plain JAX.

Pre-LN transformer: tokens = [CLS, 196 patches] + learned position embedding, then 12 blocks

    h_mid = h_in  + proj(attn(norm1(h_in)))
    h_out = h_mid + fc2(gelu(fc1(norm2(h_mid))))

and logits = head(norm(h)[CLS]). Fused qkv, 12 heads of 64, exact (erf) GELU, LayerNorm eps 1e-6.
Inputs are (N, 224, 224, 3), normalized with mean = std = 0.5 (not the ImageNet statistics).

params is a plain dict keyed like timm ("blocks.3.attn.qkv" -> {"w", "b"}). Linear weights
are stored (in, out), so a layer is x @ w + b; LayerNorms are {"scale", "bias"}. A model
with fewer blocks is the same dict without the upper blocks (see `truncate`).

`apply` can add a zero probe to every LayerNorm output. The gradient of the loss with respect
to a probe is the error delta = dL/dy at that LN output, so one backward pass returns delta at
every LN, as the BN probes do in sg.models.resnet. With return_blocks it also returns what the
closed-form backward (sg.vit_feedback) needs: per LN x_hat and 1/std, per block the residual
stream, q, k, v, the attention probabilities A, the attention output and the fc1
pre-activation.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from sg.models.resnet import pin

DEPTH = 12
WIDTH = 768
HEADS = 12
MLP = 4 * WIDTH
PATCH = 16
IMAGE = 224
LN_EPS = 1e-6
MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.5, 0.5, 0.5], np.float32)


def normalize(x_uint8):
    """uint8 NHWC -> normalized float32 for this model. Works on numpy or jax arrays."""
    return (x_uint8 / 255.0 - MEAN) / STD


def depth(params) -> int:
    return sum(1 for k in params if k.endswith(".norm1"))


def truncate(params, n_blocks):
    """The same network with only its first n_blocks blocks (then the final norm and head)."""
    keep = lambda k: not k.startswith("blocks.") or int(k.split(".")[1]) < n_blocks
    return {k: v for k, v in params.items() if keep(k)}


def ln_names(n_blocks=DEPTH) -> list[str]:
    """All LayerNorms in forward order."""
    return [f"blocks.{i}.norm{j}" for i in range(n_blocks) for j in (1, 2)] + ["norm"]


def mlp_names(n_blocks=DEPTH) -> list[str]:
    return [f"blocks.{i}.mlp.fc{j}" for i in range(n_blocks) for j in (1, 2)]


def load_timm(path: str) -> dict:
    """Load the timm safetensors file into params (float32 numpy)."""
    from safetensors.numpy import load_file

    w = load_file(path)
    n = 1 + max(int(k.split(".")[1]) for k in w if k.startswith("blocks."))
    params = {
        "patch_embed": {"w": np.transpose(w["patch_embed.proj.weight"], (2, 3, 1, 0)),  # OIHW -> HWIO
                        "b": w["patch_embed.proj.bias"]},
        "cls_token": w["cls_token"].reshape(-1),
        "pos_embed": w["pos_embed"][0],
    }
    linears = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
    for i in range(n):
        for j in (1, 2):
            pre = f"blocks.{i}.norm{j}"
            params[pre] = {"scale": w[f"{pre}.weight"], "bias": w[f"{pre}.bias"]}
        for lin in linears:
            pre = f"blocks.{i}.{lin}"
            params[pre] = {"w": np.ascontiguousarray(w[f"{pre}.weight"].T), "b": w[f"{pre}.bias"]}
    params["norm"] = {"scale": w["norm.weight"], "bias": w["norm.bias"]}
    params["head"] = {"w": np.ascontiguousarray(w["head.weight"].T), "b": w["head.bias"]}
    return params


def layernorm(p, x):
    """Over the last axis. Returns (y, x_hat, inv_std), inv_std without the feature axis."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    inv_std = jax.lax.rsqrt(var + LN_EPS)
    x_hat = pin((x - mean) * inv_std)
    return pin(x_hat * p["scale"] + p["bias"]), x_hat, inv_std[..., 0]


def linear(p, x):
    return pin(jnp.einsum("ntc,cd->ntd", x, p["w"]) + p["b"])


def embed(params, x):
    """Patchify (a 16x16 stride-16 conv as one einsum), prepend CLS, add position embedding."""
    n, hh, ww, c = x.shape
    gh, gw = hh // PATCH, ww // PATCH
    patches = x.reshape(n, gh, PATCH, gw, PATCH, c)
    tok = jnp.einsum("nyhxwc,hwcd->nyxd", patches, params["patch_embed"]["w"]) + params["patch_embed"]["b"]
    tok = tok.reshape(n, gh * gw, -1)
    cls = jnp.broadcast_to(params["cls_token"], (n, 1, tok.shape[-1])).astype(tok.dtype)
    return pin(jnp.concatenate([cls, tok], axis=1) + params["pos_embed"])


def split_heads(qkv):
    """(N, T, 3D) -> q, k, v each (N, T, heads, head_dim), the timm layout of the fused qkv."""
    n, t, _ = qkv.shape
    qkv = qkv.reshape(n, t, 3, HEADS, -1)
    return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]


def attention(q, k, v):
    """Softmax attention per head. Returns (out (N, T, heads, hd), A (N, heads, T, T))."""
    s = jnp.einsum("nqhd,nkhd->nhqk", q, k) / math.sqrt(q.shape[-1])
    a = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("nhqk,nkhd->nqhd", a, v), a


def apply(params, x, *, probes=None, stream_probes=None, return_blocks=False):
    """x: (N, 224, 224, 3) normalized images. Returns (logits, x_hats[, inv_stds, blocks]).

    probes: optional {ln_name: zeros like that LN's output (N, T, D)}; added to the output.
    stream_probes: optional [zeros like each block's output]; added to h_out, so their
        gradient is dL/dh_out of every block.
    x_hats, inv_stds: {ln_name: normalized input (N, T, D)}, {ln_name: 1 / std (N, T)}.
    blocks: per block a dict of
        h_in, h_mid, h_out  the residual stream (N, T, D) before, between and after the branches
        q, k, v             (N, T, heads, hd), before the 1/sqrt(hd) scale
        A                   attention probabilities (N, heads, T, T)
        o                   attention output, the input to proj (N, T, D)
        u                   fc1 pre-activation (N, T, 4D), for GELU'
    """
    x_hats, inv_stds, blocks = {}, {}, []

    def ln(name, h):
        y, x_hats[name], inv_stds[name] = layernorm(params[name], h)
        return y if probes is None else y + probes[name]

    h = embed(params, x)
    for i in range(depth(params)):
        pre = f"blocks.{i}"
        h_in = h
        q, k, v = split_heads(linear(params[f"{pre}.attn.qkv"], ln(f"{pre}.norm1", h)))
        o, a = attention(q, k, v)
        o = o.reshape(h.shape)
        h_mid = h + linear(params[f"{pre}.attn.proj"], o)
        u = linear(params[f"{pre}.mlp.fc1"], ln(f"{pre}.norm2", h_mid))
        h = h_mid + linear(params[f"{pre}.mlp.fc2"], jax.nn.gelu(u, approximate=False))
        if stream_probes is not None:
            h = h + stream_probes[i]
        blocks.append({"h_in": h_in, "h_mid": h_mid, "h_out": h, "q": q, "k": k, "v": v, "A": a, "o": o, "u": u})
    feats = ln("norm", h)[:, 0]
    logits = jnp.einsum("nc,ck->nk", feats, params["head"]["w"]) + params["head"]["b"]
    return (logits, x_hats, inv_stds, blocks) if return_blocks else (logits, x_hats)


def probe_shapes(params, x) -> dict:
    """Shape/dtype of every LN output for input x (traced only, no compute)."""
    return jax.eval_shape(lambda p, x: apply(p, x)[1], params, x)


def ln_grads(deltas: dict, x_hats: dict) -> dict:
    """Closed-form LN parameter gradients from error signals at the LN outputs:
    dL/dscale = sum_{n,t} delta * x_hat,  dL/dbias = sum_{n,t} delta."""
    return {
        name: {
            "scale": jnp.einsum("ntc,ntc->c", deltas[name], x_hats[name]),
            "bias": jnp.einsum("ntc->c", deltas[name]),
        }
        for name in deltas
    }
