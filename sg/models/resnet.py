"""ResNet-50 (torchvision v1 weights) in plain JAX, NHWC.

Pure functions over two pytrees:
  params: trainable weights, keyed like torchvision ("layer2.0.bn1" -> {"scale", "bias"})
  stats:  BatchNorm running {"mean", "var"}, used only when batch_stats=False

`apply` can add a zero probe to each BatchNorm output. The gradient of the loss
with respect to a probe is the error signal delta = dL/dy at that BN output,
so one backward pass returns delta at every tap. It also returns the
normalized BN inputs x_hat, which with delta give the BN parameter gradients
in closed form (see `bn_grads`).
"""

import jax
import jax.numpy as jnp
import numpy as np

STAGES = (("layer1", 3, 64, 1), ("layer2", 4, 128, 2), ("layer3", 6, 256, 2), ("layer4", 3, 512, 2))
EXPANSION = 4
BN_EPS = 1e-5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def bn_names() -> list[str]:
    """All BatchNorm layers in forward order."""
    names = ["bn1"]
    for stage, blocks, _, _ in STAGES:
        for b in range(blocks):
            names += [f"{stage}.{b}.bn1", f"{stage}.{b}.bn2", f"{stage}.{b}.bn3"]
            if b == 0:
                names.append(f"{stage}.{b}.downsample.1")
    return names


def save_npz(file, params, stats):
    """(params, stats) as an .npz that load_torchvision also reads (e.g. a fine-tuned network)."""
    np.savez(file, **{f"{kind}/{n}/{k}": np.asarray(v) for kind, tree in (("params", params), ("stats", stats))
                      for n, d in tree.items() for k, v in d.items()})


def load_torchvision(path: str) -> tuple[dict, dict]:
    """Load timm/resnet50.tv_in1k safetensors (or an .npz from save_npz) into (params, stats)."""
    if str(path).endswith(".npz"):
        out = {"params": {}, "stats": {}}
        with np.load(path) as f:
            for key in f.files:
                kind, n, k = key.split("/")
                out[kind].setdefault(n, {})[k] = f[key]
        return out["params"], out["stats"]
    from safetensors.numpy import load_file

    w = load_file(path)
    params, stats = {}, {}
    for name in bn_names():
        params[name] = {"scale": w[f"{name}.weight"], "bias": w[f"{name}.bias"]}
        stats[name] = {"mean": w[f"{name}.running_mean"], "var": w[f"{name}.running_var"]}
    for k, v in w.items():
        if k.endswith(".weight") and v.ndim == 4:  # conv: OIHW -> HWIO
            params[k[: -len(".weight")]] = {"w": np.transpose(v, (2, 3, 1, 0))}
    params["fc"] = {"w": w["fc.weight"], "b": w["fc.bias"]}  # (classes, features)
    return params, stats


_batch_sharding = None


def shard_batch(sharding):
    """Pin the batch axis of every activation to `sharding` (e.g. NamedSharding(mesh, P("batch")))
    in code traced from now on, so a stream's batch stays split over its devices instead of
    being left to XLA's guess. Per-stream code must then be vmapped with spmd_axis_name.
    None turns it off."""
    global _batch_sharding
    _batch_sharding = sharding


def pin(x):
    return x if _batch_sharding is None else jax.lax.with_sharding_constraint(x, _batch_sharding)


_einsum_convs = False


def einsum_convs(on: bool):
    """Compute k x k convs as k^2 shifted einsums in code traced from now on: needed when every
    stream carries its own k x k kernels (vmapped lax.conv kernels gave wrong sharded results)."""
    global _einsum_convs
    _einsum_convs = on


def conv_taps(x, k, stride):
    """taps[dy][dx]: the input pixels a k x k conv (zero padding (k-1)//2) multiplies by kernel
    tap (dy, dx), one per output pixel."""
    pad = (k - 1) // 2
    xp = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0))) if pad else x
    h, w = (x.shape[1] + 2 * pad - k) // stride + 1, (x.shape[2] + 2 * pad - k) // stride + 1
    return [[xp[:, dy:dy + stride * (h - 1) + 1:stride, dx:dx + stride * (w - 1) + 1:stride, :]
             for dx in range(k)] for dy in range(k)]


def conv(p, x, stride=1):
    k = p["w"].shape[0]
    if k == 1:
        # A 1x1 conv is a per-pixel matmul. As an einsum it also shards correctly when every
        # stream carries its own weights (vmapped lax.conv kernels gave wrong sharded results).
        x = x[:, ::stride, ::stride, :] if stride > 1 else x
        return pin(jnp.einsum("nhwc,cd->nhwd", x, p["w"][0, 0]))
    if _einsum_convs:
        taps = conv_taps(x, k, stride)
        return pin(sum(jnp.einsum("nhwc,cd->nhwd", taps[dy][dx], p["w"][dy, dx])
                       for dy in range(k) for dx in range(k)))
    pad = (k - 1) // 2
    return pin(jax.lax.conv_general_dilated(
        x, p["w"], (stride, stride), [(pad, pad), (pad, pad)],
        dimension_numbers=("NHWC", "HWIO", "NHWC")))


def batchnorm(p, s, x, batch_stats, mask=None):
    """Returns (y, x_hat, inv_std). Batch statistics use the biased variance, as in PyTorch;
    mask (N,) restricts them to the selected examples (as if only those were in the batch)."""
    if batch_stats and mask is not None:
        w = mask.astype(x.dtype)[:, None, None, None]
        n = jnp.sum(w) * x.shape[1] * x.shape[2]
        mean = jnp.sum(w * x, axis=(0, 1, 2)) / n
        var = jnp.sum(w * jnp.square(x - mean), axis=(0, 1, 2)) / n
    elif batch_stats:
        mean = jnp.mean(x, axis=(0, 1, 2))
        var = jnp.mean(jnp.square(x - mean), axis=(0, 1, 2))
    else:
        mean, var = s["mean"], s["var"]
    inv_std = jax.lax.rsqrt(var + BN_EPS)
    x_hat = pin((x - mean) * inv_std)
    return pin(x_hat * p["scale"] + p["bias"]), x_hat, inv_std


def maxpool(x):
    return jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), [(0, 0), (1, 1), (1, 1), (0, 0)])


def blocks() -> list[tuple[str, int, bool]]:
    """(prefix, stride, has_projection) for every bottleneck block, in forward order."""
    return [(f"{stage}.{b}", stride if b == 0 else 1, b == 0)
            for stage, n, _, stride in STAGES for b in range(n)]


def plain_bn(params, stats, batch_stats):
    """bn(name, h) -> y, without probes or recording."""
    return lambda name, h: batchnorm(params[name], stats[name], h, batch_stats)[0]


def branch(params, pre, h, stride, bn):
    """Residual branch F(h) of a bottleneck block."""
    out = jax.nn.relu(bn(f"{pre}.bn1", conv(params[f"{pre}.conv1"], h)))
    out = jax.nn.relu(bn(f"{pre}.bn2", conv(params[f"{pre}.conv2"], out, stride=stride)))
    return bn(f"{pre}.bn3", conv(params[f"{pre}.conv3"], out))


def shortcut(params, pre, h, stride, bn, projection):
    """Shortcut P(h): identity, or 1x1 conv + BN where the shape changes."""
    if not projection:
        return h
    return bn(f"{pre}.downsample.1", conv(params[f"{pre}.downsample.0"], h, stride=stride))


def head(params, h):
    feats = jnp.mean(h, axis=(1, 2))
    return jnp.einsum("nc,kc->nk", feats, params["fc"]["w"]) + params["fc"]["b"]


def apply(params, stats, x, *, batch_stats, probes=None, stream_probes=None, return_blocks=False, batch_mask=None):
    """x: (N, 224, 224, 3) normalized images. Returns (logits, x_hats[, block_io, stem, inv_stds]).

    probes: optional {bn_name: zeros like that BN's output}; added to the output.
    stream_probes: optional [zeros like each block's output]; added to h_out, so
        their gradient is dL/dh_out of every block.
    block_io: [(h_in, h_out)] for every block, i.e. the residual stream.
    stem: relu(bn1(conv1(x))), the input to the max pool.
    inv_stds: {bn_name: 1 / std} used by each BN (batch or running statistics).
    batch_mask: optional (N,) bool; batch statistics over the selected examples only.
    """
    x_hats, inv_stds = {}, {}

    def bn(name, h):
        y, x_hats[name], inv_stds[name] = batchnorm(params[name], stats[name], h, batch_stats, batch_mask)
        return y if probes is None else y + probes[name]

    stem = jax.nn.relu(bn("bn1", conv(params["conv1"], x, stride=2)))
    h = maxpool(stem)
    block_io = []
    for k, (pre, stride, projection) in enumerate(blocks()):
        h_in = h
        out = branch(params, pre, h, stride, bn)
        h = jax.nn.relu(out + shortcut(params, pre, h, stride, bn, projection))
        if stream_probes is not None:
            h = h + stream_probes[k]
        block_io.append((h_in, h))
    logits = head(params, h)
    return (logits, x_hats, block_io, stem, inv_stds) if return_blocks else (logits, x_hats)


def probe_shapes(params, stats, x) -> dict:
    """Shape/dtype of every BN output for input x (traced only, no compute)."""

    def record(p, s, x):
        _, x_hats = apply(p, s, x, batch_stats=True)
        return x_hats

    return jax.eval_shape(record, params, stats, x)


def bn_grads(deltas: dict, x_hats: dict) -> dict:
    """Closed-form BN parameter gradients from error signals:
    dL/dscale = sum_n delta_n * x_hat_n,  dL/dbias = sum_n delta_n."""
    return {
        name: {
            "scale": jnp.einsum("nhwc,nhwc->c", deltas[name], x_hats[name]),
            "bias": jnp.einsum("nhwc->c", deltas[name]),
        }
        for name in deltas
    }
