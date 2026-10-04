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


def load_torchvision(path: str) -> tuple[dict, dict]:
    """Load timm/resnet50.tv_in1k safetensors into (params, stats)."""
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


def conv(p, x, stride=1):
    k = p["w"].shape[0]
    pad = (k - 1) // 2
    return jax.lax.conv_general_dilated(
        x, p["w"], (stride, stride), [(pad, pad), (pad, pad)],
        dimension_numbers=("NHWC", "HWIO", "NHWC"))


def batchnorm(p, s, x, batch_stats):
    """Returns (y, x_hat). Batch statistics use the biased variance, as in PyTorch."""
    if batch_stats:
        mean = jnp.mean(x, axis=(0, 1, 2))
        var = jnp.mean(jnp.square(x - mean), axis=(0, 1, 2))
    else:
        mean, var = s["mean"], s["var"]
    x_hat = (x - mean) * jax.lax.rsqrt(var + BN_EPS)
    return x_hat * p["scale"] + p["bias"], x_hat


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


def apply(params, stats, x, *, batch_stats, probes=None, return_blocks=False):
    """x: (N, 224, 224, 3) normalized images. Returns (logits, x_hats[, block_io]).

    probes: optional {bn_name: zeros like that BN's output}; added to the output.
    block_io: [(h_in, h_out)] for every block, i.e. the residual stream.
    """
    x_hats = {}

    def bn(name, h):
        y, x_hats[name] = batchnorm(params[name], stats[name], h, batch_stats)
        return y if probes is None else y + probes[name]

    h = jax.nn.relu(bn("bn1", conv(params["conv1"], x, stride=2)))
    h = maxpool(h)
    block_io = []
    for pre, stride, projection in blocks():
        h_in = h
        out = branch(params, pre, h, stride, bn)
        h = jax.nn.relu(out + shortcut(params, pre, h, stride, bn, projection))
        block_io.append((h_in, h))
    logits = head(params, h)
    return (logits, x_hats, block_io) if return_blocks else (logits, x_hats)


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
