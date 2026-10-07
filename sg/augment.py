"""The test-time augmentations of CoTTA and ROID (their get_tta_transforms), in JAX.

  ColorJitterPro    brightness, contrast, saturation, hue (HSV), gamma, in a random
                    order, each clamped to [0, 1] as torchvision does
  pad + RandomAffine  padding of H/2 (edge or reflect), rotation, translation up to 1/16 of
                    the padded size, scale; bilinear; then a center crop back to H
  GaussianBlur      kernel 5 (CoTTA only)
  flip              horizontal, p = 0.5
  GaussianNoise     std 0.005 (CoTTA only)
Strong ranges by default (soft=False), as both use on ImageNet. One random draw per call
for the whole batch, as their transforms on a batched tensor do. Approximation: the blur
is applied after the crop, with edge padding (sigma <= 0.5, so it barely reaches a border).
"""

import jax
import jax.numpy as jnp
import numpy as np

GRAY = np.array([0.2989, 0.587, 0.114], np.float32)  # numpy: no JAX calls at import


def _rgb_to_hsv(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    maxc, minc = jnp.max(x, -1), jnp.min(x, -1)
    eqc = maxc == minc
    cr = maxc - minc
    s = cr / jnp.where(eqc, 1.0, maxc)
    crd = jnp.where(eqc, 1.0, cr)
    rc, gc, bc = (maxc - r) / crd, (maxc - g) / crd, (maxc - b) / crd
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = jnp.mod((hr + hg + hb) / 6.0 + 1.0, 1.0)
    return h, s, maxc


def _hsv_to_rgb(h, s, v):
    i = jnp.floor(h * 6.0)
    f = h * 6.0 - i
    i = jnp.mod(i, 6).astype(jnp.int32)
    p = jnp.clip(v * (1 - s), 0, 1)
    q = jnp.clip(v * (1 - s * f), 0, 1)
    t = jnp.clip(v * (1 - s * (1 - f)), 0, 1)
    table = jnp.stack([jnp.stack(c, -1) for c in
                       ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))])  # (6, ..., 3)
    return jnp.take_along_axis(table, i[None, ..., None], 0)[0]


def color_jitter_pro(key, x, soft=False):
    k = jax.random.split(key, 6)
    lo_hi = {"b": (0.8, 1.2) if soft else (0.6, 1.4), "c": (0.85, 1.15) if soft else (0.7, 1.3),
             "s": (0.75, 1.25) if soft else (0.5, 1.5), "h": (-0.03, 0.03) if soft else (-0.06, 0.06),
             "g": (0.85, 1.15) if soft else (0.7, 1.3)}
    u = {n: jax.random.uniform(k[i], (), minval=a, maxval=b) for i, (n, (a, b)) in enumerate(lo_hi.items())}
    clip = lambda z: jnp.clip(z, 0.0, 1.0)
    gray = lambda z: z @ GRAY

    def contrast(z):
        m = jnp.mean(gray(z), axis=(1, 2))[:, None, None, None]
        return clip(u["c"] * z + (1 - u["c"]) * m)

    def hue(z):
        h, s, v = _rgb_to_hsv(z)
        return clip(_hsv_to_rgb(jnp.mod(h + u["h"], 1.0), s, v))

    ops = [lambda z: clip(u["b"] * z), contrast,
           lambda z: clip(u["s"] * z + (1 - u["s"]) * gray(z)[..., None]), hue,
           lambda z: clip(jnp.clip(z, 1e-8, 1.0) ** u["g"])]
    order = jax.random.permutation(k[5], 5)
    for i in range(5):
        x = jax.lax.switch(order[i], ops, x)
    return x


def affine(key, x, soft=False, pad_mode="edge"):
    """Pad by H/2, random affine (bilinear) about the center, center crop back."""
    k = jax.random.split(key, 4)
    n = x.shape[1]
    deg = 8.0 if soft else 15.0
    lo, hi = (0.95, 1.05) if soft else (0.9, 1.1)
    ang = jnp.deg2rad(jax.random.uniform(k[0], (), minval=-deg, maxval=deg))
    max_t = 2 * n / 16  # 1/16 of the padded size
    tx = jnp.round(jax.random.uniform(k[1], (), minval=-max_t, maxval=max_t))
    ty = jnp.round(jax.random.uniform(k[2], (), minval=-max_t, maxval=max_t))
    sc = jax.random.uniform(k[3], (), minval=lo, maxval=hi)
    c = (n - 1) / 2
    yy, xx = jnp.meshgrid(jnp.arange(n, dtype=jnp.float32), jnp.arange(n, dtype=jnp.float32), indexing="ij")
    qx, qy = xx - c - tx, yy - c - ty
    sx = (jnp.cos(ang) * qx + jnp.sin(ang) * qy) / sc + c
    sy = (-jnp.sin(ang) * qx + jnp.cos(ang) * qy) / sc + c
    x0, y0 = jnp.floor(sx), jnp.floor(sy)
    wx, wy = (sx - x0)[None, ..., None], (sy - y0)[None, ..., None]
    if pad_mode == "edge":
        idx = lambda i: jnp.clip(i, 0, n - 1)
    else:  # reflect, without repeating the edge pixel (pad < n)
        idx = lambda i: jnp.abs(jnp.where(i > n - 1, 2 * (n - 1) - i, i))
    at = lambda yi, xi: x[:, idx(yi).astype(jnp.int32), idx(xi).astype(jnp.int32), :]
    return ((1 - wy) * ((1 - wx) * at(y0, x0) + wx * at(y0, x0 + 1))
            + wy * ((1 - wx) * at(y0 + 1, x0) + wx * at(y0 + 1, x0 + 1)))


def blur(key, x, soft=False):
    sigma = jax.random.uniform(key, (), minval=0.001, maxval=0.25 if soft else 0.5)
    w = jnp.exp(-jnp.arange(-2, 3, dtype=jnp.float32) ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    n = x.shape[1]
    pad = jnp.pad(x, ((0, 0), (2, 2), (0, 0), (0, 0)), mode="edge")
    x = sum(w[i] * pad[:, i:i + n] for i in range(5))
    pad = jnp.pad(x, ((0, 0), (0, 0), (2, 2), (0, 0)), mode="edge")
    return sum(w[i] * pad[:, :, i:i + n] for i in range(5))


def tta_augment(key, x, *, soft=False, pad_mode="edge", blur_noise=True):
    """One augmentation of a batch x (B, H, W, 3) in [0, 1]. CoTTA: blur_noise=True, edge padding;
    ROID: blur_noise=False, reflect padding."""
    k = jax.random.split(key, 5)
    x = color_jitter_pro(k[0], jnp.clip(x, 0.0, 1.0), soft)
    x = affine(k[1], x, soft, pad_mode)
    if blur_noise:
        x = blur(k[2], x, soft)
    x = jnp.where(jax.random.uniform(k[3]) < 0.5, x[:, :, ::-1], x)
    if blur_noise:
        x = x + 0.005 * jax.random.normal(k[4], x.shape)
    return jnp.clip(x, 0.0, 1.0)
