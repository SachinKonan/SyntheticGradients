"""Gate 1 (correctness), on CPU with the real pretrained weights.

1. Closed-form BN gradients from probe deltas equal jax.grad of the entropy.
2. A Tent step built from deltas equals a Tent step built from jax.grad.

Run: uv run --group dev pytest tests/test_gate1.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sg import tent
from sg.models import resnet

WEIGHTS = os.environ.get(
    "SG_WEIGHTS", "/scratch/gpfs/ZHUANGL/sk7524/data/weights/resnet50_tv_in1k.safetensors")

jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"weights not found: {WEIGHTS}")
    params, stats = resnet.load_torchvision(WEIGHTS)
    return jax.tree.map(jnp.asarray, params), jax.tree.map(jnp.asarray, stats)


@pytest.fixture(scope="module")
def batch():
    return jax.random.normal(jax.random.key(0), (4, 96, 96, 3), jnp.float32)


def autodiff_bn_grads(params, stats, x):
    def loss(bn):
        logits, _ = resnet.apply({**params, **bn}, stats, x, batch_stats=True)
        return tent.entropy(logits)

    return jax.grad(loss)(tent.bn_params(params))


def rel_err(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-30))


def test_bn_grads_match_autodiff(model, batch):
    params, stats = model
    _, deltas, x_hats = jax.jit(tent.exact_signals)(params, stats, batch)
    ours = resnet.bn_grads(deltas, x_hats)
    ref = jax.jit(autodiff_bn_grads)(params, stats, batch)
    assert set(ours) == set(ref) == set(resnet.bn_names())
    worst = max(rel_err(ours[n][k], ref[n][k]) for n in ref for k in ("scale", "bias"))
    assert worst < 1e-4, worst


def test_tent_step_matches_autodiff(model, batch):
    params, stats = model
    velocity = jax.tree.map(jnp.zeros_like, tent.bn_params(params))
    new_params, _, _, _ = jax.jit(tent.tent_step)(params, velocity, stats, batch)

    grads = jax.jit(autodiff_bn_grads)(params, stats, batch)
    ref_bn, _ = tent.sgd_momentum(tent.bn_params(params), grads, velocity, 2.5e-4, 0.9)
    for name in resnet.bn_names():
        for k in ("scale", "bias"):
            np.testing.assert_allclose(new_params[name][k], ref_bn[name][k], rtol=1e-6, atol=1e-9)
