"""The backward-over-depth code is exact when given the exact branch term.

Run: uv run --group dev pytest tests/test_feedback.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sg import feedback, tent
from sg.models import resnet

WEIGHTS = os.environ.get(
    "SG_WEIGHTS", "/scratch/gpfs/ZHUANGL/sk7524/data/weights/resnet50_tv_in1k.safetensors")


@pytest.fixture(scope="module")
def signals():
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"weights not found: {WEIGHTS}")
    with jax.enable_x64(True):
        params, stats = resnet.load_torchvision(WEIGHTS)
        params, stats = (jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), t) for t in (params, stats))
        x = jax.random.normal(jax.random.key(0), (4, 96, 96, 3), jnp.float64)
        _, deltas, _ = tent.exact_signals(params, stats, x)
        logits, x_hats, block_io = resnet.apply(params, stats, x, batch_stats=True, return_blocks=True)
        d_top, e = feedback.seed(params, block_io[-1][1], tent.entropy)
        yield params, stats, x_hats, block_io, d_top, e, deltas


def test_exact_branch_reproduces_backprop(signals):
    params, stats, _, block_io, d_top, _, deltas = signals
    with jax.enable_x64(True):
        pred = feedback.backward_over_depth(params, stats, block_io, d_top, branch_term="exact")
        assert set(pred) == set(feedback.residual_taps())
        for tap in pred:
            np.testing.assert_allclose(pred[tap], deltas[tap], rtol=1e-8, atol=1e-14 * float(jnp.abs(deltas[tap]).max()))


def test_top_tap_is_exact_for_every_method(signals):
    """The last block's taps see only the exact seed, so every method is exact there."""
    params, stats, x_hats, block_io, d_top, _, deltas = signals
    with jax.enable_x64(True):
        rec = feedback.init_recurrence(jax.random.key(1))
        rec = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), rec)
        for kwargs in ({"branch_term": "none"}, {"rec": rec}):
            pred = feedback.backward_over_depth(params, stats, block_io, d_top, **kwargs)
            m = feedback.compare(pred, deltas, x_hats)
            top = feedback.residual_taps().index("layer4.2.bn3")
            np.testing.assert_allclose(m[top], [1, 1, 1, 1], atol=1e-10)


def test_zero_U_recurrence_equals_shortcut_only(signals):
    params, stats, _, block_io, d_top, _, _ = signals
    with jax.enable_x64(True):
        rec = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), feedback.init_recurrence(jax.random.key(1)))
        a = feedback.backward_over_depth(params, stats, block_io, d_top, rec=rec)
        b = feedback.backward_over_depth(params, stats, block_io, d_top, branch_term="none")
        for tap in a:
            np.testing.assert_allclose(a[tap], b[tap], rtol=1e-12)


def test_dfa_fit_recovers_a_linear_map():
    """If mean_hw(delta) is exactly linear in e, the ridge fit recovers it."""
    key = jax.random.key(2)
    e = jax.random.normal(key, (4000, 1000))
    W = feedback.dfa_random(jax.random.key(3))
    block_io = [(None, jnp.ones((4000, 1, 1, 1)))] * len(resnet.blocks())  # mask = all ones
    deltas = feedback.dfa_predict(W, e, block_io)
    fit = feedback.dfa_solve(feedback.dfa_stats(e, deltas), ridge=1e-8)
    for tap in ["layer1.0.bn3", "layer4.2.bn3"]:
        np.testing.assert_allclose(fit[tap], W[tap], rtol=1e-3, atol=1e-3)
