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
N_BLOCKS = len(resnet.blocks())


def f64(tree):
    return jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), tree)


@pytest.fixture(scope="module")
def signals():
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"weights not found: {WEIGHTS}")
    with jax.enable_x64(True):
        params, stats = f64(resnet.load_torchvision(WEIGHTS))
        x = jax.random.normal(jax.random.key(0), (4, 96, 96, 3), jnp.float64)
        _, deltas, d_stream, x_hats, block_io, stem, _ = feedback.exact_signals(params, stats, x, tent.entropy)
        yield params, stats, x_hats, block_io, stem, d_stream, deltas


def assert_taps_equal(pred, deltas, taps=None):
    for tap in taps or pred:
        np.testing.assert_allclose(pred[tap], deltas[tap], rtol=1e-8,
                                   atol=1e-14 * float(jnp.abs(deltas[tap]).max()), err_msg=tap)


def block_of(tap):
    """'layer2.0.downsample.1' -> 'layer2.0'."""
    return ".".join(tap.split(".")[:2])


def test_exact_signals_match_tent(signals):
    params, stats, *_, deltas = signals
    with jax.enable_x64(True):
        x = jax.random.normal(jax.random.key(0), (4, 96, 96, 3), jnp.float64)
        _, ref, _ = tent.exact_signals(params, stats, x)
        assert_taps_equal(deltas, ref)


def test_exact_branch_reproduces_backprop(signals):
    """All 53 taps, including the branch BNs and the stem."""
    params, stats, x_hats, block_io, stem, d_stream, deltas = signals
    with jax.enable_x64(True):
        pred = feedback.backward_over_depth(params, stats, x_hats, block_io, stem, d_stream, branch_term="exact")
        assert set(pred) == set(feedback.all_taps())
        assert_taps_equal(pred, deltas)


@pytest.mark.parametrize("exact_top", [0, 3, 7, N_BLOCKS - 1])
def test_hybrid_taps_exact_down_to_start(signals, exact_top):
    """Every tap of blocks >= n-1-exact_top is exact for every method, traced or not."""
    params, stats, x_hats, block_io, stem, d_stream, deltas = signals
    names = [p for p, _, _ in resnet.blocks()]
    start = N_BLOCKS - 1 - exact_top
    exact_taps = [t for t in feedback.all_taps() if t != "bn1" and names.index(block_of(t)) >= start]
    with jax.enable_x64(True):
        rec = f64(feedback.init_recurrence(jax.random.key(1)))
        rec = jax.tree.map(lambda a: a + 0.01, rec)  # nonzero U, V
        for kw in ({"branch_term": "none"}, {"rec": rec}):
            run = lambda k: feedback.backward_over_depth(
                params, stats, x_hats, block_io, stem, d_stream, deltas, exact_top=k, **kw)
            pred = run(exact_top)
            assert_taps_equal(pred, deltas, exact_taps)
            traced = jax.jit(run)(exact_top)
            for t in pred:
                np.testing.assert_allclose(traced[t], pred[t], rtol=1e-8, atol=1e-30, err_msg=t)


def test_zero_heads_recurrence_equals_shortcut_only(signals):
    params, stats, x_hats, block_io, stem, d_stream, _ = signals
    with jax.enable_x64(True):
        rec = f64(feedback.init_recurrence(jax.random.key(1)))
        a = feedback.backward_over_depth(params, stats, x_hats, block_io, stem, d_stream, rec=rec)
        b = feedback.backward_over_depth(params, stats, x_hats, block_io, stem, d_stream, branch_term="none")
        for tap in a:
            np.testing.assert_allclose(a[tap], b[tap], rtol=1e-12, err_msg=tap)


def test_full_metric_is_one_for_exact(signals):
    x_hats, deltas = signals[2], signals[-1]
    with jax.enable_x64(True):
        per_tap, full = feedback.compare(deltas, deltas, x_hats)
        assert per_tap.shape == (len(feedback.all_taps()), 4)
        np.testing.assert_allclose(per_tap, 1.0, atol=1e-10)
        np.testing.assert_allclose(full, 1.0, atol=1e-10)


def test_dfa_fit_recovers_a_linear_map():
    """If mean_hw(delta) is exactly linear in e, the ridge fit recovers it."""
    e = jax.random.normal(jax.random.key(2), (4000, 1000))
    W = feedback.dfa_random(jax.random.key(3))
    masks = {t: jnp.ones((4000, 1, 1, 1), bool) for t in feedback.all_taps()}
    deltas = feedback.dfa_predict(W, e, masks)
    fit = feedback.dfa_solve(feedback.dfa_stats(e, deltas), ridge=1e-8)
    for tap in ["bn1", "layer1.0.bn1", "layer4.2.bn3"]:
        np.testing.assert_allclose(fit[tap], W[tap], rtol=1e-3, atol=1e-3)


def test_tap_widths_match_model(signals):
    x_hats = signals[2]
    widths = feedback.tap_widths()
    assert set(widths) == set(feedback.all_taps())
    for t in widths:
        assert x_hats[t].shape[-1] == widths[t], t


def test_branch_flops_match_resnet50():
    # ResNet-50 is ~4.1 GMACs; stem + head are ~0.12 G, projections ~0.27 G.
    assert 3.5e9 < feedback.branch_flops().sum() < 3.9e9
