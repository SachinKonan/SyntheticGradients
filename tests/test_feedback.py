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
        _, deltas, d_stream, x_hats, block_io, stem, _, _ = feedback.exact_signals(params, stats, x, tent.entropy)
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


def is_branch_tap(tap):
    return tap.endswith(".bn1") or tap.endswith(".bn2")


@pytest.mark.parametrize("exact_top", [0, 3, 7, N_BLOCKS - 1])
def test_hybrid_uses_exactly_what_it_pays_for(signals, exact_top):
    """exact_top = k pays for the branch VJPs of the top k blocks. Then:
    exact: residual taps of blocks >= n-1-k, branch taps of blocks >= n-k;
    not exact: the branch taps of block n-1-k and everything below.
    Holds for every method, traced or not."""
    params, stats, x_hats, block_io, stem, d_stream, deltas = signals
    names = [p for p, _, _ in resnet.blocks()]
    start = N_BLOCKS - 1 - exact_top
    idx = lambda t: names.index(block_of(t))
    exact_taps = [t for t in feedback.all_taps() if t != "bn1" and (
        idx(t) > start or (idx(t) == start and not is_branch_tap(t)))]
    unpaid = [t for t in feedback.all_taps() if t != "bn1" and is_branch_tap(t) and idx(t) <= start]
    assert f"{names[start]}.bn1" in unpaid and f"{names[start]}.bn2" in unpaid
    with jax.enable_x64(True):
        rec = f64(feedback.init_recurrence(jax.random.key(1)))
        rec = jax.tree.map(lambda a: a + 0.01, rec)  # nonzero U, V
        for kw in ({"branch_term": "none"}, {"rec": rec}):
            run = lambda k: feedback.backward_over_depth(
                params, stats, x_hats, block_io, stem, d_stream, deltas, exact_top=k, **kw)
            pred = run(exact_top)
            assert_taps_equal(pred, deltas, exact_taps)
            for t in unpaid:  # must not be the exact signal
                rel = float(jnp.linalg.norm(pred[t] - deltas[t]) / jnp.linalg.norm(deltas[t]))
                assert rel > 1e-3, (t, kw.keys(), rel)
            traced = jax.jit(run)(exact_top)
            for t in pred:
                np.testing.assert_allclose(traced[t], pred[t], rtol=1e-8, atol=1e-30, err_msg=t)


def test_exact_inputs_do_not_leak(signals):
    """Predictions at k=0 must not change if every exact quantity except the seed
    (d_stream[-1]) and the paid-for taps is replaced by garbage."""
    params, stats, x_hats, block_io, stem, d_stream, deltas = signals
    with jax.enable_x64(True):
        rec = jax.tree.map(lambda a: a + 0.01, f64(feedback.init_recurrence(jax.random.key(1))))
        junk_stream = [d * 7.0 + 1.0 for d in d_stream[:-1]] + [d_stream[-1]]
        junk_deltas = {t: v * -3.0 + 2.0 for t, v in deltas.items()}
        for kw in ({"branch_term": "none"}, {"rec": rec}):
            a = feedback.backward_over_depth(params, stats, x_hats, block_io, stem, d_stream, deltas, **kw)
            b = feedback.backward_over_depth(params, stats, x_hats, block_io, stem, junk_stream, junk_deltas, **kw)
            for t in a:
                np.testing.assert_allclose(a[t], b[t], rtol=1e-12, atol=1e-30, err_msg=t)


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
    fit = feedback.dfa_solve(feedback.dfa_stats(e, deltas, masks), ridge=1e-8)
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


def test_cost_model():
    br, pr = feedback.branch_flops(), feedback.projection_flops()
    assert 0.3e9 < pr.sum() < 0.4e9  # four 1x1 projection convs
    # Exact top 15 of 16 blocks with shortcut-only: everything except block 0's branch.
    assert np.isclose(feedback.method_cost("shortcut", 15), 1 - br[0] / (br.sum() + pr.sum()))
    # Shortcut-only still pays every projection VJP.
    assert np.isclose(feedback.method_cost("shortcut", 0), pr.sum() / (br.sum() + pr.sum()))
    # The recurrence costs more than shortcut-only at every k, and more at higher rank.
    for k in (0, 4, 12):
        r64, r256 = feedback.method_cost("recurrence", k, 64), feedback.method_cost("recurrence", k, 256)
        assert feedback.method_cost("shortcut", k) < r64 < r256
    assert feedback.method_cost("dfa", 0) < 0.02


def test_nan_safe_cosine_loss(signals):
    """The cosine targets must give finite gradients when heads start at zero."""
    from sg.experiments import gate2_predictability as g2
    params, stats, x_hats, block_io, stem, d_stream, deltas = signals
    with jax.enable_x64(True):
        rec = f64(feedback.init_recurrence(jax.random.key(1)))
        for target in ("tap_cos", "full_cos", "full_mse", "delta"):
            for k in (0, 3, 14):
                (_, _), grad = jax.value_and_grad(g2.rec_loss, has_aux=True)(
                    rec, params, stats, (x_hats, block_io, stem), d_stream, k, deltas, target)
                assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grad)), (target, k)


def test_lowrank_full_rank_is_exact(signals):
    """Full rank reproduces backprop at every tap: checks the BN backward, masks and conv transposes."""
    params, stats, x_hats, block_io, stem, d_stream, deltas = signals
    with jax.enable_x64(True):
        x = jax.random.normal(jax.random.key(0), (4, 96, 96, 3), jnp.float64)
        sig = feedback.exact_signals(params, stats, x, tent.entropy)
        lr = f64(feedback.init_lowrank(params, 1.0))
        pred = feedback.backward_over_depth(params, stats, sig.x_hats, sig.block_io, sig.stem, sig.d_stream,
                                            lowrank=lr, inv_stds=sig.inv_stds)
        assert_taps_equal(pred, sig.deltas)


def test_lowrank_rank_trends(signals):
    """Lower rank is cheaper and less accurate; every rank beats shortcut-only."""
    params, stats, *_ = signals
    with jax.enable_x64(True):
        x = jax.random.normal(jax.random.key(0), (4, 96, 96, 3), jnp.float64)
        sig = feedback.exact_signals(params, stats, x, tent.entropy)
        back = lambda **kw: feedback.backward_over_depth(params, stats, sig.x_hats, sig.block_io, sig.stem,
                                                         sig.d_stream, inv_stds=sig.inv_stds, **kw)
        cos = lambda pred: float(feedback.compare(pred, sig.deltas, sig.x_hats)[1][0])
        scores = [cos(back(lowrank=f64(feedback.init_lowrank(params, f)))) for f in (1 / 16, 1 / 4, 1.0)]
        assert scores[0] < scores[1] < scores[2] and abs(scores[2] - 1) < 1e-8, scores
        assert scores[0] > cos(back(branch_term="none")), scores
    costs = [feedback.lowrank_cost(f) for f in (1 / 16, 1 / 4, 1.0)]
    assert costs[0] < costs[1] < costs[2]
