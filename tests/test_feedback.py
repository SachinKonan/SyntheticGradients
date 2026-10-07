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


def test_gru_runs_and_starts_as_shortcut(signals):
    """Zero output heads: the GRU predictor equals shortcut-only, like the linear cell."""
    params, stats, x_hats, block_io, stem, d_stream, _ = signals
    with jax.enable_x64(True):
        gru = f64(feedback.init_gru(jax.random.key(3)))
        assert "A" not in gru and "gru" in gru
        a = feedback.backward_over_depth(params, stats, x_hats, block_io, stem, d_stream, rec=gru)
        b = feedback.backward_over_depth(params, stats, x_hats, block_io, stem, d_stream, branch_term="none")
        for tap in a:
            np.testing.assert_allclose(a[tap], b[tap], rtol=1e-12, err_msg=tap)
        gru = jax.tree.map(lambda v: v + 0.01, gru)
        c = feedback.backward_over_depth(params, stats, x_hats, block_io, stem, d_stream, rec=gru)
        assert all(bool(jnp.isfinite(v).all()) for v in c.values())


def test_phase_c_gradient_only_reaches_the_predictor(signals):
    """The outer gradient flows into the predictor and step size, never into ResNet,
    and stop-gradient on ResNet quantities leaves no second-order path."""
    params, stats, *_ = signals
    sg_ = jax.lax.stop_gradient
    with jax.enable_x64(True):
        x = jax.random.normal(jax.random.key(0), (4, 96, 96, 3), jnp.float64)
        y = jnp.arange(4)
        phi = f64(feedback.init_lowrank(params, 0.25))
        bn0 = tent.bn_params(params)

        def outer(phi, log_mult, params):
            bn = bn0
            for _ in range(2):
                p = sg_({**params, **bn})
                sig = sg_(feedback.exact_signals(p, stats, x, tent.entropy))
                d = feedback.backward_over_depth(p, stats, sig.x_hats, sig.block_io, sig.stem, sig.d_stream,
                                                 lowrank=phi, inv_stds=sig.inv_stds)
                g = resnet.bn_grads(d, sig.x_hats)
                bn = jax.tree.map(lambda b, gg: b - 1e-3 * jnp.exp(log_mult) * gg, bn, g)
            logits, _ = resnet.apply({**params, **bn}, stats, x, batch_stats=True)
            return jnp.mean(jax.nn.logsumexp(logits, -1) - logits[jnp.arange(4), y])

        g_phi, g_mult, g_params = jax.grad(outer, argnums=(0, 1, 2))(phi, jnp.float64(0.0), params)
        assert float(sum(jnp.sum(jnp.abs(v)) for v in jax.tree.leaves(g_phi))) > 0
        assert abs(float(g_mult)) > 0
        # ResNet's convolutions get a gradient only through the final forward (first order), which
        # we never apply; the predictor path itself must not depend on them.
        assert all(bool(jnp.isfinite(v).all()) for v in jax.tree.leaves(g_params))


def test_fetch_file_cache_keys_on_full_path(tmp_path):
    from sg.experiments import streams as st
    import subprocess
    calls = []
    real = subprocess.run
    def fake_run(cmd, check=True, **kw):
        calls.append(cmd)
        open(cmd[-1], "wb").write(cmd[-2].encode())
    subprocess.run = fake_run
    try:
        a = st.fetch_file("gs://b/runs/one/predictor.npz", tmp_path)
        b = st.fetch_file("gs://b/runs/two/predictor.npz", tmp_path)
        assert a != b and open(a, "rb").read() != open(b, "rb").read()
        st.fetch_file("gs://b/runs/one/predictor.npz", tmp_path)
        assert len(calls) == 2  # cached by full path
    finally:
        subprocess.run = real


def test_prefetch_surfaces_loader_errors():
    from sg.experiments import streams as st

    class Broken:
        resize = False
        def batch_records(self, step):
            raise IndexError("boom")

    with pytest.raises(IndexError):
        list(st.prefetch_batches([Broken()], 3, workers=1))


def test_filter_rule_matches_its_formula():
    """Adam-form filter: bias-corrected average / sqrt(average square), plus the anchor."""
    from sg import timerule
    bn = {n: {"scale": jnp.ones(3), "bias": jnp.zeros(3)} for n in resnet.bn_names()[:1]}
    names = list(bn)
    eta = {n: 0.1 for n in resnet.bn_names()}
    knobs = {n: v for n, v in timerule.init_knobs("filter", eta=eta, anchor=0.01).items() if n in names}
    g = {n: {"scale": jnp.array([1.0, -2.0, 0.5]), "bias": jnp.array([0.1, 0.1, 0.1])} for n in names}
    state = timerule.init_state("filter", bn)
    new, state = timerule.apply("filter", knobs, state, bn, g, bn, lr=0.0, mult=1.0)
    # First step: bias-corrected avg = g, sq = g^2, so the step is eta * g / (|g| + 1% rms(g)),
    # nearly eta * sign(g); source = start.
    gs = np.array([1.0, -2.0, 0.5])
    step = gs / (np.abs(gs) + timerule.REL_EPS * np.sqrt(np.mean(gs ** 2)))
    np.testing.assert_allclose(new[names[0]]["scale"], 1 - 0.1 * step, rtol=1e-5)


def test_filter_step_has_a_bounded_slope_near_zero_gradients():
    """phase_c differentiates through the filter; a near-zero gradient entry must not dominate."""
    from sg import timerule
    names = resnet.bn_names()[:1]
    bn = {n: {"scale": jnp.ones(3), "bias": jnp.zeros(3)} for n in names}
    knobs = {n: v for n, v in timerule.init_knobs("filter", eta={n: 1.0 for n in resnet.bn_names()}).items()
             if n in names}

    def update(gs):
        g = {n: {"scale": gs, "bias": jnp.ones(3)} for n in names}
        w, _ = timerule.apply("filter", knobs, timerule.init_state("filter", bn), bn, g, bn, lr=0.0, mult=1.0)
        return w[names[0]]["scale"]

    gs = jnp.array([1e-9, 1.0, -1.0])
    slope = jnp.abs(jnp.diag(jax.jacobian(update)(gs)))
    rms = float(jnp.sqrt(jnp.mean(gs ** 2)))
    assert float(slope.max()) <= 1.01 / (timerule.REL_EPS * rms)


def test_filter_knob_gradients_are_finite_with_zero_gradients():
    from sg import timerule
    names = resnet.bn_names()[:2]
    bn = {n: {"scale": jnp.ones(3), "bias": jnp.zeros(3)} for n in names}
    knobs = {n: v for n, v in timerule.init_knobs("filter", eta={n: 0.1 for n in resnet.bn_names()}).items()
             if n in names}
    g = {n: {"scale": jnp.array([0.0, 1.0, -1.0]), "bias": jnp.zeros(3)} for n in names}  # exact zeros

    def loss(knobs):
        state = timerule.init_state("filter", bn)
        w, state = timerule.apply("filter", knobs, state, bn, g, bn, lr=0.0, mult=1.0)
        w, state = timerule.apply("filter", knobs, state, w, g, bn, lr=0.0, mult=1.0)
        return sum(jnp.sum(v["scale"] ** 2) for v in w.values())

    grads = jax.grad(loss)(knobs)
    assert all(bool(jnp.isfinite(v)) for v in jax.tree.leaves(grads))


def test_per_stream_1x1_conv_weights_shard_correctly():
    """Each stream carrying its own 1x1 conv weights, sharded across devices, must match the
    single-device result (vmapped lax.conv kernels did not). Needs >= 2 devices, e.g.
    XLA_FLAGS=--xla_force_host_platform_device_count=2."""
    if len(jax.devices()) < 2:
        pytest.skip("needs 2 devices")
    from jax.sharding import NamedSharding, PartitionSpec as P
    key = jax.random.key(0)
    w = jax.random.normal(key, (2, 1, 1, 16, 8))  # per-stream 1x1 kernels
    x = jax.random.normal(jax.random.fold_in(key, 1), (2, 4, 6, 6, 16))
    f = jax.jit(jax.vmap(lambda w, x: resnet.conv({"w": w}, x, stride=2)))
    ref = f(w, x)
    sh = NamedSharding(jax.sharding.Mesh(np.array(jax.devices()[:2]), ("s",)), P("s"))
    out = f(jax.device_put(w, sh), jax.device_put(x, sh))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_conv1x1_grads_match_backprop(signals):
    """With exact deltas, the 1x1 conv gradients equal jax.grad of the entropy."""
    params, stats, *_ = signals
    with jax.enable_x64(True):
        x = jax.random.normal(jax.random.key(0), (4, 96, 96, 3), jnp.float64)
        sig = feedback.exact_signals(params, stats, x, tent.entropy)
        ours = feedback.conv1x1_grads(params, sig.x_hats, sig.inv_stds, sig.block_io, sig.deltas)
        names = feedback.conv1x1_names()
        assert set(ours) == set(names) and len(names) == 36

        def loss(convs):
            logits, _ = resnet.apply({**params, **convs}, stats, x, batch_stats=True)
            return tent.entropy(logits)

        ref = jax.grad(loss)({n: params[n] for n in names})
        for n in names:
            np.testing.assert_allclose(ours[n]["w"], ref[n]["w"], rtol=1e-8,
                                       atol=1e-12 * float(jnp.abs(ref[n]["w"]).max()), err_msg=n)


def test_split_batch_matches_one_device():
    """Phase C's outer gradient (conv mode, low-rank predictor) with each stream's batch split
    over 2 devices (--batch-split 2) equals the unsharded result. Needs >= 4 devices, e.g.
    XLA_FLAGS=--xla_force_host_platform_device_count=4."""
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"weights not found: {WEIGHTS}")
    from sg.experiments import streams as st
    names = feedback.conv1x1_names()
    with jax.enable_x64(True):
        params, stats = f64(resnet.load_torchvision(WEIGHTS))
        phi = feedback.init_lowrank(params, 0.25)
        key = jax.random.key(0)
        convs = {n: {"w": params[n]["w"] * (1 + 0.01 * jax.random.normal(jax.random.fold_in(key, i), (2,) + params[n]["w"].shape))}
                 for i, n in enumerate(names)}  # each of the 2 streams has its own 1x1 convs
        x = jax.random.normal(jax.random.fold_in(key, 99), (2, 4, 64, 64, 3), jnp.float64)

        def per_stream(phi, convs, x):
            def loss(phi):
                p = jax.lax.stop_gradient({**params, **convs})
                sig = jax.lax.stop_gradient(feedback.exact_signals(p, stats, x, tent.entropy))
                deltas = feedback.backward_over_depth(p, stats, sig.x_hats, sig.block_io, sig.stem, sig.d_stream,
                                                      sig.deltas, exact_top=1,
                                                      **feedback.predictor_kwargs(phi, sig.inv_stds))
                g = feedback.conv1x1_grads(p, sig.x_hats, sig.inv_stds, sig.block_io, deltas)
                new = {n: {"w": convs[n]["w"] - 0.1 * g[n]["w"]} for n in names}
                logits, _ = resnet.apply({**params, **new}, stats, x, batch_stats=True)
                return tent.entropy(logits)
            return jax.value_and_grad(loss)(phi)

        f = jax.jit(jax.vmap(per_stream, in_axes=(None, 0, 0)))
        ref = f(phi, convs, x)
        sm = st.StreamMesh(2, batch_split=2)
        args = (sm.replicate(phi), jax.device_put(convs, sm.shard), sm.put(np.asarray(x), 1))
        outs = [f(*args)]  # XLA's own choice of sharding
        from jax.sharding import NamedSharding, PartitionSpec as P
        resnet.shard_batch(NamedSharding(sm.mesh, P("batch")))  # every activation pinned (phase_c)
        try:
            outs.append(jax.jit(jax.vmap(per_stream, in_axes=(None, 0, 0), spmd_axis_name="streams"))(*args))
        finally:
            resnet.shard_batch(None)
        for out in outs:
            assert out[0].sharding.spec[0] == "streams"
            for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref)):
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * float(jnp.abs(b).max()))


def test_ram_cache_gives_the_same_batches():
    """Streams decoded into RAM (and their views, back to back) yield exactly the JPEG-path batches."""
    from pathlib import Path
    from sg.experiments import streams as st
    group = Path("/scratch/gpfs/ZHUANGL/sk7524/data/imagenet-c/packed/imagenet_c/spatter/3")
    if not group.exists():
        pytest.skip(f"data not found: {group}")
    base = st.Stream(group, 0, 8)
    base.records = base.records[:64]
    rng = np.random.default_rng(0)
    make = lambda: [st.ConcatStream([base.view(rng.permutation(64)), base.view(rng.permutation(64))], 3),
                    base.view(rng.permutation(64))]
    jpeg_streams = make()
    rng = np.random.default_rng(0)
    st.decode_in_ram([base], 4)
    ram_streams = make()
    assert all(s.decoded for s in ram_streams) and not any(s.decoded for s in jpeg_streams)
    for (xa, ya), (xb, yb) in zip(st.prefetch_batches(jpeg_streams, 6, 4), st.prefetch_batches(ram_streams, 6, 4)):
        np.testing.assert_array_equal(xa, xb)
        np.testing.assert_array_equal(ya, yb)


@pytest.mark.parametrize("k,stride", [(3, 1), (3, 2), (7, 2)])
def test_einsum_conv_matches_lax_conv(k, stride):
    key = jax.random.key(k + stride)
    x = jax.random.normal(key, (2, 15, 15, 5))
    w = jax.random.normal(jax.random.fold_in(key, 1), (k, k, 5, 4))
    ref = resnet.conv({"w": w}, x, stride)
    resnet.einsum_convs(True)
    try:
        out = resnet.conv({"w": w}, x, stride)
    finally:
        resnet.einsum_convs(False)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("einsum", [False, True])
def test_conv_grads_match_backprop(signals, einsum):
    """With exact deltas, the gradients of every conv after the stem (1x1 and 3x3) equal jax.grad."""
    params, stats, *_ = signals
    resnet.einsum_convs(einsum)
    try:
        with jax.enable_x64(True):
            x = jax.random.normal(jax.random.key(0), (4, 64, 64, 3), jnp.float64)
            sig = feedback.exact_signals(params, stats, x, tent.entropy)
            names = feedback.conv_names()
            ours = feedback.conv_grads(params, sig.x_hats, sig.inv_stds, sig.block_io, sig.deltas, names)
            assert set(ours) == set(names) and len(names) == 52

            def loss(convs):
                logits, _ = resnet.apply({**params, **convs}, stats, x, batch_stats=True)
                return tent.entropy(logits)

            ref = jax.grad(loss)({n: params[n] for n in names})
            for n in names:
                np.testing.assert_allclose(ours[n]["w"], ref[n]["w"], rtol=1e-8,
                                           atol=1e-12 * float(jnp.abs(ref[n]["w"]).max()), err_msg=n)
    finally:
        resnet.einsum_convs(False)


def test_per_stream_3x3_kernels_shard_correctly_as_einsums():
    """Per-stream 3x3 kernels sharded over devices match the single-device result with einsum convs."""
    if len(jax.devices()) < 2:
        pytest.skip("needs 2 devices")
    from jax.sharding import NamedSharding, PartitionSpec as P
    key = jax.random.key(0)
    w = jax.random.normal(key, (2, 3, 3, 16, 8))
    x = jax.random.normal(jax.random.fold_in(key, 1), (2, 4, 9, 9, 16))
    resnet.einsum_convs(True)
    try:
        f = jax.jit(jax.vmap(lambda w, x: resnet.conv({"w": w}, x, stride=2)))
        ref = f(w, x)
        sh = NamedSharding(jax.sharding.Mesh(np.array(jax.devices()[:2]), ("s",)), P("s"))
        out = f(jax.device_put(w, sh), jax.device_put(x, sh))
    finally:
        resnet.einsum_convs(False)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_masked_batch_statistics_equal_the_subset_batch():
    """resnet.apply(batch_mask=m) gives the selected examples exactly what a batch of only them gives."""
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"weights not found: {WEIGHTS}")
    params, stats = resnet.load_torchvision(WEIGHTS)
    x = jax.random.normal(jax.random.key(0), (6, 64, 64, 3), jnp.float32)
    m = jnp.array([True, False, True, True, False, True])
    full, _ = resnet.apply(params, stats, x, batch_stats=True, batch_mask=m)
    sub, _ = resnet.apply(params, stats, x[m], batch_stats=True)
    np.testing.assert_allclose(full[m], sub, rtol=2e-3, atol=2e-3)


def test_hue_matches_torchvision():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("torchvision.transforms.functional")
    from sg import augment
    x = np.random.default_rng(0).uniform(size=(2, 8, 8, 3)).astype(np.float32)
    h, s, v = augment._rgb_to_hsv(jnp.asarray(x))
    ours = augment._hsv_to_rgb(jnp.mod(h + 0.05, 1.0), s, v)
    ref = tf.adjust_hue(torch.from_numpy(x).permute(0, 3, 1, 2), 0.05).permute(0, 2, 3, 1).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5)
