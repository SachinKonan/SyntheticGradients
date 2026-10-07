"""The closed-form ViT backward (sg.vit_feedback) is exact, and its low-rank version runs.

Float64, batch 2 at 224 x 224 (the 14 x 14 patch grid), the real weights at full depth
and truncated to 2 blocks.

Run: uv run pytest tests/test_vit.py  (~1.5 min; on a shared login node pin it, e.g.
taskset -c 0-15, since XLA otherwise uses every core)
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sg import tent, vit_feedback as vf
from sg.models import vit

WEIGHTS = os.environ.get(
    "SG_VIT_WEIGHTS", "/scratch/gpfs/ZHUANGL/sk7524/data/weights/vit_b16_augreg2_in21k_ft_in1k.safetensors")
DEPTHS = [2, vit.DEPTH]


def f64(tree):
    return jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), tree)


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-300))


def assert_close(got, want, name):
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-12 * float(jnp.abs(want).max()), err_msg=name)
    assert rel(got, want) < 1e-10, (name, rel(got, want))


@pytest.fixture(scope="module")
def weights():
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"weights not found: {WEIGHTS}")
    return vit.load_timm(WEIGHTS)


@pytest.fixture(scope="module", params=DEPTHS, ids=lambda d: f"depth{d}")
def case(request, weights):
    """(params, x, exact signals, closed-form exact backward, autodiff grads)."""
    with jax.enable_x64(True):
        params = f64(vit.truncate(weights, request.param))
        x = 0.5 * jax.random.normal(jax.random.key(0), (2, 224, 224, 3), jnp.float64)
        sig = jax.jit(vf.exact_signals, static_argnums=2)(params, x, tent.entropy)
        bwd = jax.jit(vf.backward_over_depth)(params, sig.x_hats, sig.inv_stds, sig.blocks, sig.e)
        ref = jax.jit(jax.grad(lambda p: tent.entropy(vit.apply(p, x)[0])))(params)
        yield params, x, sig, bwd, ref


def test_probe_shapes_cover_every_ln(case):
    params, x, sig, *_ = case
    shapes = vit.probe_shapes(params, x)
    assert set(shapes) == set(vit.ln_names(vit.depth(params))) == set(sig.deltas)
    for s in shapes.values():
        assert s.shape == (2, 197, vit.WIDTH)


def test_exact_deltas_equal_probe_grads(case):
    """The error at every LN output and every block output equals the probe gradient."""
    _, _, sig, bwd, _ = case
    with jax.enable_x64(True):
        assert set(bwd.deltas) == set(sig.deltas)
        for name in sig.deltas:
            assert_close(bwd.deltas[name], sig.deltas[name], name)
        for i, (got, want) in enumerate(zip(bwd.d_out, sig.d_out)):
            assert_close(got, want, f"d_out[{i}]")
        worst = max(rel(bwd.deltas[k], sig.deltas[k]) for k in sig.deltas)
        print(f"depth {len(sig.blocks)}: max relative LN-error mismatch (norm) = {worst:.2e}")


def test_exact_grads_equal_autodiff(case):
    """LN (scale, bias), MLP and attention (w, b) gradients from the closed-form errors equal
    jax.grad of the entropy."""
    params, _, sig, bwd, ref = case
    with jax.enable_x64(True):
        got = (vit.ln_grads(bwd.deltas, sig.x_hats) | vf.mlp_grads(params, sig.x_hats, sig.blocks, bwd)
               | vf.attn_grads(params, sig.x_hats, sig.blocks, bwd))
        n = vit.depth(params)
        assert len(got) == len(vit.ln_names(n)) + 4 * n
        worst = 0.0
        for name, g in got.items():
            for k in g:
                assert_close(g[k], ref[name][k], f"{name}.{k}")
                worst = max(worst, rel(g[k], ref[name][k]))
        print(f"depth {n}: max relative grad error (norm) = {worst:.2e}")


def test_lowrank_full_rank_is_exact(case):
    params, _, sig, bwd, _ = case
    with jax.enable_x64(True):
        lr = f64(vf.init_lowrank(params, 1.0))
        assert lr["blocks"][0]["fc1"]["P"].shape == (vit.WIDTH, vit.WIDTH)
        low = jax.jit(vf.backward_over_depth)(params, sig.x_hats, sig.inv_stds, sig.blocks, sig.e, lowrank=lr)
        for name in bwd.deltas:
            assert_close(low.deltas[name], bwd.deltas[name], name)


def flat_grads(params, sig, bwd):
    g = vit.ln_grads(bwd.deltas, sig.x_hats) | vf.mlp_grads(params, sig.x_hats, sig.blocks, bwd)
    return jnp.concatenate([v.ravel() for v in jax.tree.leaves(g)])


def test_lowrank_runs_and_is_approximate(case):
    params, _, sig, bwd, _ = case
    with jax.enable_x64(True):
        lr = f64(vf.init_lowrank(params, 0.25))
        assert lr["blocks"][0]["qkv"]["Q"].shape == (192, 3 * vit.WIDTH)
        low = jax.jit(vf.backward_over_depth)(params, sig.x_hats, sig.inv_stds, sig.blocks, sig.e, lowrank=lr)
        a, b = flat_grads(params, sig, low), flat_grads(params, sig, bwd)
        cos = float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
        print(f"depth {vit.depth(params)}: frac 0.25 gradient cosine = {cos:.4f}")
        assert np.isfinite(np.asarray(a)).all()
        assert -1 < cos < 1 - 1e-6


def test_exact_top(case):
    """exact_top = k: the top k blocks' LN errors are exact, block n-1-k's are not (its output
    error is). Traced exact_top (lax.cond) gives the same result as a Python int."""
    params, _, sig, bwd, _ = case
    n = vit.depth(params)
    with jax.enable_x64(True):
        lr = f64(vf.init_lowrank(params, 0.25))
        args = (params, sig.x_hats, sig.inv_stds, sig.blocks, sig.e)  # arguments, not closed-over constants
        run = jax.jit(vf.backward_over_depth, static_argnames="exact_top")
        traced_run = jax.jit(vf.backward_over_depth)
        for k in sorted({0, 1, n - 1, n}):
            pred, traced = run(*args, lowrank=lr, exact_top=k), traced_run(*args, lowrank=lr, exact_top=k)
            for name in pred.deltas:
                assert_close(traced.deltas[name], pred.deltas[name], f"traced {name}")
            assert_close(pred.deltas["norm"], bwd.deltas["norm"], "norm")
            for i in range(n):
                exact = i >= n - k
                for ln in (f"blocks.{i}.norm1", f"blocks.{i}.norm2"):
                    if exact:
                        assert_close(pred.deltas[ln], bwd.deltas[ln], ln)
                    else:
                        assert rel(pred.deltas[ln], bwd.deltas[ln]) > 1e-3, (k, ln)
                if i >= n - 1 - k:
                    assert_close(pred.d_out[i], bwd.d_out[i], f"d_out[{i}]")


def test_grads_jit_and_vmap(weights):
    """The end-to-end closed-form gradient jits, and vmaps over streams with their own factors."""
    with jax.enable_x64(True):
        params = f64(vit.truncate(weights, 1))
        x = 0.5 * jax.random.normal(jax.random.key(1), (2, 1, 224, 224, 3), jnp.float64)
        lr0 = f64(vf.init_lowrank(params, 0.25))
        lr = [lr0, jax.tree.map(lambda a: 1.5 * a, lr0)]  # same shapes, different factors per stream
        stacked = jax.tree.map(lambda *a: jnp.stack(a), *lr)
        f = lambda params, x, lr: vf.grads(params, x, tent.entropy, lowrank=lr)[1]
        batched = jax.jit(jax.vmap(f, in_axes=(None, 0, 0)))(params, x, stacked)
        for s in range(2):
            one = jax.jit(f)(params, x[s], lr[s])
            for name in one:
                for k in one[name]:
                    np.testing.assert_allclose(batched[name][k][s], one[name][k], rtol=1e-9, atol=1e-30)
