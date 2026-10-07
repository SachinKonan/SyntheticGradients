"""Our JAX ViT-B/16 is timm's vit_base_patch16_224.augreg2_in21k_ft_in1k.

Logits match timm (float64, eval mode) on real val and ImageNet-C images, and the
entropy gradients of the LN and MLP parameters match PyTorch autograd.

Run: uv run --group dev pytest tests/test_vit_torch_parity.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sg import tent
from sg.data import imagenet
from sg.models import resnet, vit

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

WEIGHTS = os.environ.get(
    "SG_VIT_WEIGHTS", "/scratch/gpfs/ZHUANGL/sk7524/data/weights/vit_b16_augreg2_in21k_ft_in1k.safetensors")
PACKED = os.environ.get("SG_DATA", "/scratch/gpfs/ZHUANGL/sk7524/data/imagenet-c/packed")
N_IMAGES = 4  # per group; keeps float64 autograd on CPU small

jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(scope="module")
def torch_model():
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"weights not found: {WEIGHTS}")
    from safetensors.torch import load_file

    m = timm.create_model("vit_base_patch16_224", pretrained=False)
    m.load_state_dict(load_file(WEIGHTS))
    return m.double().eval()


@pytest.fixture(scope="module")
def images():
    """Clean val and ImageNet-C images, re-normalized for this model (mean = std = 0.5)."""
    if not os.path.exists(PACKED):
        x = np.random.default_rng(0).uniform(0, 255, (2 * N_IMAGES, 224, 224, 3))
        return vit.normalize(x).astype(np.float32)
    groups = ["imagenet_val", "imagenet_c/gaussian_noise/5"]
    x = np.stack([ex[0] for g in groups for ex in imagenet.read_group(PACKED, g, limit=N_IMAGES)])
    raw = (x * resnet.IMAGENET_STD + resnet.IMAGENET_MEAN) * 255.0
    return vit.normalize(raw).astype(np.float32)


def test_logits_match(torch_model, images):
    with torch.no_grad():
        ref = torch_model(torch.from_numpy(images).double().permute(0, 3, 1, 2)).numpy()
    with jax.enable_x64(True):
        params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), vit.load_timm(WEIGHTS))
        ours = np.asarray(jax.jit(vit.apply)(params, jnp.asarray(images, jnp.float64))[0])
    err = np.abs(ours - ref).max()
    print(f"max |logit diff| (float64) = {err:.3e}, max |logit| = {np.abs(ref).max():.2f}")
    assert err < 1e-9, err
    assert (ours.argmax(-1) == ref.argmax(-1)).all()


def test_logits_match_float32(torch_model, images):
    with torch.no_grad():
        ref = torch_model(torch.from_numpy(images).double().permute(0, 3, 1, 2)).numpy()
    ours = np.asarray(jax.jit(vit.apply)(jax.tree.map(jnp.asarray, vit.load_timm(WEIGHTS)), jnp.asarray(images))[0])
    err = np.abs(ours - ref).max()
    print(f"max |logit diff| (float32 vs float64 torch) = {err:.3e}")
    assert err < 1e-3, err
    assert (ours.argmax(-1) == ref.argmax(-1)).all()


def test_entropy_grads_match_torch(torch_model, images):
    x = images[:2]
    torch_model.zero_grad()
    logp = torch_model(torch.from_numpy(x).double().permute(0, 3, 1, 2)).log_softmax(-1)
    (-(logp.exp() * logp).sum(-1).mean()).backward()
    ref = dict(torch_model.named_parameters())
    with jax.enable_x64(True):
        params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), vit.load_timm(WEIGHTS))
        grads = jax.jit(jax.grad(lambda p: tent.entropy(vit.apply(p, jnp.asarray(x, jnp.float64))[0])))(params)
        worst = 0.0
        for name in vit.ln_names() + vit.mlp_names():
            for k, tk in (("scale", "weight"), ("bias", "bias"), ("w", "weight"), ("b", "bias")):
                if k not in grads[name]:
                    continue
                want = ref[f"{name}.{tk}"].grad.numpy()
                got = np.asarray(grads[name][k])
                got = got.T if k == "w" else got
                err = np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30)
                worst = max(worst, err)
                assert err < 1e-10, (name, k, err)
        print(f"max relative grad error vs torch = {worst:.3e}")
