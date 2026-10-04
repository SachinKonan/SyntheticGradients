"""Our JAX ResNet-50 is the torchvision ResNet-50 (IMAGENET1K_V1).

1. The safetensors weights equal the official torchvision checkpoint, bit for bit.
2. Logits match torchvision on real val images, in eval mode (running stats)
   and in train mode (batch stats, as Tent uses).
3. Tent's BN gradients match PyTorch autograd.

Run: uv run --group dev pytest tests/test_torch_parity.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sg import tent
from sg.data import imagenet
from sg.models import resnet

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

WEIGHTS_DIR = "/scratch/gpfs/ZHUANGL/sk7524/data/weights"
SAFETENSORS = f"{WEIGHTS_DIR}/resnet50_tv_in1k.safetensors"
OFFICIAL = f"{WEIGHTS_DIR}/resnet50-0676ba61.pth"
PACKED = os.environ.get("SG_DATA", "/scratch/gpfs/ZHUANGL/sk7524/data/imagenet-c/packed")

jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(scope="module")
def official():
    if not os.path.exists(OFFICIAL):
        pytest.skip("official checkpoint not downloaded")
    return torch.load(OFFICIAL, map_location="cpu")


@pytest.fixture(scope="module")
def torch_model(official):
    m = torchvision.models.resnet50()
    m.load_state_dict(official)
    return m.double()


@pytest.fixture(scope="module")
def jax_model():
    params, stats = resnet.load_torchvision(SAFETENSORS)
    return jax.tree.map(jnp.asarray, params), jax.tree.map(jnp.asarray, stats)


@pytest.fixture(scope="module")
def images():
    """16 clean val images and 16 ImageNet-C images, preprocessed."""
    clean = list(imagenet.read_group(PACKED, "imagenet_val", limit=16))
    corrupt = list(imagenet.read_group(PACKED, "imagenet_c/gaussian_noise/5", limit=16))
    x = np.stack([ex[0] for ex in clean + corrupt])
    y = np.array([ex[1] for ex in clean + corrupt])
    return x, y


def test_weights_bit_identical(official):
    from safetensors.numpy import load_file

    ours = load_file(SAFETENSORS)
    official = {k: v.detach().numpy() for k, v in official.items()}
    assert set(official) <= set(ours)
    for k, v in official.items():
        assert ours[k].dtype == v.dtype and np.array_equal(ours[k], v), k


@pytest.mark.parametrize("batch_stats", [False, True])
def test_logits_match(torch_model, jax_model, images, batch_stats):
    x, _ = images
    torch_model.train(batch_stats)
    with torch.no_grad():
        ref = torch_model(torch.from_numpy(x).double().permute(0, 3, 1, 2)).numpy()
    params, stats = jax_model
    ours, _ = jax.jit(resnet.apply, static_argnames="batch_stats")(
        params, stats, jnp.asarray(x), batch_stats=batch_stats)
    ours = np.asarray(ours)
    assert np.abs(ours - ref).max() < 1e-3, np.abs(ours - ref).max()
    assert (ours.argmax(-1) == ref.argmax(-1)).all()


def test_tent_grads_match_torch(torch_model, jax_model, images):
    x, _ = images
    torch_model.train(True)
    torch_model.zero_grad()
    logits = torch_model(torch.from_numpy(x).double().permute(0, 3, 1, 2))
    logp = logits.log_softmax(-1)
    (-(logp.exp() * logp).sum(-1).mean()).backward()
    modules = dict(torch_model.named_modules())

    # Compare in float64: in float32 the error through 50 layers is ~1e-3 from rounding alone.
    with jax.enable_x64(True):
        params, stats = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), jax_model)
        _, deltas, x_hats = jax.jit(tent.exact_signals)(params, stats, jnp.asarray(x, jnp.float64))
        ours = resnet.bn_grads(deltas, x_hats)
        for name in resnet.bn_names():
            for k, tk in (("scale", "weight"), ("bias", "bias")):
                ref = getattr(modules[name], tk).grad.numpy()
                got = np.asarray(ours[name][k])
                err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30)
                assert err < 1e-10, (name, k, err)
