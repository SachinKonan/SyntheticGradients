"""Tent (Wang et al., 2021): entropy minimization on BN affine parameters,
with BN statistics taken from the test batch.

Exact signals come from one backward pass to zero probes at every BN output
(see sg.models.resnet). The BN gradients are then formed in closed form from
(delta, x_hat), the same way a predicted delta would be used.
"""

import jax
import jax.numpy as jnp

from sg.models import resnet


def entropy(logits):
    """Mean prediction entropy over the batch."""
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(jnp.exp(logp) * logp, axis=-1))


def bn_params(params):
    return {name: params[name] for name in resnet.bn_names()}


def exact_signals(params, stats, x):
    """One forward + backward. Returns (logits, deltas, x_hats) at every BN output."""
    probes = jax.tree.map(lambda sd: jnp.zeros(sd.shape, sd.dtype),
                          resnet.probe_shapes(params, stats, x))

    def loss(probes):
        logits, x_hats = resnet.apply(params, stats, x, batch_stats=True, probes=probes)
        return entropy(logits), (logits, x_hats)

    deltas, (logits, x_hats) = jax.grad(loss, has_aux=True)(probes)
    return logits, deltas, x_hats


def sgd_momentum(params, grads, velocity, lr, momentum):
    """PyTorch-style SGD with momentum (dampening 0, no Nesterov)."""
    velocity = jax.tree.map(lambda v, g: momentum * v + g, velocity, grads)
    params = jax.tree.map(lambda p, v: p - lr * v, params, velocity)
    return params, velocity


def tent_step(params, velocity, stats, x, lr=2.5e-4, momentum=0.9):
    """Score the batch, then update BN affine params on it. Returns new state and logits.

    lr and momentum follow Tent's ImageNet setting (batch 64).
    """
    logits, deltas, x_hats = exact_signals(params, stats, x)
    grads = resnet.bn_grads(deltas, x_hats)
    new_bn, velocity = sgd_momentum(bn_params(params), grads, velocity, lr, momentum)
    return {**params, **new_bn}, velocity, logits, deltas
