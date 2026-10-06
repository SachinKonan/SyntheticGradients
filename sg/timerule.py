"""How each BN gradient becomes an update, with memory across batches.

  momentum  Tent's rule: SGD with momentum 0.9 (no learned knobs).
  filter    a diagonal filter over the BN parameters, in its Adam-like form
            (a diagonal Kalman / natural-gradient update; Ollivier 2018):
              avg  <- b1 avg + (1 - b1) g          running gradient
              sq   <- b2 sq  + (1 - b2) g^2        running size of the gradient
              w    <- w - eta * avg / sqrt(sq)     big steps when gradients agree,
                                                   small when they flip (noise)
              w    <- w - anchor * (w - w_source)  pull back toward the source model
            with bias-corrected averages. Per BN layer it has four learned knobs:
            eta (step), b1 and b2 (memory lengths), anchor (pull-back strength).

Everything is a pure function of (knobs, state, w, g, w_source), so the knobs can
be trained through the adaptation trajectory like the predictor (phase_c).
"""

import jax
import jax.numpy as jnp
import numpy as np

from sg.models import resnet

MOMENTUM = 0.9
EPS = 1e-12


def init_knobs(kind, eta=None, b1=0.9, b2=0.99, anchor=1e-4):
    """eta: {bn: initial step} (e.g. from calibrate_eta); scalars per BN layer."""
    if kind == "momentum":
        return {}
    logit = lambda p: float(np.log(p / (1 - p)))
    return {n: {"log_eta": np.float32(np.log(eta[n])), "logit_b1": np.float32(logit(b1)),
                "logit_b2": np.float32(logit(b2)), "log_anchor": np.float32(np.log(anchor))}
            for n in resnet.bn_names()}


def calibrate_eta(grads, lr):
    """Initial filter steps that match momentum SGD's per-layer step size:
    momentum moves about lr * |g| / (1 - momentum) per step; the filter moves about eta."""
    return {n: float(lr * np.sqrt(np.mean(np.concatenate([np.ravel(grads[n][k]) ** 2 for k in ("scale", "bias")])))
                     / (1 - MOMENTUM)) + 1e-12 for n in grads}


def init_state(kind, bn):
    zeros = jax.tree.map(jnp.zeros_like, bn)
    if kind == "momentum":
        return {"vel": zeros}
    return {"avg": zeros, "sq": zeros, "t": jnp.zeros(())}


def apply(kind, knobs, state, bn, g, bn_source, lr, mult):
    """One update. lr * mult scales the step for every rule (the deployment multiplier)."""
    if kind == "momentum":
        vel = jax.tree.map(lambda v, gg: MOMENTUM * v + gg, state["vel"], g)
        return jax.tree.map(lambda b, v: b - lr * mult * v, bn, vel), {"vel": vel}

    t = state["t"] + 1
    new_bn, avg, sq = {}, {}, {}
    for n in bn:
        k = knobs[n]
        b1, b2 = jax.nn.sigmoid(k["logit_b1"]), jax.nn.sigmoid(k["logit_b2"])
        eta, anchor = jnp.exp(k["log_eta"]) * mult, jnp.exp(k["log_anchor"])
        avg[n] = jax.tree.map(lambda a, gg: b1 * a + (1 - b1) * gg, state["avg"][n], g[n])
        sq[n] = jax.tree.map(lambda s, gg: b2 * s + (1 - b2) * gg * gg, state["sq"][n], g[n])
        step = jax.tree.map(lambda a, s: (a / (1 - b1 ** t)) / (jnp.sqrt(s / (1 - b2 ** t)) + EPS), avg[n], sq[n])
        new_bn[n] = jax.tree.map(lambda w, d, w0: w - eta * d - anchor * (w - w0), bn[n], step, bn_source[n])
    return new_bn, {"avg": avg, "sq": sq, "t": t}
