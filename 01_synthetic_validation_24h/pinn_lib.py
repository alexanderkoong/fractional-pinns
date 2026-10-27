#!/usr/bin/env python3
"""
JAX components shared by the parameter-recovery validation scripts.

Contents
--------
  * caputo_l1        -- L1 Caputo derivative in matrix (Toeplitz) form,
                        differentiable w.r.t. beta.  Mathematically identical
                        to caputo_derivative() in train_virality.py (the
                        production training code); the loop there is O(N)
                        traced JAX ops per sample which is prohibitively slow
                        to differentiate at N >= 384, so we use a single
                        matmul instead.  caputo_l1_loop() below is a verbatim
                        port of the production loop and verify_l1_caputo.py
                        checks the two agree to machine precision.
  * volterra_residual - integral (Volterra) form of the model residual,
                        discretized with the same product-trapezoidal weights
                        as the (corrected) ABM corrector.  Provided as an
                        alternative estimator that avoids the early-time bias
                        of the pointwise L1 residual (see
                        validate_beta_recovery.py).
  * fnn_fixed_boundary, init_network_params
                      -- the f(c) network, copied verbatim from
                        train_virality.py: output exp(NN(c)) * c * (1-c),
                        which hard-codes f(0) = f(1) = 0 and f > 0 on (0,1).
  * beta_from_raw / raw_from_beta
                      -- the production parameterization
                        beta = 0.1 + 0.89 * sigmoid(raw).

All computations run in float64 (jax_enable_x64): the validation studies
estimator bias down to ~1e-3 in beta, which should not be confounded with
float32 roundoff.
"""

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Small utilities
# ============================================================================

def gamma_fn(x):
    """Gamma function via log-gamma (numerically stable)."""
    return jnp.exp(gammaln(x))


def safe_pow(x, p):
    """
    x**p for x >= 0, safe to differentiate w.r.t. p at x = 0.

    d/dp [x**p] = x**p * log(x) is NaN at x = 0 under autodiff; the weight
    tables below contain 0**p entries whose p-gradient must be exactly 0.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    x_safe = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, x_safe ** p, 0.0)


def mse(x, y):
    return ((x - y) ** 2).sum() / x.size


# ============================================================================
# L1 Caputo derivative (matrix form + verbatim production loop)
# ============================================================================

def l1_matrix(num_timesteps, beta):
    """
    Lower-triangular Toeplitz matrix T with T[n, k] = b_{n-k} for 1 <= k <= n,
    where b_j = (j+1)^(1-beta) - j^(1-beta), so that

        (D^beta c)_n = h^(-beta)/Gamma(2-beta) * (T @ dc)_n,
        dc_k = c_k - c_{k-1} (dc_0 = 0).

    Row n = 0 is identically zero (derivative not defined at t = 0),
    matching the production implementation.
    """
    j = jnp.arange(num_timesteps, dtype=jnp.float64)
    b = safe_pow(j + 1.0, 1.0 - beta) - safe_pow(j, 1.0 - beta)
    n = j[:, None]
    k = j[None, :]
    lag = jnp.clip(n - k, 0, num_timesteps - 1).astype(int)
    return jnp.where((k >= 1) & (n - k >= 0), b[lag], 0.0)


def caputo_l1(c, beta):
    """
    L1 Caputo derivative of trajectories c, differentiable w.r.t. beta.

    Args:
        c: shape (num_samples, num_timesteps, 1), samples on the uniform
           grid t in [0, 1]
        beta: fractional order (scalar)
    Returns:
        D^beta c, same shape; entry n = 0 is 0.
    """
    num_timesteps = c.shape[1]
    h = 1.0 / (num_timesteps - 1)
    dc = jnp.diff(c, axis=1)
    dc = jnp.concatenate([jnp.zeros_like(c[:, :1, :]), dc], axis=1)
    T = l1_matrix(num_timesteps, beta)
    coef = h ** (-beta) / gamma_fn(2.0 - beta)
    return coef * jnp.einsum("nk,skf->snf", T, dc)


def caputo_l1_loop(c, t, beta):
    """
    Verbatim port of caputo_derivative() from train_virality.py (production
    training code).  Kept ONLY as the reference implementation for the
    equivalence test in verify_l1_caputo.py -- use caputo_l1() everywhere else.
    """
    num_samples, num_timesteps, num_features = c.shape
    h = 1.0 / (num_timesteps - 1)
    j = jnp.arange(num_timesteps, dtype=jnp.float64)
    weights = (j + 1.0) ** (1.0 - beta) - j ** (1.0 - beta)
    coef = (h ** (-beta)) / gamma_fn(2.0 - beta)
    dc = jnp.diff(c, axis=1)
    dc = jnp.concatenate([jnp.zeros((num_samples, 1, num_features)), dc], axis=1)
    Dbc = jnp.zeros_like(c)
    for n in range(1, num_timesteps):
        dc_slice = dc[:, 1 : n + 1, :]
        dc_reversed = jnp.flip(dc_slice, axis=1)
        w = weights[:n]
        weighted_sum = jnp.sum(w[None, :, None] * dc_reversed, axis=1, keepdims=True)
        Dbc = Dbc.at[:, n, :].set(weighted_sum[:, 0, :])
    return coef * Dbc


# ============================================================================
# Volterra (integral-form) residual
# ============================================================================

def volterra_matrix(num_timesteps, beta):
    """
    Product-trapezoidal weights for the equivalent Volterra integral equation

        c(t_n) = c0 + 1/Gamma(beta) int_0^{t_n} (t_n - s)^(beta-1)
                                     alpha f(c(s)) ds,

    discretized with piecewise-linear interpolation of f(c(s)) (the same rule
    as the corrected ABM corrector, but fully implicit -- every node uses the
    OBSERVED c, no predictor needed):

        c_n ~= c0 + h^beta/Gamma(beta+2) * sum_{j=0}^{n} V[n, j] f_j,

        V[n, n] = 1
        V[n, 0] = (n-1)^(beta+1) - (n-1-beta) n^beta
        V[n, j] = (m+1)^(beta+1) - 2 m^(beta+1) + (m-1)^(beta+1),
                  m = n - j,  1 <= j <= n-1.

    Row n = 0 is zero (the residual is defined for n >= 1).
    """
    b1 = beta + 1.0
    idx = jnp.arange(num_timesteps, dtype=jnp.float64)
    n = idx[:, None]
    j = idx[None, :]
    m = n - j

    interior = safe_pow(m + 1.0, b1) - 2.0 * safe_pow(m, b1) + safe_pow(m - 1.0, b1)
    first_col = safe_pow(n - 1.0, b1) - (n - 1.0 - beta) * safe_pow(n, beta)

    V = jnp.where((j >= 1) & (j <= n - 1), interior, 0.0)
    V = jnp.where((j == 0) & (n >= 1), first_col, V)
    V = jnp.where((j == n) & (n >= 1), 1.0, V)
    return V


def volterra_residual(c, fc, beta, alpha):
    """
    Residual of the integral form,  R_n = c_n - c0 - I_n[alpha * f],  n >= 1.

    Unlike the pointwise L1 residual (which differentiates the data and is
    dominated by the t^beta initial-layer error at early nodes), this residual
    integrates f along the trajectory, exactly like the forward solver -- it
    is the "solve-consistent" or rollout-form objective.

    Args:
        c:  shape (S, N, 1) observed trajectories
        fc: shape (S, N, 1) f evaluated at the observations
        beta, alpha: model parameters
    Returns:
        R: shape (S, N, 1); row n = 0 is 0 and should be excluded from losses.
    """
    num_timesteps = c.shape[1]
    h = 1.0 / (num_timesteps - 1)
    V = volterra_matrix(num_timesteps, beta)
    coef = alpha * h ** beta / gamma_fn(beta + 2.0)
    integral = coef * jnp.einsum("nk,skf->snf", V, fc)
    return c - c[:, :1, :] - integral


# ============================================================================
# f(c) network -- copied verbatim from train_virality.py
# ============================================================================

def random_layer_params(ind, outd, num_nn, key):
    """Initialize a single layer's weights and biases."""
    scale = jnp.sqrt(2.0 / (ind + outd))
    W = random.normal(key, (num_nn, ind, outd)) * scale
    b = jnp.zeros((num_nn, 1, outd))
    return W, b


def init_network_params(ind, outd, width, num_nn, key, num_layers):
    """Initialize parameters for a multi-layer feedforward network."""
    keys = random.split(key, num_layers)
    layers = [random_layer_params(ind, width, num_nn, keys[0])]
    for i in range(1, num_layers - 1):
        layers.append(random_layer_params(width, width, num_nn, keys[i]))
    layers.append(random_layer_params(width, outd, num_nn, keys[-1]))
    return layers


def fnn_fixed_boundary(params, x):
    """
    f(c) network with hard-coded boundary conditions f(0) = f(1) = 0:
        f(x) = exp(NN(x)) * x * (1 - x).
    """
    boundary_term = x * (1.0 - x)
    for i in range(len(params) - 1):
        W, b = params[i]
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    x = jnp.exp(x @ W + b) * boundary_term
    return x


# ============================================================================
# beta parameterization -- copied from train_virality.py
# ============================================================================

def beta_from_raw(raw):
    """beta = 0.1 + 0.89 * sigmoid(raw), constraining beta to (0.1, 0.99)."""
    return 0.1 + 0.89 * jax.nn.sigmoid(raw)


def raw_from_beta(beta):
    """Inverse of beta_from_raw (for initializing at a chosen beta)."""
    s = (beta - 0.1) / 0.89
    s = jnp.clip(s, 1e-6, 1.0 - 1e-6)
    return jnp.log(s / (1.0 - s))
