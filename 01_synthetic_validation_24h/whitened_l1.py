#!/usr/bin/env python3
"""
Whitened (data-space / errors-in-variables) L1 residual.

Duplicated and adapted from "../L1 with log term/l1_log_term_poc.py" (the
proof of concept where this loss was developed and shown to remove the
noise cheat of the raw pointwise-L1 objective while keeping the L1 scheme).
Only the winning "full" variant is carried over; the diagonal variant was
shown insufficient there (it ignores residual correlations, which the
optimizer then exploits).

    L = mean[ r^T (J J^T)^(-1) r ],     J = d r / d c_obs,

where r is the standard pointwise L1 physics residual.  This equals the
negative log-likelihood of the observations under iid Gaussian measurement
noise (log-det terms cancel), i.e. "how far the data would have to move to
satisfy the physics" -- a beta-independent metric, so the operator gain
h^(-beta) cannot be used to quiet noise.
"""

import jax
import jax.numpy as jnp

import pinn_lib as pl

JITTER = 1e-10  # linear-solve safety


def l1_expanded_matrix(num_timesteps, beta):
    """
    A(beta) with shape (N-1, N): the L1 residual's linear action on the raw
    data vector, r_n = sum_k A[n-1, k] c_k - alpha f(c_n) for n = 1..N-1.

    Row n (target node n), from the L1 weights b_j = (j+1)^(1-b) - j^(1-b):
        A[n, n] = coef * b_0    (= the operator gain g(beta))
        A[n, k] = coef * (b_{n-k} - b_{n-k-1})   for 1 <= k <= n-1
        A[n, 0] = -coef * b_{n-1}
    with coef = h^(-beta)/Gamma(2-beta).  Checked against pinn_lib.caputo_l1
    by self_test().
    """
    N = num_timesteps
    h = 1.0 / (N - 1)
    j = jnp.arange(N, dtype=jnp.float64)
    b = pl.safe_pow(j + 1.0, 1.0 - beta) - pl.safe_pow(j, 1.0 - beta)
    coef = h ** (-beta) / pl.gamma_fn(2.0 - beta)

    n = jnp.arange(1, N, dtype=jnp.float64)[:, None]
    k = jnp.arange(N, dtype=jnp.float64)[None, :]
    lag = jnp.clip(n - k, 0, N - 1).astype(int)
    A = jnp.where((k >= 1) & (k <= n - 1), b[lag] - b[lag - 1], 0.0)
    A = jnp.where(k == n, b[0], A)
    A = jnp.where(k == 0, -b[jnp.clip(n.astype(int) - 1, 0, N - 1)][:, 0:1]
                  * jnp.ones_like(A), A)
    return coef * A


def residual_and_jacobian(c_jax, beta, fc, fprime, alpha=1.0):
    """
    r : (S, N-1) pointwise L1 physics residual at nodes 1..N-1
    J : (S, N-1, N) Jacobian dr/dc_obs
    """
    S, N, _ = c_jax.shape
    A = l1_expanded_matrix(N, beta)
    r = jnp.einsum("mk,skf->smf", A, c_jax)[:, :, 0] - alpha * fc[:, 1:, 0]
    eye_target = jnp.eye(N)[1:, :]
    J = A[None, :, :] - alpha * fprime[:, 1:, 0][:, :, None] * eye_target[None, :, :]
    return r, J


def f_prime_nn(f_params, c_jax):
    """df/dc of the production network via one jvp pass."""
    _, tangent = jax.jvp(lambda x: pl.fnn_fixed_boundary(f_params, x),
                         (c_jax,), (jnp.ones_like(c_jax),))
    return tangent


def loss_whitened_full(c_jax, beta, fc, fprime, alpha=1.0):
    """Exact data-space whitening: mean of r^T (J J^T)^(-1) r."""
    r, J = residual_and_jacobian(c_jax, beta, fc, fprime, alpha)
    JJt = jnp.einsum("snk,smk->snm", J, J)
    JJt = JJt + JITTER * jnp.eye(JJt.shape[-1])[None, :, :]
    x = jnp.linalg.solve(JJt, r[:, :, None])[:, :, 0]
    return jnp.mean(r * x)


def self_test(c_jax, tol=1e-10):
    """The expanded operator must reproduce the production caputo_l1."""
    import numpy as np
    for beta in (0.3, 0.5, 0.8):
        A = l1_expanded_matrix(c_jax.shape[1], beta)
        lhs = np.array(jnp.einsum("mk,skf->smf", A, c_jax))[:, :, 0]
        rhs = np.array(pl.caputo_l1(c_jax, beta))[:, 1:, 0]
        assert np.max(np.abs(lhs - rhs)) < tol, "expanded L1 mismatch"
