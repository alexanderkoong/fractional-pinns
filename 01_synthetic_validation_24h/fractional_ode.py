#!/usr/bin/env python3
"""
Core numerical routines for the fractional virality model

    D_t^beta c(t) = alpha * f(c(t)),    c(0) = c0,    0 < beta < 1,

where D_t^beta is the Caputo fractional derivative.

This module is pure numpy/scipy (no JAX) and provides:

  * mittag_leffler        -- analytical solution kernel for the linear test ODE
  * caputo_power          -- analytical Caputo derivative of t^p
  * caputo_sin            -- analytical Caputo derivative of A*sin(w t) (series)
  * caputo_l1_np          -- the L1 discretization of the Caputo derivative
                             (numpy mirror of the training-side implementation)
  * solve_fractional_ode  -- Adams-Bashforth-Moulton (ABM) predictor-corrector
                             forward solver (Diethelm-Ford-Freed 2002)
  * generate_synthetic_data -- fine-grid solve + subsample data generation

IMPORTANT PROVENANCE NOTE
-------------------------
The ABM solver here is adapted from the original thesis validation code
("step1 full validation/validate_f_recovery.py") with ONE correction: the
j = 0 corrector weight. The original code used

    a_weights[0] = 1.0                                   # WRONG

whereas the Diethelm-Ford-Freed scheme requires

    a_{0,n+1} = n^(beta+1) - (n - beta) * (n+1)^beta     # CORRECT
                (equals beta at the first step n = 0)

The incorrect constant weight over-weights f(c0) at every step, adds a
systematic offset of order h^beta/Gamma(beta+2) * alpha * f(c0) to every
solution value, and degrades the scheme's convergence order from the
theoretical 1+beta (~1.5 for beta = 0.5) to roughly 0.4.  Measured impact
for D^0.5 c = c, c0 = 0.1, against the analytical Mittag-Leffler solution:

    N = 48 :  max error 4.17e-02 (old)  vs  2.14e-03 (corrected)
    N = 384:  max error 1.77e-02 (old)  vs  9.64e-05 (corrected)

See verify_abm_solver.py, which reproduces these numbers.

References
----------
K. Diethelm, N. J. Ford, A. D. Freed, "A predictor-corrector approach for the
numerical solution of fractional differential equations", Nonlinear Dynamics
29 (2002) 3-22.
"""

import os

import numpy as np
from scipy.special import gamma as sp_gamma


# ============================================================================
# Ground-truth reaction function used throughout the validation suite
# ============================================================================

def f_cubic(c):
    """Ground-truth reaction law used for all synthetic data: f(c) = c(1 - c^2)."""
    return c * (1.0 - c ** 2)


# ============================================================================
# Analytical solutions / derivatives (for verification)
# ============================================================================

def mittag_leffler(alpha, z, num_terms=300):
    """
    One-parameter Mittag-Leffler function E_alpha(z) = sum_k z^k / Gamma(alpha k + 1).

    The linear fractional ODE  D^beta c = lam * c, c(0) = c0  has the exact
    solution  c(t) = c0 * E_beta(lam * t^beta).

    Direct series summation in float64; adequate for |z| <~ 3, which covers
    every test in this suite (t in [0, 1], lam = 1).
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z)
    for k in range(num_terms):
        term = z ** k / sp_gamma(alpha * k + 1.0)
        out += term
        if np.all(np.abs(term) < 1e-16):
            break
    return out


def caputo_power(t, p, beta):
    """
    Analytical Caputo derivative of c(t) = t^p for p > 0, 0 < beta < 1:

        D^beta [t^p] = Gamma(p+1) / Gamma(p+1-beta) * t^(p-beta).

    Valid for any real p > 0 (not just integers); in particular p = beta gives
    D^beta [t^beta] = Gamma(1+beta), a constant -- the "initial layer" test
    function class that true solutions of the fractional IVP belong to.
    """
    t = np.asarray(t, dtype=np.float64)
    coeff = sp_gamma(p + 1.0) / sp_gamma(p + 1.0 - beta)
    with np.errstate(invalid="ignore"):
        out = coeff * np.where(t > 0, t, 1.0) ** (p - beta)
    return np.where(t > 0, out, 0.0)


def caputo_sin(t, A, omega, beta, K=80):
    """
    Analytical Caputo derivative of c(t) = A sin(omega t) via the term-by-term
    series (copied from the original verification code):

        D^beta [A sin(wt)] = A sum_{k>=0} (-1)^k w^(2k+1) t^(2k+1-beta)
                                           / Gamma(2k+2-beta).
    """
    t = np.asarray(t, dtype=np.float64)
    result = np.zeros_like(t)
    for k in range(K):
        e = 2 * k + 1
        result += (-1) ** k * omega ** e * t ** (e - beta) / sp_gamma(e + 1.0 - beta)
    return A * result


# ============================================================================
# L1 scheme for the Caputo derivative (numpy mirror of the training code)
# ============================================================================

def caputo_l1_np(c, beta, h):
    """
    L1 discretization of the Caputo derivative on a uniform grid,

        D^beta c(t_n) ~= h^(-beta)/Gamma(2-beta)
                         * sum_{j=0}^{n-1} b_j (c_{n-j} - c_{n-j-1}),
        b_j = (j+1)^(1-beta) - j^(1-beta).

    Algorithm identical to caputo_derivative() in train_virality.py (the
    production training code); this numpy version exists so the verification
    scripts do not need JAX.  pinn_lib.caputo_l1 (JAX, matrix form) is checked
    against this implementation in verify_l1_caputo.py.

    Args:
        c: 1-D array of samples on the uniform grid, shape (N,)
        beta: fractional order in (0, 1)
        h: grid spacing
    Returns:
        1-D array, entry 0 is 0 by convention (derivative undefined there).
    """
    c = np.asarray(c, dtype=np.float64)
    N = len(c)
    j = np.arange(N, dtype=np.float64)
    weights = (j + 1.0) ** (1.0 - beta) - j ** (1.0 - beta)
    coef = h ** (-beta) / sp_gamma(2.0 - beta)
    dc = np.diff(c)
    Dbc = np.zeros(N)
    for n in range(1, N):
        # dc[n-1::-1] = [c_n - c_{n-1}, ..., c_1 - c_0]
        Dbc[n] = np.dot(weights[:n], dc[n - 1 :: -1])
    return coef * Dbc


# ============================================================================
# Adams-Bashforth-Moulton forward solver (CORRECTED weights)
# ============================================================================

def abm_corrector_first_weight(n, beta):
    """Diethelm-Ford-Freed corrector weight a_{0,n+1} = n^(b+1) - (n-b)(n+1)^b."""
    return n ** (beta + 1.0) - (n - beta) * (n + 1.0) ** beta


def solve_fractional_ode(c0, t_grid, beta, f, alpha, legacy_bug=False):
    """
    Solve  D^beta c = alpha * f(c), c(0) = c0  on a uniform grid with the
    Adams-Bashforth-Moulton predictor-corrector (Diethelm-Ford-Freed 2002).

    Predictor (fractional Adams-Bashforth, rectangle rule):
        c^P_{n+1} = c0 + h^beta/Gamma(beta+1) * sum_j b_{j,n+1} f_j,
        b_{j,n+1} = (n+1-j)^beta - (n-j)^beta
    Corrector (fractional Adams-Moulton, product trapezoidal rule):
        c_{n+1}   = c0 + h^beta/Gamma(beta+2)
                    * [ a_{0,n+1} f_0 + sum_{j=1}^n a_{j,n+1} f_j + f(c^P_{n+1}) ]
        a_{0,n+1} = n^(beta+1) - (n-beta)(n+1)^beta        <-- CORRECTED
        a_{j,n+1} = (n-j+2)^(beta+1) + (n-j)^(beta+1) - 2(n-j+1)^(beta+1)

    Weights depend only on the lag (n - j), so they are precomputed once as
    arrays B and A (this also makes the solver fast enough for the fine grids
    used in data generation).

    Args:
        c0: initial condition
        t_grid: uniform time grid, shape (N,)
        beta: fractional order in (0, 1)
        f: reaction function, callable on scalars
        alpha: scaling parameter
        legacy_bug: if True, reproduce the ORIGINAL buggy weight
            a_{0,n+1} = 1.0 -- kept only so verify_abm_solver.py can
            demonstrate the size of the bug. Never use for data generation.

    Returns:
        c: solution samples on t_grid, shape (N,)
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    N = len(t_grid)
    h = t_grid[1] - t_grid[0]
    b1 = beta + 1.0

    # Toeplitz weight tables indexed by lag m = n - j (predictor) / n + 1 - j (corrector)
    m = np.arange(N + 1, dtype=np.float64)
    B = (m + 1.0) ** beta - m ** beta                     # B[m], m >= 0
    A = np.zeros(N + 1)
    A[1:] = (m[1:] + 1.0) ** b1 - 2.0 * m[1:] ** b1 + (m[1:] - 1.0) ** b1  # A[m], m >= 1

    coef_pred = h ** beta / sp_gamma(beta + 1.0)
    coef_corr = h ** beta / sp_gamma(beta + 2.0)

    c = np.empty(N)
    fv = np.empty(N)
    c[0] = c0
    fv[0] = f(c0)

    for n in range(N - 1):
        # predictor: sum_{j=0}^{n} B[n-j] * f_j
        c_pred = c0 + coef_pred * alpha * np.dot(B[n::-1], fv[: n + 1])

        # corrector: a0 * f_0 + sum_{j=1}^{n} A[n+1-j] * f_j + 1.0 * f(c_pred)
        a0 = 1.0 if legacy_bug else abm_corrector_first_weight(n, beta)
        s = a0 * fv[0] + f(c_pred)
        if n >= 1:
            s += np.dot(A[n:0:-1], fv[1 : n + 1])
        c[n + 1] = c0 + coef_corr * alpha * s
        fv[n + 1] = f(c[n + 1])

    return c


# ============================================================================
# Synthetic data generation (fine-grid solve + subsample)
# ============================================================================

def generate_synthetic_data(num_samples, num_timesteps, beta, alpha, f=f_cubic,
                            c0_range=(0.01, 0.3), seed=42, fine_points=8192):
    """
    Generate synthetic trajectories of  D^beta c = alpha f(c)  on the uniform
    observation grid t in [0, 1] with `num_timesteps` points.

    Unlike the original validation code (which solved directly at the coarse
    observation resolution, so the training data inherited the solver's
    discretization error), each trajectory is solved on a grid ~`fine_points`
    wide and then SUBSAMPLED to the observation grid.  With the corrected ABM
    scheme (interior order 1+beta) the values at the observation nodes are
    accurate to <= ~1e-5 even in the worst (smallest-beta) case studied here
    -- verified in verify_abm_solver.py TEST 4 -- i.e. the data are
    numerically exact for all practical purposes and any bias measured
    downstream is attributable to the parameter-recovery estimator, not to
    the data.

    The c0 sampling (uniform on c0_range, numpy legacy RandomState) reproduces
    the exact c0 values used in the original experiments for seed = 42.

    Args:
        num_samples: number of trajectories
        num_timesteps: observation grid size N (48 in the paper setting)
        beta, alpha, f: ground-truth model
        c0_range: initial-condition range
        seed: RNG seed for the c0 draw
        fine_points: target fine-grid size; the actual refine factor is
            round to an integer multiple of (num_timesteps - 1)

    Returns:
        t: shape (num_samples, num_timesteps, 1), the observation grid
        c: shape (num_samples, num_timesteps, 1), clean trajectories
        c0_values: shape (num_samples,)
    """
    rng = np.random.RandomState(seed)
    c0_values = rng.uniform(c0_range[0], c0_range[1], num_samples)

    refine = max(1, int(round((fine_points - 1) / (num_timesteps - 1))))
    N_fine = (num_timesteps - 1) * refine + 1
    t_fine = np.linspace(0.0, 1.0, N_fine)

    c = np.empty((num_samples, num_timesteps))
    for i in range(num_samples):
        c[i] = solve_fractional_ode(c0_values[i], t_fine, beta, f, alpha)[::refine]

    t_obs = np.linspace(0.0, 1.0, num_timesteps)
    t = np.tile(t_obs, (num_samples, 1))[:, :, np.newaxis]
    return t, c[:, :, np.newaxis], c0_values


def load_or_generate_dataset(cache_dir, num_samples, num_timesteps, beta, alpha,
                             seed=42, fine_points=8192):
    """
    Cached wrapper around generate_synthetic_data (generation is deterministic
    in all arguments, so caching only saves recompute time across the
    validation scripts).  Returns (t, c, c0_values) like the generator.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag = (f"S{num_samples}_N{num_timesteps}_b{beta:g}_a{alpha:g}"
           f"_seed{seed}_fine{fine_points}")
    path = os.path.join(cache_dir, f"synthetic_{tag}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["t"], d["c"], d["c0_values"]
    t, c, c0_values = generate_synthetic_data(
        num_samples, num_timesteps, beta, alpha,
        seed=seed, fine_points=fine_points)
    np.savez_compressed(path, t=t, c=c, c0_values=c0_values)
    return t, c, c0_values
