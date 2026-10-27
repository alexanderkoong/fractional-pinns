#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
fractional_solvers.py -- THE single fractional-ODE integrator for this repo
=============================================================================
Every inference-time rollout of  D^beta c(t) = alpha * f(c)  in this codebase
(eval_harness.pinn_reference_predict_fn, hour12_history_aware's history-aware
solve, the corrected-solver comparison script) imports its stepper from here.
Do not re-derive or inline a copy of this loop anywhere else -- that
duplication is exactly what caused the bug this module fixes.

THE BUG THIS REPLACES
----------------------
The original inference-time solver (`frankenstein_ec2.solve_from_single_point`,
copied into eval_harness.solve_fractional_from_point and inlined again in
hour12_history_aware.py) stepped with an Adams-Bashforth-Moulton-style update
(coef = h^beta / Gamma(beta+1), history sums over past f-values) but used
L1-SCHEME weights (j+1)^(1-beta) - j^(1-beta). A true fractional ABM uses
exponent beta, not 1-beta. The two coincide ONLY at beta = 0.5. Benchmarked
against the exact solution for constant f(c)=K (c(t) = c0 + alpha*K*t^beta /
Gamma(1+beta)), the old scheme delivered ~473% of the true growth increment at
beta=0.3 (overshoots ~5x; the [0,1] clip hid it), ~101% at beta=0.5
(accidentally correct), ~22% at beta=0.7, ~5% at beta=0.9. See
PROJECT_STATE_AND_ROADMAP.md Sec. 5 for the full writeup and data-level
consequences.

THE FIX
-------
Step by inverting the SAME L1 discretization of the Caputo derivative that
the PINN training loss uses (train_virality.py::caputo_derivative), so
training and rollout finally share one discretization:

    L1:  D^b c(t_n) ~ [h^-b / G(2-b)] * SUM_{j=0}^{n-1} b_j (c_{n-j} - c_{n-j-1}),
         b_j = (j+1)^(1-b) - j^(1-b)

    Setting D^b c(t_n) = alpha*f(c_n) and solving for c_n (implicit; a few
    fixed-point corrector iterations):

    c_n = c_{n-1} + G(2-b) h^b alpha f(c_n) - SUM_{j=1}^{n-1} b_j (c_{n-j} - c_{n-j-1})

Verified to <1% of exact analytic solutions across beta in [0.2, 0.9] (see
verify_analytic() below, wired into eval_harness.selftest() as a permanent
gate) and to ~1e-9 residual against the training-side discretization.
=============================================================================
"""

import math

import numpy as np


# =============================================================================
# CORE MATH
# =============================================================================
def l1_weights(beta, n):
    """L1 weights b_j = (j+1)^(1-beta) - j^(1-beta); b_0 = 1."""
    j = np.arange(n, dtype=float)
    return (j + 1.0) ** (1.0 - beta) - j ** (1.0 - beta)


def fode_solve_l1(f, alpha, beta, c0, n_steps, h,
                  n_corrector=3, corrector_tol=1e-12,
                  clip01=False, prefix=None):
    """
    Solve D^beta c(t) = alpha * f(c),  c(0) = c0, on a uniform grid of
    `n_steps` points with spacing `h`, by inverting the L1 Caputo
    discretization used in the PINN training loss.

    Update rule (implicit in f, solved by fixed-point iteration):
        c_n = c_{n-1} + G(2-b) h^b alpha f(c_n)
              - SUM_{j=1}^{n-1} b_j (c_{n-j} - c_{n-j-1})

    Args:
        f            : callable, reaction function f(c) (scalar -> scalar)
        alpha, beta  : ODE parameters (0 < beta < 1)
        c0           : initial condition (ignored if `prefix` given)
        n_steps, h   : grid size and spacing
        n_corrector  : fixed-point corrector iterations (converges fast:
                       contraction factor is G(2-b) h^b alpha |f'| << 1)
        clip01       : clip the state to [0,1] each step and evaluate f on
                       clipped values (matches the production convention and
                       stabilizes polynomial f outside [0,1])
        prefix       : optional array of already-OBSERVED values c_0..c_m.
                       The solver continues from index m+1 with the observed
                       increments included in the memory sum. This is how to
                       condition a fractional model on an observed early
                       trajectory instead of restarting at t=0 (used by
                       hour12_history_aware.py's history-aware solve).

    Returns:
        c : np.ndarray of length n_steps
    """
    assert 0.0 < beta < 1.0, "beta must be in (0, 1)"
    G = math.gamma(2.0 - beta) * (h ** beta)
    b = l1_weights(beta, n_steps)
    c = np.zeros(n_steps, dtype=float)

    if prefix is not None and len(prefix) > 0:
        m = min(len(prefix), n_steps)
        c[:m] = np.asarray(prefix, dtype=float)[:m]
        start = m
    else:
        c[0] = float(c0)
        start = 1

    if clip01:
        def _f(x):
            return float(f(min(max(x, 0.0), 1.0)))
    else:
        def _f(x):
            return float(f(x))

    for n in range(start, n_steps):
        # memory term: SUM_{j=1}^{n-1} b_j (c_{n-j} - c_{n-j-1})
        if n > 1:
            hist = float(np.dot(b[1:n], c[n - 1:0:-1] - c[n - 2::-1]))
        else:
            hist = 0.0
        base = c[n - 1] - hist

        x = base + G * alpha * _f(c[n - 1])            # explicit predictor
        for _ in range(n_corrector):                    # implicit correctors
            x_new = base + G * alpha * _f(x)
            if abs(x_new - x) < corrector_tol:
                x = x_new
                break
            x = x_new

        c[n] = min(max(x, 0.0), 1.0) if clip01 else x
    return c


def fode_solve_abm(f, alpha, beta, c0, n_steps, h):
    """
    Independent reference: the standard Diethelm-Ford-Freed fractional
    Adams-Bashforth-Moulton predictor-corrector (correct weights, exponent
    beta). Structurally different from fode_solve_l1; used to cross-validate
    it in verify_analytic() and in regression tests.
    """
    assert 0.0 < beta < 1.0
    T = n_steps
    c = np.zeros(T)
    g = np.zeros(T)
    c[0] = float(c0)
    g[0] = alpha * float(f(c[0]))
    hb = h ** beta

    for n in range(0, T - 1):
        j = np.arange(n + 1, dtype=float)
        # predictor weights: (h^b / b) [(n+1-j)^b - (n-j)^b]
        bw = (hb / beta) * ((n + 1 - j) ** beta - (n - j) ** beta)
        yp = c[0] + (1.0 / math.gamma(beta)) * float(np.dot(bw, g[:n + 1]))

        # corrector weights
        aw = np.empty(n + 1)
        aw[0] = n ** (beta + 1) - (n - beta) * (n + 1) ** beta
        if n >= 1:
            jj = np.arange(1, n + 1, dtype=float)
            aw[1:] = ((n - jj + 2) ** (beta + 1)
                      + (n - jj) ** (beta + 1)
                      - 2.0 * (n - jj + 1) ** (beta + 1))
        c[n + 1] = c[0] + (hb / math.gamma(beta + 2)) * (
            float(np.dot(aw, g[:n + 1])) + alpha * float(f(yp)))
        g[n + 1] = alpha * float(f(c[n + 1]))
    return c


def l1_caputo_derivative(c, beta, h):
    """
    TRAINING-side discretization: numpy replica of
    train_virality.py::caputo_derivative for a single 1-D trajectory. Used to
    check that a rollout actually satisfies D^b c = alpha f(c) under the
    discretization the PINN was trained with.
    """
    c = np.asarray(c, dtype=float)
    T = len(c)
    b = l1_weights(beta, T)
    d = np.diff(c)                       # d[i] = c_{i+1} - c_i
    coef = (h ** (-beta)) / math.gamma(2.0 - beta)
    out = np.zeros(T)
    for n in range(1, T):
        out[n] = coef * float(np.dot(b[:n], d[n - 1::-1]))
    return out


def old_scheme_rollout(f, alpha, beta, c0, n_steps, h):
    """
    *** DO NOT USE FOR PRODUCTION INFERENCE. ***
    Faithful numerical port of the ORIGINAL, BUGGY stepping (ABM structure
    with mismatched L1 weights -- see module docstring). Kept ONLY so
    regression tests / the old-vs-corrected comparison script can prove the
    new solver actually differs from the old one, and so the bug stays
    documented in runnable form instead of just in prose.
    """
    T = n_steps
    b = np.zeros(T)
    a = np.zeros(T)
    for j in range(T):
        b[j] = ((j + 1) ** (1 - beta) - j ** (1 - beta)) if j > 0 else 1.0
        if j == 0:
            a[j] = 1.0
        elif j < T - 1:
            a[j] = (j + 1) ** (1 - beta) - j ** (1 - beta)
        else:
            a[j] = (1 - beta) * j ** (-beta)
    coef = (h ** beta) / math.gamma(beta + 1.0)

    c = np.zeros(T)
    c[0] = min(max(float(c0), 0.0), 1.0)
    fv = np.zeros(T)
    fv[0] = float(f(c[0]))
    for i in range(1, T):
        sp = float(np.dot(b[i - 1::-1], fv[:i]))          # SUM b[i-j-1] f_j
        cp = min(max(c[0] + coef * alpha * sp, 0.0), 1.0)
        sc = float(f(cp)) * a[0] + float(np.dot(a[i:0:-1], fv[:i]))
        c[i] = min(max(c[0] + coef * alpha * sc, 0.0), 1.0)
        fv[i] = float(f(c[i]))
    return np.maximum.accumulate(c)


def mittag_leffler(beta, z, max_terms=300, tol=1e-16):
    """One-parameter Mittag-Leffler E_beta(z) = SUM z^m / Gamma(beta*m + 1).
    Fine for the moderate |z| used in verify_analytic()."""
    z = float(z)
    result = 1.0
    term = 1.0
    for m in range(1, max_terms):
        term *= z * math.exp(math.lgamma(beta * (m - 1) + 1.0)
                             - math.lgamma(beta * m + 1.0))
        result += term
        if abs(term) < tol:
            break
    return result


# =============================================================================
# ANALYTIC VERIFICATION GATE
# =============================================================================
def verify_analytic(T=48, tol_pct=2.0,
                    betas=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    verbose=False):
    """
    Check fode_solve_l1 against two closed-form solutions of
    D^beta c = alpha*f(c):

        A. constant reaction f(c) = K
           exact: c(t) = c0 + alpha*K*t^beta / Gamma(1+beta)
        B. linear reaction   f(c) = k*c
           exact: c(t) = c0 * E_beta(alpha*k*t^beta)      [Mittag-Leffler]

    Raises AssertionError if the solver's error at t=1 exceeds `tol_pct` of
    the true growth increment c(1)-c0, at any tested beta. This is the same
    style of check that originally exposed the ABM/L1-hybrid integrator bug
    (see module docstring) -- run it any time this module's numerics change,
    and never silence a failure without first checking whether the solver
    regressed.

    Returns the observed max error (%) on success.
    """
    h = 1.0 / (T - 1)
    max_err = 0.0

    K, alpha, c0 = 0.24341, 0.5, 0.05
    for beta in betas:
        exact = c0 + alpha * K / math.gamma(1.0 + beta)
        got = fode_solve_l1(lambda c: K, alpha, beta, c0, T, h)[-1]
        err = 100.0 * abs(got - exact) / (exact - c0)
        max_err = max(max_err, err)
        if verbose:
            print(f"  [const-f] beta={beta:.1f}  exact={exact:.4f}  "
                  f"got={got:.4f}  err={err:.3f}%")

    k_lin, alpha_b, c0b = 0.52697, 1.0, 0.05
    for beta in betas:
        exact = c0b * mittag_leffler(beta, alpha_b * k_lin)
        got = fode_solve_l1(lambda c: k_lin * c, alpha_b, beta, c0b, T, h)[-1]
        err = 100.0 * abs(got - exact) / (exact - c0b)
        max_err = max(max_err, err)
        if verbose:
            print(f"  [linear-f] beta={beta:.1f}  exact={exact:.4f}  "
                  f"got={got:.4f}  err={err:.3f}%")

    assert max_err < tol_pct, (
        f"fractional solver drift detected: max error {max_err:.2f}% exceeds "
        f"tolerance {tol_pct}% against analytic constant-f/linear-f solutions "
        f"(beta in {betas}). This is the same check that originally exposed "
        f"the ABM/L1-hybrid integrator bug -- do not silence it, fix the "
        f"solver in fractional_solvers.py.")
    return max_err


if __name__ == "__main__":
    err = verify_analytic(verbose=True)
    print(f"\n[ok] fractional_solvers.fode_solve_l1 matches analytic solutions "
          f"to within {err:.3f}% (max over betas tested).")
