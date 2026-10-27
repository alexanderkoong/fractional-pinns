#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fractional PINN for Social Media Virality -- whitened-full L1 objective
with a VELOCITY-CONDITIONED viral potential.

Trains the physics-informed model

    D^beta c_i(t) = alpha_i * f(c_i(t))

directly on observed tweet engagement trajectories, where:
    - D^beta is the Caputo fractional derivative, evaluated with the L1
      scheme on the uniform hourly grid (t rescaled to [0, 1])
    - beta is the shared fractional order, learned
      (parameterized beta = 0.1 + 0.89 * sigmoid(raw))
    - f(c) is the shared reaction law, a neural network with hard-coded
      boundary conditions: f(c) = exp(NN(c)) * c * (1 - c)
    - alpha_i is the per-post viral potential from a small learned head
      (below).

THE ALPHA HEAD: EARLY VELOCITY, NOT (ONLY) FOLLOWERS
----------------------------------------------------
    alpha_i = sigmoid( a * log(1 + N_i)  +  b * v_i  +  c )
    v_i     = log1p(raw total at hour 1) - log1p(raw total at hour 0)

This is the head from `final code ieee/17_velocity_alpha` (replicated in
19_velocity_final_filter_combo): after per-post max-normalization, alpha's
job is per-post SPEED, which follower count does not predict
(corr ~ 0 with hours-to-half-max on this data) but early raw velocity does.
In experiment 17 the fit abandoned followers for velocity
((a, b, c) = (+0.025, +2.614, -0.491)) and alpha spread out to 0.24-1.00;
the followers-only tanh head of earlier runs saturated at alpha = 1 for
every post (gamma unidentifiable -- the pre-diagnosed saturation regime of
the gamma-validation suite).

v_i uses RAW counts from the first two observed hours: past-only
information, no dependence on the future max that normalizes c.  In the
errors-in-variables Jacobian, v is treated as EXOGENOUS (the experiment-17
convention); the head has 3 stiff parameters, so the "reshape your own
ruler" surface this opens is small -- verified post-fit by the frozen-fit
landscape diagnostics.

Run `--velocity off` for the ablation twin: the same sigmoid head with the
velocity term removed (followers-only), so any difference is attributable
to the feature.

THE LOSS: WHITENED-FULL L1 (ported from "../L1 with log term/l1_log_term_poc.py")
---------------------------------------------------------------------------------
The pointwise L1 residual is written as a linear operator on the raw data
vector,

    r_n = [A(beta) c]_n - alpha_i * f(c_n),        n = 1 .. N-1,

and measured in DATA units through the residual's Jacobian w.r.t. the
observations, J = dr/dc_obs = A(beta) - alpha_i * f'(c_n) on the target
column:

    loss = mean over tweets of  r^T (J J^T)^(-1) r .

This is the exact Gaussian maximum-likelihood (errors-in-variables)
objective up to constants -- "how far would the observations have to move to
satisfy the physics" -- with no surrogate network, no data-fitting term, no
noise estimate, and no extra hyperparameter.  Since the loss is in squared
data units, sqrt(loss) is the implied per-node observation-noise level under
the fitted model.

The trained parameters are ONLY (f_params, alpha_params, beta).  Trajectory
forecasts are NOT produced by a fitted network: they come from
forward-solving the recovered equation (evaluation step of the paper), which
is handled by a separate script.
"""

import argparse
import os
import pickle
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit, value_and_grad
from jax.example_libraries import optimizers
from jax.scipy.special import gammaln
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# float64 throughout, matching the synthetic validation suite and the
# whitened-L1 PoC: the whitened objective solves per-tweet (N-1)x(N-1)
# linear systems, which want the extra precision, and this runs on CPU.
jax.config.update("jax_enable_x64", True)

JITTER = 1e-10  # ridge added to J J^T before solving (factorization safety)


# ============================================================================
# SECTION 1: UTILITY FUNCTIONS
# ============================================================================

def gamma_fn(x):
    """Gamma function using log-gamma for numerical stability."""
    return jnp.exp(gammaln(x))


def safe_pow(x, p):
    """
    x**p for x >= 0, safe to differentiate w.r.t. p at x = 0.

    d/dp [x**p] = x**p * log(x) is NaN at x = 0 under autodiff; the L1
    weight tables contain 0**p entries whose p-gradient must be exactly 0.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    x_safe = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, x_safe ** p, 0.0)


# ============================================================================
# SECTION 2: NEURAL NETWORK FUNCTIONS + THE ALPHA HEAD
# (only the reaction network f(c) -- there is no surrogate c(t) network)
# ============================================================================

def random_layer_params(ind, outd, num_nn, key):
    """Initialize a single layer's weights and biases."""
    scale = jnp.sqrt(2 / (ind + outd))
    W = random.normal(key, (num_nn, ind, outd)) * scale
    b = jnp.zeros((num_nn, 1, outd))
    return W, b


def init_network_params(ind, outd, width, num_nn, key, num_layers):
    """
    Initialize parameters for a multi-layer feedforward network.

    Args:
        ind: Input dimension
        outd: Output dimension
        width: Hidden layer width
        num_nn: Number of separate neural networks (1 for the shared f)
        key: JAX random key
        num_layers: Total number of layers

    Returns:
        List of (W, b) tuples for each layer
    """
    keys = random.split(key, num_layers)
    layers = [random_layer_params(ind, width, num_nn, keys[0])]
    for i in range(1, num_layers - 1):
        layers.append(random_layer_params(width, width, num_nn, keys[i]))
    layers.append(random_layer_params(width, outd, num_nn, keys[-1]))
    return layers


def fnn_fixed_boundary(params, x):
    """
    Neural network with fixed boundary conditions: f(0)=0, f(1)=0.
    Used for the reaction function f(c).

    The output is: exp(NN(x)) * x * (1-x)
    This ensures f(0) = 0 and f(1) = 0, and f > 0 on (0, 1).

    Args:
        params: List of (W, b) tuples
        x: Input tensor (engagement values in [0, 1])

    Returns:
        Output satisfying boundary conditions
    """
    boundary_term = x * (1 - x)

    for i in range(len(params) - 1):
        W, b = params[i]
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    x = jnp.exp(x @ W + b) * boundary_term

    return x


def f_prime_nn(f_params, c):
    """
    df/dc of the reaction network, computed with a jvp (one extra forward
    pass).  Needed for the residual's Jacobian w.r.t. the observations.
    """
    _, tangent = jax.jvp(lambda x: fnn_fixed_boundary(f_params, x),
                         (c,), (jnp.ones_like(c),))
    return tangent


def alpha_head(alpha_params, N_log, v, velocity_on):
    """
    The per-post viral potential (experiment-17 head):

        alpha_i = sigmoid(a * log(1+N_i) + b * v_i + c)

    Args:
        alpha_params: length-3 vector (a, b, c)
        N_log: (S, 1, 1) log(1 + followers)
        v: (S, 1, 1) early log-velocities (exogenous covariate)
        velocity_on: python bool; False drops the b*v term entirely
            (the followers-only ablation twin -- b then receives no
            gradient and stays at its initialization)

    Returns:
        alpha: (S, 1, 1) in (0, 1)
    """
    a, b, c = alpha_params[0], alpha_params[1], alpha_params[2]
    z = a * N_log + c
    if velocity_on:
        z = z + b * v
    return jax.nn.sigmoid(z)


# ============================================================================
# SECTION 3: FRACTIONAL DERIVATIVE (CAPUTO L1 SCHEME)
# Loop form (verified in the synthetic validation suite) kept as the
# self-test reference; the training loss uses the expanded matrix form.
# ============================================================================

def compute_caputo_weights(num_timesteps, beta):
    """
    Compute the weights for the L1 discretization of the Caputo derivative.

    The Caputo derivative is:
        D^β c(t) = (1/Γ(1-β)) ∫₀ᵗ c'(τ)/(t-τ)^β dτ

    Discretized using L1 scheme:
        D^β c(tₙ) ≈ (h^{-β}/Γ(2-β)) Σⱼ bⱼ [c(tₙ₋ⱼ) - c(tₙ₋ⱼ₋₁)]

    Where: bⱼ = (j+1)^{1-β} - j^{1-β}

    Args:
        num_timesteps: Number of time points
        beta: Fractional order (0 < beta < 1)

    Returns:
        weights: Array of shape (num_timesteps,) with the bⱼ values
    """
    j = jnp.arange(num_timesteps, dtype=jnp.float64)
    return safe_pow(j + 1.0, 1.0 - beta) - safe_pow(j, 1.0 - beta)


def caputo_derivative(c, t, beta):
    """
    Compute the Caputo fractional derivative using the L1 scheme (loop form).

    Kept as the reference implementation for the startup self-test and the
    physics-fit diagnostic plot; the training loss uses l1_expanded_matrix
    (verified identical at startup).

    Args:
        c: Engagement values, shape (num_samples, num_timesteps, 1)
        t: Time points (unused; uniform grid on [0, 1] is assumed)
        beta: Fractional order (scalar)

    Returns:
        D^β c: Fractional derivative, same shape as c
               (First time point is set to 0 as derivative is undefined there)
    """
    num_samples, num_timesteps, num_features = c.shape
    h = 1.0 / (num_timesteps - 1)
    weights = compute_caputo_weights(num_timesteps, beta)
    coef = (h ** (-beta)) / gamma_fn(2 - beta)

    dc = jnp.diff(c, axis=1)
    dc = jnp.concatenate([jnp.zeros((num_samples, 1, num_features)), dc], axis=1)

    Dbc = jnp.zeros_like(c)
    for n in range(1, num_timesteps):
        dc_slice = dc[:, 1:n+1, :]
        dc_reversed = jnp.flip(dc_slice, axis=1)
        w = weights[:n]
        weighted_sum = jnp.sum(w[None, :, None] * dc_reversed, axis=1, keepdims=True)
        Dbc = Dbc.at[:, n, :].set(weighted_sum[:, 0, :])

    return coef * Dbc


def l1_expanded_matrix(num_timesteps, beta):
    """
    A(beta) with shape (N-1, N): the L1 derivative's linear action on the raw
    data vector,  [A c]_n = D^beta c(t_n)  for target nodes n = 1 .. N-1.

    Row for target node n (ported from the whitened-L1 PoC; verified against
    caputo_derivative in the startup self-test):
        A[n, n] = coef * b_0
        A[n, k] = coef * (b_{n-k} - b_{n-k-1})   for 1 <= k <= n-1
        A[n, 0] = -coef * b_{n-1}
    with coef = h^(-beta)/Gamma(2-beta) and b_j the L1 weights above.
    """
    N = num_timesteps
    h = 1.0 / (N - 1)
    b = compute_caputo_weights(N, beta)
    coef = h ** (-beta) / gamma_fn(2.0 - beta)

    n = jnp.arange(1, N, dtype=jnp.float64)[:, None]   # target nodes 1..N-1
    k = jnp.arange(N, dtype=jnp.float64)[None, :]
    lag = jnp.clip(n - k, 0, N - 1).astype(int)         # n - k
    A = jnp.where((k >= 1) & (k <= n - 1), b[lag] - b[lag - 1], 0.0)
    A = jnp.where(k == n, b[0], A)
    A = jnp.where(k == 0, -b[jnp.clip(n.astype(int) - 1, 0, N - 1)][:, 0:1]
                  * jnp.ones_like(A), A)
    return coef * A


# ============================================================================
# SECTION 4: WHITENED-FULL LOSS
# ============================================================================

def residual_and_jacobian(c, beta, fc, fprime, alpha):
    """
    The physics residual and its Jacobian w.r.t. the observations.

    r : (S, N-1)     r_n = [A(beta) c]_n - alpha_i f(c_n),  n = 1 .. N-1
    J : (S, N-1, N)  dr/dc_obs = A(beta) - alpha_i f'(c_n) on the target
                     column (lower-triangular structure)

    Generalizes the PoC's residual_and_jacobian (alpha = 1 there) to
    PER-POST alpha_i of shape (S, 1, 1): alpha multiplies both the
    residual's f term and the Jacobian's f' term.  The velocity covariate
    inside alpha is treated as exogenous (see module docstring), so J takes
    no derivative through the head.
    """
    S, N, _ = c.shape
    A = l1_expanded_matrix(N, beta)                                # (N-1, N)
    a = alpha[:, 0, 0]                                             # (S,)
    r = jnp.einsum("mk,skf->smf", A, c)[:, :, 0] - a[:, None] * fc[:, 1:, 0]
    eye_target = jnp.eye(N)[1:, :]                                 # (N-1, N)
    J = A[None, :, :] \
        - (a[:, None, None] * fprime[:, 1:, 0][:, :, None]) * eye_target[None, :, :]
    return r, J


def loss_whitened_full(c, beta, fc, fprime, alpha):
    """
    Exact data-space whitening:  mean over tweets/nodes of r^T (JJ^T)^(-1) r.

    The result is in squared DATA units, so sqrt(loss) is the implied
    per-node observation-noise level under the fitted model.
    """
    r, J = residual_and_jacobian(c, beta, fc, fprime, alpha)
    JJt = jnp.einsum("snk,smk->snm", J, J)
    JJt = JJt + JITTER * jnp.eye(JJt.shape[-1])[None, :, :]
    x = jnp.linalg.solve(JJt, r[:, :, None])[:, :, 0]
    return jnp.mean(r * x)


def create_loss_function(c_data, N_log, v, velocity_on):
    """
    Create the training loss: the whitened-full physics residual evaluated
    DIRECTLY on the observations.

        loss = mean over tweets of  r^T (J J^T)^(-1) r,
        r = A(beta) c_obs - alpha_i f(c_obs),   J = dr/dc_obs,
        alpha_i = sigmoid(a log(1+N_i) + b v_i + c).

    There is no surrogate c(t) network, no data-fitting term, no burn-in,
    and no noise estimate: the whitening supplies the noise model.

    Args:
        c_data: Observed engagement (num_samples, num_timesteps, 1)
        N_log: Log-transformed follower counts (num_samples, 1, 1)
        v: Early log-velocities (num_samples, 1, 1)
        velocity_on: python bool -- False gives the followers-only ablation

    Returns:
        loss_fn(f_params, alpha_params, beta_params) -> scalar loss
    """

    def loss_fn(f_params, alpha_params, beta_params):
        # Constrain beta to (0.1, 0.99) using sigmoid
        beta = 0.1 + 0.89 * jax.nn.sigmoid(beta_params)

        alpha = alpha_head(alpha_params, N_log, v, velocity_on)

        fc = fnn_fixed_boundary(f_params, c_data)
        fprime = f_prime_nn(f_params, c_data)

        return loss_whitened_full(c_data, beta, fc, fprime, alpha)

    return loss_fn


# ============================================================================
# SECTION 5: VELOCITY COVARIATE
# ============================================================================

def compute_velocities(c_np, peak):
    """
    v_i = log1p(raw total at hour 1) - log1p(raw total at hour 0)

    The experiment-17 definition, computed from the dataset's own arrays:
    hour values are taken from the trajectory the model actually trains on
    (the dataset's hourly grid), converted back to raw units with each
    post's stored window peak -- the same E_max the dataset builder used as
    the normalizer.  This npz-based computation was verified bit-identical
    (max abs diff 0.0) against the original raw-snapshot derivation by the
    testing pipeline's independent re-computation, so the raw CSVs are not
    needed at training time.

    Hours 0-1 are past-only information: no dependence on the future values
    of the trajectory beyond E_max's role as the (already-used) normalizer.
    """
    e0 = c_np[:, 0, 0] * peak
    e1 = c_np[:, 1, 0] * peak
    return np.log1p(e1) - np.log1p(e0)


# ============================================================================
# SECTION 6: SELF-TEST
# ============================================================================

def self_test(c_data):
    """
    Run before spending hours training:
      1) the expanded operator A(beta) must reproduce the loop-form L1
         derivative on the real data to ~machine precision;
      2) gradients of the whitened loss must be finite at init-like values,
         for both the velocity-on and velocity-off heads.
    """
    c_small = c_data[:8]
    for beta in (0.3, 0.5, 0.8):
        A = l1_expanded_matrix(c_small.shape[1], beta)
        lhs = np.array(jnp.einsum("mk,skf->smf", A, c_small))[:, :, 0]
        rhs = np.array(caputo_derivative(c_small, None, beta))[:, 1:, 0]
        err = float(np.max(np.abs(lhs - rhs)))
        assert err < 1e-9, f"expanded L1 mismatch at beta={beta}: {err:.2e}"

    key = random.PRNGKey(0)
    f_params = init_network_params(1, 1, 30, 1, key, 4)
    N_log_fake = jnp.full((c_small.shape[0], 1, 1), 6.0)
    v_fake = jnp.full((c_small.shape[0], 1, 1), 0.5)
    for von in (True, False):
        loss_fn = create_loss_function(c_small, N_log_fake, v_fake, von)
        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(
            f_params, jnp.array([0.1, 1.0, 0.0]), jnp.array(0.85))
        leaves = jax.tree_util.tree_leaves(grads)
        finite = all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
        assert finite, f"non-finite gradient (velocity_on={von})"
    print("[self-test] expanded operator matches loop L1 (< 1e-9); "
          "whitened-loss gradients finite for both heads", flush=True)


# ============================================================================
# SECTION 7: TRAINING LOOP (with checkpoint/resume)
# ============================================================================

def _ckpt_path(results_dir):
    return os.path.join(results_dir, "train_checkpoint.pkl")


def train_model(c_data, N_log, v, config, results_dir="results"):
    """
    Train the fractional PINN model (f, alpha head, beta) with the
    whitened-full L1 objective.

    Checkpoints (raw parameters + step + history) are written every
    `ckpt_every` steps; if a checkpoint exists on startup, training resumes
    from it (Adam moments are rebuilt, costing only a brief warm-up).

    Args:
        c_data: Engagement data, shape (num_samples, num_timesteps, 1)
        N_log: Log-transformed follower counts, shape (num_samples,)
        v: Early log-velocities, shape (num_samples,)
        config: Dictionary with training configuration
        results_dir: Directory for checkpoints

    Returns:
        trained_params: Tuple of (f_params, alpha_params, beta_params)
        history: Dictionary with training history
    """
    num_samples, num_timesteps, _ = c_data.shape

    # Unpack config
    seed = config.get('seed', 42)
    num_steps = config.get('num_steps', 15000)
    learning_rate = config.get('learning_rate', 1e-3)
    print_every = config.get('print_every', 250)
    ckpt_every = config.get('ckpt_every', 2500)
    f_width = config.get('f_width', 30)
    f_layers = config.get('f_layers', 4)
    velocity_on = bool(config.get('velocity_on', True))

    N_log_jax = jnp.array(N_log, dtype=jnp.float64).reshape(num_samples, 1, 1)
    v_jax = jnp.array(v, dtype=jnp.float64).reshape(num_samples, 1, 1)

    print("=" * 60)
    print("FRACTIONAL PINN TRAINING (whitened-full L1, direct-on-data)")
    print("=" * 60)
    print(f"Number of tweets: {num_samples}")
    print(f"Timesteps per tweet: {num_timesteps}")
    print(f"Training steps: {num_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Loss: mean[ r^T (J J^T)^-1 r ]  (data units; no burn-in, no")
    print(f"      surrogate, no data term, no noise estimate)")
    print(f"Alpha head: sigmoid(a*log(N+1) + b*v + c), velocity "
          f"{'ON' if velocity_on else 'OFF (followers-only ablation)'}")
    print("=" * 60)

    # === Initialize Parameters (or resume from checkpoint) ===
    start_step = 0
    ckpt = _ckpt_path(results_dir)
    saved = None
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as fh:
            saved = pickle.load(fh)
        if "alpha_params" not in saved:
            print("[resume] found checkpoint from an OLDER trainer format "
                  "(no alpha head) -- ignoring it and starting fresh; "
                  "archive results/ between differently-configured runs")
            saved = None
    if saved is not None:
        f_params = [(jnp.array(W), jnp.array(b)) for W, b in saved["f_params"]]
        alpha_params = jnp.array(saved["alpha_params"])
        beta_params = jnp.array(saved["beta_params"])
        start_step = int(saved["step"]) + 1
        history = saved["history"]
        print(f"[resume] loaded checkpoint at step {saved['step']}")
    else:
        key = random.PRNGKey(seed)
        keys = random.split(key, 2)

        # f_params: 4 layers x width 30 -- the same architecture used in the
        # synthetic joint validation and the whitened-L1 PoC
        f_params = init_network_params(
            ind=1, outd=1, width=f_width, num_nn=1,
            key=keys[0], num_layers=f_layers
        )

        # alpha head (a, b, c): neutral start with mild followers weight,
        # unit velocity weight, zero bias -- same init for ON and OFF twins
        alpha_params = jnp.array([0.1, 1.0, 0.0])

        # beta_params: sigmoid(0) = 0.5 -> initial beta ~ 0.545
        beta_params = jnp.array(0.0)

        history = {
            'step': [], 'loss': [], 'beta': [],
            'head_a': [], 'head_b': [], 'head_c': [],
            'alpha_mean': [], 'alpha_min': [], 'alpha_max': [],
            'sigma_implied': []
        }

    init_alpha = alpha_head(alpha_params, N_log_jax, v_jax, velocity_on)
    a0, b0, c0 = (float(x) for x in np.asarray(alpha_params))
    print(f"f_params: {f_layers} layers, width {f_width}, 1 shared network")
    print(f"alpha head init (a, b, c) = ({a0:.3f}, {b0:.3f}, {c0:.3f}) -> "
          f"alpha min={float(init_alpha.min()):.4f}, "
          f"max={float(init_alpha.max()):.4f}, mean={float(init_alpha.mean()):.4f}")
    print(f"beta: {float(0.1 + 0.89 * jax.nn.sigmoid(beta_params)):.4f}")
    print("=" * 60)

    # === Create Loss Function ===
    loss_fn = create_loss_function(c_data, N_log_jax, v_jax, velocity_on)

    # === Setup Optimizer ===
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init((f_params, alpha_params, beta_params))

    # === JIT-compile the update step ===
    @jit
    def update_step(step_i, opt_state):
        params = get_params(opt_state)
        loss, grads = value_and_grad(loss_fn, argnums=(0, 1, 2))(*params)
        opt_state = opt_update(step_i, grads, opt_state)
        return loss, opt_state

    # === Training Loop ===
    print("Starting training...", flush=True)
    t_start = time()

    for step in range(start_step, num_steps + 1):
        loss, opt_state = update_step(step, opt_state)

        if step % print_every == 0:
            # head/beta/alpha are closed-form functions of the raw
            # parameters -- no extra forward pass needed for logging
            _, head_raw, beta_raw = get_params(opt_state)
            beta = float(0.1 + 0.89 * jax.nn.sigmoid(beta_raw))
            al = alpha_head(head_raw, N_log_jax, v_jax, velocity_on)
            ha, hb, hc = (float(x) for x in np.asarray(head_raw))
            loss_f = float(loss)
            sigma_implied = float(np.sqrt(max(loss_f, 0.0)))

            history['step'].append(step)
            history['loss'].append(loss_f)
            history['beta'].append(beta)
            history['head_a'].append(ha)
            history['head_b'].append(hb)
            history['head_c'].append(hc)
            history['alpha_mean'].append(float(al.mean()))
            history['alpha_min'].append(float(al.min()))
            history['alpha_max'].append(float(al.max()))
            history['sigma_implied'].append(sigma_implied)

            print(f"Step {step:5d} | Whitened loss: {loss_f:.6e} | "
                  f"beta: {beta:.4f} | head(a,b,c): ({ha:+.3f},{hb:+.3f},{hc:+.3f}) | "
                  f"alpha [{float(al.min()):.3f},{float(al.max()):.3f}] "
                  f"mean {float(al.mean()):.3f} | "
                  f"sigma: {sigma_implied:.5f} | {time() - t_start:5.0f}s",
                  flush=True)

        if step % ckpt_every == 0 and step > start_step:
            params = get_params(opt_state)
            with open(ckpt, "wb") as fh:
                pickle.dump({
                    "step": step,
                    "f_params": [(np.array(W), np.array(b))
                                 for W, b in params[0]],
                    "alpha_params": np.array(params[1]),
                    "beta_params": np.array(params[2]),
                    "history": history,
                }, fh)

    t_end = time()
    print("=" * 60)
    print(f"Training complete in {t_end - t_start:.2f} seconds")
    print("=" * 60)

    # === Extract Final Parameters ===
    final_params = get_params(opt_state)
    f_params, alpha_params, beta_params = final_params

    final_beta = float(0.1 + 0.89 * jax.nn.sigmoid(beta_params))
    fa, fb, fc_ = (float(x) for x in np.asarray(alpha_params))
    final_alpha = alpha_head(alpha_params, N_log_jax, v_jax, velocity_on)

    print(f"Final beta (fractional order): {final_beta:.4f}")
    print(f"Final alpha head: alpha = sigmoid({fa:+.4f}*log(N+1) "
          f"{'+ %+.4f*v ' % fb if velocity_on else ''}{fc_:+.4f})")
    print(f"  alpha: min={float(final_alpha.min()):.4f}, "
          f"max={float(final_alpha.max()):.4f}, mean={float(final_alpha.mean()):.4f}")

    return final_params, history


# ============================================================================
# SECTION 8: VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """Plot training loss and parameter trajectories."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    steps = history['step']

    # Whitened loss (sqrt = implied observation noise)
    axes[0, 0].plot(steps, history['loss'])
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Whitened loss')
    axes[0, 0].set_title('Whitened-full loss (sqrt = implied noise sigma)')
    axes[0, 0].set_yscale('log')

    # Beta evolution
    axes[0, 1].plot(steps, history['beta'])
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('beta')
    axes[0, 1].set_title('Fractional Order beta')
    axes[0, 1].set_ylim([0, 1])

    # Head coefficients
    axes[1, 0].plot(steps, history['head_a'], label='a (followers)')
    axes[1, 0].plot(steps, history['head_b'], label='b (velocity)')
    axes[1, 0].plot(steps, history['head_c'], label='c (bias)')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_title('Alpha head coefficients')
    axes[1, 0].legend()

    # Alpha spread evolution
    axes[1, 1].plot(steps, history['alpha_mean'], label='mean')
    axes[1, 1].plot(steps, history['alpha_min'], label='min')
    axes[1, 1].plot(steps, history['alpha_max'], label='max')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('alpha')
    axes[1, 1].set_title('Viral potential spread')
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history to {save_path}")


def plot_physics_fit(t, c_data, f_params, beta_params, alpha,
                     num_plots=6, save_path='physics_fit.png', max_hour=23):
    """
    Plot the two sides of the governing equation for a few tweets:
    D^beta c_data (from the data) vs alpha_i * f(c_data) (from the model).

    Note this shows the RAW pointwise residual for interpretability; the
    training objective is its whitened (data-space) norm.
    """
    num_samples = t.shape[0]
    beta = 0.1 + 0.89 * jax.nn.sigmoid(beta_params)

    Dbc = caputo_derivative(c_data, t, beta)
    physics_pred = alpha * fnn_fixed_boundary(f_params, c_data)

    num_plots = min(num_plots, num_samples)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i in range(num_plots):
        t_hours = np.array(t[i, 1:, 0]) * max_hour  # skip node 0 (undefined)
        axes[i].plot(t_hours, np.array(Dbc[i, 1:, 0]), 'b-',
                     label=r'$D^\beta c$ (data)', linewidth=2)
        axes[i].plot(t_hours, np.array(physics_pred[i, 1:, 0]), 'r--',
                     label=r'$\alpha_i f(c)$ (model)', linewidth=2)
        axes[i].set_xlabel('Hours')
        axes[i].set_ylabel('Rate')
        axes[i].set_title(f'Tweet {i+1} (alpha={float(alpha[i,0,0]):.3f})')
        axes[i].legend()

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved physics fit to {save_path}")


def plot_reaction_function(f_params, save_path='reaction_function.png'):
    """Plot the learned reaction function f(c)."""
    c_vals = jnp.linspace(0, 1, 100).reshape(1, 100, 1)
    f_vals = fnn_fixed_boundary(f_params, c_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(c_vals[0, :, 0], f_vals[0, :, 0], 'b-', linewidth=2)
    plt.xlabel('Engagement c (normalized)')
    plt.ylabel('f(c)')
    plt.title('Learned Reaction Function f(c)')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reaction function to {save_path}")


def plot_alpha_head(alpha, N_log, v, abc, velocity_on,
                    save_path='alpha_head.png'):
    """
    Scatter the fitted per-post alpha against BOTH candidate drivers:
    early velocity (left) and followers (right).  With the velocity head
    on, the left panel should carry the structure and the right should be
    flat -- the experiment-17 signature.
    """
    a, b, c = abc
    al = np.asarray(alpha)[:, 0, 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(v, al, 'r.', markersize=4, alpha=0.5)
    if velocity_on:
        vg = np.linspace(min(0, v.min()), v.max() * 1.05, 300)
        med_n = float(np.median(N_log))
        axes[0].plot(vg, 1 / (1 + np.exp(-(a * med_n + b * vg + c))), 'b-',
                     lw=2, label=f'head at median followers\n'
                                 f'sigmoid({a:+.3f}·logN {b:+.3f}·v {c:+.3f})')
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel('early velocity v = log1p(E_h1) - log1p(E_h0)')
    axes[0].set_ylabel('alpha')
    axes[0].set_title('Viral potential vs early velocity')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.05, 1.05])

    axes[1].plot(N_log, al, 'r.', markersize=4, alpha=0.5)
    axes[1].set_xlabel('log(followers + 1)')
    axes[1].set_ylabel('alpha')
    axes[1].set_title('Viral potential vs followers')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved alpha head plot to {save_path}")


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run training."""
    cli = argparse.ArgumentParser()
    cli.add_argument('--velocity', choices=['on', 'off'], default='on',
                     help="'off' = followers-only ablation twin")
    cli.add_argument('--strict-growth', action='store_true',
                     help='train only on strictly increasing trajectories')
    cli.add_argument('--no-decline', action='store_true',
                     help='drop tweets with ANY decreasing hourly step '
                          '(unlikes/deletes); plateaus are kept -- weaker '
                          'than --strict-growth, which also drops plateaus')
    cli.add_argument('--peak-quartile', type=int, choices=[1, 2, 3, 4],
                     default=None,
                     help='train on one final-engagement quartile of the '
                          '(post-filter) set: 1 = smallest peaks .. 4 = top '
                          'quartile. Boundaries (computed on the filtered '
                          'set, saved to <outdir>/quartile_bounds.json) are '
                          'what the mixture router in the testing pipeline '
                          'must reuse.')
    cli.add_argument('--outdir', default='results',
                     help='output directory (relative to this folder); lets '
                          'the per-quartile runs coexist')
    cli.add_argument('--smooth', choices=['off', 'exp11', 'very'], default='off',
                     help="smoothness filter (ported from final code ieee/11): "
                          "'exp11' = max hourly step <= 30%% of window growth, "
                          ">=10%% of growth after hour 12, >=10 growing hours "
                          "(247/403 survive); 'very' = tightened to <= 15%%, "
                          ">= 14 growing hours (107/403 survive)")
    cli.add_argument('--steps', type=int, default=15000)
    args = cli.parse_args()

    SMOOTH_PRESETS = {
        'exp11': dict(max_step_share=0.30, min_late_share=0.10,
                      late_hour=12, min_growth_hours=10),
        'very': dict(max_step_share=0.15, min_late_share=0.10,
                     late_hour=12, min_growth_hours=14),
    }

    # === Configuration ===
    config = {
        'seed': 42,
        'num_steps': args.steps,
        'learning_rate': 1e-3,
        'print_every': 250,
        'ckpt_every': 2500,
        'f_width': 30,            # f(c) network width  (4 x 30: same
        'f_layers': 4,            #   architecture as validation + PoC)
        'velocity_on': args.velocity == 'on',
        # Dataset filters (see the plateau-ablation chain: 403 -> 343 -> 86
        # all left beta at the 0.1 bound, so neither filter is on by default;
        # --strict-growth reruns the 86-post strictly-increasing subset).
        'exclude_terminal_plateau_hours': None,
        'require_strictly_increasing': args.strict_growth,
    }

    # === Data: filtered 24h training split ===
    # 403 complete 24-node trajectories (hours 0-23, metric = total,
    # per-tweet max normalization, min peak engagement 25), built
    # deterministically from the jan31 scrape by "../data/build_dataset.py"
    # (raw CSVs not shipped -- see ../data/DATA_README.md for provenance).
    # N_log and each post's window peak ship inside the npz; velocities are
    # computed from those arrays (verified identical to the raw-snapshot
    # derivation).
    here = os.path.dirname(os.path.abspath(__file__))
    data_npz = os.path.normpath(os.path.join(here, '..', 'data', 'train.npz'))
    results_dir = os.path.join(here, args.outdir)
    os.makedirs(results_dir, exist_ok=True)

    d = np.load(data_npz, allow_pickle=True)
    t_np, c_np, N_log = d['t'], d['c'], d['N_log']
    post_ids = d['post_ids']
    peak = d['peak'].astype(float)
    print(f"Loaded filtered training split: {c_np.shape[0]} tweets, "
          f"{c_np.shape[1]} hourly nodes\n  ({data_npz})")

    print("Computing early velocities from the dataset arrays ...")
    v_np = compute_velocities(c_np, peak)
    print(f"  v = log1p(E_h1) - log1p(E_h0): min {v_np.min():.3f}, "
          f"median {np.median(v_np):.3f}, max {v_np.max():.3f}")

    # === Optional dataset filters ===
    if args.smooth != 'off':
        kw = SMOOTH_PRESETS[args.smooth]

        def _is_smooth(e):
            # ported verbatim from final code ieee/11_final_filter_smoothness/
            # load_tweets.py::_is_smooth (shares are scale-invariant, so the
            # max-normalized trajectory gives identical decisions to raw)
            growth = e[-1] - e[0]
            if growth <= 0:
                return False
            dd = np.diff(e)
            if float(dd.max()) / growth > kw['max_step_share']:
                return False
            if kw['late_hour'] < len(e):
                if (e[-1] - e[kw['late_hour']]) / growth < kw['min_late_share']:
                    return False
            return int((dd > 1e-9).sum()) >= kw['min_growth_hours']

        keep = np.array([_is_smooth(c_np[i, :, 0]) for i in range(len(c_np))])
        n_drop = int((~keep).sum())
        t_np, c_np, peak = t_np[keep], c_np[keep], peak[keep]
        N_log, post_ids, v_np = N_log[keep], post_ids[keep], v_np[keep]
        print(f"Smoothness filter ('{args.smooth}': step<={kw['max_step_share']:.0%} "
              f"of growth, >={kw['min_late_share']:.0%} late growth, "
              f">={kw['min_growth_hours']} growing hours): removed {n_drop}, "
              f"kept {c_np.shape[0]}")

    if args.no_decline:
        dcv = np.diff(c_np[:, :, 0], axis=1)
        keep = (dcv >= 0).all(axis=1)
        n_drop = int((~keep).sum())
        dropped_ids = [str(p) for p in post_ids[~keep]]
        t_np, c_np, peak = t_np[keep], c_np[keep], peak[keep]
        N_log, post_ids, v_np = N_log[keep], post_ids[keep], v_np[keep]
        print(f"No-decline filter (every hourly increment >= 0): removed "
              f"{n_drop} posts with engagement loss, kept {c_np.shape[0]}")
        with open(os.path.join(results_dir, 'no_decline_dropped_ids.txt'),
                  'w') as fh:
            fh.write("\n".join(dropped_ids) + "\n")

    if args.peak_quartile is not None:
        import json
        bounds = np.percentile(peak, [25.0, 50.0, 75.0])
        q_idx = np.searchsorted(bounds, peak, side='right')  # 0..3
        keep = q_idx == (args.peak_quartile - 1)
        n_drop = int((~keep).sum())
        with open(os.path.join(results_dir, 'quartile_bounds.json'), 'w') as fh:
            json.dump({"bounds": [float(b) for b in bounds],
                       "quartile": args.peak_quartile,
                       "n_in_quartile": int(keep.sum()),
                       "n_total_filtered_set": int(len(peak)),
                       "peak_range_in_quartile": [float(peak[keep].min()),
                                                  float(peak[keep].max())]},
                      fh, indent=2)
        t_np, c_np, peak = t_np[keep], c_np[keep], peak[keep]
        N_log, post_ids, v_np = N_log[keep], post_ids[keep], v_np[keep]
        print(f"Peak-quartile filter (Q{args.peak_quartile}, bounds "
              f"{[round(float(b), 1) for b in bounds]}): removed {n_drop}, "
              f"kept {c_np.shape[0]} with final engagement in "
              f"[{peak.min():.0f}, {peak.max():.0f}]")

    if config.get('require_strictly_increasing'):
        dcv = np.diff(c_np[:, :, 0], axis=1)
        keep = (dcv > 0).all(axis=1)
        n_drop = int((~keep).sum())
        t_np, c_np, peak = t_np[keep], c_np[keep], peak[keep]
        N_log, post_ids, v_np = N_log[keep], post_ids[keep], v_np[keep]
        print(f"Strict-growth filter (every hourly increment > 0): removed "
              f"{n_drop} posts, kept {c_np.shape[0]}")

    K = config.get('exclude_terminal_plateau_hours')
    if K is not None:
        dcv = np.diff(c_np[:, :, 0], axis=1)
        n_steps = dcv.shape[1]
        has_rise = (dcv > 0).any(axis=1)
        last_inc = np.where(
            has_rise,
            (n_steps - 1) - np.argmax((dcv > 0)[:, ::-1], axis=1),
            -1)
        term_plateau = (n_steps - 1) - last_inc
        keep = has_rise & (term_plateau < K)
        n_drop = int((~keep).sum())
        t_np, c_np = t_np[keep], c_np[keep]
        N_log, post_ids, v_np = N_log[keep], post_ids[keep], v_np[keep]
        print(f"Early-plateau filter (terminal flat >= {K}h): removed "
              f"{n_drop} posts, kept {c_np.shape[0]}")

    # Keep a copy of exactly what was trained on next to the outputs
    np.savez_compressed(os.path.join(results_dir, 'loaded_data_cache.npz'),
                        t=t_np, c=c_np, N_log=N_log, post_ids=post_ids,
                        v=v_np, velocity_on=int(config['velocity_on']))

    # Convert to JAX arrays (float64)
    t = jnp.array(t_np, dtype=jnp.float64)
    c_data = jnp.array(c_np, dtype=jnp.float64)

    # === Verify numerics before the long run ===
    self_test(c_data)

    # === Train Model ===
    params, history = train_model(c_data, N_log, v_np, config,
                                  results_dir=results_dir)
    f_params, alpha_params, beta_params = params

    # === Save Results ===
    # NOTE: saved as a 4-tuple with a None placeholder in the (legacy)
    # c_params slot so that symbolic_regression.py, which unpacks
    # (c_params, f_params, alpha_params, beta_params) and uses only f_params,
    # works unchanged.  Slot 3 now holds the (a, b, c) head vector.
    with open(os.path.join(results_dir, 'params.pkl'), 'wb') as f:
        pickle.dump((None, f_params, alpha_params, beta_params), f)
    print("Saved parameters to results/params.pkl")

    with open(os.path.join(results_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print("Saved history to results/history.pkl")

    np.save(os.path.join(results_dir, 'N_log.npy'), N_log)
    np.save(os.path.join(results_dir, 'velocities.npy'), v_np)
    print("Saved follower/velocity covariates to results/")

    # === Generate Plots ===
    num_samples = c_data.shape[0]
    N_log_jax = jnp.array(N_log, dtype=jnp.float64).reshape(num_samples, 1, 1)
    v_jax = jnp.array(v_np, dtype=jnp.float64).reshape(num_samples, 1, 1)
    final_alpha = alpha_head(alpha_params, N_log_jax, v_jax,
                             config['velocity_on'])
    abc = tuple(float(x) for x in np.asarray(alpha_params))

    plot_training_history(history,
                          save_path=os.path.join(results_dir, 'training_history.png'))
    plot_physics_fit(t, c_data, f_params, beta_params, final_alpha,
                     num_plots=6,
                     save_path=os.path.join(results_dir, 'physics_fit.png'),
                     max_hour=c_data.shape[1] - 1)
    plot_reaction_function(f_params,
                           save_path=os.path.join(results_dir, 'reaction_function.png'))
    plot_alpha_head(final_alpha, N_log, v_np, abc, config['velocity_on'],
                    save_path=os.path.join(results_dir, 'alpha_head.png'))

    # === Print Final Summary ===
    final_beta = float(0.1 + 0.89 * jax.nn.sigmoid(beta_params))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY (whitened-full L1, velocity "
          f"{'ON' if config['velocity_on'] else 'OFF'})")
    print("=" * 60)
    print(f"Fractional order beta: {final_beta:.4f}")
    print(f"Alpha head (a, b, c): ({abc[0]:+.4f}, {abc[1]:+.4f}, {abc[2]:+.4f})")
    print(f"  alpha = sigmoid(a*log(N+1) "
          f"{'+ b*v ' if config['velocity_on'] else ''}+ c)")
    print(f"  alpha range: [{float(final_alpha.min()):.4f}, "
          f"{float(final_alpha.max()):.4f}], mean {float(final_alpha.mean()):.4f}")
    print(f"Implied observation noise (sqrt of final loss): "
          f"{history['sigma_implied'][-1]:.5f}")
    print()
    print("Next steps:")
    print("  1. Check results/training_history.png for loss/parameter curves")
    print("  2. Check results/alpha_head.png for the velocity-vs-followers split")
    print("  3. Check results/reaction_function.png for learned f(c)")
    print("  4. Run symbolic regression to get equation for f(c)")
    print("=" * 60)


if __name__ == '__main__':
    main()
