#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEACHER-STUDENT SEGMENTED PINNs -- four independent fractional models over
6-hour windows, teacher-forced at the boundaries.

The idea (user-specified): instead of one law for the whole 24-hour window,
train a COMPLETELY SEPARATE PINN (fresh weights: its own f_k, alpha-head_k,
beta_k) on each 6-hour segment,

    segment 0: hours  0-3    (nodes  0..3  -- the SHORT burst window)
    segment 1: hours  3-9    (nodes  3..9)
    segment 2: hours  9-16   (nodes  9..16)
    segment 3: hours 16-23   (nodes 16..23)

with each segment's initial condition anchored to the OBSERVED value at the
segment's first node (teacher forcing).  At forecast time the stitched model
picks the PINN by wall-clock hour and chains each segment's endpoint into the
next segment's initial condition (student rollout) -- see stitched_model.py.

Modeling choices (stated explicitly):
  * Each segment is its own fractional initial-value problem: the Caputo
    memory RESTARTS at the segment boundary.  This matches the forward
    solve exactly (fode_solve_l1 restarted from a seed), at the price of
    discarding cross-segment memory -- the same convention the evaluation
    harness already uses for its restart-hour sweeps.
  * The PHYSICAL hourly grid is kept: h = 1/23 (the full window's
    normalization) in every segment, so beta_k and alpha_k across segments
    live on the same time scale and the stitched forward solve is
    dimensionally consistent.  The loss machinery therefore takes h
    explicitly instead of inferring 1/(N_seg-1) from the segment length.
  * Loss, alpha head, f architecture, and the no-decline data filter are
    identical to the current single-PINN trainer
    (`../baseline/train_virality.py`, whitened-full L1 + velocity head):
    the ONLY new ingredient is time segmentation, so any performance
    difference is attributable to it.

Data: ../data/train.npz, filtered
to the 348 no-decline tweets (same as the current best single checkpoint).
Velocity covariate v = log1p(raw@h1) - log1p(raw@h0) computed from the npz's
c*peak (verified bit-identical to the trainer's snapshot-derived values by
the final-testing-pipeline README).

Outputs -> results/segment_k/params.pkl (k = 0..3, each a standard 4-tuple
checkpoint loadable by pinn_model.py), results/train_segments_report.txt
"""

import os
import pickle
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit, value_and_grad
from jax.example_libraries import optimizers
from jax.scipy.special import gammaln

jax.config.update("jax_enable_x64", True)

JITTER = 1e-10
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

NUM_NODES = 24
H_NORM = 1.0 / (NUM_NODES - 1)          # physical hourly step on [0,1]
SEG_BOUNDS = [(0, 3), (3, 9), (9, 16), (16, 23)]   # inclusive node ranges
NUM_STEPS = 15000
LR = 1e-3
F_WIDTH, F_LAYERS = 30, 4
SEED = 42


# ---------------------------------------------------------------- nets/heads
def gamma_fn(x):
    return jnp.exp(gammaln(x))


def safe_pow(x, p):
    x = jnp.asarray(x, dtype=jnp.float64)
    xs = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, xs ** p, 0.0)


def random_layer_params(ind, outd, num_nn, key):
    scale = jnp.sqrt(2 / (ind + outd))
    return (random.normal(key, (num_nn, ind, outd)) * scale,
            jnp.zeros((num_nn, 1, outd)))


def init_network_params(ind, outd, width, num_nn, key, num_layers):
    keys = random.split(key, num_layers)
    layers = [random_layer_params(ind, width, num_nn, keys[0])]
    for i in range(1, num_layers - 1):
        layers.append(random_layer_params(width, width, num_nn, keys[i]))
    layers.append(random_layer_params(width, outd, num_nn, keys[-1]))
    return layers


def fnn_fixed_boundary(params, x):
    bt = x * (1 - x)
    for i in range(len(params) - 1):
        W, b = params[i]
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return jnp.exp(x @ W + b) * bt


def f_prime_nn(f_params, c):
    _, tangent = jax.jvp(lambda x: fnn_fixed_boundary(f_params, x),
                         (c,), (jnp.ones_like(c),))
    return tangent


def alpha_head(alpha_params, N_log, v):
    a, b, cc = alpha_params[0], alpha_params[1], alpha_params[2]
    return jax.nn.sigmoid(a * N_log + b * v + cc)


# ------------------------------------------------- whitened loss, explicit h
def l1_expanded_matrix(num_timesteps, beta, h):
    """A(beta) (N-1, N) with an EXPLICIT physical step h (the segment keeps
    the full window's hourly h = 1/23 rather than rescaling to [0,1])."""
    N = num_timesteps
    j = jnp.arange(N, dtype=jnp.float64)
    b = safe_pow(j + 1.0, 1.0 - beta) - safe_pow(j, 1.0 - beta)
    coef = h ** (-beta) / gamma_fn(2.0 - beta)
    n = jnp.arange(1, N, dtype=jnp.float64)[:, None]
    k = jnp.arange(N, dtype=jnp.float64)[None, :]
    lag = jnp.clip(n - k, 0, N - 1).astype(int)
    A = jnp.where((k >= 1) & (k <= n - 1), b[lag] - b[lag - 1], 0.0)
    A = jnp.where(k == n, b[0], A)
    A = jnp.where(k == 0, -b[jnp.clip(n.astype(int) - 1, 0, N - 1)][:, 0:1]
                  * jnp.ones_like(A), A)
    return coef * A


def loss_whitened_full(c, beta, fc, fprime, alpha, h):
    S, N, _ = c.shape
    A = l1_expanded_matrix(N, beta, h)
    a = alpha[:, 0, 0]
    r = jnp.einsum("mk,skf->smf", A, c)[:, :, 0] - a[:, None] * fc[:, 1:, 0]
    eye_t = jnp.eye(N)[1:, :]
    J = A[None, :, :] - (a[:, None, None] * fprime[:, 1:, 0][:, :, None]) \
        * eye_t[None, :, :]
    JJt = jnp.einsum("snk,smk->snm", J, J) \
        + JITTER * jnp.eye(N - 1)[None, :, :]
    x = jnp.linalg.solve(JJt, r[:, :, None])[:, :, 0]
    return jnp.mean(r * x)


def train_segment(c_seg, N_log, v, seg_idx, report):
    """One fresh PINN on one segment; returns (f_params, head, beta_raw)."""
    S, N, _ = c_seg.shape
    key = random.PRNGKey(SEED + seg_idx)          # fresh weights per segment
    f_params = init_network_params(1, 1, F_WIDTH, 1, key, F_LAYERS)
    head = jnp.array([0.1, 1.0, 0.0])
    beta_raw = jnp.array(0.0)                     # beta ~ 0.545 init

    N_log_j = jnp.array(N_log, dtype=jnp.float64).reshape(S, 1, 1)
    v_j = jnp.array(v, dtype=jnp.float64).reshape(S, 1, 1)
    c_j = jnp.array(c_seg, dtype=jnp.float64)

    def loss_fn(f_p, hd, b_raw):
        beta = 0.1 + 0.89 * jax.nn.sigmoid(b_raw)
        alpha = alpha_head(hd, N_log_j, v_j)
        fc = fnn_fixed_boundary(f_p, c_j)
        fp = f_prime_nn(f_p, c_j)
        return loss_whitened_full(c_j, beta, fc, fp, alpha, H_NORM)

    opt_init, opt_update, get_params = optimizers.adam(LR)
    opt_state = opt_init((f_params, head, beta_raw))

    @jit
    def step(i, s):
        p = get_params(s)
        l, g = value_and_grad(loss_fn, argnums=(0, 1, 2))(*p)
        return l, opt_update(i, g, s)

    t0 = time()
    for i in range(NUM_STEPS + 1):
        l, opt_state = step(i, opt_state)
        if i % 5000 == 0:
            _, hd, br = get_params(opt_state)
            beta = float(0.1 + 0.89 * jax.nn.sigmoid(br))
            print(f"  seg{seg_idx} step {i:5d}  loss {float(l):.4e}  "
                  f"beta {beta:.4f}", flush=True)

    f_p, hd, br = get_params(opt_state)
    beta = float(0.1 + 0.89 * jax.nn.sigmoid(br))
    abc = tuple(float(x) for x in np.asarray(hd))
    alpha = alpha_head(hd, N_log_j, v_j)
    sig = float(np.sqrt(max(float(l), 0.0)))
    line = (f"segment {seg_idx} (nodes {SEG_BOUNDS[seg_idx][0]}-"
            f"{SEG_BOUNDS[seg_idx][1]}): beta={beta:.4f}  "
            f"head(a,b,c)=({abc[0]:+.3f},{abc[1]:+.3f},{abc[2]:+.3f})  "
            f"alpha mean {float(alpha.mean()):.3f}  implied sigma {sig:.5f}  "
            f"[{time()-t0:.0f}s]")
    print(line, flush=True)
    report.append(line)
    return f_p, hd, br


def main():
    os.makedirs(RESULTS, exist_ok=True)
    report = ["TEACHER-STUDENT SEGMENTED TRAINING (whitened-full, velocity "
              "head, no-decline 348)", "=" * 74]

    data_npz = os.path.normpath(os.path.join(HERE, "..", "data", "train.npz"))
    d = np.load(data_npz, allow_pickle=True)
    c_np, N_log, peak = d["c"], d["N_log"], d["peak"].astype(float)
    post_ids = d["post_ids"]

    keep = (np.diff(c_np[:, :, 0], axis=1) >= 0.0).all(axis=1)
    c_np, N_log, peak = c_np[keep], N_log[keep], peak[keep]
    post_ids = post_ids[keep]
    report.append(f"no-decline filter: kept {c_np.shape[0]} of {len(keep)} "
                  f"tweets")

    C_raw = c_np[:, :, 0] * peak[:, None]
    v = np.log1p(C_raw[:, 1]) - np.log1p(C_raw[:, 0])
    report.append(f"velocity v: min {v.min():.3f} median {np.median(v):.3f} "
                  f"max {v.max():.3f}")
    report.append("")

    manifest = {"seg_bounds": SEG_BOUNDS, "h_norm": H_NORM,
                "n_tweets": int(c_np.shape[0]), "velocity_on": 1}
    for k, (lo, hi) in enumerate(SEG_BOUNDS):
        print(f"=== training segment {k}: nodes {lo}..{hi} "
              f"({hi - lo + 1} nodes) ===", flush=True)
        c_seg = c_np[:, lo:hi + 1, :]
        f_p, hd, br = train_segment(c_seg, N_log, v, k, report)
        seg_dir = os.path.join(RESULTS, f"segment_{k}")
        os.makedirs(seg_dir, exist_ok=True)
        with open(os.path.join(seg_dir, "params.pkl"), "wb") as fh:
            pickle.dump((None,
                         [(np.array(W), np.array(b)) for W, b in f_p],
                         np.array(hd), np.array(br)), fh)

    np.savez_compressed(os.path.join(RESULTS, "train_cache.npz"),
                        c=c_np, N_log=N_log, v=v, peak=peak,
                        post_ids=post_ids)
    with open(os.path.join(RESULTS, "manifest.pkl"), "wb") as fh:
        pickle.dump(manifest, fh)

    text = "\n".join(report)
    with open(os.path.join(RESULTS, "train_segments_report.txt"), "w") as fh:
        fh.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
