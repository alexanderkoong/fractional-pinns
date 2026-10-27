#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter: load a trained PINN checkpoint written by
`jmlmc code/baseline/train_virality.py` and expose it through the same
`predict(c_seed, k, N_log_i, v_i=0.0)` duck-typed interface as every model
in baselines.py, so it can be scored by eval_known_max.py / eval_leak_free.py
and driven by TwoStageForecaster exactly like the classical growth curves.

TWO checkpoint generations, auto-detected from params.pkl's shape
--------------------------------------------------------------------
Both save as a 4-tuple `(None, f_params, <alpha-param>, beta_params)` with a
placeholder None in the legacy c_params slot (so symbolic_regression.py,
which unpacks 4 and uses only f_params, works unchanged either way) and
RAW/unconstrained beta_params (`beta = 0.1 + 0.89*sigmoid(beta_params)` in
both generations). They differ in slot [2]:

  PLAIN (scalar gamma_params -- the original trainer):
      alpha = tanh(softplus(gamma_params) * log(followers + 1))

  VELOCITY HEAD (length-3 alpha_params = (a, b, c) -- current trainer,
  `alpha_head()` in train_virality.py, ported from
  `final code ieee/17_velocity_alpha`):
      alpha = sigmoid(a * log(followers+1) + [b * v if velocity_on] + c)
      v     = log1p(raw engagement at hour 1) - log1p(raw engagement at hour 0)
  `velocity_on` is NOT stored in params.pkl (the OFF ablation twin just
  never gets gradient into `b`, which sits at its unchanged init value) --
  callers must say which via the `velocity_on` constructor arg, matching
  which `--velocity on|off` run actually produced the checkpoint.

f(c) = exp(NN(c)) * c * (1 - c) in both generations, reconstructed via a
dense interpolation grid (architecture-agnostic). Rollout uses
fractional_solvers.fode_solve_l1 -- the corrected L1 stepper.
"""

import pickle

import numpy as np

from fractional_solvers import fode_solve_l1


def _stable_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _stable_softplus(x):
    return np.log1p(np.exp(-abs(x))) + max(x, 0.0)


def load_checkpoint(params_path):
    """Return (kind, beta, alpha_fn, f_scalar) recovered from a
    train_virality.py params.pkl.

    kind: "plain" or "velocity_head"
    alpha_fn(N_log_i, v_i, velocity_on) -> float in (0, 1)
    f_scalar(c) -> float, vectorized-safe via np.interp
    """
    with open(params_path, "rb") as fh:
        params = pickle.load(fh)
    try:
        _c_params, f_params, alpha_params, beta_params = params
    except Exception as exc:
        raise RuntimeError(
            f"Could not unpack {params_path} as "
            "(c_params, f_params, alpha_params, beta_params) -- "
            "checkpoint format has changed; update pinn_model.load_checkpoint."
        ) from exc

    beta = 0.1 + 0.89 * _stable_sigmoid(float(np.asarray(beta_params)))

    alpha_arr = np.asarray(alpha_params)
    if alpha_arr.shape == ():
        kind = "plain"
        gamma = _stable_softplus(float(alpha_arr))

        def alpha_fn(N_log_i, v_i, velocity_on, _gamma=gamma):
            return float(np.tanh(_gamma * float(N_log_i)))
    elif alpha_arr.shape == (3,):
        kind = "velocity_head"
        a, b, c = (float(x) for x in alpha_arr)

        def alpha_fn(N_log_i, v_i, velocity_on, _a=a, _b=b, _c=c):
            z = _a * float(N_log_i) + _c
            if velocity_on:
                z += _b * float(v_i)
            return float(_stable_sigmoid(z))
    else:
        raise RuntimeError(
            f"Unrecognized alpha-param shape {alpha_arr.shape} in {params_path} "
            "-- expected scalar (plain gamma) or length-3 (velocity head); "
            "checkpoint format has changed further, update pinn_model.py."
        )

    # f(c) = exp(NN(c)) * c * (1-c), evaluated on a dense grid and interpolated
    # (architecture-agnostic: loops over however many layers f_params has).
    grid = np.linspace(0.0, 1.0, 4001)
    x = grid.reshape(1, -1, 1).astype(np.float64)
    for i in range(len(f_params) - 1):
        W, b_ = f_params[i]
        x = np.tanh(x @ np.asarray(W) + np.asarray(b_))
    W, b_ = f_params[-1]
    fv = (np.exp(x @ np.asarray(W) + np.asarray(b_))
          * (grid.reshape(1, -1, 1) * (1 - grid.reshape(1, -1, 1))))[0, :, 0]

    def f_scalar(c, _grid=grid, _fv=fv):
        return float(np.interp(c, _grid, _fv))

    return kind, beta, alpha_fn, f_scalar


class PINNCheckpoint:
    """Baseline-interface wrapper around one trained checkpoint.

    Matches baselines.Baseline's duck type (name, n_params, fitted,
    train_mean_mse, fit, predict, predict_fn) plus the optional 4th
    `v_i` argument every model in this pipeline now accepts, so it drops
    directly into the same model lists as the classical growth curves --
    including as a TwoStageForecaster shape_model in eval_leak_free.py.

    velocity_on: only meaningful for "velocity_head"-kind checkpoints (see
    load_checkpoint's docstring for why it can't be auto-detected from the
    pickle alone); ignored for "plain"-kind checkpoints.
    """
    n_params = 2

    def __init__(self, label, params_path, num_timesteps, h_norm, velocity_on=True):
        self.label = label
        self.params_path = params_path
        self.num_timesteps = num_timesteps
        self.h_norm = h_norm
        self.kind, self.beta, self._alpha_fn, self.f_scalar = load_checkpoint(params_path)
        self.velocity_on = velocity_on if self.kind == "velocity_head" else False
        suffix = "" if self.kind == "plain" else (" +velocity" if self.velocity_on else " followers-only")
        self.name = f"PINN ({label}{suffix})"
        self.fitted = {
            "kind": self.kind, "beta": self.beta,
            "velocity_on": self.velocity_on, "params_path": params_path,
        }
        self.train_mean_mse = float("nan")

    def fit(self, C, N_log):
        """No-op: parameters come from the checkpoint, not refit here. Kept
        so this class can sit in the same all_baselines()-style list and be
        `.fit(C, N_log)`-ed uniformly without special-casing."""
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        alpha = self._alpha_fn(N_log_i, v_i, self.velocity_on)
        n = self.num_timesteps - k
        return fode_solve_l1(self.f_scalar, alpha, self.beta, float(c_seed),
                             n, self.h_norm, clip01=True)

    def predict_fn(self):
        return lambda c_seed, k, N_log_i, v_i=0.0: self.predict(c_seed, k, N_log_i, v_i)

    def needs_velocity(self):
        return self.velocity_on
