#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CLASSICAL (NON-PINN) BASELINE MODELS
=============================================================================
Adapted from "beat logistic updated final filter/baselines.py" for the
CURRENT dataset (jmlmc code/24hr model verification/real_world_data_24h_T25):
24 hourly nodes, hours 0..23 (h = 1/23) -- NOT the old pipeline's 25-node
0..24 window. Every model here predicts a normalized engagement trajectory
and is scored through the same harness (eval_known_max.py / eval_leak_free.py)
as the current PINN checkpoints (pinn_model.py), so the comparison is
apples-to-apples.

Removed vs. the original: the PINNReference class (which hardcoded stale
thesis-era beta/gamma/f constants fit through a since-fixed buggy solver --
see the original file's warning comment). The current model is scored via
pinn_model.PINNCheckpoint instead, loaded from an actual trained params.pkl.

Uniform interface
-----------------
Each model is a class exposing:

    fit(C, N_log)                       # fit global params on the dense TRAIN set
    predict(c_seed, k, N_log_i)         # -> np.ndarray length (T - k), pred[0]=c_seed
    predict_fn()                        # -> the predict closure for the scorer

plus reporting attributes: name, n_params, fitted (dict), train_mean_mse.

Conventions
-----------
* Growth-curve models (logistic, Gompertz, Bass) use NORMALIZED time
  t in [0,1] over T=24 hourly points (h = 1/23), matching the PINN. Their
  ODEs are autonomous, so a restart at hour k is exact: re-anchor at
  (t_k, c_k) and evolve for elapsed time (hour - k)/23.
* Hawkes models use HOURLY INTEGER time (lags 1..23), the natural unit for a
  binned point process; the branching ratio is then directly interpretable.
* Every model clips the cumulative curve to [0,1] each step.

Fitting objective
-----------------
All models are fit by minimizing the MEAN forward-solve trajectory MSE on the
dense TRAIN set (a proper scoring objective), NOT the %<threshold metric they
are later judged on.
=============================================================================
"""

import numpy as np
from scipy.optimize import minimize, nnls

# The bounded power-law kernel can transiently overflow during the optimizer's
# excursions; the results are finite and the fit is unaffected, so silence the
# harmless over/invalid warnings rather than spam stderr.
np.seterr(over="ignore", invalid="ignore")

# Current dataset: real_world_data_24h_T25 -- 24 nodes, hours 0..23.
NUM_TIMESTEPS = 24
H_NORM = 1.0 / (NUM_TIMESTEPS - 1)

T = NUM_TIMESTEPS                 # 24
T_NORM = np.linspace(0.0, 1.0, T) # normalized time grid
C_FLOOR = 1e-3                    # seed floor (matches harness)
_CLIP_LO, _CLIP_HI = 1e-4, 1.0 - 1e-4   # for closed-form A = (1-c)/c


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _future_mse(C_pred, C):
    """Mean over tweets of MSE over hours 1..T-1 (matches the scorer, which
    grades pred[1:] vs actual[1:])."""
    return float(np.mean((C_pred[:, 1:] - C[:, 1:]) ** 2))


def _clip_seed(c):
    return min(max(float(c), _CLIP_LO), _CLIP_HI)


# ===========================================================================
# BASE CLASS
# ===========================================================================
class Baseline:
    name = "baseline"
    n_params = 0

    def __init__(self):
        self.fitted = {}
        self.train_mean_mse = float("nan")

    def fit(self, C, N_log):
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        raise NotImplementedError

    def predict_fn(self):
        return lambda c_seed, k, N_log_i, v_i=0.0: self.predict(c_seed, k, N_log_i, v_i)

    # convenience: full-batch forward-solve from c(0) for fitting/plotting
    def _batch_from_c0(self, C, N_log):
        out = np.empty_like(C)
        for i in range(C.shape[0]):
            out[i] = self.predict(max(C[i, 0], C_FLOOR), 0, float(N_log[i]))
        return out

    def _record_train_mse(self, C, N_log):
        self.train_mean_mse = _future_mse(self._batch_from_c0(C, N_log), C)


# ===========================================================================
# NAIVE / FLOOR ANCHORS
# ===========================================================================
class Persistence(Baseline):
    """c_pred(t) = c_seed for all t. The 'nothing changes' anchor."""
    name = "Persistence (c=c0)"
    n_params = 0

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        return np.full(T - k, _clip_seed(c_seed))

    def fit(self, C, N_log):
        self._record_train_mse(C, N_log)
        return self


class LinearToMax(Baseline):
    """Straight line from c_seed at hour k to 1.0 at hour T-1."""
    name = "Linear c0->1"
    n_params = 0

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        n = T - k
        if n == 1:
            return np.array([_clip_seed(c_seed)])
        return np.linspace(_clip_seed(c_seed), 1.0, n)

    def fit(self, C, N_log):
        self._record_train_mse(C, N_log)
        return self


class Climatology(Baseline):
    """Unconditional mean TRAIN trajectory -- the 'skill floor'. After the
    seed point it simply follows the average curve."""
    name = "Climatology (mean curve)"
    n_params = T   # stores the T-point mean curve (nonparametric)

    def fit(self, C, N_log):
        self.mean_curve = C.mean(axis=0)            # (T,)
        self.fitted = {"mean_curve_last": float(self.mean_curve[-1])}
        self._record_train_mse(C, N_log)
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        out = self.mean_curve[k:].copy()
        out[0] = _clip_seed(c_seed)                 # pred[0] = seed by convention
        return out


# ===========================================================================
# LOGISTIC GROWTH   dc/dt = r c (1 - c)     (advisor's required minimum)
# ===========================================================================
def _logistic_traj(c_seed, r, n):
    """Re-anchored logistic from c_seed over n points (elapsed normalized
    time 0 .. (n-1)/(T-1))."""
    c0 = _clip_seed(c_seed)
    trel = np.arange(n) / (T - 1)
    A = (1.0 - c0) / c0
    return 1.0 / (1.0 + A * np.exp(-r * trel))


class LogisticGlobal(Baseline):
    name = "Logistic (global r)"
    n_params = 1

    def fit(self, C, N_log):
        c0 = np.clip(C[:, 0], _CLIP_LO, _CLIP_HI)
        A = (1.0 - c0) / c0                          # (N,)
        def obj(log_r):
            r = np.exp(log_r[0])
            pred = 1.0 / (1.0 + A[:, None] * np.exp(-r * T_NORM[None, :]))
            return _future_mse(pred, C)
        grid = [(_g, obj([np.log(_g)])) for _g in np.linspace(2, 40, 60)]
        r0 = min(grid, key=lambda x: x[1])[0]
        res = minimize(obj, [np.log(r0)], method="Nelder-Mead")
        self.r = float(np.exp(res.x[0]))
        self.fitted = {"r": self.r}
        self._record_train_mse(C, N_log)
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        return _logistic_traj(c_seed, self.r, T - k)


class LogisticFollowerScaled(Baseline):
    """r_i = r0 * tanh(gamma * log(N_i + 1)) -- gives the logistic the SAME
    follower feature the PINN uses for alpha."""
    name = "Logistic (follower-scaled r)"
    n_params = 2

    def fit(self, C, N_log):
        c0 = np.clip(C[:, 0], _CLIP_LO, _CLIP_HI)
        A = (1.0 - c0) / c0
        Nl = np.asarray(N_log, float)
        def obj(theta):
            r0 = np.exp(theta[0]); g = np.exp(theta[1])
            r = r0 * np.tanh(g * Nl)                 # (N,)
            pred = 1.0 / (1.0 + A[:, None] * np.exp(-r[:, None] * T_NORM[None, :]))
            return _future_mse(pred, C)
        best = None
        for r0 in np.linspace(5, 45, 12):
            for g in np.linspace(0.05, 1.0, 12):
                v = obj([np.log(r0), np.log(g)])
                if best is None or v < best[0]:
                    best = (v, r0, g)
        res = minimize(obj, [np.log(best[1]), np.log(best[2])], method="Nelder-Mead")
        self.r0 = float(np.exp(res.x[0])); self.gamma = float(np.exp(res.x[1]))
        self.fitted = {"r0": self.r0, "gamma": self.gamma}
        self._record_train_mse(C, N_log)
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        r = self.r0 * np.tanh(self.gamma * float(N_log_i))
        return _logistic_traj(c_seed, r, T - k)


# ===========================================================================
# GOMPERTZ   dc/dt = b c ln(1/c)     ->   c(t) = exp(ln(c0) e^{-b t})
# ===========================================================================
def _gompertz_traj(c_seed, b, n):
    c0 = _clip_seed(c_seed)
    trel = np.arange(n) / (T - 1)
    return np.exp(np.log(c0) * np.exp(-b * trel))


class GompertzGlobal(Baseline):
    name = "Gompertz (global b)"
    n_params = 1

    def fit(self, C, N_log):
        c0 = np.clip(C[:, 0], _CLIP_LO, _CLIP_HI)
        logc0 = np.log(c0)
        def obj(log_b):
            b = np.exp(log_b[0])
            pred = np.exp(logc0[:, None] * np.exp(-b * T_NORM[None, :]))
            return _future_mse(pred, C)
        grid = [(_g, obj([np.log(_g)])) for _g in np.linspace(1, 25, 60)]
        b0 = min(grid, key=lambda x: x[1])[0]
        res = minimize(obj, [np.log(b0)], method="Nelder-Mead")
        self.b = float(np.exp(res.x[0]))
        self.fitted = {"b": self.b}
        self._record_train_mse(C, N_log)
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        return _gompertz_traj(c_seed, self.b, T - k)


class GompertzFollowerScaled(Baseline):
    name = "Gompertz (follower-scaled b)"
    n_params = 2

    def fit(self, C, N_log):
        c0 = np.clip(C[:, 0], _CLIP_LO, _CLIP_HI)
        logc0 = np.log(c0); Nl = np.asarray(N_log, float)
        def obj(theta):
            b0 = np.exp(theta[0]); g = np.exp(theta[1])
            b = b0 * np.tanh(g * Nl)
            pred = np.exp(logc0[:, None] * np.exp(-b[:, None] * T_NORM[None, :]))
            return _future_mse(pred, C)
        best = None
        for b0 in np.linspace(3, 30, 12):
            for g in np.linspace(0.05, 1.0, 12):
                v = obj([np.log(b0), np.log(g)])
                if best is None or v < best[0]:
                    best = (v, b0, g)
        res = minimize(obj, [np.log(best[1]), np.log(best[2])], method="Nelder-Mead")
        self.b0 = float(np.exp(res.x[0])); self.gamma = float(np.exp(res.x[1]))
        self.fitted = {"b0": self.b0, "gamma": self.gamma}
        self._record_train_mse(C, N_log)
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        b = self.b0 * np.tanh(self.gamma * float(N_log_i))
        return _gompertz_traj(c_seed, b, T - k)


# ===========================================================================
# BASS DIFFUSION   dc/dt = (p + q c)(1 - c)   (classic adoption/virality model)
# integrated numerically (RK4); autonomous so restart = integrate from c_seed
# ===========================================================================
def _bass_traj(c_seed, p, q, n, dt=H_NORM):
    c = np.empty(n)
    x = _clip_seed(c_seed)
    c[0] = x
    for i in range(1, n):
        def rhs(y):
            return (p + q * y) * (1.0 - y)
        k1 = rhs(x); k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2); k4 = rhs(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x = min(max(x, 0.0), 1.0)
        c[i] = x
    return c


def _bass_batch(c0_vec, p, q, dt=H_NORM):
    N = len(c0_vec)
    C_pred = np.empty((N, T))
    x = np.clip(c0_vec, _CLIP_LO, _CLIP_HI).astype(float)
    C_pred[:, 0] = x
    for i in range(1, T):
        def rhs(y):
            return (p + q * y) * (1.0 - y)
        k1 = rhs(x); k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2); k4 = rhs(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x = np.clip(x, 0.0, 1.0)
        C_pred[:, i] = x
    return C_pred


class BassDiffusion(Baseline):
    name = "Bass diffusion (p,q)"
    n_params = 2

    def fit(self, C, N_log):
        c0 = C[:, 0]
        def obj(theta):
            p = np.exp(theta[0]); q = np.exp(theta[1])
            return _future_mse(_bass_batch(c0, p, q), C)
        best = None
        for p in np.geomspace(0.05, 5.0, 10):
            for q in np.geomspace(1.0, 40.0, 10):
                v = obj([np.log(p), np.log(q)])
                if best is None or v < best[0]:
                    best = (v, p, q)
        res = minimize(obj, [np.log(best[1]), np.log(best[2])], method="Nelder-Mead")
        self.p = float(np.exp(res.x[0])); self.q = float(np.exp(res.x[1]))
        self.fitted = {"p": self.p, "q": self.q}
        self._record_train_mse(C, N_log)
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        return _bass_traj(c_seed, self.p, self.q, T - k)


# ===========================================================================
# HAWKES PROCESS   (advisor's required baseline)
# ---------------------------------------------------------------------------
# Data is aggregate hourly CUMULATIVE engagement, not event timestamps, so we
# treat hourly increments dc(n)=c(n)-c(n-1) as binned event counts and run the
# mean-field (expectation) intensity recursion in hourly time:
#
#     lambda_n = mu + sum_{m<n} dc_pred(m) * g(n - m)        (strict m<n, lag>=1)
#     dc_pred(n) = lambda_n                                  (h = 1 hour)
#     c_pred(n)  = clip(c_pred(n-1) + dc_pred(n), 0, 1)
#     realized dc_pred(n) = c_pred(n) - c_pred(n-1)          (fed back -> saturation)
#
# The branching ratio is the discrete sum sum_{lag>=1} g(lag); a free fit drives
# it toward the critical boundary, so we clamp eta < 0.999.
# ===========================================================================
def _hawkes_sim(c_seed, n_steps, gvals, mu):
    """Single-trajectory mean-field Hawkes from a cumulative seed.
    gvals[lag] = g(lag) for lag = 1..n_steps-1 (gvals[0] unused)."""
    c = np.empty(n_steps)
    dc = np.empty(n_steps)
    c[0] = _clip_seed(c_seed)
    dc[0] = c[0]                       # seed mass acts as the initial 'events'
    for n in range(1, n_steps):
        lam = mu + float(np.dot(dc[:n], gvals[n - np.arange(n)]))
        c_new = min(max(c[n - 1] + lam, 0.0), 1.0)
        dc[n] = c_new - c[n - 1]
        c[n] = c_new
    return c


def _hawkes_sim_batch(c0_vec, gvals, mu):
    """Vectorized-over-tweets mean-field Hawkes from c(0), full T steps."""
    N = len(c0_vec)
    c = np.empty((N, T)); dc = np.empty((N, T))
    c[:, 0] = np.clip(c0_vec, 0.0, 1.0)
    dc[:, 0] = c[:, 0]
    for n in range(1, T):
        g_lags = gvals[n - np.arange(n)]              # (n,)
        lam = mu + dc[:, :n] @ g_lags                 # (N,)
        c_new = np.clip(c[:, n - 1] + lam, 0.0, 1.0)
        dc[:, n] = c_new - c[:, n - 1]
        c[:, n] = c_new
    return c


def _discrete_branching_ratio(gvals):
    return float(np.sum(gvals[1:]))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return np.log(p / (1 - p))


class _HawkesParametric(Baseline):
    """Shared fit machinery for parametric SELF-EXCITING Hawkes kernels.
    mu = 0 (the post itself is the sole immigrant event; see the original
    file's rationale -- a free mu lets the optimizer zero out the kernel and
    fake saturation with background drift, which stops being a Hawkes
    process at all)."""
    n_params = 2

    def _kernel_vals(self, raw):
        raise NotImplementedError

    def _unpack(self, raw):
        raise NotImplementedError

    def fit(self, C, N_log):
        c0 = C[:, 0]
        def obj(raw):
            gvals = self._kernel_vals(raw)
            pred = _hawkes_sim_batch(c0, gvals, 0.0)
            return _future_mse(pred, C)
        best = None
        for raw0 in self._init_grid():
            v = obj(raw0)
            if best is None or v < best[0]:
                best = (v, raw0)
        res = minimize(obj, best[1], method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 2000})
        self._raw = res.x
        gvals = self._kernel_vals(res.x)
        self._gvals = gvals; self._mu = 0.0
        self.fitted = self._unpack(res.x)
        self.fitted["branching_ratio_discrete"] = _discrete_branching_ratio(gvals)
        self.fitted["mu"] = 0.0
        self._record_train_mse(C, N_log)
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        return _hawkes_sim(c_seed, T - k, self._gvals, self._mu)


class HawkesExp(_HawkesParametric):
    """g(t) = eta * omega * exp(-omega t). Continuous branching ratio = eta."""
    name = "Hawkes (exponential kernel)"

    def _init_grid(self):
        grids = []
        for eta in [0.5, 0.8, 0.95, 0.99]:
            for omega in [0.3, 0.7, 1.5, 3.0]:
                grids.append(np.array([_logit(eta / 0.999), np.log(omega)]))
        return grids

    def _kernel_vals(self, raw):
        eta = 0.999 * _sigmoid(raw[0])
        omega = np.exp(raw[1])
        lags = np.arange(T, dtype=float)
        g = eta * omega * np.exp(-omega * lags)
        g[0] = 0.0
        return g

    def _unpack(self, raw):
        return {"eta": float(0.999 * _sigmoid(raw[0])),
                "omega": float(np.exp(raw[1]))}


class HawkesPowerLaw(_HawkesParametric):
    """g(t) = eta * theta * delta^theta * (t + delta)^{-(1+theta)}.
    Heavy-tailed -- the point-process analogue of the fractional long-memory
    kernel (the thematic bridge to the PINN)."""
    name = "Hawkes (power-law kernel)"
    n_params = 3

    THETA_LO, THETA_HI = 0.1, 3.0
    DELTA_LO, DELTA_HI = 0.1, 5.0

    def _theta(self, x):
        return self.THETA_LO + (self.THETA_HI - self.THETA_LO) * _sigmoid(x)

    def _delta(self, x):
        return self.DELTA_LO + (self.DELTA_HI - self.DELTA_LO) * _sigmoid(x)

    def _init_grid(self):
        grids = []
        for eta in [0.5, 0.8, 0.95, 0.99]:
            for theta in [0.3, 0.8, 1.5, 2.5]:
                for delta in [0.5, 1.0, 2.0]:
                    grids.append(np.array([
                        _logit(eta / 0.999),
                        _logit((theta - self.THETA_LO) / (self.THETA_HI - self.THETA_LO)),
                        _logit((delta - self.DELTA_LO) / (self.DELTA_HI - self.DELTA_LO))]))
        return grids

    def _kernel_vals(self, raw):
        eta = 0.999 * _sigmoid(raw[0])
        theta = self._theta(raw[1]); delta = self._delta(raw[2])
        lags = np.arange(T, dtype=float)
        g = eta * theta * (delta ** theta) * (lags + delta) ** (-(1.0 + theta))
        g[0] = 0.0
        return g

    def _unpack(self, raw):
        return {"eta": float(0.999 * _sigmoid(raw[0])),
                "theta": float(self._theta(raw[1])),
                "delta": float(self._delta(raw[2]))}


class HawkesLearnedKernel(Baseline):
    """g(t) = sum_k w_k * phi_k(t), phi_k = exponentials at fixed decay
    rates, weights w_k >= 0 (+ background mu >= 0) found by non-negative
    least squares on teacher-forced increments; evaluated by free-running
    the mean-field simulation from c(0)."""
    name = "Hawkes (learned kernel, NNLS)"

    def __init__(self, decay_rates=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        self.decay_rates = np.asarray(decay_rates, float)
        self.n_params = len(decay_rates) + 1   # weights + mu

    def _phi(self):
        lags = np.arange(T, dtype=float)
        Phi = np.stack([np.exp(-w * lags) for w in self.decay_rates], axis=1)  # (T, K)
        Phi[0, :] = 0.0
        return Phi

    def fit(self, C, N_log):
        Phi = self._phi()                          # (T, K)
        K = Phi.shape[1]
        rows_X = []
        rows_y = []
        for i in range(C.shape[0]):
            c = C[i]
            dc = np.empty(T); dc[0] = c[0]; dc[1:] = np.diff(c)
            for n in range(1, T):
                lags = n - np.arange(n)            # n..1
                row = (dc[:n][:, None] * Phi[lags, :]).sum(axis=0)   # (K,)
                rows_X.append(np.concatenate([row, [1.0]]))          # + mu column
                rows_y.append(dc[n])
        X = np.asarray(rows_X); y = np.asarray(rows_y)
        w, _res = nnls(X, y)                       # w >= 0, length K+1
        self.weights = w[:K]
        self.mu = float(w[K])
        self._gvals = Phi @ self.weights           # (T,)
        self._gvals[0] = 0.0
        self.teacher_forced_mse = float(np.mean((X @ w - y) ** 2))
        self.fitted = {
            "weights": [float(x) for x in self.weights],
            "decay_rates": [float(x) for x in self.decay_rates],
            "mu": self.mu,
            "branching_ratio_discrete": _discrete_branching_ratio(self._gvals),
            "teacher_forced_mse": self.teacher_forced_mse,
        }
        self._record_train_mse(C, N_log)           # free-run MSE
        return self

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        return _hawkes_sim(c_seed, T - k, self._gvals, self.mu)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def all_baselines():
    """Return a fresh list of every classical baseline model instance (fit
    order). PINN checkpoints are added separately by run_pipeline.py via
    pinn_model.PINNCheckpoint -- they are not classical/fittable-here models."""
    return [
        Persistence(),
        LinearToMax(),
        Climatology(),
        LogisticGlobal(),
        LogisticFollowerScaled(),
        GompertzGlobal(),
        GompertzFollowerScaled(),
        BassDiffusion(),
        HawkesExp(),
        HawkesPowerLaw(),
        HawkesLearnedKernel(),
    ]
