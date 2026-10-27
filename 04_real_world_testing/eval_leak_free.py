#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
TRACK B -- LEAK-FREE, UNKNOWN-FINAL-MAX SCORING (real-world deployment track)
=============================================================================
Adapted from "beat logistic updated final filter/run_realworld_forecast.py",
retargeted at the current 24-node (hours 0..23) dataset and RAW counts
reconstructed as (per-tweet normalized c) x (per-tweet true peak), both of
which are already stored in the current dataset's .npz files -- see
data/SOURCE_DATASET_README.md.

WHAT "LEAK-FREE" MEANS HERE
-----------------------------
For every restart hour k_start, a model sees ONLY:
  * the author's follower count, and
  * raw engagement observed strictly BEFORE k_start (the value at hour
    k_start-1; cumulative engagement is monotone-ish and that single value
    is a sufficient statistic for everything before it, matching every
    growth-curve ODE's Markovian state),
and must predict raw engagement at k_start, k_start+1, .... It NEVER sees
the tweet's true final/peak value -- that is precisely what distinguishes
this from eval_known_max.py's Track A, where per-tweet max-normalization
implicitly hands the model the answer.

k_start = 0 is a genuine cold start: no engagement history at all, only
follower count.

TWO-STAGE DESIGN (unchanged from the original)
------------------------------------------------
Stage 1 (scale): a ridge regression fit on a POOLED PANEL across every
train tweet x every candidate seed hour, predicting log1p(final raw
engagement) from [log1p(seed value), N_log, hour_frac]. A separate
followers-only cold-start regressor handles k_start=0.
Stage 2 (shape): any fitted classical/PINN model exposing
`.predict(c_seed, k, N_log_i)` (baselines.py's classes and
pinn_model.PINNCheckpoint both already match this), re-seeded at k_start and
driven by the STAGE-1 PREDICTED scale -- never the true one.

METRIC
------
Raw counts span orders of magnitude (25 to 17,000+ in this dataset), so
plain MSE would be dominated entirely by a handful of viral outliers -- the
original pipeline's own choice, kept here unchanged, is log1p-MSE:
    mean( (log1p(pred) - log1p(actual))^2 )
Reported as "% of tweets with log1p-trajectory-MSE below threshold" at
0.10 / 0.05 / 0.01. This is NOT the same metric space as Track A's plain
c-space MSE -- see run_pipeline.py's report for the explicit distinction.
=============================================================================
"""

import numpy as np

NUM_TIMESTEPS = 24                        # hours 0..23 (current dataset)
MAX_SEED_HOUR = NUM_TIMESTEPS - 2         # largest seed hour ever used (22)
C_FLOOR_NORM = 1e-3                       # floor for predicted-seed fed to shape models

DEFAULT_THRESHOLDS = (0.10, 0.05, 0.01)   # exactly what was requested


# =============================================================================
# STAGE 1 -- SCALE REGRESSION
# =============================================================================
class RidgeScaleModel:
    """log1p(final) ~ log1p(h_seed) + N_log + hour_frac, fit on a POOLED
    PANEL across every tweet x every candidate seed hour (0..MAX_SEED_HOUR).
    The hour_frac feature (seed_hour / (T-1)) lets one model serve every
    restart point without over-predicting late-hour remaining growth."""
    name = "Ridge scale regressor (pooled across seed hours)"

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.coef_ = None
        self.r2_ = float("nan")
        self.rmse_ = float("nan")

    def _design(self, log_h, n_log, hour_frac):
        return np.column_stack([np.ones_like(log_h), log_h, n_log, hour_frac])

    def fit(self, log_h, n_log, hour_frac, log_final):
        X = self._design(log_h, n_log, hour_frac)
        A = X.T @ X + self.l2 * np.eye(X.shape[1])
        b = X.T @ log_final
        self.coef_ = np.linalg.solve(A, b)
        pred = X @ self.coef_
        resid = log_final - pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((log_final - log_final.mean()) ** 2))
        self.r2_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.rmse_ = float(np.sqrt(np.mean(resid ** 2)))
        return self

    def predict_log_final(self, log_h, n_log, hour_frac):
        X = self._design(np.atleast_1d(log_h), np.atleast_1d(n_log), np.atleast_1d(hour_frac))
        return X @ self.coef_


class ColdStartScaleModel:
    """log1p(final) ~ N_log only -- the k_start=0 case, where there is no
    engagement history at all, only the author's follower count."""
    name = "Cold-start scale regressor (followers only)"

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.coef_ = None
        self.r2_ = float("nan")
        self.rmse_ = float("nan")

    def _design(self, n_log):
        n_log = np.atleast_1d(n_log)
        return np.column_stack([np.ones_like(n_log), n_log])

    def fit(self, n_log, log_final):
        X = self._design(n_log)
        A = X.T @ X + self.l2 * np.eye(X.shape[1])
        b = X.T @ log_final
        self.coef_ = np.linalg.solve(A, b)
        pred = X @ self.coef_
        resid = log_final - pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((log_final - log_final.mean()) ** 2))
        self.r2_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.rmse_ = float(np.sqrt(np.mean(resid ** 2)))
        return self

    def predict_log_final(self, n_log):
        X = self._design(n_log)
        return X @ self.coef_


def build_scale_panel(C_raw, N_log, max_seed_hour=MAX_SEED_HOUR):
    """Pool (log1p(h_seed), N_log, hour_frac, log1p(final)) rows across every
    tweet x every candidate seed hour 0..max_seed_hour."""
    N, T = C_raw.shape
    log_final = np.log1p(C_raw[:, -1])
    logh_list, nlog_list, hourfrac_list, target_list = [], [], [], []
    for s in range(0, max_seed_hour + 1):
        logh_list.append(np.log1p(np.clip(C_raw[:, s], 0, None)))
        nlog_list.append(N_log)
        hourfrac_list.append(np.full(N, s / (T - 1)))
        target_list.append(log_final)
    return (np.concatenate(logh_list), np.concatenate(nlog_list),
            np.concatenate(hourfrac_list), np.concatenate(target_list))


def _predict_final_from_seed(scale_model, h_seed, n_log_i, k):
    hour_frac = k / (NUM_TIMESTEPS - 1)
    log_final = scale_model.predict_log_final(np.log1p(h_seed), n_log_i, hour_frac)[0]
    return max(np.expm1(log_final), h_seed, 1.0)


def _predict_final_cold(cold_scale_model, n_log_i):
    log_final = cold_scale_model.predict_log_final(n_log_i)[0]
    return max(np.expm1(log_final), 1.0)


# =============================================================================
# FORECASTERS
#   predict_raw(h_seed, n_log_i, k)  -> length (T-k) array, hours k.., [0]=h_seed
#   predict_cold(n_log_i)            -> length T array, hours 0.. (no history)
# =============================================================================
class PersistenceForecaster:
    """'Nothing changes' anchor: future engagement stays at the last known
    (seed) value."""
    name = "Persistence (raw, c=seed)"

    def predict_raw(self, h_seed, n_log_i, k, v_i=0.0):
        n = NUM_TIMESTEPS - k
        return np.full(n, max(h_seed, 0.0))

    def predict_cold(self, n_log_i, v_i=0.0):
        return np.zeros(NUM_TIMESTEPS)


class FollowerRegressionForecaster:
    """Stage-1 only: predicts the eventual SCALE but assumes the tweet has
    already reached it (flat future trajectory) -- isolates how much of a
    two-stage model's skill is coming from Stage 1 alone."""
    name = "Follower-adjusted regression (scale only)"

    def __init__(self, scale_model, cold_scale_model):
        self.scale_model = scale_model
        self.cold_scale_model = cold_scale_model

    def predict_raw(self, h_seed, n_log_i, k, v_i=0.0):
        n = NUM_TIMESTEPS - k
        final_pred = _predict_final_from_seed(self.scale_model, h_seed, n_log_i, k)
        traj = np.full(n, final_pred)
        traj[0] = max(h_seed, 0.0)
        return traj

    def predict_cold(self, n_log_i, v_i=0.0):
        final_pred = _predict_final_cold(self.cold_scale_model, n_log_i)
        return np.full(NUM_TIMESTEPS, final_pred)


class TwoStageForecaster:
    """Stage 1 (predicted scale) + Stage 2 (shape from a pre-fitted normalized
    growth-curve/PINN model, restartable at any hour k). Leak-free: the shape
    model never sees the tweet's true final engagement -- shape_model can be
    any baselines.py classical model OR a pinn_model.PINNCheckpoint, both
    exposing the same .predict(c_seed, k, n_log_i) interface."""

    def __init__(self, name, shape_model, scale_model, cold_scale_model):
        self.name = name
        self.shape_model = shape_model
        self.scale_model = scale_model
        self.cold_scale_model = cold_scale_model

    def predict_raw(self, h_seed, n_log_i, k, v_i=0.0):
        final_pred = _predict_final_from_seed(self.scale_model, h_seed, n_log_i, k)
        c_seed = min(max(h_seed / final_pred, C_FLOOR_NORM), 1.0 - 1e-4)
        frac = self.shape_model.predict(c_seed, k, n_log_i, v_i)   # length T-k, in [0,1]
        traj = frac * final_pred
        traj[0] = max(h_seed, 0.0)
        return traj

    def predict_cold(self, n_log_i, v_i=0.0):
        final_pred = _predict_final_cold(self.cold_scale_model, n_log_i)
        frac = self.shape_model.predict(C_FLOOR_NORM, 0, n_log_i, v_i)   # full curve from ~0
        return frac * final_pred


# =============================================================================
# METRICS
# =============================================================================
def _log_traj_mse(pred_future, actual_future):
    return float(np.mean((np.log1p(np.clip(pred_future, 0, None))
                           - np.log1p(np.clip(actual_future, 0, None))) ** 2))


def _threshold_pcts(values, thresholds):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {thr: float("nan") for thr in thresholds}
    return {thr: 100.0 * float(np.mean(values < thr)) for thr in thresholds}


def _virality_capture(final_actual, final_pred, frac=0.10):
    final_actual = np.asarray(final_actual, dtype=float)
    final_pred = np.asarray(final_pred, dtype=float)
    n = len(final_actual)
    k = max(1, int(round(n * frac)))
    true_top = set(np.argsort(-final_actual)[:k])
    pred_top = set(np.argsort(-final_pred)[:k])
    return 100.0 * len(true_top & pred_top) / k


def _summarize(traj_mse, final_pred, final_actual, thresholds, n):
    traj_mse = np.asarray(traj_mse, dtype=float)
    final_pred = np.asarray(final_pred, dtype=float)
    final_actual = np.asarray(final_actual, dtype=float)

    log_final_pred = np.log1p(np.clip(final_pred, 0, None))
    log_final_actual = np.log1p(np.clip(final_actual, 0, None))
    abs_log_err = np.abs(log_final_pred - log_final_actual)

    ratio = final_pred / np.clip(final_actual, 1e-9, None)
    within2x = 100.0 * float(np.mean((ratio >= 0.5) & (ratio <= 2.0))) if n else float("nan")
    within5x = 100.0 * float(np.mean((ratio >= 0.2) & (ratio <= 5.0))) if n else float("nan")

    out = {
        "n": n,
        "mean_log_traj_mse": float(np.mean(traj_mse)) if n else float("nan"),
        "median_log_traj_mse": float(np.median(traj_mse)) if n else float("nan"),
        "final_median_abs_log_err": float(np.median(abs_log_err)) if n else float("nan"),
        "final_rmse_log": float(np.sqrt(np.mean((log_final_pred - log_final_actual) ** 2))) if n else float("nan"),
        "within_2x_pct": within2x,
        "within_5x_pct": within5x,
        "top10pct_capture": _virality_capture(final_actual, final_pred, 0.10) if n >= 10 else float("nan"),
    }
    for thr, pct in _threshold_pcts(traj_mse, thresholds).items():
        out[f"traj_mse_<{thr}"] = pct
    return out


# =============================================================================
# EVALUATION -- parameterized by restart hour k_start
# =============================================================================
def evaluate_dense(model, C_RAW, N_LOG, thresholds, k_start, V=None):
    """k_start=0: cold start (score against ALL hours). k_start>=1: seed at
    hour (k_start-1), score against hours k_start..end.

    V: optional (N,) array of per-tweet early-velocity covariates -- FIXED
    per tweet regardless of k_start (v is defined from hours 0-1, not from
    the seed hour), passed to shape models with a velocity head. Ignored
    (defaults to 0.0) by every other model."""
    N = C_RAW.shape[0]
    traj_mse = np.empty(N)
    final_pred = np.empty(N)
    final_actual = C_RAW[:, -1]

    for i in range(N):
        v_i = float(V[i]) if V is not None else 0.0
        if k_start == 0:
            pred_future = model.predict_cold(float(N_LOG[i]), v_i)
            actual_future = C_RAW[i]
        else:
            seed_hour = k_start - 1
            h_seed = max(float(C_RAW[i, seed_hour]), 0.0)
            pred_from_seed = model.predict_raw(h_seed, float(N_LOG[i]), k=seed_hour, v_i=v_i)
            pred_future = pred_from_seed[1:]        # drop the known seed point
            actual_future = C_RAW[i, k_start:]

        traj_mse[i] = _log_traj_mse(pred_future, actual_future)
        final_pred[i] = pred_future[-1]

    return _summarize(traj_mse, final_pred, final_actual, thresholds, n=N)


def evaluate_sparse(model, tweets, nlog, thresholds, k_start, vlog=None):
    """Same restart-hour logic as evaluate_dense, but only scored at each
    tweet's actually-observed hours. `tweets[pid] = {'hours': int[],
    'engagement': RAW float[]}` (NOT normalized -- see run_pipeline.py).

    vlog: optional {post_id: v} dict (see eval_known_max.score_sparse_full --
    same exclusion behavior: a tweet missing from vlog or mapped to NaN is
    excluded whenever vlog is passed, i.e. only when scoring a model that
    actually needs it)."""
    traj_mse, final_pred, final_actual = [], [], []

    for pid, tw in tweets.items():
        if vlog is not None and (pid not in vlog or np.isnan(vlog[pid])):
            continue
        v_i = float(vlog[pid]) if vlog is not None else 0.0
        hours = tw["hours"]
        eng = tw["engagement"]

        if k_start == 0:
            if len(hours) < 2:
                continue
            pred_full = model.predict_cold(float(nlog[pid]), v_i)
            pred_at_obs = pred_full[hours]
            fa = eng
        else:
            seed_hour = k_start - 1
            if seed_hour not in hours:
                continue
            future_mask = hours > seed_hour
            if int(np.sum(future_mask)) < 1:
                continue
            seed_idx = int(np.where(hours == seed_hour)[0][0])
            h_seed = max(float(eng[seed_idx]), 0.0)
            pred_from_seed = model.predict_raw(h_seed, float(nlog[pid]), k=seed_hour, v_i=v_i)
            fh = hours[future_mask]
            fa = eng[future_mask]
            pred_at_obs = pred_from_seed[fh - seed_hour]

        traj_mse.append(_log_traj_mse(pred_at_obs, fa))
        final_pred.append(float(pred_at_obs[-1]))
        final_actual.append(float(fa[-1]))

    return _summarize(np.array(traj_mse), np.array(final_pred),
                       np.array(final_actual), thresholds, n=len(traj_mse))
