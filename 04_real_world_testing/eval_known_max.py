#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
TRACK A -- KNOWN-FINAL-MAX (SHAPE-ONLY) SCORING
=============================================================================
Adapted from "beat logistic updated final filter/eval_harness.py"'s
score_dense_full / score_sparse_full, retargeted at the current 24-node
(hours 0..23) dataset. Unchanged in substance -- same metric, same
single-point-restart design.

WHAT "KNOWN FINAL MAX" MEANS HERE
----------------------------------
Every trajectory C[i] is normalized by tweet i's OWN true final/peak
engagement (per-tweet max-normalization, done once at dataset-build time --
see data/SOURCE_DATASET_README.md). Scoring a model's forward-solve against
this normalized curve therefore tests SHAPE ONLY: the model is implicitly
told the eventual scale (that's what the normalization bakes in), and is
graded purely on whether its growth DYNAMICS match the real trajectory's.
This is the easier, "diagnostic" track -- it isolates dynamics quality from
the much harder scale-forecasting problem that eval_leak_free.py (Track B)
grades.

METRIC
------
Plain (non-log) MSE between predicted and true normalized-c trajectories,
over the FUTURE points only (pred[0] == the seed is excluded). Reported as
"% of tweets with MSE below threshold" at thresholds 0.10 / 0.05 / 0.01 (the
stricter two are not in the original file's default set as *headline*
numbers, but ARE already part of its DEFAULT_THRESHOLDS tuple -- no metric
redefinition here, just surfacing the values requested).
=============================================================================
"""

import numpy as np

NUM_TIMESTEPS = 24                        # hours 0..23 (current dataset)
H_NORM = 1.0 / (NUM_TIMESTEPS - 1)

DEFAULT_THRESHOLDS = (0.10, 0.05, 0.01)   # exactly what was requested
DEFAULT_RESTART_HOURS = (0, 1, 2, 3, 6, 12, 18)


def _threshold_pcts(mses, thresholds):
    mses = np.asarray(mses, dtype=float)
    return {thr: 100.0 * float(np.mean(mses < thr)) for thr in thresholds}


def score_dense_full(C, N_log, predict_fn, thresholds=DEFAULT_THRESHOLDS,
                     restart_hours=DEFAULT_RESTART_HOURS, early_window=(1, 7),
                     c_floor=1e-3, V=None):
    """
    Score a model on a DENSE (complete-trajectory) set via single-point
    restart at each hour in `restart_hours`. The h0 entry (k=0) is the
    headline metric: forward-solve from c(0), MSE over hours 1..NUM_TIMESTEPS-1.

    V: optional (N,) array of per-tweet early-velocity covariates, passed
    through to predict_fn as its 4th arg for models whose alpha depends on
    it (pinn_model.PINNCheckpoint with a velocity head); ignored (defaults
    to 0.0 per tweet) by every other model. Irrelevant for the DENSE set
    since hour 0 and hour 1 are always both observed here.

    Returns a dict:
        {
          'h0':        {thr: pct, ...},          # forward solve from c(0)
          'h0_mean_mse', 'h0_median_mse',
          'early':     {thr: pct, ...},          # MSE over hours [1,7) only, from c(0)
          'early_mean_mse',
          'per_hour':  {k: {thr: pct, ...}, ...} # restart at each k
          'n': N,
        }
    """
    N = C.shape[0]
    e_lo, e_hi = early_window
    out = {"n": N, "per_hour": {}}

    for k in restart_hours:
        if k >= NUM_TIMESTEPS - 1:
            continue
        mses = np.empty(N)
        for i in range(N):
            actual = C[i]
            c_k = max(float(actual[k]), c_floor)
            v_i = float(V[i]) if V is not None else 0.0
            pred = predict_fn(c_k, k, float(N_log[i]), v_i)  # length NUM_TIMESTEPS-k
            mses[i] = float(np.mean((pred[1:] - actual[k + 1:]) ** 2))
        out["per_hour"][k] = _threshold_pcts(mses, thresholds)
        if k == 0:
            out["h0"] = _threshold_pcts(mses, thresholds)
            out["h0_mean_mse"] = float(np.mean(mses))
            out["h0_median_mse"] = float(np.median(mses))
            early = np.empty(N)
            for i in range(N):
                actual = C[i]
                c0 = max(float(actual[0]), c_floor)
                v_i = float(V[i]) if V is not None else 0.0
                pred = predict_fn(c0, 0, float(N_log[i]), v_i)
                early[i] = float(np.mean((pred[e_lo:e_hi] - actual[e_lo:e_hi]) ** 2))
            out["early"] = _threshold_pcts(early, thresholds)
            out["early_mean_mse"] = float(np.mean(early))
    return out


def score_sparse_full(tweets, nlog, predict_fn, thresholds=DEFAULT_THRESHOLDS,
                      restart_hours=DEFAULT_RESTART_HOURS, c_floor=1e-3, vlog=None):
    """
    Score a model on a SPARSE (partial/irregularly-observed trajectory) set.
    A tweet is eligible at hour k iff it has an observed snapshot at k AND at
    least 2 observed snapshots after k; predictions are compared only at the
    observed future hours.

    vlog: optional {post_id: v} dict of early-velocity covariates (see
    score_dense_full). Sparse tweets don't all have both hour 0 and hour 1
    observed, so v isn't always computable -- a tweet missing from vlog, or
    mapped to NaN, is additionally excluded from eligibility whenever vlog
    is passed (i.e. only when scoring a model that actually needs it; pass
    vlog=None, the default, to score every otherwise-eligible tweet with
    v_i=0.0, which is correct/harmless for every non-velocity model).

    Returns the same structure as score_dense_full (minus the early-window
    block), plus per-hour eligible counts under 'n_per_hour'.
    """
    pids = list(tweets.keys())
    out = {"per_hour": {}, "n_per_hour": {}}

    for k in restart_hours:
        if k >= NUM_TIMESTEPS - 1:
            continue
        eligible = [p for p in pids
                    if k in tweets[p]["hours"]
                    and int(np.sum(tweets[p]["hours"] > k)) >= 2]
        if vlog is not None:
            eligible = [p for p in eligible
                       if p in vlog and not np.isnan(vlog[p])]
        out["n_per_hour"][k] = len(eligible)
        if not eligible:
            out["per_hour"][k] = {thr: float("nan") for thr in thresholds}
            continue

        mses = np.empty(len(eligible))
        for idx, pid in enumerate(eligible):
            tw = tweets[pid]
            k_idx = int(np.where(tw["hours"] == k)[0][0])
            c_k = max(float(tw["engagement"][k_idx]), c_floor)
            v_i = float(vlog[pid]) if vlog is not None else 0.0
            pred = predict_fn(c_k, k, float(nlog[pid]), v_i)  # length NUM_TIMESTEPS-k
            fmask = tw["hours"] > k
            fh = tw["hours"][fmask]
            fa = tw["engagement"][fmask]
            mses[idx] = float(np.mean((pred[fh - k] - fa) ** 2))
        out["per_hour"][k] = _threshold_pcts(mses, thresholds)
        if k == 0:
            out["h0"] = _threshold_pcts(mses, thresholds)
            out["h0_mean_mse"] = float(np.mean(mses))
            out["h0_median_mse"] = float(np.median(mses))
            out["n"] = len(eligible)
    return out
