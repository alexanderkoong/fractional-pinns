#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StitchedPINN (manifest-aware) -- recombines segment PINNs into one
forecaster, reading each results directory's own `manifest.pkl` for its
segment boundaries. This lets the same class load BOTH this folder's
3h-burst model (0-3/3-9/9-16/16-23) and any other segmentation's results directory for a direct comparison.

Semantics identical to the original: pick the segment PINN by hour, roll it
to the segment end with the corrected L1 stepper, chain the endpoint into
the next segment (student rollout across teacher-forced boundaries).
Duck-types `04_real_world_testing/pinn_model.PINNCheckpoint`.
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.normpath(os.path.join(HERE, "..", "04_real_world_testing"))
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from fractional_solvers import fode_solve_l1          # noqa: E402
from pinn_model import load_checkpoint                # noqa: E402


class StitchedPINN:
    n_params = 2

    def __init__(self, label, results_dir, num_timesteps, h_norm,
                 velocity_on=True):
        self.label = label
        self.num_timesteps = num_timesteps
        self.h_norm = h_norm
        self.velocity_on = velocity_on

        with open(os.path.join(results_dir, "manifest.pkl"), "rb") as fh:
            manifest = pickle.load(fh)
        self.seg_bounds = [tuple(b) for b in manifest["seg_bounds"]]

        self.segments = []
        betas = []
        for k in range(len(self.seg_bounds)):
            path = os.path.join(results_dir, f"segment_{k}", "params.pkl")
            kind, beta, alpha_fn, f_scalar = load_checkpoint(path)
            assert kind == "velocity_head", f"segment {k} unexpected kind {kind}"
            self.segments.append({"beta": beta, "alpha_fn": alpha_fn,
                                  "f": f_scalar})
            betas.append(beta)
        self.beta = float(np.mean(betas))
        self.betas = betas
        self.name = f"PINN ({label}{' +velocity' if velocity_on else ''})"
        self.fitted = {"kind": "stitched", "betas": betas,
                       "seg_bounds": self.seg_bounds,
                       "velocity_on": velocity_on}
        self.train_mean_mse = float("nan")

    def fit(self, C, N_log):
        return self

    def _segment_of(self, hour):
        for idx, (lo, hi) in enumerate(self.seg_bounds):
            if hour < hi:
                return idx
        return len(self.seg_bounds) - 1

    def predict(self, c_seed, k, N_log_i, v_i=0.0):
        out = [float(c_seed)]
        c = float(c_seed)
        hour = int(k)
        last = self.num_timesteps - 1
        while hour < last:
            s = self._segment_of(hour)
            seg_end = min(self.seg_bounds[s][1], last)
            n = seg_end - hour + 1
            seg = self.segments[s]
            alpha = seg["alpha_fn"](N_log_i, v_i, self.velocity_on)
            traj = fode_solve_l1(seg["f"], alpha, seg["beta"], c, n,
                                 self.h_norm, clip01=True)
            out.extend(float(x) for x in traj[1:])
            c = float(traj[-1])
            hour = seg_end
        return np.array(out)

    def predict_fn(self):
        return lambda c_seed, k, N_log_i, v_i=0.0: self.predict(
            c_seed, k, N_log_i, v_i)

    def needs_velocity(self):
        return self.velocity_on
