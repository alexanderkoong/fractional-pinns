#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PySR on each segment's learned f_k (same config as the single-PINN
symbolic_regression.py). Outputs results/symbolic_segments.txt."""

import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
C_GRID = np.linspace(0.0, 1.0, 500)


def f_on_grid(f_params):
    x = C_GRID.reshape(1, -1, 1).astype(np.float64)
    for i in range(len(f_params) - 1):
        W, b = f_params[i]
        x = np.tanh(x @ np.asarray(W) + np.asarray(b))
    W, b = f_params[-1]
    return (np.exp(x @ np.asarray(W) + np.asarray(b))
            * (C_GRID.reshape(1, -1, 1) * (1 - C_GRID.reshape(1, -1, 1))))[0, :, 0]


def main():
    from pysr import PySRRegressor
    report = ["PER-SEGMENT SYMBOLIC REGRESSION (PySR, standard config)",
              "=" * 74]
    for k in range(4):
        with open(os.path.join(RESULTS, f"segment_{k}", "params.pkl"),
                  "rb") as fh:
            _, f_params, head, beta_raw = pickle.load(fh)
        beta = 0.1 + 0.89 / (1.0 + np.exp(-float(np.asarray(beta_raw))))
        fv = f_on_grid(f_params)
        c_fit = np.concatenate([C_GRID, [0.0, 1.0]])
        f_fit = np.concatenate([fv, [0.0, 0.0]])
        w = np.concatenate([np.ones(len(C_GRID)), [10.0, 10.0]])
        model = PySRRegressor(
            model_selection="best", niterations=300, maxsize=14,
            binary_operators=["+", "-", "*", "/"], unary_operators=["exp"],
            complexity_of_operators={"/": 2, "exp": 3},
            elementwise_loss="loss(x, y, w) = w*(x - y)^2",
            output_directory=os.path.join(RESULTS, "pysr"),
            run_id=f"segment_{k}", progress=False, verbosity=0)
        model.fit(c_fit.reshape(-1, 1), f_fit, weights=w)
        expr = str(model.sympy())
        pred = model.predict(C_GRID.reshape(-1, 1))
        corr = float(np.corrcoef(fv, pred)[0, 1])
        report.append(f"segment {k}: beta={beta:.4f}   f(c) = {expr}   "
                      f"(corr vs NN {corr:.5f})")
        print(report[-1], flush=True)
    text = "\n".join(report)
    with open(os.path.join(RESULTS, "symbolic_segments.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
