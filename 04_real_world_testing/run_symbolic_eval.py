#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYMBOLIC TRACK A -- the evaluation behind the paper's reported accuracies.

run_final_eval.py rolls out the trained NEURAL f reconstructed from each
params.pkl. The paper instead reports forward solves of the SYMBOLIC laws
distilled from those networks by PySR (the single-window law, Eq. (14), and
the teacher-student chain of the four Table-4 laws). The two variants give
materially different numbers:

  test_dense no-decline (n=86), h0, thresholds 0.10 / 0.05 / 0.025 / 0.01
    teacher-student 3h burst, NN f ....... 96.5 / 87.2 / 79.1 / 66.3
    teacher-student 3h burst, symbolic ... 100.0 / 91.9 / 81.4 / 64.0   <- paper
    single-window baseline, NN f ......... 59.3 / 52.3 / 40.7 / 12.8
    single-window baseline, symbolic ..... 64.0 / 53.5 / 39.5 / 10.5    <- paper

  train no-decline (n=348), h0
    teacher-student 3h burst, symbolic ... 87.6 (<0.025) / 72.1 (<0.01) <- paper

The paper's numbers reproduce ONLY with the FULL-PRECISION PySR coefficients
below; the rounded forms printed in the paper give slightly different results
(both variants are scored for the record).

Symbolic sources (verbatim):
  single window : ../02_baseline_pinn/results/symbolic_results.txt
                    f(c) = (c - 0.029468223/(1.0295491 - c)) * 1.0972929
  segments      : ../03_teacher_student_3h_burst/results/symbolic_segments.txt
                    seg0 (0-3h) : 1.7939605*c + 0.18553144
                                  + 0.18497752/(c - 1.0927554)
                    seg1 (3-9h) : c*(3.0545263 - 2.9919586*c)
                    seg2 (9-16h): c*(c - 0.9759003)*(-4.7721124)
                    seg3(16-23h): c*(-5.80983 + 7.8357487/(c + 0.28688404))

The script also scores the classical baselines (fit on the no-decline
train subset, exactly as in run_final_eval.py) on the same splits, so its
output contains the paper's full model-comparison table (Table 6: seen =
train no-decline n=348, unseen = test_dense no-decline n=86).

Everything else is identical to run_final_eval.py: alpha comes from each
checkpoint's fitted velocity head, beta from each checkpoint's fitted value,
rollouts use fractional_solvers.fode_solve_l1 (clip01=True, h = 1/23), and
scoring is eval_known_max.score_dense_full from observed c(0) (floored at
1e-3), MSE over hours 1..23, on the same no-decline splits (train
403 -> 348, test_dense 100 -> 86).

Outputs: results/SYMBOLIC_EVAL.md (incl. Table 6),
         results/track_a_symbolic_full.csv
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
BURST = os.path.join(ROOT, "03_teacher_student_3h_burst")
for p in (HERE, BURST):
    if p not in sys.path:
        sys.path.insert(0, p)

import baselines as B                            # noqa: E402
import eval_known_max as EKM                     # noqa: E402
from pinn_model import PINNCheckpoint            # noqa: E402
from run_final_eval import load_dense_npz        # noqa: E402
from stitched_model import StitchedPINN          # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(HERE, "results")
SINGLE_CKPT = os.path.join(ROOT, "02_baseline_pinn", "results", "params.pkl")
BURST_RESULTS = os.path.join(BURST, "results")

THRESHOLDS = (0.10, 0.05, 0.025, 0.01)

# --- distilled laws, full precision (sources in module docstring) -----------
SYM_SINGLE = lambda c: (c - 0.029468223 / (1.0295491 - c)) * 1.0972929  # noqa: E731
SYM_SEGS = [
    lambda c: c * 1.7939605 + 0.18553144 + 0.18497752 / (c - 1.0927554),
    lambda c: c * (3.0545263 - 2.9919586 * c),
    lambda c: c * (c - 0.9759003) * (-4.7721124),
    lambda c: c * (-5.80983 + 7.8357487 / (c + 0.28688404)),
]

# --- the same laws as ROUNDED in the paper (Eq. 14 / Table 4) ---------------
PAPER_SINGLE = lambda c: 1.097 * (c - 0.0295 / (1.0295 - c))  # noqa: E731
PAPER_SEGS = [
    lambda c: 1.79 * c + 0.19 + 0.18 / (c - 1.09),
    lambda c: 3.05 * c * (1 - 0.98 * c),
    lambda c: 4.77 * c * (0.98 - c),
    lambda c: c * (7.84 / (c + 0.29) - 5.81),
]
PAPER_BETAS = [0.228, 0.744, 0.989, 0.989]


def no_decline_mask(C):
    return (np.diff(C, axis=1) >= 0.0).all(axis=1)


def stitched(label, laws=None, betas=None):
    """Burst-window StitchedPINN; optionally swap each segment's f (and
    beta) for a symbolic law while keeping the checkpoint alpha heads."""
    m = StitchedPINN(label, BURST_RESULTS, EKM.NUM_TIMESTEPS, EKM.H_NORM,
                     velocity_on=True)
    if laws is not None:
        for k, seg in enumerate(m.segments):
            seg["f"] = laws[k]
            if betas is not None:
                seg["beta"] = betas[k]
    m.name = label
    return m


def single(label, law=None, beta=None):
    """Single-window checkpoint; optionally swap the NN f (and beta) for a
    symbolic law while keeping the checkpoint alpha head."""
    m = PINNCheckpoint(label, SINGLE_CKPT, EKM.NUM_TIMESTEPS, EKM.H_NORM,
                       velocity_on=True)
    if law is not None:
        m.f_scalar = law
    if beta is not None:
        m.beta = beta
    m.name = label
    return m


def main():
    os.makedirs(RESULTS, exist_ok=True)

    train = load_dense_npz(os.path.join(DATA_DIR, "train.npz"))
    test = load_dense_npz(os.path.join(DATA_DIR, "test_dense.npz"))
    v_tr = np.log1p(train["C_raw"][:, 1]) - np.log1p(train["C_raw"][:, 0])
    v_te = np.log1p(test["C_raw"][:, 1]) - np.log1p(test["C_raw"][:, 0])
    m_tr = no_decline_mask(train["C_norm"])
    m_te = no_decline_mask(test["C_norm"])
    print(f"train {len(m_tr)} -> {int(m_tr.sum())} no-decline; "
          f"test_dense {len(m_te)} -> {int(m_te.sum())} no-decline")

    splits = (
        ("train_nodecline", train["C_norm"][m_tr], train["N_log"][m_tr],
         v_tr[m_tr]),
        ("test_dense_nodecline", test["C_norm"][m_te], test["N_log"][m_te],
         v_te[m_te]),
        ("test_dense_full", test["C_norm"], test["N_log"], v_te),
    )

    models = [
        stitched("TS 3h burst -- SYMBOLIC (paper numbers)", SYM_SEGS),
        stitched("TS 3h burst -- NN f (run_final_eval.py)"),
        stitched("TS 3h burst -- symbolic as rounded in paper Table 4",
                 PAPER_SEGS, PAPER_BETAS),
        single("single window -- SYMBOLIC (paper numbers)", SYM_SINGLE),
        single("single window -- NN f (run_final_eval.py)"),
        single("single window -- symbolic as rounded in paper Eq. (14)",
               PAPER_SINGLE, beta=0.136),
    ]

    classical = []
    for m in B.all_baselines():
        m.fit(train["C_norm"][m_tr], train["N_log"][m_tr])
        classical.append(m)

    rows, score = [], {}
    md = ["# Symbolic-embedded Track A (the paper's evaluation)", "",
          "Forward solves use the DISTILLED PySR laws (full precision) in "
          "place of the neural f; alpha/beta from the fitted checkpoints; "
          "protocol identical to run_final_eval.py (h0 restart, clip01 L1 "
          "rollout, MSE over hours 1..23). See the module docstring of "
          "run_symbolic_eval.py for sources.",
          ""]
    for split_label, C, N, V in splits:
        md += [f"## {split_label} (n={len(C)})", "",
               "| Model | <0.10 | <0.05 | <0.025 | <0.01 |",
               "|---|---|---|---|---|"]
        print(f"== {split_label} (n={len(C)})")
        for m in models + classical:
            needs_v = getattr(m, "needs_velocity", lambda: False)()
            d = EKM.score_dense_full(C, N, m.predict_fn(),
                                     thresholds=THRESHOLDS,
                                     restart_hours=(0,),
                                     V=V if needs_v else None)
            t = d["h0"]
            score[(m.name, split_label)] = t
            cells = " | ".join(f"{t[thr]:.2f}%" for thr in THRESHOLDS)
            md.append(f"| {m.name} | {cells} |")
            print(f"  {m.name:<52} " +
                  "  ".join(f"<{thr}={t[thr]:6.2f}" for thr in THRESHOLDS))
            row = {"model": m.name, "split": split_label,
                   "restart_hour": 0, "n": d["n"]}
            row.update({f"pct_below_{thr}": t[thr] for thr in THRESHOLDS})
            rows.append(row)
        md.append("")

    md += ["## Table 6 -- seen (train no-decline, n=348) vs unseen "
           "(test_dense no-decline, n=86)", "",
           "| Model | seen <0.025 | seen <0.01 | unseen <0.025 | unseen <0.01 |",
           "|---|---|---|---|---|"]
    for m in [models[0]] + classical:
        cells = [score[(m.name, sp)][thr]
                 for sp in ("train_nodecline", "test_dense_nodecline")
                 for thr in (0.025, 0.01)]
        md.append(f"| {m.name} | " +
                  " | ".join(f"{x:.1f}%" for x in cells) + " |")
    md.append("")

    csv_path = os.path.join(RESULTS, "track_a_symbolic_full.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md_path = os.path.join(RESULTS, "SYMBOLIC_EVAL.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
