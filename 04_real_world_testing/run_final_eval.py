#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL REAL-WORLD EVALUATION -- the paper's model-comparison run.

Scores, on the held-out test set:
  * the 11 classical baselines (fit on the no-decline train subset),
  * the single-window baseline PINN (02_baseline_pinn's checkpoint --
    whitened-full L1 + velocity head, the loss validated in
    01_synthetic_validation_24h),
  * the stitched teacher-student model (03_teacher_student_3h_burst's four
    segment checkpoints, windows 0-3/3-9/9-16/16-23, routed by hour with
    chained endpoints),

on Track A (known final max: shape only, forward-solve from hour 0) and
Track B (leak-free two-stage forecasting by restart hour).  The held-out
test_dense split is scored both no-decline-filtered (n = 86; the headline --
matches the training filter) and unfiltered (n = 100).

Expected output (deterministic given the shipped checkpoints):

    Track A, no-decline test_dense      <0.10   <0.05   <0.01
    teacher-student 3h burst (NN f)     96.5%   87.2%   66.3%
    single-window baseline PINN (NN f)  59.3%   52.3%   12.8%
    best classical (Climatology/Bass)  100.0%   95.3%   44.2%

The PINN rows here roll out the neural f. The paper's PINN accuracies are
forward solves of the distilled symbolic laws -- see run_symbolic_eval.py.
The classical-baseline rows (incl. the <0.025 column) are the paper's.

Outputs: results/FINAL_LEADERBOARD.md, results/track_[ab]_final.csv
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.normpath(os.path.join(HERE, ".."))

import baselines as B                       # noqa: E402
import eval_known_max as EKM                # noqa: E402
import eval_leak_free as ELF                # noqa: E402
from pinn_model import PINNCheckpoint       # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "03_teacher_student_3h_burst"))
from stitched_model import StitchedPINN     # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(HERE, "results")
SINGLE_CKPT = os.path.join(ROOT, "02_baseline_pinn", "results", "params.pkl")
STITCHED_DIR = os.path.join(ROOT, "03_teacher_student_3h_burst", "results")

THRESHOLDS = (0.10, 0.05, 0.025, 0.01)
RESTART_HOURS = (0, 1, 3, 6, 12)


def load_dense_npz(path):
    """train.npz / test_dense.npz -> normalized C, raw C_raw, N_log, peak."""
    d = np.load(path, allow_pickle=True)
    c = d["c"][:, :, 0].astype(float)
    peak = d["peak"].astype(float)
    return {"C_norm": c, "C_raw": c * peak[:, None],
            "N_log": d["N_log"].astype(float), "post_ids": d["post_ids"],
            "peak": peak}


def compute_velocity_dense(C_raw):
    """v_i = log1p(raw at hour 1) - log1p(raw at hour 0)."""
    return np.log1p(C_raw[:, 1]) - np.log1p(C_raw[:, 0])


def no_decline_mask(C):
    return (np.diff(C, axis=1) >= 0.0).all(axis=1)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    os.makedirs(RESULTS, exist_ok=True)

    train = load_dense_npz(os.path.join(DATA_DIR, "train.npz"))
    test_dense = load_dense_npz(os.path.join(DATA_DIR, "test_dense.npz"))
    test_v_all = compute_velocity_dense(test_dense["C_raw"])

    m_tr = no_decline_mask(train["C_norm"])
    m_te = no_decline_mask(test_dense["C_norm"])
    tr = {k: (v[m_tr] if isinstance(v, np.ndarray) and len(v) == len(m_tr)
              else v) for k, v in train.items()}
    te = {k: (v[m_te] if isinstance(v, np.ndarray) and len(v) == len(m_te)
              else v) for k, v in test_dense.items()}
    te_v = test_v_all[m_te]
    print(f"train {len(m_tr)} -> {int(m_tr.sum())} no-decline; "
          f"held-out test_dense {len(m_te)} -> {int(m_te.sum())}")

    models = []
    for m in B.all_baselines():
        m.fit(tr["C_norm"], tr["N_log"])
        models.append(m)

    pinns = []
    stitched = StitchedPINN("teacher-student 3h burst 0-3/3-9/9-16/16-23",
                            STITCHED_DIR, EKM.NUM_TIMESTEPS, EKM.H_NORM,
                            velocity_on=True)
    print("stitched segment betas:", [f"{b:.4f}" for b in stitched.betas])
    pinns.append(stitched)
    if os.path.exists(SINGLE_CKPT):
        pinns.append(PINNCheckpoint("single-window baseline", SINGLE_CKPT,
                                    EKM.NUM_TIMESTEPS, EKM.H_NORM,
                                    velocity_on=True))
    else:
        print(f"[note] single-window checkpoint not found at {SINGLE_CKPT} "
              f"(train 02_baseline_pinn first); scoring stitched only")

    # ---- TRACK A ----------------------------------------------------------
    a_rows, a_head = [], {}
    for split_label, Cn, Nn, Vn in (
            ("test_dense_nodecline", te["C_norm"], te["N_log"], te_v),
            ("test_dense_full", test_dense["C_norm"], test_dense["N_log"],
             test_v_all)):
        print(f"== TRACK A {split_label}")
        for m in models + pinns:
            needs_v = getattr(m, "needs_velocity", lambda: False)()
            d = EKM.score_dense_full(Cn, Nn, m.predict_fn(),
                                     thresholds=THRESHOLDS,
                                     restart_hours=RESTART_HOURS,
                                     V=Vn if needs_v else None)
            a_head.setdefault(m.name, {"is_pinn": m in pinns})[split_label] = {
                thr: d["h0"].get(thr, float("nan")) for thr in THRESHOLDS}
            for k in RESTART_HOURS:
                row = d["per_hour"].get(k)
                if row is not None:
                    out = {"model": m.name, "split": split_label,
                           "restart_hour": k, "n": d["n"]}
                    out.update({f"pct_below_{t}": p for t, p in row.items()})
                    a_rows.append(out)
            print(f"  {m.name:<58} h0<0.10="
                  f"{a_head[m.name][split_label][0.10]:5.1f}%", flush=True)

    # ---- TRACK B ----------------------------------------------------------
    log_h, nlp, hf, lfp = ELF.build_scale_panel(tr["C_raw"], tr["N_log"])
    scale_model = ELF.RidgeScaleModel(l2=1.0).fit(log_h, nlp, hf, lfp)
    cold = ELF.ColdStartScaleModel(l2=1.0).fit(
        tr["N_log"], np.log1p(tr["C_raw"][:, -1]))
    forecasters = [ELF.PersistenceForecaster(),
                   ELF.FollowerRegressionForecaster(scale_model, cold)]
    for m in models + pinns:
        forecasters.append(ELF.TwoStageForecaster(
            f"Two-stage: {m.name}", m, scale_model, cold))

    b_rows, b_head = [], {}
    print("== TRACK B test_dense_nodecline")
    for fc in forecasters:
        sm = getattr(fc, "shape_model", None)
        needs_v = sm is not None and getattr(sm, "needs_velocity",
                                             lambda: False)()
        b_head[fc.name] = {"k": {}, "is_pinn": sm is not None and sm in pinns}
        for k in RESTART_HOURS:
            sc = ELF.evaluate_dense(fc, te["C_raw"], te["N_log"], THRESHOLDS,
                                    k_start=k, V=te_v if needs_v else None)
            out = {"model": fc.name, "split": "test_dense_nodecline",
                   "restart_hour": k, "n": sc["n"]}
            for t in THRESHOLDS:
                out[f"pct_below_{t}"] = sc.get(f"traj_mse_<{t}", float("nan"))
            out["mean_log_traj_mse"] = sc["mean_log_traj_mse"]
            b_rows.append(out)
            b_head[fc.name]["k"][k] = {
                t: sc.get(f"traj_mse_<{t}", float("nan")) for t in THRESHOLDS}
        h = b_head[fc.name]["k"]
        print(f"  {fc.name:<66} k0={h[0][0.10]:4.1f}%  k6={h[6][0.10]:4.1f}%  "
              f"k12={h[12][0.10]:4.1f}%", flush=True)

    write_csv(os.path.join(RESULTS, "track_a_final.csv"), a_rows)
    write_csv(os.path.join(RESULTS, "track_b_final.csv"), b_rows)

    # ---- leaderboard -------------------------------------------------------
    lines = ["# Final leaderboard -- PINNs vs classical baselines "
             "(held-out test set)", ""]
    lines.append(f"Stitched segment betas: "
                 f"{', '.join(f'{b:.3f}' for b in stitched.betas)} on bounds "
                 f"{stitched.seg_bounds}. Classical baselines + Track B scale "
                 f"models fit on the {int(m_tr.sum())}-tweet no-decline train "
                 f"subset; test_dense scored no-decline (n={int(m_te.sum())}) "
                 f"and full (n={len(m_te)}).")
    lines.append("")
    for split in ("test_dense_nodecline", "test_dense_full"):
        lines.append(f"## Track A -- shape only, {split}")
        lines.append("")
        lines.append("| Model | <0.10 | <0.05 | <0.025 | <0.01 |")
        lines.append("|---|---|---|---|---|")
        for name, d in sorted(a_head.items(),
                              key=lambda kv: -kv[1][split][0.10]):
            t = d[split]
            tag = "**PINN**" if d["is_pinn"] else ""
            lines.append(f"| {tag} {name} | {t[0.10]:.1f}% | {t[0.05]:.1f}% "
                         f"| {t[0.025]:.1f}% | {t[0.01]:.1f}% |")
        lines.append("")
    lines.append("## Track B -- leak-free, test_dense_nodecline, <0.10 by "
                 "restart hour")
    lines.append("")
    lines.append("| Model | " + " | ".join(f"k={k}" for k in RESTART_HOURS)
                 + " |")
    lines.append("|---|" + "---|" * len(RESTART_HOURS))
    for name, d in b_head.items():
        tag = "**PINN**" if d["is_pinn"] else ""
        cells = [f"{d['k'][k][0.10]:.0f}" for k in RESTART_HOURS]
        lines.append(f"| {tag} {name} | " + " | ".join(cells) + " |")
    lines.append("")

    cl = {n: d["test_dense_nodecline"][0.10] for n, d in a_head.items()
          if not d["is_pinn"]}
    pv = {n: d["test_dense_nodecline"][0.10] for n, d in a_head.items()
          if d["is_pinn"]}
    bc, bp = max(cl, key=cl.get), max(pv, key=pv.get)
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Track A (no-decline test_dense, <0.10): best classical = "
                 f"{bc} ({cl[bc]:.1f}%); best PINN = {bp} ({pv[bp]:.1f}%).")
    with open(os.path.join(RESULTS, "FINAL_LEADERBOARD.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote results/FINAL_LEADERBOARD.md")


if __name__ == "__main__":
    main()
