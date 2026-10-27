# Fractional PINNs for social media virality

Code for *Fractional physics-informed neural networks to model social media
virality* (A. Koong & G. E. Karniadakis). Each post's 24-hour engagement
curve is modelled by a Caputo fractional ODE, D^β c = α f(c). A PINN learns
β, a per-post rate α, and the reaction law f. PySR then distills f into a
closed-form equation.

## Layout

| Folder | Contents | Main entry point |
|---|---|---|
| `01_synthetic_validation_24h/` | Synthetic recovery of β and f(c) at N = 24 nodes; the solver is checked against exact Mittag-Leffler solutions, and the residual losses are compared (whitened L1 is the one adopted) | `verify_24h.py`, `make_fig_whitened_beta.py` |
| `02_baseline_pinn/` | Single-window PINN (whitened L1 + velocity α head) and its symbolic f(c) | `train_virality.py --velocity on --no-decline`, `symbolic_regression.py` |
| `03_teacher_student_3h_burst/` | Teacher–student PINNs on the 0–3 / 3–9 / 9–16 / 16–23 h windows, stitched by hour | `train_segments.py`, `symbolic_segments.py` |
| `04_real_world_testing/` | Held-out evaluation against 11 classical baselines (growth curves, Hawkes processes, persistence, climatology) | `run_symbolic_eval.py`, `run_final_eval.py` |
| `data/` | Processed dataset: 403 train / 100 test-dense / 333 test-sparse posts, with the frozen split | see `data/DATA_README.md` |

Trained checkpoints and result reports are already in each `results/` folder,
so the evaluations run without retraining.

## Quick start

```bash
pip install -r requirements.txt   # Python 3.9, CPU-only JAX; PySR installs Julia on first use
cd 04_real_world_testing
python3 run_symbolic_eval.py      # paper's accuracies + Table 6 -> results/SYMBOLIC_EVAL.md (seconds)
python3 run_final_eval.py         # NN-f rollouts, Track A/B leaderboard -> results/FINAL_LEADERBOARD.md
```

Retraining from scratch (stages 1–3, seeds fixed) rewrites the shipped
`results/` folders: stage 1 takes ~1–2 h, stages 2 and 3 take ~5 min each.
PySR may print expression strings that differ cosmetically between runs.

## Headline result

This is the % of held-out posts (no-decline test set, n = 86) whose trajectory MSE falls below
each tolerance, forward-solved from c(0) (paper Table 6):

| Model | < 0.025 | < 0.01 |
|---|---|---|
| **Teacher–student PINN (symbolic laws)** | **81.4%** | **64.0%** |
| Bass diffusion | 75.6% | 44.2% |
| Gompertz | 70.9% | 34.9% |
| Climatology (mean curve) | 66.3% | 33.7% |

The fitted fractional order goes from β = 0.23 in the 0–3 h burst to 0.74 over 3–9 h,
then 0.99 during relaxation.

**Note:** the paper's PINN numbers are forward solves of the *distilled
symbolic laws*, using the full-precision PySR coefficients
(`run_symbolic_eval.py`). `run_final_eval.py` and the stage-3 leaderboard
roll out the *neural* f instead, so their PINN rows differ (e.g. 79.1% / 66.3%).
The coefficients rounded for print in the paper also shift the scores
slightly, and `SYMBOLIC_EVAL.md` reports all three variants.

## Data

The engagement metric is likes + retweets + replies + quotes. It is sampled at 24 hourly
nodes and normalized by each post's own peak, so raw counts are `c * peak`. Posts that
ever decline are dropped at load time (train 403 → 348, test 100 → 86).
The raw January 2026 X/Twitter scrape used by `data/build_dataset.py` is not
included.
