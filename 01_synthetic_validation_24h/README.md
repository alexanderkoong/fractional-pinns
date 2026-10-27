# 24-Hour Model Verification

The real-world dataset now covers **24 hourly observations** per post
instead of 48. This folder reruns the parameter-recovery validation at
N = 24 nodes on the rescaled interval t in [0, 1] (grid spacing h = 1/23).
Halving the observation count matters for two reasons: less information per
trajectory, and a coarser grid — the L1 scheme's initial-layer error lives
at fixed *node index* (it is resolution-independent per node), so the
corrupted early nodes now occupy a larger fraction of the window.

Scope (per the current experiment plan — nothing else):

1. **Beta alone** (f, alpha at truth): exact landscape argmin — clean data
   at true beta in {0.3, 0.5, 0.7}, perturbed data at
   sigma in {0.005, 0.02, 0.05}.
2. **Joint (beta, f), clean**: 2 beta-inits x 2 net seeds per objective,
   PySR on each objective's best run.
3. **Joint (beta, f), perturbed**: 3 sigmas x 2 noise draws, per-objective
   cheat ratio L(learned)/L(true params), PySR at sigma = 0.02.

Objectives: `l1_full` (the paper's Eq. 4, control), `l1_burn` (early-node
burn-in), `volterra` (integral-form reference), `l1_whitened` (data-space
whitened L1 — the errors-in-variables ML loss developed in
`../L1 with log term/`, which removes the noise cheat while keeping the L1
scheme).

Everything else is identical to the 48-hour suites for comparability: same
ground truth `D^0.5 c = c(1-c^2)`, alpha = 1 fixed, same 200-trajectory c0
draw (seed 42), numerically exact data (corrected ABM solver, fine-grid
solve + subsample), same production f-network / beta parameterization /
training budget, same noise seeds. N = 48 numbers quoted in the report are
citations from the earlier folders, not reruns.

## Files

* `verify_24h.py` — the whole suite (checkpointed under
  `results/checkpoints/`; rerunning resumes).
* `whitened_l1.py` — data-space whitened L1 loss, duplicated/adapted from
  `../L1 with log term/l1_log_term_poc.py` (winning variant only).
* `fractional_ode.py`, `pinn_lib.py` — byte-identical duplicates of
  `../model validation/`'s numerics libraries (self-contained folder).
* `results/verify_24h.txt` — full report; `results/verify_24h_summary.png`.

## Results (full tables in `results/verify_24h.txt`)

**Clean data, N = 24.** Beta remains identifiable and accurately
recoverable at 24 points. Beta-only argmin errors across true beta in
{0.3, 0.5, 0.7}: l1_full −0.022..−0.027 (the coarser grid roughly doubles
its N = 48 bias, as the fixed-node-index layer error predicts), l1_burn
<= 0.005, volterra <= 0.004, whitened <= 0.020. Clean joint recovery
(true beta 0.5; N = 48 references in parentheses):

| objective | joint beta_hat | f corr | PySR on best run |
|---|---|---|---|
| l1_full (paper Eq. 4) | 0.4574 (0.4729) | 0.9978 | exp-form drift, corr 0.9977 |
| l1_burn | 0.4945 (0.4967) | 1.0000 | near-cubic, corr 0.99998 |
| volterra | 0.4974 (0.4991) | 1.0000 | **`c(0.9998 − c^2)` — exact law** |
| l1_whitened | 0.4872 (0.5026) | 0.9998 | cubic, roots ~{0, 1.0014, −0.94}, corr 0.9998 |

All objectives remain init/seed-consistent (spread <= 4e-4).

**Perturbed data, N = 24** (joint recovery, mean over 2 noise draws;
cheat = L(learned)/L(true params) under each objective):

| sigma | l1_full | volterra | l1_whitened | whitened cheat |
|-------|---------|----------|-------------|----------------|
| 0.005 | 0.381 (cheats) | 0.497 | 0.488 | 0.95 |
| 0.02  | **0.103 (pinned)** | 0.475 | **0.490** | 0.99 |
| 0.05  | 0.100 | 0.17 (breaks) | 0.402 | 0.97 |

* The paper's raw L1 objective cheats at N = 24 exactly as at N = 48
  (beta pinned at the 0.1 bound by sigma = 0.02, cheat ratio down to 0.06).
* **The whitened L1 is the strongest objective under noise at N = 24**: at
  sigma = 0.02 it recovers beta to −0.010 with f corr 0.9986 and PySR
  returning `0.98 c (1 − c)(1 + 1.04 c)` (corr 0.99998) — beating volterra
  (−0.025) while keeping the L1 scheme. At sigma = 0.05 — where volterra
  collapses to ~0.17 — whitened degrades honestly to ~0.40 (cheat 0.97):
  biased by the flexible-f variance channel, but nowhere near the bound.
* Beta-only landscapes under noise (Part 1b) confirm the estimator-level
  ranking: whitened argmin moves only −0.010 → −0.018 from sigma 0 → 0.05,
  volterra −0.002 → −0.040, while raw/burn-in collapse to the bound.

**Bottom line: halving the observation window to 24 hourly points does not
break parameter recovery.** Clean-data recovery passes for every objective
except the raw pointwise L1 (whose known bias roughly doubles), and the
noise robustness ranking from the 48h studies is preserved — with the
whitened L1 emerging as the best noisy-data objective at this resolution.

## Reproducing

```
pip install -r ../requirements.txt
python3 verify_24h.py     # ~1.5-2 h on a laptop CPU (34 training runs + 6 PySR fits)
```
