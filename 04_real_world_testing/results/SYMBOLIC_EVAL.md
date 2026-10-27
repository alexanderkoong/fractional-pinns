# Symbolic-embedded Track A (the paper's evaluation)

Forward solves use the DISTILLED PySR laws (full precision) in place of the neural f; alpha/beta from the fitted checkpoints; protocol identical to run_final_eval.py (h0 restart, clip01 L1 rollout, MSE over hours 1..23). See the module docstring of run_symbolic_eval.py for sources.

## train_nodecline (n=348)

| Model | <0.10 | <0.05 | <0.025 | <0.01 |
|---|---|---|---|---|
| TS 3h burst -- SYMBOLIC (paper numbers) | 97.70% | 94.83% | 87.64% | 72.13% |
| TS 3h burst -- NN f (run_final_eval.py) | 95.40% | 91.95% | 85.63% | 69.54% |
| TS 3h burst -- symbolic as rounded in paper Table 4 | 97.70% | 95.11% | 90.23% | 72.70% |
| single window -- SYMBOLIC (paper numbers) | 65.80% | 55.46% | 41.38% | 17.53% |
| single window -- NN f (run_final_eval.py) | 59.20% | 51.15% | 38.22% | 17.82% |
| single window -- symbolic as rounded in paper Eq. (14) | 65.80% | 55.75% | 41.38% | 17.82% |
| Persistence (c=c0) | 3.74% | 1.44% | 0.86% | 0.86% |
| Linear c0->1 | 88.51% | 60.63% | 37.07% | 16.09% |
| Climatology (mean curve) | 95.11% | 83.05% | 63.51% | 39.66% |
| Logistic (global r) | 94.54% | 83.62% | 60.92% | 27.87% |
| Logistic (follower-scaled r) | 94.54% | 83.62% | 60.92% | 27.87% |
| Gompertz (global b) | 97.41% | 88.22% | 74.14% | 43.68% |
| Gompertz (follower-scaled b) | 97.41% | 88.22% | 74.14% | 43.68% |
| Bass diffusion (p,q) | 96.84% | 89.37% | 76.15% | 49.14% |
| Hawkes (exponential kernel) | 41.67% | 34.77% | 27.01% | 15.80% |
| Hawkes (power-law kernel) | 37.36% | 26.72% | 20.69% | 15.23% |
| Hawkes (learned kernel, NNLS) | 81.03% | 55.75% | 39.37% | 24.14% |

## test_dense_nodecline (n=86)

| Model | <0.10 | <0.05 | <0.025 | <0.01 |
|---|---|---|---|---|
| TS 3h burst -- SYMBOLIC (paper numbers) | 100.00% | 91.86% | 81.40% | 63.95% |
| TS 3h burst -- NN f (run_final_eval.py) | 96.51% | 87.21% | 79.07% | 66.28% |
| TS 3h burst -- symbolic as rounded in paper Table 4 | 100.00% | 96.51% | 84.88% | 63.95% |
| single window -- SYMBOLIC (paper numbers) | 63.95% | 53.49% | 39.53% | 10.47% |
| single window -- NN f (run_final_eval.py) | 59.30% | 52.33% | 40.70% | 12.79% |
| single window -- symbolic as rounded in paper Eq. (14) | 63.95% | 53.49% | 41.86% | 11.63% |
| Persistence (c=c0) | 4.65% | 2.33% | 0.00% | 0.00% |
| Linear c0->1 | 86.05% | 63.95% | 41.86% | 13.95% |
| Climatology (mean curve) | 100.00% | 83.72% | 66.28% | 33.72% |
| Logistic (global r) | 95.35% | 81.40% | 54.65% | 26.74% |
| Logistic (follower-scaled r) | 95.35% | 81.40% | 54.65% | 26.74% |
| Gompertz (global b) | 98.84% | 90.70% | 70.93% | 34.88% |
| Gompertz (follower-scaled b) | 98.84% | 90.70% | 70.93% | 34.88% |
| Bass diffusion (p,q) | 98.84% | 95.35% | 75.58% | 44.19% |
| Hawkes (exponential kernel) | 45.35% | 39.53% | 26.74% | 12.79% |
| Hawkes (power-law kernel) | 41.86% | 27.91% | 22.09% | 13.95% |
| Hawkes (learned kernel, NNLS) | 77.91% | 55.81% | 36.05% | 20.93% |

## test_dense_full (n=100)

| Model | <0.10 | <0.05 | <0.025 | <0.01 |
|---|---|---|---|---|
| TS 3h burst -- SYMBOLIC (paper numbers) | 98.00% | 91.00% | 82.00% | 66.00% |
| TS 3h burst -- NN f (run_final_eval.py) | 96.00% | 87.00% | 80.00% | 68.00% |
| TS 3h burst -- symbolic as rounded in paper Table 4 | 99.00% | 95.00% | 85.00% | 66.00% |
| single window -- SYMBOLIC (paper numbers) | 66.00% | 56.00% | 40.00% | 12.00% |
| single window -- NN f (run_final_eval.py) | 62.00% | 55.00% | 41.00% | 14.00% |
| single window -- symbolic as rounded in paper Eq. (14) | 66.00% | 56.00% | 42.00% | 13.00% |
| Persistence (c=c0) | 8.00% | 6.00% | 3.00% | 3.00% |
| Linear c0->1 | 83.00% | 63.00% | 42.00% | 17.00% |
| Climatology (mean curve) | 99.00% | 79.00% | 62.00% | 33.00% |
| Logistic (global r) | 94.00% | 81.00% | 55.00% | 29.00% |
| Logistic (follower-scaled r) | 94.00% | 81.00% | 55.00% | 29.00% |
| Gompertz (global b) | 97.00% | 89.00% | 70.00% | 37.00% |
| Gompertz (follower-scaled b) | 97.00% | 89.00% | 70.00% | 37.00% |
| Bass diffusion (p,q) | 97.00% | 92.00% | 74.00% | 46.00% |
| Hawkes (exponential kernel) | 48.00% | 43.00% | 30.00% | 16.00% |
| Hawkes (power-law kernel) | 45.00% | 33.00% | 27.00% | 19.00% |
| Hawkes (learned kernel, NNLS) | 77.00% | 57.00% | 39.00% | 25.00% |

## Table 6 -- seen (train no-decline, n=348) vs unseen (test_dense no-decline, n=86)

| Model | seen <0.025 | seen <0.01 | unseen <0.025 | unseen <0.01 |
|---|---|---|---|---|
| TS 3h burst -- SYMBOLIC (paper numbers) | 87.6% | 72.1% | 81.4% | 64.0% |
| Persistence (c=c0) | 0.9% | 0.9% | 0.0% | 0.0% |
| Linear c0->1 | 37.1% | 16.1% | 41.9% | 14.0% |
| Climatology (mean curve) | 63.5% | 39.7% | 66.3% | 33.7% |
| Logistic (global r) | 60.9% | 27.9% | 54.7% | 26.7% |
| Logistic (follower-scaled r) | 60.9% | 27.9% | 54.7% | 26.7% |
| Gompertz (global b) | 74.1% | 43.7% | 70.9% | 34.9% |
| Gompertz (follower-scaled b) | 74.1% | 43.7% | 70.9% | 34.9% |
| Bass diffusion (p,q) | 76.1% | 49.1% | 75.6% | 44.2% |
| Hawkes (exponential kernel) | 27.0% | 15.8% | 26.7% | 12.8% |
| Hawkes (power-law kernel) | 20.7% | 15.2% | 22.1% | 14.0% |
| Hawkes (learned kernel, NNLS) | 39.4% | 24.1% | 36.0% | 20.9% |

