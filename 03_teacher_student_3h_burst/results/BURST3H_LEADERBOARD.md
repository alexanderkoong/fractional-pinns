# 3h-burst vs uniform 6h segmentation -- leaderboard

3h-burst segment betas: 0.228, 0.744, 0.989, 0.989 on bounds [(0, 3), (3, 9), (9, 16), (16, 23)]. Same training recipe/data as the 4x6h model; only the window layout differs.

## Track A -- shape only, test_dense_nodecline

| Model | <0.10 | <0.05 | <0.01 |
|---|---|---|---|
|  Climatology (mean curve) | 100.0% | 83.7% | 33.7% |
|  Gompertz (global b) | 98.8% | 90.7% | 34.9% |
|  Gompertz (follower-scaled b) | 98.8% | 90.7% | 34.9% |
|  Bass diffusion (p,q) | 98.8% | 95.3% | 44.2% |
| **PINN** PINN (teacher-student 3h burst 0-3/3-9/9-16/16-23 +velocity) | 96.5% | 87.2% | 66.3% |
|  Logistic (global r) | 95.3% | 81.4% | 26.7% |
|  Logistic (follower-scaled r) | 95.3% | 81.4% | 26.7% |
| **PINN** PINN (teacher-student 4x6h uniform +velocity) | 93.0% | 81.4% | 62.8% |
|  Linear c0->1 | 86.0% | 64.0% | 14.0% |
|  Hawkes (learned kernel, NNLS) | 77.9% | 55.8% | 20.9% |
| **PINN** PINN (valpha ON /348-no-decline single +velocity) | 59.3% | 52.3% | 12.8% |
|  Hawkes (exponential kernel) | 45.3% | 39.5% | 12.8% |
|  Hawkes (power-law kernel) | 41.9% | 27.9% | 14.0% |
|  Persistence (c=c0) | 4.7% | 2.3% | 0.0% |

## Track A -- shape only, test_dense_full

| Model | <0.10 | <0.05 | <0.01 |
|---|---|---|---|
|  Climatology (mean curve) | 99.0% | 79.0% | 33.0% |
|  Gompertz (global b) | 97.0% | 89.0% | 37.0% |
|  Gompertz (follower-scaled b) | 97.0% | 89.0% | 37.0% |
|  Bass diffusion (p,q) | 97.0% | 92.0% | 46.0% |
| **PINN** PINN (teacher-student 3h burst 0-3/3-9/9-16/16-23 +velocity) | 96.0% | 87.0% | 68.0% |
|  Logistic (global r) | 94.0% | 81.0% | 29.0% |
|  Logistic (follower-scaled r) | 94.0% | 81.0% | 29.0% |
| **PINN** PINN (teacher-student 4x6h uniform +velocity) | 92.0% | 82.0% | 64.0% |
|  Linear c0->1 | 83.0% | 63.0% | 17.0% |
|  Hawkes (learned kernel, NNLS) | 77.0% | 57.0% | 25.0% |
| **PINN** PINN (valpha ON /348-no-decline single +velocity) | 62.0% | 55.0% | 14.0% |
|  Hawkes (exponential kernel) | 48.0% | 43.0% | 16.0% |
|  Hawkes (power-law kernel) | 45.0% | 33.0% | 19.0% |
|  Persistence (c=c0) | 8.0% | 6.0% | 3.0% |

## Track B -- leak-free, test_dense_nodecline, <0.10 by restart hour

| Model | k=0 | k=1 | k=3 | k=6 | k=12 |
|---|---|---|---|---|---|
|  Persistence (raw, c=seed) | 0 | 3 | 21 | 52 | 90 |
|  Follower-adjusted regression (scale only) | 5 | 16 | 5 | 8 | 33 |
|  Two-stage: Persistence (c=c0) | 0 | 3 | 21 | 52 | 90 |
|  Two-stage: Linear c0->1 | 0 | 16 | 47 | 45 | 92 |
|  Two-stage: Climatology (mean curve) | 1 | 22 | 38 | 26 | 72 |
|  Two-stage: Logistic (global r) | 0 | 29 | 23 | 22 | 63 |
|  Two-stage: Logistic (follower-scaled r) | 0 | 29 | 23 | 22 | 63 |
|  Two-stage: Gompertz (global b) | 0 | 28 | 40 | 28 | 83 |
|  Two-stage: Gompertz (follower-scaled b) | 0 | 28 | 40 | 28 | 83 |
|  Two-stage: Bass diffusion (p,q) | 2 | 23 | 41 | 35 | 94 |
|  Two-stage: Hawkes (exponential kernel) | 0 | 23 | 30 | 20 | 45 |
|  Two-stage: Hawkes (power-law kernel) | 0 | 17 | 35 | 21 | 45 |
|  Two-stage: Hawkes (learned kernel, NNLS) | 0 | 23 | 35 | 20 | 36 |
| **PINN** Two-stage: PINN (teacher-student 3h burst 0-3/3-9/9-16/16-23 +velocity) | 0 | 29 | 24 | 36 | 91 |
| **PINN** Two-stage: PINN (teacher-student 4x6h uniform +velocity) | 0 | 29 | 24 | 23 | 91 |
| **PINN** Two-stage: PINN (valpha ON /348-no-decline single +velocity) | 0 | 26 | 15 | 15 | 76 |

## Verdict

Track A (no-decline test_dense, <0.10): best classical = Climatology (mean curve) (100.0%); best PINN = PINN (teacher-student 3h burst 0-3/3-9/9-16/16-23 +velocity) (96.5%). classical still wins.
