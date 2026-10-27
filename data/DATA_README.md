# Real-world dataset: 24h window, T = 25, 80/20 dense split

Built 2026-07-12 by build_dataset.py (deterministic; seed 42). Source: `baseline/final_training_data/` (jan31 scrape).

| set | tweets | role |
|---|---|---|
| train | 403 | model fitting (complete 24h grids) |
| test-dense | 100 | held-out complete trajectories: cold-start + closure eval |
| test-sparse | 333 | never-trainable partial trajectories: multi-horizon eval |
| **test total** | **433** | |

## Training-set characteristics (403 tweets, 24 hourly nodes)

| characteristic | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| peak total engagement | 28 | 39 | 92 | 249 | 696 |
| quantization sigma_eff (0.58/peak) | 0.0008 | 0.0023 | 0.0063 | 0.0149 | 0.0207 |
| plateau share (flat hourly steps) | 0% | 4% | 30% | 52% | 70% |
| largest single-hour jump / total | 9% | 11% | 15% | 22% | 31% |
| author followers | 321 | 1,866 | 12,146 | 51,929 | 263,947 |
| c(0) (normalized engagement at hour 0) | 0.02 | 0.05 | 0.13 | 0.27 | 0.51 |

- interpolated tweets (had 1-2h gaps filled): 286 (71%)
- peak range: 25 to 17189; followers range: 1 to 36,598,675
- every trajectory: complete hours 0-23, max-normalized (ends at its own peak = 1.0)

## Test-set composition

- test-dense: 100 complete trajectories (all hour-0-seedable), peak median 90 -- stratified to match the train peak distribution
- test-sparse: 333 partial trajectories, median 14 of 24 hours observed; 58 have hour 0, 112 have hour 1
- hour-0-seedable test tweets total: 158

Files: `train.npz`, `test_dense.npz`, `test_sparse.npz`, `split_manifest.json` (frozen id lists). Rebuild with `python3 build_dataset.py`.
