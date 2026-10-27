#!/usr/bin/env python3
"""
Build the filtered real-world dataset: 24-hour window, T = 25 staircase filter,
stratified 80/20 dense split.

Deterministic end-to-end: rerunning this script from the raw CSVs reproduces
byte-identical splits (fixed RNG seed, stable orderings). All outputs land in
this folder.

DEFINITIONS (decided 2026-07-12)
--------------------------------
Window      : hour buckets 0..23 (24 hourly nodes, t rescaled to [0,1],
              h = 1/23). Motivation: the median tweet reaches 100% of its
              48-hour engagement by hour 23; hours 24-47 are almost pure
              plateau and actively conflict with the strictly-positive f.
Filter      : keep a tweet only if its raw TOTAL engagement (likes + RTs +
              replies + quotes) reaches at least T = 25 inside the window.
              One engagement then moves a normalized curve by <= 4%;
              quantization noise sigma_eff ~= 0.58/T <= 0.023, at the edge
              of the sigma <= 0.02 regime where the whitened-L1 estimator
              was validated on synthetic data.
DENSE pool  : tweets passing the training loader (hour 0 present, gaps <= 2h
              interpolated, non-flat) AND the T filter. These are the only
              trainable tweets (complete uniform grid).
Split       : stratified 80/20 of the dense pool into TRAIN / TEST-DENSE.
              Strata = peak-engagement deciles, so both sides span small to
              viral. RNG: numpy RandomState(42), applied within strata over
              a peak-sorted, post_id-tiebroken ordering.
SPARSE pool : TEST-only. Tweets NOT in the dense pool, with >= 1 in-window
              snapshot, growing (max > min), peak >= T, and follower data --
              mirrors load_sparse_unseen() from the filtered eval pipeline.
              Variable hourly coverage; unusable for training by construction.

OUTPUTS
-------
  train.npz        t (Ntr,24,1), c (Ntr,24,1), N_log (Ntr,), post_ids,
                   peak (raw), interpolated (bool: any gap-filled hours)
  test_dense.npz   same fields for the held-out dense tweets
  test_sparse.npz  object arrays: post_ids, hours (int arrays),
                   engagement (normalized float arrays), N_log, peak,
                   has_h0, has_h1
  split_manifest.json   filters, seed, counts, and the full id lists
  README.md        provenance + summary tables (written by this script)
"""

import json
import os

import numpy as np
import pandas as pd

from load_tweets import load_tweet_data_robust, load_follower_counts

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/alexkoong/Desktop/jmlmc code/baseline/final_training_data"

END_HOUR = 23          # window = hour buckets 0..23 (24 nodes)
MIN_PEAK = 25          # T: staircase filter on raw total engagement
TEST_FRAC = 0.20       # dense holdout share
SPLIT_SEED = 42
N_STRATA = 10          # peak-decile stratification
BUILD_DATE = "2026-07-12"


def build():
    snapshots = os.path.join(DATA_DIR, "snapshots.csv")
    posts = os.path.join(DATA_DIR, "posts.csv")
    authors = os.path.join(DATA_DIR, "authors.csv")

    # ---------------- raw peaks within the window ----------------
    df = pd.read_csv(snapshots)
    df["total"] = df["likes"] + df["retweets"] + df["replies"] + df["quotes"]
    df["bucket"] = df["hours_since_post"].astype(int)
    dfw = df[(df.bucket >= 0) & (df.bucket <= END_HOUR)]
    peak_lookup = dfw.groupby("post_id")["total"].max().to_dict()

    # ---------------- DENSE pool (trainable) ----------------
    t, c, ids, stats = load_tweet_data_robust(
        snapshots, metric="total", start_hour=0, end_hour=END_HOUR,
        max_gap_to_interpolate=2, normalize=True, verbose=False)
    ids = [int(p) for p in ids]
    interp_ids = set(int(k) for k in stats.interpolation_details)

    peak = np.array([peak_lookup.get(p, 0) for p in ids], dtype=float)
    keep = peak >= MIN_PEAK
    t, c = t[keep], c[keep]
    dense_ids = [p for p, k in zip(ids, keep) if k]
    peak = peak[keep]
    interpolated = np.array([p in interp_ids for p in dense_ids])

    N_log = load_follower_counts(posts, authors, dense_ids, verbose=False)

    # ---------------- stratified 80/20 split ----------------
    order = np.lexsort((np.array(dense_ids), peak))   # peak asc, id tiebreak
    strata = np.array_split(order, N_STRATA)
    rng = np.random.RandomState(SPLIT_SEED)
    test_idx = []
    for s in strata:
        n_test = int(round(TEST_FRAC * len(s)))
        test_idx.extend(rng.permutation(s)[:n_test])
    test_mask = np.zeros(len(dense_ids), dtype=bool)
    test_mask[np.array(test_idx, dtype=int)] = True

    def pack(mask):
        return dict(
            t=t[mask], c=c[mask],
            N_log=N_log[mask],
            post_ids=np.array([p for p, m in zip(dense_ids, mask) if m],
                              dtype=np.int64),
            peak=peak[mask], interpolated=interpolated[mask],
        )

    train = pack(~test_mask)
    test_dense = pack(test_mask)

    # ---------------- SPARSE pool (test-only) ----------------
    posts_df = pd.read_csv(posts, usecols=["post_id", "author_id"])
    authors_df = pd.read_csv(authors, usecols=["author_id", "followers_count"])
    follower = (posts_df.merge(authors_df, on="author_id", how="left")
                .set_index("post_id")["followers_count"].to_dict())

    dense_set = set(dense_ids)
    sp_ids, sp_hours, sp_eng, sp_nlog, sp_peak = [], [], [], [], []
    for pid, grp in dfw.groupby("post_id"):
        if pid in dense_set:
            continue
        grp = (grp.sort_values("hours_since_post")
               .drop_duplicates("bucket", keep="first")
               .sort_values("bucket"))
        tot = grp["total"].values.astype(float)
        if tot.size == 0 or tot.max() <= tot.min():
            continue                                    # must be growing
        if tot.max() < MIN_PEAK:
            continue                                    # staircase filter
        fc = follower.get(pid, np.nan)
        if fc is None or (isinstance(fc, float) and np.isnan(fc)):
            continue                                    # needs follower data
        sp_ids.append(int(pid))
        sp_hours.append(grp["bucket"].values.astype(int))
        sp_eng.append(tot / tot.max())
        sp_nlog.append(float(np.log(fc + 1)))
        sp_peak.append(float(tot.max()))

    has_h0 = np.array([0 in set(h) for h in sp_hours])
    has_h1 = np.array([1 in set(h) for h in sp_hours])

    # ---------------- self-checks ----------------
    n_dense = len(dense_ids)
    assert train["c"].shape[1] == END_HOUR + 1 == 24
    assert len(train["post_ids"]) + len(test_dense["post_ids"]) == n_dense
    tr_set = set(train["post_ids"].tolist())
    td_set = set(test_dense["post_ids"].tolist())
    sp_set = set(sp_ids)
    assert not (tr_set & td_set), "train/test-dense overlap"
    assert not (tr_set & sp_set), "train/sparse overlap"
    assert not (td_set & sp_set), "test-dense/sparse overlap"
    for d in (train, test_dense):
        assert d["peak"].min() >= MIN_PEAK
        assert np.all(d["c"] >= 0) and np.all(d["c"] <= 1)
        assert np.allclose(d["c"].max(axis=1), 1.0), "max-normalization broken"
    assert min(sp_peak) >= MIN_PEAK
    print(f"[self-check] dense {n_dense} = train {len(tr_set)} + "
          f"test-dense {len(td_set)}; sparse {len(sp_set)}; "
          f"no overlaps; all peaks >= {MIN_PEAK}; grids complete")

    # ---------------- save ----------------
    np.savez_compressed(os.path.join(HERE, "train.npz"), **train)
    np.savez_compressed(os.path.join(HERE, "test_dense.npz"), **test_dense)
    np.savez_compressed(
        os.path.join(HERE, "test_sparse.npz"),
        post_ids=np.array(sp_ids, dtype=np.int64),
        hours=np.array(sp_hours, dtype=object),
        engagement=np.array(sp_eng, dtype=object),
        N_log=np.array(sp_nlog), peak=np.array(sp_peak),
        has_h0=has_h0, has_h1=has_h1)

    manifest = dict(
        build_date=BUILD_DATE, source_data=DATA_DIR,
        window_hours=[0, END_HOUR], num_nodes=END_HOUR + 1,
        metric="total = likes + retweets + replies + quotes",
        min_peak_engagement=MIN_PEAK, test_frac=TEST_FRAC,
        split_seed=SPLIT_SEED, n_strata=N_STRATA,
        counts=dict(dense=n_dense, train=len(tr_set),
                    test_dense=len(td_set), test_sparse=len(sp_set),
                    test_total=len(td_set) + len(sp_set)),
        train_ids=sorted(tr_set), test_dense_ids=sorted(td_set),
        test_sparse_ids=sorted(sp_set),
    )
    with open(os.path.join(HERE, "split_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    return train, test_dense, dict(peak=np.array(sp_peak), has_h0=has_h0,
                                   has_h1=has_h1,
                                   n_hours=np.array([len(h) for h in sp_hours])), manifest


def summarize(train, test_dense, sparse, manifest):
    """Print + persist the characteristics tables."""
    def pct(x, q):
        return np.percentile(x, q)

    lines = []
    lines.append("# Real-world dataset: 24h window, T = 25, 80/20 dense split")
    lines.append("")
    lines.append(f"Built {manifest['build_date']} by build_dataset.py "
                 f"(deterministic; seed {manifest['split_seed']}). Source: "
                 f"`baseline/final_training_data/` (jan31 scrape).")
    lines.append("")
    lines.append("| set | tweets | role |")
    lines.append("|---|---|---|")
    c = manifest["counts"]
    lines.append(f"| train | {c['train']} | model fitting (complete 24h grids) |")
    lines.append(f"| test-dense | {c['test_dense']} | held-out complete "
                 f"trajectories: cold-start + closure eval |")
    lines.append(f"| test-sparse | {c['test_sparse']} | never-trainable partial "
                 f"trajectories: multi-horizon eval |")
    lines.append(f"| **test total** | **{c['test_total']}** | |")
    lines.append("")

    dc = np.diff(train["c"][:, :, 0], axis=1)
    plateau = (dc == 0).mean(axis=1)
    max_jump = dc.max(axis=1)
    followers = np.exp(train["N_log"]) - 1
    c0 = train["c"][:, 0, 0]
    pk = train["peak"]

    lines.append("## Training-set characteristics "
                 f"({c['train']} tweets, 24 hourly nodes)")
    lines.append("")
    lines.append("| characteristic | p10 | p25 | median | p75 | p90 |")
    lines.append("|---|---|---|---|---|---|")
    rows = [
        ("peak total engagement", pk, "{:.0f}"),
        ("quantization sigma_eff (0.58/peak)", 0.58 / pk, "{:.4f}"),
        ("plateau share (flat hourly steps)", plateau, "{:.0%}"),
        ("largest single-hour jump / total", max_jump, "{:.0%}"),
        ("author followers", followers, "{:,.0f}"),
        ("c(0) (normalized engagement at hour 0)", c0, "{:.2f}"),
    ]
    for name, x, fmt in rows:
        cells = " | ".join(fmt.format(pct(x, q)) for q in (10, 25, 50, 75, 90))
        lines.append(f"| {name} | {cells} |")
    lines.append("")
    lines.append(f"- interpolated tweets (had 1-2h gaps filled): "
                 f"{int(train['interpolated'].sum())} "
                 f"({100 * train['interpolated'].mean():.0f}%)")
    lines.append(f"- peak range: {pk.min():.0f} to {pk.max():.0f}; "
                 f"followers range: {followers.min():.0f} to "
                 f"{followers.max():,.0f}")
    lines.append(f"- every trajectory: complete hours 0-23, max-normalized "
                 f"(ends at its own peak = 1.0)")
    lines.append("")

    lines.append("## Test-set composition")
    lines.append("")
    n_h0 = int(sparse["has_h0"].sum())
    n_h1 = int(sparse["has_h1"].sum())
    lines.append(f"- test-dense: {c['test_dense']} complete trajectories "
                 f"(all hour-0-seedable), peak median "
                 f"{np.median(test_dense['peak']):.0f} -- stratified to match "
                 f"the train peak distribution")
    lines.append(f"- test-sparse: {c['test_sparse']} partial trajectories, "
                 f"median {int(np.median(sparse['n_hours']))} of 24 hours "
                 f"observed; {n_h0} have hour 0, {n_h1} have hour 1")
    lines.append(f"- hour-0-seedable test tweets total: "
                 f"{c['test_dense'] + n_h0}")
    lines.append("")
    lines.append("Files: `train.npz`, `test_dense.npz`, `test_sparse.npz`, "
                 "`split_manifest.json` (frozen id lists). Rebuild with "
                 "`python3 build_dataset.py`.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(HERE, "README.md"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    parts = build()
    summarize(*parts)
