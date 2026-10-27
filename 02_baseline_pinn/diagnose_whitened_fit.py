#!/usr/bin/env python3
"""
Post-training diagnostics for the whitened-full real-data fit.

Run AFTER train_virality.py.  Loads results/params.pkl (or the latest
checkpoint) and the cached data, then:

  1. BETA LANDSCAPE AT THE FIT: the whitened loss as a function of beta with
     (f, gamma) frozen at their learned values.  Distinguishes "beta = 0.1 is
     a wall-hugging minimizer (the objective wants to go lower)" from
     "beta = 0.1 is an interior-like optimum that happens to sit near the
     bound".  Also evaluated on the raw pointwise-L1 objective for reference.

  2. DATA-SHAPE DIAGNOSTICS: real trajectories are max-normalized integer
     staircases -- long plateaus punctuated by jumps.  Reports the plateau
     fraction (zero increments), the share of exactly-flat tails, and where
     the whitened residual concentrates (plateau nodes vs jump nodes).
     Mechanism being probed: on a plateau, D^beta c decays like
     (t - t_jump)^(-beta) after a jump, so SMALL beta keeps the observed
     "rate" elevated where the hard-coded f > 0 demands a positive rate --
     a model-class interaction that can genuinely prefer small beta on
     staircase data, independent of any estimator cheat.

  3. IMPLIED NOISE: sqrt(loss) per post (heterogeneity check: the whitening
     assumes one shared noise scale; max-normalized small posts violate it).

Outputs -> results/diagnose_whitened_fit.txt, results/diagnose_whitened_fit.png
"""

import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_virality as tv

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BETA_GRID = np.round(np.arange(0.101, 0.951, 0.01), 4)


def load_fit():
    """Prefer final params.pkl; fall back to the latest checkpoint."""
    p = os.path.join(RESULTS, "params.pkl")
    if os.path.exists(p):
        with open(p, "rb") as fh:
            _, f_params, alpha_params, beta_params = pickle.load(fh)
        src = "params.pkl"
    else:
        with open(os.path.join(RESULTS, "train_checkpoint.pkl"), "rb") as fh:
            saved = pickle.load(fh)
        f_params = [(jnp.array(W), jnp.array(b)) for W, b in saved["f_params"]]
        alpha_params = jnp.array(saved["alpha_params"])
        beta_params = jnp.array(saved["beta_params"])
        src = f"train_checkpoint.pkl (step {saved['step']})"
    return f_params, alpha_params, beta_params, src


def main():
    report = []
    d = np.load(os.path.join(RESULTS, "loaded_data_cache.npz"),
                allow_pickle=True)
    c_np, N_log = d["c"], d["N_log"]
    c = jnp.array(c_np, dtype=jnp.float64)
    S = c.shape[0]
    N_log_jax = jnp.array(N_log, dtype=jnp.float64).reshape(S, 1, 1)

    f_params, alpha_params, beta_params, src = load_fit()
    beta_hat = float(0.1 + 0.89 * jax.nn.sigmoid(beta_params))
    velocity_on = bool(int(d["velocity_on"])) if "velocity_on" in d.files else True
    v_jax = jnp.array(d["v"], dtype=jnp.float64).reshape(S, 1, 1)
    alpha = tv.alpha_head(jnp.array(alpha_params), N_log_jax, v_jax,
                          velocity_on)
    abc = tuple(float(x) for x in np.asarray(alpha_params))

    report.append("=" * 74)
    report.append("POST-FIT DIAGNOSTICS -- whitened-full L1 on real data")
    report.append("=" * 74)
    report.append(f"loaded fit from: {src}")
    report.append(f"beta_hat = {beta_hat:.4f}   alpha head (a,b,c) = "
                  f"({abc[0]:+.4f}, {abc[1]:+.4f}, {abc[2]:+.4f}) "
                  f"[velocity {'ON' if velocity_on else 'OFF'}]   "
                  f"alpha mean = {float(alpha.mean()):.4f}, "
                  f"min = {float(alpha.min()):.4f}")
    report.append("")

    fc = tv.fnn_fixed_boundary(f_params, c)
    fprime = tv.f_prime_nn(f_params, c)

    # ------------------------------------------------------------------
    # 1) beta landscapes with (f, gamma) frozen at the fit
    # ------------------------------------------------------------------
    @jax.jit
    def whitened_at(b):
        return tv.loss_whitened_full(c, b, fc, fprime, alpha)

    @jax.jit
    def raw_at(b):
        r, _ = tv.residual_and_jacobian(c, b, fc, fprime, alpha)
        return jnp.mean(r ** 2)

    wl = np.array([float(whitened_at(b)) for b in BETA_GRID])
    rl = np.array([float(raw_at(b)) for b in BETA_GRID])

    i_min = int(np.argmin(wl))
    report.append("BETA LANDSCAPE (f, gamma frozen at the learned fit)")
    report.append("-" * 74)
    report.append(f"  whitened loss: min at beta = {BETA_GRID[i_min]:.3f} "
                  f"(loss {wl[i_min]:.4e}); at the fitted beta_hat: "
                  f"{float(whitened_at(beta_hat)):.4e}")
    report.append(f"  gradient direction at the 0.1 bound: "
                  f"{'still decreasing toward the bound (wall-hugging)' if i_min == 0 else 'interior minimum'}")
    for b_probe in (0.101, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9):
        j = int(np.argmin(np.abs(BETA_GRID - b_probe)))
        report.append(f"    beta = {BETA_GRID[j]:.3f}: whitened {wl[j]:.4e}   "
                      f"raw-L1 {rl[j]:.4e}")
    report.append("")

    # ------------------------------------------------------------------
    # 2) staircase / plateau structure and where the residual lives
    # ------------------------------------------------------------------
    dc = np.diff(c_np[:, :, 0], axis=1)                      # (S, N-1)
    plateau = dc == 0.0
    frac_plateau = float(plateau.mean())
    frac_decrease = float((dc < 0).mean())
    half = dc.shape[1] // 2
    flat_tail = float(np.mean(np.all(dc[:, half:] == 0.0, axis=1)))

    r, J = tv.residual_and_jacobian(c, beta_hat, fc, fprime, alpha)
    JJt = jnp.einsum("snk,smk->snm", J, J) \
        + tv.JITTER * jnp.eye(J.shape[1])[None, :, :]
    x = jnp.linalg.solve(JJt, r[:, :, None])[:, :, 0]
    per_node = np.array(r * x)                               # (S, N-1) contributions
    per_post = per_node.mean(axis=1)

    on_plateau = float(per_node[plateau].mean())
    on_jump = float(per_node[~plateau].mean())
    report.append("DATA SHAPE AND RESIDUAL LOCATION")
    report.append("-" * 74)
    report.append(f"  zero-increment (plateau) share of all steps: "
                  f"{100*frac_plateau:.1f}%")
    report.append(f"  negative increments (engagement removed): "
                  f"{100*frac_decrease:.2f}%")
    report.append(f"  posts fully flat over the second half of the window: "
                  f"{100*flat_tail:.1f}%")
    report.append(f"  mean whitened residual contribution: plateau nodes "
                  f"{on_plateau:.3e} vs jump nodes {on_jump:.3e}")
    report.append("")

    # ------------------------------------------------------------------
    # 3) implied per-post noise (heterogeneity of the whitening's sigma)
    # ------------------------------------------------------------------
    sig_post = np.sqrt(np.maximum(per_post, 0.0))
    report.append("IMPLIED PER-POST NOISE (whitening assumes ONE shared sigma)")
    report.append("-" * 74)
    qs = np.percentile(sig_post, [5, 25, 50, 75, 95])
    report.append("  percentiles of sqrt(per-post whitened loss): "
                  + ", ".join(f"p{p}={v:.4f}" for p, v in
                              zip([5, 25, 50, 75, 95], qs)))
    # correlation with post size proxy (peak raw engagement is gone after
    # normalization; use follower count as the available size proxy)
    corr_nf = float(np.corrcoef(np.log(sig_post + 1e-9), N_log)[0, 1])
    report.append(f"  corr(log implied sigma, log followers) = {corr_nf:+.3f}")
    report.append("")

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].semilogy(BETA_GRID, wl, "b-", label="whitened loss")
    axes[0].axvline(beta_hat, color="r", ls=":", lw=2,
                    label=f"fitted beta = {beta_hat:.3f}")
    axes[0].set_xlabel("beta (f, gamma frozen)")
    axes[0].set_ylabel("whitened loss")
    axes[0].set_title("beta landscape at the fit")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(BETA_GRID, rl, "g-")
    axes[1].axvline(beta_hat, color="r", ls=":", lw=2)
    axes[1].set_xlabel("beta (f, gamma frozen)")
    axes[1].set_ylabel("raw pointwise-L1 loss")
    axes[1].set_title("raw-L1 landscape (reference)")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(sig_post, bins=60)
    axes[2].set_xlabel("sqrt(per-post whitened loss)")
    axes[2].set_ylabel("posts")
    axes[2].set_title("implied per-post noise (shared-sigma check)")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "diagnose_whitened_fit.png"), dpi=150)
    plt.close()

    text = "\n".join(report)
    print(text)
    with open(os.path.join(RESULTS, "diagnose_whitened_fit.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
