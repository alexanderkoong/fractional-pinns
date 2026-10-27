#!/usr/bin/env python3
"""
Regenerate the paper figure fig_whitened_beta.pdf at N = 24.

The original figure came from the N = 48 studies.  This script produces the
same two-panel comparison from the N = 24 experiments:

  (a) exact minimizer of each objective with f fixed at truth, vs sigma;
  (b) joint (beta, f) training, vs sigma (means over the two noise draws).

Three objectives: the pointwise L1 residual (paper Eq. 4), the DIAGONALLY
whitened L1 (per-node variance normalization + log-variance terms), and the
FULLY whitened L1 (data-space / errors-in-variables metric).

The pointwise and fully whitened results are read from the existing
verify_24h.py checkpoints; the diagonal variant was not part of the 24h
verification scope, so its runs are performed here (duplicated from
"../L1 with log term/l1_log_term_poc.py", adapted to N = 24), checkpointed
under results/checkpoints/ with a fig_diag_ prefix.

Output -> results/fig_whitened_beta.pdf (copy into the LaTeX folder),
          results/fig_whitened_beta.txt (the numbers behind the figure)
"""

import os
from time import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit, random, value_and_grad
from jax.example_libraries import optimizers

import fractional_ode as fo
import pinn_lib as pl
import whitened_l1 as wl
import verify_24h as v

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

TRUE_BETA = 0.5
SIGMAS = [0.0, 0.005, 0.02, 0.05]
NOISE_SEEDS = [101, 102]
NUM_STEPS = 30000
INIT_BETA = 0.7
NET_SEED = 1


# ----------------------------------------------------------------------------
# Diagonally whitened loss (duplicated from ../L1 with log term/, N-agnostic)
# ----------------------------------------------------------------------------
def estimate_sigma(c):
    """Robust noise level from second differences (MAD; Var(d2) = 6 sigma^2)."""
    d2 = np.diff(np.asarray(c)[:, :, 0], n=2, axis=1)
    return float(np.median(np.abs(d2)) / 0.6745 / np.sqrt(6.0))


def loss_whitened_diag(c_jax, beta, fc, fprime, sigma_hat):
    """Per-node variance normalization + log-variance term (diagonal ML)."""
    r, J = wl.residual_and_jacobian(c_jax, beta, fc, fprime, 1.0)
    v_row = jnp.sum(J ** 2, axis=-1)
    return jnp.mean(r ** 2 / (sigma_hat ** 2 * v_row)) + jnp.mean(jnp.log(v_row))


def make_diag_beta_loss(c_jax, sigma_hat):
    c_np = np.asarray(c_jax)
    fc = jnp.array(fo.f_cubic(c_np))
    fprime = jnp.array(1.0 - 3.0 * c_np ** 2)
    return jit(lambda b: loss_whitened_diag(c_jax, b, fc, fprime, sigma_hat))


def train_diag_joint(c_jax, sigma_hat, ckpt_label):
    path = os.path.join(CKPT_DIR, f"{ckpt_label}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return float(d["beta_hat"]), True

    key = random.PRNGKey(NET_SEED)
    f_params = pl.init_network_params(1, 1, v.F_WIDTH, 1, key, v.F_LAYERS)
    raw0 = pl.raw_from_beta(jnp.array(INIT_BETA))

    def loss_fn(params):
        f_p, raw = params
        beta = pl.beta_from_raw(raw)
        fc = pl.fnn_fixed_boundary(f_p, c_jax)
        fprime = wl.f_prime_nn(f_p, c_jax)
        return loss_whitened_diag(c_jax, beta, fc, fprime, sigma_hat)

    opt_init, opt_update, get_params = optimizers.adam(v.LR)
    opt_state = opt_init((f_params, raw0))

    @jit
    def step(i, opt_state):
        params = get_params(opt_state)
        l, g = value_and_grad(loss_fn)(params)
        return l, opt_update(i, g, opt_state)

    for i in range(NUM_STEPS + 1):
        l, opt_state = step(i, opt_state)

    f_params, raw = get_params(opt_state)
    beta_hat = float(pl.beta_from_raw(raw))
    flat = {f"W{i}": np.array(W) for i, (W, _) in enumerate(f_params)}
    flat.update({f"b{i}": np.array(b) for i, (_, b) in enumerate(f_params)})
    np.savez_compressed(path, beta_hat=beta_hat, final_loss=float(l),
                        n_layers=len(f_params), **flat)
    return beta_hat, False


def ckpt_beta(label):
    return float(np.load(os.path.join(CKPT_DIR, f"{label}.npz"))["beta_hat"])


# ----------------------------------------------------------------------------
def main():
    _, c_clean, _ = fo.load_or_generate_dataset(
        os.path.join(RESULTS_DIR, "data_cache"), v.NUM_SAMPLES, v.N_OBS,
        TRUE_BETA, v.ALPHA, seed=v.SEED_DATA)

    def perturbed(sigma, seed):
        if sigma == 0.0:
            return c_clean.copy()
        rng = np.random.RandomState(seed)
        return np.clip(c_clean + rng.normal(0.0, sigma, c_clean.shape), 0.0, 1.0)

    lines = ["fig_whitened_beta.pdf source numbers (N = 24)", "=" * 60]

    # --- panel (a): beta-only argmin (one draw, seed 101, as in Part 1b) ---
    print("panel (a): beta-only landscapes ...", flush=True)
    panel_a = {"pointwise": [], "diag": [], "full": []}
    for sigma in SIGMAS:
        c_s = perturbed(sigma, NOISE_SEEDS[0])
        c_jax = jnp.array(c_s)
        s_hat = max(estimate_sigma(c_s), 1e-6)
        panel_a["pointwise"].append(v.argmin_beta(v.make_beta_only_loss(c_jax, "l1_full")))
        panel_a["diag"].append(v.argmin_beta(make_diag_beta_loss(c_jax, s_hat)))
        panel_a["full"].append(v.argmin_beta(v.make_beta_only_loss(c_jax, "l1_whitened")))
    lines.append("panel (a) beta-only argmin vs sigma " + str(SIGMAS) + ":")
    for k in panel_a:
        lines.append(f"  {k:>9}: " + ", ".join(f"{b:.4f}" for b in panel_a[k]))

    # --- panel (b): joint training ---
    print("panel (b): joint results ...", flush=True)
    panel_b = {"pointwise": [], "diag": [], "full": []}

    # sigma = 0 points: the init-0.7/seed-1 clean runs from verify_24h
    panel_b["pointwise"].append(ckpt_beta(
        f"clean_l1_full_initb{INIT_BETA}_seed{NET_SEED}_steps{NUM_STEPS}"))
    panel_b["full"].append(ckpt_beta(
        f"clean_l1_whitened_initb{INIT_BETA}_seed{NET_SEED}_steps{NUM_STEPS}"))
    c_jax = jnp.array(c_clean)
    s_hat = max(estimate_sigma(c_clean), 1e-6)
    bh, cached = train_diag_joint(c_jax, s_hat, f"fig_diag_sigma0_steps{NUM_STEPS}")
    panel_b["diag"].append(bh)
    print(f"  diag sigma=0: beta_hat={bh:.4f} ({'cached' if cached else 'trained'})",
          flush=True)

    # sigma > 0: means over the two noise draws
    for sigma in SIGMAS[1:]:
        for name, prefix in [("pointwise", "pert_l1_full"),
                             ("full", "pert_l1_whitened")]:
            vals = [ckpt_beta(f"{prefix}_sigma{sigma:g}_nseed{ns}_steps{NUM_STEPS}")
                    for ns in NOISE_SEEDS]
            panel_b[name].append(float(np.mean(vals)))
        vals = []
        for ns in NOISE_SEEDS:
            t0 = time()
            c_s = perturbed(sigma, ns)
            s_hat = max(estimate_sigma(c_s), 1e-6)
            bh, cached = train_diag_joint(
                jnp.array(c_s), s_hat,
                f"fig_diag_sigma{sigma:g}_nseed{ns}_steps{NUM_STEPS}")
            vals.append(bh)
            print(f"  diag sigma={sigma} nseed={ns}: beta_hat={bh:.4f} "
                  f"({'cached' if cached else f'{time()-t0:.0f}s'})", flush=True)
        panel_b["diag"].append(float(np.mean(vals)))

    lines.append("panel (b) joint beta_hat vs sigma " + str(SIGMAS) +
                 " (means over 2 draws for sigma > 0):")
    for k in panel_b:
        lines.append(f"  {k:>9}: " + ", ".join(f"{b:.4f}" for b in panel_b[k]))

    # --- figure ---
    style = {"pointwise": dict(color="tab:red", marker="o",
                               label="pointwise L1"),
             "diag": dict(color="tab:orange", marker="s",
                          label="diagonally whitened"),
             "full": dict(color="tab:blue", marker="^",
                          label="fully whitened")}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, panel, title in [(axes[0], panel_a, "(a) $f$ fixed at truth"),
                             (axes[1], panel_b, "(b) joint training of $(\\beta, f)$")]:
        for k in ["pointwise", "diag", "full"]:
            ax.plot(SIGMAS, panel[k], lw=1.6, ms=5.5, **style[k])
        ax.axhline(TRUE_BETA, color="k", ls=":", lw=1.4)
        ax.set_xlabel(r"perturbation level $\sigma$")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(SIGMAS)
        ax.set_xticklabels(["0", "0.005", "0.02", "0.05"], fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"recovered $\hat{\beta}$")
    axes[0].set_ylim(0.05, 0.62)
    axes[1].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_whitened_beta.pdf")
    plt.savefig(out)
    plt.close()
    lines.append(f"saved: {out}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(RESULTS_DIR, "fig_whitened_beta.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
