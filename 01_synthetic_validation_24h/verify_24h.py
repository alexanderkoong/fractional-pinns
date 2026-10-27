#!/usr/bin/env python3
"""
24-HOUR MODEL VERIFICATION
==========================

The real-world dataset now covers 24 hourly observations per post instead of
48.  This suite reruns the parameter-recovery validations at N = 24
observation nodes on the rescaled interval t in [0, 1] (grid spacing
h = 1/23 instead of 1/47 -- a coarser grid, which stresses the L1
discretization harder: the initial-layer error lives at fixed node index,
so it now occupies a larger fraction of the observation window).

Scope (per the current experiment plan -- nothing else):
  PART 1  recover beta ALONE (f, alpha at truth): exact landscape argmin,
          clean data at true beta in {0.3, 0.5, 0.7}, and perturbed data
          (true beta = 0.5) at sigma in {0.005, 0.02, 0.05}.
  PART 2  JOINT (beta, f) recovery on clean data: 2 beta-inits x 2 net
          seeds per objective, PySR symbolic regression on each
          objective's best run.
  PART 3  JOINT (beta, f) recovery on PERTURBED data: sigma x 2 noise
          draws, cheat ratio L(learned)/L(true params) under each
          objective, PySR at sigma = 0.02 for the honest objectives.

Objectives compared (all use the L1 scheme except volterra, kept as the
integral-form reference):
    l1_full         pointwise L1, all nodes -- the paper's Eq. (4) (control)
    l1_burn         pointwise L1 excluding the first 4 interior nodes
    volterra        integral-form (solve-consistent) residual
    l1_whitened     data-space whitened L1 (errors-in-variables ML; the
                    winner of the "L1 with log term" proof of concept)

Everything else matches the 48-hour suites for comparability: same ground
truth D^0.5 c = c(1-c^2), alpha = 1 fixed, 200 trajectories with the same
c0 draw (seed 42), numerically exact data (corrected ABM, fine-grid solve +
subsample), same production f-network and beta parameterization, adam 30k
steps, additive clipped Gaussian perturbations with the same noise seeds.

N = 48 reference numbers quoted in the report are citations from the prior
folders ("model validation", "perturbed synthetic data full recovery",
"L1 with log term"), not reruns.

Outputs -> results/verify_24h.txt, results/verify_24h_*.png
"""

import os
from time import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import jax.numpy as jnp
from jax import jit, random, value_and_grad
from jax.example_libraries import optimizers

import fractional_ode as fo
import pinn_lib as pl
import whitened_l1 as wl

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "data_cache")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PYSR_DIR = os.path.join(RESULTS_DIR, "pysr")

N_OBS = 24                       # <-- the change under test (was 48)
TRUE_BETA = 0.5
TRUE_BETAS_PART1 = [0.3, 0.5, 0.7]
ALPHA = 1.0
NUM_SAMPLES = 200
SEED_DATA = 42

SIGMAS = [0.005, 0.02, 0.05]
NOISE_SEEDS = [101, 102]

INIT_BETAS = [0.3, 0.7]
NET_SEEDS = [1, 2]
NUM_STEPS = 30000
LR = 1e-3
F_WIDTH = 30
F_LAYERS = 4
BURN_IN = 5                      # same node count as the 48h suite: the layer
                                 # error lives at fixed node index, not time
# scan floor just above the joint parameterization's 0.1 bound, so noisy
# landscapes report their true (possibly collapsed) argmin rather than a
# scan-boundary artifact
BETA_SCAN = np.round(np.arange(0.11, 0.951, 0.01), 4)

ESTIMATORS = ["l1_full", "l1_burn", "volterra", "l1_whitened"]
JOINT_PERTURBED = ["l1_full", "volterra", "l1_whitened"]
EST_LABEL = {
    "l1_full": "pointwise L1, all nodes (paper Eq. 4)",
    "l1_burn": f"pointwise L1, burn-in (nodes >= {BURN_IN})",
    "volterra": "Volterra integral form (solve-consistent)",
    "l1_whitened": "data-space whitened L1 (EIV maximum likelihood)",
}

C_GRID = np.linspace(0.0, 1.0, 500)
F_TRUE_GRID = fo.f_cubic(C_GRID)


# ----------------------------------------------------------------------------
# Losses (shared core: fc / fprime arrays so true-f and NN cases share code)
# ----------------------------------------------------------------------------
def loss_of(method, c_jax, beta, fc, fprime):
    if method == "l1_full":
        r = pl.caputo_l1(c_jax, beta) - ALPHA * fc
        return pl.mse(r[:, 1:, :], jnp.zeros_like(r[:, 1:, :]))
    if method == "l1_burn":
        r = pl.caputo_l1(c_jax, beta) - ALPHA * fc
        return pl.mse(r[:, BURN_IN:, :], jnp.zeros_like(r[:, BURN_IN:, :]))
    if method == "volterra":
        r = pl.volterra_residual(c_jax, ALPHA * fc, beta, 1.0)
        return pl.mse(r[:, 1:, :], jnp.zeros_like(r[:, 1:, :]))
    if method == "l1_whitened":
        return wl.loss_whitened_full(c_jax, beta, fc, fprime, ALPHA)
    raise ValueError(method)


def make_beta_only_loss(c_jax, method):
    """Loss as a function of beta alone, f fixed at TRUTH (analytic f')."""
    c_np = np.asarray(c_jax)
    fc = jnp.array(fo.f_cubic(c_np))
    fprime = jnp.array(1.0 - 3.0 * c_np ** 2)
    return jit(lambda b: loss_of(method, c_jax, b, fc, fprime))


def make_joint_loss(c_jax, method):
    needs_fprime = method == "l1_whitened"

    def fn(params):
        f_params, raw = params
        beta = pl.beta_from_raw(raw)
        fc = pl.fnn_fixed_boundary(f_params, c_jax)
        # fprime only enters the whitened loss; skip the extra jvp otherwise
        fprime = wl.f_prime_nn(f_params, c_jax) if needs_fprime else fc
        return loss_of(method, c_jax, beta, fc, fprime)
    return fn


def argmin_beta(loss):
    scan = np.array([float(loss(b)) for b in BETA_SCAN])
    i0 = int(np.argmin(scan))
    lo = BETA_SCAN[max(0, i0 - 2)]
    hi = BETA_SCAN[min(len(BETA_SCAN) - 1, i0 + 2)]
    res = minimize_scalar(lambda b: float(loss(b)), bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-7})
    return float(res.x)


# ----------------------------------------------------------------------------
# Training (checkpointed) and evaluation
# ----------------------------------------------------------------------------
def train_joint(c_jax, method, init_beta, net_seed, ckpt_label):
    path = os.path.join(CKPT_DIR, f"{ckpt_label}.npz")
    if os.path.exists(path):
        d = np.load(path)
        f_params = [(jnp.array(d[f"W{i}"]), jnp.array(d[f"b{i}"]))
                    for i in range(int(d["n_layers"]))]
        return f_params, float(d["beta_hat"]), float(d["final_loss"]), True

    key = random.PRNGKey(net_seed)
    f_params = pl.init_network_params(1, 1, F_WIDTH, 1, key, F_LAYERS)
    raw0 = pl.raw_from_beta(jnp.array(init_beta))
    loss_fn = make_joint_loss(c_jax, method)

    opt_init, opt_update, get_params = optimizers.adam(LR)
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
    return f_params, beta_hat, float(l), False


def evaluate_f(f_params):
    c_in = jnp.array(C_GRID.reshape(1, -1, 1))
    f_learned = np.array(pl.fnn_fixed_boundary(f_params, c_in))[0, :, 0]
    corr = float(np.corrcoef(F_TRUE_GRID, f_learned)[0, 1])
    scale = float(np.max(f_learned) / np.max(F_TRUE_GRID))
    return f_learned, corr, scale


def loss_at_truth(c_jax, method):
    return float(make_beta_only_loss(c_jax, method)(TRUE_BETA))


def run_pysr(f_learned, label):
    from pysr import PySRRegressor
    c_fit = np.concatenate([C_GRID, [0.0, 1.0]])
    f_fit = np.concatenate([f_learned, [0.0, 0.0]])
    weights = np.concatenate([np.ones(len(C_GRID)), [10.0, 10.0]])
    model = PySRRegressor(
        model_selection="best", niterations=300, maxsize=14,
        binary_operators=["+", "-", "*", "/"], unary_operators=["exp"],
        complexity_of_operators={"/": 2, "exp": 3},
        elementwise_loss="loss(x, y, w) = w*(x - y)^2",
        output_directory=PYSR_DIR, run_id=label, progress=False, verbosity=0,
    )
    model.fit(c_fit.reshape(-1, 1), f_fit, weights=weights)
    expr = str(model.sympy())
    f_sym = model.predict(C_GRID.reshape(-1, 1))
    corr = float(np.corrcoef(F_TRUE_GRID, f_sym)[0, 1])
    return expr, corr


# ----------------------------------------------------------------------------
def main():
    for d in (RESULTS_DIR, CACHE_DIR, CKPT_DIR, PYSR_DIR):
        os.makedirs(d, exist_ok=True)

    report = []
    report.append("=" * 78)
    report.append(f"24-HOUR MODEL VERIFICATION  (N = {N_OBS} observation nodes, "
                  f"h = 1/{N_OBS - 1})")
    report.append("=" * 78)
    report.append(f"Ground truth: D^beta c = c(1 - c^2), alpha = {ALPHA} (fixed), "
                  f"{NUM_SAMPLES} trajectories, c0 ~ U(0.01, 0.3)")
    report.append("All settings identical to the 48h suites except N; N = 48")
    report.append("numbers quoted below are citations from those folders.")
    report.append("")

    data = {}
    for tb in TRUE_BETAS_PART1:
        _, c, _ = fo.load_or_generate_dataset(
            CACHE_DIR, NUM_SAMPLES, N_OBS, tb, ALPHA, seed=SEED_DATA)
        data[tb] = c
    c_clean = data[TRUE_BETA]
    wl.self_test(jnp.array(c_clean))
    print("[self-test] whitened operator matches production L1 at N=24",
          flush=True)

    def perturbed(sigma, noise_seed):
        if sigma == 0.0:
            return c_clean.copy()
        rng = np.random.RandomState(noise_seed)
        return np.clip(c_clean + rng.normal(0.0, sigma, c_clean.shape),
                       0.0, 1.0)

    # ------------------------------------------------------------------
    # PART 1: beta alone
    # ------------------------------------------------------------------
    print("PART 1: beta-only landscapes ...", flush=True)
    report.append("PART 1: BETA-ONLY RECOVERY (f and alpha at truth; exact argmin)")
    report.append("-" * 78)
    report.append("1a. Clean data, three ground truths:")
    report.append(f"  {'true beta':>10} | " +
                  " | ".join(f"{m:>16}" for m in ESTIMATORS))
    part1a = {}
    for tb in TRUE_BETAS_PART1:
        c_jax = jnp.array(data[tb])
        cells = []
        for m in ESTIMATORS:
            bhat = argmin_beta(make_beta_only_loss(c_jax, m))
            part1a[(tb, m)] = bhat
            cells.append(f"{bhat:.4f} ({bhat - tb:+.3f})")
        report.append(f"  {tb:>10} | " + " | ".join(f"{c:>16}" for c in cells))
    report.append("  (N=48 clean reference, true beta 0.5: l1_full 0.4838, "
                  "l1_burn 0.4978,")
    report.append("   volterra 0.4994, l1_whitened 0.4970)")
    report.append("")

    report.append("1b. Perturbed data (true beta = 0.5, noise draw seed "
                  f"{NOISE_SEEDS[0]}):")
    report.append(f"  {'sigma':>7} | " +
                  " | ".join(f"{m:>16}" for m in ESTIMATORS))
    part1b = {}
    for sigma in [0.0] + SIGMAS:
        c_jax = jnp.array(perturbed(sigma, NOISE_SEEDS[0]))
        cells = []
        for m in ESTIMATORS:
            bhat = argmin_beta(make_beta_only_loss(c_jax, m))
            part1b[(sigma, m)] = bhat
            cells.append(f"{bhat:.4f} ({bhat - TRUE_BETA:+.3f})")
        report.append(f"  {sigma:>7} | " + " | ".join(f"{c:>16}" for c in cells))
    report.append("")

    # ------------------------------------------------------------------
    # PART 2: joint (beta, f) on clean data
    # ------------------------------------------------------------------
    print("PART 2: clean joint recovery ...", flush=True)
    report.append("PART 2: JOINT (beta, f) RECOVERY -- CLEAN DATA")
    report.append(f"({len(INIT_BETAS)} beta-inits x {len(NET_SEEDS)} net seeds "
                  f"per objective, {NUM_STEPS} adam steps)")
    report.append("-" * 78)
    c_clean_jax = jnp.array(c_clean)
    part2 = {}
    for method in ESTIMATORS:
        recs = []
        for ib in INIT_BETAS:
            for ns in NET_SEEDS:
                t0 = time()
                label = f"clean_{method}_initb{ib}_seed{ns}_steps{NUM_STEPS}"
                f_params, beta_hat, final_loss, cached = train_joint(
                    c_clean_jax, method, ib, ns, label)
                f_learned, corr, scale = evaluate_f(f_params)
                recs.append({"init": ib, "seed": ns, "beta_hat": beta_hat,
                             "f_corr": corr, "f_scale": scale,
                             "f_learned": f_learned, "loss": final_loss})
                print(f"  {label}: beta_hat={beta_hat:.4f} f_corr={corr:.4f} "
                      f"({'cached' if cached else f'{time()-t0:.0f}s'})",
                      flush=True)
        part2[method] = recs
        betas = np.array([r["beta_hat"] for r in recs])
        report.append(f"  {method:>12}: beta_hat {betas.mean():.4f} "
                      f"(err {betas.mean() - TRUE_BETA:+.4f}, spread "
                      f"{betas.max() - betas.min():.1e}), "
                      f"f corr {np.mean([r['f_corr'] for r in recs]):.4f}, "
                      f"f scale {np.mean([r['f_scale'] for r in recs]):.4f}")
    report.append("  (N=48 clean joint reference: l1_full 0.4729, l1_burn "
                  "0.4967, volterra 0.4991,")
    report.append("   l1_whitened 0.5026)")
    report.append("")

    report.append("2b. PySR on each objective's best clean run:")
    pysr_clean = {}
    for method in ESTIMATORS:
        best = min(part2[method], key=lambda r: abs(r["beta_hat"] - TRUE_BETA))
        try:
            expr, corr = run_pysr(best["f_learned"], f"clean_{method}")
            pysr_clean[method] = (expr, corr)
            report.append(f"  {method:>12}: f(c) = {expr}   (corr {corr:.5f})")
        except Exception as exc:
            pysr_clean[method] = (None, np.nan)
            report.append(f"  {method:>12}: PySR FAILED ({exc!r})")
    report.append("")

    # ------------------------------------------------------------------
    # PART 3: joint (beta, f) on perturbed data
    # ------------------------------------------------------------------
    print("PART 3: perturbed joint recovery ...", flush=True)
    report.append("PART 3: JOINT (beta, f) RECOVERY -- PERTURBED DATA")
    report.append(f"(init beta 0.7, net seed 1, {len(NOISE_SEEDS)} noise draws "
                  f"per sigma; cheat = L(learned)/L(true params), per objective)")
    report.append("-" * 78)
    report.append(f"  {'method':>12} | {'sigma':>6} | "
                  f"{'beta_hat (per draw)':>19} | {'mean err':>9} | "
                  f"{'f corr':>7} | {'cheat':>7}")
    part3 = {}
    for method in JOINT_PERTURBED:
        for sigma in SIGMAS:
            recs = []
            for nsd in NOISE_SEEDS:
                t0 = time()
                c_jax = jnp.array(perturbed(sigma, nsd))
                label = (f"pert_{method}_sigma{sigma:g}_nseed{nsd}"
                         f"_steps{NUM_STEPS}")
                f_params, beta_hat, final_loss, cached = train_joint(
                    c_jax, method, 0.7, 1, label)
                f_learned, corr, scale = evaluate_f(f_params)
                l_true = loss_at_truth(c_jax, method)
                recs.append({"beta_hat": beta_hat, "f_corr": corr,
                             "f_scale": scale, "f_learned": f_learned,
                             "cheat": final_loss / l_true if l_true else np.nan})
                print(f"  {label}: beta_hat={beta_hat:.4f} f_corr={corr:.4f} "
                      f"cheat={recs[-1]['cheat']:.3f} "
                      f"({'cached' if cached else f'{time()-t0:.0f}s'})",
                      flush=True)
            part3[(method, sigma)] = recs
            bh = [r["beta_hat"] for r in recs]
            report.append(
                f"  {method:>12} | {sigma:>6} | "
                f"{', '.join(f'{b:.4f}' for b in bh):>19} | "
                f"{np.mean(bh) - TRUE_BETA:>+9.4f} | "
                f"{np.mean([r['f_corr'] for r in recs]):>7.4f} | "
                f"{np.mean([r['cheat'] for r in recs]):>7.3f}")
    report.append("  (N=48 perturbed joint reference, sigma 0.005/0.02/0.05: "
                  "l1_full 0.339/0.100/0.100,")
    report.append("   volterra 0.499/0.483/0.17-0.26, l1_whitened "
                  "-/0.5011/0.3122)")
    report.append("")

    report.append("3b. PySR at sigma = 0.02 (honest objectives):")
    for method in ["volterra", "l1_whitened"]:
        recs = part3[(method, 0.02)]
        med = sorted(recs, key=lambda r: r["beta_hat"])[len(recs) // 2]
        try:
            expr, corr = run_pysr(med["f_learned"], f"pert_{method}_sigma0.02")
            report.append(f"  {method:>12}: f(c) = {expr}   (corr {corr:.5f})")
        except Exception as exc:
            report.append(f"  {method:>12}: PySR FAILED ({exc!r})")
    report.append("")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    report.append("=" * 78)
    report.append("VERDICT AT N = 24")
    report.append("=" * 78)
    for method in ESTIMATORS:
        clean_err = abs(np.mean([r["beta_hat"] for r in part2[method]])
                        - TRUE_BETA)
        line = f"  {method:>12}: clean joint |beta err| = {clean_err:.4f}"
        if method in JOINT_PERTURBED:
            errs = {s: abs(np.mean([r["beta_hat"] for r in part3[(method, s)]])
                           - TRUE_BETA) for s in SIGMAS}
            ok = [s for s in SIGMAS if errs[s] <= 0.05]
            line += (f"; perturbed |err| <= 0.05 up to sigma = "
                     f"{max(ok) if ok else 'NONE'}"
                     f"  ({', '.join(f'{s}: {errs[s]:.3f}' for s in SIGMAS)})")
        report.append(line)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    x = np.arange(len(TRUE_BETAS_PART1))
    for m in ESTIMATORS:
        ax.plot(TRUE_BETAS_PART1,
                [part1a[(tb, m)] - tb for tb in TRUE_BETAS_PART1],
                "o-", label=m)
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("true beta")
    ax.set_ylabel("argmin error (beta-only, clean)")
    ax.set_title(f"Part 1a: clean beta-only bias at N={N_OBS}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for m in ESTIMATORS:
        ax.plot([0.0] + SIGMAS, [part1b[(s, m)] for s in [0.0] + SIGMAS],
                "o-", label=f"{m} (beta-only)")
    for m, mk in zip(JOINT_PERTURBED, ["x", "s", "^"]):
        ax.plot(SIGMAS,
                [np.mean([r["beta_hat"] for r in part3[(m, s)]]) for s in SIGMAS],
                mk, ms=9, mfc="none", mew=2, ls="none", label=f"{m} (joint)")
    ax.axhline(TRUE_BETA, color="k", ls=":", lw=1.5)
    ax.set_xlabel("noise sigma")
    ax.set_ylabel("recovered beta")
    ax.set_title(f"Parts 1b + 3: recovery under noise at N={N_OBS}")
    ax.set_ylim(0.05, 0.8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for m in ESTIMATORS:
        best = min(part2[m], key=lambda r: abs(r["beta_hat"] - TRUE_BETA))
        ax.plot(C_GRID, best["f_learned"], lw=1.5, label=f"{m} (clean best)")
    ax.plot(C_GRID, F_TRUE_GRID, "k--", lw=2.2, label="true c(1-c^2)")
    ax.set_xlabel("c")
    ax.set_ylabel("f(c)")
    ax.set_title("Part 2: learned f on clean data")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "verify_24h_summary.png"), dpi=150)
    plt.close()

    text = "\n".join(report)
    print("\n" + text)
    with open(os.path.join(RESULTS_DIR, "verify_24h.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
