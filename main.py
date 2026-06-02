"""
Bayesian CPT mixture model — entry point.

Usage
-----
    python main.py                        # full run (GlobalDraws/Chains)
    python main.py --draws 500            # quick smoke-test
    python main.py --load                 # skip sampling, load saved trace

Key steps
---------
1. Load + preprocess data (delegates to Estimation project)
2. Set up the likelihood (module-level state in likelihood.py)
3. Build the PyMC model
4. Sample with SMC (Sequential Monte Carlo)
5. Fix label switching, extract cluster parameters, compute responsibilities
6. Save trace + summary CSV + trace plots
"""

import argparse
import os
import numpy as np
import pandas as pd

from GlobalSettings import (
    GlobalCluster, GlobalMethod,
    GlobalDraws, GlobalChains, GlobalSeed,
    GlobalTracePath, GlobalSummaryCSV, GlobalPlotsDir,
)
from preprocessing import load_and_preprocess
from likelihood import setup_likelihood
from bayesian_mixture import build_model, run_sampling, save_trace, load_trace
from analysis import (
    fix_label_switching,
    extract_cluster_params,
    compute_responsibilities,
    print_summary,
    plot_posteriors,
    check_convergence,
)


def _finite_sample_stat_values(data_array):
    """Flatten numeric values from regular or ragged SMC sample_stats arrays."""
    values = data_array.values

    def _iter_numeric(obj):
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            for item in obj.flat:
                yield from _iter_numeric(item)
            return
        arr = np.asarray(obj, dtype=float).ravel()
        for value in arr:
            if np.isfinite(value):
                yield float(value)

    return list(_iter_numeric(values))


def main(draws=None, chains=None, load=False):
    draws  = draws  or GlobalDraws
    chains = chains or GlobalChains

    os.makedirs("outputs/plots", exist_ok=True)

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    print("Loading and preprocessing data …")
    preproc, subjects, subj_index, lotteries = load_and_preprocess()
    N = len(subjects)
    print(f"  {N} subjects, {preproc[0].shape[0]} observations")

    # ── 2. Set up likelihood (module-level state) ─────────────────────────────
    setup_likelihood(preproc, lotteries, subjects, GlobalMethod, GlobalCluster)

    # ── 3 & 4. Build model and sample ─────────────────────────────────────────
    if load and os.path.exists(GlobalTracePath):
        print(f"Loading trace from {GlobalTracePath} …")
        idata = load_trace(GlobalTracePath)
    else:
        print(f"Building model  (C={GlobalCluster}, method={GlobalMethod}) …")
        model = build_model(subjects, GlobalMethod, GlobalCluster)

        print(f"Sampling  (SMC: particles={draws}, chains={chains}) …")
        idata = run_sampling(
            model,
            draws=draws,
            chains=chains,
            seed=GlobalSeed,
        )
        save_trace(idata, GlobalTracePath)

    # ── 5. Post-processing ────────────────────────────────────────────────────
    print("Fixing label switching …")
    idata = fix_label_switching(idata, GlobalMethod, GlobalCluster)

    print("Extracting cluster parameters …")
    cluster_df = extract_cluster_params(idata, GlobalMethod, GlobalCluster)

    print("Computing cluster responsibilities (may take a minute) …")
    resp = compute_responsibilities(
        idata, preproc, subjects, lotteries, GlobalMethod, GlobalCluster
    )

    # ── 6. Save + display ─────────────────────────────────────────────────────
    cluster_df.to_csv(GlobalSummaryCSV, index=False)
    print(f"Summary saved → {GlobalSummaryCSV}")

    print_summary(cluster_df, resp, subjects)

    print("Saving posterior plots …")
    plot_posteriors(idata, GlobalPlotsDir, method=GlobalMethod)

    # SMC diagnostic summary across independent runs / posterior particles.
    print("\nSMC diagnostics (R-hat/ESS across runs and particles):")
    conv_df  = check_convergence(idata, GlobalMethod, GlobalCluster)
    finite_rhat = conv_df["r_hat"].dropna()
    finite_ess_per_chain = conv_df["ess_per_chain"].dropna()
    max_rhat = finite_rhat.max() if not finite_rhat.empty else float("nan")
    min_ess_per_chain = (
        finite_ess_per_chain.min() if not finite_ess_per_chain.empty else float("nan")
    )
    flag = (
        "OK"
        if max_rhat < 1.05 and min_ess_per_chain >= 100
        else "WARNING: poor agreement / low particle ESS"
    )
    print(
        f"  max finite R-hat = {max_rhat:.4f}; "
        f"min ESS/chain = {min_ess_per_chain:.1f}  ({flag})"
    )
    print(conv_df.to_string(index=False))

    # ── Log marginal likelihood (SMC byproduct) ───────────────────────────────
    try:
        log_ml_values = _finite_sample_stat_values(
            idata.sample_stats["log_marginal_likelihood"]
        )
        log_ml = float(np.mean(log_ml_values))
        print(f"\nLog marginal likelihood: {log_ml:.2f}  "
              f"(use to compare C values via Bayes Factor)")
    except Exception:
        pass

    return idata, cluster_df, resp


if __name__ == "__main__":
    import arviz as az

    parser = argparse.ArgumentParser(description="Bayesian CPT mixture estimation")
    parser.add_argument("--draws",  type=int, default=None,
                        help="number of SMC particles (default: GlobalDraws)")
    parser.add_argument("--chains", type=int, default=None,
                        help="independent SMC runs (default: GlobalChains)")
    parser.add_argument("--load",   action="store_true",
                        help="load saved trace instead of sampling")
    args = parser.parse_args()

    main(draws=args.draws, chains=args.chains, load=args.load)
