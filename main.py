"""
Bayesian CPT mixture model with continuous reference-point weights.

Usage:
    python main.py
    python main.py --draws 500
    python main.py --load
"""

import argparse
import os
import numpy as np

from GlobalSettings import (
    GlobalCluster,
    GlobalMethod,
    GlobalDraws,
    GlobalChains,
    GlobalSeed,
    GlobalTracePath,
    GlobalSummaryCSV,
    GlobalPlotsDir,
    GlobalKsiMode,
)
from preprocessing import load_and_preprocess
from likelihood import setup_likelihood
from bayesian_mixture import build_model, run_sampling, save_trace, load_trace
from analysis import (
    fix_label_switching,
    extract_cluster_params,
    extract_reference_weights,
    compute_cluster_responsibilities,
    print_summary,
    plot_posteriors,
    check_convergence,
)


def _finite_sample_stat_values(data_array):
    # SMC sample_stats may contain ragged object arrays (one entry per tempering stage).
    # This helper recursively flattens them into a list of finite floats.
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


def _validate_trace_matches_data(idata, n_subjects):
    # Guard against loading a stale trace produced by an older parameterisation
    # or a different dataset (different number of subjects).
    if GlobalKsiMode not in {"estimated", "fixed", "marginalized"}:
        raise ValueError(
            "GlobalKsiMode must be one of: 'estimated', 'fixed', 'marginalized'."
        )
    if "theta_raw" in idata.posterior:
        raise ValueError(
            "Saved trace was produced by the sigmoid/raw parameterisation. "
            "Re-run sampling without --load."
        )
    for required_var in ("theta_rest", "a_weights"):
        if required_var not in idata.posterior:
            raise ValueError(
                f"Saved trace is missing {required_var!r}. "
                "Re-run sampling without --load."
            )
    has_ksi = "ksi" in idata.posterior
    has_mu_ksi = "mu_ksi" in idata.posterior

    if GlobalKsiMode == "marginalized":
        if has_ksi or has_mu_ksi:
            raise ValueError(
                "Saved trace contains ksi variables, but GlobalKsiMode is "
                "'marginalized'. Re-run sampling without --load."
            )
        return
    if GlobalKsiMode == "estimated" and not has_mu_ksi:
        raise ValueError(
            "Saved trace is missing 'mu_ksi', but GlobalKsiMode is 'estimated'. "
            "Re-run sampling without --load."
        )
    if GlobalKsiMode == "fixed" and has_mu_ksi:
        raise ValueError(
            "Saved trace contains estimated ksi variables, but GlobalKsiMode is "
            "'fixed'. Re-run sampling without --load."
        )
    if not has_ksi:
        raise ValueError(
            f"Saved trace is missing 'ksi', but GlobalKsiMode is {GlobalKsiMode!r}. "
            "Re-run sampling without --load."
        )

    # ksi has one entry per subject, so its last dimension encodes dataset size.
    trace_n = idata.posterior["ksi"].shape[-1]
    if trace_n != n_subjects:
        raise ValueError(
            f"Saved trace has ksi dimension {trace_n}, but the current preprocessing "
            f"produced {n_subjects} subjects. Re-run sampling without --load."
        )


def main(draws=None, chains=None, clusters=None, load=False):
    # CLI overrides take precedence; fall back to GlobalSettings values.
    draws = draws or GlobalDraws
    chains = chains or GlobalChains
    C = clusters if clusters is not None else GlobalCluster

    os.makedirs(GlobalPlotsDir, exist_ok=True)

    # --- Data ---
    print("Loading and preprocessing data ...")
    preproc, subjects, subj_index, lotteries = load_and_preprocess()
    del subj_index  # only needed inside preprocessing
    print(f"  {len(subjects)} subjects, {preproc[0].shape[0]} observations")

    # Register the compiled likelihood op with the global lottery/data arrays so
    # PyTensor can call it inside the PyMC model graph.
    setup_likelihood(preproc, lotteries, subjects, GlobalMethod, C)

    # --- Sampling or loading ---
    if load and os.path.exists(GlobalTracePath):
        print(f"Loading trace from {GlobalTracePath} ...")
        idata = load_trace(GlobalTracePath)
    else:
        print(
            f"Building model (C={C} clusters, method={GlobalMethod}, "
            f"ksi={GlobalKsiMode}) ..."
        )
        model = build_model(subjects, GlobalMethod, C)

        print(f"Sampling (SMC: particles={draws}, chains={chains}) ...")
        idata = run_sampling(model, draws=draws, chains=chains, seed=GlobalSeed)

    # Catch mismatches between a saved trace and the current data/parameterisation.
    _validate_trace_matches_data(idata, len(subjects))

    # --- Post-processing ---
    # For C=2, sort clusters by lambda so cluster 0 is always the low-loss-aversion type.
    print("Fixing label switching ...")
    idata = fix_label_switching(idata, GlobalMethod, C)

    # Only write the trace after a fresh sampling run, not when reloading.
    if not (load and os.path.exists(GlobalTracePath)):
        save_trace(idata, GlobalTracePath)

    # --- Posterior summaries ---
    print("Extracting cluster parameters ...")
    cluster_df = extract_cluster_params(idata, GlobalMethod, C)

    print("Extracting reference-point weights ...")
    ref_df = extract_reference_weights(idata, C)

    # Soft assignment probabilities P(cluster k | subject i), averaged over posterior draws.
    print("Computing cluster responsibilities ...")
    resp = compute_cluster_responsibilities(
        idata, preproc, subjects, lotteries, GlobalMethod, C
    )

    # --- Output ---
    cluster_df.to_csv(GlobalSummaryCSV, index=False)
    print(f"Cluster summary saved -> {GlobalSummaryCSV}")

    ref_path = GlobalSummaryCSV.replace(".csv", "_reference_weights.csv")
    ref_df.to_csv(ref_path, index=False)
    print(f"Reference weights saved -> {ref_path}")

    print_summary(cluster_df, ref_df, resp, subjects)

    print("Saving posterior plots ...")
    plot_posteriors(idata, GlobalPlotsDir, method=GlobalMethod)

    # --- Convergence diagnostics ---
    # R-hat compares within-chain vs between-chain variance across SMC runs.
    # ESS measures effective particle count after importance resampling.
    print("\nSMC diagnostics (R-hat/ESS across runs and particles):")
    conv_df = check_convergence(idata, GlobalMethod, C)
    finite_rhat = conv_df["r_hat"].dropna()
    finite_ess = conv_df["ess_per_chain"].dropna()
    max_rhat = finite_rhat.max() if not finite_rhat.empty else float("nan")
    min_ess = finite_ess.min() if not finite_ess.empty else float("nan")
    flag = "OK" if max_rhat < 1.05 and min_ess >= 100 else "WARNING: poor agreement / low particle ESS"
    print(f"  max finite R-hat = {max_rhat:.4f}; min ESS/chain = {min_ess:.1f} ({flag})")
    print(conv_df.to_string(index=False))

    # Log marginal likelihood is the SMC estimator of log p(data | model),
    # useful for Bayes-factor model comparison across different C or methods.
    try:
        log_ml_values = _finite_sample_stat_values(
            idata.sample_stats["log_marginal_likelihood"]
        )
        log_ml = float(np.mean(log_ml_values))
        print(f"\nLog marginal likelihood: {log_ml:.2f}")
    except Exception:
        pass

    return idata, cluster_df, ref_df, resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian CPT mixture")
    parser.add_argument("--draws",    type=int,  default=None, help="SMC particles per chain (overrides GlobalDraws)")
    parser.add_argument("--chains",   type=int,  default=None, help="number of independent SMC runs (overrides GlobalChains)")
    parser.add_argument("--clusters", type=int,  default=None, help="number of latent clusters C (overrides GlobalCluster)")
    parser.add_argument("--load",     action="store_true",     help="skip sampling and reload trace from GlobalTracePath")
    args = parser.parse_args()

    main(draws=args.draws, chains=args.chains, clusters=args.clusters, load=args.load)
