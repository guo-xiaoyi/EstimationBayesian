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
    GlobalDiscounting,
    GlobalMethod,
    GlobalDraws,
    GlobalChains,
    GlobalSeed,
    GlobalOutputDir,
    GlobalTracePath,
    GlobalSummaryCSV,
    GlobalPlotsDir,
    GlobalKsiMode,
    GlobalLotterySetName,
)
from preprocessing import load_and_preprocess
from likelihood import get_free_bounds, setup_likelihood
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


def _resolve_output_paths(output_dir=None):
    if output_dir is None:
        return GlobalTracePath, GlobalSummaryCSV, GlobalPlotsDir
    return (
        os.path.join(output_dir, "trace.nc"),
        os.path.join(output_dir, "summary.csv"),
        os.path.join(output_dir, "plots"),
    )


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
    bounds_rest = get_free_bounds(GlobalMethod)
    expected_rest_dim = len(bounds_rest)
    trace_rest_dim = idata.posterior["theta_rest"].shape[-1]
    if trace_rest_dim != expected_rest_dim:
        raise ValueError(
            f"Saved trace has theta_rest dimension {trace_rest_dim}, but the "
            f"current parameterisation expects {expected_rest_dim}. Re-run "
            "sampling without --load."
        )
    if "theta" in idata.posterior:
        expected_theta_dim = expected_rest_dim + 3
        trace_theta_dim = idata.posterior["theta"].shape[-1]
        if trace_theta_dim != expected_theta_dim:
            raise ValueError(
                f"Saved trace has theta dimension {trace_theta_dim}, but the "
                f"current parameterisation expects {expected_theta_dim}. Re-run "
                "sampling without --load."
            )
    lower_rest = np.array([b[0] for b in bounds_rest], dtype=float)
    upper_rest = np.array([b[1] for b in bounds_rest], dtype=float)
    fixed_mask = np.isclose(lower_rest, upper_rest)
    if np.any(fixed_mask):
        fixed_draws = idata.posterior["theta_rest"].values[..., fixed_mask]
        fixed_values = lower_rest[fixed_mask]
        if not np.allclose(fixed_draws, fixed_values, rtol=1e-8, atol=1e-8):
            raise ValueError(
                "Saved trace does not match the current fixed structural "
                "parameter bounds. Re-run sampling without --load."
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


def main(
    draws=None,
    chains=None,
    clusters=None,
    load=False,
    output_dir=None,
    seed=None,
    smc_cores=None,
    progressbar=None,
):
    # CLI overrides take precedence; fall back to GlobalSettings values.
    draws = GlobalDraws if draws is None else draws
    chains = GlobalChains if chains is None else chains
    C = clusters if clusters is not None else GlobalCluster
    seed = GlobalSeed if seed is None else seed
    trace_path, summary_csv, plots_dir = _resolve_output_paths(output_dir)

    os.makedirs(plots_dir, exist_ok=True)

    # --- Data ---
    print(f"Loading and preprocessing data (lottery_set={GlobalLotterySetName}) ...")
    preproc, subjects, subj_index, lotteries = load_and_preprocess()
    del subj_index  # only needed inside preprocessing
    print(f"  {len(subjects)} subjects, {preproc[0].shape[0]} observations")

    # Register the compiled likelihood op with the global lottery/data arrays so
    # PyTensor can call it inside the PyMC model graph.
    setup_likelihood(preproc, lotteries, subjects, GlobalMethod, C)

    # --- Sampling or loading ---
    if load and os.path.exists(trace_path):
        print(f"Loading trace from {trace_path} ...")
        idata = load_trace(trace_path)
    else:
        print(
            f"Building model (C={C} clusters, method={GlobalMethod}, "
            f"discounting={GlobalDiscounting}, ksi={GlobalKsiMode}) ..."
        )
        model = build_model(subjects, GlobalMethod, C)

        cores_label = smc_cores if smc_cores is not None else "auto"
        print(
            f"Sampling (SMC: particles={draws}, chains={chains}, "
            f"cores={cores_label}, seed={seed}) ..."
        )
        idata = run_sampling(
            model,
            draws=draws,
            chains=chains,
            seed=seed,
            cores=smc_cores,
            progressbar=progressbar,
        )

    # Catch mismatches between a saved trace and the current data/parameterisation.
    _validate_trace_matches_data(idata, len(subjects))

    # --- Post-processing ---
    # Sort clusters by lambda so labels represent low-to-high loss aversion.
    print("Fixing label switching ...")
    idata = fix_label_switching(idata, GlobalMethod, C)

    # Only write the trace after a fresh sampling run, not when reloading.
    if not (load and os.path.exists(trace_path)):
        save_trace(idata, trace_path)

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
    cluster_df.to_csv(summary_csv, index=False)
    print(f"Cluster summary saved -> {summary_csv}")

    ref_path = summary_csv.replace(".csv", "_reference_weights.csv")
    ref_df.to_csv(ref_path, index=False)
    print(f"Reference weights saved -> {ref_path}")

    print_summary(cluster_df, ref_df, resp, subjects)

    print("Saving posterior plots ...")
    plot_posteriors(idata, plots_dir, method=GlobalMethod)

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
    parser.add_argument("--seed",     type=int,  default=None, help="random seed (overrides GlobalSeed)")
    parser.add_argument("--smc-cores", type=int, default=None, help="cores used inside PyMC SMC for this run")
    parser.add_argument("--output-dir", default=None, help=f"directory for trace, summaries, and plots (default: {GlobalOutputDir})")
    parser.add_argument("--no-progress", action="store_true", help="disable PyMC progress bars")
    parser.add_argument("--load",     action="store_true",     help="skip sampling and reload trace from output-dir/trace.nc")
    args = parser.parse_args()

    main(
        draws=args.draws,
        chains=args.chains,
        clusters=args.clusters,
        load=args.load,
        output_dir=args.output_dir,
        seed=args.seed,
        smc_cores=args.smc_cores,
        progressbar=not args.no_progress,
    )
