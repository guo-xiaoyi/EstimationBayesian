"""
Bayesian hierarchical finite-mixture model for CPT parameters.

Model
-----
Weakly informative Normal-Inverse-Gamma-style hierarchical prior.
Cluster parameters are drawn from a population LogNormal whose own parameters
are random, regularised toward theoretically neutral benchmarks.

    π          ~ Dirichlet(α₀, …, α₀)                mixing weights        (C,)

    — Normal-Inverse-Gamma-style hyperprior on (μ_G, σ_G²) ————————————————
    σ_G²[j]    ~ InverseGamma(α_IG, β_IG)             variance hyperprior   (5,)
    μ_G[j]     ~ Normal(μ_0[j], σ_G[j])               mean hyperprior       (5,)
      μ_0 benchmarks (log-scale): log(1)=0 for α,λ,γ/pα,δ (linearity /
      no-distortion / no-loss-aversion); log-geometric-midpoint for r.

    θ_rest_k   ~ TruncatedLogNormal(μ_G, σ_G; bounds) [r,α,λ,γ/pα,δ]       (C, 5)
    a_k        ~ Dirichlet(1, 1, 1, 1)                ref-point weights     (C, 4)
    θ_k        = [r,α,λ,γ/pα, a1,a2,a3, δ]           assembled             (C, 8)

    if GlobalEstimateKsi:
      μ_κ      ~ HalfNormal(σ_prior)                  noise hyperprior
      κ_raw_i  ~ HalfNormal(1)                        non-centered noise     (N,)
      κ_i      = μ_κ · κ_raw_i                        individual noise       (N,)
    else:
      κ_i      = GlobalFixedKsi                        fixed noise            (N,)

    log p(data | θ, π, κ) attached as a pm.Potential via the blackbox op
    from likelihood.py (no gradient required by SMC).

Reference-point weights (a1, a2, a3, a4):
    a_k ~ Dirichlet(1,1,1,1) gives a uniform prior on the 3-simplex,
    treating all four reference-point types symmetrically.
    a4 = 1 - a1 - a2 - a3 (forward-looking residual) is implicit.

Sampler: pm.sample_smc
    Sequential Monte Carlo with tempered likelihood stages.
    Gradient-free and better suited than random-walk MCMC for multimodal
    mixture posteriors.

Usage
-----
    Call setup_likelihood() from likelihood.py first, then:

        model = build_model(subjects, method, C)
        idata = run_sampling(model, draws=2000, chains=4)
        save_trace(idata, "outputs/trace.nc")
"""

import numpy as np
import pytensor.tensor as pt
import pymc as pm
import arviz as az
import xarray as xr

from likelihood import cpt_loglik_op, get_free_bounds
from GlobalSettings import (
    GlobalPriorDirichlet, GlobalPriorKsiSigma,
    GlobalPriorIG_Alpha, GlobalPriorIG_Beta, GlobalPriorMu0,
    GlobalEstimateKsi, GlobalFixedKsi,
    GlobalSMCCores,
)


def _benchmark_mu0(lower, upper):
    """
    Return log-scale benchmark means μ_0 for [r, alpha, lamb, gamma/palpha, delta].

    Uses GlobalPriorMu0 values where defined; falls back to the log geometric
    midpoint (the least-informative choice on a log scale) for the discount-rate
    parameter r (index 0), whose admissible range is method-dependent.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mu0 = np.empty(len(lower))
    for j in range(len(lower)):
        if GlobalPriorMu0[j] is not None:
            mu0[j] = GlobalPriorMu0[j]
        else:
            # geometric midpoint in log-space (used for r)
            safe_lo = max(lower[j], 1e-8)
            mu0[j] = 0.5 * (np.log(safe_lo) + np.log(upper[j]))
    return mu0


def build_model(subjects, method, C):
    """
    Construct the PyMC model.

    Parameters
    ----------
    subjects : list of subject labels (length N)
    method   : "prelec" or "tk"
    C        : number of clusters

    Returns
    -------
    pm.Model

    Variables in the model
    ----------------------
    pi         : (C,)    Dirichlet — mixture weights
    sigma_G_sq : (5,)    InverseGamma — variance hyperprior for log-scale CPT params
    sigma_G    : (5,)    Deterministic — sqrt(sigma_G_sq)
    mu_G       : (5,)    Normal(mu0, sigma_G) — mean hyperprior for log-scale CPT params
    theta_rest : (C, 5)  Truncated LogNormal(mu_G, sigma_G) — [r, alpha, lamb, gamma/palpha, delta]
    a_weights  : (C, 4)  Dirichlet — reference-point weights [a1,a2,a3,a4]
    theta      : (C, 8)  Deterministic — assembled full param vector
    mu_ksi     : scalar  HalfNormal, only when GlobalEstimateKsi=True
    ksi_raw    : (N,)    HalfNormal, only when GlobalEstimateKsi=True
    ksi        : (N,)    Deterministic individual noise
    """
    N           = len(subjects)
    bounds_rest = get_free_bounds(method)   # 5 bounds: [r, alpha, lamb, gamma/palpha, delta]
    lower_rest  = np.array([b[0] for b in bounds_rest])
    upper_rest  = np.array([b[1] for b in bounds_rest])
    lower_rest_for_prior = np.maximum(lower_rest, 1e-6)
    mu0 = _benchmark_mu0(lower_rest, upper_rest)

    with pm.Model() as model:
        # ── Mixture weights ───────────────────────────────────────────────────
        pi = pm.Dirichlet("pi", a=np.ones(C) * GlobalPriorDirichlet)

        # ── Normal-Inverse-Gamma-style hyperprior on (μ_G, σ_G²) ─────────────
        # σ_G²[j] ~ InverseGamma(α, β)   E[σ_G²] = β/(α-1) = 1.0
        #   Governs how spread out cluster parameters are in log-space.
        #   Large prior scale keeps the prior diffuse while excluding σ → 0.
        sigma_G_sq = pm.InverseGamma(
            "sigma_G_sq",
            alpha=GlobalPriorIG_Alpha,
            beta=GlobalPriorIG_Beta,
            shape=5,
        )
        sigma_G = pm.Deterministic("sigma_G", pt.sqrt(sigma_G_sq))

        # μ_G[j] | σ_G[j] ~ Normal(μ_0[j], σ_G[j])
        #   Centered at theoretically neutral benchmarks (log(1)=0 for
        #   linearity / no-distortion / no-loss-aversion; log-midpoint for r).
        #   Variance inherits from the same σ_G drawn above.
        mu_G = pm.Normal("mu_G", mu=mu0, sigma=sigma_G, shape=5)

        # ── Structural CPT parameters (non-simplex) ───────────────────────────
        # θ_rest_k[j] ~ TruncatedLogNormal(μ_G[j], σ_G[j])
        # [r, alpha, lamb, gamma (TK) / palpha (Prelec), delta]
        theta_rest_dist = pm.LogNormal.dist(
            mu=mu_G,
            sigma=sigma_G,
            shape=(C, 5),
        )
        theta_rest = pm.Truncated(
            "theta_rest",
            theta_rest_dist,
            lower=lower_rest_for_prior,
            upper=upper_rest,
            shape=(C, 5),
        )

        # ── Reference-point weights ───────────────────────────────────────────
        # Uniform prior on the 3-simplex: Dirichlet(1,1,1,1).
        # a4 = 1-a1-a2-a3 (forward-looking residual) is the 4th component.
        a_weights = pm.Dirichlet("a_weights", a=np.ones(4), shape=(C, 4))

        # ── Assemble full parameter matrix (C, 8) ─────────────────────────────
        # [r, alpha, lamb, gamma/palpha] + [a1, a2, a3] + [delta]
        theta = pm.Deterministic(
            "theta",
            pt.concatenate(
                [theta_rest[:, :4], a_weights[:, :3], theta_rest[:, 4:]], axis=1
            ),
        )

        # ── Noise scale κ_i ──────────────────────────────────────────────────
        if GlobalEstimateKsi:
            # Non-centered hierarchical noise: μ_κ → κ_i
            mu_ksi = pm.HalfNormal("mu_ksi", sigma=GlobalPriorKsiSigma)
            ksi_raw = pm.HalfNormal("ksi_raw", sigma=1.0, shape=N)
            ksi = pm.Deterministic("ksi", mu_ksi * ksi_raw)
        else:
            ksi = pm.Deterministic("ksi", pt.ones(N) * GlobalFixedKsi)

        # ── CPT mixture log-likelihood ────────────────────────────────────────
        pm.Potential("ll", cpt_loglik_op(theta, pi, ksi))

    return model


def run_sampling(model, draws=2000, chains=4, seed=42):
    """
    Sample from the posterior using Sequential Monte Carlo (SMC).

    SMC is better suited than MCMC for multimodal posteriors (mixture models):
    particles are initialised from the prior and the likelihood is introduced
    gradually (β: 0→1), so all modes are discovered before they can be missed.

    A key byproduct is log_marginal_likelihood, which enables direct Bayes
    Factor comparison across different values of C:
        idata.sample_stats["log_marginal_likelihood"]

    Parameters
    ----------
    model  : pm.Model from build_model()
    draws  : number of SMC particles — more particles → better approximation
    chains : independent SMC runs (for diagnostics; each is a full SMC pass)
    seed   : random seed

    Returns
    -------
    arviz.InferenceData  (posterior + sample_stats incl. log_marginal_likelihood)
    """
    cores = chains if GlobalSMCCores is None else min(chains, GlobalSMCCores)

    with model:
        idata = pm.sample_smc(
            draws=draws,
            chains=chains,
            random_seed=seed,
            progressbar=True,
            cores=cores,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )
    return idata


def _as_float_sequence(value):
    """Convert scalar/list/array SMC stats to a flat finite-or-nan float vector."""
    if isinstance(value, (list, tuple)):
        parts = [_as_float_sequence(item) for item in value]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    if isinstance(value, np.ndarray) and value.dtype == object:
        parts = [_as_float_sequence(item) for item in value.flat]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    try:
        return np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _netcdf_safe_idata(idata):
    """
    Convert ragged SMC sample_stats object arrays into padded numeric arrays.

    PyMC stores SMC stage diagnostics as one variable-length sequence per chain.
    NetCDF cannot serialize those object/list arrays directly, so we preserve
    them as float arrays padded with NaN over a common smc_stage dimension.
    """
    if not hasattr(idata, "sample_stats"):
        return idata

    needs_rewrite = any(
        data_array.values.dtype == object
        for data_array in idata.sample_stats.data_vars.values()
    )
    if not needs_rewrite:
        return idata

    safe = idata.copy()
    coords = {}
    data_vars = {}
    object_sequences = {}
    max_stage = 0

    for name, data_array in idata.sample_stats.data_vars.items():
        values = data_array.values
        if values.dtype != object:
            data_vars[name] = data_array
            continue

        sequences = [_as_float_sequence(item) for item in values.reshape(-1)]
        object_sequences[name] = sequences
        max_stage = max(max_stage, max((len(seq) for seq in sequences), default=0))

    for name, sequences in object_sequences.items():
        padded = np.full((len(sequences), max_stage), np.nan, dtype=float)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq

        data_vars[name] = (("chain", "smc_stage"), padded)

    n_chains = safe.posterior.sizes.get("chain", None)
    if n_chains is not None:
        coords["chain"] = np.arange(n_chains)
    if max_stage:
        coords["smc_stage"] = np.arange(max_stage)

    sample_stats = xr.Dataset(data_vars=data_vars, coords=coords)
    sample_stats.attrs.update(idata.sample_stats.attrs)
    delattr(safe, "sample_stats")
    safe.add_groups({"sample_stats": sample_stats})
    return safe


def save_trace(idata, path="outputs/trace.nc"):
    idata = _netcdf_safe_idata(idata)
    idata.to_netcdf(path)
    print(f"Trace saved → {path}")


def load_trace(path="outputs/trace.nc"):
    idata = az.from_netcdf(path)
    stats = set(idata.sample_stats.data_vars) if hasattr(idata, "sample_stats") else set()
    if "log_marginal_likelihood" not in stats:
        print(
            "WARNING: loaded trace has no SMC log_marginal_likelihood; "
            "it may be an older non-SMC trace."
        )
    return idata
