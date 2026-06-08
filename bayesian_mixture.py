"""
Bayesian hierarchical finite-mixture model for CPT parameters.

Reference points are represented by continuous, cluster-specific weights:

    a_weights[k] = [a_SQ, a_PA, a_LE, a_FE],  k = 1, ..., C

The full cluster parameter vector is assembled as:

    theta_k = [r, alpha, lamb, gamma/palpha, a1, a2, a3, delta]

where a4 is stored in a_weights but omitted from theta because it is the
forward-looking component on the simplex. Structural parameters have a
hierarchical truncated LogNormal prior on their admissible bounds.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "base_compiledir" not in _pytensor_flags:
    suffix = "base_compiledir=/tmp/pytensor"
    os.environ["PYTENSOR_FLAGS"] = f"{_pytensor_flags},{suffix}" if _pytensor_flags else suffix

import numpy as np
import pytensor.tensor as pt
import pymc as pm
import arviz as az
import xarray as xr

from likelihood import cpt_loglik_marginal_ksi_op, cpt_loglik_op, get_free_bounds
from GlobalSettings import (
    GlobalPriorDirichlet,
    GlobalPriorDirichletRP,
    GlobalPriorKsiSigma,
    GlobalPriorIG_Alpha,
    GlobalPriorIG_Beta,
    GlobalPriorMu0,
    GlobalFixedKsi,
    GlobalKsiMode,
    GlobalSMCCores,
)


def _benchmark_mu0(lower, upper):
    """Return log-scale benchmark means for [r, alpha, lamb, gamma/palpha, delta]."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mu0 = np.empty(len(lower))
    for j in range(len(lower)):
        if GlobalPriorMu0[j] is not None:
            mu0[j] = GlobalPriorMu0[j]
        else:
            safe_lo = max(lower[j], 1e-8)
            mu0[j] = 0.5 * (np.log(safe_lo) + np.log(upper[j]))
    return mu0


def build_model(subjects, method, C, K=None):
    """
    Construct the PyMC model.

    K is accepted for backwards compatibility; continuous reference-point
    weights always use the four canonical components SQ, PA, LE, and FE.
    """
    if K not in (None, 4):
        raise ValueError("Continuous reference-point weights require K=4.")

    N = len(subjects)
    bounds_rest = get_free_bounds(method)
    lower_rest = np.array([b[0] for b in bounds_rest])
    upper_rest = np.array([b[1] for b in bounds_rest])
    lower_for_prior = np.maximum(lower_rest, 1e-6)
    mu0 = _benchmark_mu0(lower_for_prior, upper_rest)

    with pm.Model() as model:
        pi = pm.Dirichlet("pi", a=np.ones(C) * GlobalPriorDirichlet)

        sigma_G_sq = pm.InverseGamma(
            "sigma_G_sq",
            alpha=GlobalPriorIG_Alpha,
            beta=GlobalPriorIG_Beta,
            shape=5,
        )
        sigma_G = pm.Deterministic("sigma_G", pt.sqrt(sigma_G_sq))
        mu_G = pm.Normal("mu_G", mu=mu0, sigma=sigma_G, shape=5)

        theta_rest_dist = pm.LogNormal.dist(
            mu=mu_G,
            sigma=sigma_G,
            shape=(C, 5),
        )
        theta_rest = pm.Truncated(
            "theta_rest",
            theta_rest_dist,
            lower=lower_for_prior,
            upper=upper_rest,
            shape=(C, 5),
        )

        # One continuous reference-point weight vector per cluster.
        a_weights = pm.Dirichlet(
            "a_weights",
            a=np.ones(4) * GlobalPriorDirichletRP,
            shape=(C, 4),
        )

        theta = pm.Deterministic(
            "theta",
            pt.concatenate(
                [theta_rest[:, :4], a_weights[:, :3], theta_rest[:, 4:]],
                axis=1,
            ),
        )

        if GlobalKsiMode == "estimated":
            mu_ksi = pm.HalfNormal("mu_ksi", sigma=GlobalPriorKsiSigma)
            ksi_raw = pm.HalfNormal("ksi_raw", sigma=1.0, shape=N)
            ksi = pm.Deterministic("ksi", mu_ksi * ksi_raw)
            pm.Potential("ll", cpt_loglik_op(theta, pi, ksi))
        elif GlobalKsiMode == "fixed":
            ksi = pm.Deterministic("ksi", pt.ones(N) * GlobalFixedKsi)
            pm.Potential("ll", cpt_loglik_op(theta, pi, ksi))
        elif GlobalKsiMode == "marginalized":
            pm.Potential("ll", cpt_loglik_marginal_ksi_op(theta, pi))
        else:
            raise ValueError(
                "GlobalKsiMode must be one of: 'estimated', 'fixed', 'marginalized'."
            )

    return model


def run_sampling(model, draws=2000, chains=4, seed=42):
    """Sample from the posterior using PyMC Sequential Monte Carlo."""
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
    """Pad ragged SMC stage-diagnostic arrays so NetCDF can serialise them."""
    if not hasattr(idata, "sample_stats"):
        return idata

    n_chains = (
        idata.posterior.sizes.get("chain")
        if hasattr(idata, "posterior")
        else None
    )

    def _has_chain_conflict(da):
        return (
            "chain" in da.dims
            and n_chains is not None
            and da.sizes.get("chain") != n_chains
        )

    needs_rewrite = any(
        da.values.dtype == object or _has_chain_conflict(da)
        for da in idata.sample_stats.data_vars.values()
    )
    if not needs_rewrite:
        return idata

    safe = idata.copy()
    coords = {}
    data_vars = {}
    stage_vars = {}
    max_stage = 0

    def _numeric_like(da, sequences):
        values = np.full(da.shape, np.nan, dtype=float)
        values_flat = values.reshape(-1)
        for i, seq in enumerate(sequences):
            if len(seq):
                values_flat[i] = seq[0]
        da_coords = {dim: da.coords[dim] for dim in da.dims if dim in da.coords}
        return xr.DataArray(values, dims=da.dims, coords=da_coords, attrs=da.attrs)

    def _chain_sequences(da):
        chain_axis = da.get_axis_num("chain")
        by_chain = np.moveaxis(da.values, chain_axis, 0)
        sequences = []
        for chain_values in by_chain:
            parts = [
                _as_float_sequence(value)
                for value in np.asarray(chain_values, dtype=object).reshape(-1)
            ]
            sequences.append(
                np.concatenate(parts) if parts else np.array([], dtype=float)
            )
        return sequences

    def _add_stage_var(name, first_dim, sequences, attrs):
        nonlocal max_stage
        stage_vars[name] = (first_dim, sequences, attrs)
        max_stage = max(max_stage, max((len(seq) for seq in sequences), default=0))

    for name, da in idata.sample_stats.data_vars.items():
        if da.values.dtype == object:
            seqs = [_as_float_sequence(v) for v in da.values.reshape(-1)]
            scalar_like = all(len(seq) <= 1 for seq in seqs)
            if scalar_like and not _has_chain_conflict(da):
                data_vars[name] = _numeric_like(da, seqs)
            elif (
                "chain" in da.dims
                and n_chains is not None
                and da.sizes.get("chain") == n_chains
            ):
                _add_stage_var(name, "chain", _chain_sequences(da), da.attrs)
            elif n_chains and len(seqs) % n_chains == 0:
                n_stages = len(seqs) // n_chains
                values = np.array(
                    [seq[0] if len(seq) else np.nan for seq in seqs],
                    dtype=float,
                ).reshape(n_chains, n_stages)
                _add_stage_var(name, "chain", list(values), da.attrs)
            else:
                _add_stage_var(name, f"{name}_entry", seqs, da.attrs)
        elif _has_chain_conflict(da):
            # log_marginal_likelihood is stored flat as (n_chains * n_stages,);
            # reshape to (n_chains, n_stages) so the 'chain' coord stays consistent.
            flat = da.values.ravel()
            if n_chains and len(flat) % n_chains == 0:
                n_stages = len(flat) // n_chains
                reshaped = flat.reshape(n_chains, n_stages)
                _add_stage_var(name, "chain", list(reshaped), da.attrs)
            else:
                data_vars[name] = (
                    (f"{name}_entry",),
                    flat,
                    dict(da.attrs),
                )
                coords[f"{name}_entry"] = np.arange(len(flat))
        else:
            data_vars[name] = da

    for name, (first_dim, seqs, attrs) in stage_vars.items():
        padded = np.full((len(seqs), max_stage), np.nan, dtype=float)
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = seq
        data_vars[name] = (
            (first_dim, "smc_stage"),
            padded,
            dict(attrs),
        )
        if first_dim != "chain":
            coords[first_dim] = np.arange(len(seqs))

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
    print(f"Trace saved -> {path}")


def load_trace(path="outputs/trace.nc"):
    idata = az.from_netcdf(path)
    stats = set(idata.sample_stats.data_vars) if hasattr(idata, "sample_stats") else set()
    if "log_marginal_likelihood" not in stats:
        print("WARNING: loaded trace has no SMC log_marginal_likelihood.")
    return idata
