"""
Post-sampling analysis for the Bayesian CPT mixture model with continuous,
cluster-specific reference-point weights.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "base_compiledir" not in _pytensor_flags:
    suffix = "base_compiledir=/tmp/pytensor"
    os.environ["PYTENSOR_FLAGS"] = f"{_pytensor_flags},{suffix}" if _pytensor_flags else suffix

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import logsumexp

from likelihood import (
    compute_marginal_ksi_subject_loglik,
    _compute_ce_th,
    _free_to_full,
    get_free_param_names,
    get_param_names,
    RP_COMPONENT_NAMES,
)


def fix_label_switching(idata, method, C=2):
    """
    Sort clusters in every draw by lambda, ascending.
    """
    if C < 2:
        return idata

    theta = idata.posterior["theta"].values
    theta_rest = idata.posterior["theta_rest"].values
    a_weights = idata.posterior["a_weights"].values
    pi = idata.posterior["pi"].values

    lamb_idx = get_param_names(method).index("lamb")
    order = np.argsort(theta[..., lamb_idx], axis=2)

    theta = np.take_along_axis(theta, order[..., None], axis=2)
    theta_rest = np.take_along_axis(theta_rest, order[..., None], axis=2)
    a_weights = np.take_along_axis(a_weights, order[..., None], axis=2)
    pi = np.take_along_axis(pi, order, axis=2)

    idata.posterior["theta"].values = theta
    idata.posterior["theta_rest"].values = theta_rest
    idata.posterior["a_weights"].values = a_weights
    idata.posterior["pi"].values = pi
    return idata


def _hdi_1d(values):
    return az.hdi(values[np.newaxis, :, np.newaxis], hdi_prob=0.94)[0]


def extract_cluster_params(idata, method, C):
    """
    Posterior mean and 94% HDI per cluster for pi, theta, and a4.
    """
    theta_draws = idata.posterior["theta"].values
    pi_draws = idata.posterior["pi"].values
    a_draws = idata.posterior["a_weights"].values

    n_chains, n_draws, _, P = theta_draws.shape
    S = n_chains * n_draws
    theta_flat = theta_draws.reshape(S, C, P)
    pi_flat = pi_draws.reshape(S, C)
    a_flat = a_draws.reshape(S, C, 4)

    names = get_param_names(method)
    rows = []

    for k in range(C):
        th_k = theta_flat[:, k, :]
        pi_k = pi_flat[:, k]
        a_k = a_flat[:, k, :]

        hdi_th = az.hdi(th_k[np.newaxis], hdi_prob=0.94)
        hdi_pi = _hdi_1d(pi_k)

        rows.append({
            "cluster": k,
            "param": "pi",
            "mean": float(pi_k.mean()),
            "hdi_low": float(hdi_pi[0]),
            "hdi_high": float(hdi_pi[1]),
        })
        for j, name in enumerate(names):
            rows.append({
                "cluster": k,
                "param": name,
                "mean": float(th_k[:, j].mean()),
                "hdi_low": float(hdi_th[j, 0]),
                "hdi_high": float(hdi_th[j, 1]),
            })

        hdi_a4 = _hdi_1d(a_k[:, 3])
        rows.append({
            "cluster": k,
            "param": "a4",
            "mean": float(a_k[:, 3].mean()),
            "hdi_low": float(hdi_a4[0]),
            "hdi_high": float(hdi_a4[1]),
        })

    return pd.DataFrame(rows)


def extract_reference_weights(idata, C):
    """
    Posterior mean and 94% HDI for a_weights by cluster and RP component.
    """
    a_draws = idata.posterior["a_weights"].values
    n_chains, n_draws = a_draws.shape[:2]
    a_flat = a_draws.reshape(n_chains * n_draws, C, 4)
    rows = []

    for k in range(C):
        for j, name in enumerate(RP_COMPONENT_NAMES):
            values = a_flat[:, k, j]
            hdi = _hdi_1d(values)
            rows.append({
                "cluster": k,
                "component": name,
                "mean": float(values.mean()),
                "hdi_low": float(hdi[0]),
                "hdi_high": float(hdi[1]),
            })
    return pd.DataFrame(rows)


def compute_cluster_responsibilities(idata, preproc, subjects, lotteries,
                                     method, C, max_draws=500):
    """
    Posterior-averaged soft cluster assignments.
    """
    (y_proc, EL_arr, EL_fe_arr, EL_c1_arr,
     t_arr, Z1_arr, Z2_arr, Zt_arr,
     si_arr, spread_arr, obs_arr) = preproc
    si_arr = si_arr.astype(int)

    N = len(subjects)
    theta_draws = idata.posterior["theta"].values
    pi_draws = idata.posterior["pi"].values
    has_ksi = "ksi" in idata.posterior
    if has_ksi:
        ksi_draws = idata.posterior["ksi"].values
        if ksi_draws.shape[-1] != N:
            raise ValueError(
                f"Trace has ksi dimension {ksi_draws.shape[-1]}, but current data has "
                f"{N} subjects. Re-run sampling instead of using this saved trace."
            )

    n_chains, n_draws = theta_draws.shape[:2]
    S = n_chains * n_draws

    rng = np.random.default_rng(1)
    idxs = rng.choice(S, size=min(max_draws, S), replace=False)

    theta_flat = theta_draws.reshape(S, C, -1)[idxs]
    pi_flat = pi_draws.reshape(S, C)[idxs]
    ksi_flat = ksi_draws.reshape(S, N)[idxs] if has_ksi else None

    resp_sum = np.zeros((N, C))

    for s in range(len(idxs)):
        if has_ksi:
            sig_arr = ksi_flat[s][si_arr] * spread_arr
            log_l = np.zeros((N, C))

            for k in range(C):
                params_full = _free_to_full(theta_flat[s, k], method)
                ce_th = _compute_ce_th(
                    params_full, method,
                    EL_arr, EL_fe_arr, EL_c1_arr,
                    t_arr, Z1_arr, Z2_arr, Zt_arr,
                    lotteries, y_proc,
                )
                log_pdf = norm.logpdf(obs_arr, loc=ce_th, scale=sig_arr)
                np.add.at(log_l[:, k], si_arr, log_pdf)
        else:
            log_l = compute_marginal_ksi_subject_loglik(
                theta_flat[s],
                method,
                preproc,
                lotteries,
                n_subjects=N,
            )

        log_pi = np.log(np.clip(pi_flat[s], 1e-300, None))
        log_terms = log_l + log_pi[np.newaxis, :]
        log_denom = logsumexp(log_terms, axis=1, keepdims=True)
        resp_sum += np.exp(log_terms - log_denom)

    return resp_sum / len(idxs)


def print_summary(cluster_df, ref_df, resp, subjects):
    print("\n" + "=" * 70)
    print("  BAYESIAN CPT MIXTURE - POSTERIOR SUMMARY")
    print("=" * 70)

    C = cluster_df["cluster"].nunique()
    for k in range(C):
        sub = cluster_df[cluster_df["cluster"] == k]
        pi_row = sub[sub["param"] == "pi"].iloc[0]
        print(f"\n  Cluster {k}  (pi = {pi_row['mean']:.3f}  "
              f"[{pi_row['hdi_low']:.3f}, {pi_row['hdi_high']:.3f}])")
        name_width = max(10, int(sub["param"].astype(str).str.len().max()))
        print(f"  {'Param':<{name_width}} {'Mean':>8}  {'94% HDI':>20}")
        print("  " + "-" * (name_width + 32))
        for _, row in sub[sub["param"] != "pi"].iterrows():
            print(f"  {row['param']:<{name_width}} {row['mean']:>8.4f}  "
                  f"[{row['hdi_low']:>7.4f}, {row['hdi_high']:>7.4f}]")

    print("\n  Reference-point weights by cluster:")
    print(f"  {'Cluster':>7} {'Component':<10} {'Mean':>8}  {'94% HDI':>20}")
    print("  " + "-" * 54)
    for _, row in ref_df.iterrows():
        print(f"  {int(row['cluster']):>7} {row['component']:<10} "
              f"{row['mean']:>8.4f}  "
              f"[{row['hdi_low']:>7.4f}, {row['hdi_high']:>7.4f}]")

    max_resp = resp.max(axis=1)
    assignment = resp.argmax(axis=1)
    uncertain = np.argsort(max_resp)[:5]
    print("\n  Most uncertain cluster assignments:")
    print(f"  {'Subject':<40} {'Cluster':>7}  {'Max P':>7}")
    print("  " + "-" * 58)
    for i in uncertain:
        print(f"  {str(subjects[i]):<40} {assignment[i]:>7}  {max_resp[i]:>7.3f}")
    print("=" * 70 + "\n")


def check_convergence(idata, method, C):
    """
    Per-parameter R-hat and ESS for theta_rest, a_weights, pi, and noise vars.
    """
    rest_names = get_free_param_names(method)
    a_names = ["a1", "a2", "a3", "a4"]
    posterior = idata.posterior
    estimate_ksi = "mu_ksi" in posterior

    rhat_vars = ["theta_rest", "a_weights"]
    if C > 1:
        rhat_vars.append("pi")
    if estimate_ksi:
        rhat_vars.extend(["mu_ksi", "ksi"])

    rhats = az.rhat(idata, var_names=rhat_vars)
    ess_vars = [v for v in rhat_vars if v != "ksi"]
    ess = az.ess(idata, var_names=ess_vars, method="bulk")
    n_chains = posterior.sizes["chain"]
    rows = []

    def _r4(value):
        value = float(value)
        return round(value, 4) if np.isfinite(value) else np.nan

    def _ess_cols(value):
        value = float(value)
        if not np.isfinite(value):
            return {"ess_bulk": np.nan, "ess_per_chain": np.nan}
        return {"ess_bulk": round(value, 1), "ess_per_chain": round(value / n_chains, 1)}

    theta_mean = posterior["theta_rest"].mean(dim=("chain", "draw")).values
    theta_rhat = rhats["theta_rest"].values
    theta_ess = ess["theta_rest"].values
    for k in range(C):
        for j, pname in enumerate(rest_names):
            rows.append({
                "cluster": k,
                "param": pname,
                "mean": _r4(theta_mean[k, j]),
                "r_hat": _r4(theta_rhat[k, j]),
                **_ess_cols(theta_ess[k, j]),
            })

    a_mean = posterior["a_weights"].mean(dim=("chain", "draw")).values
    a_rhat = rhats["a_weights"].values
    a_ess = ess["a_weights"].values
    for k in range(C):
        for j, aname in enumerate(a_names):
            rows.append({
                "cluster": k,
                "param": aname,
                "mean": _r4(a_mean[k, j]),
                "r_hat": _r4(a_rhat[k, j]),
                **_ess_cols(a_ess[k, j]),
            })

    if C == 1:
        rows.append({
            "cluster": 0,
            "param": "pi",
            "mean": _r4(float(posterior["pi"].values.mean())),
            "r_hat": np.nan,
            "ess_bulk": np.nan,
            "ess_per_chain": np.nan,
        })
    else:
        pi_mean = posterior["pi"].mean(dim=("chain", "draw")).values
        pi_rhat = rhats["pi"].values
        pi_ess = ess["pi"].values
        for k in range(C):
            rows.append({
                "cluster": k,
                "param": "pi",
                "mean": _r4(pi_mean[k]),
                "r_hat": _r4(pi_rhat[k]),
                **_ess_cols(pi_ess[k]),
            })

    if estimate_ksi:
        rows.append({
            "cluster": None,
            "param": "mu_ksi",
            "mean": _r4(float(posterior["mu_ksi"].mean(dim=("chain", "draw")).values)),
            "r_hat": _r4(float(rhats["mu_ksi"].values)),
            **_ess_cols(float(ess["mu_ksi"].values)),
        })
        ksi_rhat = rhats["ksi"].values
        ksi_mean = posterior["ksi"].mean(dim=("chain", "draw")).values
        rows.append({
            "cluster": None,
            "param": "ksi (mean R-hat)",
            "mean": _r4(np.nanmean(ksi_mean)),
            "r_hat": _r4(np.nanmean(ksi_rhat)),
            "ess_bulk": np.nan,
            "ess_per_chain": np.nan,
        })
        rows.append({
            "cluster": None,
            "param": "ksi (worst R-hat)",
            "mean": float("nan"),
            "r_hat": _r4(np.nanmax(ksi_rhat)),
            "ess_bulk": np.nan,
            "ess_per_chain": np.nan,
        })
    elif "ksi" in posterior:
        ksi_mean = posterior["ksi"].mean(dim=("chain", "draw")).values
        rows.append({
            "cluster": None,
            "param": "ksi (fixed)",
            "mean": _r4(np.nanmean(ksi_mean)),
            "r_hat": np.nan,
            "ess_bulk": np.nan,
            "ess_per_chain": np.nan,
        })
    else:
        rows.append({
            "cluster": None,
            "param": "ksi (marginalized)",
            "mean": np.nan,
            "r_hat": np.nan,
            "ess_bulk": np.nan,
            "ess_per_chain": np.nan,
        })

    return pd.DataFrame(rows)


def plot_posteriors(idata, output_dir="outputs/plots", method="tk"):
    os.makedirs(output_dir, exist_ok=True)

    trace_vars = ["theta", "a_weights"]
    if idata.posterior["pi"].shape[-1] > 1:
        trace_vars.insert(1, "pi")
    if "mu_ksi" in idata.posterior:
        trace_vars.append("mu_ksi")

    for var in trace_vars:
        axes = az.plot_trace(idata, var_names=[var])
        fig = axes.ravel()[0].get_figure()
        path = os.path.join(output_dir, f"trace_{var}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved -> {path}")

    theta = idata.posterior["theta"].values
    _, _, C, P = theta.shape
    theta_flat = theta.reshape(-1, C, P)
    names = get_param_names(method)

    fig, axes = plt.subplots(P, C, figsize=(4 * C, 2 * P), squeeze=False)
    for k in range(C):
        for j, name in enumerate(names):
            ax = axes[j][k]
            ax.hist(theta_flat[:, k, j], bins=50, density=True, alpha=0.7,
                    color=f"C{k}", edgecolor="none")
            ax.set_title(f"Cluster {k} - {name}", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    path = os.path.join(output_dir, "posterior_theta.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved -> {path}")
