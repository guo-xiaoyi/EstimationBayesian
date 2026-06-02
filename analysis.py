"""
Post-sampling analysis for the Bayesian CPT mixture model.

Functions
---------
fix_label_switching(idata, method, C)
    Reorder clusters in every draw so cluster 0 always has the smaller lambda.

extract_cluster_params(idata, method, C)
    Posterior mean ± 94 % HDI per cluster, returned as a DataFrame.
    Includes a4 (forward-looking weight) read directly from a_weights.

compute_responsibilities(idata, preproc, subjects, lotteries, method, C)
    Posterior-averaged soft cluster assignments (N × C matrix).

print_summary(cluster_df, resp, subjects)
    Formatted console output.

plot_posteriors(idata, output_dir, method)
    Save ArviZ trace plots for all variables.
"""

import os
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import logsumexp

from likelihood import _compute_ce_th, _free_to_full, PARAM_NAMES


def fix_label_switching(idata, method, C=2):
    """
    For each posterior draw, sort clusters so cluster 0 has the smaller lambda
    (index 2 in the free-param vector for both Prelec and TK).
    Modifies idata.posterior["theta"], ["theta_rest"], ["a_weights"], ["pi"] in-place.
    Only implemented for C=2; no-op for C>2.
    """
    if C != 2:
        return idata

    theta = idata.posterior["theta"].values         # (chains, draws, C, 8)
    pi    = idata.posterior["pi"].values            # (chains, draws, C)
    theta_rest = idata.posterior["theta_rest"].values  # (chains, draws, C, 5)
    a_weights  = idata.posterior["a_weights"].values   # (chains, draws, C, 4)

    lamb_idx = 2   # index of lambda in the assembled theta vector
    mask = theta[:, :, 0, lamb_idx] > theta[:, :, 1, lamb_idx]

    theta[mask]      = theta[mask][:, [1, 0], :]
    pi[mask]         = pi[mask][:, [1, 0]]
    theta_rest[mask] = theta_rest[mask][:, [1, 0], :]
    a_weights[mask]  = a_weights[mask][:, [1, 0], :]

    idata.posterior["theta"].values      = theta
    idata.posterior["pi"].values         = pi
    idata.posterior["theta_rest"].values = theta_rest
    idata.posterior["a_weights"].values  = a_weights
    return idata


def extract_cluster_params(idata, method, C=2):
    """
    Return a DataFrame with posterior mean and 94 % HDI for each cluster.

    Columns: cluster | param | mean | hdi_low | hdi_high

    Params reported:
      - pi
      - All 8 entries of PARAM_NAMES[method]: r, alpha, lamb, gamma/palpha,
        a1, a2, a3, delta  (a1,a2,a3 come directly from a_weights[:,:3])
      - a4  (the forward-looking weight — 4th column of a_weights)
    """
    theta_draws     = idata.posterior["theta"].values       # (chains, draws, C, 8)
    pi_draws        = idata.posterior["pi"].values          # (chains, draws, C)
    a_weights_draws = idata.posterior["a_weights"].values   # (chains, draws, C, 4)

    n_chains, n_draws, _, P = theta_draws.shape
    S = n_chains * n_draws
    theta_flat     = theta_draws.reshape(S, C, P)
    pi_flat        = pi_draws.reshape(S, C)
    a_weights_flat = a_weights_draws.reshape(S, C, 4)

    names = PARAM_NAMES[method]
    rows  = []

    for k in range(C):
        th_k = theta_flat[:, k, :]      # (S, 8)
        pi_k = pi_flat[:, k]            # (S,)
        aw_k = a_weights_flat[:, k, :]  # (S, 4)

        hdi_th = az.hdi(th_k[np.newaxis], hdi_prob=0.94)          # (8, 2)
        hdi_pi = az.hdi(pi_k[np.newaxis, :, np.newaxis], hdi_prob=0.94)[0]

        rows.append({
            "cluster":  k,
            "param":    "pi",
            "mean":     float(pi_k.mean()),
            "hdi_low":  float(hdi_pi[0]),
            "hdi_high": float(hdi_pi[1]),
        })
        for j, name in enumerate(names):
            rows.append({
                "cluster":  k,
                "param":    name,
                "mean":     float(th_k[:, j].mean()),
                "hdi_low":  float(hdi_th[j, 0]),
                "hdi_high": float(hdi_th[j, 1]),
            })

        # a4: forward-looking residual weight, directly from a_weights
        a4 = aw_k[:, 3]
        hdi_a4 = az.hdi(a4[np.newaxis, :, np.newaxis], hdi_prob=0.94)[0]
        rows.append({
            "cluster":  k,
            "param":    "a4",
            "mean":     float(a4.mean()),
            "hdi_low":  float(hdi_a4[0]),
            "hdi_high": float(hdi_a4[1]),
        })

    return pd.DataFrame(rows)


def compute_responsibilities(idata, preproc, subjects, lotteries, method, C):
    """
    Posterior-averaged soft cluster assignments.

    For each posterior draw (θ, π, κ), compute:
        r_{ik} = π_k · L_ik / Σ_j π_j L_ij

    and average over draws.  Returns (N × C) array.
    """
    (y_proc, EL_arr, EL_fe_arr, EL_c1_arr,
     t_arr, Z1_arr, Z2_arr, Zt_arr,
     si_arr, spread_arr, obs_arr) = preproc
    si_arr = si_arr.astype(int)

    N = len(subjects)

    theta_draws = idata.posterior["theta"].values   # (chains, draws, C, 8)
    pi_draws    = idata.posterior["pi"].values
    ksi_draws   = idata.posterior["ksi"].values     # (chains, draws, N)

    n_chains, n_draws = theta_draws.shape[:2]
    S = n_chains * n_draws

    theta_flat = theta_draws.reshape(S, C, -1)
    pi_flat    = pi_draws.reshape(S, C)
    ksi_flat   = ksi_draws.reshape(S, N)

    resp_sum = np.zeros((N, C))

    for s in range(S):
        th  = theta_flat[s]    # (C, 8) — assembled theta (a1,a2,a3 already in place)
        pi  = pi_flat[s]       # (C,)
        ksi = ksi_flat[s]      # (N,)

        ksi_row = ksi[si_arr]
        sig_arr = ksi_row * spread_arr

        log_L = np.zeros((N, C))
        for k in range(C):
            params_full = _free_to_full(th[k], method)
            ce_th = _compute_ce_th(
                params_full, method,
                EL_arr, EL_fe_arr, EL_c1_arr,
                t_arr, Z1_arr, Z2_arr, Zt_arr,
                lotteries, y_proc,
            )
            log_pdf = norm.logpdf(obs_arr, loc=ce_th, scale=sig_arr)
            np.add.at(log_L[:, k], si_arr, log_pdf)

        log_pi    = np.log(np.clip(pi, 1e-300, None))
        log_terms = log_L + log_pi[np.newaxis, :]
        log_denom = logsumexp(log_terms, axis=1, keepdims=True)
        resp_s    = np.exp(log_terms - log_denom)
        resp_sum += resp_s

    return resp_sum / S


def print_summary(cluster_df, resp, subjects):
    """Print cluster parameter table and top-5 hardest-to-assign subjects."""
    print("\n" + "=" * 70)
    print("  BAYESIAN CPT MIXTURE — POSTERIOR SUMMARY")
    print("=" * 70)

    C = cluster_df["cluster"].nunique()
    for k in range(C):
        sub = cluster_df[cluster_df["cluster"] == k]
        pi_row = sub[sub["param"] == "pi"].iloc[0]
        print(f"\n  Cluster {k}  (π = {pi_row['mean']:.3f}  "
              f"[{pi_row['hdi_low']:.3f}, {pi_row['hdi_high']:.3f}])")
        print(f"  {'Param':<10} {'Mean':>8}  {'94% HDI':>20}")
        print("  " + "-" * 42)
        for _, row in sub[sub["param"] != "pi"].iterrows():
            print(f"  {row['param']:<10} {row['mean']:>8.4f}  "
                  f"[{row['hdi_low']:>7.4f}, {row['hdi_high']:>7.4f}]")

    max_resp   = resp.max(axis=1)
    assignment = resp.argmax(axis=1)
    uncertain  = np.argsort(max_resp)[:5]
    print(f"\n  {'Subject':<40} {'Cluster':>7}  {'Max P':>7}")
    print("  " + "-" * 58)
    for i in uncertain:
        print(f"  {subjects[i]:<40} {assignment[i]:>7}  {max_resp[i]:>7.3f}")
    print("=" * 70 + "\n")


def check_convergence(idata, method, C):
    """
    Per-parameter R-hat diagnostics with interpretable names.

    Checks sampled variables such as theta_rest, a_weights, pi, and optionally
    mu_ksi/ksi when the noise scale is estimated.

    Returns a DataFrame with columns:
        cluster | param | mean | r_hat | ess_bulk | ess_per_chain
    ksi is summarised across subjects with mean + worst-case R-hat.
    """
    names      = PARAM_NAMES[method]
    rest_names = [n for n in names if n not in ("a1", "a2", "a3")]  # [r,α,λ,γ/pα,δ]
    a_names    = ["a1", "a2", "a3", "a4"]

    rows = []
    posterior = idata.posterior
    estimate_ksi = "mu_ksi" in posterior
    rhat_vars = ["theta_rest", "a_weights"]
    if C > 1:
        rhat_vars.append("pi")
    if estimate_ksi:
        rhat_vars.extend(["mu_ksi", "ksi"])
    rhats = az.rhat(idata, var_names=rhat_vars)
    ess_vars = ["theta_rest", "a_weights"]
    if C > 1:
        ess_vars.append("pi")
    if estimate_ksi:
        ess_vars.append("mu_ksi")
    ess = az.ess(idata, var_names=ess_vars, method="bulk")
    n_chains = posterior.sizes["chain"]

    def _round4(value):
        value = float(value)
        return round(value, 4) if np.isfinite(value) else np.nan

    def _ess_cols(value):
        value = float(value)
        if not np.isfinite(value):
            return {"ess_bulk": np.nan, "ess_per_chain": np.nan}
        return {
            "ess_bulk": round(value, 1),
            "ess_per_chain": round(value / n_chains, 1),
        }

    # ── theta_rest (C, 5): non-simplex structural parameters ─────────────────
    theta_mean = posterior["theta_rest"].mean(dim=("chain", "draw")).values
    theta_rhat = rhats["theta_rest"].values
    theta_ess = ess["theta_rest"].values
    for k in range(C):
        for j, pname in enumerate(rest_names):
            rows.append({"cluster": k, "param": pname,
                         "mean": _round4(theta_mean[k, j]),
                         "r_hat": _round4(theta_rhat[k, j]),
                         **_ess_cols(theta_ess[k, j])})

    # ── a_weights (C, 4): reference-point weights ─────────────────────────────
    a_mean = posterior["a_weights"].mean(dim=("chain", "draw")).values
    a_rhat = rhats["a_weights"].values
    a_ess = ess["a_weights"].values
    for k in range(C):
        for j, aname in enumerate(a_names):
            rows.append({"cluster": k, "param": aname,
                         "mean": _round4(a_mean[k, j]),
                         "r_hat": _round4(a_rhat[k, j]),
                         **_ess_cols(a_ess[k, j])})

    # ── pi (C,): mixture weights ───────────────────────────────────────────────
    # With C == 1, pi is a degenerate Dirichlet draw fixed at 1.0, so R-hat is
    # undefined and ArviZ may name the row pi[0] rather than pi.
    if C == 1:
        pi_mean = float(posterior["pi"].values.mean())
        rows.append({"cluster": 0, "param": "pi",
                     "mean": _round4(pi_mean),
                     "r_hat": np.nan,
                     "ess_bulk": np.nan,
                     "ess_per_chain": np.nan})
    else:
        pi_mean = posterior["pi"].mean(dim=("chain", "draw")).values
        pi_rhat = rhats["pi"].values
        pi_ess = ess["pi"].values
        for k in range(C):
            rows.append({"cluster": k, "param": "pi",
                         "mean": _round4(pi_mean[k]),
                         "r_hat": _round4(pi_rhat[k]),
                         **_ess_cols(pi_ess[k])})

    if estimate_ksi:
        # ── mu_ksi (scalar) ────────────────────────────────────────────────────
        mu_ksi_mean = posterior["mu_ksi"].mean(dim=("chain", "draw")).values
        mu_ksi_rhat = rhats["mu_ksi"].values
        mu_ksi_ess = ess["mu_ksi"].values
        rows.append({"cluster": None, "param": "mu_ksi",
                     "mean": _round4(mu_ksi_mean),
                     "r_hat": _round4(mu_ksi_rhat),
                     **_ess_cols(mu_ksi_ess)})

        # ── ksi (N subjects): report mean + worst-case, not all N rows ────────
        ksi_mean = posterior["ksi"].mean(dim=("chain", "draw")).values
        ksi_rhat = rhats["ksi"].values
        rows.append({"cluster": None, "param": "ksi  (mean R-hat, N subjects)",
                     "mean": _round4(np.nanmean(ksi_mean)),
                     "r_hat": _round4(np.nanmean(ksi_rhat)),
                     "ess_bulk": np.nan,
                     "ess_per_chain": np.nan})
        rows.append({"cluster": None, "param": "ksi  (worst R-hat)",
                     "mean": float("nan"),
                     "r_hat": _round4(np.nanmax(ksi_rhat)),
                     "ess_bulk": np.nan,
                     "ess_per_chain": np.nan})
    else:
        ksi_mean = posterior["ksi"].mean(dim=("chain", "draw")).values
        rows.append({"cluster": None, "param": "ksi  (fixed)",
                     "mean": _round4(np.nanmean(ksi_mean)),
                     "r_hat": np.nan,
                     "ess_bulk": np.nan,
                     "ess_per_chain": np.nan})

    return pd.DataFrame(rows)


def plot_posteriors(idata, output_dir="outputs/plots", method="prelec"):
    """Save ArviZ trace plots for theta, pi, optional mu_ksi, and a_weights."""
    os.makedirs(output_dir, exist_ok=True)

    trace_vars = ["theta", "a_weights"]
    if idata.posterior["pi"].shape[-1] > 1:
        trace_vars.insert(1, "pi")
    if "mu_ksi" in idata.posterior:
        trace_vars.insert(-1, "mu_ksi")

    for var in trace_vars:
        axes = az.plot_trace(idata, var_names=[var])
        fig  = axes.ravel()[0].get_figure()
        path = os.path.join(output_dir, f"trace_{var}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved → {path}")

    # Posterior histograms per cluster for assembled theta params
    theta = idata.posterior["theta"].values   # (chains, draws, C, 8)
    n_chains, n_draws, C, P = theta.shape
    theta_flat = theta.reshape(-1, C, P)
    names = PARAM_NAMES[method]

    fig, axes = plt.subplots(P, C, figsize=(4 * C, 2 * P), squeeze=False)
    for k in range(C):
        for j, name in enumerate(names):
            ax = axes[j][k]
            ax.hist(theta_flat[:, k, j], bins=50, density=True, alpha=0.7,
                    color=f"C{k}", edgecolor="none")
            ax.set_title(f"Cluster {k} — {name}", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    path = os.path.join(output_dir, "posterior_theta.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved → {path}")
