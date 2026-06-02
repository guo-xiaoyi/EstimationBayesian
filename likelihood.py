"""
CPT mixture log-likelihood wrapped as a PyTensor blackbox op so it can be
used inside a PyMC model with gradient-free samplers such as SMC.

The likelihood marginalises over cluster assignments:

    log p(data | θ, π, κ) = Σ_i log [ Σ_k π_k · ∏_l N(CE_obs; CE_th(θ_k), κ_i·spread_l) ]

theta  : (C, P) cluster parameters — [r, alpha, lamb, gamma/palpha, a1, a2, a3, delta]
         a1,a2,a3 are the reference-point weights directly (no stick-breaking).
pi     : (C,)  mixing weights
ksi    : (N,)  individual noise multipliers

Call setup_likelihood() once after preprocessing before building the model.

Core CPT functions (_compute_ce_th, _preprocess_rows) are local copies from
Estimation/Mixture.py, adapted to accept a1,a2,a3 directly without stick-breaking.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

import functions as f
from GlobalSettings import GlobalPrelecBounds, GlobalTKBounds, GlobalRSQMode

# ── Module-level state set by setup_likelihood() ──────────────────────────────
_preproc      = None   # 11-tuple from _preprocess_rows
_lotteries    = None
_N            = None   # number of subjects
_C            = None   # number of clusters
_method       = None   # "prelec" or "tk"
_si_arr       = None   # subject index per row (pre-extracted for speed)
_ckey_groups  = None   # precomputed {ckey: np.array(row_indices)} — never changes

# Parameter names for the assembled theta (C, 8):
# positions 4,5,6 are now a1,a2,a3 directly — no stick-breaking.
PARAM_NAMES = {
    "prelec": ["r", "alpha", "lamb", "palpha", "a1", "a2", "a3", "delta"],
    "tk":     ["r", "alpha", "lamb", "gamma",  "a1", "a2", "a3", "delta"],
}


def get_free_bounds(method):
    """
    Bounds for the non-simplex structural parameters only:
      TK:     [r, alpha, lamb, gamma, delta]   — 5 bounds
      Prelec: [r, alpha, lamb, palpha, delta]  — 5 bounds
    The reference-point weights (a1,a2,a3,a4) are handled by pm.Dirichlet.
    """
    if method == "prelec":
        b = GlobalPrelecBounds
        # order in GlobalPrelecBounds: r(0) alpha(1) lamb(2) beta(3) palpha(4) s1(5) s2(6) s3(7) delta(8)
        return [b[0], b[1], b[2], b[4], b[8]]
    else:
        b = GlobalTKBounds
        # order in GlobalTKBounds: r(0) alpha(1) lamb(2) gamma(3) s1(4) s2(5) s3(6) delta(7)
        return [b[0], b[1], b[2], b[3], b[7]]


def _free_to_full(params_free, method):
    """Insert fixed beta=1 at position 3 for Prelec; TK passes through."""
    if method == "prelec":
        return np.concatenate([params_free[:3], [1.0], params_free[3:]])
    return params_free


def setup_likelihood(preproc, lotteries, subjects, method, C):
    """Initialise module-level state.  Must be called before building the model."""
    global _preproc, _lotteries, _N, _C, _method, _si_arr, _ckey_groups
    _preproc   = preproc
    _lotteries = lotteries
    _N         = len(subjects)
    _C         = C
    _method    = method
    _si_arr    = preproc[8].astype(int)   # si_arr is index 8 in the tuple
    # Precompute groupby structure once — y_proc never changes after preprocessing.
    # Avoids re-creating a pandas GroupBy object on every likelihood evaluation.
    _ckey_groups = {
        key: idx.values
        for key, idx in preproc[0].groupby("_ckey", sort=False).groups.items()
    }


# ── CPT core (adapted from Estimation/Mixture.py) ────────────────────────────

def _compute_ce_th(params, method, EL_arr, EL_fe_arr, EL_c1_arr,
                   t_arr, Z1_arr, Z2_arr, Zt_arr, lotteries, y):
    """
    Compute theoretical CEs for all rows under one cluster's params.

    params is in full space (beta inserted for Prelec by _free_to_full):
      TK:     [r, alpha, lamb, gamma, a1, a2, a3, delta]
      Prelec: [r, alpha, lamb, beta, palpha, a1, a2, a3, delta]

    a1, a2, a3 are the reference-point weights directly — no stick-breaking.
    a4 = 1 - a1 - a2 - a3 (forward-looking residual).
    """
    if method == "tk":
        r, alpha, lamb, gamma, a1, a2, a3, delta = params[:8]
        beta, palpha = 1, 1
    elif method == "prelec":
        r, alpha, lamb, beta, palpha, a1, a2, a3, delta = params[:9]
        gamma = 0.61
    else:
        raise ValueError(f"Unknown method: {method!r}")

    a4 = max(0.0, 1.0 - a1 - a2 - a3)

    RA  = np.zeros(len(y))
    m1  = t_arr == 1
    m2  = t_arr == 2
    RA[m1] = Z1_arr[m1] / (delta + 1.0)
    RA[m2] = (delta * Z1_arr[m2] + Z2_arr[m2]) / (delta**2 + delta + 1.0)

    RLE = np.zeros(len(y))
    RLE[m1] = EL_arr[m1]
    RLE[m2] = (delta * EL_arr[m2] + EL_c1_arr[m2]) / (delta + 1.0)

    RSQ = Zt_arr if GlobalRSQMode == "zt" else np.zeros(len(y))
    R_l = a1 * RSQ + a2 * RA + a3 * RLE + a4 * EL_fe_arr

    ce_th_base = np.empty(len(y))
    for key, idx in _ckey_groups.items():
        lid = key[0]
        rl  = float(R_l[idx[0]])
        ev  = f.evaluation(r=r, R=rl, alpha=alpha, lamb=lamb,
                           gamma=gamma, lotteries={lid: lotteries[lid]},
                           method=method, beta=beta, palpha=palpha)
        ce_th_base[idx] = f.u_inv(ev[lid]["V"], rl, alpha, lamb)

    return ce_th_base - Zt_arr


def _preprocess_rows(y, lotteries, subj_index):
    """
    Filter and annotate observations; convert to numpy arrays for the hot path.

    Returns 11-tuple:
        (y_proc, EL_arr, EL_fe_arr, EL_c1_arr,
         t_arr, Z1_arr, Z2_arr, Zt_arr, si_arr, spread_arr, obs_arr)
    """
    def _parse_col(col):
        return pd.to_numeric(
            col.str.replace("£", "", regex=False), errors="coerce"
        ).fillna(0).astype(int)

    y_proc = (y[y["lottery_id"].isin(lotteries.keys())]
              .dropna(subset=["ce_observed"])
              .copy()
              .reset_index(drop=True))
    y_proc["spread"] = y_proc["lottery_id"].map(
        {lid: lotteries[lid]["spread"] for lid in lotteries}
    )
    y_proc["_EL"] = y_proc["lottery_id"].map(
        {lid: f.expected_payoff(lotteries[lid]["outcomes"]) for lid in lotteries}
    )
    s2 = y_proc["round_number"] == 17
    s3 = y_proc["round_number"] == 18
    y_proc["_t"]  = s2.astype(int) + 2 * s3.astype(int)
    y_proc["_Z1"] = np.where(s2 | s3, _parse_col(y_proc["realized_period1_label"]), 0)
    y_proc["_Z2"] = np.where(s3,      _parse_col(y_proc["realized_period2_label"]), 0)
    y_proc["_Zt"] = y_proc["_Z1"] + y_proc["_Z2"]
    y_proc["_si"] = y_proc["participant_label"].map(subj_index)
    y_proc["_ckey"] = list(zip(
        y_proc["lottery_id"], y_proc["_t"], y_proc["_Z1"], y_proc["_Z2"]
    ))

    el_fe_cache = {}
    el_c1_cache = {}
    for key in y_proc["_ckey"].unique():
        lid, t_val, z1, z2 = key
        outcomes = lotteries[lid]["outcomes"]
        el_fe_cache[key] = f.conditional_el(outcomes, int(z1), int(z2), int(t_val))
        if int(t_val) == 2:
            c1_key = (lid, z1)
            if c1_key not in el_c1_cache:
                el_c1_cache[c1_key] = f.conditional_el(outcomes, int(z1), 0, 1)

    y_proc["_EL_fe"] = y_proc["_ckey"].map(el_fe_cache)
    y_proc["_EL_c1"] = 0.0
    mask_t2 = y_proc["_t"] == 2
    y_proc.loc[mask_t2, "_EL_c1"] = (
        y_proc.loc[mask_t2]
        .apply(lambda row: el_c1_cache.get((row["lottery_id"], row["_Z1"]), 0.0), axis=1)
        .values
    )

    EL_arr     = y_proc["_EL"].values.astype(float)
    EL_fe_arr  = y_proc["_EL_fe"].values.astype(float)
    EL_c1_arr  = y_proc["_EL_c1"].values.astype(float)
    t_arr      = y_proc["_t"].values.astype(float)
    Z1_arr     = y_proc["_Z1"].values.astype(float)
    Z2_arr     = y_proc["_Z2"].values.astype(float)
    Zt_arr     = y_proc["_Zt"].values.astype(float)
    si_arr     = y_proc["_si"].values
    spread_arr = y_proc["spread"].values.astype(float)
    obs_arr    = y_proc["ce_observed"].values.astype(float)

    return (y_proc, EL_arr, EL_fe_arr, EL_c1_arr,
            t_arr, Z1_arr, Z2_arr, Zt_arr, si_arr, spread_arr, obs_arr)


# ── Likelihood computation ─────────────────────────────────────────────────────

def _compute_ll_numpy(theta, pi, ksi):
    """
    Pure-numpy CPT mixture log-likelihood.

    theta : (C, 8) float64 — [r, alpha, lamb, gamma/palpha, a1, a2, a3, delta]
    pi    : (C,)   float64
    ksi   : (N,)   float64
    Returns scalar float64.
    """
    (y_proc, EL_arr, EL_fe_arr, EL_c1_arr,
     t_arr, Z1_arr, Z2_arr, Zt_arr,
     si_arr, spread_arr, obs_arr) = _preproc

    ksi_row = ksi[_si_arr]
    sig_arr = ksi_row * spread_arr

    log_L = np.zeros((_N, _C))
    for k in range(_C):
        params_full = _free_to_full(theta[k], _method)
        ce_th = _compute_ce_th(
            params_full, _method,
            EL_arr, EL_fe_arr, EL_c1_arr,
            t_arr, Z1_arr, Z2_arr, Zt_arr,
            _lotteries, y_proc,
        )
        log_pdf = norm.logpdf(obs_arr, loc=ce_th, scale=sig_arr)
        np.add.at(log_L[:, k], _si_arr, log_pdf)

    log_pi    = np.log(np.clip(pi, 1e-300, None))
    log_terms = log_L + log_pi[np.newaxis, :]
    return np.array(logsumexp(log_terms, axis=1).sum(), dtype=np.float64)


@as_op(
    itypes=[pt.dmatrix, pt.dvector, pt.dvector],
    otypes=[pt.dscalar],
)
def cpt_loglik_op(theta, pi, ksi):
    """PyTensor op wrapping _compute_ll_numpy.  No gradient is provided."""
    return _compute_ll_numpy(theta, pi, ksi)
