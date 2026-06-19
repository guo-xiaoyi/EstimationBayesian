"""
CPT mixture log-likelihood wrapped as a PyTensor blackbox op.

The model uses continuous, cluster-specific reference-point weights:

    theta_k = [discount params, alpha_plus, alpha_minus, lamb,
               gamma/palpha, a1, a2, a3, delta]
    a_k     = [a1, a2, a3, a4] on the 4-simplex, one vector per cluster

The likelihood marginalises over cluster assignments:

    log p(data | theta, pi, kappa)
      = sum_i log sum_k pi_k prod_l N(CE_obs_il; CE_th(theta_k), kappa_i * spread_l)

Call setup_likelihood() once after preprocessing before building the model.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import gammaln, logsumexp

_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "base_compiledir" not in _pytensor_flags:
    suffix = "base_compiledir=/tmp/pytensor"
    os.environ["PYTENSOR_FLAGS"] = f"{_pytensor_flags},{suffix}" if _pytensor_flags else suffix

import pytensor.tensor as pt
from pytensor.compile.ops import as_op

import functions as f
from GlobalSettings import (
    GlobalDiscounting,
    GlobalPrelecBounds,
    GlobalPriorKsiIGAlpha,
    GlobalPriorKsiIGBeta,
    GlobalQuasiHyperbolicBounds,
    GlobalRSQMode,
    GlobalTKBounds,
)

# Module-level state set by setup_likelihood().
_preproc = None
_lotteries = None
_N = None
_C = None
_method = None
_si_arr = None
_ckey_groups = None
_conditional_lottery_cache = None
_BAD_LOGP = -1e300

PARAM_NAMES = {
    "prelec": [
        "r", "alpha_plus", "alpha_minus", "lamb", "palpha",
        "a1", "a2", "a3", "delta",
    ],
    "tk": [
        "r", "alpha_plus", "alpha_minus", "lamb", "gamma",
        "a1", "a2", "a3", "delta",
    ],
}

RP_COMPONENT_NAMES = ["SQ", "PA", "LE", "FE"]


def normalise_discounting(discounting=None):
    """Return the canonical discounting key used for parameter layouts."""
    raw = GlobalDiscounting if discounting is None else discounting
    key = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "exp": "exponential",
        "hyp": "hyperbolic",
        "quasihyperbolic": "quasi_hyperbolic",
        "qh": "quasi_hyperbolic",
        "beta_delta": "quasi_hyperbolic",
    }
    return aliases.get(key, key)


def _normalise_method(method):
    key = str(method).strip().lower()
    if key not in {"tk", "prelec"}:
        raise ValueError(f"Unknown method: {method!r}")
    return key


def get_discounting_param_names(discounting=None):
    discounting = normalise_discounting(discounting)
    if discounting in {"exponential", "hyperbolic"}:
        return ["r"]
    if discounting == "quasi_hyperbolic":
        return ["beta_qh", "delta_qh"]
    raise ValueError(f"Unknown discounting method: {discounting!r}")


def get_free_param_names(method, discounting=None):
    """Names for theta_rest, excluding simplex reference-point weights."""
    method = _normalise_method(method)
    probability_param = "palpha" if method == "prelec" else "gamma"
    return (
        get_discounting_param_names(discounting)
        + ["alpha_plus", "alpha_minus", "lamb", probability_param, "delta"]
    )


def get_pre_rp_param_count(method, discounting=None):
    """Number of theta_rest columns that appear before a1/a2/a3 in theta."""
    _normalise_method(method)
    return len(get_discounting_param_names(discounting)) + 4


def get_param_names(method, discounting=None):
    """Names for sampled theta columns, excluding the implicit a4."""
    names = get_free_param_names(method, discounting)
    split = get_pre_rp_param_count(method, discounting)
    return names[:split] + ["a1", "a2", "a3"] + names[split:]


def get_full_param_names(method, discounting=None):
    """Names after inserting Prelec's fixed probability-weight beta=1."""
    names = get_param_names(method, discounting)
    if _normalise_method(method) == "prelec":
        insert_at = len(get_discounting_param_names(discounting)) + 3
        names = names[:insert_at] + ["beta"] + names[insert_at:]
    return names


def get_free_bounds(method, discounting=None):
    """
    Bounds for non-simplex structural params:
    [discount params, alpha_plus, alpha_minus, lamb, gamma/palpha, delta].
    Reference-point weights are handled by a Dirichlet prior over a_weights.
    """
    method = _normalise_method(method)
    discounting = normalise_discounting(discounting)
    if discounting in {"exponential", "hyperbolic"}:
        discount_bounds = None
    elif discounting == "quasi_hyperbolic":
        discount_bounds = list(GlobalQuasiHyperbolicBounds)
    else:
        raise ValueError(f"Unknown discounting method: {discounting!r}")

    if method == "prelec":
        b = GlobalPrelecBounds
        base = [b[1], b[2], b[3], b[5], b[9]]
        return ([b[0]] if discount_bounds is None else discount_bounds) + base

    b = GlobalTKBounds
    base = [b[1], b[2], b[3], b[4], b[8]]
    return ([b[0]] if discount_bounds is None else discount_bounds) + base


def _free_to_full(params_free, method, discounting=None):
    """Insert fixed probability-weight beta=1 for Prelec; TK passes through."""
    method = _normalise_method(method)
    if method != "prelec":
        return params_free

    free_names = get_param_names(method, discounting)
    full_names = get_full_param_names(method, discounting)
    values = dict(zip(free_names, params_free))
    return np.array(
        [1.0 if name == "beta" else values[name] for name in full_names],
        dtype=float,
    )


def setup_likelihood(preproc, lotteries, subjects, method, C, K=None):
    """Initialise module-level state. Must be called before building the model."""
    del K  # Kept for backwards-compatible callers; continuous RP weights use 4 components.

    global _preproc, _lotteries, _N, _C, _method, _si_arr, _ckey_groups
    global _conditional_lottery_cache

    _preproc = preproc
    _lotteries = lotteries
    _N = len(subjects)
    _C = C
    _method = method
    _si_arr = preproc[8].astype(int)
    _conditional_lottery_cache = {}

    y_proc = preproc[0]
    _ckey_groups = {
        key: idx.values
        for key, idx in y_proc.groupby("_ckey", sort=False).groups.items()
    }


def _conditional_lottery_for_key(lid, t_val, z1, z2, lotteries):
    """
    Return lottery `lid` conditioned on the realised history at session t.

    The payoff streams are kept on the original total-payoff scale so the model
    can compute a total CE and subtract Z_t afterwards. Branch probabilities are
    renormalised within the realised history.
    """
    key = (lid, int(t_val), int(z1), int(z2))
    cached = _conditional_lottery_cache.get(key)
    if cached is not None:
        return cached

    base = lotteries[lid]
    t_int = int(t_val)
    if t_int <= 0:
        _conditional_lottery_cache[key] = base
        return base

    matched = []
    for path in base["outcomes"].values():
        prob = float(next(iter(path.keys())))
        stream = list(next(iter(path.values())))

        if t_int == 1:
            keep = len(stream) >= 2 and stream[1] == int(z1)
        elif t_int == 2:
            keep = len(stream) >= 3 and stream[1] == int(z1) and stream[2] == int(z2)
        else:
            keep = False

        if keep:
            matched.append((prob, stream))

    total_prob = sum(prob for prob, _ in matched)
    if not matched or total_prob <= 0:
        _conditional_lottery_cache[key] = base
        return base

    outcomes = {
        i: {prob / total_prob: stream}
        for i, (prob, stream) in enumerate(matched)
    }
    conditioned = {
        "name": base.get("name", lid),
        "spread": base["spread"],
        "outcomes": outcomes,
    }
    _conditional_lottery_cache[key] = conditioned
    return conditioned


def _compute_ce_th(params, method, EL_arr, EL_fe_arr, EL_c1_arr,
                   t_arr, Z1_arr, Z2_arr, Zt_arr, lotteries, y):
    """
    Compute theoretical CEs for all observations under one cluster's parameters.

    Parameter names are generated from the CPT method and GlobalDiscounting.
    Prelec's probability-weight beta is fixed to 1 and inserted by _free_to_full.
    """
    method = _normalise_method(method)
    names = get_full_param_names(method)
    values = dict(zip(names, params[:len(names)]))

    r = values.get("r", 0.0)
    beta_qh = values.get("beta_qh", 1.0)
    delta_qh = values.get("delta_qh")
    alpha_plus = values["alpha_plus"]
    alpha_minus = values["alpha_minus"]
    lamb = values["lamb"]
    gamma = values.get("gamma", 0.61)
    beta = values.get("beta", 1.0)
    palpha = values.get("palpha", 1.0)
    a1 = values["a1"]
    a2 = values["a2"]
    a3 = values["a3"]
    delta = values["delta"]

    a4 = max(0.0, 1.0 - a1 - a2 - a3)

    n = len(y)
    m1 = t_arr == 1
    m2 = t_arr == 2

    ra = np.zeros(n)
    ra[m1] = Z1_arr[m1] / (delta + 1.0)
    ra[m2] = (delta * Z1_arr[m2] + Z2_arr[m2]) / (delta**2 + delta + 1.0)

    rle = np.zeros(n)
    rle[m1] = EL_arr[m1]
    rle[m2] = (delta * EL_arr[m2] + EL_c1_arr[m2]) / (delta + 1.0)

    rsq = Zt_arr if GlobalRSQMode == "zt" else np.zeros(n)
    r_l = a1 * rsq + a2 * ra + a3 * rle + a4 * EL_fe_arr

    ce_th_base = np.empty(n)
    for key, idx in _ckey_groups.items():
        lid, t_val, z1, z2 = key
        rl = float(r_l[idx[0]])
        conditioned_lottery = _conditional_lottery_for_key(lid, t_val, z1, z2, lotteries)
        ev = f.evaluation(
            r=r,
            R=rl,
            alpha_plus=alpha_plus,
            alpha_minus=alpha_minus,
            lamb=lamb,
            gamma=gamma,
            lotteries={lid: conditioned_lottery},
            method=method,
            beta=beta,
            palpha=palpha,
            discounting=GlobalDiscounting,
            beta_qh=beta_qh,
            delta_qh=delta_qh,
        )
        ce_th_base[idx] = f.u_inv(
            ev[lid]["V"],
            rl,
            lamb=lamb,
            alpha_plus=alpha_plus,
            alpha_minus=alpha_minus,
        )

    return ce_th_base - Zt_arr


def _preprocess_rows(y, lotteries, subj_index):
    """
    Filter and annotate observations; convert to numpy arrays for the hot path.

    Returns:
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
    y_proc["_t"] = s2.astype(int) + 2 * s3.astype(int)
    y_proc["_Z1"] = np.where(s2 | s3, _parse_col(y_proc["realized_period1_label"]), 0)
    y_proc["_Z2"] = np.where(s3, _parse_col(y_proc["realized_period2_label"]), 0)
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

    EL_arr = y_proc["_EL"].values.astype(float)
    EL_fe_arr = y_proc["_EL_fe"].values.astype(float)
    EL_c1_arr = y_proc["_EL_c1"].values.astype(float)
    t_arr = y_proc["_t"].values.astype(float)
    Z1_arr = y_proc["_Z1"].values.astype(float)
    Z2_arr = y_proc["_Z2"].values.astype(float)
    Zt_arr = y_proc["_Zt"].values.astype(float)
    si_arr = y_proc["_si"].values
    spread_arr = y_proc["spread"].values.astype(float)
    obs_arr = y_proc["ce_observed"].values.astype(float)

    return (y_proc, EL_arr, EL_fe_arr, EL_c1_arr,
            t_arr, Z1_arr, Z2_arr, Zt_arr, si_arr, spread_arr, obs_arr)


def _compute_ll_numpy(theta, pi, ksi):
    """
    Pure NumPy CPT mixture log-likelihood.

    theta : (C, 9) cluster params
            [r, alpha_plus, alpha_minus, lamb, gamma/palpha, a1, a2, a3, delta]
    pi    : (C,) cluster mixing weights
    ksi   : (N,) subject-level noise multipliers
    """
    (y_proc, EL_arr, EL_fe_arr, EL_c1_arr,
     t_arr, Z1_arr, Z2_arr, Zt_arr,
     si_arr, spread_arr, obs_arr) = _preproc

    si_arr = si_arr.astype(int)
    sig_arr = ksi[si_arr] * spread_arr

    log_l = np.zeros((_N, _C))
    for k in range(_C):
        params_full = _free_to_full(theta[k], _method)
        ce_th = _compute_ce_th(
            params_full, _method,
            EL_arr, EL_fe_arr, EL_c1_arr,
            t_arr, Z1_arr, Z2_arr, Zt_arr,
            _lotteries, y_proc,
        )
        if not np.all(np.isfinite(ce_th)):
            log_l[:, k] = _BAD_LOGP
            continue
        log_pdf = norm.logpdf(obs_arr, loc=ce_th, scale=sig_arr)
        log_pdf = np.nan_to_num(log_pdf, nan=_BAD_LOGP, neginf=_BAD_LOGP, posinf=_BAD_LOGP)
        np.add.at(log_l[:, k], si_arr, log_pdf)

    log_pi = np.log(np.clip(pi, 1e-300, None))
    log_terms = log_l + log_pi[np.newaxis, :]
    return np.array(logsumexp(log_terms, axis=1).sum(), dtype=np.float64)


def compute_marginal_ksi_subject_loglik(
    theta,
    method,
    preproc,
    lotteries,
    alpha=GlobalPriorKsiIGAlpha,
    beta=GlobalPriorKsiIGBeta,
    n_subjects=None,
):
    """
    Subject-by-cluster log likelihood after integrating out subject noise.

    The integrated nuisance parameter is v_i = ksi_i^2 with
    v_i ~ InverseGamma(alpha, beta). Conditional observation noise is
    Normal(CE_th, v_i * spread_l^2).
    """
    (y_proc, EL_arr, EL_fe_arr, EL_c1_arr,
     t_arr, Z1_arr, Z2_arr, Zt_arr,
     si_arr, spread_arr, obs_arr) = preproc

    si_arr = si_arr.astype(int)
    if n_subjects is None:
        n_subjects = int(si_arr.max()) + 1

    spread_arr = spread_arr.astype(float)
    if np.any(spread_arr <= 0):
        raise ValueError("Marginalized ksi likelihood requires positive spreads.")

    n_obs = np.bincount(si_arr, minlength=n_subjects).astype(float)
    log_spread_sum = np.bincount(
        si_arr,
        weights=np.log(spread_arr),
        minlength=n_subjects,
    )
    posterior_shape = alpha + 0.5 * n_obs
    base = (
        -log_spread_sum
        - 0.5 * n_obs * np.log(2.0 * np.pi)
        + alpha * np.log(beta)
        - gammaln(alpha)
        + gammaln(posterior_shape)
    )

    C = theta.shape[0]
    log_l = np.zeros((n_subjects, C))
    for k in range(C):
        params_full = _free_to_full(theta[k], method)
        ce_th = _compute_ce_th(
            params_full, method,
            EL_arr, EL_fe_arr, EL_c1_arr,
            t_arr, Z1_arr, Z2_arr, Zt_arr,
            lotteries, y_proc,
        )
        if not np.all(np.isfinite(ce_th)):
            log_l[:, k] = _BAD_LOGP
            continue
        scaled_resid_sq = ((obs_arr - ce_th) / spread_arr) ** 2
        q = np.bincount(si_arr, weights=scaled_resid_sq, minlength=n_subjects)
        log_l[:, k] = base - posterior_shape * np.log(beta + 0.5 * q)
        log_l[:, k] = np.nan_to_num(
            log_l[:, k],
            nan=_BAD_LOGP,
            neginf=_BAD_LOGP,
            posinf=_BAD_LOGP,
        )

    return log_l


def _compute_ll_numpy_marginal_ksi(theta, pi):
    """Pure NumPy CPT mixture log-likelihood with ksi analytically marginalized."""
    log_l = compute_marginal_ksi_subject_loglik(
        theta,
        _method,
        _preproc,
        _lotteries,
        alpha=GlobalPriorKsiIGAlpha,
        beta=GlobalPriorKsiIGBeta,
        n_subjects=_N,
    )
    log_pi = np.log(np.clip(pi, 1e-300, None))
    log_terms = log_l + log_pi[np.newaxis, :]
    return np.array(logsumexp(log_terms, axis=1).sum(), dtype=np.float64)


@as_op(
    itypes=[pt.dmatrix, pt.dvector, pt.dvector],
    otypes=[pt.dscalar],
)
def cpt_loglik_op(theta, pi, ksi):
    """PyTensor op wrapping _compute_ll_numpy. No gradient is provided."""
    return _compute_ll_numpy(theta, pi, ksi)


@as_op(
    itypes=[pt.dmatrix, pt.dvector],
    otypes=[pt.dscalar],
)
def cpt_loglik_marginal_ksi_op(theta, pi):
    """PyTensor op wrapping the ksi-marginalized likelihood."""
    return _compute_ll_numpy_marginal_ksi(theta, pi)
