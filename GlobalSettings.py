"""
Settings for the EstimationBayesian project.

Fully self-contained — no dependency on the Estimation project directory.
All shared code (lotteries, functions, data) is maintained as local copies.
"""

import os

import numpy as np


def _env_str(name, default):
    return os.environ.get(name, default)


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_optional_int(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _env_float_pair(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain two comma-separated floats.")
    lower, upper = float(parts[0]), float(parts[1])
    if lower > upper:
        raise ValueError(f"{name} lower bound must be <= upper bound.")
    return (lower, upper)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}

# ── Lottery sets ───────────────────────────────────────────────────────────────
import lotteries as _lot
_HIGH_STAKE = {'lottery_2', 'lottery_7', 'cali_mix', 'cali_mix_pres', 'lottery_8'}
_NO_CALI    = {'cali_mix', 'cali_mix_pres', 'lottery_8'}
GlobalLotteryFull      = _lot.lotteries_v2
GlobalLotteryLowStake  = {k: v for k, v in _lot.lotteries_v2.items() if k not in _HIGH_STAKE}
GlobalLotteryNoCaliMix = {k: v for k, v in _lot.lotteries_v2.items() if k not in _NO_CALI}
GlobalLotterySets = {
    "full": GlobalLotteryFull,
    "all": GlobalLotteryFull,
    "low_stake": GlobalLotteryLowStake,
    "low-stake": GlobalLotteryLowStake,
    "no_cali_mix": GlobalLotteryNoCaliMix,
    "no-cali-mix": GlobalLotteryNoCaliMix,
}


def _select_lottery_set(name, ids_csv=None):
    if name not in GlobalLotterySets:
        choices = ", ".join(sorted(GlobalLotterySets))
        raise ValueError(f"Unknown lottery set {name!r}. Choose one of: {choices}.")

    selected = dict(GlobalLotterySets[name])
    if ids_csv not in (None, ""):
        ids = [item.strip() for item in ids_csv.split(",") if item.strip()]
        missing = [item for item in ids if item not in GlobalLotteryFull]
        if missing:
            raise ValueError(f"Unknown lottery ids in EB_GLOBAL_LOTTERY_IDS: {missing}")
        selected = {key: GlobalLotteryFull[key] for key in ids}
    return selected


GlobalLotterySetName = _env_str("EB_GLOBAL_LOTTERY_SET", "full")
GlobalLotteryIdsCSV = _env_str("EB_GLOBAL_LOTTERY_IDS", "")
GlobalLottery = _select_lottery_set(GlobalLotterySetName, GlobalLotteryIdsCSV)

# ── Parameter bounds ───────────────────────────────────────────────────────────
# Full CPT params per cluster:
#   TK     : [r, alpha_plus, alpha_minus, lamb, gamma, a1, a2, a3, delta]
#   Prelec : [r, alpha_plus, alpha_minus, lamb, beta, palpha, a1, a2, a3, delta]
#   Quasi-hyperbolic discounting replaces r with [beta_qh, delta_qh],
#   where beta_qh is present bias and delta_qh is the long-run discount factor.
# alpha_plus and alpha_minus are CARA absolute risk-aversion coefficients.
# Non-simplex structural params are sampled hierarchically; reference-point
# weights are continuous cluster-specific a_weights on the 4-simplex.
_GlobalDeltaBoundsOverride = _env_float_pair("EB_GLOBAL_DELTA_BOUNDS", None)
GlobalTKDeltaBounds = _env_float_pair(
    "EB_GLOBAL_TK_DELTA_BOUNDS",
    _GlobalDeltaBoundsOverride if _GlobalDeltaBoundsOverride is not None else (1, 1),
)
GlobalPrelecDeltaBounds = _env_float_pair(
    "EB_GLOBAL_PRELEC_DELTA_BOUNDS",
    _GlobalDeltaBoundsOverride if _GlobalDeltaBoundsOverride is not None else (0.0, 1),
)
GlobalTKBounds = [
    (1e-6, 0.01), (1e-5, 0.01), (1e-5, 0.01), (0.5, 5.0), (0.2, 1.0),
    (0, 1), (0.0, 1), (0.0, 1), GlobalTKDeltaBounds,
]
GlobalPrelecBounds = [
    (1e-6, 0.1), (1e-5, 0.01), (1e-5, 0.01), (0.5, 3.0), (1.0, 1.0), (0.4, 1),
    (0, 1), (0.0, 1), (0.0, 1), GlobalPrelecDeltaBounds,
]
GlobalQuasiHyperbolicBounds = [
    (0.05, 1.0),  # beta_qh: present-bias factor
    (0.50, 1.0),  # delta_qh: long-run per-period discount factor
]
GlobalKsiBounds = (1e-4, 1.5)

# ── Model settings ─────────────────────────────────────────────────────────────
GlobalMethod   = _env_str("EB_GLOBAL_METHOD", "tk")
GlobalDiscounting = _env_str("EB_GLOBAL_DISCOUNTING", "hyperbolic")
GlobalCluster  = _env_int("EB_GLOBAL_CLUSTER", 2)      # number of latent preference clusters (C)

# Reference point components (Baillon, Bleichrodt & Spinu, 2015)
# Continuous cluster-specific weights a_weights[k] live on these 4 components:
# SQ = status quo, PA = partial adaptation, LE = lagged expectation, FE = forward expectation.
GlobalK        = 4      # kept for compatibility; continuous RP weights require 4 components

GlobalTol      = 1e-4
GlobalInterMax = 1000
GlobalData     = _env_str("EB_GLOBAL_DATA", "/home/ubuntu/Estimation/gamedata.csv")
GlobalRSQMode  = "zero"
GlobalFixTau   = False

GlobalNSeeds     = 1000
GlobalMasterSeed = 42
GlobalSeedsSet   = np.random.default_rng(GlobalMasterSeed).integers(0, 10000, size=GlobalNSeeds).tolist()

# ── SMC settings ───────────────────────────────────────────────────────────────
GlobalDraws    = _env_int("EB_GLOBAL_DRAWS", 1000)
GlobalTune     = 5000   # unused by SMC; kept for legacy imports
GlobalChains   = _env_int("EB_GLOBAL_CHAINS", 5)
GlobalSMCCores = _env_optional_int("EB_GLOBAL_SMC_CORES", None)
GlobalSeed     = _env_int("EB_GLOBAL_SEED", 42)
GlobalProgressBar = _env_bool("EB_GLOBAL_PROGRESSBAR", True)

# ── Prior hyperparameters ──────────────────────────────────────────────────────
GlobalPriorDirichlet = _env_float("EB_GLOBAL_PRIOR_DIRICHLET", 1.0)   # concentration for π (cluster mixing weights)
GlobalPriorDirichletRP = _env_float("EB_GLOBAL_PRIOR_DIRICHLET_RP", 1.0) # concentration for cluster-specific a_weights
GlobalKsiMode        = _env_str("EB_GLOBAL_KSI_MODE", "marginalized")  # one of: "estimated", "fixed", "marginalized"
GlobalPriorKsiSigma  = _env_float("EB_GLOBAL_PRIOR_KSI_SIGMA", 0.5)
GlobalFixedKsi       = _env_float("EB_GLOBAL_FIXED_KSI", 0.5)
GlobalPriorKsiIGAlpha = _env_float("EB_GLOBAL_PRIOR_KSI_IG_ALPHA", 3.0)
GlobalPriorKsiIGBeta  = _env_float(
    "EB_GLOBAL_PRIOR_KSI_IG_BETA",
    (GlobalPriorKsiIGAlpha - 1.0) * GlobalFixedKsi**2,
)
GlobalEstimateKsi    = GlobalKsiMode == "estimated"  # legacy compatibility

# Normal-Inverse-Gamma hyperprior on log-scale structural parameters.
# theta_rest follows a truncated LogNormal on the admissible bounds.
GlobalPriorIG_Alpha = _env_float("EB_GLOBAL_PRIOR_IG_ALPHA", 3.0)
GlobalPriorIG_Beta  = _env_float("EB_GLOBAL_PRIOR_IG_BETA", 2.0)

# Benchmark log-scale neutral values for
# [r, alpha_plus, alpha_minus, lamb, gamma/palpha, delta].
GlobalPriorCARAEta0 = _env_float("EB_GLOBAL_PRIOR_CARA_ETA0", 1e-3)
if GlobalPriorCARAEta0 <= 0:
    raise ValueError("EB_GLOBAL_PRIOR_CARA_ETA0 must be positive.")
GlobalPriorCARAMu0 = float(np.log(GlobalPriorCARAEta0))
GlobalPriorMu0 = [None, GlobalPriorCARAMu0, GlobalPriorCARAMu0, 0.0, 0.0, 0.0]
GlobalPriorMu0ByName = {
    "r": None,
    "beta_qh": 0.0,
    "delta_qh": 0.0,
    "alpha_plus": GlobalPriorCARAMu0,
    "alpha_minus": GlobalPriorCARAMu0,
    "lamb": 0.0,
    "gamma": 0.0,
    "palpha": 0.0,
    "delta": 0.0,
}

# ── Output paths ───────────────────────────────────────────────────────────────
GlobalOutputDir  = _env_str("EB_GLOBAL_OUTPUT_DIR", "outputs")
GlobalTracePath  = _env_str(
    "EB_GLOBAL_TRACE_PATH",
    os.path.join(GlobalOutputDir, "trace.nc"),
)
GlobalSummaryCSV = _env_str(
    "EB_GLOBAL_SUMMARY_CSV",
    os.path.join(GlobalOutputDir, "summary.csv"),
)
GlobalPlotsDir   = _env_str(
    "EB_GLOBAL_PLOTS_DIR",
    os.path.join(GlobalOutputDir, "plots"),
)
