"""
Settings for the EstimationBayesian project.

Fully self-contained — no dependency on the Estimation project directory.
All shared code (lotteries, functions, data) is maintained as local copies.
"""

import numpy as np

# ── Lottery sets ───────────────────────────────────────────────────────────────
import lotteries as _lot
_HIGH_STAKE = {'lottery_2', 'lottery_7', 'cali_mix', 'cali_mix_pres', 'lottery_8'}
_NO_CALI    = {'cali_mix', 'cali_mix_pres', 'lottery_8'}
GlobalLotteryLowStake  = {k: v for k, v in _lot.lotteries_v2.items() if k not in _HIGH_STAKE}
GlobalLotteryNoCaliMix = {k: v for k, v in _lot.lotteries_v2.items() if k not in _NO_CALI}
GlobalLottery = _lot.lotteries_v2

# ── Parameter bounds (identical to Estimation/GlobalSettings.py) ───────────────
# TK:     [r, alpha, lamb, gamma, s1, s2, s3, delta]
GlobalTKBounds = [
    (1e-6, 0.01), (0.5, 1.5), (0.99, 3.0), (0.2, 1.0),
    (0, 1), (0.0, 1), (0.0, 1), (0.2, 1),
]
# Prelec: [r, alpha, lamb, beta, palpha, s1, s2, s3, delta]  (beta fixed at 1)
GlobalPrelecBounds = [
    (1e-6, 0.1), (0.5, 1.5), (0.99, 3.5), (1.0, 1.0), (0.4, 1),
    (0, 1), (0.0, 1), (0.0, 1), (0.0, 1),
]
GlobalKsiBounds = (1e-4, 1.5)

# ── Variables Mixture.py / data.py / EM imports at load time ──────────────────
GlobalMethod   = "tk"
GlobalCluster  = 2
GlobalTol      = 1e-4
GlobalInterMax = 1000
GlobalData     = "/home/ubuntu/Estimation/gamedata.csv"
GlobalRSQMode  = "zero"
GlobalFixTau   = False

GlobalNSeeds     = 1000
GlobalMasterSeed = 42
GlobalSeedsSet   = np.random.default_rng(GlobalMasterSeed).integers(0, 10000, size=GlobalNSeeds).tolist()

# ── SMC settings ───────────────────────────────────────────────────────────────
GlobalDraws  = 1000     # particles per independent SMC run
GlobalTune   = 5000    # unused by SMC; kept only for old scripts/imports
GlobalChains = 4     # independent SMC runs
GlobalSMCCores = None # None: use one worker per chain; set lower to queue chains
GlobalSeed   = 42

# ── Prior hyperparameters ──────────────────────────────────────────────────────
GlobalPriorDirichlet = 1.0   # Dirichlet concentration for mixing weights
GlobalPriorKsiSigma  = 0.3   # HalfNormal scale for the ksi hyperprior
GlobalEstimateKsi    = False # False: fix ksi_i = GlobalFixedKsi for all subjects
GlobalFixedKsi       = 0.3   # noise multiplier used when GlobalEstimateKsi is False

# Normal-Inverse-Gamma-style hyperprior on (μ_G, σ_G²) for cluster-level LogNormals.
#
# σ_G²[j] ~ InverseGamma(GlobalPriorIG_Alpha, GlobalPriorIG_Beta)
#   E[σ_G²] = Beta/(Alpha-1) = 1.0  →  σ_G ≈ 1 on the log-scale (diffuse but finite)
#   This allows the prior to cover the full admissible range without dominating the posterior.
#
# μ_G[j] | σ_G[j] ~ Normal(μ_0[j], σ_G[j])
#   μ_0[j] are the benchmark values in log-space (see GlobalPriorMu0 below).
#   Centering at log(1)=0 for alpha, lamb, gamma/palpha, delta encodes the
#   theoretically neutral points (linearity / no distortion / no loss aversion).
#   The r benchmark uses the log geometric midpoint of its admissible range.
GlobalPriorIG_Alpha = 3.0    # InverseGamma shape  (α > 2 ensures finite variance)
GlobalPriorIG_Beta  = 2.0    # InverseGamma scale  (E[σ_G²] = 2/(3-1) = 1.0)

# Benchmark log-scale means μ_0 for [r, alpha, lamb, gamma/palpha, delta].
# Index:  0=r  1=alpha  2=lamb  3=gamma_or_palpha  4=delta
# log(1) = 0 encodes the theoretically neutral point for power/weighting parameters.
# r has no strong behavioral anchor, so we leave it as None to let build_model()
# compute the geometric midpoint from the method-specific bounds at runtime.
GlobalPriorMu0 = [None, 0.0, 0.0, 0.0, 0.0]

# ── Output paths ───────────────────────────────────────────────────────────────
GlobalOutputDir  = "outputs"
GlobalTracePath  = "outputs/trace.nc"
GlobalSummaryCSV = "outputs/summary.csv"
GlobalPlotsDir   = "outputs/plots"
