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

# ── Parameter bounds ───────────────────────────────────────────────────────────
# Structural CPT params per cluster: [r, alpha, lamb, gamma/palpha, delta]
# Reference-point weights are continuous cluster-specific a_weights on the 4-simplex.
GlobalTKBounds = [
    (1e-6, 0.01), (0.5, 3), (0.5, 3.0), (0.2, 1.0),
    (0, 1), (0.0, 1), (0.0, 1), (0.2, 1),
]
GlobalPrelecBounds = [
    (1e-6, 0.1), (0.5, 1.5), (0.99, 3.5), (1.0, 1.0), (0.4, 1),
    (0, 1), (0.0, 1), (0.0, 1), (0.0, 1),
]
GlobalKsiBounds = (1e-4, 1.5)

# ── Model settings ─────────────────────────────────────────────────────────────
GlobalMethod   = "tk"
GlobalCluster  = 1      # number of latent preference clusters (C)

# Reference point components (Baillon, Bleichrodt & Spinu, 2015)
# Continuous cluster-specific weights a_weights[k] live on these 4 components:
# SQ = status quo, PA = partial adaptation, LE = lagged expectation, FE = forward expectation.
GlobalK        = 4      # kept for compatibility; continuous RP weights require 4 components

GlobalTol      = 1e-4
GlobalInterMax = 1000
GlobalData     = "/home/ubuntu/Estimation/gamedata.csv"
GlobalRSQMode  = "zero"
GlobalFixTau   = False

GlobalNSeeds     = 1000
GlobalMasterSeed = 42
GlobalSeedsSet   = np.random.default_rng(GlobalMasterSeed).integers(0, 10000, size=GlobalNSeeds).tolist()

# ── SMC settings ───────────────────────────────────────────────────────────────
GlobalDraws    = 3000
GlobalTune     = 5000   # unused by SMC; kept for legacy imports
GlobalChains   = 5
GlobalSMCCores = None
GlobalSeed     = 42

# ── Prior hyperparameters ──────────────────────────────────────────────────────
GlobalPriorDirichlet = 1.0   # concentration for π (cluster mixing weights)
GlobalPriorDirichletRP = 1.0 # concentration for cluster-specific a_weights
GlobalKsiMode        = "marginalized"  # one of: "estimated", "fixed", "marginalized"
GlobalPriorKsiSigma  = 0.3
GlobalFixedKsi       = 0.3
GlobalPriorKsiIGAlpha = 3.0
GlobalPriorKsiIGBeta  = (GlobalPriorKsiIGAlpha - 1.0) * GlobalFixedKsi**2
GlobalEstimateKsi    = GlobalKsiMode == "estimated"  # legacy compatibility

# Normal-Inverse-Gamma hyperprior on log-scale structural parameters.
# theta_rest follows a truncated LogNormal on the admissible bounds.
GlobalPriorIG_Alpha = 3.0
GlobalPriorIG_Beta  = 2.0

# Benchmark log-scale neutral values for [r, alpha, lamb, gamma/palpha, delta].
GlobalPriorMu0 = [None, 0.0, 0.0, 0.0, 0.0]

# ── Output paths ───────────────────────────────────────────────────────────────
GlobalOutputDir  = "outputs"
GlobalTracePath  = "outputs/trace.nc"
GlobalSummaryCSV = "outputs/summary.csv"
GlobalPlotsDir   = "outputs/plots"
