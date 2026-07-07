"""
Batch settings for running multiple Bayesian CPT estimations.

Edit RUNS, then launch:

    python3 run_batch.py

Each run writes to its own folder under OUTPUT_ROOT. The scheduler keeps at
most MAX_CORES allocated across all active runs.
"""

MAX_CORES = 32
OUTPUT_ROOT = "batch_outputs"
SKIP_COMPLETED = True
POLL_SECONDS = 5

# Values here are inherited by every entry in RUNS unless that run overrides
# them. "cores" is the number of PyMC SMC workers to allocate for this run.
DEFAULTS = {
    "method": "tk",
    # Utility function: "cara" (current baseline) or "power".
    "utility": "cara",
    "discounting": "hyperbolic",
    "lottery_set": "full",
    "clusters": 1,
    "draws": 5000,
    "chains": 5,
    "cores": 5,
    "seed": 42,
    "ksi_mode": "marginalized",
    # Structural reference-point delta. Equal endpoints fix it; an interval estimates it.
    "delta_bounds": (1e-6, 1.0),
    "progressbar": False,
}

RUNS = [
    {    
    "method": "tk",
    "clusters": 1,
    "lottery_set": "full",
    "draws": 5000,
    "chains": 5,
    "cores": 5,
    "seed": 42,
    "ksi_mode": "marginalized",
    "progressbar": False,
    "delta_bounds": (1e-6, 1.0),
    },

    {    
    "method": "tk",
    "clusters": 2,
    "lottery_set": "full",
    "draws": 5000,
    "chains": 5,
    "cores": 5,
    "seed": 42,
    "ksi_mode": "marginalized",
    "progressbar": False,
    "delta_bounds": (1e-6, 1.0),
    },



    {
    "method": "tk",
    "discounting": "quasi_hyperbolic",
    "clusters": 1,
    "lottery_set": "full",
    "draws": 5000,
    "chains": 5,
    "cores": 5,
    "seed": 42,
    "ksi_mode": "marginalized",
    "progressbar": False,
    "delta_bounds": (1e-6, 1.0),
    },

    {
    "method": "tk",
    "discounting": "quasi_hyperbolic",
    "clusters": 2,
    "lottery_set": "full",
    "draws": 5000,
    "chains": 5,
    "cores": 5,
    "seed": 42,
    "ksi_mode": "marginalized",
    "progressbar": False,
    "delta_bounds": (1e-6, 1.0),
    },


    {
    "method": "tk",
    "discounting": "exponential",
    "clusters": 1,
    "lottery_set": "full",
    "draws": 5000,
    "chains": 5,
    "cores": 5,
    "seed": 42,
    "ksi_mode": "marginalized",
    "progressbar": False,
    "delta_bounds": (1e-6, 1.0),
    },

    {
    "method": "tk",
    "discounting": "exponential",
    "clusters": 2,
    "lottery_set": "full",
    "draws": 5000,
    "chains": 5,
    "cores": 5,
    "seed": 42,
    "ksi_mode": "marginalized",
    "progressbar": False,
    "delta_bounds": (1e-6, 1.0),
    },

    {
    "method": "tk",
    "discounting": "quasi_hyperbolic",
    "clusters": 3,
    "lottery_set": "full",
    "draws": 5000,
    "chains": 5,
    "cores": 5,
    "seed": 42,
    "ksi_mode": "marginalized",
    "progressbar": False,
    "delta_bounds": (1e-6, 1.0),
    },
        {
    "method": "tk",
    "discounting": "quasi_hyperbolic",
    "clusters": 3,
    "lottery_set": "full",
    "draws": 5000,
    "chains": 5,
    "cores": 5,
    "seed": 42,
    "ksi_mode": "marginalized",
    "progressbar": False,
    "delta_bounds": (1e-6, 1.0),
    },

]
