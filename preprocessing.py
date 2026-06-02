"""
Load and preprocess data for Bayesian CPT estimation.
All imports are now local — no dependency on the Estimation project.
"""

import numpy as np
import pandas as pd
import functions as f
from data import process
from likelihood import _preprocess_rows
from GlobalSettings import GlobalLottery


def get_observed_ce():
    """Return the six columns needed for likelihood computation."""
    data, _, _, _ = process()
    return data[[
        "participant_label", "lottery_id", "round_number",
        "ce_observed", "realized_period1_label", "realized_period2_label",
    ]].copy()


def load_and_preprocess(lottery_set=None):
    """
    Full pipeline: load data → transform lotteries → run _preprocess_rows.

    Parameters
    ----------
    lottery_set : dict or None
        Raw lottery dict (pre-transform).  Defaults to GlobalLottery.

    Returns
    -------
    preproc   : 11-tuple from _preprocess_rows
                (y_proc, EL_arr, EL_fe_arr, EL_c1_arr,
                 t_arr, Z1_arr, Z2_arr, Zt_arr, si_arr, spread_arr, obs_arr)
    subjects  : sorted list of participant labels
    subj_index : dict  label → integer index
    lotteries  : transformed lottery dict (has 'outcomes' key)
    """
    if lottery_set is None:
        lottery_set = GlobalLottery

    lotteries = f.transform(lottery_set)

    y = get_observed_ce()
    subjects   = sorted(y["participant_label"].unique())
    subj_index = {s: i for i, s in enumerate(subjects)}

    preproc = _preprocess_rows(y, lotteries, subj_index)
    return preproc, subjects, subj_index, lotteries
