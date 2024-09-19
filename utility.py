# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:18:04 2024

@author: A R Fogg
"""

import numpy as np
import pandas as pd

from scipy import interpolate


def interpolate_mlt(desired_timestamps, data_df, mlt_flag='decimal_gseLT'):
    """
    Interpolate magnetic local time / MLT. Requires special
    approach as MLT is a periodic parameter (i.e. 24 == 0).

    Parameters
    ----------
    desired_timestamps : np.array
        pd.Timestamps for MLT to be calculated at.
    data_df : pd.DataFrame
        DataFrame containing MLT (mlt_flag) as a function of
        'unix'. These are used to create the interpolation
        function.
    mlt_flag : string, optional
        Column in data_df containing MLT. The default is
        'decimal_gseLT'.

    Returns
    -------
    out_mlt : np.array
        Interpolated MLT as a function of desired_timestamps.

    """
    # Unwrap MLT
    unwrapped_mlts = np.unwrap(data_df[mlt_flag], period=24)
    # Generate interpolation function
    mlt_func = interpolate.interp1d(data_df['unix'], unwrapped_mlts)

    # Initialise empty MLT array
    out_mlt = np.full(desired_timestamps.size, np.nan)

    # For each desired timestamp, estimate MLT
    for i in range(desired_timestamps.size):
        unwrapped_interp_mlt = mlt_func(
            pd.Timestamp(desired_timestamps[i]).timestamp())
        out_mlt[i] = unwrapped_interp_mlt % 24

    return out_mlt
