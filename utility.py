# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:18:04 2024

@author: A R Fogg
"""

import numpy as np
import pandas as pd

from scipy import interpolate


def interpolate_mlt(desired_timestamps, data_df, mlt_flag='decimal_gseLT'):
    # data_df needs unix and irish_mlt
    # desired_timestamps is a numpy array

    unwrapped_mlts = np.unwrap(data_df[mlt_flag], period=24)
    mlt_func = interpolate.interp1d(data_df['unix'], unwrapped_mlts)

    out_mlt = np.full(desired_timestamps.size, np.nan)

    for i in range(desired_timestamps.size):
        unwrapped_interp_mlt = mlt_func(
            pd.Timestamp(desired_timestamps[i]).timestamp())
        out_mlt[i] = unwrapped_interp_mlt % 24

    return out_mlt
