# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:18:04 2024

@author: A R Fogg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate

import aaft


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


def generate_random_phase_surrogate(data, plot=False):

    print('Generating Random Phase Surrogate')
    surrogate = aaft.AAFTsur(data)

    if plot:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

        # Histogram
        bins = np.linspace(np.nanmin(data), np.nanmax(data), 51)
        ax[0].hist(data, bins=bins, label='Data', color='lightgrey')
        ax[0].hist(surrogate, bins=bins, label='Surrogate',
                   histtype='step', color='palevioletred')

        ax[0].set_xlabel('Data units')
        ax[0].set_ylabel('Occurrence')
        ax[0].set_title('Distribution')
        ax[0].legend()

        # Timeseries
        i_lims = [0, 100]
        ax[1].plot(data[i_lims[0]:i_lims[1]], color='grey', label='Data')
        ax[1].plot(surrogate[i_lims[0]:i_lims[1]], color='palevioletred',
                   label='Surrogate')

        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Magnitude')
        ax[1].set_title('Timeseries')
        ax[1].legend()

        fig.tight_layout()

    return surrogate


def calc_longitude_of_sun(data, lon_tag='lon_gsm', plot=False):
    # data is a DataFrame containing datetime, decimal_gseLT, lon_gsm
    # NEED TO CHECK WITH SIYUAN ABOUT LON GSM OR GSE!!
    
    print('hello')
    
    lon_sol = data[lon_tag] - ((12. - data.decimal_gseLT) * 15.)
    lon_sol = lon_sol % 360
    
    if plot:
        fig, ax = plt.subplots()
        ax.set_facecolor('lightgrey')

        ax.hist([data[lon_tag], lon_sol], color=['white', 'blue'],
                edgecolor='black', density=True,
                label=['$\lambda_{sc}$', '$\lambda_{sun}$'])

        ax.legend()
        ax.set_xlabel("Longitude ($^{\circ}$)")
        ax.set_ylabel("Density")
    
    return lon_sol