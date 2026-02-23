# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:19:11 2024

@author: A R Fogg
"""

import os
import sys

import pandas as pd
import numpy as np

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\wind_utility')
import calc_integrated_power

fig_dir = os.path.join("C:" + os.sep,
                       r"Users\Alexandra\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")


def define_freq_bands():
    """
    Return frequency bands to integrate power over.

    Returns
    -------
    freq_bands : np.array
        Upper and lower frequencies for several bands. Of
        shape number of bands x 2.

    """
    freq_bands = np.array([[20, 50],
                           [50, 100],
                           [100, 300],
                           [300, 500],
                           [500, 700],
                           [700, 850],
                           [100, 400],
                           [100, 650],
                           [150, 400],
                           [150, 650],
                           [650, 800]
                           ])

    return freq_bands


def select_akr_intervals(interval,
                         freq_cols=['ipwr_100_400kHz', 'ipwr_50_100kHz']):
    """
    Load in AKR intensity over a defined interval, over multiple
    frequency bands.

    Parameters
    ----------
    interval : string
        Name of defined interval. Options are defined in
        interval_options from return_test_intervals.

    Returns
    -------
    output_df : pd.DataFrame
        DataFrame containing requested AKR data.

    """

    interval_options = return_test_intervals()

    # Check parsed interval is correct
    if interval not in np.array(interval_options.tag):
        print('Parsed interval not a valid option.')
        print('Exiting...')
        return
    else:
        selected = interval_options.loc[interval_options.tag == interval,
                                        :].reset_index()

    # Which years are needed?
    syear = selected.stime[0].year
    eyear = selected.etime[0].year

    # Read in data
    freq_bands = define_freq_bands()
    akr_df = calc_integrated_power.read_run_store_power_with_wind_location(
        syear, eyear, freq_bands=freq_bands)
    # Select only the interval requested
    akr_df = akr_df.loc[(akr_df.datetime >= selected.stime[0]) &
                        (akr_df.datetime <= selected.etime[0]),
                        :].reset_index(drop=True)
    akr_df['unix'] = [t.timestamp() for t in akr_df.datetime]
    
    # Preprocess data if requested
    if selected.logged[0]:
        for freq_col in freq_cols:
            # Smoothing 
            averaging_df = akr_df.copy(deep=True)
            averaging_df.set_index('datetime', inplace=True)
            smoothed = averaging_df[freq_col].rolling('3h').mean()
            akr_df["smoothed_" + freq_col] = smoothed.values
                       
            # Log the power
            akr_df['log_smoothed_' + freq_col] = np.log(akr_df["smoothed_" + freq_col])
    
            # Smoothed, logged, but zero -> np.nan
            zero_i, = np.where(akr_df["smoothed_" + freq_col] == 0)
            akr_df.loc[zero_i, 'log_smoothed_' + freq_col] = np.nan


    return akr_df


def return_test_intervals():
    """
    Function where test intervals are defined.

    Returns
    -------
    interval_options : pd.DataFrame
        Containing start, end and tag for test intervals.

    """

    interval_options = pd.DataFrame(
        {'tag': ['full_archive',
                 'cassini_flyby',
                 'long_nightside_period',
                 'cassini_flyby_logged'],
         'title': ['Decadal Archive\n(1995-2004)',
                   'Wind-Cassini Conjunction\n(15th Aug 1999-14th Sep 1999)',
                   'Nightside Interval\n(22:36 11th Oct 2003 - 09:36 2nd Mar 2004)',
                   'Wind-Cassini Conjunction [log intensity]\n(15th Aug 1999-14th Sep 1999)'],
         'label': ['Decadal archive', 'Wind-Cassini Conjunction',
                   'Nightside interval', 'Wind-Cassini Conjunction\n[log intensity]'],
         'color': ['mediumpurple', 'gold', 'salmon', 'grey'],
         'stime': [pd.Timestamp(1995, 1, 1, 0),
                   pd.Timestamp(1999, 8, 15, 0),
                   pd.Timestamp(2003, 10, 11, 22, 36),
                   pd.Timestamp(1999, 8, 15, 0)],
         'etime': [pd.Timestamp(2004, 12, 31, 23, 59),
                   pd.Timestamp(1999, 9, 14, 23, 59),
                   pd.Timestamp(2004, 3, 2, 9, 36),
                   pd.Timestamp(1999, 9, 14, 23, 59)],
         'logged': [False, False, False, True]})

    return interval_options
