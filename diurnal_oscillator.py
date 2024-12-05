# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:43:04 2024

@author: A R Fogg

Definitions of data with a 24 hour periodicity.
"""

import numpy as np

import matplotlib.pyplot as plt

from neurodsp.sim import sim_oscillation


def oscillating_signal(osc_freq, plot=False):
    """
    Function to create a timeseries of oscillating
    signal using neurodsp package.

    Parameters
    ----------
    osc_freq : float
        Period of the desired oscillation in hours.
    plot : bool, optional
        If plot == True, a diagnostic plot of the
        generated signal is presented. The default
        is False.

    Returns
    -------
    time : np.array
        Time axis in seconds.
    akr_osc : np.array
        Signal.

    """
    # Create fake time axis
    yr_secs = 365*24*60*60    # One year in seconds
    res_secs = 3*60   # Temporal resolution of 3 minutes

    time = np.arange(0, yr_secs, res_secs)

    akr_osc = sim_oscillation(yr_secs, 1/res_secs, 1/(osc_freq*60*60),
                              cycle='sine')
    akr_osc = (akr_osc+2.0) * 1e6
    # ^ make positive and put to the order of AKR power

    if plot:
        fig, ax = plt.subplots()
        ax.plot(time, akr_osc)
        ax.set_xlim(0, 4*osc_freq*60*60)

    return time, akr_osc