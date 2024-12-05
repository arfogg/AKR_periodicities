# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:43:04 2024

@author: A R Fogg

Definitions of data with a 24 hour periodicity.
"""

import numpy as np

import matplotlib.pyplot as plt

from neurodsp.sim import sim_oscillation


def oscillating_signal(osc_freq, add_noise=True, noise_level=0.1,
                       plot=False, fontsize=15):
    """
    Function to create a timeseries of oscillating
    signal using neurodsp package.

    Parameters
    ----------
    osc_freq : float
        Period of the desired oscillation in hours.
    add_noise : bool, optional
        If True, random noise is added to the signal. The default is True.
    noise_level : float, optional
        Fraction of the intensity mean that the normal distribution scale
        is defined as. The default is 0.1.
    plot : bool, optional
        If plot == True, a diagnostic plot of the
        generated signal is presented. The default
        is False.
    fontsize: int, optional
        Defines fontsize on plot. The default is 15.

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

    akr_osc = sim_oscillation(yr_secs, 1/res_secs, 1/(osc_freq * 60 * 60),
                              cycle='sine')
    akr_osc = (akr_osc + 2.0) * 1e6
    # ^ make positive and put to the order of AKR power

    if add_noise:
        akr_osc = akr_osc + np.random.normal(loc=np.mean(akr_osc),
                                             scale=noise_level *
                                             np.mean(akr_osc),
                                             size=akr_osc.size)

    if plot:
        n_plot_osc = 4
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time, akr_osc, linewidth=1.0, color='black')
        ax.set_xlim(0, n_plot_osc*osc_freq*60*60)
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.set_ylabel('Amplitude', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

        for i in range(n_plot_osc):
            ax.axvline(float(i)*osc_freq*60*60, linestyle='dashed',
                       color='grey', linewidth=1.)

    return time, akr_osc
