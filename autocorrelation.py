# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:19:36 2025

@author: A R Fogg
"""

import numpy as np

import matplotlib.pyplot as plt


def autocorrelation(y, n_shifts, temporal_resolution=180, starting_lag=7200):
    """
    Calculate the autocorrelation (ACF) for a signal y
    across various lags.

    Parameters
    ----------
    y : np.array
        Signal to be analysed.
    n_shifts : int
        Number of lags to calculate the ACF for.
    temporal_resolution : int, optional
        Temporal resolution of y in seconds. The
        default is 180.
    starting_lag : int, optional
        Starting lag, in seconds. The default is 7200.

    Returns
    -------
    acf : np.array
        ACF as a function of lags.
    lags : np.array
        Lag in seconds.

    """
    print('Calculating the ACF for ' + str(n_shifts) + ' lags')
    starting_lag_i = int(starting_lag / temporal_resolution)

    # Initialise arrays
    acf = np.full(n_shifts, np.nan)
    lags = np.full(n_shifts, np.nan)

    for i in range(0, n_shifts):
        # Shift y
        shifted_y = np.append(y[-(i+starting_lag_i):],
                              y[:-(i+starting_lag_i)])
        # Calculate ACF
        acf[i] = np.correlate(y, shifted_y)
        # Calculate lag in seconds
        lags[i] = temporal_resolution * (i + starting_lag_i)

    return acf, lags


def plot_autocorrelogram(lags, acf, fontsize=15, tick_sep_hrs=12.,
                         highlight_period=24.,
                         highlight_fmt={'color': 'grey',
                                        'linestyle': 'dashed',
                                        'linewidth': 1.},
                         acf_fmt={'color': 'forestgreen',
                                  'linewidth': 1.}):
    """
    Plot the ACF as a function of lags.

    Parameters
    ----------
    lags : np.array
        Lag in seconds.
    acf : np.array
        ACF as a function of lags.
    fontsize : int, optional
        Fontsize parsed to matplotlib. The default is 15.
    tick_sep_hrs : float, optional
        Seperation of x ticks in hours. The default is 12.
    highlight_period : float, optional
        Vertical line is drawn at each integer number of this
        repeating interval (in hours) on the x axis. The
        default is 24.
    highlight_fmt : dict, optional
        Format of the vertical highlight lines to be parsed
        to matplotlib.
    acf_fmt : dict, optional
        Format of the ACF line to be parsed to matplotlib.

    Returns
    -------
    fig : matplotlib figure object
        Figure containing plot.
    ax : matplotlib axes object
        Axes containing plot.

    """

    # Define plotting window
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot ACF as a function of lag
    ax.plot(lags, acf, **acf_fmt)

    # Convert x ticks from seconds to readable format
    n_ticks = int(np.floor(np.max(ax.get_xticks()) /
                           (tick_sep_hrs * 60. * 60.)))
    tick_pos = []
    tick_str = []
    for i in range(n_ticks):
        tick_pos.append(i * (tick_sep_hrs * 60. * 60.))
        tick_str.append(str(int(tick_sep_hrs * i)))
    ax.set_xticks(tick_pos, tick_str)

    # Label axes
    ax.set_ylabel('ACF', fontsize=fontsize)
    ax.set_xlabel('Lag (hours)', fontsize=fontsize)

    # Draw vertical lines each highlight period
    n_vert = int(np.floor((ax.get_xlim()[1]) / (highlight_period * 60. * 60.)))
    for i in range(n_vert):
        ax.axvline(((i + 1) * highlight_period)*(60. * 60.), **highlight_fmt)

    # Formatting
    ax.tick_params(labelsize=fontsize)

    return fig, ax
