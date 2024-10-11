# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:16:17 2024

@author: A R Fogg

Various functions to assess periodicities across a generic AKR
intensity timeseries.

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from numpy.fft import fft, ifft


def generic_fft_function(time, y, temporal_resolution):
    """

    Parameters
    ----------
    time : np.array
        Time axis for y in seconds. Unix time is
        recommended for real data.
    y : np.array
        Signal.
    temporal_resolution : pd.Timedelta
        Seperation of consecutive points in time.

    Returns
    -------
    freq : np.array
        Frequency of FFT calculation in Hz.
    period : np.array
        Period of FFT calculation in hours.
    fft_amp : np.array
        Amplitude of FFT calculation.
    inverse_signal : np.array
        Inverse FFT, comparable to input y.

    """

    # Calculate sampling rate in Hz
    sampling_rate = 1 / (temporal_resolution.total_seconds())

    X = fft(y)  # y axis for FFT plot
    N = len(X)  # number of FFT points
    n = np.arange(N)    # 0 - N array. integers
    T = N/sampling_rate    # number of FFT points / number of obs per sec
    freq = n/T  # freqs fft is evaluated at

    # Functions to convert between period in hours
    #   and frequency in Hz
    # period = 1 / freq
    # period = period / (60*60)   # period in hours
    def period_to_freq(x):
        ticks = []
        for tick in x:
            if tick != 0:
                ticks.append(1. / (tick * (60.*60.)))
            else:
                ticks.append(0)
        return np.array(ticks)

    def freq_to_period(x):
        ticks = []
        for tick in x:
            if tick != 0:
                ticks.append((1. / tick) / (60.*60.))
            else:
                ticks.append(0)
        return np.array(ticks)

    period = freq_to_period(freq)

    fft_amp = np.abs(X)

    inverse_signal = ifft(X)
    
    return freq, period, fft_amp, inverse_signal


def plot_fft_summary(time, y, temporal_resolution,
                     freq, period, fft_amp, inverse_signal,
                     surrogate_period=None, surrogate_fft_amp=None,
                     fft_xlims=[0, 36],
                     signal_xlims=[np.nan, np.nan],
                     fontsize=15,
                     vertical_indicators=[],
                     unix_to_dtime=False,
                     resolution_lim=True,
                     signal_y_log=False,
                     input_fmt={'color': 'royalblue', 'linewidth': 1.},
                     ifft_fmt={'color': 'royalblue', 'linewidth': 1.}):
    """

    Parameters
    ----------
    time : np.array
        Time axis for y in seconds. Unix time is
        recommended for real data.
    y : np.array
        Signal.
    temporal_resolution : pd.Timedelta
        Seperation of consecutive points in time.
    freq : np.array
        Frequency of FFT calculation in Hz.
    period : np.array
        Period of FFT calculation in hours.
    fft_amp : np.array
        Amplitude of FFT calculation.
    inverse_signal : np.array
        Inverse FFT, comparable to input y.
    surrogate_period : np.array
        Period of FFT for surrogate intensity.
    surrogate_fft_amp : np.array
        Amplitude of FFt calculation for surrogate intensity.
    fft_xlims : list, optional
        X limits for FFT periodogram in hours. The default
        is [0, 36].
    signal_xlims : list, optional
        If provided, the xaxis of the signal and IFFT
        plots are limits to these values. This can allow
        nicer plots of a few oscillations. The default
        is [np.nan, np.nan].
    fontsize : int, optional
        Fontsize for plotting. The default is 15.
    vertical_indicators : list, optional
        A list of positions to draw a vertical line on the
        centre FFT plot (in hours). The default is [].
    unix_to_dtime : bool, optional
        If True, the xaxis for signal axes will be displayed
        in YYYY MM/DD HH:MM format. The default is False.
    resolution_lim : bool, optional
        If True, periods below twice the resolution of
        the data are not presented. The default is True.
    signal_y_log : bool, optional
        If True, change the y scale to log for signal axes.
        The default is False.
    input_fmt : dict, optional
        Dictionary containing formatting options for input
        signal plot. The default is {'color': 'royalblue',
                                     'linewidth': 1.}.
    ifft_fmt : dict, optional
        Dictionary containing formatting options for IFFT
        signal plot. The default is {'color': 'royalblue',
                                     'linewidth': 1.}.

    Returns
    -------
    fig : matplotlib figure
        Figure containing the plot.
    ax : array of matplotlib axes
        Axes containing plots.

    """
    if resolution_lim:
        j, = np.where(period > (2. * (temporal_resolution.total_seconds() /
                                      (60*60))))
        freq = freq[j]
        period = period[j]
        fft_amp = fft_amp[j]

    fig, ax = plt.subplots(ncols=3, figsize=(18, 6))

    # Plot original signal
    ax[0].plot(time, y, **input_fmt)

    ax[0].set_ylabel('Amplitude', fontsize=fontsize)
    ax[0].set_xlabel('Time (s)', fontsize=fontsize)
    ax[0].set_title('Input', fontsize=fontsize)
    ax[0].tick_params(labelsize=fontsize)

    t = ax[0].text(0.05, 0.95, '(a)', transform=ax[0].transAxes,
                   fontsize=fontsize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    # Plot FFT periodogram
    ax[1].plot(period, fft_amp, color='grey', label='Input')
    if surrogate_fft_amp is not None:
        ax[1].plot(surrogate_period, surrogate_fft_amp,
                   color='salmon', label='Surrogate', linewidth=1., alpha=0.5)
        ax[1].plot(surrogate_period,
                   pd.Series(surrogate_fft_amp).rolling(window=50).mean(),
                   color='brown', label='Surrogate 50 point\nrolling mean')

    ax[1].legend(fontsize=fontsize)
    ax[1].set_xlabel('Period (hours)', fontsize=fontsize)
    ax[1].set_ylabel('FFT Amplitude', fontsize=fontsize)
    ax[1].set_title('FFT of input', fontsize=fontsize)
    ax[1].tick_params(labelsize=fontsize)
    ax[1].set_xlim(fft_xlims)

    # Calculate y lims
    k, = np.where((period >= fft_xlims[0]) & (period <= fft_xlims[1]))
    fft_ylims = [0.9*np.nanmin(fft_amp[k]), 1.1*np.nanmax(fft_amp[k])]
    ax[1].set_ylim(fft_ylims)

    t = ax[1].text(0.05, 0.95, '(b)', transform=ax[1].transAxes,
                   fontsize=fontsize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    if vertical_indicators != []:
        for h in vertical_indicators:
            ax[1].axvline(h, color='navy', linestyle='dashed',
                          linewidth=1.5)
            trans = transforms.blended_transform_factory(ax[1].transData,
                                                         ax[1].transAxes)
            ax[1].text(h, 1.05, str(h), transform=trans,
                       fontsize=fontsize, va='top', ha='center',
                       color='navy')

    # Plot inverse FFT signal
    ax[2].plot(time, inverse_signal, **ifft_fmt)

    ax[2].set_xlabel('Time (s)', fontsize=fontsize)
    ax[2].set_ylabel('Amplitude', fontsize=fontsize)
    ax[2].set_title('Inverse FFT', fontsize=fontsize)
    ax[2].tick_params(labelsize=fontsize)

    t = ax[2].text(0.05, 0.95, '(c)', transform=ax[2].transAxes,
                   fontsize=fontsize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    # Adjust x axes limits if required
    if (~np.isnan(signal_xlims[0])) & (~np.isnan(signal_xlims[1])):
        ax[0].set_xlim(signal_xlims)
        ax[2].set_xlim(signal_xlims)

        # Calculate y lims
        q, = np.where((time >= signal_xlims[0]) &
                      (time <= signal_xlims[1]))
        signal_ylims = [0.9*np.nanmin(y[q]), 1.1*np.nanmax(y[q])]
        ax[0].set_ylim(signal_ylims)
        ax[2].set_ylim(signal_ylims)

    if unix_to_dtime:
        tick_loc = ax[0].get_xticks()
        tick_lab = pd.to_datetime(pd.Series(tick_loc),
                                  unit='s').dt.strftime('%Y\n%m/%d\n%H:%M')
        ax[0].set_xticks(tick_loc, tick_lab, fontsize=fontsize)
        ax[2].set_xticks(tick_loc, tick_lab, fontsize=fontsize)

    if signal_y_log:
        ax[0].set_yscale('log')
        ax[2].set_yscale('log')

    fig.tight_layout()

    return fig, ax


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
    shifted_y : np.ndarray
        Array containing y shifted by lags, of shape
        y.size x n_shifts.
    acf : np.array
        ACF as a function of lags.
    lags : np.array
        Lag in seconds.

    """

    starting_lag_i = int(starting_lag / temporal_resolution)

    # Initialise arrays
    shifted_y = np.full((y.size, n_shifts), np.nan)
    acf = np.full(n_shifts, np.nan)
    lags = np.full(n_shifts, np.nan)

    for i in range(0, n_shifts):
        # Shift y
        shifted_y[:, i] = np.append(y[-(i+starting_lag_i):],
                                    y[:-(i+starting_lag_i)])
        # Calculate ACF
        acf[i] = np.correlate(y, shifted_y[:, i])
        # Calculate lag in seconds
        lags[i] = temporal_resolution * (i + starting_lag_i)

    return shifted_y, acf, lags


def plot_autocorrelogram(lags, acf, fontsize=15):
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

    Returns
    -------
    fig : matplotlib figure object
        Figure containing plot.
    ax : matplotlib axes object
        Axes containing plot.

    """
    fig, ax = plt.subplots()

    ax.plot(lags, acf, linewidth=1., color='black')

    # Convert x ticks from seconds to readable format
    # Rough - plotting a tick every 2 hours
    n_ticks = int(np.floor(np.max(ax.get_xticks()) / (2. * 60. * 60.)))
    tick_pos = []
    tick_str = []
    for i in range(n_ticks):
        tick_pos.append(i * (2. * 60. * 60.))
        tick_str.append(str(int(2. * i)))
    ax.set_xticks(tick_pos, tick_str)

    ax.set_ylabel('ACF', fontsize=fontsize)
    ax.set_xlabel('Lag (hours)', fontsize=fontsize)

    return fig, ax


def test_acf():
    y = np.sin(np.linspace(0, 11, 51))
    temporal_resolution = pd.Timedelta(minutes=3)
    n_shifts = 5
    lags = np.array(range(n_shifts)) * temporal_resolution
    shifted_y, acf = autocorrelation(y, n_shifts)

    fig, ax = plot_autocorrelogram(lags, acf)