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


def generic_fft_function(time, y, temporal_resolution,
                         plot=True, resolution_lim=True,
                         fft_xlims=[0, 36],
                         signal_xlims=[np.nan, np.nan],
                         fontsize=15,
                         vertical_indicators=[],
                         unix_to_dtime=False):
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
    plot : BOOL, optional
        If plot==True, a diagnostic plot for the FFT is
        presented. The default is True.
    resolution_lim : bool, optional
        If True, periods below twice the resolution of
        the data are not presented. The default is True.
    fft_xlims : list, optional
        If provided, the limits for the FFT x axis. The
        default is [0, 36].
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

    Returns
    -------
    freq : np.array
        Frequency of FFT calculation in Hz.
    period : np.array
        Period of FFT calculation in hours.
    fft_amp : np.array
        Amplitude of FFT calculation.

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

    if resolution_lim:
        j, = np.where(period > (2. * (temporal_resolution.total_seconds() /
                                      (60*60))))
        freq = freq[j]
        period = period[j]
        fft_amp = fft_amp[j]

    if plot:
        fig, ax = plt.subplots(ncols=3, figsize=(18, 6))

        # Plot original signal
        ax[0].plot(time, y, 'crimson')

        ax[0].set_ylabel('Amplitude', fontsize=fontsize)
        ax[0].set_xlabel('Time (s)', fontsize=fontsize)
        ax[0].set_title('Input', fontsize=fontsize)
        ax[0].tick_params(labelsize=fontsize)

        t = ax[0].text(0.05, 0.95, '(a)', transform=ax[0].transAxes,
                       fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

        # Plot FFT periodogram
        ax[1].stem(period, fft_amp, 'c',
                   markerfmt=" ", basefmt="-c")

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
                ax[1].axvline(h, color='grey', linestyle='dashed',
                              linewidth=1.5)
                trans = transforms.blended_transform_factory(ax[1].transData,
                                                             ax[1].transAxes)
                ax[1].text(h, 1.05, str(h), transform=trans,
                           fontsize=fontsize, va='top', ha='center',
                           color='grey')

        # Plot inverse FFT signal
        ax[2].plot(time, ifft(X), 'rebeccapurple')

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

        fig.tight_layout()

        return freq, period, fft_amp, fig, ax

    else:
        # No plot to return
        return freq, period, fft_amp
