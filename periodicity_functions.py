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
import scipy.signal as signal


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



    period = freq_to_period(freq)

    fft_amp = np.abs(X)

    inverse_signal = ifft(X)
    
    return freq, period, fft_amp, inverse_signal


def plot_fft_summary(time, y, temporal_resolution,
                     freq, period, fft_amp, inverse_signal,
                     surrogate_period=None, surrogate_fft_amp=None,
                     fft_xlims=[0, 36],
                     signal_xlims=[np.nan, np.nan], signal_ymin=1.,
                     fontsize=15,
                     vertical_indicators=[],
                     unix_to_dtime=False,
                     resolution_lim=True,
                     signal_y_log=False,
                     input_fmt={'color': 'royalblue', 'linewidth': 1.},
                     ifft_fmt={'color': 'royalblue', 'linewidth': 1.},
                     input_ax=None, panel_label=True):
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
    SIGNAL_YMIN
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
    input_ax : np.array of three matplotlib axes
        Axes to do plotting on. The default is None.

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

    if input_ax is None:
        fig, ax = plt.subplots(ncols=3, figsize=(18, 6))
    else:
        ax = input_ax

    # Remove data below signal_ymin
    i_ind, = np.where(y <= signal_ymin)
    y[i_ind] = np.nan
    # combined_rounded_df.loc[
    #             combined_rounded_df.integrated_power == 0].index
        # pwr = np.array(combined_rounded_df.
        #                integrated_power.copy(deep=True))
        # pwr[r_ind] = np.nan
    o_ind, = np.where(inverse_signal <= signal_ymin)
    inverse_signal[o_ind] = np.nan

    # Plot original signal
    ax[0].plot(time, y, **input_fmt)

    ax[0].set_ylabel('Amplitude', fontsize=fontsize)
    ax[0].set_xlabel('Time (UT)', fontsize=fontsize)
    ax[0].set_title('Input', fontsize=fontsize)
    ax[0].tick_params(labelsize=fontsize)

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

    ax[2].set_xlabel('Time (UT)', fontsize=fontsize)
    ax[2].set_ylabel('Amplitude', fontsize=fontsize)
    ax[2].set_title('Inverse FFT', fontsize=fontsize)
    ax[2].tick_params(labelsize=fontsize)

    # Label panels if requested
    if panel_label:
        t = ax[0].text(0.05, 0.95, '(a)', transform=ax[0].transAxes,
                       fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))
        t = ax[1].text(0.05, 0.95, '(b)', transform=ax[1].transAxes,
                       fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))
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

    if input_ax is None:
        fig.tight_layout()
        return fig, ax
    else:
        return ax


def DEP_autocorrelation(y, n_shifts, temporal_resolution=180, starting_lag=7200):
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
    print('Calculating the ACF for ' + str(n_shifts) + ' lags')
    starting_lag_i = int(starting_lag / temporal_resolution)

    # Initialise arrays
    # shifted_y = np.full((y.size, n_shifts), np.nan)
    acf = np.full(n_shifts, np.nan)
    lags = np.full(n_shifts, np.nan)

    for i in range(0, n_shifts):
        # Shift y
        # shifted_y[:, i] = np.append(y[-(i+starting_lag_i):],
        #                            y[:-(i+starting_lag_i)])
        shifted_y = np.append(y[-(i+starting_lag_i):],
                              y[:-(i+starting_lag_i)])
        # Calculate ACF
        acf[i] = np.correlate(y, shifted_y)
        # Calculate lag in seconds
        lags[i] = temporal_resolution * (i + starting_lag_i)

    return acf, lags


def DEP_plot_autocorrelogram(lags, acf, fontsize=15, tick_sep_hrs=12.,
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
    fig, ax = plt.subplots(figsize=(10, 7))

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

    ax.set_ylabel('ACF', fontsize=fontsize)
    ax.set_xlabel('Lag (hours)', fontsize=fontsize)

    # Draw vertical lines each highlight period
    n_vert = int(np.floor((ax.get_xlim()[1]) / (highlight_period * 60. * 60.)))
    for i in range(n_vert):
        ax.axvline(((i + 1) * highlight_period)*(60. * 60.), **highlight_fmt)

    # Formatting
    ax.tick_params(labelsize=fontsize)

    return fig, ax

# def test_LS():
        
#     rng = np.random.default_rng()

#     A = 2.
#     w0 = 1.  # rad/sec
#     nin = 150
#     nout = 100000

#     time = rng.uniform(0, 10*np.pi, nin)

#     y = A * np.cos(w0*time)

#     freqs = np.linspace(0.01, 10, nout)
#     periods = freq_to_period(freqs)
    
#     ls_pgram = generic_lomb_scargle(time, y, freqs)
    
    
#     plot_LS_summary(time, y, freqs, periods, ls_pgram,
#                     vertical_indicators=[])


# def generic_lomb_scargle(time, y, freqs):
    
#     # time in seconds
#     # NaN rows removed
    


#     ls_pgram = signal.lombscargle(time, y, freqs, normalize=True)

#     return ls_pgram

# def plot_LS_summary(time, y, freqs, periods, ls_pgram,
#                     fontsize=15,
#                     vertical_indicators=[],
#                     pgram_fmt={'color': 'dimgrey', 'linewidth': 1.5},
#                     vertical_ind_col='royalblue'):
#                      # surrogate_period=None, surrogate_fft_amp=None,
#                      # fft_xlims=[0, 36],
#                      # signal_xlims=[np.nan, np.nan], signal_ymin=1.,
#                      # fontsize=15,
#                      # vertical_indicators=[],
#                      # unix_to_dtime=False,
#                      # resolution_lim=True,
#                      # signal_y_log=False,
#                      # input_fmt={'color': 'royalblue', 'linewidth': 1.},
#                      # ifft_fmt={'color': 'royalblue', 'linewidth': 1.},
#                      # input_ax=None, panel_label=True
    
#     # fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
#     # ax_t.plot(time, y, 'b+')
#     # ax_t.set_xlabel('Time [s]')

#     # ax_w.plot(freqs, ls_pgram)
#     # ax_w.set_xlabel('Angular frequency [rad/s]')
#     # ax_w.set_ylabel('Normalized amplitude')
#     # plt.show()
    

    
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     ax.plot(freqs, ls_pgram, **pgram_fmt)
#     ax.set_xscale('log')
    
#     # Formatting
#     ax.set_ylabel('Lomb-Scargle Normalised Amplitude', fontsize=fontsize)
#     ax.set_xlabel('Period (hours)', fontsize=fontsize)
#     ax.tick_params(labelsize=fontsize)
    
    
#     if vertical_indicators != []:
#         for h in vertical_indicators:
#             ax.axvline(h, color=vertical_ind_col, linestyle='dashed',
#                           linewidth=1.5)
#             trans = transforms.blended_transform_factory(ax.transData,
#                                                          ax.transAxes)
#             ax.text(h, 1.05, str(h), transform=trans,
#                        fontsize=fontsize, va='top', ha='center',
#                        color=vertical_ind_col)
            

# Functions to convert between period in hours
#   and frequency in Hz
# period = 1 / freq
# period = period / (60*60)   # period in hours
def period_to_freq(period):
    
    if len(period[period == 0]) > 0:
        print('ERROR periodicity_functions.period_to_freq')
        print('Input periods contains period == 0')
        print('Please rerun without entry where period == 0')
        raise ValueError('Input data contains 0(s)')

    freq = [1 / (p * 60. * 60.) for p in period]
    # freq = []
    # for p in period:
    #     freq.append(1. / (p * (60.*60.)))
    return np.array(freq)

def freq_to_period(freq):
    
    if len(freq[freq == 0]) > 0:
        print('ERROR periodicity_functions.freq_to_period')
        print('Input periods contains freq == 0')
        print('Please rerun without entry where freq == 0')
        raise ValueError('Input data contains 0(s)')

    period = [(1/f) / (60. * 60.) for f in freq]

    # period = []
    # for f in freq:
    #     period.append((1. / f) / (60.*60.))

    return np.array(period)    


# def DEPRECATED_test_acf():
#     y = np.sin(np.linspace(0, 11, 51))
#     temporal_resolution = pd.Timedelta(minutes=3)
#     n_shifts = 5
#     lags = np.array(range(n_shifts)) * temporal_resolution
#     shifted_y, acf = autocorrelation(y, n_shifts)

#     fig, ax = plot_autocorrelogram(lags, acf)