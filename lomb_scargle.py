# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:50:12 2024

@author: A R Fogg

Functions to run Lomb-Scargle analysis.
"""

import pathlib
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import scipy.signal as signal

from joblib import Parallel, delayed


def define_frequency_bins(T, f_min, f_max, n0=5):
    """
    Function to define the Lomb-Scargle bins in frequency space. Theory
    of bin selection from Vanderplas (2018)
    https://doi.org/10.3847/1538-4365/aab766

    Parameters
    ----------
    T : int
        Length of entire dataset in seconds. The lowest possible freq
        is 1/total amount of time observed.
    f_min : float
        Desired minimum frequency.
    f_max : float
        Desired maximum frequency.
    n0 : TYPE, optional
        Number of samples needed to define a peak in a periodogram. The
        default is 5.

    Returns
    -------
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    N_f : int
        Number of frequencies.
    sample_f : np.array
        Calculated frequency bins.

    """

    # Number of frequencies to be sampled
    N_f = int(n0 * T * f_max)

    # Distribute frequencies logarithmically
    sample_f = np.logspace(np.log10(f_min), np.log10(f_max), N_f)

    return f_min, f_max, N_f, sample_f


def generic_lomb_scargle(time, y, freqs):
    """
    Function to run Lomb Scargle as implemented in SciPy

    Parameters
    ----------
    time : np.array
        Time axis of the data in seconds.
    y : np.array
        Amplitude of the data. NaN rows must be removed.
    freqs : np.array
        Frequency bins to be sampled.

    Returns
    -------
    ls_pgram : np.array
        Normalised Lomb-Scargle amplitude as a function of freqs.

    """

    ls_pgram = signal.lombscargle(time, y, freqs, normalize=True)

    return ls_pgram


def plot_LS_summary(periods, ls_pgram,
                    fontsize=15,
                    vertical_indicators=[],
                    pgram_fmt={'color': 'dimgrey', 'linewidth': 1.5},
                    vertical_ind_col='royalblue',
                    ax=None):
    """
    Function to plot Lomb-Scargle periodogram.

    Parameters
    ----------
    periods : np.array
        Periods for the x axis.
    ls_pgram : np.array
        Lomb-scargle amplitude for the y axis.
    fontsize : float, optional
        Fontsize for all text. The default is 15.
    vertical_indicators : list, optional
        X axis positions (in hours) to draw vertical arrows at. The
        default is [].
    pgram_fmt : dict, optional
        Formatting options for the periodogram curve. The default is
        {'color': 'dimgrey', 'linewidth': 1.5}.
    vertical_ind_col : string, optional
        Color for the vertical indicators. The default is 'royalblue'.
    ax : matplotlib axis, optional
        Axis to draw the plot on. The default is None.

    Returns
    -------
    ax : matplotlib axis
        Axis object containing the plot.

    """

    # Define plotting window if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot periodogram
    ax.plot(periods, ls_pgram, **pgram_fmt)

    # Formatting
    ax.set_xscale('log')
    ax.set_ylabel('Lomb-Scargle\nNormalised Amplitude', fontsize=fontsize)
    ax.set_xlabel('Period (hours)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    # Draw vertical indicators
    if vertical_indicators != []:
        for h in vertical_indicators:
            trans = transforms.blended_transform_factory(ax.transData,
                                                         ax.transAxes)
            ax.annotate(str(h), xy=(h, 1.0), xytext=(h, 1.15),
                        xycoords=trans, arrowprops={'facecolor': 'black'},
                        fontsize=fontsize, va='top', ha='center',
                        color=vertical_ind_col)

    return ax


def compute_lomb_scargle_peak(time, signal, freqs, i, directory, keyword):
    """ Compute Lomb-Scargle and save the result to a unique file. """
    
    fname = f"{directory}/{keyword}_LS_peak_bootstrap_{i}.csv"    
    
    
    #def compute_lomb_scargle_peak(time, signal, freqs, i, directory, keyword):
    print(f"Computing for bootstrap {i}...")
    print(f"Time type: {type(time)}, Signal type: {type(signal)}, Freqs type: {type(freqs)}")
    #breakpoint()
    # Rest of your function...
    time = np.asarray(time)
    signal = np.asarray(signal)
    
    
    if pathlib.Path(fname).is_file():
        peak_magnitude_df = pd.read_csv(fname, float_precision="round_trip")
        peak_magnitude = peak_magnitude_df['Peak_Magnitude'].values[0]
    else:
        peak_magnitude = np.nanmax(generic_lomb_scargle(time, signal, freqs))
        # Save each iteration separately
        pd.DataFrame({'Bootstrap_Index': [i],
                      'Peak_Magnitude': [peak_magnitude]}
                      ).to_csv(fname, index=False)

    return peak_magnitude

def false_alarm_probability(n_bootstrap, BS_signal, time, freqs,
                            FAP_peak_directory, FAP_peak_keyword, FAP_fname):
    #breakpoint()
    # bootstrap_peak_magnitudes = np.full(n_bootstrap, np.nan)
    # for i in range(n_bootstrap):
        
    #     pgram = generic_lomb_scargle(time, BS_signal[:, i], freqs)
    #     bootstrap_peak_magnitudes[i] = np.nanmax(pgram)

    print(generic_lomb_scargle)
    print(type(time))
    print(type(freqs))

    if pathlib.Path(FAP_fname).is_file():
        with open(FAP_fname, 'rb') as f:
            bootstrap_peak_magnitudes = pickle.load(f)
            FAP = pickle.load(f)

    else:
        # Run compute_lomb_scargle_peak in parallel
        bootstrap_peak_magnitudes = Parallel(n_jobs=-2)(
            delayed(compute_lomb_scargle_peak)(time, BS_signal[:, i].copy(), freqs, i,
                                               FAP_peak_directory,
                                               FAP_peak_keyword
                                               ) for i in range(n_bootstrap)
        )
        bootstrap_peak_magnitudes = np.array(bootstrap_peak_magnitudes) 
        # Compute FAP
        FAP = np.nanmean(bootstrap_peak_magnitudes)

        with open(FAP_fname, 'wb') as f:
            pickle.dump(bootstrap_peak_magnitudes, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(FAP, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    return bootstrap_peak_magnitudes, FAP

def DEPRECATED_detect_peak(ls_pgram, periods, freqs):

    i = np.argmax(ls_pgram)

    peak_height = ls_pgram[i]
    peak_freq = freqs[i]
    peak_period = periods[i]

    return peak_height, peak_freq, peak_period
