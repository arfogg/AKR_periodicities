# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:50:12 2024

@author: A R Fogg

Functions to run Lomb-Scargle analysis.
"""

import pathlib
import pickle
# import psutil

from astropy.timeseries import LombScargle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


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


def generic_lomb_scargle(time, y, f_min, f_max, n0=5):
    """
    Function to run Lomb Scargle as implemented in SciPy

    Parameters
    ----------
    time : np.array
        Time axis of the data in seconds.
    y : np.array
        Amplitude of the data. NaN rows must be removed.
    f_min : float
        Desired minimum frequency.
    f_max : float
        Desired maximum frequency.
    n0 : int, optional
        Number of samples needed to define a peak in a periodogram. The
        default is 5.

    Returns
    -------
    ls_object : object
        Astropy LombScargle object containing Lomb Scargle results.
    freqs : np.array
        Frequencies along the periodogram axis.
    ls_pgram : np.array
        Normalised Lomb-Scargle amplitude as a function of freqs.

    """

    # Run Lomb Scargle analysis, creating LombScargle object
    ls_object = LombScargle(time, y, normalization='standard')

    # Extract frequencies and power
    # breakpoint()
    freqs, ls_pgram = ls_object.autopower(
        minimum_frequency=f_min, maximum_frequency=f_max, samples_per_peak=n0)

    return ls_object, freqs, ls_pgram


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


def compute_lomb_scargle_peak(time, signal, f_min, f_max, i,
                              directory, keyword, n0=5):
    """
    Compute the peak of a LS periodogram and save to file.

    Parameters
    ----------
    time : np.array
        Time for timeseries to be analysed.
    signal : np.array
        Amplitude for timeseries to be analysed.
    f_min : float
        Desired minimum frequency.
    f_max : float
        Desired maximum frequency.
    i : int
        Bootstrap number.
    directory : string
        Directory for file to be saved.
    keyword : string
        Keyword to put into file.
    n0 : int, optional
        Number of samples needed to define a peak in a periodogram. The
        default is 5.

    Returns
    -------
    peak_magnitude : float
        Magnitude of LS peak.

    """

    # Output filename for peak magnitude
    fname = f"{directory}/{keyword}_LS_peak_bootstrap_{i}.csv"

    # Ensure inputs are in the right type
    time = np.asarray(time)
    signal = np.asarray(signal)

    # Already saved, load in
    if pathlib.Path(fname).is_file():
        peak_magnitude_df = pd.read_csv(fname, float_precision="round_trip")
        peak_magnitude = peak_magnitude_df['Peak_Magnitude'].values[0]
    # Else, calculate
    else:
        ls_object, freqs, ls_pgram = generic_lomb_scargle(time, signal,
                                                          f_min, f_max, n0=n0)
        peak_magnitude = np.nanmax(ls_pgram)
        # Save each iteration separately
        pd.DataFrame({'Bootstrap_Index': [i],
                      'Peak_Magnitude': [peak_magnitude]}
                     ).to_csv(fname, index=False)

    return peak_magnitude


def false_alarm_probability(n_bootstrap, BS_signal, time, f_min, f_max,
                            FAP_peak_directory, FAP_peak_keyword, FAP_fname,
                            n0=5):
    """
    Calculate the false alarm probability on a LS periodogram using
    bootstrapped signals.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstraps.
    BS_signal : np.ndarray
        Of shape len(time) x n_bootstrap. Contains the bootstrapped signal.
    time : np.array
        Time axis for LS peridogram.
    f_min : float
        Desired minimum frequency.
    f_max : float
        Desired maximum frequency.
    FAP_peak_directory : string
        Directory to save FAP peaks in.
    FAP_peak_keyword : string
        Keyword to put into FAP peak filenames.
    FAP_fname : string
        Filename to save overall FAP to.
    n0 : int, optional
        Number of samples needed to define a peak in a periodogram. The
        default is 5.

    Returns
    -------
    bootstrap_peak_magnitudes : TYPE
        DESCRIPTION.
    FAP : TYPE
        DESCRIPTION.

    """

    # FAP already calculated, loading
    if pathlib.Path(FAP_fname).is_file():
        with open(FAP_fname, 'rb') as f:
            bootstrap_peak_magnitudes = pickle.load(f)
            FAP = pickle.load(f)

    # Otherwise, calculate
    else:
        bootstrap_peak_magnitudes = np.full(n_bootstrap, np.nan)

        for i in range(n_bootstrap):
            bootstrap_peak_magnitudes[i] = compute_lomb_scargle_peak(
                time, BS_signal[:, i], f_min, f_max, i, FAP_peak_directory,
                FAP_peak_keyword, n0=n0)

        # Compute FAP, 97.7th percentile / 2 sigma
        # FAP = np.nanmean(bootstrap_peak_magnitudes)
        FAP = np.percentile(bootstrap_peak_magnitudes, 97.7)
        # Save FAP to file
        with open(FAP_fname, 'wb') as f:
            pickle.dump(bootstrap_peak_magnitudes, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(FAP, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bootstrap_peak_magnitudes, FAP
