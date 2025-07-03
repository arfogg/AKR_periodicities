# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:50:12 2024

@author: A R Fogg

Functions to run Lomb-Scargle analysis.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pathlib
import pickle
import psutil
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import scipy.signal as signal

from joblib import Parallel, delayed, parallel_backend


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

def generic_lomb_scargle_chunked(time, y, freqs, chunk_size=200):
    ls_pgram_full = []
    j=0
    for start in range(0, len(freqs), chunk_size):
        print('chunk ', j)
        chunk_freqs = freqs[start:start+chunk_size]
        ls_chunk = signal.lombscargle(time, y, chunk_freqs, normalize=True)
        ls_pgram_full.append(ls_chunk)
        j=j+1
    return np.concatenate(ls_pgram_full)


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
    """
    Compute the peak of a LS periodogram and save to file.

    Parameters
    ----------
    time : np.array
        Time for timeseries to be analysed.
    signal : np.array
        Amplitude for timeseries to be analysed.
    freqs : np.array
        Frequency bins to be assessed.
    i : int
        Bootstrap number.
    directory : string
        Directory for file to be saved.
    keyword : string
        Keyword to put into file.

    Returns
    -------
    peak_magnitude : float
        Magnitude of LS peak.

    """

    fname = f"{directory}/{keyword}_LS_peak_bootstrap_{i}.csv"

    print(f"Computing for bootstrap {i}...")
    print(f"Time type: {type(time)}, Signal type: {type(signal)}, Freqs type: {type(freqs)}")

    # Ensure inputs are in the right type
    time = np.asarray(time)
    signal = np.asarray(signal)

    # Already saved, load in
    if pathlib.Path(fname).is_file():
        peak_magnitude_df = pd.read_csv(fname, float_precision="round_trip")
        peak_magnitude = peak_magnitude_df['Peak_Magnitude'].values[0]
    # Else, calculate
    else:
        #peak_magnitude = np.nanmax(generic_lomb_scargle(time, signal, freqs))
        peak_magnitude = np.nanmax(generic_lomb_scargle_chunked(time, signal, freqs))
        # Save each iteration separately
        pd.DataFrame({'Bootstrap_Index': [i],
                      'Peak_Magnitude': [peak_magnitude]}
                     ).to_csv(fname, index=False)

    return peak_magnitude



def calc_safe_njobs(est_mem_per_job_gb, frac_safe=0.5):
    """
    Dynamically scale n_jobs based on available memory to prevent crashes.

    Parameters
    ----------
    task_list : list
        List of arguments (as tuples) to pass to the function.
    func : callable
        Function to be called with arguments from task_list.
    est_mem_per_job_gb : float
        Estimated memory use per job, in gigabytes.
    backend : str
        Joblib backend. Default is 'loky' (process-based). Use 'threading' if needed.

    Returns
    -------
    results : list
        List of function results.
    """
    # Get available memory in GB
    available_mem_gb = psutil.virtual_memory().available / 1e9
    print(available_mem_gb, ' GB memory available')

    # Leave some safety buffer (e.g., 30% of available RAM)
    safe_mem_gb = available_mem_gb * frac_safe
    print('Using only ',frac_safe*100., '%, i.e.:', safe_mem_gb, ' GB memory')

    # Determine max safe number of parallel jobs
    # Double / rounds down
    max_safe_jobs = max(1, int(safe_mem_gb // est_mem_per_job_gb))

    print(f"Using up to {max_safe_jobs} parallel jobs")

    return max_safe_jobs


def estimate_lombscargle_memory_gb(time, freqs, dtype=np.float64):
    """
    Estimate memory usage in gigabytes for one Lomb-Scargle job.

    Parameters
    ----------
    time : np.ndarray
        Time array (1D).
    freqs : np.ndarray
        Frequency bins (1D).
    dtype : numpy dtype, optional
        Data type of array elements. Default is float64.

    Returns
    -------
    mem_gb : float
        Estimated memory usage in gigabytes.
    """
    n_time = len(time)
    n_freqs = len(freqs)
    bytes_per_element = np.dtype(dtype).itemsize

    # Estimate memory for internal matrix: shape (n_time, n_freqs)
    total_bytes = n_time * n_freqs * bytes_per_element
    mem_gb = total_bytes / 1e9

    return mem_gb


def false_alarm_probability(n_bootstrap, BS_signal, time, freqs,
                            FAP_peak_directory, FAP_peak_keyword, FAP_fname):
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
    freqs : np.array
        Frequency bins to be assessed.
    FAP_peak_directory : string
        Directory to save FAP peaks in.
    FAP_peak_keyword : string
        Keyword to put into FAP peak filenames.
    FAP_fname : string
        Filename to save overall FAP to.

    Returns
    -------
    bootstrap_peak_magnitudes : TYPE
        DESCRIPTION.
    FAP : TYPE
        DESCRIPTION.

    """

    # print(generic_lomb_scargle)
    # print(type(time))
    # print(type(freqs))
    # print("DEBUG TYPES:")
    # print("  time:", type(time))
    # print("  signal:", type(BS_signal[:, 0]))
    # print("  freqs:", type(freqs))

    # FAP already calculated, loading
    if pathlib.Path(FAP_fname).is_file():
        with open(FAP_fname, 'rb') as f:
            bootstrap_peak_magnitudes = pickle.load(f)
            FAP = pickle.load(f)

    # Otherwise, calculate
    else:
        print(f"time size: {len(time)}")
        print(f"freqs size: ", freqs.size)
        print(f"Estimated memory for lombscargle matrix (GB): {(len(time) * len(freqs) * 8) / 1e9:.2f}")
        print(f"Estimated memory for lombscargle matrix (GB): {(len(time) * 100 * 8) / 1e9:.2f}")

        est_mem_per_job_gb = estimate_lombscargle_memory_gb(time, freqs[0:199], dtype=np.float64)
        max_safe_jobs = calc_safe_njobs(est_mem_per_job_gb)


        for i in range(n_bootstrap):
            print('BS ', i)
            bootstrap_peak_magnitudes[i] = compute_lomb_scargle_peak(
                time, BS_signal[:, i], freqs, i, FAP_peak_directory,
                FAP_peak_keyword)
            # print(f"Bootstrap {i}: peak = {peak}")


        #breakpoint()
        # Run individual LS calculations in parallel.
        # n_jobs=-3 uses all but 2 available processors
        # with parallel_backend("threading"):
        # with parallel_backend("loky"):
        #     # Run compute_lomb_scargle_peak in parallel
        #     bootstrap_peak_magnitudes = Parallel(n_jobs=max_safe_jobs)(
        #         delayed(compute_lomb_scargle_peak)(time,
        #                                            BS_signal[:, i].copy(),
        #                                            freqs, i,
        #                                            FAP_peak_directory,
        #                                            FAP_peak_keyword
        #                                            ) for i in range(
        #                                                n_bootstrap)
        #                                                )
        bootstrap_peak_magnitudes = np.array(bootstrap_peak_magnitudes)

        # Compute FAP
        FAP = np.nanmean(bootstrap_peak_magnitudes)
        # Save FAP to file
        with open(FAP_fname, 'wb') as f:
            pickle.dump(bootstrap_peak_magnitudes, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(FAP, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bootstrap_peak_magnitudes, FAP

def normalise_bootstrapped_LS_peaks(n_bootstrap, BS_signal,
                                    bootstrap_peak_magnitudes,
                                    FAP_peak_directory, FAP_peak_keyword,
                                    FAP_fname):
    # Ideally, if we rerun we'll do this automatically in the scipy
    # lomb scargle function, but this is a quick fix as rerunning is
    # very computationally intensive.

    # Initialise array
    norm_bs_peak_magnitude = np.full()
    # Loop through, normalising
    for i in range(n_bootstrap):
        #power = lombscargle(t, y, angular_freqs, precenter=True)
        norm_bs_peak_magnitude[i] /= 0.5 * np.var(BS_signal[:, i])


    # then save to a dir
# def DEPRECATED_detect_peak(ls_pgram, periods, freqs):

#     i = np.argmax(ls_pgram)

#     peak_height = ls_pgram[i]
#     peak_freq = freqs[i]
#     peak_period = periods[i]

#     return peak_height, peak_freq, peak_period
