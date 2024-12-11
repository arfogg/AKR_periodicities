# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:50:12 2024

@author: A R Fogg

Functions to run Lomb-Scargle analysis.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import scipy.signal as signal



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

def define_frequency_bins(T, f_min, f_max, n0=5):
    
    # T = length of entire dataset in seconds. i.e. the lowest
    #   possible freq is 1/total amount of time observed
    # f_max = desired maximum frequency, decided based on prior science
    #   knowledge of the system
    # n0 = number of samples needed to define a peak in a periodogram
    #
    # Vanderplas 2018 https://doi.org/10.3847/1538-4365/aab766

    print(f_min < f_max)
    
    # Number of frequencies to be sampled
    N_f = int(n0 * T * f_max)
    
    # Distribute frequencies logarithmically
    sample_f = np.logspace(np.log10(f_min), np.log10(f_max), N_f)
    
    return f_min, f_max, N_f, sample_f

def generic_lomb_scargle(time, y, freqs):
    
    # time in seconds
    # NaN rows removed
    


    ls_pgram = signal.lombscargle(time, y, freqs, normalize=True)

    return ls_pgram

def detect_peak(ls_pgram, periods, freqs):
    
    
    
    
    i = np.argmax(ls_pgram)
    
    peak_height = ls_pgram[i]
    peak_freq = freqs[i]
    peak_period = periods[i]
    
    
    
    # calc FAP here????
    # not sure this is the best wway to calc the peak!
    
    
    
    return peak_height, peak_freq, peak_period

def plot_LS_summary(periods, ls_pgram,
                    fontsize=15,
                    vertical_indicators=[],
                    pgram_fmt={'color': 'dimgrey', 'linewidth': 1.5},
                    vertical_ind_col='royalblue',
                    ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(periods, ls_pgram, **pgram_fmt)
    # lims=ax.get_ylim()
    # ax.plot(periods, np.repeat((lims[1]-lims[0])/2, periods.size),
    #         linewidth=0., marker='^', alpha=0.5, color='red',
    #         fillstyle='none')
    ax.set_xscale('log')

    # Formatting
    ax.set_ylabel('Lomb-Scargle\nNormalised Amplitude', fontsize=fontsize)
    ax.set_xlabel('Period (hours)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    if vertical_indicators != []:
        for h in vertical_indicators:
            ax.axvline(h, color=vertical_ind_col, linestyle='dashed',
                       linewidth=1.5)
            trans = transforms.blended_transform_factory(ax.transData,
                                                         ax.transAxes)
            ax.text(h, 1.075, str(h), transform=trans,
                    fontsize=fontsize, va='top', ha='center',
                    color=vertical_ind_col)

    return ax