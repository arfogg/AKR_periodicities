# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:47:27 2024

@author: A R Fogg
"""

#import read_supermag
#import read_omni
#import read_wind_position
#import read_integrated_power
import sys
import os
#import pathlib
import string

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import scipy.signal as signal
import matplotlib.transforms as transforms
import matplotlib as mpl
#from numpy.fft import fft, ifft

#from neurodsp.sim import sim_oscillation

#import fastgoertzel as G

import periodicity_functions
import feature_importance
import read_and_tidy_data
import binning_averaging
import wind_location
import main_plotting

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\wind_utility')

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\readers')


fontsize = 15
alphabet = list(string.ascii_lowercase)
axes_labels = []
for a in alphabet:
    axes_labels.append('(' + a + ')')

fig_dir = os.path.join("C:" + os.sep,
                       r"Users\admin\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")


def fft_multiple_freqs(temporal_resolution=pd.Timedelta(minutes=3),
                       fft_xlims=[0, 36], resolution_lim=True,
                       vertical_indicators=[12, 24], fontsize=15):

    freq_bands = read_and_tidy_data.define_freq_bands()

    cmap = mpl.colormaps['Set3']

    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, freq_bands.shape[0]))

    # Read in data
    # !!!!!!!!!!!!!

    # Initialise plotting window
    fig, ax = plt.subplots(ncols=2, figsize=(15, 7))

    # Fake signal for now
    time, akr_osc = main_plotting.oscillating_signal(24)
    freq, period, fft_amp, inverse_signal = periodicity_functions.\
        generic_fft_function(time, akr_osc, pd.Timedelta(minutes=3))

    if resolution_lim:
        j, = np.where(period > (2. * (temporal_resolution.total_seconds() /
                                      (60*60))))
        freq = freq[j]
        period = period[j]
        fft_amp = fft_amp[j]

    period_a_max = np.full(freq_bands.shape[0], np.nan)
    freq_str = np.full(freq_bands.shape[0], 'xxxxxxxxxxxxxxxx')
    for i in range(freq_bands.shape[0]):
        # Define name for this Frequency Band
        freq_str[i] = str(str(freq_bands[i, 0]) + '-' + str(freq_bands[i, 1]))

        # Run FFT here
        print('FFT for ' + freq_str[i])

        # Plot FFT periodogram
        ax[0].plot(period, fft_amp, label=freq_str[i], color=colors[i])

        period_a_max[i] = period[np.argmax(fft_amp)]

    # Plot vertical lines at desired periods
    if vertical_indicators != []:
        for h in vertical_indicators:
            ax[0].axvline(h, color='black', linestyle='dashed',
                          linewidth=1.)
            trans = transforms.blended_transform_factory(ax[0].transData,
                                                         ax[0].transAxes)
            ax[0].text(h, 1.05, str(h), transform=trans,
                       fontsize=fontsize, va='top', ha='center',
                       color='black')
    # Formatting
    ax[0].tick_params(labelsize=fontsize)
    ax[0].set_xlim(fft_xlims)
    ax[0].legend(fontsize=fontsize)
    ax[0].set_xlabel('Period (hours)', fontsize=fontsize)
    ax[0].set_ylabel('FFT amplitude', fontsize=fontsize)
    t = ax[0].text(0.05, 0.95, axes_labels[0], transform=ax[0].transAxes,
                   fontsize=fontsize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    # Bar chart
    rects = ax[1].bar(freq_str, period_a_max, label=freq_str, color=colors)

    # Label bar tops
    for j, (f, p, r) in enumerate(zip(freq_str, period_a_max, rects)):
        ax[1].annotate(str("%.2f" % p),
                       xy=(r.get_x() + r.get_width() / 2, p * 0.98),
                       xytext=(0, 2), textcoords='offset points',
                       rotation=-90, ha='center', va='top', fontsize=fontsize)

    # Formatting
    ax[1].tick_params(axis='x', labelrotation=-90)
    ax[1].tick_params(labelsize=fontsize)
    ax[1].set_xlabel('Frequency band (kHz)', fontsize=fontsize)
    ax[1].set_ylabel('Period at peak FFT amplitude\n(hours)',
                     fontsize=fontsize)
    t = ax[1].text(0.05, 0.95, axes_labels[1], transform=ax[1].transAxes,
                   fontsize=fontsize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    fig.tight_layout()
