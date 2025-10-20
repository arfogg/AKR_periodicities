# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:01:45 2025

@author: A R Fogg
"""

import os
import sys
import string
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import matplotlib.transforms as transforms

import periodicity_functions
import read_and_tidy_data
import binning_averaging
import wind_location
import diurnal_oscillator
import lomb_scargle
import autocorrelation
import bootstrap_functions
import utility
import main_plotting

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\wind_utility')
import read_wind_position
#import read_sunspot_n

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\readers')
import read_omni
import read_supermag
import read_sunspot_n

fontsize = 15
alphabet = list(string.ascii_lowercase)
axes_labels = []
for a in alphabet:
    axes_labels.append('(' + a + ')')

fig_dir = os.path.join("C:" + os.sep,
                       r"Users\Alexandra\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")


def repeat_waters2021_FFT():

    # Make the data evenly sampled to 3 hour points
    
    # Log transform the data
    
    # FFT
    
    # Normalise the FFT
    
    
    
    # Initialising variables
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    samples_per_peak = 5

    vertical_indicators = [12, 24]
    vertical_ind_col = 'black'

    annotate_bbox = {"facecolor": "white", "edgecolor": "grey", "pad": 5.}

    # Different frequency channels
    freq_column = 'ipwr_100_400kHz'
    n = '100-400 kHz'
    c = 'dimgrey'
    interval_tag = 'cassini_flyby'

    LS_fig = os.path.join(fig_dir, "recreate_waters_FFT.png")

    # Read in interval data
    interval_options = read_and_tidy_data.return_test_intervals()

    # Initialise plotting window
    fig, ax = plt.subplots(figsize=(12.5, 5))

    print('Running FFT for ', interval_tag)

    # base_dir = pathlib.Path(data_dir) / 'waters_FFT'
    # file_paths = [
    #     base_dir / f"LS_{interval_tag}_{f}.csv" for f in [freq_column]]
    # file_checks = [file_path.is_file() for file_path in file_paths]

    # if all(file_checks) is False:
    akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)

    breakpoint()

    

    # print('Frequency band: ', freq_column)
    # fft_csv = os.path.join(data_dir, 'lomb_scargle', 'LS_' +
    #                       interval_tag + '_' + freq_column + '.csv')
    # if pathlib.Path(fft_csv).is_file() is False:
    #     freq_df = akr_df.dropna(subset=[freq_column])
    #     t1 = pd.Timestamp.now()
    #     print('starting LS at ', t1)
    #     ls_object, freqs, ls_pgram = lomb_scargle.generic_lomb_scargle(
    #         freq_df.unix, freq_df[freq_column], f_min, f_max,
    #         n0=samples_per_peak)
    #     t2 = pd.Timestamp.now()
    #     print('LS finished, time elapsed: ', t2-t1)
    #     # Write to file
    #     periods = periodicity_functions.freq_to_period(freqs)
    #     ls_df = pd.DataFrame({'period_hr': periods,
    #                           'angular_freq': freqs,
    #                           'ls_pgram': ls_pgram})
    #     t2 = pd.Timestamp.now()
    #     print('LS finished, time elapsed: ', t2-t1)
    #     # Write to file
    #     ls_df.to_csv(fft_csv, index=False)

    # else:
    #     ls_df = pd.read_csv(ls_csv, delimiter=',',
    #                         float_precision='round_trip')

    #     ls_pgram = np.array(ls_df.ls_pgram)
    #     periods = np.array(ls_df.period_hr)

    # # Plot FAP here
    # FAP_pkl = os.path.join(
    #     data_dir, "lomb_scargle",
    #     interval_tag + '_' + freq_column + "_FAP_" + str(n_bootstrap)
    #     + "_BSs.pkl")
    # # Read in bootstrap
    # ftime_cl, BS = main_plotting.read_subset_bootstraps(interval_tag,
    #                                       freq_ch=freq_column,
    #                                       n_bootstrap=n_bootstrap)
    # # Convert ftime_cl to unix
    # ftime_unix = [pd.Timestamp(t).timestamp() for t in ftime_cl]

    # # Read in/calc peak magnitudes for bootstraps and FAP
    # bootstrap_peak_magnitudes, FAP = lomb_scargle.false_alarm_probability(
    #     n_bootstrap, BS, ftime_unix, f_min, f_max, FAP_peaks_dir,
    #     interval_tag + '_' + freq_column, FAP_pkl, n0=samples_per_peak)

    # ax[i + 1].plot(periods, ls_pgram, linewidth=1.5, color=c, label=n)
    # if j == 0:
    #     trans = transforms.blended_transform_factory(
    #         ax[i + 1].transAxes, ax[i + 1].transData)
    #     ax[i + 1].annotate("FAL\n" + "{:.3e}".format(FAP),
    #                        xy=(0.2, FAP), xytext=(0.1, FAP),
    #                        xycoords=trans, arrowprops={'facecolor': c},
    #                        fontsize=fontsize, va='center', ha='right',
    #                        color=c, bbox=annotate_bbox,
    #                        fontweight="bold")
    # elif j == 1:
    #     trans = transforms.blended_transform_factory(
    #         ax[i + 1].transAxes, ax[i + 1].transData)
    #     ax[i + 1].annotate("FAL\n" + "{:.3e}".format(FAP),
    #                        xy=(0.8, FAP), xytext=(0.9, FAP),
    #                        xycoords=trans, arrowprops={'facecolor': c},
    #                        fontsize=fontsize, va='center', ha='left',
    #                        color=c, bbox=annotate_bbox, fontweight="bold")

    # ax[i + 1].set_xscale('log')

    # # Formatting
    # ax[i + 1].set_ylabel('Lomb-Scargle\nNormalised Amplitude',
    #                      fontsize=fontsize)
    # ax[i + 1].set_xlabel('Period (hours)', fontsize=fontsize)
    # ax[i + 1].tick_params(labelsize=fontsize)
    # ax[i + 1].legend(fontsize=fontsize, loc='upper left')
    # if vertical_indicators != []:
    #     for h in vertical_indicators:
    #         trans = transforms.blended_transform_factory(
    #             ax[i + 1].transData, ax[i + 1].transAxes)
    #         ax[i + 1].annotate(str(h), xy=(h, 1.0), xytext=(h, 1.15),
    #                            xycoords=trans,
    #                            arrowprops={'facecolor': 'black'},
    #                            fontsize=fontsize, va='top', ha='center',
    #                            color=vertical_ind_col)

    # # Label panels
    # titles = np.append('Synthetic', interval_options.label)
    # for (i, a) in enumerate(ax):
    #     t = a.text(0.005, 1.05, axes_labels[i], transform=a.transAxes,
    #                fontsize=fontsize, va='bottom', ha='left')
    #     t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    #     tit = a.text(1.0, 1.05, titles[i], transform=a.transAxes,
    #                  fontsize=1.25 * fontsize, va='center', ha='right')

    # # Adjust margins etc
    # fig.tight_layout()
    # # Save to file
    # # fig.savefig(LS_fig)
