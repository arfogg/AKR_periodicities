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
from scipy.stats import linregress

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

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\qq_plot')
import qq_plot

#fontsize = 15
alphabet = list(string.ascii_lowercase)
axes_labels = []
for a in alphabet:
    axes_labels.append('(' + a + ')')

fig_dir = os.path.join("C:" + os.sep,
                       r"Users\Alexandra\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")


# Set up fontsizes
fontsize = 15
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize


def run_analyses():

    # correlate james power with current power
    # try 100-650 band
    # log intensity
    # 3 hr smoothing
    # constant gap fill (see james email)
    # do FFT
    print('hello')

    # (1) basic correlation mine vs James ipower
    # (2) comparison of mine vs James (correlation, timeseries) at each process stage (power, smoothed, logged)
    # (3) FFT on both
    # (4) LS on both


def compare_two_timeseries(
        n1='Name1', n2='Name2', c1='black', c2='palevioletred'):
    # x1, y1, x2, y2, n1='1', n2='2'):
    x1 = np.linspace(0, 100, 1000)
    x2 = np.linspace(0, 100, 1000)
    y1 = np.sin(x1)
    y2 = np.sin(x2) + np.random.normal(0., 200.*np.mean(y1), len(x2))

    fig = plt.figure(figsize=(12.5, 10))
    ax = [plt.subplot(221), plt.subplot(222), plt.subplot(212)]

    # Correlate
    ax[0].set_facecolor('lightgrey')
    h, xedges, yedges, im = ax[0].hist2d(y1, y2, bins=40, cmin=1, cmap='RdPu')
    # ax[0].plot(y1, y2, linewidth=0., marker='x', color='black')
    fig.colorbar(im, ax=ax[0], label='Counts')
    linear_fit = linregress(y1, y2)

    x_mod = np.linspace(np.min([y1, y2]), np.max([y1, y2]), 100)
    y_mod = (linear_fit.slope * x_mod) + linear_fit.intercept
    ax[0].plot(x_mod, y_mod, color='black', linewidth=1.5,
               label='y = %3.7sx + %3.7s\nr = %3.7s' % (
                   linear_fit.slope, linear_fit.intercept, linear_fit.rvalue))
    # ax[0].text(0.05, 0.95,
    #            'y = %3.7sx + %3.7s'%(linear_fit.slope, linear_fit.intercept),
    #            transform=ax[0].transAxes, ha='left', va='top')
    # breakpoint()
    ax[0].set_xlabel(n1)
    ax[0].set_ylabel(n2)
    ax[0].legend(loc='lower right')

    # breakpoint()
    # QQ Plot
    ax[1] = qq_plot.qq_plot(y1, y2, ax[1])
    ax[1].set_xlabel(n1)
    ax[1].set_ylabel(n2)

    # Timeseries
    ax[2].plot(x2, y2, label=n2, color=c2, linewidth=2., linestyle='dashed')
    ax[2].plot(x1, y1, label=n1, color=c1)

    ax[2].legend(loc='upper right')
    ax[2].set_xlabel('UT')
    ax[2].set_ylabel('Intensity')

    for (i, a) in enumerate(ax):
        t = a.text(0.05, 0.95, axes_labels[i], transform=a.transAxes,
                   fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    fig.tight_layout()
    # breakpoint()


def analog_waters_data():
    """
    Here we create an analog of the data from Waters+ 2021 paper for the
    Wind-Cassini conjunction.

    Returns
    -------
    smoothed_df : pd.DataFrame
        Contains data preprocessed as per Waters 2021.

    """

    interval_tag = 'cassini_flyby'
    # freq_column = 'ipwr_100_650kHz'

    # Read in interval data
    # interval_options = read_and_tidy_data.return_test_intervals()
    akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)

    akr_df.drop(columns=['ipwr_20_50kHz', 'ipwr_50_100kHz', 'ipwr_100_300kHz',
                         'ipwr_300_500kHz', 'ipwr_500_700kHz',
                         'ipwr_700_850kHz', 'ipwr_100_400kHz',
                         'ipwr_150_400kHz', 'ipwr_150_650kHz',
                         'ipwr_650_800kHz'], inplace=True)

    # We have no NaNs - needs investigating
    # Could be ok, since we are using a more up to date version of Waters data

    # Waters fig. 9 caption:
    # "Analysis is performed on the integrated powers after applying a 3-hr
    # rolling window and log-transforming the data"

    # 3 hour smoothing
    averaging_df = akr_df.copy(deep=True)
    averaging_df.set_index('datetime', inplace=True)
    smoothed = averaging_df['ipwr_100_650kHz'].rolling('3h').mean()
    smoothed_df = pd.DataFrame({'datetime': smoothed.index,
                                'ipwr_100_650kHz': smoothed.values})

    # Log power
    smoothed_df['log_pwr'] = np.log(smoothed_df.ipwr_100_650kHz)

    # well, anything with zero power is inf when logged.
    # not clear what the previous paper did to deal with these.
    # for now, we will put a zero in each place so zero stays zero
    zero_i, = np.where(smoothed_df.ipwr_100_650kHz == 0)
    smoothed_df.loc[zero_i, "log_pwr"] = 0

    return smoothed_df




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

    fft_png = os.path.join(fig_dir, "recreate_waters_FFT.png")

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


    hf_df = akr_df.copy()
    hf_df.drop(columns=['ipwr_20_50kHz', 'ipwr_50_100kHz', 'ipwr_100_300kHz',
           'ipwr_300_500kHz', 'ipwr_500_700kHz', 'ipwr_700_850kHz',
           'ipwr_100_400kHz','ipwr_150_400kHz',
    'ipwr_150_650kHz', 'ipwr_650_800kHz'], inplace=True)
    
    # aparently there are no NaNs
    
    # "averaged over a 3-hr window" but "input at a 3-min resolution"
    averaging_df = hf_df.copy(deep=True)
    averaging_df.set_index('datetime', inplace=True)
    smoothed = averaging_df['ipwr_100_650kHz'].rolling('3h').mean()
    smoothed_df = pd.DataFrame({'datetime': smoothed.index,
                                'ipwr_100_650kHz': smoothed.values})

    # "log-transforming the data"
    # natural log is the default
    smoothed_df['log_pwr'] = np.log(smoothed_df.ipwr_100_650kHz)

    # well, anything with zero power is inf when logged.
    # not clear what the previous paper did to deal with these.
    # for now, we will put a zero in each place so zero stays zero
    zero_i, = np.where(smoothed_df.ipwr_100_650kHz==0)
    smoothed_df.log_pwr.iloc[zero_i] = 0
    

    freq, period, fft_amp, inverse_signal = \
        periodicity_functions.generic_fft_function(smoothed_df.datetime,
                                                   smoothed_df.log_pwr,
                                                   pd.Timedelta(minutes=3))

    # "normalized the FFT output by the maximum spectral power"
    fft_norm = fft_amp / np.max(fft_amp)

    ax.plot(period, fft_norm)
    ax.set_xscale('log')
    ax.set_xlabel('Period (hours)', fontsize=fontsize)
    ax.set_ylabel('Relative Spectral Power', fontsize=fontsize)
    ax.axvline(24., linestyle='dotted', color='grey')
    ax.axvline(12., linestyle='dashed', color='grey')

