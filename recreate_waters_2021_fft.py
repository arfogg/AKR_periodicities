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
import read_waters_masked_data
#import read_sunspot_n

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\readers')
import read_omni
import read_supermag
import read_sunspot_n

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\qq_plot')
import qq_plot

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\wind_waves_pipeline')
import spectrogram_plotter


fontsize = 15
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

    # Read in both
    waters_df = read_waters_data()
    fogg_df = analog_waters_data()

    
    # breakpoint()
    # Check the times align
    # Times are different beyond ~5/6th decimal place (decimal seconds)
    # This is presumably due to us using different versions of the masked data
    # Hence, round all datetimes to nearest second
    waters_df['datetime_rounded'] = waters_df.datetime_ut.dt.round('s')
    fogg_df['datetime_rounded'] = fogg_df.datetime.dt.round('s')
    
    # Tidy up for merging
    waters_df['waters_ipwr_100_650kHz'] = waters_df['P_Wsr-1_100_650_kHz']
    waters_df.drop(columns=['daily_min_index', 'GSE_LAT', 'GSE_LCT_T',
                            'P_Wsr-1_100_650_kHz',
                            'P_Wsr-1_30_100_kHz', 'P_Wsr-1_30_650_kHz',
                            'RADIUS', 'SM_LAT', 'datetime_ut',
                            'unix', 'date_i'], inplace=True)





    # Merge DataFrames
    merged_ipwr_df = pd.merge(fogg_df, waters_df,
                              on='datetime_rounded', how='inner')

    # Record where zeros exist in both datasets
    waters_zero_ind = np.array(merged_ipwr_df.loc[
        merged_ipwr_df.waters_ipwr_100_650kHz == 0.0].index)
    fogg_zero_ind = np.array(merged_ipwr_df.loc[
        merged_ipwr_df.ipwr_100_650kHz == 0.0].index)
    waters_zero_dt = merged_ipwr_df.datetime_rounded.iloc[waters_zero_ind]
    fogg_zero_dt = merged_ipwr_df.datetime_rounded.iloc[fogg_zero_ind]

    # Replace zeros with NaN
    merged_ipwr_df.loc[
        merged_ipwr_df['waters_ipwr_100_650kHz'] == 0.0,
        "waters_ipwr_100_650kHz"] = np.nan
    merged_ipwr_df.loc[
        merged_ipwr_df['ipwr_100_650kHz'] == 0.0,
        "ipwr_100_650kHz"] = np.nan
    


    ipwr_max = np.nanmax([np.array(merged_ipwr_df.waters_ipwr_100_650kHz),
                          np.array(merged_ipwr_df.ipwr_100_650kHz)])
    ipwr_min = np.nanmin([np.array(merged_ipwr_df.waters_ipwr_100_650kHz),
                          np.array(merged_ipwr_df.ipwr_100_650kHz)])




    hist_bins = np.logspace(np.log10(ipwr_min), np.log10(ipwr_max), 50)
    #breakpoint()




    # (1) basic correlation mine vs James ipower
    compare_two_timeseries(merged_ipwr_df.datetime_rounded,
                           merged_ipwr_df.ipwr_100_650kHz,
                           merged_ipwr_df.datetime_rounded,
                           merged_ipwr_df.waters_ipwr_100_650kHz,
                           n1='Fogg', n2='Waters',
                           title='Wind-Cassini Conjunction, comparing Waters/Fogg data\nAll data (excluding zeros)',
                           hist_bins=hist_bins)
    # breakpoint()
    
    # Where are the zeros?
    where_are_the_zeros(fogg_zero_dt, waters_zero_dt, n1='Fogg', n2='Waters')

    # Check the spectrogram over some zoom ins
    detailed_dive_spectrogram(pd.Timestamp(1999, 8, 17, 00),
                              pd.Timestamp(1999, 8, 20, 0),
                              merged_ipwr_df, 8,
                              vlines=[pd.Timestamp(1999, 8, 18, 00),
                                      pd.Timestamp(1999, 8, 19, 00)])

    detailed_dive_spectrogram(pd.Timestamp(1999, 8, 24, 00),
                              pd.Timestamp(1999, 8, 29, 0),
                              merged_ipwr_df, 8,
                              vlines=[pd.Timestamp(1999, 8, 25, 00),
                                      pd.Timestamp(1999, 8, 26, 00),
                                      pd.Timestamp(1999, 8, 27, 00),
                                      pd.Timestamp(1999, 8, 28, 00)])

    detailed_dive_spectrogram(pd.Timestamp(1999, 9, 7, 00),
                              pd.Timestamp(1999, 9, 10, 0),
                              merged_ipwr_df, 9,
                              vlines=[pd.Timestamp(1999, 9, 8, 00),
                                      pd.Timestamp(1999, 9, 9, 00)])


    # return waters_zero_dt, fogg_zero_dt
    
    # (2) comparison of mine vs James (correlation, timeseries) at each process stage (power, smoothed, logged)
    # (3) FFT on both
    # (4) LS on both

def detailed_dive_spectrogram(stime, etime, merged_ipwr_df, month, vlines=[],
                              n1='Fogg', n2='Waters'):
    
    print('hello')
    
    fig, ax = plt.subplots(nrows=2, figsize=(14, 15))


    akr_df = read_waters_masked_data.concat_monthly_data(1999, month)

    ax[0], x_arr, y_arr, z_arr = spectrogram_plotter.return_spectrogram(
        akr_df, ax[0], no_cbar=True, flux_tag='akr_flux_si_1au',
        cmap='gist_gray')




    # Timeseries
    ax[1].plot(merged_ipwr_df.datetime_rounded,
               merged_ipwr_df.waters_ipwr_100_650kHz, label=n2,
               color='palevioletred', linewidth=3., linestyle='dashed')
    ax[1].plot(merged_ipwr_df.datetime_rounded,
               merged_ipwr_df.ipwr_100_650kHz, label=n1,
               linewidth=3., color='grey')

    ax[1].set_yscale('log')
    ax[1].legend(loc='upper right', fontsize=fontsize)
    ax[1].set_xlabel('UT', fontsize=fontsize)
    ax[1].set_ylabel('Intensity', fontsize=fontsize)


    for (i, a) in enumerate(ax):
        t = a.text(0.05, 0.95, axes_labels[i], transform=a.transAxes,
                   fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))
        a.set_xlim(stime, etime)
        for t in vlines:
            a.axvline(t, linewidth=5., color='darkorchid')

    
    

def where_are_the_zeros(dt1, dt2, n1='Name1', n2='Name2'):
    # dt1 is list of datetimes with power ==0 for 1, ditto dt2 for 2
    
    dt1_day = dt1.dt.round('D')
    dt2_day = dt2.dt.round('D')
    
    n1 = n1 + " (" + str(len(dt1_day)) + " total zeros)"
    n2 = n2 + " (" + str(len(dt2_day)) + " total zeros)"
    # counts_frame1 = dt1_day.value_counts().to_frame().reset_index()
    # counts_frame2 = dt2_day.value_counts().to_frame().reset_index()
    c1 = dt1_day.value_counts()
    counts_frame1 = pd.DataFrame({'datetime': c1.index.values,
                                  'n': c1.values})
    c2 = dt2_day.value_counts()
    counts_frame2 = pd.DataFrame({'datetime': c2.index.values,
                                  'n': c2.values})    
    
    bar_width = pd.Timedelta(hours=10)

    fig, ax = plt.subplots(figsize=(14, 7))


    ax.bar(counts_frame1.datetime - bar_width/2., counts_frame1.n, bar_width,
           align='center', color='grey', label=n1)
    ax.bar(counts_frame2.datetime + bar_width/2., counts_frame2.n, bar_width,
           align='center', color='palevioletred', label=n2)

    ax.legend(loc='upper left', fontsize=fontsize)
    
    ax.tick_params(labelsize=fontsize)
    ax.set_ylabel("Number of I==0 fields")
    ax.set_xlabel("Day (in the Wind-Cassini Flyby)")
    
    ax.set_title("Number of times Intensity==0 was observed each day",
                 fontsize=fontsize*1.75)
    

def compare_two_timeseries(x1, y1, x2, y2,
                           n1='Name1', n2='Name2',
                           c1='black', c2='palevioletred', title='',
                           hist_bins=40):
    # x1, y1, x2, y2, n1='1', n2='2'):
    # x1 = np.linspace(0, 100, 1000)
    # x2 = np.linspace(0, 100, 1000)
    # y1 = np.sin(x1)
    # y2 = np.sin(x2) + np.random.normal(0., 200.*np.mean(y1), len(x2))

    fig = plt.figure(figsize=(12.5, 10))
    ax = [plt.subplot(221), plt.subplot(222), plt.subplot(212)]

    # Correlate
    ax[0].set_facecolor('lightgrey')
    h, xedges, yedges, im = ax[0].hist2d(y1, y2, bins=hist_bins, cmin=1, cmap='RdPu')
    # ax[0].plot(y1, y2, linewidth=0., marker='x', color='black')
    fig.colorbar(im, ax=ax[0], label='Counts')

    no_nan_df = pd.DataFrame({'y1': y1, 'y2': y2})
    nan_rows = no_nan_df.loc[(np.isnan(no_nan_df.y1)) | (np.isnan(no_nan_df.y2))].index
    no_nan_df.drop(nan_rows, axis=0, inplace=True)

    linear_fit = linregress(no_nan_df.y1, no_nan_df.y2)

    #breakpoint()
    x_mod = np.linspace(np.nanmin([y1, y2]), np.nanmax([y1, y2]), 100)
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


    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    # breakpoint()
    # QQ Plot
    ax[1] = qq_plot.qq_plot(y1, y2, ax[1])
    ax[1].set_xlabel(n1)
    ax[1].set_ylabel(n2)
    
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    # Timeseries
    ax[2].plot(x2, y2, label=n2, color=c2, linewidth=2., linestyle='dashed')
    ax[2].plot(x1, y1, label=n1, color=c1)

    ax[2].set_yscale('log')
    ax[2].legend(loc='upper right')
    ax[2].set_xlabel('UT')
    ax[2].set_ylabel('Intensity')

    for (i, a) in enumerate(ax):
        t = a.text(0.05, 0.95, axes_labels[i], transform=a.transAxes,
                   fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    fig.suptitle(title, x=0.5, y=0.95, fontsize=fontsize*1.75)

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


def read_waters_data():

    data_csv = os.path.join(fig_dir, "waters_2021_data",
                            "power_1999doy227_257.csv")
    
    data_df = pd.read_csv(data_csv, header=0,
                          names=['daily_min_index', 'GSE_LAT', 'GSE_LCT_T',
                                 'P_Wsr-1_100_650_kHz', 'P_Wsr-1_30_100_kHz',
                                 'P_Wsr-1_30_650_kHz', 'RADIUS', 'SM_LAT',
                                 'datetime_ut', 'unix', 'date_i'],
                          parse_dates=['datetime_ut'])
    
    return data_df