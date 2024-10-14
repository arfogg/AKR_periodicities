# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:15:12 2024

@author: A R Fogg
"""

import sys
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal

from numpy.fft import fft, ifft

from neurodsp.sim import sim_oscillation

import fastgoertzel as G

import periodicity_functions
import feature_importance
import read_and_tidy_data
import binning_averaging

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\wind_utility')
import read_integrated_power

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\readers')
import read_omni
import read_supermag

fig_dir = os.path.join("C:" + os.sep,
                       r"Users\admin\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")

# interesting stuff here on simulating intermittent oscillations
# https://neurodsp-tools.github.io/neurodsp/auto_tutorials/sim/plot_SimulatePeriodic.html


def test_with_oscillator():
    # Testing everything on the nice, fake oscillating signal
    time, akr_osc = oscillating_signal(24)

    freq, period, fft_amp, fft_fig, fft_ax = periodicity_functions.\
        generic_fft_function(time, akr_osc, pd.Timedelta(minutes=3),
                             signal_xlims=[0, 72.*60.*60.],
                             vertical_indicators=[12, 24])

    return


def generate_plots():

    # intervals = np.array(['full_archive', 'cassini_flyby'])
    intervals = np.array(['full_archive'])

    fft_signal_x_start = np.array([pd.Timestamp(1999, 8, 18, 0).timestamp()])
    fft_signal_x_width = np.array([5. * 24. * 60. * 60.])

    for i in range(intervals.size):
        print('Running analyses for ', intervals[i])
        combined_rounded_df = read_and_tidy_data.\
            combine_rounded_akr_omni(intervals[i])

        # -- FFT --
        fft_png = os.path.join(fig_dir, intervals[i] + '_fft.png')
        if (pathlib.Path(fft_png).is_file()) is False:
            # First 3 days of data
            signal_xlims = [fft_signal_x_start[i],
                            fft_signal_x_start[i] + fft_signal_x_width[i]]

            freq, period, fft_amp, inverse_signal = periodicity_functions.\
                generic_fft_function(combined_rounded_df.unix,
                                     combined_rounded_df['integrated_power'],
                                     pd.Timedelta(minutes=3))
            freq_sur, period_sur, fft_amp_sur, inverse_signal_sur = \
                periodicity_functions.\
                generic_fft_function(
                    combined_rounded_df.unix,
                    combined_rounded_df['surrogate_integrated_power'],
                    pd.Timedelta(minutes=3))
            # Change zeros to nans for plotting intensity
            r_ind = combined_rounded_df.loc[
                combined_rounded_df.integrated_power == 0].index
            pwr = np.array(combined_rounded_df.
                           integrated_power.copy(deep=True))
            pwr[r_ind] = np.nan

            fft_fig, fft_ax = periodicity_functions.plot_fft_summary(
                combined_rounded_df.unix, pwr, pd.Timedelta(minutes=3),
                freq, period, fft_amp, inverse_signal,
                surrogate_period=period_sur,
                surrogate_fft_amp=fft_amp_sur,
                fontsize=15,
                fft_xlims=[0, 36],
                signal_xlims=signal_xlims,
                signal_y_log=True,
                vertical_indicators=[12, 24],
                unix_to_dtime=True,
                resolution_lim=True)
            fft_fig.savefig(fft_png)
        # -- END FFT --

        # -- ACF --
        temporal_resolution = 3. * 60.  # in seconds
        n_shifts = 5000
        acf_png = os.path.join(fig_dir, intervals[i] + '_' + str(n_shifts)
                               + '_acf.png')
        acf_csv = os.path.join(data_dir, intervals[i] + '_' + str(n_shifts)
                               + '_acf.csv')

        if (pathlib.Path(acf_png).is_file()) is False:
            #lags = np.array(range(n_shifts)) * temporal_resolution
            if (pathlib.Path(acf_csv).is_file()) is False:
                acf, lags = periodicity_functions.autocorrelation(
                    combined_rounded_df['integrated_power'], n_shifts,
                    temporal_resolution=temporal_resolution, starting_lag=7200)
                acf_df = pd.DataFrame({'acf': acf,
                            'lags': lags})
                acf_df.to_csv(index=False)
            else:
                acf_df = pd.read_csv(acf_csv, float_precision='round trip')
                acf = acf_df['acf']
                lags = acf_df['lags']

            acf_fig, acf_ax = periodicity_functions.\
                plot_autocorrelogram(lags, acf, tick_sep_hrs=36.,
                                         highlight_period=24.,)
            
            acf_fig.savefig(acf_png)
        breakpoint()
        # -- END ACF --

        # -- FEATURE IMPORTANCE --
        loc_fi_png = os.path.join(fig_dir, intervals[i] + '_loc_fi.png')
        geo_fi_png = os.path.join(fig_dir, intervals[i] + '_geo_fi.png')
        all_fi_png = os.path.join(fig_dir, intervals[i] + '_all_fi.png')
        combined_fi_png = os.path.join(fig_dir, intervals[i] +
                                       '_combined_fi.png')

        location_fi_csv = os.path.join(data_dir, intervals[i] +
                                       '_location_only_fidata.csv')
        geophysical_fi_csv = os.path.join(data_dir, intervals[i] +
                                          '_geophysical_only_fidata.csv')
        all_fi_csv = os.path.join(data_dir, intervals[i] +
                                  '_loc_geophysical_fidata.csv')

        if (pathlib.Path(loc_fi_png).is_file() is False) | (pathlib.Path(geo_fi_png).is_file() is False) | (pathlib.Path(all_fi_png).is_file() is False) | (pathlib.Path(combined_fi_png).is_file() is False):

            # Location Feature Importance
            loc_feature_name = np.array(['LT', 'Latitude', 'Radial Distance'])
            if pathlib.Path(location_fi_csv).is_file() is False:
                loc_fig, loc_ax, loc_importance = feature_importance.\
                    plot_feature_importance(
                        np.array(combined_rounded_df.integrated_power),
                        np.array(combined_rounded_df[['decimal_gseLT',
                                                      'lat_gse', 'lon_gse']]),
                        feature_names=loc_feature_name,
                        seed=1993, fontsize=20, record_number=True)
                location_fi_df = pd.DataFrame({
                    'feature_name': loc_feature_name,
                    'importance': loc_importance})
                location_fi_df.to_csv(location_fi_csv, index=False)

            else:
                location_fi_df = pd.read_csv(location_fi_csv, delimiter=',',
                                             float_precision='round_trip')
                loc_fig, loc_ax, importance = feature_importance.\
                    plot_feature_importance(
                        np.array(combined_rounded_df.integrated_power),
                        np.array(combined_rounded_df[['decimal_gseLT',
                                                      'lat_gse', 'lon_gse']]),
                        importance=np.array(location_fi_df.importance),
                        feature_names=loc_feature_name,
                        seed=1993, fontsize=20, record_number=True)
            loc_fig.savefig(loc_fi_png, bbox_inches='tight')

            # Geophysical Feature Importance
            geo_feature_name = np.array(['Bx', 'By', 'Bz', 'Bt',
                                         'ClockAngle', 'Vsw', 'Nsw',
                                         'Psw', 'AE', 'AL', 'AU', 'SYM-H',
                                         'SME', 'SMU', 'SML', 'SMR'])
            geo_feature_tag = np.array(['bx', 'by_gsm', 'bz_gsm', 'b_total',
                                        'clock_angle', 'flow_speed',
                                        'proton_density', 'flow_pressure',
                                        'ae', 'al', 'au', 'symh',
                                        'SME', 'SMU', 'SML', 'SMR'])
            geo_df = combined_rounded_df.dropna(subset=geo_feature_tag
                                                ).reset_index(drop=True)
            if pathlib.Path(geophysical_fi_csv).is_file() is False:
                geo_fig, geo_ax, geo_importance = feature_importance.\
                    plot_feature_importance(np.array(geo_df.integrated_power),
                                            np.array(geo_df[geo_feature_tag]),
                                            feature_names=geo_feature_name,
                                            seed=1993, fontsize=20,
                                            record_number=True)
                geophysical_fi_df = pd.DataFrame({
                    'feature_name': geo_feature_name,
                    'importance': geo_importance})
                geophysical_fi_df.to_csv(geophysical_fi_csv, index=False)
            else:
                geophysical_fi_df = pd.read_csv(geophysical_fi_csv,
                                                delimiter=',',
                                                float_precision='round_trip')
                geo_fig, geo_ax, geo_importance = feature_importance.\
                    plot_feature_importance(
                        np.array(geo_df.integrated_power),
                        np.array(geo_df[geo_feature_tag]),
                        importance=np.array(geophysical_fi_df.importance),
                        feature_names=geo_feature_name,
                        seed=1993, fontsize=20, record_number=True)
            geo_fig.savefig(geo_fi_png, bbox_inches='tight')

            # All Features
            all_feature_name = np.array(['LT', 'Latitude', 'Radial Distance',
                                         'Bx', 'By', 'Bz', 'Bt',
                                         'ClockAngle', 'Vsw', 'Nsw',
                                         'Psw', 'AE', 'AL', 'AU', 'SYM-H',
                                         'SME', 'SMU', 'SML', 'SMR'])
            all_feature_tag = np.array(['decimal_gseLT', 'lat_gse', 'lon_gse',
                                        'bx', 'by_gsm', 'bz_gsm', 'b_total',
                                        'clock_angle', 'flow_speed',
                                        'proton_density', 'flow_pressure',
                                        'ae', 'al', 'au', 'symh',
                                        'SME', 'SMU', 'SML', 'SMR'])
            all_df = combined_rounded_df.dropna(subset=all_feature_tag
                                                ).reset_index(drop=True)
            if pathlib.Path(all_fi_csv).is_file() is False:
                all_fig, all_ax, all_importance = feature_importance.\
                    plot_feature_importance(np.array(all_df.integrated_power),
                                            np.array(all_df[all_feature_tag]),
                                            feature_names=all_feature_name,
                                            seed=1993, fontsize=20,
                                            record_number=True)
                all_fi_df = pd.DataFrame({
                    'feature_name': all_feature_name,
                    'importance': all_importance})
                all_fi_df.to_csv(all_fi_csv, index=False)
            else:
                all_fi_df = pd.read_csv(all_fi_csv,
                                        delimiter=',',
                                        float_precision='round_trip')
                all_fig, all_ax, geo_importance = feature_importance.\
                    plot_feature_importance(
                        np.array(all_df.integrated_power),
                        np.array(all_df[all_feature_tag]),
                        importance=np.array(all_fi_df.importance),
                        feature_names=all_feature_name,
                        seed=1993, fontsize=20, record_number=True)
            all_fig.savefig(all_fi_png, bbox_inches='tight')

            # Combined features panel
            combined_fig, combined_ax = feature_importance.\
                feature_importance_3panel(
                    loc_feature_name, geo_feature_name, all_feature_name,
                    np.array(location_fi_df.importance),
                    np.array(geophysical_fi_df.importance),
                    np.array(all_fi_df.importance), titles=[
                        '(a) Visibility', '(b) Geophysical',
                        '(c) Visibility and Geophysical'])
            combined_fig.savefig(combined_fi_png, bbox_inches='tight')
        # -- END FEATURE IMPORTANCE --
        
        # -- BINNING / AVERAGING --
        binned_median_png = os.path.join(fig_dir, intervals[i] + '_MLT_UT_binning_median.png')
        binned_boxplot_png = os.path.join(fig_dir, intervals[i] + '_MLT_UT_binning_boxplot.png')

        if (pathlib.Path(binned_median_png).is_file() is False) | (pathlib.Path(binned_boxplot_png).is_file() is False):
            fig_m, fig_b = binning_averaging.plot_UT_trend(combined_rounded_df)
        
            fig_m.savefig(binned_median_png, bbox_inches='tight')
            fig_b.savefig(binned_boxplot_png, bbox_inches='tight')
        
    return





def lomb_scargle(intensity_df):
    """
    TEST FUNCTION TO DO A LOMB SCARGLE
    ON AKR PERIODICITIES - WORK
    IN PROGRESS

    Parameters
    ----------
    intensity_df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    freqs_in_hrs=np.linspace(0,15,16)
    
    pgram=signal.lombscargle(intensity_df['datetime_ut'], intensity_df['P_Wsr-1_100_650_kHz'], freqs_in_hrs)
    
    fig,ax=plt.subplots()
    ax.plot(freqs_in_hrs, pgram)

    
def test_goertzel():
    
    
    def wave(amp, freq, phase, x):
        return amp * np.sin(2*np.pi * freq * x + phase)


    x = np.arange(0, 512)
    y = wave(1, 1/128, 0, x)

    amp, phase = G.goertzel(y, 1/128)
    print(f'Goertzel Amp: {amp:.4f}, phase: {phase:.4f}')



    # Compared to max amplitude FFT output 
    ft = np.fft.fft(y)
    FFT = pd.DataFrame()
    FFT['amp'] = np.sqrt(ft.real**2 + ft.imag**2) / (len(y) / 2)
    FFT['freq'] = np.fft.fftfreq(ft.size, d=1)
    FFT['phase'] = np.arctan2(ft.imag, ft.real)

    max_ = FFT.iloc[FFT['amp'].idxmax()]
    print(f'FFT amp: {max_["amp"]:.4f}, '
            f'phase: {max_["phase"]:.4f}, '
            f'freq: {max_["freq"]:.4f}')
    
    a=np.full(len(FFT), np.nan)
    p=np.full(len(FFT), np.nan)
    for i in range(len(FFT)):
        a[i], p[i] = G.goertzel(y, FFT['freq'][i])
        
    fig,ax=plt.subplots(nrows=2)

    ax[0].plot(x,y)

    ax[1].plot(FFT['phase'], FFT['amp'], marker='.', linewidth=0.0, label='fft')
    #ax[1].plot(phase,amp, marker='*', fillstyle='none', markersize=15.0, linewidth=0.0, label='goertzel')
    ax[1].plot(p,a, marker='*', fillstyle='none', markersize=15.0, linewidth=0.0, label='goertzel')
    ax[1].legend()
    
    
    
    
    # WITH DATA

def oscillating_signal(osc_freq, plot=False):
    """
    Function to create a timeseries of oscillating
    signal using neurodsp package.

    Parameters
    ----------
    osc_freq : float
        Period of the desired oscillation in hours.
    plot : bool, optional
        If plot == True, a diagnostic plot of the
        generated signal is presented. The default
        is False.

    Returns
    -------
    time : np.array
        Time axis in seconds.
    akr_osc : np.array
        Signal.

    """
    # Create fake time axis
    yr_secs = 365*24*60*60    # One year in seconds
    res_secs = 3*60   # Temporal resolution of 3 minutes

    time = np.arange(0, yr_secs, res_secs)

    akr_osc = sim_oscillation(yr_secs, 1/res_secs, 1/(osc_freq*60*60),
                              cycle='sine')
    akr_osc = (akr_osc+2.0) * 1e6
    # ^ make positive and put to the order of AKR power

    if plot:
        fig, ax = plt.subplots()
        ax.plot(time, akr_osc)
        ax.set_xlim(0, 4*osc_freq*60*60)

    return time, akr_osc


def generic_fft_function(time, y, temporal_resolution, plot=True,
                         xlims=[np.nan, np.nan]):
    """
    

    Parameters
    ----------
    time : np.array
        Time axis for y in seconds.
    y : np.array
        Signal.
    temporal_resolution : pd.Timedelta
        Seperation of consecutive points in time.
    plot : BOOL, optional
        If plot==True, a diagnostic plot for the FFT is 
        presented. The default is True.
    xlims : list, optional
        If provided, the xaxis of the signal and IFFT
        plots are limits to these values. This can allow 
        nicer plots of a few oscillations. The default
        is [np.nan, np.nan].

    Returns
    -------
    freq : np.array
        Frequency of FFT calculation in Hz.
    period : np.array
        Period of FFT calculation in hours.
    fft_amp : np.array
        Amplitude of FFT calculation.  

    """
    # temporal_resolution is a pd.Timedelta
    
    # Calculate sampling rate in Hz
    sampling_rate = 1 / (temporal_resolution.total_seconds())
    
    X = fft(y)  # y axis for FFT plot
    N = len(X)  # number of FFT points
    n = np.arange(N)    # 0 - N array. integers
    T = N/sampling_rate    # number of FFT points / number of obs per sec
    freq = n/T  # freqs fft is evaluated at
    
    # Functions to convert between period in hours
    #   and frequency in Hz
    def period_to_freq(x):
        ticks=[]
        for tick in x:
            if tick != 0:
                ticks.append(1. / (tick * (60.*60.)))
            else:
                ticks.append(0)
        #print('pf2', x, ticks)
        return ticks
    def freq_to_period(x):
        ticks=[]
        for tick in x:
            if tick != 0:
                ticks.append((1. / tick) / (60.*60.))
            else:
                ticks.append(0)
        #print('f2p', x, ticks)
        return ticks
    
    
    
    
    # period = 1 / freq
    # period = period / (60*60)   # period in hours
    period = freq_to_period(freq)
    #fig,ax=plt.subplots(ncols=2, figsize = (12, 6))

    fft_amp=np.abs(X)

    if plot :
        
        fig,ax=plt.subplots(ncols=3, figsize = (18, 6))
        
        # Plot original signal
        ax[0].plot(time, y, 'orange')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_title('Observations')
        
        # Plot FFT periodogram
        ax[1].stem(period, fft_amp, 'c', \
                  markerfmt=" ", basefmt="-c") 
        ax[1].set_xlabel('Period (hours)')
        ax[1].set_ylabel('FFT Amplitude |X(freq)|')
        ax[1].set_title('FFT of observations')
        ax[1].set_xlim([-5,36])
        # # Top axis with freq - doesn't work yet
        # top_ax=ax[1].twiny()
        # top_ax.set_xticks(period_to_freq(ax[1].get_xticks()))
        
        
        # Plot inverse FFT signal
        ax[2].plot(time, ifft(X), 'blueviolet')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Amplitude')
        ax[2].set_title('Inverse FFT')
        
        if (~np.isnan(xlims[0])) & (~np.isnan(xlims[1])):
            ax[0].set_xlim(xlims)
            ax[2].set_xlim(xlims)
        
        fig.tight_layout()
    
    return freq, period, fft_amp
    
    
    