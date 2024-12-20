# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:15:12 2024

@author: A R Fogg
"""

import sys
import os
import pathlib
import string

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import matplotlib.transforms as transforms

from numpy.fft import fft, ifft

from neurodsp.sim import sim_oscillation

import fastgoertzel as G

import periodicity_functions
import feature_importance
import read_and_tidy_data
import binning_averaging
import wind_location
import diurnal_oscillator
import lomb_scargle

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\wind_utility')
#import read_integrated_power
import read_wind_position

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\readers')
import read_omni
import read_supermag


fontsize = 15
alphabet = list(string.ascii_lowercase)
axes_labels = []
for a in alphabet:
    axes_labels.append('(' + a + ')')

fig_dir = os.path.join("C:" + os.sep,
                       r"Users\admin\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")

# interesting stuff here on simulating intermittent oscillations
# https://neurodsp-tools.github.io/neurodsp/auto_tutorials/sim/plot_SimulatePeriodic.html


def trajectory_plots():
    """
    Create and save trajectory plots for the three
    Wind intervals.

    Returns
    -------
    None.

    """

    interval_options = read_and_tidy_data.return_test_intervals()

    years = np.arange(1995, 2004 + 1)
    # Read Wind position data
    wind_position_df = read_wind_position.concat_data_years(years)

    fig = plt.figure(figsize=(15, 15))
    axes = np.array([fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2),
                     fig.add_subplot(2, 2, 3),
                     fig.add_subplot(2, 2, 4, projection='polar')])

    for i, ax in enumerate(axes.reshape(-1)[:-1]):
        ax = wind_location.plot_trajectory(interval_options.stime.iloc[i],
                                           interval_options.etime.iloc[i],
                                           wind_position_df, ax,
                                           fontsize=fontsize)

        # Formatting
        ax.set_title(interval_options.title.iloc[i], fontsize=fontsize)
        t = ax.text(0.05, 0.95, axes_labels[i], transform=ax.transAxes,
                    fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

        # LT histogram
        draw_ticks = True if i == 2 else False
        axes.reshape(-1)[-1] = wind_location.lt_hist(
            interval_options.stime.iloc[i], interval_options.etime.iloc[i],
            wind_position_df, axes.reshape(-1)[-1],
            bar_fmt={'color': interval_options.color.iloc[i],
                     'edgecolor': 'black', 'alpha': 0.4,
                     'label': interval_options.label.iloc[i]},
            draw_ticks=draw_ticks)

    t = axes.reshape(-1)[-1].text(0.05, 0.95, axes_labels[3],
                                  transform=axes.reshape(-1)[-1].transAxes,
                                  fontsize=fontsize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    fig.tight_layout()

    traj_fig = os.path.join(fig_dir, "three_interval_traj.png")
    fig.savefig(traj_fig)


def run_lomb_scargle():

    # Initialising variables
    # periods = np.logspace(np.log10(1), np.log10(48), 500)  # in hours
    # freqs = periodicity_functions.period_to_freq(periods)
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    T = (pd.Timestamp(2005, 1, 1, 0) - pd.Timestamp(1995, 1, 1, 0)).total_seconds()
    f_min, f_max, N_f, freqs = lomb_scargle.define_frequency_bins(T, f_min, f_max, n0=5)
    freqs = freqs[::-1]
    angular_freqs = 2 * np.pi * freqs
    periods = periodicity_functions.freq_to_period(freqs)
    vertical_indicators = [12, 24]
    vertical_ind_col = 'black'

    # Different frequency channels
    freq_tags = np.array(['ipwr_100_400kHz', 'ipwr_50_100kHz'  # ,
                          #'ipwr_100_650kHz'
                          ])
    freq_colors = np.array(['dimgrey', 'darkorange', 'rebeccapurple'])

    LS_fig = os.path.join(fig_dir, "three_interval_lomb_scargle.png")

    # Read in interval data
    interval_options = read_and_tidy_data.return_test_intervals()



    # Initialise plotting window
    fig, ax = plt.subplots(nrows=4, figsize=(10, 17))




    # # Run Lomb-Scargle over the fake oscillator
    # print('Running Lomb-Scargle analysis on fake oscillating signal')
    # ftime, fsignal = diurnal_oscillator.oscillating_signal(24., plot=False,
    #                                                        add_noise=True)

    # ls_pgram = lomb_scargle.generic_lomb_scargle(ftime, fsignal, angular_freqs)
    # ax[0] = lomb_scargle.plot_LS_summary(periods, ls_pgram,
    #                              vertical_indicators=vertical_indicators,
    #                              ax=ax[0], vertical_ind_col=vertical_ind_col)






    for (i, interval_tag) in enumerate(interval_options['tag']):
        print('Running Lomb-Scargle for ', interval_tag)
        
        base_dir = pathlib.Path(data_dir) / 'lomb_scargle'
        file_paths = [base_dir / f"LS_{interval_tag}_{f}.csv" for f in freq_tags]
        file_checks = [file_path.is_file() for file_path in file_paths]


        if all(file_checks) is False:
            
            akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)

        # Remove any rows where intensity == np.nan
        for (j, (freq_column, c)) in enumerate(zip(freq_tags, freq_colors)):
            print('Frequency band: ', freq_column)
            ls_csv = os.path.join(data_dir, 'lomb_scargle', 'LS_' +
                                  interval_tag + '_' + freq_column + '.csv')


            if pathlib.Path(ls_csv).is_file() is False:

                freq_df = akr_df.dropna(subset=[freq_column])
                t1 = pd.Timestamp.now()
                print('starting LS at ', t1)
                ls_pgram = lomb_scargle.generic_lomb_scargle(freq_df.unix,
                                                             freq_df[freq_column],
                                                             angular_freqs)
                t2 = pd.Timestamp.now()
                print('LS finished, time elapsed: ', t2-t1)
                ls_df = pd.DataFrame({'period_hr': periods,
                                      'freq_Hz': freqs,
                                      'angular_freq': angular_freqs,
                                      'ls_pgram': ls_pgram})
                ls_df.to_csv(ls_csv, index=False)
                
            else:
                ls_df = pd.read_csv(ls_csv, delimiter=',',
                                    float_precision='round_trip')

                ls_pgram = np.array(ls_df.ls_pgram)

            # ax[i + 1] = lomb_scargle.plot_LS_summary(periods, ls_pgram,
            #                                          vertical_indicators=[12.,
            #                                                               24.],
            #                                          ax=ax[i+1])
            ax[i + 1].plot(periods, ls_pgram, linewidth=1.5, color=c, label=freq_column)

        ax[i + 1].set_xscale('log')

        # Formatting
        ax[i + 1].set_ylabel('Lomb-Scargle\nNormalised Amplitude', fontsize=fontsize)
        ax[i + 1].set_xlabel('Period (hours)', fontsize=fontsize)
        ax[i + 1].tick_params(labelsize=fontsize)
        ax[i + 1].legend(fontsize=fontsize, loc='center left')

        if vertical_indicators != []:
            for h in vertical_indicators:
                ax[i + 1].axvline(h, color=vertical_ind_col, linestyle='dashed',
                           linewidth=1.5)
                trans = transforms.blended_transform_factory(ax[i + 1].transData,
                                                             ax[i + 1].transAxes)
                ax[i + 1].text(h, 1.075, str(h), transform=trans,
                        fontsize=fontsize, va='top', ha='center',
                        color=vertical_ind_col)

    # Label panels
    for (i, a) in enumerate(ax):
        t = a.text(0.025, 0.95, axes_labels[i], transform=a.transAxes,
                   fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    # Adjust margins etc
    fig.tight_layout()

    # Save to file
    fig.savefig(LS_fig)




















def TEMP_neat_fft_plot():
    png_name = os.path.join(fig_dir, "neat_single_10_year_fft.png")

    interval_options = read_and_tidy_data.return_test_intervals()
    intervals = np.array(interval_options.tag)

    fft_signal_x_start = np.array([pd.Timestamp(1999, 8, 15, 0).timestamp(),
                                   pd.Timestamp(1999, 8, 15, 0).timestamp(),
                                   pd.Timestamp(2003, 10, 11,
                                                22, 36).timestamp()])
    fft_signal_x_width = np.repeat([5. * 24. * 60. * 60.], len(intervals))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    print('Running analyses for ', intervals[0])
    combined_rounded_df = read_and_tidy_data.\
        combine_rounded_akr_omni(intervals[0])

    # First 3 days of data
    signal_xlims = [fft_signal_x_start[0],
                    fft_signal_x_start[0] +
                    fft_signal_x_width[0]]

    freq, period, fft_amp, inverse_signal = periodicity_functions.\
        generic_fft_function(combined_rounded_df.unix,
                             combined_rounded_df['integrated_power'],
                             pd.Timedelta(minutes=3))
    # freq_sur, period_sur, fft_amp_sur, inverse_signal_sur = \
    #         periodicity_functions.\
    #         generic_fft_function(
    #             combined_rounded_df.unix,
    #             combined_rounded_df['surrogate_integrated_power'],
    #             pd.Timedelta(minutes=3))

    axes = periodicity_functions.plot_fft_summary(
                combined_rounded_df.unix,
                np.array(combined_rounded_df.integrated_power),
                pd.Timedelta(minutes=3),
                freq, period, fft_amp, inverse_signal,
                # surrogate_period=period_sur,
                # surrogate_fft_amp=fft_amp_sur,
                fontsize=fontsize,
                fft_xlims=[0, 36],
                signal_xlims=signal_xlims,
                signal_y_log=True,
                vertical_indicators=[12, 24],
                unix_to_dtime=True,
                resolution_lim=True,
                # input_ax=ax[i, :],
                panel_label=False)
    ax = axes[1]

        # y_l_ax = ax[i, 2].twinx()
        # y_l_ax.set_yticks([])
        # y_l_ax.set_ylabel(interval_options.title.iloc[i], fontsize=fontsize,
        #                   weight='heavy', rotation=-90, labelpad=35)

    # # Panel labels
    # for i, (lab, a) in enumerate(zip(axes_labels, ax.reshape(-1)[:-1])):
    #     t = a.text(0.05, 0.95, lab, transform=a.transAxes,
    #                fontsize=fontsize, va='top', ha='left')
    #     t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    fig.tight_layout()

    # fig.savefig(png_name)
    return



def test_with_oscillator():
    # Testing everything on the nice, fake oscillating signal
    time, akr_osc = oscillating_signal(24)

    freq, period, fft_amp, inverse_signal = periodicity_functions.\
        generic_fft_function(time, akr_osc, pd.Timedelta(minutes=3))

    return freq, period, fft_amp





def generate_fft_plot():

    png_name = os.path.join(fig_dir, "three_interval_fft.png")

    interval_options = read_and_tidy_data.return_test_intervals()
    intervals = np.array(interval_options.tag)

    fft_signal_x_start = np.array([pd.Timestamp(1999, 8, 15, 0).timestamp(),
                                   pd.Timestamp(1999, 8, 15, 0).timestamp(),
                                   pd.Timestamp(2003, 10, 11,
                                                22, 36).timestamp()])
    fft_signal_x_width = np.repeat([5. * 24. * 60. * 60.], len(intervals))

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))

    for i in range(intervals.size):
        print('Running analyses for ', intervals[i])
        combined_rounded_df = read_and_tidy_data.\
            combine_rounded_akr_omni(intervals[i])

        # First 3 days of data
        signal_xlims = [fft_signal_x_start[i],
                        fft_signal_x_start[i] +
                        fft_signal_x_width[i]]

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

        ax[i, :] = periodicity_functions.plot_fft_summary(
                combined_rounded_df.unix,
                np.array(combined_rounded_df.integrated_power),
                pd.Timedelta(minutes=3),
                freq, period, fft_amp, inverse_signal,
                surrogate_period=period_sur,
                surrogate_fft_amp=fft_amp_sur,
                fontsize=fontsize,
                fft_xlims=[0, 36],
                signal_xlims=signal_xlims,
                signal_y_log=True,
                vertical_indicators=[12, 24],
                unix_to_dtime=True,
                resolution_lim=True,
                input_ax=ax[i, :], panel_label=False)

        y_l_ax = ax[i, 2].twinx()
        y_l_ax.set_yticks([])
        y_l_ax.set_ylabel(interval_options.title.iloc[i], fontsize=fontsize,
                          weight='heavy', rotation=-90, labelpad=35)

    # Panel labels
    for i, (lab, a) in enumerate(zip(axes_labels, ax.reshape(-1)[:-1])):
        t = a.text(0.05, 0.95, lab, transform=a.transAxes,
                   fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    fig.tight_layout()

    fig.savefig(png_name)
    return


def generate_individual_plots():

    interval_options = read_and_tidy_data.return_test_intervals()
    # intervals = np.array(['full_archive', 'cassini_flyby'])
    intervals = np.array(interval_options.tag)

    fft_signal_x_start = np.array(interval_options.stime)
    # fft_signal_x_start = np.array([pd.Timestamp(1999, 8, 18, 0).timestamp()])
    fft_signal_x_width = np.repeat([5. * 24. * 60. * 60.], len(intervals))

    for i in range(intervals.size):
        print('Running analyses for ', intervals[i])
        combined_rounded_df = read_and_tidy_data.\
            combine_rounded_akr_omni(intervals[i])

        # -- FFT --
        fft_png = os.path.join(fig_dir, intervals[i] + '_fft.png')
        if (pathlib.Path(fft_png).is_file()) is False:
            # First 3 days of data
            signal_xlims = [pd.Timestamp(fft_signal_x_start[i]).timestamp(),
                            pd.Timestamp(fft_signal_x_start[i]).timestamp() + fft_signal_x_width[i]]

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

        # -- END ACF --
    return

def DEPRECATED_run_feature_importance():

    interval_options = read_and_tidy_data.return_test_intervals()
    # intervals = np.array(['full_archive', 'cassini_flyby'])
    intervals = np.array(interval_options.tag)

    fft_signal_x_start = np.array([pd.Timestamp(1999, 8, 18, 0).timestamp()])
    fft_signal_x_width = np.array([5. * 24. * 60. * 60.])

    for i in range(intervals.size):
        print('Running analyses for ', intervals[i])
        combined_rounded_df = read_and_tidy_data.\
            combine_rounded_akr_omni(intervals[i])

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





# def lomb_scargle(intensity_df):
#     """
#     TEST FUNCTION TO DO A LOMB SCARGLE
#     ON AKR PERIODICITIES - WORK
#     IN PROGRESS

#     Parameters
#     ----------
#     intensity_df : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     freqs_in_hrs=np.linspace(0,15,16)
    
#     pgram=signal.lombscargle(intensity_df['datetime_ut'], intensity_df['P_Wsr-1_100_650_kHz'], freqs_in_hrs)
    
#     fig,ax=plt.subplots()
#     ax.plot(freqs_in_hrs, pgram)

    
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
    
    
    