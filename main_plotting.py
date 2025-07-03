# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:15:12 2024

@author: A R Fogg
"""

# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import os
import pathlib
import string
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import matplotlib.transforms as transforms

from numpy.fft import fft, ifft

from neurodsp.sim import sim_oscillation

# import fastgoertzel as G

import periodicity_functions
# import feature_importance
import read_and_tidy_data
import binning_averaging
import wind_location
import diurnal_oscillator
import lomb_scargle
import autocorrelation
import bootstrap_functions
import utility

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\wind_utility')
#import read_integrated_power
import read_wind_position

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

# interesting stuff here on simulating intermittent oscillations
# https://neurodsp-tools.github.io/neurodsp/auto_tutorials/sim/plot_SimulatePeriodic.html



def read_synthetic_oscillator():
    """
    Run and/or read in the synthetic AKR periodic signal.

    Returns
    -------
    time : np.array
        Time axis in seconds.
    akr_osc : np.array
        Simulated AKR intensity, arbitrary units.

    """
    synthetic_signal_csv = os.path.join(data_dir,
                                        "synthetic_oscillatory_signal.csv")
    synthetic_signal_fig = os.path.join(fig_dir,
                                        "synthetic_oscillatory_signal.png")

    if (pathlib.Path(synthetic_signal_csv).is_file()) & \
            (pathlib.Path(synthetic_signal_fig).is_file()):
        signal_df = pd.read_csv(synthetic_signal_csv, delimiter=',',
                                float_precision='round_trip')
        time = np.array(signal_df.time)
        akr_osc = np.array(signal_df.akr_osc)
    else:
        time, akr_osc, fig, ax = diurnal_oscillator.\
            oscillating_signal(24, plot=True)

        # Write to file
        signal_df = pd.DataFrame({'time': time, 'akr_osc': akr_osc})
        signal_df.to_csv(synthetic_signal_csv, index=False)

        fig.savefig(synthetic_signal_fig)

    return time, akr_osc


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


def read_subset_bootstraps(subset_n, freq_ch='ipwr_100_400kHz',
                           n_bootstrap=100):

    if subset_n == 'synthetic':
        # Run Lomb-Scargle over the fake oscillator
        ftime, fsignal = read_synthetic_oscillator()
        # Remove NaN rows
        clean_ind, = np.where(~np.isnan(fsignal))
        ftime = ftime[clean_ind]
        fsignal = fsignal[clean_ind]
        # Define fname
        synthetic_BS_pkl = os.path.join(data_dir, "synthetic_signal_"
                                        + str(n_bootstrap) + "_BSs.pkl")
        if pathlib.Path(synthetic_BS_pkl).is_file():
            # Read in bootstraps
            with open(synthetic_BS_pkl, 'rb') as f:
                BS = pickle.load(f)
        else:
            print('Generating ')
            BS = np.full((fsignal.size, n_bootstrap), np.nan)
            for i in range(n_bootstrap):
                # Generate bootstrap
                BS[:, i] = bootstrap_functions.generate_bootstrap(fsignal)
            with open(synthetic_BS_pkl, 'wb') as f:
                pickle.dump(BS, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        akr_df = read_and_tidy_data.select_akr_intervals(subset_n)

        ftime = np.array(akr_df.datetime)
        fsignal = np.array(akr_df[freq_ch])

        # Remove NaN rows
        clean_ind, = np.where(~np.isnan(fsignal))
        ftime = ftime[clean_ind]
        fsignal = fsignal[clean_ind]
        # Define fname
        BS_pkl = os.path.join(data_dir, subset_n + "_" + freq_ch +
                              "_" + str(n_bootstrap) + "_BSs.pkl")

        if pathlib.Path(BS_pkl).is_file():
            # Read in bootstraps
            with open(BS_pkl, 'rb') as f:
                BS = pickle.load(f)
        else:
            print('Generating bootstraps for ', subset_n, freq_ch)
            BS = np.full((fsignal.size, n_bootstrap), np.nan)
            for i in range(n_bootstrap):
                # Generate bootstrap
                BS[:, i] = bootstrap_functions.generate_bootstrap(fsignal)
            with open(BS_pkl, 'wb') as f:
                pickle.dump(BS, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ftime, BS


def run_lomb_scargle():

    # Initialising variables
    # periods = np.logspace(np.log10(1), np.log10(48), 500)  # in hours
    # freqs = periodicity_functions.period_to_freq(periods)
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    T = (pd.Timestamp(2005, 1, 1, 0) -
         pd.Timestamp(1995, 1, 1, 0)).total_seconds()
    samples_per_peak = 5
    # f_min, f_max, N_f, freqs = lomb_scargle.define_frequency_bins(T, f_min,
    #                                                               f_max, n0=5)

    # freqs = freqs[::-1]
    # angular_freqs = 2 * np.pi * freqs
    # periods = periodicity_functions.freq_to_period(freqs)
    vertical_indicators = [12, 24]
    vertical_ind_col = 'black'

    annotate_bbox = {"facecolor": "white", "edgecolor": "grey", "pad": 5.}


    # Different frequency channels
    freq_tags = np.array(['ipwr_100_400kHz', 'ipwr_50_100kHz'  # ,
                          #'ipwr_100_650kHz'
                          ])
    freq_labels = np.array(['100-400 kHz', '50-100 kHz'])
    freq_colors = np.array(['dimgrey', 'darkorange', 'rebeccapurple'])

    LS_fig = os.path.join(fig_dir, "three_interval_lomb_scargle.png")

    # Number of bootstrap
    n_bootstrap = 100

    # FAP filenames
    FAP_peaks_dir = os.path.join(data_dir, "lomb_scargle", 'LS_peaks_for_FAP')
    synthetic_FAP_pkl = os.path.join(data_dir, "lomb_scargle", "synthetic_FAP_"
                                     + str(n_bootstrap) + "_BSs.pkl")
    # FAP_fmt_dict = {'linewidth': 2.,
    #                 'linestyle': 'dashed',
    #                 'color': 'blueviolet'}

    # Read in interval data
    print('Reading AKR data over requested intervals')
    interval_options = read_and_tidy_data.return_test_intervals()

    # Initialise plotting window
    fig, ax = plt.subplots(nrows=4, figsize=(12.5, 17))

    # Run Lomb-Scargle over the fake oscillator
    ftime, fsignal = read_synthetic_oscillator()

    # Remove NaN rows
    clean_ind, = np.where(~np.isnan(fsignal))
    ftime = ftime[clean_ind]
    fsignal = fsignal[clean_ind]

    ls_csv = os.path.join(data_dir, 'lomb_scargle', 'LS_synthetic.csv')
    if pathlib.Path(ls_csv).is_file():
        ls_df = pd.read_csv(ls_csv, delimiter=',',
                            float_precision='round_trip')

        ls_pgram = np.array(ls_df.ls_pgram)
        periods = np.array(ls_df.period_hr)

    else:
        print('Running LS analysis on synthetic oscillator')
        t1 = pd.Timestamp.now()
        print('starting LS at ', t1)
        ls_object, freqs, ls_pgram = lomb_scargle.generic_lomb_scargle(
            ftime, fsignal, f_min, f_max, n0=samples_per_peak)
        t2 = pd.Timestamp.now()
        print('LS finished, time elapsed: ', t2-t1)
        # Write to file
        periods = periodicity_functions.freq_to_period(freqs)
        ls_df = pd.DataFrame({'period_hr': periods,
                              'angular_freq': freqs,
                              'ls_pgram': ls_pgram})
        ls_df.to_csv(ls_csv, index=False)

    ax[0].plot(periods, ls_pgram, linewidth=1.5,
               color=freq_colors[0], label='Synthetic')
    ax[0].set_xscale('log')

    # Read in bootstrap
    ftime_cl, synthetic_BS = read_subset_bootstraps("synthetic",
                                                    n_bootstrap=n_bootstrap)
    # Read in peaks for bootstraps and FAP
    bootstrap_peak_magnitudes, FAP = lomb_scargle.false_alarm_probability(
        n_bootstrap, synthetic_BS, ftime_cl, f_min, f_max, FAP_peaks_dir,
        'synthetic', synthetic_FAP_pkl, n0=samples_per_peak)

    # Plot the FAP
    # ax[0].axhline(FAP, **FAP_fmt_dict, label='FAP')
    # Draw arrow for FAP
    trans = transforms.blended_transform_factory(ax[0].transData,
                                                 ax[0].transData)
    ax[0].annotate("FAP\n" + "{:.3e}".format(FAP), xy=(9., FAP), xytext=(8., FAP),
                   xycoords=trans, arrowprops={'facecolor': freq_colors[0]},
                   fontsize=fontsize, va='center', ha='right',
                   color=freq_colors[0],
                   bbox=annotate_bbox, fontweight="bold")

    # Formatting
    ax[0].set_ylabel('Lomb-Scargle\nNormalised Amplitude', fontsize=fontsize)
    ax[0].set_xlabel('Period (hours)', fontsize=fontsize)
    ax[0].tick_params(labelsize=fontsize)
    ax[0].legend(fontsize=fontsize, loc='upper left')

    if vertical_indicators != []:
        for h in vertical_indicators:
            trans = transforms.blended_transform_factory(ax[0].transData,
                                                         ax[0].transAxes)
            ax[0].annotate(str(h), xy=(h, 1.0), xytext=(h, 1.15),
                           xycoords=trans, arrowprops={'facecolor': 'black'},
                           fontsize=fontsize, va='top', ha='center',
                           color=vertical_ind_col)

    # breakpoint()

    for (i, interval_tag) in enumerate(interval_options['tag']):
        print('Running Lomb-Scargle for ', interval_tag)

        base_dir = pathlib.Path(data_dir) / 'lomb_scargle'
        file_paths = [base_dir / f"LS_{interval_tag}_{f}.csv" for f in freq_tags]
        file_checks = [file_path.is_file() for file_path in file_paths]

        if all(file_checks) is False:
            
            print('banana')
            akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)

        # Remove any rows where intensity == np.nan
        for (j, (freq_column, c, n)) in enumerate(zip(freq_tags,
                                                      freq_colors,
                                                      freq_labels)):
            print('Frequency band: ', freq_column)
            ls_csv = os.path.join(data_dir, 'lomb_scargle', 'LS_' +
                                  interval_tag + '_' + freq_column + '.csv')
            print('apple')
            if pathlib.Path(ls_csv).is_file() is False:
                print('pineapple')
                freq_df = akr_df.dropna(subset=[freq_column])
                t1 = pd.Timestamp.now()
                print('starting LS at ', t1)
                # ls_pgram = lomb_scargle.generic_lomb_scargle(freq_df.unix,
                #                                              freq_df[freq_column],
                #                                              angular_freqs)
                ls_object, freqs, ls_pgram = lomb_scargle.generic_lomb_scargle(
                    freq_df.unix, freq_df[freq_column], f_min, f_max, n0=samples_per_peak)
                t2 = pd.Timestamp.now()
                print('LS finished, time elapsed: ', t2-t1)
                # Write to file
                periods = periodicity_functions.freq_to_period(freqs)
                ls_df = pd.DataFrame({'period_hr': periods,
                                      'angular_freq': freqs,
                                      'ls_pgram': ls_pgram})
                ls_df.to_csv(ls_csv, index=False)
                t2 = pd.Timestamp.now()
                print('LS finished, time elapsed: ', t2-t1)
                # ls_df = pd.DataFrame({'period_hr': periods,
                #                       'freq_Hz': freqs,
                #                       'angular_freq': angular_freqs,
                #                       'ls_pgram': ls_pgram})
                ls_df.to_csv(ls_csv, index=False)

            else:
                print('guava')
                ls_df = pd.read_csv(ls_csv, delimiter=',',
                                    float_precision='round_trip')

                ls_pgram = np.array(ls_df.ls_pgram)
                periods = np.array(ls_df.period_hr)

            # Plot FAP here
            FAP_pkl = os.path.join(data_dir, "lomb_scargle",
                                   interval_tag + '_' + freq_column + "_FAP_" + str(n_bootstrap)
                                   + "_BSs.pkl")
            # Read in bootstrap
            ftime_cl, BS = read_subset_bootstraps(interval_tag,
                                                  freq_ch=freq_column,
                                                  n_bootstrap=n_bootstrap)
            # Read in/calc peak magnitudes for bootstraps and FAP
            print('tomato')
            # bootstrap_peak_magnitudes, FAP = lomb_scargle.false_alarm_probability(
            #     n_bootstrap, BS, ftime_cl, angular_freqs, FAP_peaks_dir,
            #     interval_tag + '_' + freq_column, FAP_pkl)
            bootstrap_peak_magnitudes, FAP = lomb_scargle.false_alarm_probability(
                n_bootstrap, BS, ftime_cl, f_min, f_max, FAP_peaks_dir,
                'synthetic', synthetic_FAP_pkl, n0=samples_per_peak)
            # f, a = plt.subplots()
            # a.hist(bootstrap_peak_magnitudes)
            # Plot the FAP
            #ax[i + 1].axhline(FAP, **FAP_fmt_dict, label='FAP')
            ax[i + 1].plot(periods, ls_pgram, linewidth=1.5, color=c, label=n)
            if j == 0:
                #ax[i + 1].plot(periods, ls_pgram, linewidth=1.5, color=c, label=n)
                
                trans = transforms.blended_transform_factory(ax[i + 1].transAxes,
                                                             ax[i + 1].transData)
                ax[i + 1].annotate("FAP\n" + "{:.3e}".format(FAP), xy=(0.2, FAP), xytext=(0.1, FAP),
                               xycoords=trans, arrowprops={'facecolor': c},
                               fontsize=fontsize, va='center', ha='right',
                               color=c,
                               bbox=annotate_bbox, fontweight="bold")
            elif j == 1:
                #twax = ax[i + 1].twinx()
                #twax.plot(periods, ls_pgram, linewidth=1.5, color=c, label=n)
                
                trans = transforms.blended_transform_factory(ax[i + 1].transAxes,
                                                             ax[i + 1].transData)
                ax[i + 1].annotate("FAP\n" + "{:.3e}".format(FAP), xy=(0.8, FAP), xytext=(0.9, FAP),
                               xycoords=trans, arrowprops={'facecolor': c},
                               fontsize=fontsize, va='center', ha='left',
                               color=c,
                               bbox=annotate_bbox, fontweight="bold")
            
            # ax[i + 1].axhline(np.percentile(bootstrap_peak_magnitudes, 99), **FAP_fmt_dict, label='FAPpc')

            # ax[i + 1] = lomb_scargle.plot_LS_summary(periods, ls_pgram,
            #                                          vertical_indicators=[12.,
            #                                                               24.],
            #                                          ax=ax[i+1])

            # breakpoint()

        ax[i + 1].set_xscale('log')

        

        # Formatting
        ax[i + 1].set_ylabel('Lomb-Scargle\nNormalised Amplitude', fontsize=fontsize)
        ax[i + 1].set_xlabel('Period (hours)', fontsize=fontsize)
        ax[i + 1].tick_params(labelsize=fontsize)
        ax[i + 1].legend(fontsize=fontsize, loc='upper left')




        if vertical_indicators != []:
            for h in vertical_indicators:
                trans = transforms.blended_transform_factory(
                    ax[i + 1].transData, ax[i + 1].transAxes)
                ax[i + 1].annotate(str(h), xy=(h, 1.0), xytext=(h, 1.15),
                            xycoords=trans, arrowprops={'facecolor': 'black'},
                            fontsize=fontsize, va='top', ha='center',
                            color=vertical_ind_col)

    # Label panels
    titles = np.append('Synthetic', interval_options.label)
    for (i, a) in enumerate(ax):
        
        t = a.text(0.005, 1.05, axes_labels[i], transform=a.transAxes,
                   fontsize=fontsize, va='bottom', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))
        
        # tit = a.text(1.025, 0.5, titles[i], transform=a.transAxes,
        #              fontsize=1.25 * fontsize, va='center', ha='center',
        #              rotation=-90.)
        tit = a.text(1.0, 1.05, titles[i], transform=a.transAxes,
                     fontsize=1.25 * fontsize, va='center', ha='right')
                

    # Adjust margins etc
    fig.tight_layout()

    # Save to file
    fig.savefig(LS_fig)


def run_ACF():

    # Plotting variables
    detrended_acf_col = 'blue'
    normalised_acf_col = 'black'
    shm_fit_col = 'mediumseagreen'
    ci_shade_alpha = 0.3
    acf_lw = 1.

    # Initialise variables
    temporal_resolution = 3. * 60.  # in seconds
    n_shifts = 5000
    tick_sep_hrs = 24.
    highlight_period = 24.
    highlight_fmt = {'color': 'grey',
                     'linestyle': 'dashed',
                     'linewidth': 1.}

    # Different frequency channels
    freq_tags = np.array(['ipwr_100_400kHz', 'ipwr_50_100kHz'])
    freq_labels = np.array(['100-400 kHz', '50-100 kHz'])
    freq_colors = np.array(['dimgrey', 'darkorange'])

    ACF_fig = os.path.join(fig_dir, "three_interval_ACF.png")

    # Read in interval data
    print('Reading AKR data over requested intervals')
    interval_options = read_and_tidy_data.return_test_intervals()

    # Initialise plotting window
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 17))

    # Run ACF over the fake oscillator
    ftime, fsignal = read_synthetic_oscillator()
    # Remove NaN rows
    clean_ind, = np.where(~np.isnan(fsignal))
    ftime = ftime[clean_ind]
    fsignal = fsignal[clean_ind]

    # Define output ACF data csv
    acf_csv = os.path.join(data_dir, 'acf', 'ACF_synthetic.csv')

    if (pathlib.Path(acf_csv).is_file()) is False:
        acf, lags = autocorrelation.autocorrelation(
            fsignal, n_shifts, temporal_resolution=temporal_resolution,
            starting_lag=7200)
        acf_df = pd.DataFrame({'acf': acf, 'lags': lags})
        acf_df.to_csv(acf_csv, index=False)
    else:
        acf_df = pd.read_csv(acf_csv, float_precision='round_trip')
        acf = acf_df['acf']
        lags = acf_df['lags']

    # Initialise ACF fit class
    synthetic_acf_fit = autocorrelation.decay_shm_fit(lags, acf)

    # Plot Synthetic ACF
    p1, = ax[0, 0].plot(synthetic_acf_fit.lags, synthetic_acf_fit.acf,
                        color=freq_colors[0], linewidth=2., label='ACF')

    # Initial guess parameters
    A0 = 1.0
    gamma0 = 1e-6
    omega0 = 2 * np.pi / 100000  # Guessing ~100,000 lag period
    phi0 = 0
    # Detrend, Normalise and Fit
    synthetic_acf_fit.fit_SHM(A0=A0, gamma0=gamma0, omega0=omega0, phi0=phi0)
    # Extract required text labels
    synthetic_acf_fit.create_text_labels()
    # Calculate confidence interval
    synthetic_acf_fit.calc_confidence_interval()

    # Plot line of best fit
    p2, = ax[0, 0].plot(synthetic_acf_fit.lags,
                        synthetic_acf_fit.linear_detrend_y,
                        color=detrended_acf_col, linestyle='dashed',
                        linewidth=2., label='Linear Trend')
    # Plot detrended ACF
    ax_detrend = ax[0, 0].twinx()
    p3, = ax_detrend.plot(synthetic_acf_fit.lags,
                          synthetic_acf_fit.linear_detrended_acf,
                          color=detrended_acf_col, linewidth=1.,
                          label='Detrended ACF')
    # Decor
    ax[0, 0].set_ylabel('ACF', fontsize=fontsize)
    ax_detrend.tick_params(axis='y', labelcolor=detrended_acf_col,
                           labelsize=fontsize)
    ax_detrend.spines['right'].set_color(detrended_acf_col)
    ax_detrend.set_ylabel('Detrended ACF', color=detrended_acf_col,
                          fontsize=fontsize)
    lines = [p1, p2, p3]
    ax_detrend.legend(lines, [l.get_label() for l in lines], fontsize=fontsize)
    ax_detrend.yaxis.offsetText.set_fontsize(fontsize)

    # Normalised ACF
    ax[0, 1].plot(synthetic_acf_fit.lags, synthetic_acf_fit.normalised_acf,
                  color=normalised_acf_col, linewidth=1.,
                  label='Normalised ACF')

    # Plotting fitted SHM
    ax[0, 1].plot(synthetic_acf_fit.lags, synthetic_acf_fit.y_fitted,
                  color=shm_fit_col, linewidth=2.,
                  linestyle='dashed', label='Decaying SHM fit')
    ax[0, 1].fill_between(synthetic_acf_fit.lags, synthetic_acf_fit.y_ci[:, 0],
                          synthetic_acf_fit.y_ci[:, 1], alpha=ci_shade_alpha,
                          color=shm_fit_col, label='95% CI')

    # Decor
    ax[0, 1].legend(fontsize=fontsize)
    ax[0, 1].set_ylabel('Normalised ACF Amplitude', fontsize=fontsize)

    return synthetic_acf_fit
    # Print fitting details into a table!!!!
    # Including some overall parameter for SHM fit, the linear fit, the SHM fit

    # Loop over datasets, calculating and plotting
    for (i, interval_tag) in enumerate(interval_options['tag']):
        print('Running autocorrelation for ', interval_tag)

        base_dir = pathlib.Path(data_dir) / 'acf'
        file_paths = [base_dir / f"ACF_{interval_tag}_{f}.csv" for f in freq_tags]
        file_checks = [file_path.is_file() for file_path in file_paths]

        # If no ACF calculations done, read integrated power data
        if all(file_checks) is False:
            akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)

        # Looping over frequency bands
        for (j, (freq_column, c, n)) in enumerate(zip(freq_tags,
                                                      freq_colors,
                                                      freq_labels)):
            print('Frequency band: ', freq_column)
            acf_csv = os.path.join(data_dir, 'acf', 'ACF_' +
                                   interval_tag + '_' + freq_column + '.csv')

            if pathlib.Path(acf_csv).is_file() is False:
                freq_df = akr_df.dropna(subset=[freq_column])
                t1 = pd.Timestamp.now()
                print('starting ACF at ', t1)
                acf, lags = autocorrelation.autocorrelation(
                    freq_df[freq_column], n_shifts,
                    temporal_resolution=temporal_resolution, starting_lag=7200)
                t2 = pd.Timestamp.now()
                print('ACF finished, time elapsed: ', t2-t1)
                acf_df = pd.DataFrame({'acf': acf, 'lags': lags})
                acf_df.to_csv(acf_csv, index=False)

            else:

                acf_df = pd.read_csv(acf_csv, delimiter=',',
                                     float_precision='round_trip')
                acf = acf_df['acf']
                lags = acf_df['lags']

            if freq_column == 'ipwr_50_100kHz':
                tax = ax[i + 1, 0].twinx()
                tax.plot(lags, acf, color=c, linewidth=acf_lw, label=n)
                tax.set_ylabel('ACF\n(' + n + ')', color=c, fontsize=fontsize)
                tax.tick_params(axis='y', labelcolor=(c), labelsize=fontsize)
                tax.spines['right'].set_color(c)
                tax.yaxis.offsetText.set_fontsize(fontsize)

                # Initialise ACF fit class
                LFE_acf_fit = autocorrelation.decay_shm_fit(lags, acf)
                # Initial guess parameters
                A0 = 1.0
                gamma0 = 1e-6
                omega0 = 2 * np.pi / 100000  # Guessing ~100,000 lag period
                phi0 = 0
                # Detrend, Normalise and Fit
                LFE_acf_fit.fit_SHM(A0=A0, gamma0=gamma0, omega0=omega0,
                                    phi0=phi0)
                # Extract required text labels
                LFE_acf_fit.create_text_labels()
                # Calculate confidence intervals on fit
                LFE_acf_fit.calc_confidence_interval()
                # Plotting fitted SHM
                ax[i + 1, 1].plot(LFE_acf_fit.lags, LFE_acf_fit.y_fitted,
                                  color=c, linewidth=2.,
                                  linestyle='dashed', label=n)
                ax[i + 1, 1].fill_between(LFE_acf_fit.lags,
                                          LFE_acf_fit.y_ci[:, 0],
                                          LFE_acf_fit.y_ci[:, 1],
                                          alpha=ci_shade_alpha, color=c,
                                          label='95% CI')

            else:
                ax[i + 1, 0].plot(lags, acf, color=c, linewidth=acf_lw,
                                  label=n)

                # Initialise ACF fit class
                mAKR_acf_fit = autocorrelation.decay_shm_fit(lags, acf)
                # Initial guess parameters
                A0 = 1.0
                gamma0 = 1e-6
                omega0 = 2 * np.pi / 100000  # Guessing ~100,000 lag period
                phi0 = 0
                # Detrend, Normalise and Fit
                mAKR_acf_fit.fit_SHM(A0=A0, gamma0=gamma0, omega0=omega0,
                                     phi0=phi0)
                # Extract required text labels
                mAKR_acf_fit.create_text_labels()
                # Calculate confidence intervals on fit
                mAKR_acf_fit.calc_confidence_interval()
                # Plotting fitted SHM
                ax[i + 1, 1].plot(mAKR_acf_fit.lags, mAKR_acf_fit.y_fitted,
                                  color=c, linewidth=2.,
                                  linestyle='dashed', label=n)
                ax[i + 1, 1].fill_between(mAKR_acf_fit.lags,
                                          mAKR_acf_fit.y_ci[:, 0],
                                          mAKR_acf_fit.y_ci[:, 1],
                                          alpha=ci_shade_alpha, color=c,
                                          label='95% CI')

    # Format all axes
    ii = 0
    titles = np.append('Synthetic', interval_options.label)
    for (j, axes) in enumerate(ax):
        for (k, a) in enumerate(axes):
            # Convert x ticks from seconds to readable format
            n_ticks = int(np.floor(np.max(a.get_xticks()) /
                                   (tick_sep_hrs * 60. * 60.)))
            tick_pos = []
            tick_str = []
            for i in range(n_ticks):
                tick_pos.append(i * (tick_sep_hrs * 60. * 60.))
                tick_str.append(str(int(tick_sep_hrs * i)))
            a.set_xticks(tick_pos, tick_str)

            if j < 1:
                a.set_ylabel('ACF', fontsize=fontsize)
            elif (j >= 1) & (k == 0):
                a.set_ylabel('ACF\n(' + freq_labels[0] + ')',
                             fontsize=fontsize)
                from matplotlib.lines import Line2D

                custom_lines = [Line2D([0], [0], color=freq_colors[0],
                                       lw=acf_lw),
                                Line2D([0], [0], color=freq_colors[1],
                                       lw=acf_lw)]
                a.legend(custom_lines, freq_labels, fontsize=fontsize,
                         loc='upper right')
            else:
                a.set_ylabel('Normalised ACF',
                             fontsize=fontsize)
                a.legend(fontsize=fontsize)

            a.set_xlabel('Lag (hours)', fontsize=fontsize)

            # Draw vertical lines each highlight period
            n_vert = int(np.floor((a.get_xlim()[1]) /
                                  (highlight_period * 60. * 60.)))

            for i in range(n_vert):
                a.axvline(((i + 1) * highlight_period) * (60. * 60.),
                          **highlight_fmt)

            # Formatting
            a.tick_params(labelsize=fontsize)
            a.yaxis.offsetText.set_fontsize(fontsize)
            t = a.text(0.02, 0.95, axes_labels[ii], transform=a.transAxes,
                       fontsize=1.5 * fontsize, va='top', ha='left')
            t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

            tit = a.text(0.5, 1.01, titles[j], transform=a.transAxes,
                         fontsize=1.25 * fontsize, va='bottom', ha='center')

            a.set_xlim(left=0., right=np.max(lags))

            ii = ii + 1

    fig.tight_layout()

    # Save to file
    # fig.savefig(ACF_fig)


def run_MLT_binning_overlayed(n_mlt_sectors='four'):


    if n_mlt_sectors=='four':
        # Initialise variables
        region_centres = [0, 6, 12, 18]
        region_width = 6
        region_names = ['midn', 'dawn', 'noon', 'dusk']
        region_mrkrs = ['o', '^', '*', 'x']
        region_flags = [0, 1, 2, 3]
        region_colors = ['grey', "#8d75ca", "#a39143", "#bb6c82"]
    elif n_mlt_sectors=='eight':
        # Initialise variables
        region_centres = [0, 3, 6, 9, 12, 15, 18, 21]
        region_width = 3
        region_names = ['0', '3', '6', '9', '12', '15', '18', '21']
        region_mrkrs = ['o', '^', '*', 'x', '+', 'X', "2", "."]
        region_flags = [0, 1, 2, 3, 4, 5, 6, 7]
        region_colors = ['grey', "#c18b40", "#a361c7", "#7ca343", "#6587cd",
                         "#cc5643", "#49ae8a", "#c65c8a"]       

    lon_bin_width = 30.

    # Different frequency channels
    freq_tags = np.array(['ipwr_100_400kHz', 'ipwr_50_100kHz'  # ,
                          #'ipwr_100_650kHz'
                          ])
    freq_labels = np.array(['100-400 kHz', '50-100 kHz'])
    freq_colors = np.array(['dimgrey', 'darkorange', 'rebeccapurple'])

    MLT_fig = os.path.join(fig_dir, "three_interval_MLT_binned_"+n_mlt_sectors+"MLT.png")

    # Read in interval data
    print('Reading AKR data over requested intervals')
    interval_options = read_and_tidy_data.return_test_intervals()

    # Initialise plotting window
    fig, ax = plt.subplots(nrows=interval_options['tag'].size,
                           ncols=freq_tags.size, figsize=(21, 13))

    # # Run ACF over the fake oscillator
    # ftime, fsignal = read_synthetic_oscillator()
    # # Remove NaN rows
    # clean_ind, = np.where(~np.isnan(fsignal))
    # ftime = ftime[clean_ind]
    # fsignal = fsignal[clean_ind]


    # acf_csv = os.path.join(data_dir, 'acf', 'ACF_synthetic.csv')

    # if (pathlib.Path(acf_csv).is_file()) is False:
    #     acf, lags = autocorrelation.autocorrelation(
    #         fsignal, n_shifts, temporal_resolution=temporal_resolution,
    #         starting_lag=7200)
    #     acf_df = pd.DataFrame({'acf': acf, 'lags': lags})
    #     acf_df.to_csv(acf_csv, index=False)
    # else:
    #     acf_df = pd.read_csv(acf_csv, float_precision='round_trip')
    #     acf = acf_df['acf']
    #     lags = acf_df['lags']

    # ax[0].plot(lags, acf, color=freq_colors[0], linewidth=1.)

    # # DO I NEED TO DROP NANS???


    abc_label_counter = 0
    for (i, interval_tag) in enumerate(interval_options['tag']):
        print('Running autocorrelation for ', interval_tag)

        base_dir = pathlib.Path(data_dir) / 'MLT_binning'
        file_paths = [base_dir / f"MLT_binned_{n_mlt_sectors}sector_{interval_tag}_{f}.csv" for f in freq_tags]
        file_checks = [file_path.is_file() for file_path in file_paths]

        if all(file_checks) is False:
            # Read in AKR intensity data
            akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)
            # akr_df['lon_sol'] = utility.calc_longitude_of_sun(akr_df)
            mlt_flag, mlt_name = binning_averaging.calc_LT_flag(
                akr_df, region_centres=region_centres,
                region_width=region_width, region_names=region_names,
                region_flags=region_flags)
            akr_df['mlt_flag'] = mlt_flag
            akr_df['mlt_name'] = mlt_name


        # Remove any rows where intensity == np.nan
        for (j, (freq_column, c, n)) in enumerate(zip(freq_tags, freq_colors, freq_labels)):
            print('Frequency band: ', freq_column)
            MLT_csv = os.path.join(data_dir, 'MLT_binning', 'MLT_binned_' + n_mlt_sectors +'sector_' +
                                  interval_tag + '_' + freq_column + '.csv')

            if pathlib.Path(MLT_csv).is_file() is False:

                freq_df = akr_df.dropna(subset=[freq_column])
                t1 = pd.Timestamp.now()
                print('starting MLT binning at ', t1)    
                
                UT_df =[]
                # UT_df = binning_averaging.return_UT_trend(
                #         akr_df, region_centres=region_centres,
                #         region_width=region_width, region_names=region_names,
                #         region_flags=region_flags, UT_bin_width=UT_bin_width,
                #         ipower_tag=freq_column)
                lon_df = binning_averaging.return_lon_trend(akr_df, region_centres=region_centres,
                        region_width=region_width, region_names=region_names,
                        region_flags=region_flags,
                        lon_bin_width=30.,
                                    ipower_tag=freq_column,
                                    lon_sol_tag="lon_sol", lon_sc_tag="lon_gsm")
                breakpoint()
                t2 = pd.Timestamp.now()
                print('MLT binning finished, time elapsed: ', t2-t1)
                #UT_df.to_csv(MLT_csv, index=False)                
            else:

                UT_df = pd.read_csv(MLT_csv, delimiter=',',
                                    float_precision='round_trip')
    


            if freq_column == 'ipwr_100_400kHz':
                for k, (MLT_n, c, mrkr) in enumerate(zip(region_names, region_colors,
                                                   region_mrkrs)):
                    ax[i, 0].plot(UT_df.UT_bin_centre, UT_df[MLT_n + '_median_norm_no0'],
                                  color=c, label=MLT_n, marker=mrkr,
                                  markersize=fontsize)
                    ax[i, 0].fill_between(UT_df.UT_bin_centre,
                                          UT_df[MLT_n + '_median_norm_no0'] -
                                          UT_df[MLT_n + '_mad_norm_no0'],
                                          UT_df[MLT_n + '_median_norm_no0'] +
                                          UT_df[MLT_n + '_mad_norm_no0'],
                                          color=c, alpha=0.2)


            elif freq_column == 'ipwr_50_100kHz':
                for k, (MLT_n, c, mrkr) in enumerate(zip(region_names, region_colors,
                                                   region_mrkrs)):
                    ax[i, 1].plot(UT_df.UT_bin_centre, UT_df[MLT_n + '_median_norm_no0'],
                                  color=c, label=MLT_n, marker=mrkr,
                                  markersize=fontsize)
                    ax[i, 1].fill_between(UT_df.UT_bin_centre,
                                          UT_df[MLT_n + '_median_norm_no0'] -
                                          UT_df[MLT_n + '_mad_norm_no0'],
                                          UT_df[MLT_n + '_median_norm_no0'] +
                                          UT_df[MLT_n + '_mad_norm_no0'],
                                          color=c, alpha=0.2)

            # Formatting
            ax[i, j].text(0.5, 1.01,
                          interval_options.label[i] + ' (' + n + ')',
                          transform=ax[i, j].transAxes,
                          fontsize=1.25 * fontsize, va='bottom', ha='center')
            ax[i, j].set_ylabel('Normalised median integrated power', fontsize=fontsize)
            ax[i, j].set_xlabel('UT (hours)', fontsize=fontsize)
            ax[i, j].set_xlim(left=0., right=24.)
            ax[i, j].tick_params(labelsize=fontsize)
            ax[i, j].legend(fontsize=fontsize, loc='best')
            t = ax[i, j].text(0.02, 0.95, axes_labels[abc_label_counter],
                              transform=ax[i, j].transAxes, fontsize=fontsize,
                              va='top', ha='left')
            t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))
            abc_label_counter = abc_label_counter + 1

    fig.tight_layout()

    # Save to file
    fig.savefig(MLT_fig)


def run_MLT_binning_seperate(n_mlt_sectors='four'):

    fontsize = 20

    if n_mlt_sectors == 'four':
        # Initialise variables
        region_centres = [0, 6, 12, 18]
        region_width = 6
        region_names = ['midn', 'dawn', 'noon', 'dusk']
        region_mrkrs = ['o', '^', '*', 'x']
        region_flags = [0, 1, 2, 3]
        region_colors = ['grey', "#8d75ca", "#a39143", "#bb6c82"]
    elif n_mlt_sectors == 'eight':
        # Initialise variables
        region_centres = [0, 3, 6, 9, 12, 15, 18, 21]
        region_width = 3
        region_names = ['0', '3', '6', '9', '12', '15', '18', '21']
        region_mrkrs = ['o', '^', '*', 'x', '+', 'X', "2", "."]
        region_flags = [0, 1, 2, 3, 4, 5, 6, 7]
        region_colors = ['grey', "#c18b40", "#a361c7", "#7ca343", "#6587cd",
                         "#cc5643", "#49ae8a", "#c65c8a"]

    # UT bin is the same always
    # UT_bin_width = 2
    lon_bin_width = 30.

    # Different frequency channels
    freq_tags = np.array(['ipwr_100_400kHz', 'ipwr_50_100kHz'])
    freq_labels = np.array(['100-400 kHz', '50-100 kHz'])
    freq_colors = np.array(['dimgrey', 'darkorange', 'rebeccapurple'])

    # Read in interval data
    print('Reading AKR data over requested intervals')
    interval_options = read_and_tidy_data.return_test_intervals()

    # Loop through different intervals / datasets
    for (i, interval_tag) in enumerate(interval_options['tag']):
        print('Running autocorrelation for ', interval_tag)

        base_dir = pathlib.Path(data_dir) / 'MLT_binning'
        file_paths = [base_dir /
                      f"MLT_lon_binned_{n_mlt_sectors}sector_{interval_tag}_{f}.csv"
                      for f in freq_tags]
        file_checks = [file_path.is_file() for file_path in file_paths]

        if all(file_checks) is False:
            # Read in AKR intensity data
            akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)
            akr_df['lon_sol'] = utility.calc_longitude_of_sun(akr_df, plot=True)
            mlt_flag, mlt_name = binning_averaging.calc_LT_flag(
                akr_df, region_centres=region_centres,
                region_width=region_width, region_names=region_names,
                region_flags=region_flags)
            akr_df['mlt_flag'] = mlt_flag
            akr_df['mlt_name'] = mlt_name

        # Loop through different frequency columns
        for (j, (freq_column, c, n)) in enumerate(zip(freq_tags,
                                                      freq_colors,
                                                      freq_labels)):
            print('Frequency band: ', freq_column)
            MLT_csv = os.path.join(data_dir, 'MLT_binning',
                                   'MLT_lon_binned_' + n_mlt_sectors + 'sector_' +
                                   interval_tag + '_' + freq_column + '.csv')

            # Initialise plotting window
            fig, ax = plt.subplots(nrows=len(region_centres), ncols=1,
                                   figsize=(15, len(region_centres) * 3.))
            fig_name = os.path.join(fig_dir, interval_tag + "_" + freq_column
                                    + "_" + n_mlt_sectors + "MLT.png")

            # Bin data into UT bins
            if pathlib.Path(MLT_csv).is_file() is False:

                # freq_df = akr_df.dropna(subset=[freq_column])
                # t1 = pd.Timestamp.now()
                # print('starting MLT binning at ', t1)
                # UT_df = binning_averaging.return_UT_trend(
                #         akr_df, region_centres=region_centres,
                #         region_width=region_width, region_names=region_names,
                #         region_flags=region_flags, UT_bin_width=UT_bin_width,
                #         ipower_tag=freq_column)
                lon_df = binning_averaging.return_lon_trend(
                    akr_df, region_centres=region_centres,
                    region_width=region_width, region_names=region_names,
                    region_flags=region_flags, lon_bin_width=30.,
                    ipower_tag=freq_column,
                    lon_sol_tag="lon_sol", lon_sc_tag="lon_gsm")
                breakpoint()
                UT_df=[]
                # t2 = pd.Timestamp.now()
                # print('MLT binning finished, time elapsed: ', t2-t1)
                UT_df.to_csv(MLT_csv, index=False)
            # Else read in pre-sorted data
            else:
                UT_df = pd.read_csv(MLT_csv, delimiter=',',
                                    float_precision='round_trip')

            # Loop through each MLT sector, plotting
            for k, (MLT_n, c, mrkr) in enumerate(zip(region_names,
                                                     region_colors,
                                                     region_mrkrs)):
                # Set x axis limits
                ax[k].set_xlim(left=0., right=24.)

                # Plot number of observations
                ax[k].bar(UT_df.UT_bin_centre, UT_df[MLT_n + "n"],
                          zorder=0.5, color='gold', linewidth=.75,
                          edgecolor='black', alpha=0.75, label='# all')
                ax[k].bar(UT_df.UT_bin_centre, UT_df[MLT_n + "n_no0"],
                          zorder=0.6, color='deeppink', linewidth=.75,
                          edgecolor='black', alpha=0.75, label="# > 0")

                # Create twin axis
                twax = ax[k].twinx()

                # Plot intensity trend
                twax.plot(UT_df.UT_bin_centre,
                          UT_df[MLT_n + '_median_norm_no0'],
                          color='black', label="median", marker='o',
                          markersize=fontsize, linewidth=1.5, zorder=2.5)
                twax.fill_between(UT_df.UT_bin_centre,
                                  UT_df[MLT_n + '_median_norm_no0'] -
                                  UT_df[MLT_n + '_mad_norm_no0'],
                                  UT_df[MLT_n + '_median_norm_no0'] +
                                  UT_df[MLT_n + '_mad_norm_no0'],
                                  color='grey', alpha=0.5, label="MAD",
                                  zorder=2)

                # Formatting
                # Panel MLT sector
                t = ax[k].text(0.98, 0.95, MLT_n + ' MLT',
                               transform=ax[k].transAxes, fontsize=fontsize,
                               va='top', ha='right')
                t.set_bbox(dict(facecolor='white', alpha=0.75,
                                edgecolor='grey'))
                # Y labels
                ax[k].set_ylabel("# averaged", fontsize=fontsize)
                twax.set_ylabel('Normalised median\nintegrated power',
                                fontsize=fontsize)
                # Fontsize
                ax[k].tick_params(labelsize=fontsize)
                # Panel label
                t = ax[k].text(0.02, 0.95, axes_labels[k],
                               transform=ax[k].transAxes, fontsize=fontsize,
                               va='top', ha='left')
                t.set_bbox(dict(facecolor='white', alpha=0.75,
                                edgecolor='grey'))

            # More formatting
            ax[0].text(0.5, 1.01,
                       interval_options.label[i] + ' (' + n + ')',
                       transform=ax[0].transAxes,
                       fontsize=1.25 * fontsize, va='bottom', ha='center')
            leg_ln = [*ax[k].get_legend_handles_labels()[0],
                      *twax.get_legend_handles_labels()[0]]
            leg_lab = [*ax[k].get_legend_handles_labels()[1],
                       *twax.get_legend_handles_labels()[1]]
            # ax[0].legend(leg_ln, leg_lab, fontsize=0.65*fontsize,
            #              loc='lower center', ncol=3)
            ax[0].legend(leg_ln, leg_lab, bbox_to_anchor=(1.13, 0.0, 0.2, 1.0),
                         loc="center left", fontsize=fontsize)
            ax[k].set_xlabel('UT (hours)', fontsize=fontsize)

            fig.tight_layout()

            # Save to file
            fig.savefig(fig_name)
            #return

   
def lomb_scargle_cassini():
    
    # Run the Lomb-scargle analysis over each year of AKR intensity
    # with sunspot number as another panel
    
    # Define time periods
    years = np.arange(1995, 2004 + 1)
    # stime = pd.Timestamp(years[0], 1, 1, 0, 0, 0)
    # etime = pd.Timestamp(years[-1]+1, 1, 1, 0, 0, 0)
    stime = 1994.5
    etime = 2004.5
    
    # Define Lomb-Scargle freqs etc
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    T = (pd.Timestamp(2005, 1, 1, 0) -
         pd.Timestamp(1995, 1, 1, 0)).total_seconds()
    samples_per_peak = 5
    # f_min = 1 / (48. * 60. * 60.)
    # f_max = 1 / (8. * 60. * 60.)
    # T = (pd.Timestamp(2005, 1, 1, 0) - pd.Timestamp(1995, 1, 1, 0)).total_seconds()
    # f_min, f_max, N_f, freqs = lomb_scargle.define_frequency_bins(T, f_min, f_max, n0=5)
    # freqs = freqs[::-1]
    # angular_freqs = 2 * np.pi * freqs
    # periods = periodicity_functions.freq_to_period(freqs)    
    
    # Read in interval data
    interval_options = read_and_tidy_data.return_test_intervals()
    interval_details = interval_options.loc[
        interval_options.tag == "cassini_flyby"]
    interval_stime = interval_details.stime.iloc[0]
    interval_etime = interval_details.etime.iloc[0]
    # Read in *all* AKR integrated power
    akr_df = read_and_tidy_data.select_akr_intervals("full_archive")
    
    # Sliding parameters
    slides = 2
    slide_width = pd.Timedelta(days=10)
    slide_width_multiplier = np.linspace(0, slides, slides+1) - slides/2

    ut_s = np.full(slides + 1, np.nan)
    ut_e = np.full(slides + 1, np.nan)

    x_lim = [interval_stime - ((slides / 2) * slide_width),
             interval_etime + ((slides / 2) * slide_width)]

    for i, factor in enumerate(slide_width_multiplier):
        ut_s[i] = interval_stime - (factor * slide_width)
        ut_e[i] = interval_etime - (factor * slide_width)

   #  # LS analysis here
   #  ls_pgram = np.full((periods.size, slides + 1), np.nan)
   #  for i, yr in enumerate(slides + 1):
   #      print('Running Lomb-Scargle analysis for ', yr)
        
   #      # Subselect AKR df
    
   #      # Run Lomb-Scargle
        
   #      # Current placeholder data
   #      ls_pgram[:, i] = np.repeat(i, periods.size)
    
   #  # Find edges of pixels on period axis
   #  period_edges = np.full(periods.size + 1, np.nan)
   #  for k in range(period_edges.size):
   #      if k == period_edges.size-2:
   #          period_edges[k] = periods[k-1] + ((periods[k]-periods[k-1])/2)
   #      elif k == period_edges.size-1:
   #          period_edges[k] = periods[k-1] + ((periods[k-1]-periods[k-2])/2)
   #      else:
   #          period_edges[k] = periods[k] - ((periods[k+1]-periods[k])/2)
    
   #  # Perhaps a panel with discreet years and another with some smoothing?
   #  # IF TIME
    
   # # breakpoint()
    
    
   #  # Initialise plotting
   #  fig, ax = plt.subplots(nrows=2, figsize=(16,8))
   
   #  # Plot Lomb-Scargles
   #  X, Y = np.meshgrid(np.append(years-0.5, years[-1]+0.5), period_edges)
   #  pcm = ax[0].pcolormesh(X, Y, ls_pgram, cmap='plasma')
   #  cbar = plt.colorbar(pcm, ax=ax[0], label="Lomb-Scargle\nNormalised Amplitude")
   #  cbar.ax.tick_params(labelsize=fontsize)
   
   #  ax[0].set_ylabel('Period (hours)', fontsize=fontsize)

   #  ax_pos = ax[0].get_position().bounds




   #  # Make width same as (a)
   #  pos=ax[1].get_position().bounds
   #  ax[1].set_position([ax_pos[0], pos[1], ax_pos[2], pos[3]])
    
    
   #  # Formatting
   #  for j, a in enumerate(ax):
   #      a.set_xlim(stime, etime)
   #      a.tick_params(labelsize=fontsize)

   #      t = a.text(0.02, 0.92, axes_labels[j], transform=a.transAxes,
   #                 fontsize=fontsize, va='top', ha='left')
   #      t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))
                






































    
def LS_solar_cyc(sunspot_n_fdict={'color': 'lightgrey',
                                  'label': 'Mean',
                                  'linewidth': 1.},
                 smoothsn_fdict={'color': 'black',
                                 'label': '13-month Smoothed'}):
    
    # Run the Lomb-scargle analysis over each year of AKR intensity
    # with sunspot number as another panel
    
    # Define time periods
    years = np.arange(1995, 2004 + 1)
    # stime = pd.Timestamp(years[0], 1, 1, 0, 0, 0)
    # etime = pd.Timestamp(years[-1]+1, 1, 1, 0, 0, 0)
    stime = 1994.5
    etime = 2004.5
    
    # Define Lomb-Scargle freqs etc
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    T = (pd.Timestamp(2005, 1, 1, 0) - pd.Timestamp(1995, 1, 1, 0)).total_seconds()
    f_min, f_max, N_f, freqs = lomb_scargle.define_frequency_bins(T, f_min, f_max, n0=5)
    freqs = freqs[::-1]
    angular_freqs = 2 * np.pi * freqs
    periods = periodicity_functions.freq_to_period(freqs)    
    
    # Read in Sunspot Number
    sunspot_df = read_sunspot_n.read_monthly_sunspot()
    smoothed_sunspot_df = read_sunspot_n.read_monthly_smoothed_sunspot()
    
    # Read in AKR integrated power
        
    # LS analysis here
    ls_pgram = np.full((periods.size, years.size), np.nan)
    for i, yr in enumerate(years):
        print('Running Lomb-Scargle analysis for ', yr)
        
        # Subselect AKR df
    
        # Run Lomb-Scargle
        
        # Current placeholder data
        ls_pgram[:, i] = np.repeat(i, periods.size)
    
    # Find edges of pixels on period axis
    period_edges = np.full(periods.size + 1, np.nan)
    for k in range(period_edges.size):
        if k == period_edges.size-2:
            period_edges[k] = periods[k-1] + ((periods[k]-periods[k-1])/2)
        elif k == period_edges.size-1:
            period_edges[k] = periods[k-1] + ((periods[k-1]-periods[k-2])/2)
        else:
            period_edges[k] = periods[k] - ((periods[k+1]-periods[k])/2)
    
    # Perhaps a panel with discreet years and another with some smoothing?
    # IF TIME
    
   # breakpoint()
    
    
    # Initialise plotting
    fig, ax = plt.subplots(nrows=2, figsize=(16,8))
   
    # Plot Lomb-Scargles
    X, Y = np.meshgrid(np.append(years-0.5, years[-1]+0.5), period_edges)
    pcm = ax[0].pcolormesh(X, Y, ls_pgram, cmap='plasma')
    cbar = plt.colorbar(pcm, ax=ax[0], label="Lomb-Scargle\nNormalised Amplitude")
    cbar.ax.tick_params(labelsize=fontsize)
   
    ax[0].set_ylabel('Period (hours)', fontsize=fontsize)

    ax_pos = ax[0].get_position().bounds

    # Plot Sunspot Number
    ax[1].plot(sunspot_df.year_frac, sunspot_df.mean_s_n, **sunspot_n_fdict)
    ax[1].plot(smoothed_sunspot_df.year_frac, smoothed_sunspot_df.smooth_s_n, **smoothsn_fdict)

    # Set limits
    sunspot_ymax = np.max(
        [sunspot_df.loc[(sunspot_df.year_frac >= stime) & (sunspot_df.year_frac <= etime), 'mean_s_n'].max(),
        smoothed_sunspot_df.loc[(smoothed_sunspot_df.year_frac >= stime) & (smoothed_sunspot_df.year_frac <= etime), 'smooth_s_n'].max()])
    ax[1].set_ylim(0, 1.1 * sunspot_ymax)
    ax[1].set_xlim(stime, etime)
    
    ax[1].set_ylabel('Sunspot Number', fontsize=fontsize)
    ax[1].set_xlabel('Year', fontsize=fontsize)
    ax[1].legend(fontsize=fontsize, loc='upper right')
    # Make width same as (a)
    pos=ax[1].get_position().bounds
    ax[1].set_position([ax_pos[0], pos[1], ax_pos[2], pos[3]])
    
    
    # Formatting
    for j, a in enumerate(ax):
        a.set_xlim(stime, etime)
        a.tick_params(labelsize=fontsize)

        t = a.text(0.02, 0.92, axes_labels[j], transform=a.transAxes,
                   fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))
                








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

# def DEPRECATED_run_feature_importance():

#     interval_options = read_and_tidy_data.return_test_intervals()
#     # intervals = np.array(['full_archive', 'cassini_flyby'])
#     intervals = np.array(interval_options.tag)

#     fft_signal_x_start = np.array([pd.Timestamp(1999, 8, 18, 0).timestamp()])
#     fft_signal_x_width = np.array([5. * 24. * 60. * 60.])

#     for i in range(intervals.size):
#         print('Running analyses for ', intervals[i])
#         combined_rounded_df = read_and_tidy_data.\
#             combine_rounded_akr_omni(intervals[i])

#         # -- FEATURE IMPORTANCE --
#         loc_fi_png = os.path.join(fig_dir, intervals[i] + '_loc_fi.png')
#         geo_fi_png = os.path.join(fig_dir, intervals[i] + '_geo_fi.png')
#         all_fi_png = os.path.join(fig_dir, intervals[i] + '_all_fi.png')
#         combined_fi_png = os.path.join(fig_dir, intervals[i] +
#                                        '_combined_fi.png')

#         location_fi_csv = os.path.join(data_dir, intervals[i] +
#                                        '_location_only_fidata.csv')
#         geophysical_fi_csv = os.path.join(data_dir, intervals[i] +
#                                           '_geophysical_only_fidata.csv')
#         all_fi_csv = os.path.join(data_dir, intervals[i] +
#                                   '_loc_geophysical_fidata.csv')

#         if (pathlib.Path(loc_fi_png).is_file() is False) | (pathlib.Path(geo_fi_png).is_file() is False) | (pathlib.Path(all_fi_png).is_file() is False) | (pathlib.Path(combined_fi_png).is_file() is False):

#             # Location Feature Importance
#             loc_feature_name = np.array(['LT', 'Latitude', 'Radial Distance'])
#             if pathlib.Path(location_fi_csv).is_file() is False:
#                 loc_fig, loc_ax, loc_importance = feature_importance.\
#                     plot_feature_importance(
#                         np.array(combined_rounded_df.integrated_power),
#                         np.array(combined_rounded_df[['decimal_gseLT',
#                                                       'lat_gse', 'lon_gse']]),
#                         feature_names=loc_feature_name,
#                         seed=1993, fontsize=20, record_number=True)
#                 location_fi_df = pd.DataFrame({
#                     'feature_name': loc_feature_name,
#                     'importance': loc_importance})
#                 location_fi_df.to_csv(location_fi_csv, index=False)

#             else:
#                 location_fi_df = pd.read_csv(location_fi_csv, delimiter=',',
#                                              float_precision='round_trip')
#                 loc_fig, loc_ax, importance = feature_importance.\
#                     plot_feature_importance(
#                         np.array(combined_rounded_df.integrated_power),
#                         np.array(combined_rounded_df[['decimal_gseLT',
#                                                       'lat_gse', 'lon_gse']]),
#                         importance=np.array(location_fi_df.importance),
#                         feature_names=loc_feature_name,
#                         seed=1993, fontsize=20, record_number=True)
#             loc_fig.savefig(loc_fi_png, bbox_inches='tight')

#             # Geophysical Feature Importance
#             geo_feature_name = np.array(['Bx', 'By', 'Bz', 'Bt',
#                                          'ClockAngle', 'Vsw', 'Nsw',
#                                          'Psw', 'AE', 'AL', 'AU', 'SYM-H',
#                                          'SME', 'SMU', 'SML', 'SMR'])
#             geo_feature_tag = np.array(['bx', 'by_gsm', 'bz_gsm', 'b_total',
#                                         'clock_angle', 'flow_speed',
#                                         'proton_density', 'flow_pressure',
#                                         'ae', 'al', 'au', 'symh',
#                                         'SME', 'SMU', 'SML', 'SMR'])
#             geo_df = combined_rounded_df.dropna(subset=geo_feature_tag
#                                                 ).reset_index(drop=True)
#             if pathlib.Path(geophysical_fi_csv).is_file() is False:
#                 geo_fig, geo_ax, geo_importance = feature_importance.\
#                     plot_feature_importance(np.array(geo_df.integrated_power),
#                                             np.array(geo_df[geo_feature_tag]),
#                                             feature_names=geo_feature_name,
#                                             seed=1993, fontsize=20,
#                                             record_number=True)
#                 geophysical_fi_df = pd.DataFrame({
#                     'feature_name': geo_feature_name,
#                     'importance': geo_importance})
#                 geophysical_fi_df.to_csv(geophysical_fi_csv, index=False)
#             else:
#                 geophysical_fi_df = pd.read_csv(geophysical_fi_csv,
#                                                 delimiter=',',
#                                                 float_precision='round_trip')
#                 geo_fig, geo_ax, geo_importance = feature_importance.\
#                     plot_feature_importance(
#                         np.array(geo_df.integrated_power),
#                         np.array(geo_df[geo_feature_tag]),
#                         importance=np.array(geophysical_fi_df.importance),
#                         feature_names=geo_feature_name,
#                         seed=1993, fontsize=20, record_number=True)
#             geo_fig.savefig(geo_fi_png, bbox_inches='tight')

#             # All Features
#             all_feature_name = np.array(['LT', 'Latitude', 'Radial Distance',
#                                          'Bx', 'By', 'Bz', 'Bt',
#                                          'ClockAngle', 'Vsw', 'Nsw',
#                                          'Psw', 'AE', 'AL', 'AU', 'SYM-H',
#                                          'SME', 'SMU', 'SML', 'SMR'])
#             all_feature_tag = np.array(['decimal_gseLT', 'lat_gse', 'lon_gse',
#                                         'bx', 'by_gsm', 'bz_gsm', 'b_total',
#                                         'clock_angle', 'flow_speed',
#                                         'proton_density', 'flow_pressure',
#                                         'ae', 'al', 'au', 'symh',
#                                         'SME', 'SMU', 'SML', 'SMR'])
#             all_df = combined_rounded_df.dropna(subset=all_feature_tag
#                                                 ).reset_index(drop=True)
#             if pathlib.Path(all_fi_csv).is_file() is False:
#                 all_fig, all_ax, all_importance = feature_importance.\
#                     plot_feature_importance(np.array(all_df.integrated_power),
#                                             np.array(all_df[all_feature_tag]),
#                                             feature_names=all_feature_name,
#                                             seed=1993, fontsize=20,
#                                             record_number=True)
#                 all_fi_df = pd.DataFrame({
#                     'feature_name': all_feature_name,
#                     'importance': all_importance})
#                 all_fi_df.to_csv(all_fi_csv, index=False)
#             else:
#                 all_fi_df = pd.read_csv(all_fi_csv,
#                                         delimiter=',',
#                                         float_precision='round_trip')
#                 all_fig, all_ax, geo_importance = feature_importance.\
#                     plot_feature_importance(
#                         np.array(all_df.integrated_power),
#                         np.array(all_df[all_feature_tag]),
#                         importance=np.array(all_fi_df.importance),
#                         feature_names=all_feature_name,
#                         seed=1993, fontsize=20, record_number=True)
#             all_fig.savefig(all_fi_png, bbox_inches='tight')

#             # Combined features panel
#             combined_fig, combined_ax = feature_importance.\
#                 feature_importance_3panel(
#                     loc_feature_name, geo_feature_name, all_feature_name,
#                     np.array(location_fi_df.importance),
#                     np.array(geophysical_fi_df.importance),
#                     np.array(all_fi_df.importance), titles=[
#                         '(a) Visibility', '(b) Geophysical',
#                         '(c) Visibility and Geophysical'])
#             combined_fig.savefig(combined_fi_png, bbox_inches='tight')
#         # -- END FEATURE IMPORTANCE --
        
#         # -- BINNING / AVERAGING --
#         binned_median_png = os.path.join(fig_dir, intervals[i] + '_MLT_UT_binning_median.png')
#         binned_boxplot_png = os.path.join(fig_dir, intervals[i] + '_MLT_UT_binning_boxplot.png')

#         if (pathlib.Path(binned_median_png).is_file() is False) | (pathlib.Path(binned_boxplot_png).is_file() is False):
#             fig_m, fig_b = binning_averaging.plot_UT_trend(combined_rounded_df)
        
#             fig_m.savefig(binned_median_png, bbox_inches='tight')
#             fig_b.savefig(binned_boxplot_png, bbox_inches='tight')
        
#     return


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
    
    
    