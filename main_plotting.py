# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:15:12 2024

@author: A R Fogg
"""

import sys
import os
import pathlib
import string
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.transforms as transforms

from numpy.fft import fft, ifft
from matplotlib.gridspec import GridSpec
from neurodsp.sim import sim_oscillation
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm


import periodicity_functions
import read_and_tidy_data
import binning_averaging
import wind_location
import diurnal_oscillator
import lomb_scargle
import autocorrelation
import bootstrap_functions
import utility

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\wind_utility')
import read_wind_position
#import read_sunspot_n

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\readers')
import read_omni
import read_supermag
import read_sunspot_n

fontsize = 18
alphabet = list(string.ascii_lowercase)
axes_labels = []
for a in alphabet:
    axes_labels.append('(' + a + ')')

fig_dir = os.path.join("C:" + os.sep,
                       r"Users\Alexandra\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")


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
        t = ax.text(0.07, 0.93, axes_labels[i], transform=ax.transAxes,
                    fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

        # LT histogram
        draw_ticks = True if i == 2 else False
        if i == 2:
            half_magnitude = True
            # leg_lab = interval_options.label.iloc[i] + "\n(* 50%)"
            
            leg_lab = interval_options.label.iloc[i].split()[0] + "\n" + interval_options.label.iloc[i].split()[1] + "\n(* 50%)"
        else:
            half_magnitude = False
            # leg_lab = interval_options.label.iloc[i]
            leg_lab = interval_options.label.iloc[i].split()[0] + "\n" + interval_options.label.iloc[i].split()[1]
        axes.reshape(-1)[-1] = wind_location.lt_hist(
            interval_options.stime.iloc[i], interval_options.etime.iloc[i],
            wind_position_df, axes.reshape(-1)[-1],
            bar_fmt={'color': interval_options.color.iloc[i],
                     'edgecolor': 'black', 'alpha': 0.4,
                     'label': leg_lab},
            draw_ticks=draw_ticks, half_magnitude=half_magnitude,
            fontsize=fontsize)

    t = axes.reshape(-1)[-1].text(0.05, 0.95, axes_labels[3],
                                  transform=axes.reshape(-1)[-1].transAxes,
                                  fontsize=fontsize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    fig.tight_layout()

    traj_fig = os.path.join(fig_dir, "three_interval_traj.png")
    fig.savefig(traj_fig)

def geomag_ac_plots():
    
    interval_options = read_and_tidy_data.return_test_intervals()

    supermag_df = read_supermag.concat_indices_years(
        np.array(range(1995, 2005)))    
    
    
    smr_bins = np.linspace(-175, 0, 50)
    sme_bins = np.linspace(0, 2000, 50)
    
    akr_bins = np.logspace(0, 6, 50)
    #lf_bins = np.logspace(0, 6, 50)
    
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 12.5))
    
    label_counter = 0
    for (i, interval_tag) in enumerate(['full_archive', 'cassini_flyby',
                              'long_nightside_period']):
        #print(interval_tag)
        interval = interval_options.loc[interval_options.tag == interval_tag]
        #print(interval.stime.values[0])
        #print(interval.etime.values[0])
        interval_df = supermag_df.loc[(supermag_df.Date_UTC >= interval.stime.values[0]) & (supermag_df.Date_UTC <= interval.etime.values[0])] 
    
        for (j, (index, binz)) in enumerate(zip(['SMR', 'SME'], [smr_bins, sme_bins])):
            ax[i, j].hist(interval_df[index], bins=binz, color=interval.color,
                          density=True, label=index)
            
            if index == "SME":
                ax[i, j].hist(-1 * interval_df["SML"], bins=binz,
                              histtype='step', color='black', density=True,
                              label='-SML')
                ax[i, j].legend(loc='lower right', fontsize=fontsize)
            
            # Label mean, std
            mean = np.nanmean(interval_df[index])
            std = np.nanstd(interval_df[index])
            var = np.nanvar(interval_df[index])
            
            ax[i, j].axvline(mean, color='black', linewidth=1.5, linestyle='dashed')
            if index == "SME":
                horiza = "left"
                opp_ha = "right"
                plus = 100.
                lab_x = 0.95
            elif index == "SMR":
                horiza = "right"
                opp_ha = "left"
                plus = -10.
                lab_x = 0.05
            ax[i, j].text(mean + plus, 0.9,
                          r"$\bar{x}$ = " + str(np.round(mean, 2)) +
                          "\n$\sigma$ = " + str(np.round(std, 2)) +
                          "\n$\sigma^{2}$ = " + str(np.round(var, 2)) +
                          "\nN = " + str(len(interval_df)),
                          transform=ax[i, j].get_xaxis_transform(),
                          ha=horiza, va='top', fontsize=fontsize)

            # x y labels
            ax[i, j].set_xlabel(index + " (nT)", fontsize=fontsize)
            ax[i, j].set_ylabel("Normalised Occurrence", fontsize=fontsize)
            # # Label interval
            # if j == 1:
            #     ax[i, j].text(1.05, 0.5, interval.label.values[0],
            #                   va='center', ha='left',
            #                   transform=ax[i, j].transAxes,
            #                   rotation='vertical', fontsize=fontsize)
            # Fontsize
            ax[i, j].tick_params(labelsize=fontsize)
            
            # a b c labels
            t = ax[i, j].text(lab_x, 0.95, axes_labels[label_counter],
                              transform=ax[i, j].transAxes,
                              fontsize=fontsize, va='top', ha=opp_ha)
            t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

            label_counter = label_counter + 1
        
         
        akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)
        # breakpoint()
        for (k, (f_band, title)) in enumerate(zip(
                ['ipwr_100_400kHz', 'ipwr_50_100kHz'],
                ["100-400 kHz Power (W $sr^{-1}$)",
                 "50-100 kHz Power (W $sr^{-1}$)"]
                )):


            # if f_band == "ipwr_50_100kHz":
            #     ax[i, k + 2].hist(akr_df[f_band], bins=akr_bins,
            #                       color=interval.color, density=True,
            #                       label=f_band)
            #     #ax[i, k + 2].legend(loc='lower right', fontsize=fontsize)

            # elif f_band == "ipwr_100_400kHz":
            ax[i, k + 2].hist(akr_df[f_band], bins=akr_bins, color=interval.color,
                                  density=True, label=f_band)

            ax[i, k + 2].set_xscale('log')

            # Label mean, std
            median = np.nanmedian(akr_df.loc[akr_df[f_band] > 0., f_band])
            std = np.nanstd(akr_df.loc[akr_df[f_band] > 0., f_band])
            var = np.nanvar(akr_df.loc[akr_df[f_band] > 0., f_band])
            
            ax[i, k + 2].axvline(median, color='black', linewidth=1.5,
                                 linestyle='dashed')
            # if index == "SME":
            #     horiza = "left"
            #     opp_ha = "right"
            #     plus = 100.
            #lab_x = 0.95
            # elif index == "SMR":
            horiza = "right"
            opp_ha = "left"
            #plus = -10.
            lab_x = 0.05
            # breakpoint()
            # ax[i, k + 2].text(median, 0.9,
            #               r"$\eta$ = " + str(np.round(median, 2)) +
            #               "\n$\sigma$ = " + str(np.round(std, 2)) +
            #               "\n$\sigma^{2}$ = " + str(np.round(var, 2)) +
            #               "\nN = " + str(len(akr_df)),
            #               transform=ax[i, k + 2].get_xaxis_transform(),
            #               ha=horiza, va='top', fontsize=fontsize)

            ax[i, k + 2].text(median - (median * 0.15), 0.85,
                          r"$\eta$ = " + f"{median:.2e}" +
                          "\n$\sigma$ = " + f"{std:.2e}" +
                          "\n$\sigma^{2}$ = " + f"{var:.2e}" +
                          "\nN = " + str(len(akr_df)),
                          transform=ax[i, k + 2].get_xaxis_transform(),
                          ha=horiza, va='top', fontsize=fontsize)


            # # x y labels
            ax[i, k + 2].set_xlabel(title,
                                    fontsize=fontsize)
            ax[i, k + 2].set_ylabel("Normalised Occurrence",
                                    fontsize=fontsize)
            # Label interval
            if k == 1:
                ax[i, k + 2].text(1.05, 0.5, interval.label.values[0],
                                  va='center', ha='left',
                                  transform=ax[i, k + 2].transAxes,
                                  rotation='vertical', fontsize=fontsize)
            # Fontsize
            ax[i, k + 2].tick_params(labelsize=fontsize)
            
            # a b c labels
            t = ax[i, k + 2].text(lab_x, 0.95, axes_labels[label_counter],
                              transform=ax[i, k + 2].transAxes,
                              fontsize=fontsize, va='top', ha=opp_ha)
            t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

            label_counter = label_counter + 1
            
            

    fig.tight_layout()
    
    fig_png = os.path.join(fig_dir, "three_interval_SMR_SME_AKR_hist.png")
    fig.savefig(fig_png)

def read_subset_bootstraps(subset_n, freq_ch='ipwr_100_400kHz',
                           n_bootstrap=100):
    """
    Read in bootstraps

    Parameters
    ----------
    subset_n : string
        String name for the interval, or synthetic.
    freq_ch : string, optional
        Frequency channel. The default is 'ipwr_100_400kHz'.
    n_bootstrap : int, optional
        Number of bootstraps. The default is 100.

    Returns
    -------
    ftime : np.array
        Time axis for bootstraps.
    BS : np.array
        Bootstrapped intensity.

    """
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
    """
    Run the Lomb Scargle analysis, creating and saving plots.

    Returns
    -------
    None.

    """
    # Initialising variables
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    samples_per_peak = 5

    vertical_indicators = [12, 24]
    vertical_ind_col = 'black'

    annotate_bbox = {"facecolor": "white", "edgecolor": "grey", "pad": 5.}

    # Different frequency channels
    freq_tags = np.array(['ipwr_100_400kHz', 'ipwr_50_100kHz'])
    freq_labels = np.array(['100-400 kHz', '50-100 kHz'])
    freq_colors = np.array(['dimgrey', 'darkorange', 'rebeccapurple'])

    LS_fig = os.path.join(fig_dir, "two_interval_lomb_scargle.png")

    # Number of bootstraps
    n_bootstrap = 100

    # FAP filenames
    FAP_peaks_dir = os.path.join(data_dir, "lomb_scargle", 'LS_peaks_for_FAP')
    synthetic_FAP_pkl = os.path.join(data_dir, "lomb_scargle", "synthetic_FAP_"
                                     + str(n_bootstrap) + "_BSs.pkl")
    # Read in interval data
    print('Reading AKR data over requested intervals')
    interval_options = read_and_tidy_data.return_test_intervals()

    # Select only desired intervals
    desired_intervals = ['full_archive', 'long_nightside_period']
    desired_i, = np.where(interval_options.tag.isin(desired_intervals) == True)
    labels = interval_options.label[desired_i].values

    # Initialise plotting window
    fig, ax = plt.subplots(nrows=3, figsize=(12.5, 13))

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

    # Draw arrow for FAP
    trans = transforms.blended_transform_factory(ax[0].transData,
                                                 ax[0].transData)
    ax[0].annotate("FAL\n" + "{:.3e}".format(FAP),
                   xy=(9., FAP), xytext=(8., FAP),
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

    for (i, interval_tag) in enumerate(desired_intervals):
        print('Running Lomb-Scargle for ', interval_tag)

        base_dir = pathlib.Path(data_dir) / 'lomb_scargle'
        file_paths = [
            base_dir / f"LS_{interval_tag}_{f}.csv" for f in freq_tags]
        file_checks = [file_path.is_file() for file_path in file_paths]

        if all(file_checks) is False:
            akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)

        # Remove any rows where intensity == np.nan
        for (j, (freq_column, c, n)) in enumerate(zip(freq_tags,
                                                      freq_colors,
                                                      freq_labels)):
            print('Frequency band: ', freq_column)
            ls_csv = os.path.join(data_dir, 'lomb_scargle', 'LS_' +
                                  interval_tag + '_' + freq_column + '.csv')
            if pathlib.Path(ls_csv).is_file() is False:
                freq_df = akr_df.dropna(subset=[freq_column])
                t1 = pd.Timestamp.now()
                print('starting LS at ', t1)
                ls_object, freqs, ls_pgram = lomb_scargle.generic_lomb_scargle(
                    freq_df.unix, freq_df[freq_column], f_min, f_max,
                    n0=samples_per_peak)
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
                # Write to file
                ls_df.to_csv(ls_csv, index=False)

            else:
                ls_df = pd.read_csv(ls_csv, delimiter=',',
                                    float_precision='round_trip')

                ls_pgram = np.array(ls_df.ls_pgram)
                periods = np.array(ls_df.period_hr)

            # Plot FAP here
            FAP_pkl = os.path.join(
                data_dir, "lomb_scargle",
                interval_tag + '_' + freq_column + "_FAP_" + str(n_bootstrap)
                + "_BSs.pkl")
            # Read in bootstrap
            ftime_cl, BS = read_subset_bootstraps(interval_tag,
                                                  freq_ch=freq_column,
                                                  n_bootstrap=n_bootstrap)
            # Convert ftime_cl to unix
            ftime_unix = [pd.Timestamp(t).timestamp() for t in ftime_cl]

            # Read in/calc peak magnitudes for bootstraps and FAP
            bootstrap_peak_magnitudes, FAP = lomb_scargle.false_alarm_probability(
                n_bootstrap, BS, ftime_unix, f_min, f_max, FAP_peaks_dir,
                interval_tag + '_' + freq_column, FAP_pkl, n0=samples_per_peak)

            ax[i + 1].plot(periods, ls_pgram, linewidth=1.5, color=c, label=n)
            if j == 0:
                trans = transforms.blended_transform_factory(
                    ax[i + 1].transAxes, ax[i + 1].transData)
                ax[i + 1].annotate("FAL\n" + "{:.3e}".format(FAP),
                                   xy=(0.2, FAP), xytext=(0.1, FAP),
                                   xycoords=trans, arrowprops={'facecolor': c},
                                   fontsize=fontsize, va='center', ha='right',
                                   color=c, bbox=annotate_bbox,
                                   fontweight="bold")
            elif j == 1:
                trans = transforms.blended_transform_factory(
                    ax[i + 1].transAxes, ax[i + 1].transData)
                ax[i + 1].annotate("FAL\n" + "{:.3e}".format(FAP),
                                   xy=(0.8, FAP), xytext=(0.9, FAP),
                                   xycoords=trans, arrowprops={'facecolor': c},
                                   fontsize=fontsize, va='center', ha='left',
                                   color=c, bbox=annotate_bbox, fontweight="bold")
        ax[i + 1].set_xscale('log')

        # Formatting
        ax[i + 1].set_ylabel('Lomb-Scargle\nNormalised Amplitude',
                             fontsize=fontsize)
        ax[i + 1].set_xlabel('Period (hours)', fontsize=fontsize)
        ax[i + 1].tick_params(labelsize=fontsize)
        ax[i + 1].legend(fontsize=fontsize, loc='upper left')

        if vertical_indicators != []:
            for h in vertical_indicators:
                trans = transforms.blended_transform_factory(
                    ax[i + 1].transData, ax[i + 1].transAxes)
                ax[i + 1].annotate(str(h), xy=(h, 1.0), xytext=(h, 1.15),
                                   xycoords=trans,
                                   arrowprops={'facecolor': 'black'},
                                   fontsize=fontsize, va='top', ha='center',
                                   color=vertical_ind_col)
                
    # Add in more ticks
    majticks = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    majlabels = [str(t) for t in majticks]
    #ax[0].set_xticks(majticks, labels=majlabels)

    # Label panels
    titles = np.append('Synthetic', labels)
    for (i, a) in enumerate(ax):
        t = a.text(0.005, 1.05, axes_labels[i], transform=a.transAxes,
                   fontsize=fontsize, va='bottom', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

        tit = a.text(1.0, 1.05, titles[i], transform=a.transAxes,
                     fontsize=1.25 * fontsize, va='center', ha='right')
        a.set_xticks(majticks, labels=majlabels)

    # Adjust margins etc
    fig.tight_layout()
    # Save to file
    fig.savefig(LS_fig)



def wind_cassini_lomb_scargle():

    # Initialising variables
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    samples_per_peak = 5

    vertical_indicators = [12, 24]
    vertical_ind_col = 'black'

    annotate_bbox = {"facecolor": "white", "edgecolor": "grey", "pad": 5.}

    FAL_color = 'darkmagenta'
    tru_color = 'coral'

    # Different frequency channels
    freq_tags = np.array(['ipwr_100_400kHz'])#, 'ipwr_50_100kHz'])
    freq_labels = np.array(['100-400 kHz', '50-100 kHz'])
    freq_colors = np.array(['dimgrey', 'darkorange', 'rebeccapurple'])

    LS_fig = os.path.join(fig_dir, "LS_wind_cassini_smoothed.png")

    # Number of bootstraps
    n_bootstrap = 100

    # FAP filenames
    FAP_peaks_dir = os.path.join(data_dir, "lomb_scargle", 'LS_peaks_for_FAP')
    # synthetic_FAP_pkl = os.path.join(data_dir, "lomb_scargle", "synthetic_FAP_"
    #                                  + str(n_bootstrap) + "_BSs.pkl")
    # Read in interval data
    print('Reading AKR data over requested intervals')
    interval_options = read_and_tidy_data.return_test_intervals()
    # interval_options = interval_options.loc[interval_options['tag'] == 'cassini_flyby']
    desired_intervals = ['cassini_flyby', 'cassini_flyby_logged']
    desired_i, = np.where(interval_options.tag.isin(desired_intervals) == True)
    labels = interval_options.label[desired_i].values
    
    # Initialise plotting window
    #fig, ax = plt.subplots(nrows=2, figsize=(12.5, 8))
    
    fig = plt.figure(figsize=(17.5, 8))
    gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2.5, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax = [ax1, ax2, ax3, ax4]

    if vertical_indicators != []:
        for h in vertical_indicators:
            trans = transforms.blended_transform_factory(ax[0].transData,
                                                         ax[0].transAxes)
            ax[0].annotate(str(h), xy=(h, 1.0), xytext=(h, 1.15),
                           xycoords=trans, arrowprops={'facecolor': 'black'},
                           fontsize=fontsize, va='top', ha='center',
                           color=vertical_ind_col)

    ax_counter = 0
    for (i, tag) in enumerate(desired_intervals):
        print('Running Lomb-Scargle for ', tag)

        base_dir = pathlib.Path(data_dir) / 'lomb_scargle'
        file_paths = [
            base_dir / f"LS_{tag}_{f}.csv" for f in freq_tags]
        file_checks = [file_path.is_file() for file_path in file_paths]

        if all(file_checks) is False:
            akr_df = read_and_tidy_data.select_akr_intervals(tag)

        # Remove any rows where intensity == np.nan
        for (j, (freq_column, c, n)) in enumerate(zip(freq_tags,
                                                      freq_colors,
                                                      freq_labels)):
            print('Frequency band: ', freq_column, ' tag: ', tag)
            ls_csv = os.path.join(data_dir, 'lomb_scargle', 'LS_' +
                                  tag + '_' + freq_column + '.csv')
            
            if tag == 'cassini_flyby_logged':   
                intensity_col = 'log_smoothed_' + freq_column
            else:
                intensity_col = freq_column
        
            if pathlib.Path(ls_csv).is_file() is False:

                freq_df = akr_df.copy(deep=True)

                    
                freq_df = freq_df.dropna(subset=[intensity_col])
                print(intensity_col)
                
                t1 = pd.Timestamp.now()
                print('starting LS at ', t1)
                ls_object, freqs, ls_pgram = lomb_scargle.generic_lomb_scargle(
                    freq_df.unix, freq_df[intensity_col], f_min, f_max,
                    n0=samples_per_peak)
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
                # Write to file
                ls_df.to_csv(ls_csv, index=False)

            else:
                ls_df = pd.read_csv(ls_csv, delimiter=',',
                                    float_precision='round_trip')

                ls_pgram = np.array(ls_df.ls_pgram)
                periods = np.array(ls_df.period_hr)


            # Plot FAP here
            FAP_pkl = os.path.join(
                data_dir, "lomb_scargle",
                tag + '_' + intensity_col + "_FAP_" + str(n_bootstrap)
                + "_BSs.pkl")
            # Read in bootstrap
            ftime_cl, BS = read_subset_bootstraps(tag,
                                                  freq_ch=intensity_col,
                                                  n_bootstrap=n_bootstrap)
            # Convert ftime_cl to unix
            ftime_unix = [pd.Timestamp(t).timestamp() for t in ftime_cl]

            # Read in/calc peak magnitudes for bootstraps and FAP
            bootstrap_peak_magnitudes, FAP = lomb_scargle.false_alarm_probability(
                n_bootstrap, BS, ftime_unix, f_min, f_max, FAP_peaks_dir,
                tag + '_' + freq_column, FAP_pkl, n0=samples_per_peak)

            ax[ax_counter].plot(periods, ls_pgram, linewidth=1.5, color=c, label=n)
            if j == 0:
                trans = transforms.blended_transform_factory(
                    ax[ax_counter].transAxes, ax[ax_counter].transData)
                ax[ax_counter].annotate("FAL\n" + "{:.3e}".format(FAP),
                                   xy=(0.2, FAP), xytext=(0.1, FAP),
                                   xycoords=trans, arrowprops={'facecolor': FAL_color},
                                   fontsize=fontsize, va='center', ha='right',
                                   color=FAL_color, bbox=annotate_bbox,
                                   fontweight="bold")
            elif j == 1:
                trans = transforms.blended_transform_factory(
                    ax[ax_counter].transAxes, ax[ax_counter].transData)
                ax[ax_counter].annotate("FAL\n" + "{:.3e}".format(FAP),
                                   xy=(0.8, FAP), xytext=(0.9, FAP),
                                   xycoords=trans, arrowprops={'facecolor': c},
                                   fontsize=fontsize, va='center', ha='left',
                                   color=c, bbox=annotate_bbox, fontweight="bold")
                
            ax[ax_counter + 1].hist(bootstrap_peak_magnitudes,
                                    bins=np.linspace(0.0005, 0.0016, 25),
                                    color=c, alpha=0.5,
                                    label=str(n_bootstrap) + "\nbootstraps"
                                    )
            pgram_max = np.nanmax(ls_pgram)
            pgram_max_period = periods[np.nanargmax(ls_pgram)]
            ax[ax_counter].plot(pgram_max_period, pgram_max,
                                linewidth=0., color=tru_color, marker='*',
                                markersize=1.25 * fontsize, label='H',
                                fillstyle='none')
            ax[ax_counter + 1].axvline(pgram_max,
                                       linewidth=1.5, linestyle='dashed', 
                                       color=tru_color, label="H")
            ax[ax_counter + 1].axvline(FAP,
                                       linewidth=1.5, linestyle='dashed', 
                                       color=FAL_color, label="FAL")
            mean_bs = np.nanmean(bootstrap_peak_magnitudes)
            ax[ax_counter + 1].axvline(mean_bs,
                                       linewidth=1.5, linestyle='dotted', 
                                       color='black', label="mean")
            stats = "H = " + str(np.round((pgram_max / FAP), 2)) + "*FAL\nH = " + str(np.round((pgram_max / mean_bs), 2)) + "*mean"
            # #"Highest peak " + str(np.round((pgram_max / FAP), 2)) "times larger than FAL, and "
            # breakpoint()
            
            ax[ax_counter + 1].text(1.0, 1.0, stats,
                                    transform=ax[ax_counter + 1].transAxes,
                                    fontsize=0.9 * fontsize, ha='right', va='bottom')
        
        ax[ax_counter].set_xscale('log')

        # Formatting
        ax[ax_counter].set_ylabel('Lomb-Scargle\nNormalised Amplitude',
                                  fontsize=fontsize)
        ax[ax_counter].set_xlabel('Period (hours)', fontsize=fontsize)
        ax[ax_counter].legend(fontsize=fontsize, loc='upper left')
        
        ax[ax_counter].text(1.0, 1.0, labels[i],
                            transform=ax[ax_counter].transAxes,
                            fontsize=fontsize, ha='right', va='bottom')

        if vertical_indicators != []:
            for h in vertical_indicators:
                trans = transforms.blended_transform_factory(
                    ax[ax_counter].transData, ax[ax_counter].transAxes)
                ax[ax_counter].annotate(str(h), xy=(h, 1.0), xytext=(h, 1.15),
                                   xycoords=trans,
                                   arrowprops={'facecolor': 'black'},
                                   fontsize=fontsize, va='top', ha='center',
                                   color=vertical_ind_col)
        
        # Histograms
        ax[ax_counter + 1].legend(loc="upper right", fontsize=0.8 * fontsize)
        ax[ax_counter + 1].set_xlabel("Magnitude of LS Peak", fontsize=fontsize)
        ax[ax_counter + 1].set_ylabel("Occurrence", fontsize=fontsize)
        ax[ax_counter + 1].set_xscale('log')
        
        ax_counter = ax_counter + 2
        
    # Add in more ticks
    majticks = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    majlabels = [str(t) for t in majticks]
    #ax[0].set_xticks(majticks, labels=majlabels)

    for (i, a) in enumerate(ax):
        t = a.text(0.005, 1.05, axes_labels[i], transform=a.transAxes,
                   fontsize=fontsize, va='bottom', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

        # tit = a.text(1.0, 1.025, labels[i], transform=a.transAxes,
        #              fontsize=1.25 * fontsize, va='bottom', ha='right')
        # a.set_xticks(majticks, labels=majlabels)
        a.tick_params(labelsize=fontsize)

    # Adjust margins etc
    fig.tight_layout()
    # Save to file
    fig.savefig(LS_fig) 
    

def run_ACF():
    """
    Run the ACF analysis and create plots.

    Returns
    -------
    None.

    """
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

    # Initialise table arrays
    table_title = [' ']
    table_subtitle = [' ']
    table_linear_trend = ['Linear Trend']
    table_linear_Rsq = ['$R^{2}$']
    table_detrend_mean = ['Normalisation mean']
    table_detrend_std = ['Normalisation STD']
    table_shm_fit = ['SHM fit']
    table_smh_omega = ['$2\pi$ / $\omega (hours)$']
    table_shm_chisq = ['$\chi^{2}$']

    # Add in Synthetic table things
    table_title.append('Synthetic')
    table_subtitle.append(' ')
    table_linear_trend.append(synthetic_acf_fit.text_linear_trend)
    table_linear_Rsq.append(synthetic_acf_fit.text_linear_trend_pearson_r)
    table_detrend_mean.append(synthetic_acf_fit.text_normalisation_mean)
    table_detrend_std.append(synthetic_acf_fit.text_normalisation_std)
    table_shm_fit.append(synthetic_acf_fit.text_shm_trend)
    table_smh_omega.append("{:.2f}".format(((2* np.pi) / synthetic_acf_fit.omega) / (60 * 60)))
    table_shm_chisq.append(synthetic_acf_fit.text_shm_chi_sq)

    # Add latex formatting
    temp = [t + ' & ' for t in table_title[:-1]]
    temp.append(table_title[-1])
    temp.append(' \\')
    table_title_fm = temp.copy()

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

    # Loop over datasets, calculating and plotting
    for (i, interval_tag) in enumerate(interval_options['tag']):
        print('Running autocorrelation for ', interval_tag)

        base_dir = pathlib.Path(data_dir) / 'acf'
        file_paths = [base_dir / f"ACF_{interval_tag}_{f}.csv" for f in freq_tags]
        file_checks = [file_path.is_file() for file_path in file_paths]

        # If no ACF calculations done, read integrated power data
        if all(file_checks) is False:
            akr_df = read_and_tidy_data.select_akr_intervals(interval_tag)

        # Latex table title
        table_title.append("\multicolumn{2}{|c|}{" + interval_options['label'].iloc[i] + "}")

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
                tax.set_ylabel('ACF ($W^{2}$ $sr^{-2}$)\n(' + n + ')',
                               color=c, fontsize=fontsize)
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
                table_linear_trend.append(LFE_acf_fit.text_linear_trend)
                table_linear_Rsq.append(
                    LFE_acf_fit.text_linear_trend_pearson_r)
                table_detrend_mean.append(LFE_acf_fit.text_normalisation_mean)
                table_detrend_std.append(LFE_acf_fit.text_normalisation_std)
                table_shm_fit.append(LFE_acf_fit.text_shm_trend)
                table_smh_omega.append(
                    "{:.2f}".format(((2* np.pi) / LFE_acf_fit.omega) / (60*60)))
                table_shm_chisq.append(LFE_acf_fit.text_shm_chi_sq)

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

                table_linear_trend.append(mAKR_acf_fit.text_linear_trend)
                table_linear_Rsq.append(
                    mAKR_acf_fit.text_linear_trend_pearson_r)
                table_detrend_mean.append(mAKR_acf_fit.text_normalisation_mean)
                table_detrend_std.append(mAKR_acf_fit.text_normalisation_std)
                table_shm_fit.append(mAKR_acf_fit.text_shm_trend)
                table_smh_omega.append(
                    "{:.2f}".format(((2* np.pi) / mAKR_acf_fit.omega) / (60*60)))
                table_shm_chisq.append(mAKR_acf_fit.text_shm_chi_sq)
            table_subtitle.append(n)

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
                a.set_ylabel('ACF ($W^{2}$ $sr^{-2}$)', fontsize=fontsize)
            elif (j >= 1) & (k == 0):
                a.set_ylabel(
                    'ACF ($W^{2}$ $sr^{-2}$)\n(' + freq_labels[0] + ')',
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
    fig.savefig(ACF_fig)

    # Add latex formatting
    temp = [t + ' & ' for t in table_title[:-1]]
    temp.append(table_title[-1])
    temp.append(' \\')
    table_title_fm = temp.copy()
    print(''.join(table_title_fm))

    # Print out latex table
    temp = [t + ' & ' for t in table_subtitle[:-1]]
    temp.append(table_subtitle[-1])
    temp.append(' \\')
    table_subtitle_fm = temp.copy()
    print(''.join(table_subtitle_fm))

    temp = [t + ' & ' for t in table_linear_trend[:-1]]
    temp.append(table_linear_trend[-1])
    temp.append(' \\')
    table_linear_trend_fm = temp.copy()
    print(''.join(table_linear_trend_fm))

    temp = [t + ' & ' for t in table_linear_Rsq[:-1]]
    temp.append(table_linear_Rsq[-1])
    temp.append(' \\')
    table_linear_Rsq_fm = temp.copy()
    print(''.join(table_linear_Rsq_fm))

    temp = [t + ' & ' for t in table_detrend_mean[:-1]]
    temp.append(table_detrend_mean[-1])
    temp.append(' \\')
    table_detrend_mean_fm = temp.copy()
    print(''.join(table_detrend_mean_fm))

    temp = [t + ' & ' for t in table_detrend_std[:-1]]
    temp.append(table_detrend_std[-1])
    temp.append(' \\')
    table_detrend_std_fm = temp.copy()
    print(''.join(table_detrend_std_fm))

    temp = [t + ' & ' for t in table_shm_fit[:-1]]
    temp.append(table_shm_fit[-1])
    temp.append(' \\')
    table_shm_fit_fm = temp.copy()
    print(''.join(table_shm_fit_fm))

    temp = [t + ' & ' for t in table_smh_omega[:-1]]
    temp.append(table_smh_omega[-1])
    temp.append(' \\')
    table_shm_fit_fm = temp.copy()
    print(''.join(table_shm_fit_fm))

    temp = [t + ' & ' for t in table_shm_chisq[:-1]]
    temp.append(table_shm_chisq[-1])
    temp.append(' \\')
    table_shm_chisq_fm = temp.copy()
    print(''.join(table_shm_chisq_fm))


def run_MLT_binning_overlayed(n_mlt_sectors='four'):
    """
    Run the MLT binning with plots overlayed.

    Parameters
    ----------
    n_mlt_sectors : string, optional
        How many MLT sectors to divide data into. The default is 'four'.

    Returns
    -------
    None.

    """

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

    lon_bin_width = 30.

    # Different frequency channels
    freq_tags = np.array(['ipwr_100_400kHz', 'ipwr_50_100kHz'])
    freq_labels = np.array(['100-400 kHz', '50-100 kHz'])
    freq_colors = np.array(['dimgrey', 'darkorange', 'rebeccapurple'])

    MLT_fig = os.path.join(fig_dir,
                           "three_interval_MLT_binned_" + n_mlt_sectors + "MLT.png")

    # Read in interval data
    print('Reading AKR data over requested intervals')
    interval_options = read_and_tidy_data.return_test_intervals()

    # Initialise plotting window
    fig, ax = plt.subplots(nrows=interval_options['tag'].size,
                           ncols=freq_tags.size, figsize=(21, 13))

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
        for (j, (freq_column, c, n)) in enumerate(zip(freq_tags,
                                                      freq_colors,
                                                      freq_labels)):
            print('Frequency band: ', freq_column)
            MLT_csv = os.path.join(data_dir, 'MLT_binning',
                                   'MLT_binned_' + n_mlt_sectors + 'sector_' +
                                   interval_tag + '_' + freq_column + '.csv')

            if pathlib.Path(MLT_csv).is_file() is False:
                freq_df = akr_df.dropna(subset=[freq_column])
                t1 = pd.Timestamp.now()
                print('starting MLT binning at ', t1)

                UT_df = []
                lon_df = binning_averaging.return_lon_trend(
                    akr_df, region_centres=region_centres,
                    region_width=region_width, region_names=region_names,
                    region_flags=region_flags, lon_bin_width=30.,
                    ipower_tag=freq_column, lon_sol_tag="lon_sol",
                    lon_sc_tag="lon_gsm")
                breakpoint()
                t2 = pd.Timestamp.now()
                print('MLT binning finished, time elapsed: ', t2-t1)

            else:

                UT_df = pd.read_csv(MLT_csv, delimiter=',',
                                    float_precision='round_trip')

            if freq_column == 'ipwr_100_400kHz':
                for k, (MLT_n, c, mrkr) in enumerate(zip(region_names,
                                                         region_colors,
                                                         region_mrkrs)):
                    ax[i, 0].plot(UT_df.UT_bin_centre,
                                  UT_df[MLT_n + '_median_norm_no0'],
                                  color=c, label=MLT_n, marker=mrkr,
                                  markersize=fontsize)
                    ax[i, 0].fill_between(UT_df.UT_bin_centre,
                                          UT_df[MLT_n + '_median_norm_no0'] -
                                          UT_df[MLT_n + '_mad_norm_no0'],
                                          UT_df[MLT_n + '_median_norm_no0'] +
                                          UT_df[MLT_n + '_mad_norm_no0'],
                                          color=c, alpha=0.2)

            elif freq_column == 'ipwr_50_100kHz':
                for k, (MLT_n, c, mrkr) in enumerate(zip(region_names,
                                                         region_colors,
                                                         region_mrkrs)):
                    ax[i, 1].plot(UT_df.UT_bin_centre,
                                  UT_df[MLT_n + '_median_norm_no0'],
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
            ax[i, j].set_ylabel('Normalised median integrated power',
                                fontsize=fontsize)
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
    """
    Run MLT binning with seperate plots.

    Parameters
    ----------
    n_mlt_sectors : string, optional
        How many MLT sectors to divide data into. The default is 'four'.

    Returns
    -------
    None.

    """

    fontsize = 20
    lon_sc_tag = 'lon_geo'
    lon_sol_tag = 'lon_sol'

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

    # Longitude bin is the same always
    lon_bin_width = 30.
    bwidth = lon_bin_width * 0.8

    # Different frequency channels
    freq_tags = np.array(['ipwr_100_400kHz', 'ipwr_50_100kHz'])
    freq_labels = np.array(['100-400 kHz', '50-100 kHz'])
    freq_colors = np.array(['dimgrey', 'darkorange', 'rebeccapurple'])

    # Read in interval data
    print('Reading AKR intervals')
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

            # Convert from GSE to GEO lat and lon
            geo_df = utility.full_archive_geo_coord()
            akr_df['lon_geo'] = geo_df.lon_geo
            akr_df['lat_geo'] = geo_df.lat_geo

            akr_df['lon_sol'] = utility.calc_longitude_of_sun(
                akr_df, lon_tag=lon_sc_tag, plot=True)

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
                                   'MLT_lon_binned_' +
                                   n_mlt_sectors + 'sector_' +
                                   interval_tag + '_' + freq_column + '.csv')
            # Bin data into lon bins
            if pathlib.Path(MLT_csv).is_file() is False:
                lon_df = binning_averaging.return_lon_trend(
                        akr_df, region_centres=region_centres,
                        region_width=region_width, region_names=region_names,
                        region_flags=region_flags, lon_bin_width=30.,
                        ipower_tag=freq_column,
                        lon_sol_tag=lon_sol_tag, lon_sc_tag=lon_sc_tag)

                lon_df.to_csv(MLT_csv, index=False)
            # Else read in pre-sorted data
            else:
                lon_df = pd.read_csv(MLT_csv, delimiter=',',
                                     float_precision='round_trip')

            # Loop through lon sun and lon sc
            for o, (tg, f_tg, tit_tg) in enumerate(zip(
                    [lon_sol_tag, lon_sc_tag],
                    ['sol', 'sc'],
                    ['Longitude of the Sun ($^{\circ}$)',
                     'Spacecraft Geographic Longitude ($^{\circ}$)'])):

                # Initialise plotting window
                fig, ax = plt.subplots(nrows=len(region_centres), ncols=1,
                                       figsize=(15, len(region_centres) * 3.))
                fig_name = os.path.join(fig_dir, interval_tag +
                                        "_" + freq_column +
                                        "_" + f_tg +
                                        "_" + n_mlt_sectors + "MLT.png")

                # Loop through each MLT sector, plotting
                for k, (MLT_n, c, mrkr) in enumerate(zip(region_names,
                                                         region_colors,
                                                         region_mrkrs)):
                    # Set x axis limits
                    ax[k].set_xlim(left=0., right=360.)

                    # Plot number of observations
                    ax[k].bar(lon_df.lon_bin_centre,
                              lon_df[MLT_n + "_n_" + f_tg],
                              zorder=0.5, color='gold', linewidth=.75,
                              edgecolor='black', alpha=0.75, label='# all',
                              width=bwidth)
                    ax[k].bar(lon_df.lon_bin_centre,
                              lon_df[MLT_n + "_n_no0_" + f_tg],
                              zorder=0.6, color='deeppink', linewidth=.75,
                              edgecolor='black', alpha=0.75, label="# > 0",
                              width=bwidth)

                    # Create twin axis
                    twax = ax[k].twinx()

                    # Plot intensity trend
                    twax.plot(lon_df.lon_bin_centre,
                              lon_df[MLT_n + '_median_norm_no0' + "_" + f_tg],
                              color='black', label="median", marker='o',
                              markersize=fontsize, linewidth=1.5, zorder=2.5)
                    twax.fill_between(
                        lon_df.lon_bin_centre,
                        lon_df[MLT_n + '_median_norm_no0' + "_" + f_tg] -
                        lon_df[MLT_n + '_mad_norm_no0' + "_" + f_tg],
                        lon_df[MLT_n + '_median_norm_no0' + "_" + f_tg] +
                        lon_df[MLT_n + '_mad_norm_no0' + "_" + f_tg],
                        color='grey', alpha=0.5, label="MAD", zorder=2)

                    # Formatting
                    # Panel MLT sector
                    t = ax[k].text(0.98, 0.95, MLT_n + ' MLT',
                                   transform=ax[k].transAxes,
                                   fontsize=fontsize,
                                   va='top', ha='right')
                    t.set_bbox(dict(facecolor='white', alpha=0.75,
                                    edgecolor='grey'))
                    # Y labels
                    ax[k].set_ylabel("# averaged", fontsize=fontsize)
                    twax.set_ylabel('Normalised median\nintegrated power',
                                    fontsize=fontsize)
                    # Fontsize
                    ax[k].tick_params(labelsize=fontsize)
                    twax.tick_params(labelsize=fontsize)
                    # Panel label
                    t = ax[k].text(0.02, 0.95, axes_labels[k],
                                   transform=ax[k].transAxes,
                                   fontsize=fontsize,
                                   va='top', ha='left')
                    t.set_bbox(dict(facecolor='white', alpha=0.75,
                                    edgecolor='grey'))

                    if interval_tag == "full_archive":

                        imax = lon_df[MLT_n + '_median_norm_no0' + "_" +
                                      f_tg].idxmax()

                        twax.plot([lon_df.lon_bin_centre.iloc[imax]],
                                  [lon_df[MLT_n + '_median_norm_no0' + "_" +
                                          f_tg].iloc[imax]],
                                  marker='*', linewidth=0, fillstyle='none',
                                  markeredgecolor='black', zorder=5,
                                  markersize=fontsize*3., markeredgewidth=2.5)

                # More formatting
                ax[0].text(0.5, 1.01,
                           interval_options.label[i] + ' (' + n + ')',
                           transform=ax[0].transAxes,
                           fontsize=1.25 * fontsize, va='bottom', ha='center')
                leg_ln = [*ax[k].get_legend_handles_labels()[0],
                          *twax.get_legend_handles_labels()[0]]
                leg_lab = [*ax[k].get_legend_handles_labels()[1],
                           *twax.get_legend_handles_labels()[1]]
                ax[0].legend(leg_ln, leg_lab,
                             bbox_to_anchor=(1.13, 0.0, 0.2, 1.0),
                             loc="center left", fontsize=fontsize)
                ax[k].set_xlabel(tit_tg, fontsize=fontsize)

                fig.tight_layout()

                # Save to file
                fig.savefig(fig_name)


def lomb_scargle_cassini_sliding():
    """
    Run Lomb Scargle analysis over a sliding window, starting with the Cassini
    flyby interval.

    Returns
    -------
    None.

    """

    # Define Lomb-Scargle freqs etc
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    # T = (pd.Timestamp(2005, 1, 1, 0) -
    #      pd.Timestamp(1995, 1, 1, 0)).total_seconds()
    samples_per_peak = 5
    # freqs = lomb_scargle.define_frequency_bins(T, f_min, f_max, n0=5)
    freq_column = "ipwr_100_400kHz"

    # Read in interval data
    interval_options = read_and_tidy_data.return_test_intervals()
    interval_details = interval_options.loc[
        interval_options.tag == "cassini_flyby"]
    interval_stime = interval_details.stime.iloc[0]
    interval_etime = interval_details.etime.iloc[0]
    interval_midtime = ((interval_etime - interval_stime) /
                        2.) + interval_stime
    # Read in *all* AKR integrated power
    akr_df = read_and_tidy_data.select_akr_intervals("full_archive")

    # Sliding parameters
    slides = 20
    slide_width = pd.Timedelta(days=10)
    slide_width_multiplier = np.linspace(0, slides, slides+1) - slides/2

    x_lim = [interval_stime - ((slides / 2) * slide_width),
             interval_etime + ((slides / 2) * slide_width)]

    ut_s, ut_e, ut_mid = [], [], []
    for i, factor in enumerate(slide_width_multiplier):

        ut_s.append(interval_stime + (factor * slide_width))
        ut_e.append(interval_etime + (factor * slide_width))
        ut_mid.append(((ut_e[i] - ut_s[i]) / 2.) + ut_s[i])

    ut_s = np.array(ut_s)
    ut_e = np.array(ut_e)
    ut_mid = np.array(ut_mid)

    # LS analysis here
    for i in range(slides + 1):
        # Subselect AKR df
        subset_df = akr_df.loc[(akr_df.datetime >= ut_s[i]) &
                               (akr_df.datetime <= ut_e[i]),
                               :].reset_index(drop=True)
        if i == 0:
            # Run Lomb-Scargle
            ls_object, freqs, ls_pgram_0 = lomb_scargle.generic_lomb_scargle(
                subset_df.unix, subset_df[freq_column], f_min, f_max,
                n0=samples_per_peak)
            ls_pgram = np.full((freqs.size, slides + 1), np.nan)
            ls_pgram[:, i] = ls_pgram_0
            periods = periodicity_functions.freq_to_period(freqs)
        else:
            ls_object, freqs, ls_pgram[:, i] = lomb_scargle.generic_lomb_scargle(
                subset_df.unix, subset_df[freq_column], f_min, f_max,
                n0=samples_per_peak)            

    # Find edges of pixels on period axis
    period_edges = np.full(periods.size + 1, np.nan)
    for k in range(period_edges.size):
        if k == period_edges.size-2:
            period_edges[k] = periods[k-1] + ((periods[k]-periods[k-1])/2)
        elif k == period_edges.size-1:
            period_edges[k] = periods[k-1] + ((periods[k-1]-periods[k-2])/2)
        else:
            period_edges[k] = periods[k] - ((periods[k+1]-periods[k])/2)

    # Initialise plotting
    fig, ax = plt.subplots(nrows=2, figsize=(16, 8))

    # Plot Lomb-Scargles
    X, Y = np.meshgrid(np.append(ut_mid - (slide_width/2.),
                                 ut_mid[-1] + (slide_width/2.)),
                       period_edges)

    pcm = ax[0].pcolormesh(X, Y, ls_pgram, cmap='binary_r')
    cbar = plt.colorbar(pcm, ax=ax[0],
                        label="Lomb-Scargle\nNormalised Amplitude")
    cbar.ax.tick_params(labelsize=fontsize)

    ax[0].set_ylabel('Period (hours)', fontsize=fontsize)

    for h in [12., 24., 36.]:
        ax[0].axhline(h, linestyle='dashed', linewidth=2., color='orangered')
        trans = transforms.blended_transform_factory(ax[0].transAxes,
                                                     ax[0].transData)
        ax[0].text(0.025, h * 1.05, str(int(h)), color='orangered',
                   fontsize=fontsize, va='bottom', ha='left',
                   transform=trans)

    ax_pos = ax[0].get_position().bounds

    # Highlight Cassini period
    ax[0].axvline(interval_midtime - (slide_width/2.), color='royalblue',
                  linewidth=2., linestyle='dashed')
    ax[0].axvline(interval_midtime + (slide_width/2.), color='royalblue',
                  linewidth=2., linestyle='dashed')

    # Make width same as (a)
    pos = ax[1].get_position().bounds
    ax[1].set_position([ax_pos[0], pos[1], ax_pos[2], pos[3]])


def lomb_scargle_cassini_expanding(annotation_colour='grey'):
    """
    Run the Lomb-Scargle analysis, gradually expanding the Cassini flyby
    interval.

    Returns
    -------
    None.

    """
    fig_png_nonlog = os.path.join(fig_dir, "expanding_cassini_lomb_scargle_nonlog.png")
    #fig_png_dots = os.path.join(fig_dir, "expanding_cassini_lomb_scargle_dots.png")

    # Define Lomb-Scargle freqs etc
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    samples_per_peak = 5

    freq_column = "ipwr_100_400kHz"

    # Read in interval data
    interval_options = read_and_tidy_data.return_test_intervals()
    interval_details = interval_options.loc[
        interval_options.tag == "cassini_flyby"]
    interval_stime = interval_details.stime.iloc[0]
    interval_etime = interval_details.etime.iloc[0]
    interval_midtime = ((interval_etime - interval_stime) /
                        2.) + interval_stime
    # Read in *all* AKR integrated power
    akr_df = read_and_tidy_data.select_akr_intervals("full_archive")
    
    # Read in SMR, SML, SME
    supermag_df = read_supermag.concat_indices_years(
        np.array(range(1995, 2005)))

    # Sliding parameters
    slides = 50
    slide_width = pd.Timedelta(days=5)
    slide_width_multiplier = np.linspace(0, slides, slides+1)
    
    # Number of top peaks to collect
    n_peaks = 5

    x_lim = [interval_stime - ((slides / 2) * slide_width),
             interval_etime + ((slides / 2) * slide_width)]

    ut_s, ut_e, length = [], [], []
    for i, factor in enumerate(slide_width_multiplier):

        ut_s.append(interval_stime - (factor * slide_width))
        ut_e.append(interval_etime + (factor * slide_width))
        length.append((ut_e[i] - ut_s[i]).total_seconds())

    ut_s = np.array(ut_s)
    ut_e = np.array(ut_e)
    length = np.array(length)

    # LS analysis here
    variable_freqs = []
    variable_periods = []
    ls_pgram =[]

    peak_freq = np.full(slides + 1, np.nan)
    peak_height = np.full(slides + 1, np.nan)
    top_peaks_freq = np.full((slides + 1, n_peaks), np.nan)
    top_peaks_height = np.full((slides + 1, n_peaks), np.nan)
    
    mean_akr_i = np.full(slides + 1, np.nan)
    std_akr_i = np.full(slides + 1, np.nan)
    
    std_SMR = np.full(slides + 1, np.nan)
    std_SME = np.full(slides + 1, np.nan)
    std_SML = np.full(slides + 1, np.nan)

    for i in range(slides + 1):
        # Subselect AKR df
        subset_df = akr_df.loc[(akr_df.datetime >= ut_s[i]) &
                               (akr_df.datetime <= ut_e[i]),
                               :].reset_index(drop=True)
        mean_akr_i[i] = np.nanmean(subset_df[freq_column])
        std_akr_i[i] = np.nanstd(subset_df[freq_column])

        output = lomb_scargle.generic_lomb_scargle(subset_df.unix,
                                                   subset_df[freq_column],
                                                   f_min, f_max,
                                                   n0=samples_per_peak)

        variable_freqs.append(output[1])
        ls_pgram.append(output[2])
        variable_periods.append(
            periodicity_functions.freq_to_period(output[1]))

        arg = output[2].argmax()
        peak_height[i] = output[2][arg]
        peak_freq[i] = output[1][arg]
        
        sorted_index = np.argsort(output[2])[::-1]
        top_peaks_freq[i, :] = output[1][sorted_index[0:n_peaks]]
        top_peaks_height[i, :] = output[2][sorted_index[0:n_peaks]]
        #breakpoint()
        
        subset_supermag = supermag_df.loc[(supermag_df.Date_UTC >= ut_s[i]) &
                               (supermag_df.Date_UTC <= ut_e[i]),
                               :].reset_index(drop=True)
        std_SMR[i] = np.nanstd(subset_supermag['SMR'])
        std_SME[i] = np.nanstd(subset_supermag['SME'])
        std_SML[i] = np.nanstd(subset_supermag['SML'])


    peak_period = periodicity_functions.freq_to_period(peak_freq)
    top_peaks_period = periodicity_functions.freq_to_period(top_peaks_freq)

    # 2D IMAGE OF LS POWER AS A FUNCTION OF LENGTH OF ARCHIVE
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 18))

    length_days = length / (60. * 60 * 24)

    # Build X bins
    x_centers = length_days  # shape (251,)
    x_edges = centers_to_edges(x_centers)

    # Normalise LS pgram at each X
    ls_pgram_norm = []
    
    for z in ls_pgram:
        z = np.asarray(z)
    
        zmin = np.nanmin(z)
        zmax = np.nanmax(z)
    
        if zmax > zmin:
            z_norm = (z - zmin) / (zmax - zmin)
        else:
            z_norm = np.zeros_like(z)  # or np.nan
    
        ls_pgram_norm.append(z_norm)

    verts = []
    colors = []
    
    for i, x in enumerate(x_centers):
    
        y_centers = variable_periods[i]
        z_vals    = ls_pgram_norm[i]
    
        y_edges = centers_to_edges(y_centers)
    
        for j in range(len(y_centers)):
            verts.append([
                (x_edges[i],   y_edges[j]),
                (x_edges[i+1], y_edges[j]),
                (x_edges[i+1], y_edges[j+1]),
                (x_edges[i],   y_edges[j+1])
            ])
            colors.append(z_vals[j])
    
    colors = np.asarray(colors)



    pc = PolyCollection(
        verts,
        array=colors,
        cmap="plasma",
        linewidths=0,
        zorder=0
    )
    
    # vmin = np.min(ls_pgram_norm[-1])
    # vmax = 1.5 * np.max(ls_pgram_norm[-1])
    # pc.set_clim(vmin, vmax)

    ax[0].add_collection(pc)
    ax[0].autoscale()
    
    cbar = fig.colorbar(pc, ax=ax[0])
    cbar.set_label("Normalised LS Power [0-1]", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax[0].set_ylabel("Period (hours)\n", fontsize=fontsize)
    #ax[0].set_xlabel("Length of archive (days)", fontsize=fontsize)

    # Set limits
    ax[0].set_ylim([np.min(y_centers), np.max(y_centers)])  # based on the lims of last loop

    xlims = [np.min(x_edges), np.max(x_edges)]
    for a in ax:
        a.set_xlim(xlims)



    for l in [12, 24, 36, 48]:

        print('guava', l, ax[0].get_xlim()[1])
        ax[0].annotate(
            str(l),
            (ax[0].get_xlim()[1], l),
            xytext=(ax[0].get_xlim()[1]*1.025, l),
            xycoords='data',
            textcoords='data',#ax[0].get_xaxis_transform(),
            color=annotation_colour,
            fontsize=fontsize,
            va='center', ha='left',
            arrowprops=dict(facecolor=annotation_colour, shrink=0.05),
            clip_on=False)
        

    # ------------------------
    # PLOTTING THE TOP POINTS IN EACH PERIODOGRAM
    # fig, ax = plt.subplots()
    ax[1].plot(length_days, top_peaks_period[:, 0], linewidth=0.,
               marker='*', markersize=fontsize, fillstyle='none',
               alpha=0.65, markeredgecolor='black', label="1", zorder=1.)

    peak_colors = ["#6c93c5", "#8f9e49", "#a763bd", "#ca5b56"]
    mrs = ["o", "^", "x", "+"]
    for i in range(1, n_peaks):
        ax[1].plot(length_days, top_peaks_period[:, i], linewidth=0.,
                marker=mrs[i-1], markersize=0.5*fontsize, fillstyle='none',
                alpha=0.65, label=str(i+1), color=peak_colors[i-1], zorder=i*0.1)#,
                #markeredgecolor='deeppink')

    ax[1].legend(loc="upper right", fontsize=fontsize)

    ax[1].set_ylabel("Period of LS peak (hours)\n", fontsize=fontsize)

    for l in [12, 24, 36, 48]:
        ax[1].annotate(str(l), (ax[1].get_xlim()[1], l),
                    xytext=(ax[1].get_xlim()[1]*1.025, l),
                    xycoords='data',
                    color=annotation_colour, fontsize=fontsize,
                    va='center', ha='left',
                    arrowprops=dict(facecolor=annotation_colour, shrink=0.05),
                    clip_on=False)
    
    ax[1].axhline(24., linewidth=1.5, linestyle='dashed', color='black', zorder=0.)

    ax[1].set_ylim([17, 40])

 
    
    # Variability in AKR
    # ax[2].plot(length_days, mean_akr_i,
    #            color='black', zorder=1., label='Mean AKR intensity')
    
    # ax[2].fill_between(length_days,
    #                    mean_akr_i + std_akr_i,
    #                    mean_akr_i - std_akr_i,
    #                    color='grey', alpha=0.75, zorder=0., label='$\sigma$')
    ax[2].plot(length_days, std_akr_i, color='black', linewidth=1.5)
    ax[2].set_ylabel("Standard Deviation in\nAKR Integrated Power (W $sr^{-1}$)",
                     fontsize=fontsize)
    

    # ax[2].legend(loc="upper right", fontsize=fontsize)

    # Variability in SMR, SME, SML
    lines = []
    for k, (index, n, c) in enumerate(zip([std_SMR, std_SME, std_SML],
                                       ["SMR", "SME", "SML"],
                                       ["black", "darkviolet", "orange"]
                                       )):
        if n == "SMR":
            twax = ax[3].twinx()
            l, = twax.plot(length_days, index, color=c, label="$\sigma$" + n)
        else:
            l, = ax[3].plot(length_days, index, color=c, label="$\sigma$" + n)
        lines.append(l)

    #lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    
    ax[3].legend(lines, labels, loc="upper right", fontsize=fontsize)

    ax[3].set_ylabel("Standard Deviation in\nSME, SML (nT)", fontsize=fontsize)
    twax.set_ylabel("Standard Deviation in SMR (nT)", fontsize=fontsize)
    twax.tick_params(labelsize=fontsize)

    # fontsize on ax ticks

    
    cbar_pos = ax[0].get_position().bounds
    for (m, a) in enumerate(ax):
        a.tick_params(labelsize=fontsize)
        a.set_xlabel("Length of archive (days)", fontsize=fontsize)



        if m > 0:
            # print('bananas', m)
            pos = a.get_position().bounds
            a.set_position([cbar_pos[0], pos[1], cbar_pos[2], pos[3]])
            
        t = a.text(0.01, 0.95, axes_labels[m],
                          transform=a.transAxes,
                          fontsize=fontsize, va='top', ha="left")
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))


    # Adjust margins etc
    # fig.tight_layout()

    fig.savefig(fig_png_nonlog)

def centers_to_edges(c):
    c = np.asarray(c)
    edges = np.zeros(len(c) + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - (c[1] - c[0]) / 2
    edges[-1] = c[-1] + (c[-1] - c[-2]) / 2
    return edges
    



def lomb_scargle_expanding(n_rand=100):
    """
    Run the Lomb-Scargle analysis, gradually expanding n_rand random intervals.

    Returns
    -------
    None.

    """
    
    fig_png = os.path.join(fig_dir, "expanding_lomb_scargle_random_.png")

    # Define Lomb-Scargle freqs etc
    f_min = 1 / (48. * 60. * 60.)
    f_max = 1 / (8. * 60. * 60.)
    samples_per_peak = 5

    freq_column = "ipwr_100_400kHz"

    annotation_colour = 'purple'

    # # Read in interval data
    # interval_options = read_and_tidy_data.return_test_intervals()
    # interval_details = interval_options.loc[
    #     interval_options.tag == "cassini_flyby"]
    # interval_stime = interval_details.stime.iloc[0]
    # interval_etime = interval_details.etime.iloc[0]
    # interval_midtime = ((interval_etime - interval_stime) /
    #                     2.) + interval_stime

    # Read in *all* AKR integrated power
    akr_df = read_and_tidy_data.select_akr_intervals("full_archive")


    # Choose random start points
    np.random.seed(0)
    random_starts = np.random.choice(np.array(range(len(akr_df))), n_rand,
                                     replace=False)

    starts = akr_df.datetime.iloc[random_starts].to_numpy()

    # Sliding parameters
    slides = 50
    slide_width = pd.Timedelta(days=5)
    slide_width_multiplier = np.linspace(0, slides, slides+1)
    first_window_width = pd.Timedelta(days=30) 

    t_length = np.full((n_rand, slides + 1), pd.NaT)
    period_of_peak = np.full((n_rand, slides + 1), np.nan)

    for j in range(n_rand):
        print('Analysing random window ', j)
 
        x_lim = [starts[j],
                 starts[j] + first_window_width + (slides * slide_width)]
    
        ut_s, ut_e, length = [], [], []
        for i, factor in enumerate(slide_width_multiplier):
    
            ut_s.append(starts[j])
            ut_e.append(starts[j] + first_window_width + (factor * slide_width))
            length.append((ut_e[i] - ut_s[i]).total_seconds())
    
        ut_s = np.array(ut_s)
        ut_e = np.array(ut_e)
        # breakpoint()
        # length = np.array(length)
        # breakpoint()
        t_length[j, :] = length
        # t_length = np.append(t_length, length)
    
        # LS analysis here
        variable_freqs = []
        variable_periods = []
        ls_pgram =[]

        peak_freq = np.full(slides + 1, np.nan)
        peak_height = np.full(slides + 1, np.nan)
        for i in range(slides + 1):
            # Subselect AKR df
            subset_df = akr_df.loc[(akr_df.datetime >= ut_s[i]) &
                                   (akr_df.datetime <= ut_e[i]),
                                   :].reset_index(drop=True)
    
            output = lomb_scargle.generic_lomb_scargle(subset_df.unix,
                                                       subset_df[freq_column],
                                                       f_min, f_max,
                                                       n0=samples_per_peak)
    
            variable_freqs.append(output[1])
            ls_pgram.append(output[2])
            variable_periods.append(
                periodicity_functions.freq_to_period(output[1]))
    
            arg = output[2].argmax()
            peak_height[i] = output[2][arg]
            peak_freq[i] = output[1][arg]

        period_of_peak[j, :] = periodicity_functions.freq_to_period(peak_freq)


    fig, ax = plt.subplots(figsize=(12, 6))

    length_days = t_length / (60. * 60 * 24)
    
    mean_peak = np.mean(period_of_peak, axis=0)
    median_peak = np.median(period_of_peak, axis=0)
    std = np.std(period_of_peak, axis=0)

    
    # return length_days, period_of_peak, mean_peak, median_peak, std
    # print('apple')
    # for j in range(n_rand):
    #     ax.plot(length_days[j, :], period_of_peak[j, :], linewidth=0.,
    #             marker='o', markersize=fontsize,# color='deeppink',
    #             alpha=0.65, markeredgecolor='black', label=str(j))
    # print('banana')
    # ax.set_ylabel("Period of LS peak (hours)\n", fontsize=fontsize)
    # ax.set_xlabel("Length of archive (days)", fontsize=fontsize)
    # # ax.set_ylim([20, 40])
    # ax.axhline(24., linestyle='dashed', linewidth=2.,
    #            zorder=0.5, color='black')
    # ax.text(20, 24.25, "24 hours", ha='left', va='bottom',
    #         transform=ax.transData, fontsize=fontsize, color='black')
    # # ax.legend()

    # # Adjust margins etc
    # fig.tight_layout()

    # Save to file
    # fig.savefig(fig_png)


    
    # x_bin_centres = np.linspace(first_window_width.days, (first_window_width + (slide_width_multiplier[-1] * slide_width)).days, slides + 1)
    # x_bin_edges = x_bin_centres - (slide_width.days / 2.)
    # x_bin_edges = np.append(x_bin_edges, x_bin_centres[-1] + (slide_width.days / 2.))
    # y_bin_edges = np.linspace(5, 50, 91)
    
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.hist2d(length_days.flatten(), period_of_peak.flatten(), bins=[x_bin_edges, y_bin_edges], cmin=1, cmap='grey')

    # x_bin_centres = np.linspace(first_window_width.days, (first_window_width + (slide_width_multiplier[-1] * slide_width)).days, slides + 1)
    # x_bin_edges = x_bin_centres - (slide_width.days / 2.)
    # x_bin_edges = np.append(x_bin_edges, x_bin_centres[-1] + (slide_width.days / 2.))
    # y_bin_edges = np.linspace(5, 50, 46) + 0.5
    
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.hist2d(length_days.flatten(), period_of_peak.flatten(), bins=[x_bin_edges, y_bin_edges], cmin=1, cmap='copper', norm='log')
    # ax.set_facecolor('gainsboro')

    # x_bin_centres = np.linspace(first_window_width.days, (first_window_width + (slide_width_multiplier[-1] * slide_width)).days, slides + 1)
    # x_bin_edges = x_bin_centres - (slide_width.days / 2.)
    # x_bin_edges = np.append(x_bin_edges, x_bin_centres[-1] + (slide_width.days / 2.))
    # y_bin_edges = np.linspace(5, 50, 46) + 0.5
    
    # fig, ax = plt.subplots(figsize=(12, 6))
    # h, x, y, im = ax.hist2d(length_days.flatten(), period_of_peak.flatten(), bins=[x_bin_edges, y_bin_edges], cmin=1, cmap='copper', norm='log')
    # ax.set_facecolor('gainsboro')
    
    # ax.set_ylabel("Period of LS peak (hours)\n", fontsize=fontsize)
    # ax.set_xlabel("Length of archive (days)", fontsize=fontsize)

    # ax.axhline(24., linestyle='dashed', linewidth=2., color='black')
    # ax.text(20, 24.25, "24 hours", ha='left', va='bottom',
    #         transform=ax.transData, fontsize=fontsize, color='black')

    
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.ax.tick_params(labelsize=fontsize)
    # cbar.set_label('Occurrence (counts)', fontsize=fontsize)
    
    # x_bin_centres = np.linspace(first_window_width.days, (first_window_width + (slide_width_multiplier[-1] * slide_width)).days, slides + 1)
    # x_bin_edges = x_bin_centres - (slide_width.days / 2.)
    # x_bin_edges = np.append(x_bin_edges, x_bin_centres[-1] + (slide_width.days / 2.))
    # y_bin_edges = np.linspace(5, 50, 46) + 0.5

    # fig, ax = plt.subplots(figsize=(12, 6))
    # h, x, y, im = ax.hist2d(length_days.flatten(), period_of_peak.flatten(), bins=[x_bin_edges, y_bin_edges], cmin=1, cmap='magma', norm='log')
    # ax.set_facecolor('gainsboro')

    # ax.set_ylabel("Period of LS peak (hours)\n", fontsize=fontsize)
    # ax.set_xlabel("Length of archive (days)", fontsize=fontsize)

    # for l in [12, 24, 36, 48]:
    #     ax.annotate(str(l), (ax.get_xlim()[1], l), xytext=(ax.get_xlim()[1]*1.025, l), xycoords='data', color=annotation_colour, fontsize=fontsize, va='center', ha='left',
    #         arrowprops=dict(facecolor=annotation_colour, shrink=0.05))
        

    # cbar = fig.colorbar(im, ax=ax)
    # cbar.ax.tick_params(labelsize=fontsize)
    # cbar.set_label('Occurrence (counts)', fontsize=fontsize)
    
    
        
    x_bin_centres = np.linspace(first_window_width.days, (first_window_width + (slide_width_multiplier[-1] * slide_width)).days, slides + 1)
    x_bin_edges = x_bin_centres - (slide_width.days / 2.)
    x_bin_edges = np.append(x_bin_edges, x_bin_centres[-1] + (slide_width.days / 2.))
    y_bin_edges = np.linspace(5, 50, 46) + 0.5
    
    #fig, ax = plt.subplots(figsize=(12, 14))
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    h, x, y, im = ax1.hist2d(length_days.flatten(), period_of_peak.flatten(), bins=[x_bin_edges, y_bin_edges], cmin=1, cmap='magma', norm='log')
    ax1.set_facecolor('gainsboro')
    
    ax1.set_ylabel("Period of LS peak (hours)\n", fontsize=fontsize)
    ax1.set_xlabel("Length of archive (days)", fontsize=fontsize)
    
    for l in [12, 24, 36, 48]:
        # ax.axhline(24., linestyle='dashed', linewidth=2., color='purple')
        # ax.text(20, 24.25, "24 hours", ha='left', va='bottom',
        #        transform=ax.transData, fontsize=fontsize, color='black')
        ax1.annotate(str(l), (ax1.get_xlim()[1], l),
                     xytext=(ax1.get_xlim()[1]*1.025, l),
                     xycoords='data',
                     color=annotation_colour, fontsize=fontsize,
                     va='center', ha='left',
                     arrowprops=dict(facecolor=annotation_colour, shrink=0.05))
        
    
    cbar = fig.colorbar(im, ax=ax1)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label('Occurrence (counts)', fontsize=fontsize)
    
    
    ax_cbar_pos = ax1.get_position().bounds
    # Average
    ax2.plot(length_days[0, :], mean_peak, color='crimson', linewidth=2., label='APPROXMean')
    ax2.plot(length_days[0, :], median_peak, color='grey', linewidth=2., label='APPROXMedian')
    
    ax2.set_ylabel('Average period of\nLS peak (hours)\nAPPROX ATM', fontsize=fontsize)
    ax2.set_xlabel('Length of archive (days)', fontsize=fontsize)
    ax2.legend(fontsize=fontsize)
    
    # Standard dev
    ax3.plot(length_days[0, :], std, color='crimson', linewidth=2., label='Standard\nDeviation')
    
    ax3.set_ylabel('Standard deviation of\nperiods of LS peak (hours)', fontsize=fontsize)
    ax3.set_xlabel('Length of archive (days)', fontsize=fontsize)
    
    # Make width same as (a)
    pos = ax2.get_position().bounds
    ax2.set_position([ax_cbar_pos[0], pos[1], ax_cbar_pos[2], pos[3]])
    pos = ax3.get_position().bounds
    ax3.set_position([ax_cbar_pos[0], pos[1], ax_cbar_pos[2], pos[3]])
        
    
    for i, a in enumerate([ax1, ax2, ax3]):
        # increase tick fontsize
        a.tick_params(labelsize=fontsize)
        t = a.text(0.02, 0.95, axes_labels[i], transform=a.transAxes,
                       fontsize=1.5 * fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    # Save to file
    fig.savefig(fig_png)
    print('NEED TO MAKE AXES LIMITS OF B AND C ALIGN WITH A')


def plot_sliding_spectrogram():
    """
    Plot the LSSA modulation spectrograms.

    Returns
    -------
    None.

    """

    ylim = [22., 26.]
    xlim = [pd.Timestamp(1995, 1, 1, 0), pd.Timestamp(2005, 1, 1, 0)]

    # Read Wind position data
    years = np.arange(1995, 2004 + 1)
    wind_position_df = read_wind_position.concat_data_years(years)

    # Read Sunspot data
    sunspot_df = read_sunspot_n.read_monthly_smoothed_sunspot()

    for k, (freq, freq_tit) in enumerate(zip(['high', 'low'],
                                             ['100 - 800 kHz',
                                              '30 - 100 kHz'])):
        # define fig filename
        # initialise fig
        fig_png = os.path.join(fig_dir, "modulation_spectrogram_" +
                               freq + '.png')

        fig, axes = plt.subplots(nrows=5, figsize=(12, 18))

        for i, (lon, lon_tit) in enumerate(zip(['sun', 'sc'],
                                               ['$\lambda_{sun}$',
                                                '$\lambda_{SC}$'])):

            out_dict = utility.read_wu_period(lon, freq)

            time_axis = np.array(out_dict['timestamp'])  # 189
            period_axis = np.array(out_dict['period_hours'])  # 1201
            spectrogram = np.array(out_dict['spectrogram'])  # 1201, 189

            # Restrict spectrogram
            t_ind, = np.where(time_axis <= pd.Timestamp(2005, 1, 10))
            ind, = np.where((period_axis >= ylim[0]) &
                            (period_axis <= ylim[1]))
            time_axis = time_axis[t_ind]
            period_axis = period_axis[ind]
            spectrogram = spectrogram[ind, :]
            spectrogram = spectrogram[:, t_ind]

            norm = mpl.colors.Normalize(vmin=np.nanmin(spectrogram),
                                        vmax=0.9 * np.nanmax(spectrogram))

            psm = axes[i].pcolormesh(time_axis, period_axis, spectrogram,
                                     norm=norm, cmap='plasma',
                                     shading='nearest')
            cbar = fig.colorbar(psm, ax=axes[i])
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label('Normalised Power [proxy]', fontsize=fontsize)

            axes[i].set_title(freq_tit + ', organised by ' + lon_tit,
                              fontsize=fontsize)

        # Difference plot here on axes[2]
        sun_dict = utility.read_wu_period('sun', freq)
        sc_dict = utility.read_wu_period('sc', freq)

        time_axis = np.array(sun_dict['timestamp'])  # 189
        period_axis = np.array(sun_dict['period_hours'])  # 1201
        spectrogram = np.array(sun_dict['spectrogram']
                               - sc_dict['spectrogram'])

        # Restrict axes
        t_ind, = np.where(time_axis <= pd.Timestamp(2005, 1, 10))
        ind, = np.where((period_axis >= ylim[0]) &
                        (period_axis <= ylim[1]))
        time_axis = time_axis[t_ind]
        period_axis = period_axis[ind]
        spectrogram = spectrogram[ind, :]
        spectrogram = spectrogram[:, t_ind]

        norm = mpl.colors.CenteredNorm()

        psm = axes[2].pcolormesh(time_axis, period_axis, spectrogram,
                                 norm=norm, cmap='PuOr', shading='nearest')

        axes[2].set_title('Difference plot: $\lambda_{sun}$ - $\lambda_{SC}$',
                          fontsize=fontsize)
        cbar = fig.colorbar(psm, ax=axes[2])
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label('Difference', fontsize=fontsize)

        # Wind LT on axes[3]
        axes[3].plot(wind_position_df.datetime,
                     wind_position_df.decimal_gseLT,
                     color='black', linewidth=1.)

        # Plot a dot for every nightside observation
        nightside_lt_dtime = wind_position_df.loc[
            (wind_position_df.decimal_gseLT >= 18.) |
            (wind_position_df.decimal_gseLT <= 6.)]
        axes[3].plot(nightside_lt_dtime.datetime,
                     np.repeat(23.5, len(nightside_lt_dtime)),
                     linewidth=0., color='lightcoral', marker='.')

        # Sunspot Number on axes[4]
        axes[4].plot(sunspot_df.month_mid_dtime, sunspot_df.smooth_s_n,
                     color='darkmagenta', linewidth=2.)
        axes[4].set_ylim(top=215)

        # Formatting
        axes[3].set_ylabel('Wind Spacecraft LT\n(GSE, hours)',
                           fontsize=fontsize)
        axes[4].set_ylabel('Sunspot Number', fontsize=fontsize)
        for (j, a) in enumerate(axes):
            a.tick_params(labelsize=fontsize)
            a.set_xlabel('Year', fontsize=fontsize)
            t = a.text(0.02, 0.92, axes_labels[j], transform=a.transAxes,
                       fontsize=fontsize, va='top', ha='left')
            t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

            # a.set_ylim(ylim)
            a.set_xlim(xlim)
            if j < 3:
                a.set_ylabel('Period [proxy, hours]', fontsize=fontsize)

        fig.tight_layout()

        # Adjust axis width
        pos_cbar = axes[2].get_position().bounds
        pos_lt = axes[3].get_position().bounds
        axes[3].set_position([pos_cbar[0], pos_lt[1],
                              pos_cbar[2], pos_lt[3]])

        pos_sn = axes[4].get_position().bounds
        axes[4].set_position([pos_cbar[0], pos_sn[1],
                              pos_cbar[2], pos_sn[3]])

        fig.savefig(fig_png)


def LS_solar_cyc(sunspot_n_fdict={'color': 'lightgrey',
                                  'label': 'Mean',
                                  'linewidth': 1.},
                 smoothsn_fdict={'color': 'black',
                                 'label': '13-month Smoothed'}):
    """
    Run Lomb Scargle wrt Solar cycle.

    Parameters
    ----------
    sunspot_n_fdict : dictionary, optional
        Format details for sunspot number. The default is
        {'color': 'lightgrey', 'label': 'Mean', 'linewidth': 1.}.
    smoothsn_fdict : dictionary, optional
        Format details for smoothed sunspot number. The default is
        {'color': 'black', 'label': '13-month Smoothed'}.

    Returns
    -------
    None.

    """
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
    T = (pd.Timestamp(2005, 1, 1, 0) - pd.Timestamp(
        1995, 1, 1, 0)).total_seconds()
    f_min, f_max, N_f, freqs = lomb_scargle.define_frequency_bins(T,
                                                                  f_min, f_max,
                                                                  n0=5)
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

    # Initialise plotting
    fig, ax = plt.subplots(nrows=2, figsize=(16, 8))

    # Plot Lomb-Scargles
    X, Y = np.meshgrid(np.append(years-0.5, years[-1] + 0.5), period_edges)
    pcm = ax[0].pcolormesh(X, Y, ls_pgram, cmap='plasma')
    cbar = plt.colorbar(pcm, ax=ax[0],
                        label="Lomb-Scargle\nNormalised Amplitude")
    cbar.ax.tick_params(labelsize=fontsize)

    ax[0].set_ylabel('Period (hours)', fontsize=fontsize)

    ax_pos = ax[0].get_position().bounds

    # Plot Sunspot Number
    ax[1].plot(sunspot_df.year_frac, sunspot_df.mean_s_n, **sunspot_n_fdict)
    ax[1].plot(smoothed_sunspot_df.year_frac, smoothed_sunspot_df.smooth_s_n,
               **smoothsn_fdict)

    # Set limits
    sunspot_ymax = np.max(
        [sunspot_df.loc[(sunspot_df.year_frac >= stime) &
                        (sunspot_df.year_frac <= etime), 'mean_s_n'].max(),
         smoothed_sunspot_df.loc[(smoothed_sunspot_df.year_frac >= stime) &
                                 (smoothed_sunspot_df.year_frac <= etime),
                                 'smooth_s_n'].max()])
    ax[1].set_ylim(0, 1.1 * sunspot_ymax)
    ax[1].set_xlim(stime, etime)

    ax[1].set_ylabel('Sunspot Number', fontsize=fontsize)
    ax[1].set_xlabel('Year', fontsize=fontsize)
    ax[1].legend(fontsize=fontsize, loc='upper right')
    # Make width same as (a)
    pos = ax[1].get_position().bounds
    ax[1].set_position([ax_pos[0], pos[1], ax_pos[2], pos[3]])

    # Formatting
    for j, a in enumerate(ax):
        a.set_xlim(stime, etime)
        a.tick_params(labelsize=fontsize)

        t = a.text(0.02, 0.92, axes_labels[j], transform=a.transAxes,
                   fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))
