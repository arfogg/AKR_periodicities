# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:28:54 2024

@author: A R Fogg
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\wind_utility')
import read_wind_position

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\misc_utility')
import draw_emoji


fig_dir = os.path.join("C:" + os.sep,
                       r"Users\admin\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")


# find period where Wind on the nightside
# plot orbits for the different periods


def find_nigthside_period():
    """
    Find periods where Wind is on the nightside of the Earth.

    Returns
    -------
    midnight_periods_df : pd.DataFrame
        DataFrame containing the start (sdtime), end(edtime), and length
        of the periods. Sorted so that longest period is first.

    """

    midnight_periods_csv = os.path.join(data_dir, "midnight_periods.csv")

    if pathlib.Path(midnight_periods_csv).is_file():
        midnight_periods_df = pd.read_csv(midnight_periods_csv, delimiter=',',
                                          float_precision='round_trip',
                                          parse_dates=['sdtime', 'edtime'])
    else:
        # Define desired nightside range
        lower_lt = 18.
        upper_lt = 6.
        # Over these years
        years = np.arange(1995, 2004 + 1)

        # Read Wind position data
        position_df = read_wind_position.concat_data_years(years)

        # Find timestamps where wind is in nightside
        nightside_df = position_df.\
            query("@lower_lt <= decimal_gseLT | @upper_lt >= decimal_gseLT").\
            reset_index(drop=True)
        # Timedelta between each row
        nightside_df['time_diff'] = nightside_df['datetime'].diff()

        # Position data temporal resolution
        temp_res = pd.Timedelta(minutes=12)
        sdtimes = []
        edtimes = []
        length = []
        next_start_i = 0
        # For each timedelta, find where the jump is greater than the temporal
        # resolution. This jump indicates a gap between two periods.
        for i in range(1, len(nightside_df)):
            if nightside_df['time_diff'].iloc[i] > temp_res:
                sdtimes.append(nightside_df['datetime'].iloc[next_start_i])
                edtimes.append(nightside_df['datetime'].iloc[i-1])
                length.append(nightside_df['datetime'].iloc[i-1]
                              - nightside_df['datetime'].iloc[next_start_i])
                next_start_i = i

        midnight_periods_df = pd.DataFrame({'sdtime': sdtimes,
                                            'edtime': edtimes,
                                            'length': length})
        midnight_periods_df.sort_values('length', ascending=False,
                                        inplace=True, ignore_index=True)

        midnight_periods_df.to_csv(midnight_periods_csv, index=False)

    return midnight_periods_df


def plot_trajectory(sdtime, edtime, wind_position_df, ax, fontsize=15):
    """
    Draw the Wind trajectory onto a parsed axis.

    Parameters
    ----------
    sdtime : pd.Timestamp
        Start time of the orbital path.
    edtime : pd.Timestamp
        End time of the orbital path.
    wind_position_df : pd.DataFrame
        Pandas DataFrame containing columns datetime, x_gse, y_gse.
    ax : matplotlib axis
        Matplotlib axis to draw the trajectory on.
    fontsize : int, optional
        Fontsize parsed to matplotlib. The default is 15.

    Returns
    -------
    ax : matplotlib axis
        Completed trajectory plot.

    """

    # Limit Trajectory DataFrame to desired time window
    position_df = wind_position_df.loc[
        (wind_position_df.datetime >= sdtime) &
        (wind_position_df.datetime <= edtime)].copy()

    # Dashed lines at x, y == 0
    ax.axvline(0, linestyle='dashed', color='black', linewidth=1.)
    ax.axhline(0, linestyle='dashed', color='black', linewidth=1.)

    # Plot Trajectory in the X-Y GSM plane
    ax.plot(position_df['x_gse'], position_df['y_gse'],
            linewidth=1.0, color='dimgrey', label='Wind\nTrajectory')

    # Invert x axis so Sun is on left
    ax.invert_xaxis()

    # Label the Earth
    ax.plot(0, 0, marker="*", markersize=fontsize, color='mediumblue',
            linewidth=0., label='Earth')
    # # Draw the Earth
    # draw_emoji.draw_emoji('earth_africa', ax, coords=(0., 0.), zoom=0.03)

    # Label the beginning and end of the orbit
    ax.plot(position_df['x_gse'].iloc[0], position_df['y_gse'].iloc[0],
            marker="o", markersize=0.6*fontsize, color='green', linewidth=0.,
            label='Start')
    ax.plot(position_df['x_gse'].iloc[-1], position_df['y_gse'].iloc[-1],
            marker="s", markersize=0.6*fontsize, color='red', linewidth=0.,
            label='End')

    # Formatting
    ax.set_xlabel('$X$ $GSE$ $(R_{E})$', fontsize=fontsize)
    ax.set_ylabel('$Y$ $GSE$ $(R_{E})$', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc='upper right')

    return ax


def lt_hist(sdtime, edtime, wind_position_df, ax,
            lt_bin_centres=np.linspace(0, 23, 24), lt_bin_width=1.,
            fontsize=15, bar_fmt={'color': 'grey', 'edgecolor': 'black',
                                  'alpha': 0.5},
            draw_ticks=True):
    # first bin is the one going over midnight
    
    # Limit Trajectory DataFrame to desired time window
    position_df = wind_position_df.loc[
        (wind_position_df.datetime >= sdtime) &
        (wind_position_df.datetime <= edtime)].copy()

    # Initialise arrays
    n_obs = np.full(lt_bin_centres.size, np.nan)
    # n_pc = np.empty(lt_bin_centres.size, np.nan)

    n_total = len(position_df.decimal_gseLT)

    for i in range(lt_bin_centres.size):
        if i == 0:
            # Midnight bin
            n_obs[i] = len(position_df.loc[
                (position_df.decimal_gseLT > (24-(lt_bin_width/2))) |
                (position_df.decimal_gseLT <=
                 lt_bin_centres[i]+(lt_bin_width/2)),
                'decimal_gseLT'])
        else:
            # All other bins
            n_obs[i] = len(position_df.loc[
                (position_df.decimal_gseLT >
                 lt_bin_centres[i]-(lt_bin_width/2)) &
                (position_df.decimal_gseLT <=
                 lt_bin_centres[i]+(lt_bin_width/2)),
                'decimal_gseLT'])

    n_pc = (n_obs / n_total) * 100.

    # Calculate bin centres in degrees
    theta = 2 * np.pi * (lt_bin_centres/24.)
    ax.bar(theta, n_pc, width=((2.*np.pi)/(len(lt_bin_centres))),
           bottom=0.0, **bar_fmt)

    ax.set_theta_zero_location('S')
    ax.set_theta_direction('counterclockwise')

    if draw_ticks:
        xtickpos = []
        xticklab = []
        for i in range(0, 12):
            xtickpos.append((i * 2) * ((2 * np.pi)/24.0))
            xticklab.append('%02d' % (i * 2))
        ax.set_xticks(xtickpos)
        ax.set_xticklabels(xticklab)

        label_position = ax.get_rlabel_position()
        ax.text(np.radians(label_position + 20), ax.get_rmax()/2.,
                '% observing time', rotation=label_position-83,
                ha='center', va='center', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.legend(fontsize=fontsize)

    return ax
