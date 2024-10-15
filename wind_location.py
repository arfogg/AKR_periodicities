# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:28:54 2024

@author: A R Fogg
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\readers')
import read_wind_position

# find period where Wind on the nightside
# plot orbits for the different periods


def find_nigthside_period(lower_lt=18., upper_lt=6.):

    years = np.arange(1995, 2004 + 1)
    position_df = read_wind_position.concat_data_years(years)

    # only timestamps where wind is in nightside
    nightside_df = position_df.query("@lower_lt <= decimal_gseLT | @upper_lt >= decimal_gseLT").reset_index(drop=True)
    # timedelta between each row
    nightside_df['time_diff'] = nightside_df['datetime'].diff()
    
    temp_res = pd.Timedelta(minutes=12)
    sdtimes = []
    edtimes = []
    length = []
    next_start_i = 0
    for i in range(1, len(nightside_df)):
        if nightside_df['time_diff'].iloc[i] > temp_res:
            sdtimes.append(nightside_df['datetime'].iloc[next_start_i])
            edtimes.append(nightside_df['datetime'].iloc[i-1])
            length.append(nightside_df['datetime'].iloc[i-1] - nightside_df['datetime'].iloc[next_start_i])
            next_start_i = i
    
    midnight_periods_df = pd.DataFrame({'sdtime': sdtimes, 'edtime': edtimes, 'length': length})
    midnight_periods_df.sort_values('length', ascending=False, inplace=True, ignore_index=True)

    # NEED TO SAVE THIS TO FILE
    
def plot_trajectory(sdtime, edtime, wind_position_df, fontsize=15):
    
    # eventually parse these
    # sdtime = pd.Timestamp(1999, 8, 15, 0)
    # edtime = pd.Timestamp(1999, 9, 14, 23, 59)
    # years = np.arange(sdtime.year, edtime.year+1)
    # position_df = read_wind_position.concat_data_years(years)

    # Limit Trajectory DataFrame to desired time window
    position_df = wind_position_df.loc[(wind_position_df.datetime >= sdtime)
                                       & (wind_position_df.datetime <= edtime)].copy()


    fig, ax = plt.subplots(figsize=(6, 6))

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
    # ASPECT RATIO
    # LEGEND