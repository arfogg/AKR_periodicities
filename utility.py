# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:18:04 2024

@author: A R Fogg
"""

import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from spacepy import coordinates as coord
from spacepy.time import Ticktock

import aaft

import read_and_tidy_data

fig_dir = os.path.join("C:" + os.sep,
                       r"Users\Alexandra\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")


def interpolate_mlt(desired_timestamps, data_df, mlt_flag='decimal_gseLT'):
    """
    Interpolate magnetic local time / MLT. Requires special
    approach as MLT is a periodic parameter (i.e. 24 == 0).

    Parameters
    ----------
    desired_timestamps : np.array
        pd.Timestamps for MLT to be calculated at.
    data_df : pd.DataFrame
        DataFrame containing MLT (mlt_flag) as a function of
        'unix'. These are used to create the interpolation
        function.
    mlt_flag : string, optional
        Column in data_df containing MLT. The default is
        'decimal_gseLT'.

    Returns
    -------
    out_mlt : np.array
        Interpolated MLT as a function of desired_timestamps.

    """
    # Unwrap MLT
    unwrapped_mlts = np.unwrap(data_df[mlt_flag], period=24)
    # Generate interpolation function
    mlt_func = interpolate.interp1d(data_df['unix'], unwrapped_mlts)

    # Initialise empty MLT array
    out_mlt = np.full(desired_timestamps.size, np.nan)

    # For each desired timestamp, estimate MLT
    for i in range(desired_timestamps.size):
        unwrapped_interp_mlt = mlt_func(
            pd.Timestamp(desired_timestamps[i]).timestamp())
        out_mlt[i] = unwrapped_interp_mlt % 24

    return out_mlt


def generate_random_phase_surrogate(data, plot=False):

    print('Generating Random Phase Surrogate')
    surrogate = aaft.AAFTsur(data)

    if plot:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

        # Histogram
        bins = np.linspace(np.nanmin(data), np.nanmax(data), 51)
        ax[0].hist(data, bins=bins, label='Data', color='lightgrey')
        ax[0].hist(surrogate, bins=bins, label='Surrogate',
                   histtype='step', color='palevioletred')

        ax[0].set_xlabel('Data units')
        ax[0].set_ylabel('Occurrence')
        ax[0].set_title('Distribution')
        ax[0].legend()

        # Timeseries
        i_lims = [0, 100]
        ax[1].plot(data[i_lims[0]:i_lims[1]], color='grey', label='Data')
        ax[1].plot(surrogate[i_lims[0]:i_lims[1]], color='palevioletred',
                   label='Surrogate')

        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Magnitude')
        ax[1].set_title('Timeseries')
        ax[1].legend()

        fig.tight_layout()

    return surrogate

# def chunk_gse_to_geo(data):
    
#     print(data['datetime'][0], data['datetime'][len(data)-1])
    
# #     o_csv = os.path.join(data_dir, 'coord_transform')

# def full_archive_geo_coord_parallel():
    
#     interval_options = read_and_tidy_data.return_test_intervals()    
#     data = read_and_tidy_data.select_akr_intervals(interval_options['tag'][0])

#     chunk_gse_to_geo(data)

def convert_gse_to_geo(data, rad_tag='radius', lat_tag='lat_gse',
                       lon_tag='lon_gse', unix_tag='unix'):
    # data is a DataFrame containing datetime, decimal_gseLT, lon_gse
    # dim (n,3)
    # rad, lat, lon
    
    # Check for output file
    
    input_coord_array = np.array([data[rad_tag].values,
                                        data[lat_tag].values,
                                        data[lon_tag].values]).T
    
    # Initialise coord class
    # print('banana')
    coord_cls = coord.Coords(input_coord_array, 'GSE', 'sph')
    
    # Fold in time
    # print('apple')
    coord_cls.ticks = Ticktock(data[unix_tag].values, 'UNX')
    
    # from spacepy import coordinates as coord
    # cvals = coord.Coords([[1,2,4],[1,2,2]], 'GEO', 'car')
    # cvals.x  # returns all x coordinates
    # from spacepy.time import Ticktock
    # cvals.ticks = Ticktock(['2002-02-02T12:00:00', '2002-02-02T12:00:00'], 'ISO') # add ticks
    # newcoord = cvals.convert('GSM', 'sph')
    # newcoord
    # print('guava')
    newcoord = coord_cls.convert('GEO', 'sph')
    
    geo_lat = newcoord.lati
    geo_lon = newcoord.long
    
    return geo_lat, geo_lon

def full_archive_geo_coord():

    interval_options = read_and_tidy_data.return_test_intervals()    
    data = read_and_tidy_data.select_akr_intervals(interval_options['tag'][0])

    print('full archive len ', len(data))
    #data = data.iloc[0:100000]
    geo_coords_csv = os.path.join(data_dir, "full_archive_GEO_coords.csv")
    
    if pathlib.Path(geo_coords_csv).is_file():
        out_df = pd.read_csv(geo_coords_csv, delimiter=',',
                             float_precision='round_trip')
        
        
    else:
        print('Converting from GSE to GEO coordinates for full archive')
        t0 = pd.Timestamp.now()
        print('Started at:', t0)
        # Run this over chunks, so that it's quicker
        chunk_length = 10000
        n_loops = int(np.ceil(len(data)/chunk_length))

        geo_lat, geo_lon = np.array([]), np.array([])
        for i in range(n_loops):
            print('Chunk ', i, 'time elapsed ', pd.Timestamp.now() - t0)
            starting_i = (i * chunk_length)
            ending_i = (i + 1) * chunk_length
            if ending_i > len(data):
                ending_i = len(data)

            g_lat, g_lon = convert_gse_to_geo(data[starting_i:ending_i])
            geo_lat = np.append(geo_lat, g_lat)
            geo_lon = np.append(geo_lon, g_lon)
        
        # geo_lat, geo_lon = convert_gse_to_geo(data)
        t2 = pd.Timestamp.now()
        print('Ended at:', t2)
        print('Time elapsed:', t2-t0)
        out_df = pd.DataFrame({'datetime': data['datetime'],
                               'unix': data['unix'],
                               'lon_geo': geo_lon,
                               'lat_geo': geo_lat})
        out_df.to_csv(geo_coords_csv, index=False)
        
    return out_df

def calc_longitude_of_sun(data, lon_tag='lon_gse', plot=False):
    # data is a DataFrame containing datetime, decimal_gseLT, lon_tag
    # NEED TO CHECK WITH SIYUAN ABOUT LON GSM OR GSE!!

    lon_sol = (data[lon_tag] % 360) - ((12. - data.decimal_gseLT) * 15.)
    lon_sol = lon_sol % 360

    if plot:
        fig, ax = plt.subplots()
        ax.set_facecolor('lightgrey')

        ax.hist([data[lon_tag] % 360, lon_sol], color=['white', 'blue'],
                edgecolor='black', density=True,
                label=['$\lambda_{sc}$', '$\lambda_{sun}$'])

        ax.legend()
        ax.set_xlabel("Longitude ($^{\circ}$)")
        ax.set_ylabel("Density")

    return lon_sol
