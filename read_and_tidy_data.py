# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:19:11 2024

@author: A R Fogg
"""

import os
import sys
import pathlib

import pandas as pd
import numpy as np

from scipy import interpolate

import utility

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\wind_utility')
#import read_integrated_power
import calc_integrated_power

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\readers')
import read_omni
import read_supermag

fig_dir = os.path.join("C:" + os.sep,
                       r"Users\Alexandra\Documents\figures\akr_periodicities")
data_dir = os.path.join(fig_dir, "data_quickloads")


def define_freq_bands():
    """
    Return frequency bands to integrate power over.

    Returns
    -------
    freq_bands : np.array
        Upper and lower frequencies for several bands. Of
        shape number of bands x 2.

    """
    freq_bands = np.array([[20, 50],
                           [50, 100],
                           [100, 300],
                           [300, 500],
                           [500, 700],
                           [700, 850],
                           [100, 400],
                           [100, 650],
                           [150, 400],
                           [150, 650],
                           [650, 800]
                          ])

    return freq_bands


# def read_intensity_over_freq_bands(year):
#     """
#     Run code to read in the intensity data over many frequency bands.

#     Parameters
#     ----------
#     year : int
#         Year to read data for in YYYY format.

#     Returns
#     -------
#     i_loc_df : pd.DataFrame
#         DataFrame containing datetime, location information, and the
#         integrated power across the requested bands.

#     """
#     freq_bands = define_freq_bands()

#     i_loc_df = read_integrated_power.\
#         read_integrated_power_n_freq_bands(year, 'waters',
#                                            'periodicity', freq_bands)

#     return i_loc_df


# def concat_intensity_over_freq_bands(years):

#     print('Reading integrated power data (multiple freq bands) for year: ',
#           years)

#     appended_data = []
#     for year in years:
#         temp_df = read_intensity_over_freq_bands(year)
#         appended_data.append(temp_df)

#     appended_df = pd.concat(appended_data, ignore_index=True)

#     return appended_df


def select_akr_intervals(interval):
    """
    Load in AKR intensity over a defined interval, over multiple
    frequency bands.

    Parameters
    ----------
    interval : string
        Name of defined interval. Options are defined in
        interval_options from return_test_intervals.

    Returns
    -------
    output_df : pd.DataFrame
        DataFrame containing requested AKR data.

    """

    interval_options = return_test_intervals()

    # Check parsed interval is correct
    if interval not in np.array(interval_options.tag):
        print('Parsed interval not a valid option.')
        print('Exiting...')
        return
    else:
        selected = interval_options.loc[interval_options.tag == interval,
                                        :].reset_index()

    # Which years are needed?
    # years = list(range(selected.stime[0].year, selected.etime[0].year+1, 1))

    syear = selected.stime[0].year
    eyear = selected.etime[0].year

    # Read in data
    # akr_df = concat_intensity_over_freq_bands(years)
    freq_bands = define_freq_bands()
    # akr_df = calc_integrated_power.read_run_store_power(syear, eyear,
    #                                                     freq_bands=freq_bands)
    akr_df = calc_integrated_power.read_run_store_power_with_wind_location(
        syear, eyear, freq_bands=freq_bands)
    # Select only the interval requested
    akr_df = akr_df.loc[(akr_df.datetime >= selected.stime[0]) &
                        (akr_df.datetime <= selected.etime[0]),
                        :].reset_index(drop=True)
    akr_df['unix'] = [t.timestamp() for t in akr_df.datetime]


    # # ----- ROUNDED VERSION -----
    # # Round to a nice, clean, 3 minute resolution
    # # (this is important for FFTs etc)
    # print('Temporally rounding AKR onto an even resolution')
    # # s_time = akr_df.datetime_ut.iloc[0].ceil(freq='min')
    # # e_time = akr_df.datetime_ut.iloc[-1].floor(freq='min')

    # new_time_axis = akr_df.datetime_ut.dt.round(freq='min')
    # new_time_axis = new_time_axis[:-1]
    # unix_time_axis = [t.timestamp() for t in new_time_axis]
    # rounded_akr_df = pd.DataFrame({'datetime_ut': new_time_axis,
    #                                'unix': unix_time_axis})

    # func = interpolate.interp1d(akr_df.unix, akr_df['P_Wsr-1_100_650_kHz'])
    # rounded_akr_df['P_Wsr-1_100_650_kHz'] = func(unix_time_axis)

    # # Interpolating Wind location
    # tags = np.array(['x_gse', 'y_gse', 'z_gse', 'lat_gse', 'lon_gse',
    #                  'x_gsm', 'y_gsm', 'z_gsm', 'lat_gsm', 'lon_gsm',
    #                  'radius', 'decimal_gseLT'])
    # for t in tags:
    #     print('Interpolating Wind position: ', t)

    #     if t == 'decimal_gseLT':
    #         print('Interpolating Local Time:')
    #         print('Using special approach as LT is periodic')

    #         rounded_akr_df[t] = utility.interpolate_mlt(
    #                 np.array(rounded_akr_df.datetime_ut), akr_df,
    #                 mlt_flag='decimal_gseLT')

    #     else:
    #         # Create a function defining the interpolation between points
    #         # in time defined by akr_df.unix, over the variable akr_df[t]
    #         func = interpolate.interp1d(akr_df.unix, akr_df[t])
    #         # Feed the function the time resolution of the akr data
    #         # (defined by interpolated_akr_df.unix), and get the values
    #         # of the current tag out, feed straight back into the
    #         # akr dataframe
    #         rounded_akr_df[t] = func(rounded_akr_df.unix)
    #         rounded_akr_df[t] = pd.to_numeric(rounded_akr_df[t])

    # # Append a random, uniform variable to the dataframe
    # rounded_akr_df['random_uniform'] = np.random.uniform(
    #     0., 1., len(rounded_akr_df))
    # # Append a random, gaussian variable to the dataframe
    # rounded_akr_df['random_gaussian'] = np.random.normal(
    #     100., 25., len(rounded_akr_df))

    # return akr_df, rounded_akr_df
    return akr_df


def return_test_intervals():
    """
    Function where test intervals are defined.

    Returns
    -------
    interval_options : pd.DataFrame
        Containing start, end and tag for test intervals.

    """

    interval_options = pd.DataFrame(
        {'tag': ['full_archive',
                 'cassini_flyby',
                 'long_nightside_period'],
         'title': ['Decadal Archive\n(1995-2004)',
                   'Cassini flyby\n(15th Aug 1999-14th Sep 1999)',
                   'Nightside Interval\n(22:36 11th Oct 2003 - 09:36 2nd Mar 2004)'],
         # 'label': ['1995-2004', 'Cassini fly-by', 'Nightside viewing'],
         'label': ['Decadal archive', 'Cassini flyby', 'Nightside interval'],
         'color': ['mediumpurple', 'gold', 'salmon'],
         'stime': [pd.Timestamp(1995, 1, 1, 0),
                   pd.Timestamp(1999, 8, 15, 0),
                   pd.Timestamp(2003, 10, 11, 22, 36)],
         'etime': [pd.Timestamp(2004, 12, 31, 23, 59),
                   pd.Timestamp(1999, 9, 14, 23, 59),
                   pd.Timestamp(2004, 3, 2, 9, 36)]})

    return interval_options


# def select_akr_intervals(interval, interpolated=False, rounded=False):
#     """
#     Load in AKR intensity over a defined interval.

#     Parameters
#     ----------
#     interval : string
#         Name of defined interval. Options are defined in
#         interval_options from return_test_intervals.
#     interpolated : Bool, optional
#         If True, interpolated AKR intensity is returned. The
#         default is False.
#     rounded : Bool, optional
#         If True, AKR on rounded Timestamps is returned. The
#         default is False.

#     Returns
#     -------
#     output_df : pd.DataFrame
#         DataFrame containing requested AKR data.

#     """

#     interval_options = return_test_intervals()

#     # Check parsed interval is correct
#     if interval not in np.array(interval_options.tag):
#         print('Parsed interval not a valid option.')
#         print('Exiting...')
#         return
#     else:
#         selected = interval_options.loc[interval_options.tag == interval,
#                                         :].reset_index()

#     # Which years are needed?
#     years = list(range(selected.stime[0].year, selected.etime[0].year+1, 1))

#     # Read in data
#     akr_df = read_integrated_power.concat_integrated_power_years(years,
#                                                                  'waters')

#     # Select only the interval requested
#     akr_df = akr_df.loc[(akr_df.datetime_ut >= selected.stime[0]) &
#                         (akr_df.datetime_ut <= selected.etime[0]),
#                         :].reset_index()

#     # ----- INTERPOLATED VERSION -----
#     if interpolated is True:
#         # Resample to a nice, clean, 3 minute resolution
#         # (this is important for FFTs etc)
#         print('Temporally resampling AKR onto an even resolution')
#         s_time = akr_df.datetime_ut.iloc[0].ceil(freq='min')
#         e_time = akr_df.datetime_ut.iloc[-1].floor(freq='min')
#         n_periods = np.floor((e_time-s_time)/pd.Timedelta(minutes=3))

#         new_time_axis = pd.date_range(s_time, periods=n_periods, freq='3T')
#         unix_time_axis = (new_time_axis - pd.Timestamp('1970-01-01')) / (
#             pd.Timedelta(seconds=1))
#         interpolated_akr_df = pd.DataFrame({'datetime_ut': new_time_axis,
#                                             'unix': unix_time_axis})

#         func = interpolate.interp1d(akr_df.unix, akr_df['P_Wsr-1_100_650_kHz'])
#         interpolated_akr_df['P_Wsr-1_100_650_kHz'] = func(unix_time_axis)

#         # Interpolating Wind location
#         tags = np.array(['x_gse', 'y_gse', 'z_gse', 'lat_gse', 'lon_gse',
#                          'x_gsm', 'y_gsm', 'z_gsm', 'lat_gsm', 'lon_gsm',
#                          'radius', 'decimal_gseLT'])
#         for t in tags:
#             print('Interpolating Wind position: ', t)

#             if t == 'decimal_gseLT':
#                 print('Interpolating Local Time:')
#                 print('Using special approach as LT is periodic')

#                 interpolated_akr_df[t] = utility.interpolate_mlt(
#                     np.array(interpolated_akr_df.datetime_ut), akr_df,
#                     mlt_flag='decimal_gseLT')

#             else:
#                 # Create a function defining the interpolation between points
#                 # in time defined by akr_df.unix, over the variable akr_df[t]
#                 func = interpolate.interp1d(akr_df.unix, akr_df[t])
#                 # Feed the function the time resolution of the akr data
#                 # (defined by interpolated_akr_df.unix), and get the values
#                 # of the current tag out, feed straight back into the
#                 # akr dataframe
#                 interpolated_akr_df[t] = func(interpolated_akr_df.unix)
#                 interpolated_akr_df[t] = pd.to_numeric(interpolated_akr_df[t])

#     # ----- END INTERPOLATED VERSION -----

#     # ----- ROUNDED VERSION -----
#     if rounded is True:
#         # Round to a nice, clean, 3 minute resolution
#         # (this is important for FFTs etc)
#         print('Temporally rounding AKR onto an even resolution')
#         s_time = akr_df.datetime_ut.iloc[0].ceil(freq='min')
#         e_time = akr_df.datetime_ut.iloc[-1].floor(freq='min')

#         new_time_axis = akr_df.datetime_ut.dt.round(freq='min')
#         new_time_axis = new_time_axis[:-1]
#         unix_time_axis = [t.timestamp() for t in new_time_axis]
#         rounded_akr_df = pd.DataFrame({'datetime_ut': new_time_axis,
#                                        'unix': unix_time_axis})

#         func = interpolate.interp1d(akr_df.unix, akr_df['P_Wsr-1_100_650_kHz'])
#         rounded_akr_df['P_Wsr-1_100_650_kHz'] = func(unix_time_axis)

#         # Interpolating Wind location
#         tags = np.array(['x_gse', 'y_gse', 'z_gse', 'lat_gse', 'lon_gse',
#                          'x_gsm', 'y_gsm', 'z_gsm', 'lat_gsm', 'lon_gsm',
#                          'radius', 'decimal_gseLT'])
#         for t in tags:
#             print('Interpolating Wind position: ', t)

#             if t == 'decimal_gseLT':
#                 print('Interpolating Local Time:')
#                 print('Using special approach as LT is periodic')

#                 rounded_akr_df[t] = utility.interpolate_mlt(
#                     np.array(rounded_akr_df.datetime_ut), akr_df,
#                     mlt_flag='decimal_gseLT')

#             else:
#                 # Create a function defining the interpolation between points
#                 # in time defined by akr_df.unix, over the variable akr_df[t]
#                 func = interpolate.interp1d(akr_df.unix, akr_df[t])
#                 # Feed the function the time resolution of the akr data
#                 # (defined by interpolated_akr_df.unix), and get the values
#                 # of the current tag out, feed straight back into the
#                 # akr dataframe
#                 rounded_akr_df[t] = func(rounded_akr_df.unix)
#                 rounded_akr_df[t] = pd.to_numeric(rounded_akr_df[t])

#         # Append a random, uniform variable to the dataframe
#         rounded_akr_df['random_uniform'] = np.random.uniform(
#             0., 1., len(rounded_akr_df))
#         # Append a random, gaussian variable to the dataframe
#         rounded_akr_df['random_gaussian'] = np.random.normal(
#             100., 25., len(rounded_akr_df))

#     if (interpolated is False) and (rounded is False):
#         return akr_df
#     elif (interpolated is True) and (rounded is False):
#         return akr_df, interpolated_akr_df
#     elif (interpolated is False) and (rounded is True):
#         return akr_df, rounded_akr_df
#     elif (interpolated is True) and (rounded is True):
#         return akr_df, interpolated_akr_df, rounded_akr_df


# def combine_rounded_akr_omni(interval, omni_cols=['bx',
#                                                   'by_gsm', 'bz_gsm',
#                                                   'b_total', 'clock_angle',
#                                                   'flow_speed',
#                                                   'proton_density',
#                                                   'flow_pressure', 'ae', 'al',
#                                                   'au', 'symh', 'pc_n'],
#                              supermag_cols=['SME', 'SMU', 'SML', 'SMR']):
#     """
#     Function that reads combines AKR on rounded Timestamps
#     with SuperMAG and OMNI data.

#     Parameters
#     ----------
#     interval : string
#         Name for the desired interval.
#     omni_cols : list, optional
#         Columns to extract from OMNI. The default is ['bx','by_gsm', 'bz_gsm',
#         'b_total', 'clock_angle', 'flow_speed', 'proton_density',
#         'flow_pressure', 'ae', 'al', 'au', 'symh', 'pc_n']
#     supermag_cols : list, optional
#         Columns to extract from SuperMAG indices data. The default is
#         ['SME', 'SMU', 'SML', 'SMR'].

#     Returns
#     -------
#     output_df : pd.DataFrame
#         DataFrame containing the requested AKR data.

#     """
#     print('Combining AKR intensity, SuperMAG, and OMNI into one DataFrame')
#     output_csv = os.path.join(data_dir, "rounded_akr_omni_supermag_" +
#                               str(interval) + '.csv')

#     if pathlib.Path(output_csv).is_file():
#         output_df = pd.read_csv(output_csv, delimiter=',',
#                                 float_precision='round_trip',
#                                 parse_dates=['datetime'])
#     else:
#         # Read in data
#         akr_df, rounded_akr_df = select_akr_intervals(interval, rounded=True)

#         # Create output_df
#         output_df = rounded_akr_df.copy(deep=True)
#         output_df.rename(columns={"P_Wsr-1_100_650_kHz": "integrated_power",
#                                   "datetime_ut": "datetime"},
#                          inplace=True)

#         # There are some duplicate rows in here
#         output_df.drop_duplicates(inplace=True, ignore_index=True,
#                                   subset=['datetime'])

#         # Generate and store Random Phase Surrogate
#         surrogate_intensity = utility.\
#             generate_random_phase_surrogate(output_df.integrated_power)
#         output_df['surrogate_integrated_power'] = surrogate_intensity

#         # Define the years to read in
#         syear = rounded_akr_df.datetime_ut[0].year
#         eyear = rounded_akr_df.datetime_ut[
#             len(rounded_akr_df) - 1].year
#         years = np.linspace(syear, eyear, eyear-syear + 1).astype(int)

#         # Read in OMNI data
#         omni_df = read_omni.concat_local_years_low_memory(years)

#         # Add OMNI data to output_df
#         for c in omni_cols:
#             output_df[c] = omni_df.loc[
#                 omni_df['datetime'].isin(output_df['datetime']),
#                 c].values

#         # Read SuperMAG indices
#         supermag_df = read_supermag.concat_indices_years(years)
#         # Add SuperMAG data to output_df
#         for c in supermag_cols:
#             output_df[c] = supermag_df.loc[
#                 supermag_df['Date_UTC'].isin(output_df['datetime']),
#                 c].values

#         # Write to a csv
#         output_df.to_csv(output_csv, index=False)

#     return output_df


# def combine_interp_akr_omni(interval, omni_cols=['bx', 'bz_gse',
#                                                  'by_gsm', 'bz_gsm',
#                                                  'b_total', 'clock_angle',
#                                                  'flow_speed',
#                                                  'proton_density',
#                                                  'flow_pressure', 'ae', 'al',
#                                                  'au', 'symh', 'pc_n']):

#     akr_df, interpolated_akr_df = select_akr_intervals(interval,
#                                                        interpolated=True)

#     output_csv = os.path.join(data_dir, "interp_akr_omni_supermag_" +
#                               str(interval) + '.csv')

#     if pathlib.Path(output_csv).is_file():
#         output_df = pd.read_csv(output_csv, delimiter=',',
#                                 float_precision='round_trip',
#                                 parse_dates=['datetime'])
#     else:

#         # Create output_df
#         output_df = interpolated_akr_df.copy(deep=True)
#         output_df.rename(columns={"P_Wsr-1_100_650_kHz": "integrated_power",
#                                   "datetime_ut": "datetime"},
#                          inplace=True)

#         # Define the years to read in
#         syear = interpolated_akr_df.datetime_ut[0].year
#         eyear = interpolated_akr_df.datetime_ut[
#             len(interpolated_akr_df) - 1].year
#         years = np.linspace(syear, eyear, eyear-syear + 1).astype(int)

#         # Read in OMNI data
#         omni_df = read_omni.concat_local_years_low_memory(years)

#         # Add OMNI data to output_df
#         for c in omni_cols:
#             output_df[c] = omni_df.loc[
#                 omni_df['datetime'].isin(output_df['datetime']),
#                 c].values

#         # Read SuperMAG indices
#         supermag_df = read_supermag.concat_indices_years(years)
#         # Add SuperMAG data to output_df
#         output_df['sme'] = supermag_df.loc[
#             supermag_df['Date_UTC'].isin(output_df['datetime']), 'SME'].values
#         output_df['smu'] = supermag_df.loc[
#             supermag_df['Date_UTC'].isin(output_df['datetime']), 'SMU'].values
#         output_df['sml'] = supermag_df.loc[
#             supermag_df['Date_UTC'].isin(output_df['datetime']), 'SML'].values
#         output_df['smr'] = supermag_df.loc[
#             supermag_df['Date_UTC'].isin(output_df['datetime']), 'SMR'].values

#         # Write to a csv
#         output_df.to_csv(output_csv, index=False)

#     return output_df
