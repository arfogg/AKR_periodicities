# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:19:27 2024

@author: A R Fogg

Create averages over different LT regions
"""

import sys
import string
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\misc_utility')
import statistical_metrics

alphabet = list(string.ascii_lowercase)
labels = []
for a in alphabet:
    labels.append('(' + a + ')')


def calc_LT_flag(data_df,
                 region_centres=[0, 6, 12, 18],
                 region_width=6,
                 region_names=['midn', 'dawn', 'noon', 'dusk'],
                 region_flags=[0, 1, 2, 3]):
    """
    Function to determine the MLT sector flag for rows in a DataFrame.

    Parameters
    ----------
    data_df : pd.DataFrame
        Must have columns 'datetime' and 'decimal_gseLT'
    region_centres : list, optional
        Centres of MLT regions in hours. The first entry must be the region
        crossing midnight. The default is [0, 6, 12, 18].
    region_width : list, optional
        Width of MLT regions in hours. The default is 6.
    region_names : list, optional
        String names of MLT regions. The default is ['midn', 'dawn',
                                                     'noon', 'dusk'].
    region_flags : list, optional
        Flags for MLT regions. The default is [0, 1, 2, 3].

    Returns
    -------
    mlt_flag : np.array
        Flag describing the MLT region of each row, component-wise,
        according to the parsed flags.
    mlt_name : np.array
        String describing the MLT region of each row, component-wise,
        according to the parsed names.

    """
    print('Selecting Local Time flags for parsed DataFrame')

    # Initialise arrays
    mlt_flag = np.full(len(data_df), np.nan)
    mlt_name = np.full(len(data_df), '')

    # Loop through different LT regions, and apply the flag to related to
    # each row in data_df
    for i, (c, n, f) in enumerate(zip(region_centres, region_names,
                                      region_flags)):

        # Find indices for this MLT iteration
        if i == 0:
            # Special procedure for midnight bin
            mlt_i, = np.where((data_df.decimal_gseLT >=
                               (24. - region_width/2.))
                              | (data_df.decimal_gseLT <
                                 (0. + region_width/2.)))
        else:
            mlt_i, = np.where((data_df.decimal_gseLT >= (c - region_width/2.))
                              & (data_df.decimal_gseLT <
                                 (c + region_width/2.)))

        # Assign the name and flag to the selected rows
        mlt_flag[mlt_i] = f
        mlt_name[mlt_i] = n

    return mlt_flag, mlt_name


def return_UT_trend(data_df, region_centres=[0, 6, 12, 18],
                    region_width=6,
                    region_names=['midn', 'dawn', 'noon', 'dusk'],
                    region_flags=[0, 1, 2, 3], UT_bin_width=2,
                    ipower_tag='integrated_power'):
    """
    Function to sort data by MLT and UT

    Parameters
    ----------
    data_df : pd.DataFrame
        Pandas DataFrame containing 'datetime', ipower_tag, 'mlt_flag'.
    region_centres : list, optional
        Centres for the MLT bins. The default is [0, 6, 12, 18].
    region_width : int, optional
        Width of the MLT bins. The default is 6.
    region_names : list, optional
        Names for the MLT bins. The default is ['midn', 'dawn', 'noon',
                                                'dusk'].
    region_flags : list, optional
        Int flags for the MLT bins. The default is [0, 1, 2, 3].
    UT_bin_width : int, optional
        Width of the UT bins in hours. The default is 2.
    ipower_tag : string, optional
        String which selects integrated power from data_df. The default
        is 'integrated_power'.

    Returns
    -------
    UT_df : pd.DataFrame
        Returns sorted and averaged results.

    """

    UT_bins = np.linspace(0, 24-UT_bin_width, int(24/UT_bin_width)) +\
        (UT_bin_width/2)

    # Calculate the decimal hour of the observation in UT
    data_df['decimal_hr'] = ((data_df['datetime'] -
                              data_df['datetime'].dt.normalize())
                             / pd.Timedelta(hours=1))

    UT_df = pd.DataFrame({'UT_bin_centre': UT_bins})

    # Iterate through MLT sectors
    for i, (c, n, f) in enumerate(zip(region_centres, region_names,
                                      region_flags)):

        LT_data_df = data_df.loc[data_df.mlt_flag == f].reset_index()

        UT_median = np.full(UT_bins.size, np.nan)
        UT_mad = np.full(UT_bins.size, np.nan)

        UT_median_no0 = np.full(UT_bins.size, np.nan)
        UT_mad_no0 = np.full(UT_bins.size, np.nan)

        UT_n = np.full(UT_bins.size, np.nan)
        UT_n_no0 = np.full(UT_bins.size, np.nan)

        for j in range(UT_bins.size):
            UT_ind, = np.where((LT_data_df.decimal_hr >=
                                (UT_bins[j]-UT_bin_width/2))
                               & (LT_data_df.decimal_hr <
                                  (UT_bins[j]+UT_bin_width/2)))
            dist_ = np.array(
                LT_data_df[ipower_tag].iloc[UT_ind].values)
            UT_mad[j], UT_median[j] = statistical_metrics.\
                median_absolute_deviation(dist_)
            UT_mad_no0[j], UT_median_no0[j] = statistical_metrics.\
                median_absolute_deviation(dist_[dist_ > 0.])

            UT_n[j] = dist_.size
            UT_n_no0[j] = dist_[dist_ > 0.].size

        UT_df[n + '_median'] = UT_median
        UT_df[n + '_median_norm'] = UT_median / np.nanmax(UT_median)
        UT_df[n + '_mad'] = UT_mad
        UT_df[n + '_mad_norm'] = UT_mad / np.nanmax(UT_median)

        UT_df[n + '_median_no0'] = UT_median_no0
        UT_df[n + '_median_norm_no0'] = UT_median_no0 / np.nanmax(
            UT_median_no0)
        UT_df[n + '_mad_no0'] = UT_mad_no0
        UT_df[n + '_mad_norm_no0'] = UT_mad_no0 / np.nanmax(UT_median_no0)

        UT_df[n + 'n'] = UT_n
        UT_df[n + 'n_no0'] = UT_n_no0

    return UT_df


def plot_UT_trend(data_df, region_centres=[0, 6, 12, 18],
                  region_width=6,
                  region_names=['midn', 'dawn', 'noon', 'dusk'],
                  region_flags=[0, 1, 2, 3], UT_bin_width=2,
                  region_titles=['Midnight (21-3 LT)', 'Dawn (3-9 LT)',
                                 'Noon (9-15 LT)', 'Dusk (15-21 LT)'],
                  fontsize=15):
    """
    Plot the UT/MLT trend

    Parameters
    ----------
    data_df : pd.DataFrame
        Pandas DataFrame containing 'datetime', 'integrated_power'.
    region_centres : list, optional
        Centres for the MLT bins. The default is [0, 6, 12, 18].
    region_width : int, optional
        Width of the MLT bins. The default is 6.
    region_names : list, optional
        Names for the MLT bins. The default is ['midn', 'dawn', 'noon',
                                                'dusk'].
    region_flags : list, optional
        Int flags for the MLT bins. The default is [0, 1, 2, 3].
    UT_bin_width : int, optional
        Width of the UT bins in hours. The default is 2.
    region_titles : list, optional
        String titles for MLT panels. The default is ['Midnight (21-3 LT)',
                                                      'Dawn (3-9 LT)',
                                                      'Noon (9-15 LT)',
                                                      'Dusk (15-21 LT)'].
    fontsize : float, optional
        Fontsize applied to labels/text. The default is 15.

    Returns
    -------
    fig_med : matplotlib figure
        Output figure.
    fig_bp : matplotlib figure
        Output figure.

    """

    # Remove all rows where intensity is zero.
    zero_ind, = np.where(data_df.integrated_power == 0.)
    data_df.drop(index=zero_ind, inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    # Record the LT flag
    f, n = calc_LT_flag(data_df, region_centres=region_centres,
                        region_width=region_width, region_names=region_names,
                        region_flags=region_flags)
    data_df['mlt_flag'] = f
    data_df['mlt_name'] = n

    UT_bins = np.linspace(0, 24-UT_bin_width, int(24/UT_bin_width)) +\
        (UT_bin_width/2)
    data_df['decimal_hr'] = (data_df.datetime -
                             data_df.datetime.values.astype('datetime64[D]')).\
        astype('timedelta64[m]') / 60.

    # Initialise plotting window
    fig_bp, axes_bp = plt.subplots(nrows=len(region_names),
                                   figsize=(10, 3*len(region_names)))

    fig_med, axes_med = plt.subplots(nrows=len(region_names),
                                     figsize=(9, 3*len(region_names)))

    # Iterate through MLT sectors
    for i, (c, n, f, t, ax_bp, ax_md) in enumerate(zip(region_centres,
                                                       region_names,
                                                       region_flags,
                                                       region_titles,
                                                       axes_bp, axes_med)):

        LT_data_df = data_df.loc[data_df.mlt_flag == f].reset_index()

        UT_median = np.full(UT_bins.size, np.nan)
        UT_mad = np.full(UT_bins.size, np.nan)
        UT_dist = []

        UT_n = np.full(UT_bins.size, np.nan)

        for j in range(UT_bins.size):

            UT_ind, = np.where((LT_data_df.decimal_hr >=
                                (UT_bins[j]-UT_bin_width/2))
                               & (LT_data_df.decimal_hr <
                                  (UT_bins[j]+UT_bin_width/2)))
            dist_ = np.array(
                LT_data_df['integrated_power'].iloc[UT_ind].values)
            UT_mad[j], UT_median[j] = statistical_metrics.\
                median_absolute_deviation(dist_)

            UT_n[j] = dist_.size

            UT_dist.append(dist_)

        UT_cmap = matplotlib.colormaps['cool_r']
        UT_norm = matplotlib.colors.Normalize(vmin=np.nanmin(UT_n),
                                              vmax=np.nanmax(UT_n))
        UT_color = UT_cmap(UT_norm(UT_n))
        # PLOT MEDIAN
        ax_md.errorbar(UT_bins, UT_median, UT_mad,
                       marker='x', fillstyle='none', markersize=0.75*fontsize,
                       capsize=0.5*fontsize,
                       color='black', linestyle='dashed', linewidth=1.0)

        # DECOR
        ax_md.set_title(t, fontsize=fontsize)
        ax_md.tick_params(labelsize=fontsize)
        ax_md.set_ylabel('Median Integrated\nPower (W sr$^{-1}$)',
                         fontsize=fontsize)
        ax_md.set_xlabel('UT (hours)', fontsize=fontsize)

        txt = ax_md.text(0.025, 0.93, labels[i],
                         transform=ax_md.transAxes, fontsize=fontsize,
                         va='top', ha='left')
        txt.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

        # BOXPLOT
        box = ax_bp.boxplot(UT_dist, positions=UT_bins,
                            whis=(15, 85), showfliers=False,
                            patch_artist=True,
                            medianprops={'color': 'black', 'linewidth': 2.})

        for patch, color in zip(box['boxes'], UT_color):
            patch.set_facecolor(color)

        # COLORBAR
        cbar = fig_bp.colorbar(
                matplotlib.cm.ScalarMappable(norm=UT_norm, cmap=UT_cmap),
                ax=ax_bp, pad=0.025)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label('N', fontsize=fontsize)

        # DECOR
        ax_bp.set_title(t, fontsize=fontsize)
        ax_bp.tick_params(labelsize=fontsize)
        ax_bp.set_ylabel('Median Integrated\nPower (W sr$^{-1}$)',
                         fontsize=fontsize)
        ax_bp.set_xlabel('UT (hours)', fontsize=fontsize)

        txt = ax_bp.text(0.025, 0.93, labels[i],
                         transform=ax_bp.transAxes, fontsize=fontsize,
                         va='top', ha='left')
        txt.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))

    fig_med.tight_layout()
    fig_bp.tight_layout()

    return fig_med, fig_bp


def return_lon_trend(data_df, region_centres=[0, 6, 12, 18],
                     region_width=6,
                     region_names=['midn', 'dawn', 'noon', 'dusk'],
                     region_flags=[0, 1, 2, 3],
                     lon_bin_width=30.,
                     ipower_tag='integrated_power',
                     lon_sol_tag="lon_sol", lon_sc_tag="lon_gse"):
    """
    Bin and average data by longitude.

    Parameters
    ----------
    data_df : pd.DataFrame
        Pandas dataframe containing 'datetime', 'mlt_flag', lon_sol_tag,
        lon_sc_tag, ipower_tag.
    region_centres : list, optional
        Centres for the MLT bins. The default is [0, 6, 12, 18].
    region_width : int, optional
        Width of the MLT bins. The default is 6.
    region_names : list, optional
        Names for the MLT bins. The default is ['midn', 'dawn', 'noon',
                                                'dusk'].
    region_flags : list, optional
        Int flags for the MLT bins. The default is [0, 1, 2, 3].
    lon_bin_width : float, optional
        Width of longitude bins. The default is 30..
    ipower_tag : string, optional
        Column header for data_df containing AKR integrated power. The
        default is 'integrated_power'.
    lon_sol_tag : string, optional
        Column header for data_df containing longitude of the Sun. The
        default is "lon_sol".
    lon_sc_tag : string, optional
        Column header for data_df containing spacecraft longitude. The
        default is "lon_gse".

    Returns
    -------
    lon_df : pd.DataFrame
        Dataframe containing binned and averaged data.

    """
    data_df[lon_sc_tag] = data_df[lon_sc_tag] % 360.
    data_df[lon_sol_tag] = data_df[lon_sol_tag] % 360.

    lon_bins = np.linspace(0, 360.-lon_bin_width, int(360./lon_bin_width)) +\
        (lon_bin_width/2)

    data_df['decimal_hr'] = ((data_df['datetime'] -
                              data_df['datetime'].dt.normalize())
                             / pd.Timedelta(hours=1))

    lon_df = pd.DataFrame({'lon_bin_centre': lon_bins})

    # Iterate through MLT sectors
    for i, (c, n, f) in enumerate(zip(region_centres, region_names,
                                      region_flags)):

        LT_data_df = data_df.loc[data_df.mlt_flag == f].reset_index()

        # Initialise variables for longitude bins
        median_lon = np.full((lon_bins.size, 2), np.nan)
        mad_lon = np.full((lon_bins.size, 2), np.nan)

        median_no0_lon = np.full((lon_bins.size, 2), np.nan)
        mad_no0_lon = np.full((lon_bins.size, 2), np.nan)

        n_lon = np.full((lon_bins.size, 2), np.nan)
        n_no0_lon = np.full((lon_bins.size, 2), np.nan)

        for j in range(lon_bins.size):

            for [k, ln_tg] in enumerate([lon_sol_tag, lon_sc_tag]):
                # Find the right indices
                lon_ind, = np.where((LT_data_df[ln_tg] >=
                                    (lon_bins[j] - lon_bin_width/2))
                                    & (LT_data_df[ln_tg] <
                                       (lon_bins[j] + lon_bin_width/2)))
                dist_ = np.array(
                    LT_data_df[ipower_tag].iloc[lon_ind].values)
                mad_lon[j, k], median_lon[j, k] = statistical_metrics.\
                    median_absolute_deviation(dist_)
                mad_no0_lon[j, k], median_no0_lon[j, k] = statistical_metrics.\
                    median_absolute_deviation(dist_[dist_ > 0.])

                n_lon[j, k] = dist_.size
                n_no0_lon[j, k] = dist_[dist_ > 0.].size

        for [l, tg] in enumerate(['sol', 'sc']):
            lon_df[n + '_median_' + tg] = median_lon[:, l]
            lon_df[n + '_median_norm_' + tg] = median_lon[:, l] / np.nanmax(
                median_lon[:, l])
            lon_df[n + '_mad_' + tg] = mad_lon[:, l]
            lon_df[n + '_mad_norm_' + tg] = mad_lon[:, l] / np.nanmax(
                mad_lon[:, l])

            lon_df[n + '_median_no0_' + tg] = median_no0_lon[:, l]
            lon_df[n + '_median_norm_no0_' + tg] = median_no0_lon[:, l] / np.nanmax(median_no0_lon[:, l])
            lon_df[n + '_mad_no0_' + tg] = mad_no0_lon[:, l]
            lon_df[n + '_mad_norm_no0_' + tg] = mad_no0_lon[:, l] / np.nanmax(
                mad_no0_lon[:, l])

            lon_df[n + '_n_' + tg] = n_lon[:, l]
            lon_df[n + '_n_no0_' + tg] = n_no0_lon[:, l]

    return lon_df
