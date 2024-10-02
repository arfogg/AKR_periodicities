# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:19:27 2024

@author: A R Fogg

Create averages over different LT regions
"""

import numpy as np
import matplotlib.pyplot as plt


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


def plot_UT_trend(data_df, region_centres=[0, 6, 12, 18],
                  region_width=6,
                  region_names=['midn', 'dawn', 'noon', 'dusk'],
                  region_flags=[0, 1, 2, 3], UT_bin_width=2):

    # data_df needs columns
    #   datetime, decimal_gseLT, integrated_power
    # UT_bin_width is bin width in hours

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

    # violin_quantiles = [[0.25, 0.75] for w in range(UT_bins.size)]

    # Initialise plotting window
    fig, axes = plt.subplots(nrows=len(region_names),
                             figsize=(9, 2*len(region_names)))

    # Iterate through MLT sectors
    for i, (c, n, f, ax) in enumerate(zip(region_centres, region_names,
                                      region_flags, axes)):
        # print(i, c, n, f)

        LT_data_df = data_df.loc[data_df.mlt_flag == f].reset_index()

        UT_median = np.full(UT_bins.size, np.nan)
        UT_dist = []
        for j in range(UT_bins.size):

            UT_ind, = np.where((LT_data_df.decimal_hr >=
                                (UT_bins[j]-UT_bin_width/2))
                               & (LT_data_df.decimal_hr <
                                  (UT_bins[j]+UT_bin_width/2)))
            UT_median[j] = np.nanmedian(LT_data_df['integrated_power'].
                                        iloc[UT_ind].values)

            UT_dist.append(LT_data_df['integrated_power'].iloc[UT_ind].values)

            # if i == 1:
            #     breakpoint()
            # print(UT_bins[j], UT_median[j])
            # THERE'S A TONNE OF ZEROS IN INTEGRATED INTENSITY, WHAT DO WE
            # WANT TO DO WITH THEM?
            # BUT ZERO INTENSITY =/= NO AKR? OR IS IT ==?
            # throw out zeros/low data

        # PLOT MEDIAN
        ax.plot(UT_bins, UT_median, marker='o', fillstyle='none',
        linewidth=0.)

        # VIOLIN PLOT
        # ax.violinplot(UT_dist, positions=UT_bins,
        #               #quantiles=violin_quantiles,
        #               showmeans=False, showmedians=True)

        # BOXPLOT
        #ax.boxplot(UT_dist, positions=UT_bins,
        #           whis=(0.1, 0.9), showfliers=False)
        ax.set_title(n)
