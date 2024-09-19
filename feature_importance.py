# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:43:20 2024

@author: A R Fogg

Code to assess feature importance of different parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


def plot_feature_importance(baseline, features,
                            importance=None,
                            feature_names=None, seed=1993,
                            fontsize=20., record_number=True):
    """
    Function to calculate and plot the feature importance of various
    features compared with a baseline.

    Parameters
    ----------
    baseline : np.array
        Baseline parameter to compare the features against.
    features : np.array
        Shape len(baseline) x number of features. Matrix of features
        on the same time axis as baseline.
    importance : np.array, optional
        Precalculated importance, of length = number of features. The
        default is None.
    feature_names : list, optional
        String names describing the features. If not parsed, then the
        features are given integers as names. The default is None.
    seed : int, optional
        Random state parsed to RandomForestRegressor. The default
        is 1993.
    fontsize : float, optional
        Fontsize parsed to matplotlib. The default is 20.0.
    record_number : Bool, optional
        If True, then the value of the feature importance is recorded
        above the bar on the plot. The default is True.

    Returns
    -------
    fig : matplotlib figure
        Figure containing the output plot.
    ax : matplotlib axis
        Axis containing the output plot.
    importance : np.array
        Importance of each feature in order features are parsed in.

    """

    # Give indexes if features are unnamed
    if feature_names is None:
        feature_names = np.array(range(features.shape[1])).astype('str')
    else:
        feature_names = np.array(feature_names)

    if importance is None:
        # Initialise model
        model = RandomForestRegressor(random_state=seed)
        # Fitting
        t1 = pd.Timestamp.now()
        print('Starting model fitting for '+str(len(feature_names))
              + ' features at time ', t1)
        model.fit(features, baseline)
        print('Model fit complete, time taken: ', pd.Timestamp.now()-t1)
        # Extract the feature importance
        importance = model.feature_importances_

    # Plot results
    fig, ax = plt.subplots(figsize=((2 + (len(feature_names) * 1.5)), 10))

    # Rank the features by importance
    sort_i = np.argsort(importance)
    sorted_feature_names = feature_names[sort_i][::-1]
    sorted_importance = importance[sort_i][::-1]

    # Plot bar chart
    ax.bar(sorted_feature_names, sorted_importance,
           color='paleturquoise', edgecolor='darkcyan', linewidth=2.0)

    # Record value on top of each bar
    if record_number is True:
        for i in range(sorted_importance.size):
            ax.text(sorted_feature_names[i], sorted_importance[i],
                    str("%.3f" % sorted_importance[i]),
                    fontsize=fontsize, ha='center', va='bottom')

    # Formatting
    ax.set_xlabel('Features\n(in order of importance)', fontsize=fontsize)
    ax.set_ylabel('Feature Importance', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    return fig, ax, importance


def feature_importance_3panel(fin1, fin2, fin3,
                              im1, im2, im3,
                              record_number=True,
                              fontsize=15.,
                              titles=['(a)', '(b)', '(c)']):
    """
    Function to plot the feature importance comparing three
    different panels. Does not calculate importance, only plots!

    Parameters
    ----------
    fin1 : np.array
        Feature names for set 1.
    fin2 : np.array
        Feature names for set 2.
    fin3 : np.array
        Feature names for set 3.
    im1 : np.array
        Importance for features in set 1.
    im2 : np.array
        Importance for features in set 2.
    im3 : np.array
        Importance for features in set 3.
    record_number : Bool, optional
        If True, the importance is recorded above the bar on the
        plot. The default is True.
    fontsize : float, optional
        Fontsize parsed to matplotlib. The default is 15.0.
    titles : list, optional
        String titles to apply to each panel. The default is
        ['(a)', '(b)', '(c)'].

    Returns
    -------
    fig : matplotlib figure
        Figure containing the output plot.
    ax : matplotlib axis
        Axis containing the output plot.

    """

    # titles = ['(a) Visibility',
    #           '(b) Geophysical',
    #           '(c) Visibility and Geophysical']
    fill_colors = ['orange', 'paleturquoise', 'palevioletred']
    edge_colors = ['darkgoldenrod', 'darkcyan', 'crimson']

    fig = plt.figure(figsize=(20, 10))

    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=3)
    ax3 = plt.subplot(2, 1, 2)

    for i, (fin, im, ax) in enumerate(zip([fin1, fin2, fin3],
                                          [im1, im2, im3], [ax1, ax2, ax3])):
        # print(i, fi, ax)

        sort_i = np.argsort(im)
        sorted_feature_names = fin[sort_i][::-1]
        sorted_importance = im[sort_i][::-1]

        ax.bar(sorted_feature_names, sorted_importance,
               color=fill_colors[i], edgecolor=edge_colors[i],
               linewidth=2.0, alpha=0.3)

        if record_number is True:
            for j in range(sorted_importance.size):
                ax.text(sorted_feature_names[j], sorted_importance[j],
                        str("%.3f" % sorted_importance[j]),
                        fontsize=fontsize, ha='center', va='bottom')

        # Formatting
        ax.set_xlabel('Features (in order of importance)', fontsize=fontsize)
        ax.set_ylabel('Feature Importance', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_title(titles[i], fontsize=1.25*fontsize)

    fig.tight_layout()

    return fig, ax
