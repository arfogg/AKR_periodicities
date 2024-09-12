# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:43:20 2024

@author: A R Fogg
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


def example():
    # example from sklearn
    # random forest for feature importance on a regression problem
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from matplotlib import pyplot
    # define dataset
    X, y = make_regression(n_samples=1000, n_features=10,
                           n_informative=5, random_state=1)
    # X 1000x10. 1000 datapoints, 10 variables
    # y is 1000 AKR intensity points
    # all on same time axis
    # define the model
    model = RandomForestRegressor()
    # good to test a few random states
    #    RandomForestRegressor(random_state=200)
    # N randoms states -> average results and then you
    #   get the average result (but it won't add to 1)
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

    # interpolate not-AKR
    # round to nearest 3 min


def plot_feature_importance(baseline, features, feature_names=None, seed=1993,
                       fontsize=20, record_number=True):

    # features has shape len(baseline) x number of features

    # Give indexes if features are unnamed
    if feature_names is None:
        feature_names = np.array(range(features.shape[1])).astype('str')

    # Initialise model
    model = RandomForestRegressor(random_state=seed)
    # Fitting
    model.fit(features, baseline)
    # Extract the feature importance
    importance = model.feature_importances_

    # Plot results
    fig, ax = plt.subplots(figsize=((2 + (feature_names.size * 1.5)), 10))

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
