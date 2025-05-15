# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:15:47 2024

@author: A R Fogg

Functions to generate bootstrap samples.
"""

import numpy as np


def produce_single_bootstrap_sample(data, extract_length):
    """
    Function to product a bootstrapping sample from one given
    data set.

    Based on code by Dr Dáire Healy here -
    https://github.com/arfogg/bi_multi_variate_eva/blob/c8f7911b5aa911a074a33b862224563460357ce3/r_code/bootstrapped_chi.R

    Parameters
    ----------
    data : np.array
        The input data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    extract_length : int
        On average, the data samples appended to the bootstrap
        will be extract_length number of data points long.

    Returns
    -------
    bootstrap_sample : np.array
        A sample of length data.size of bootstrapped data.

    """

    # Initialise empty list
    bootstrap_sample = []

    # While our output is shorter than the input data
    #   we will continue adding data
    while len(bootstrap_sample) < len(data):

        # Choose a random start and end point from the
        #   input data to resample
        start_point = int(np.random.choice(data.size, size=1)[0])
        end_point = int(start_point + np.random.geometric(1.0 / extract_length,
                                                          size=1)[0])

        # If not beyond the end of the data, append
        #   some data to the new array
        if end_point < len(data):
            bootstrap_sample = np.append(bootstrap_sample,
                                         data[start_point:end_point])

    # Ensure output sample isn't longer that the original sample
    bootstrap_sample = np.array(bootstrap_sample[0:len(data)])

    return bootstrap_sample


def generate_bootstrap(data):
    """
    Function to product a bootstrapping sample from one given
    data set.

    Parameters
    ----------
    data : np.array
        The input data to be bootstrapped. Must be np.array,
        pd.Series won't work.

    Returns
    -------
    bootstrap_sample : np.array
        A sample of length data.size of bootstrapped data.

    """
    # Random selection from the data
    bootstrap_sample = np.random.choice(data, size=len(data))

    return bootstrap_sample
