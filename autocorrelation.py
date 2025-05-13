# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:19:36 2025

@author: A R Fogg
"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.optimize import least_squares
from scipy.optimize import curve_fit


def autocorrelation(y, n_shifts, temporal_resolution=180, starting_lag=7200):
    """
    Calculate the autocorrelation (ACF) for a signal y
    across various lags.

    Parameters
    ----------
    y : np.array
        Signal to be analysed.
    n_shifts : int
        Number of lags to calculate the ACF for.
    temporal_resolution : int, optional
        Temporal resolution of y in seconds. The
        default is 180.
    starting_lag : int, optional
        Starting lag, in seconds. The default is 7200.

    Returns
    -------
    acf : np.array
        ACF as a function of lags.
    lags : np.array
        Lag in seconds.

    """
    print('Calculating the ACF for ' + str(n_shifts) + ' lags')
    starting_lag_i = int(starting_lag / temporal_resolution)

    # Initialise arrays
    acf = np.full(n_shifts, np.nan)
    lags = np.full(n_shifts, np.nan)

    for i in range(0, n_shifts):
        # Shift y
        shifted_y = np.append(y[-(i+starting_lag_i):],
                              y[:-(i+starting_lag_i)])
        # Calculate ACF
        acf[i] = np.correlate(y, shifted_y)
        # Calculate lag in seconds
        lags[i] = temporal_resolution * (i + starting_lag_i)

    return acf, lags


def plot_autocorrelogram(lags, acf, fontsize=15, tick_sep_hrs=12.,
                         highlight_period=24.,
                         highlight_fmt={'color': 'grey',
                                        'linestyle': 'dashed',
                                        'linewidth': 1.},
                         acf_fmt={'color': 'forestgreen',
                                  'linewidth': 1.}):
    """
    Plot the ACF as a function of lags.

    Parameters
    ----------
    lags : np.array
        Lag in seconds.
    acf : np.array
        ACF as a function of lags.
    fontsize : int, optional
        Fontsize parsed to matplotlib. The default is 15.
    tick_sep_hrs : float, optional
        Seperation of x ticks in hours. The default is 12.
    highlight_period : float, optional
        Vertical line is drawn at each integer number of this
        repeating interval (in hours) on the x axis. The
        default is 24.
    highlight_fmt : dict, optional
        Format of the vertical highlight lines to be parsed
        to matplotlib.
    acf_fmt : dict, optional
        Format of the ACF line to be parsed to matplotlib.

    Returns
    -------
    fig : matplotlib figure object
        Figure containing plot.
    ax : matplotlib axes object
        Axes containing plot.

    """

    # Define plotting window
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot ACF as a function of lag
    ax.plot(lags, acf, **acf_fmt)

    # Convert x ticks from seconds to readable format
    n_ticks = int(np.floor(np.max(ax.get_xticks()) /
                           (tick_sep_hrs * 60. * 60.)))
    tick_pos = []
    tick_str = []
    for i in range(n_ticks):
        tick_pos.append(i * (tick_sep_hrs * 60. * 60.))
        tick_str.append(str(int(tick_sep_hrs * i)))
    ax.set_xticks(tick_pos, tick_str)

    # Label axes
    ax.set_ylabel('ACF', fontsize=fontsize)
    ax.set_xlabel('Lag (hours)', fontsize=fontsize)

    # Draw vertical lines each highlight period
    n_vert = int(np.floor((ax.get_xlim()[1]) / (highlight_period * 60. * 60.)))
    for i in range(n_vert):
        ax.axvline(((i + 1) * highlight_period)*(60. * 60.), **highlight_fmt)

    # Formatting
    ax.tick_params(labelsize=fontsize)

    return fig, ax



# def fit_exp_envelope(lags, acf):
    
#     from scipy.signal import hilbert
    
#     acf_envelope = np.abs(hilbert(acf))
    
#     return acf_envelope

def remove_linear_trend(lags, acf):
    
    linear_fit = linregress(lags, acf)
    
    line_y = linear_fit.intercept + linear_fit.slope * lags
    
    detrended_acf = acf - line_y
    
    return linear_fit, line_y, detrended_acf

def normalise_with_mean_subtraction(acf):
    
    norm_acf = (acf - np.nanmean(acf)) / np.std(acf)
    
    return norm_acf

def fit_decaying_sinusoid(lags, acf, A0, gamma0, omega0, phi0):
    print('Fitting a decaying sinusoid to the parsed parameters')

    initial_guess = (A0, gamma0, omega0, phi0)
    #initial_guess = (5E18, 0.05, 0.0006, 0.0)

    # Curve fitting
    popt, pcov = curve_fit(damped_oscillator, lags, acf, p0=initial_guess)

    # Extract fitted parameters
    A_fit, gamma_fit, omega_fit, phi_fit = popt

    y_fit = damped_oscillator(lags, *popt)
    
    return A_fit, gamma_fit, omega_fit, phi_fit, y_fit, popt, pcov

    # # input_parameters = [A0, tau, omega, b]
    # popt, pcov = curve_fit(func_decaying_sinusoid, lags, acf,
    #                         p0=input_parameters)
    #popt, pcov = curve_fit(func_linear_decaying_sinusoid, lags, acf)
    
    # return popt, pcov, func_linear_decaying_sinusoid(*popt, lags)
    
    # res_lsq = least_squares(func_decaying_sinusoid_residual,
    #               input_parameters, args=(lags, acf))

    # return res_lsq, func_decaying_sinusoid(res_lsq.x, lags)

def damped_oscillator(x, A, gamma, omega, phi):
    return A * np.exp(-gamma * x) * np.cos(omega * x + phi)


def damped_osc_ci(A, gamma, omega, phi, pcov,
                  lags, n_bootstrap=100, ci=[2.5, 97.5]):
    
    print('hello')
    
    errors = np.sqrt(np.diag(pcov))
    

    bs_A = np.random.normal(A, errors[0], n_bootstrap)
    bs_gamma = np.random.normal(gamma, errors[1], n_bootstrap)
    bs_omega = np.random.normal(omega, errors[2], n_bootstrap)
    bs_phi = np.random.normal(phi, errors[3], n_bootstrap)
    
    y_bs = np.full((len(lags), n_bootstrap), np.nan)
    y_ci = np.full((len(lags), 2), np.nan)

    for i in range(n_bootstrap):
        y_bs[:, i] = damped_oscillator(lags, bs_A[i], bs_gamma[i],
                                       bs_omega[i], bs_phi[i])
    for i in range(len(lags)):
        y_ci[i, :] = np.percentile(y_bs[i, :], ci)

    return bs_A, bs_gamma, bs_omega, bs_phi, y_bs, y_ci   

class decay_shm_fit():
    
    
    def __init__(self, lags, acf):

        # Store the ACF data
        self.lags = lags
        self.acf = acf
        
    def fit_SHM(self, A0=1.0, gamma0=1e-6,
                omega0=2 * np.pi / 100000, phi0=0):
        
        # Detrend the data, removing a linear trend
        self.linear_detrend_fit, self.linear_detrend_y,\
            self.linear_detrended_acf =\
            remove_linear_trend(self.lags, self.acf)

        # self.linear_detrend_fit = linear_fit
        # self.linear_detrend_y = line_y
        # self.linear_detrended_acf = detrended_acf
        
        # Normalise the Data
        self.normalised_acf = normalise_with_mean_subtraction(self.linear_detrended_acf)

        self.A, self.gamma, self.omega, self.phi, self.y_fitted, self.popt,\
            self.pcov = fit_decaying_sinusoid(self.lags,
                                              self.normalised_acf,
                                              A0, gamma0, omega0, phi0)
            
        # Confidence interval here?? Not sure on this code atm!
        sigma_ab = np.sqrt(np.diagonal(self.pcov))
        self.ci_upper = damped_oscillator(self.lags, *(self.popt + sigma_ab))
        self.ci_lower = damped_oscillator(self.lags, *(self.popt - sigma_ab))
    def create_text_labels(self):
        
        self.text_linear_trend = "y = " + "{:.2e}".format(self.linear_detrend_fit.slope) +\
            " x\n+ " + "{:.2e}".format(self.linear_detrend_fit.intercept)

        self.text_shm_trend = "{:.2f}".format(self.A) + "exp(-" + "{:.2e}".format(self.gamma) + "x)" +\
            "cos(" + "{:.2e}".format(self.omega) + "x + " + "{:.2e}".format(self.phi) + ")"
        #A * np.exp(-gamma * x) * np.cos(omega * x + phi)
        
    def calc_confidence_interval(self):
        
        self.bs_A, self.bs_gamma, self.bs_omega, self.bs_phi,\
            self.y_bs, self.y_ci = \
            damped_osc_ci(self.A, self.gamma, self.omega, self.phi,
                          self.pcov, self.lags, n_bootstrap=10, ci=95)