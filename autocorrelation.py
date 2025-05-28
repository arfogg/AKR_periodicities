# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:19:36 2025

@author: A R Fogg
"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import linregress, chisquare
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


def remove_linear_trend(lags, acf):
    """
    Detrend an ACF timeseries by fitting and removing a linear trend.

    Parameters
    ----------
    lags : np.array
        Lags (i.e. x axis of ACF plot).
    acf : np.array
        ACF (i.e. y axis of ACF plot).

    Returns
    -------
    linear_fit : object
        Class from scipy linregress, contains intercept and slope.
    line_y : np.array
        Linear fit y values as a function of input lags.
    detrended_acf : np.array
        Detrended ACF as a function of input lags.

    """

    # Generate linear fit to ACF
    linear_fit = linregress(lags, acf)

    # Return y values on line of best fit
    line_y = linear_fit.intercept + linear_fit.slope * lags

    # Subtract line of best fit from ACF
    detrended_acf = acf - line_y

    return linear_fit, line_y, detrended_acf


def normalise_with_mean_subtraction(acf):
    """
    Normalise ACF values by subtracting the mean.

    Parameters
    ----------
    acf : np.array
        ACF values.

    Returns
    -------
    norm_acf : np.array
        ACF values with mean subtracted.

    """

    # Mean ACF
    mean_acf = np.nanmean(acf)
    # Standard deviation
    std_acf = np.std(acf)
    # Subtract mean from ACF
    norm_acf = (acf - mean_acf) / std_acf

    return norm_acf, mean_acf, std_acf


def fit_decaying_sinusoid(lags, acf, A0, gamma0, omega0, phi0):
    """
    Fit a damped simple harmonic oscillator curve to ACF as a function of lag.

    Parameters
    ----------
    lags : np.array
        Lags (i.e. x axis of ACF plot).
    acf : np.array
        ACF (i.e. y axis of ACF plot). Better fit if detrended and normalised
        before inputting to this function.
    A0 : float
        Initial guess for A.
    gamma0 : float
        Initial guess for gamma.
    omega0 : float
        Initial guess for omega.
    phi0 : float
        Initial guess for phi.

    Returns
    -------
    A_fit : float
        A for fitted curve.
    gamma_fit : float
        Gamma for fitted curve.
    omega_fit : float
        Omega for fitted curve.
    phi_fit : float
        Phi for fitted curve.
    y_fit : np.array
        Y values for fitted curve as a function of input lags.
    popt : array
        Parameters from curve fit.
    pcov : array
        Covariance of popt.

    """
    print('Fitting a decaying sinusoid to the parsed parameters')

    initial_guess = (A0, gamma0, omega0, phi0)

    # Curve fitting
    popt, pcov = curve_fit(damped_oscillator, lags, acf, p0=initial_guess)

    # Extract fitted parameters
    A_fit, gamma_fit, omega_fit, phi_fit = popt

    # Calculate y values of fitted curve as a function of lags
    y_fit = damped_oscillator(lags, *popt)

    return A_fit, gamma_fit, omega_fit, phi_fit, y_fit, popt, pcov


def damped_oscillator(x, A, gamma, omega, phi):
    """
    Function defining a damped simple harmonic oscillator.

    Parameters
    ----------
    x : np.array
        X axis, i.e. lags in the ACF case.
    A : float
        Free parameter.
    gamma : float
        Free parameter.
    omega : float
        Free parameter.
    phi : float
        Free parameter.

    Returns
    -------
    np.array
        Y axis of curve as a function of x, given the input
        free parameters.

    """
    return A * np.exp(-gamma * x) * np.cos(omega * x + phi)


def damped_osc_ci(A, gamma, omega, phi, pcov,
                  lags, n_bootstrap=100, ci=[2.5, 97.5]):
    """
    Calculate confidence interval on SHM fit.

    Parameters
    ----------
    A : float
        Fitted free parameter A.
    gamma : float
        Fitted free parameter gamma.
    omega : float
        Fitted free parameter omega.
    phi : float
        Fitted free parameter phi.
    pcov : array
        Covariance of free parameters.
    lags : np.array
        X axis for model.
    n_bootstrap : int, optional
        Number of bootstraps to create. The default is 100.
    ci : list, optional
        Position for np.percentile to calculate confidence
        interval. The default is [2.5, 97.5].

    Returns
    -------
    bs_A : np.array
        Bootstraps of free parameter A.
    bs_gamma : np.array
        Bootstraps of free parameter gamma.
    bs_omega : np.array
        Bootstraps of free parameter omega.
    bs_phi : np.array
        Bootstraps of free parameter phi.
    y_bs : np.array
        Array of shape len(lags) x n_bootstrap containing y as
        a function of lags for each bootstrap.
    y_ci : np.array
        Array of shape len(lags) x 2 containing upper and lower
        confidence interval.

    """
    # Calculate the 1SD errors on parameters
    errors = np.sqrt(np.diag(pcov))

    # Define a random distribution of each of the parameters
    # within the 1SD errors
    bs_A = np.random.normal(A, errors[0], n_bootstrap)
    bs_gamma = np.random.normal(gamma, errors[1], n_bootstrap)
    bs_omega = np.random.normal(omega, errors[2], n_bootstrap)
    bs_phi = np.random.normal(phi, errors[3], n_bootstrap)

    # Initialise empty arrays to contain bootstrapped y and
    # confidence interval
    y_bs = np.full((len(lags), n_bootstrap), np.nan)
    y_ci = np.full((len(lags), 2), np.nan)

    # Estimate y values for n_bootstrap cases
    for i in range(n_bootstrap):
        y_bs[:, i] = damped_oscillator(lags, bs_A[i], bs_gamma[i],
                                       bs_omega[i], bs_phi[i])

    # Calculate the confidence interval on the bootstraps
    for i in range(len(lags)):
        y_ci[i, :] = np.percentile(y_bs[i, :], ci)

    return bs_A, bs_gamma, bs_omega, bs_phi, y_bs, y_ci


class decay_shm_fit():
    
    def __init__(self, lags, acf):
        """
        Initialise decay_shm_fit class.

        Parameters
        ----------
        lags : np.array
            Lags (i.e. x axis of the ACF plot).
        acf : np.array
            ACF (i.e. y axis of the ACF plot).

        Returns
        -------
        None.

        """
        # Store the ACF data
        self.lags = lags
        self.acf = acf

    def fit_SHM(self, A0=1.0, gamma0=1e-6,
                omega0=2 * np.pi / 100000, phi0=0):
        """
        Fit a damped SHM curve to the ACF as a function
        of lag.

        Parameters
        ----------
        A0 : float, optional
            Initial guess for A. The default is 1.0.
        gamma0 : float, optional
            Initial guess for gamma. The default is 1e-6.
        omega0 : float, optional
            Initial guess for omega. The default is 2 * np.pi / 100000.
        phi0 : float, optional
            Initial guess for phi. The default is 0.

        Returns
        -------
        None.

        """
        # Detrend the data, removing a linear trend
        self.linear_detrend_fit, self.linear_detrend_y,\
            self.linear_detrended_acf =\
            remove_linear_trend(self.lags, self.acf)
                
        # Normalise the Data
        self.normalised_acf, self.normalisation_mean, self.normalisation_std =\
            normalise_with_mean_subtraction(self.linear_detrended_acf)

        # Fit damped SHM
        self.A, self.gamma, self.omega, self.phi, self.y_fitted, self.popt,\
            self.pcov = fit_decaying_sinusoid(self.lags,
                                              self.normalised_acf,
                                              A0, gamma0, omega0, phi0)
        self.shm_chi_sq, self.shm_chi_p = chisquare(self.normalised_acf, self.y_fitted)
        # # Calculate confidence interval
        # sigma_ab = np.sqrt(np.diagonal(self.pcov))
        # self.ci_upper = damped_oscillator(self.lags, *(self.popt + sigma_ab))
        # self.ci_lower = damped_oscillator(self.lags, *(self.popt - sigma_ab))

    def create_text_labels(self):
        """
        Create attributes containing text strings for plot labels.

        Returns
        -------
        None.

        """
        # Linear Trend
        self.text_linear_trend = "y = " + \
            "{:.2e}".format(self.linear_detrend_fit.slope) + \
            " x\n+ " + "{:.2e}".format(self.linear_detrend_fit.intercept)
        # Linear Trend Pearson r
        self.text_linear_trend_pearson_r = '{:.2e}'.format(
            self.linear_detrend_fit.rvalue)

        # Normalisation mean and std
        self.text_normalisation_mean = "{:.2e}".format(self.normalisation_mean)
        self.text_normalisation_std = "{:.2e}".format(self.normalisation_std)

        # SHM fit
        self.text_shm_trend = "{:.2f}".format(self.A) + \
            "exp(-" + "{:.2e}".format(self.gamma) + "x)" + \
            "cos(" + "{:.2e}".format(self.omega) + "x + " + \
            "{:.2e}".format(self.phi) + ")"
        # SHM fit statistics
        self.text_shm_chi_sq = "{:.2f}".format(self.shm_chi_sq)
        self.text_shm_chi_p = "{:.2f}".format(self.shm_chi_p)

    def calc_confidence_interval(self):
        """
        Calculate the confidence interval on fit.

        Returns
        -------
        None.

        """
        self.bs_A, self.bs_gamma, self.bs_omega, self.bs_phi,\
            self.y_bs, self.y_ci = \
            damped_osc_ci(self.A, self.gamma, self.omega, self.phi,
                          self.pcov, self.lags, n_bootstrap=100,
                          ci=[2.5, 97.5])
