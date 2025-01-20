# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:43:04 2024

@author: A R Fogg

Definitions of data with a 24 hour periodicity.
"""

import string

import numpy as np

import matplotlib.pyplot as plt

from neurodsp.sim import sim_oscillation, sim_powerlaw
from neurodsp.sim.utils import modulate_signal


alphabet = list(string.ascii_lowercase)
axes_labels = []
for a in alphabet:
    axes_labels.append('(' + a + ')')



def oscillating_signal(osc_freq, add_noise=True, noise_level=0.1,
                       add_amplitude_modulation=True,
                       create_data_gaps=True,
                       plot=False, n_plot_osc=4, fontsize=15,
                       noise_color='slateblue', mod_color='yellowgreen',
                       modf_color='deeppink', gaps_color='darkgrey'):
    """
    Function to create a timeseries of oscillating
    signal using neurodsp package.
    NEEDS UPDATING

    Parameters
    ----------
    osc_freq : float
        Period of the desired oscillation in hours.
    add_noise : bool, optional
        If True, random noise is added to the signal. The default is True.
    noise_level : float, optional
        Fraction of the intensity mean that the normal distribution scale
        is defined as. The default is 0.1.
    plot : bool, optional
        If plot == True, a diagnostic plot of the generated signal is
        presented. The default is False.
    n_plot_osc : int, optional
        Number of oscillations to show on the plot. The default is 4.
    fontsize: int, optional
        Defines fontsize on plot. The default is 15.

    Returns
    -------
    time : np.array
        Time axis in seconds.
    akr_osc : np.array
        Signal.

    """
    # Create fake time axis
    yr_secs = 365*24*60*60    # One year in seconds
    res_secs = 3*60   # Temporal resolution of 3 minutes

    time = np.arange(0, yr_secs, res_secs)

    akr_osc = sim_oscillation(n_seconds=yr_secs,
                              fs=1/res_secs,
                              freq=1/(osc_freq * 60 * 60),
                              cycle='gaussian',
                              std=osc_freq * 6 * 60)
    akr_osc = (akr_osc + 2.0) * 1e6
    # ^ make positive and put to the order of AKR power
    clean = akr_osc

    if add_noise:
        noisy = akr_osc + np.random.normal(loc=np.mean(akr_osc),
                                             scale=noise_level *
                                             np.mean(akr_osc),
                                             size=akr_osc.size)
        akr_osc = noisy
        
    if add_amplitude_modulation:
        # Simulate a different modulating signal, this time
        modulation = np.abs(sim_powerlaw(n_seconds=yr_secs,
                                         fs=1/res_secs, exponent=-2))

        # Apply the amplitude modulation to the signal
        amplitude_modulated = modulate_signal(akr_osc, modulation)
        akr_osc = amplitude_modulated

    if create_data_gaps:
        # The average length of the data gaps
        #   (in terms of number of data points)
        extract_length = 20  # 20 data points -> 60 minutes
        # Number of synthetic data gaps
        #   (approx 2 per day)
        n_gaps = int((yr_secs/(24*60*60))*2)
        start_point = np.random.choice(akr_osc.size, size=n_gaps)
        end_point = start_point + np.random.geometric(1.0 / extract_length, size=n_gaps)
        
        synthetic_gaps = akr_osc.copy()
        for i in range(n_gaps):
            if end_point[i] < len(akr_osc):
                synthetic_gaps[start_point[i]:end_point[i]] = np.nan
            else:
                synthetic_gaps[start_point[i]:] = np.nan
        akr_osc = synthetic_gaps.copy()

        

    if plot:
        fig, ax = plt.subplots(1, 2,
                               gridspec_kw={'width_ratios': [2, 1]},
                               figsize = (12, 5))
        # fig, ax = plt.subplots(figsize=(8, 5))
        
        # Time series
        ax[0].plot(time, clean, linewidth=1.0, color='black',
                label='Pure oscillator')

        xmax = n_plot_osc*osc_freq*60*60
        yi, = np.where(time <= xmax)
        yval = clean[yi]

        if add_noise:
            ax[0].plot(time, noisy, linewidth=1.0, label='Random Gaussian Noise',
                    color=noise_color)
            yval = np.append(yval, noisy[yi])
    
        if add_amplitude_modulation:
            # Plot modulated signal
            ax[0].plot(time, amplitude_modulated, linewidth=1.0,
                    label='Amplitude Modulated', color=mod_color)
            yval = np.append(yval, amplitude_modulated[yi])

            # Plot modulation
            mod_ax = ax[0].twinx()
            mod_leg = mod_ax.plot(time, modulation, linewidth=1.0,
                        label='Modulation factor', color=modf_color,
                        linestyle='dashed')
            mod_ax.set_ylabel('Modulation factor', color=modf_color,
                              fontsize=fontsize)
            mod_ax.tick_params(axis='y', labelcolor=(modf_color),
                               labelsize=fontsize)
            mod_ax.spines['right'].set_color(modf_color)

        if create_data_gaps:
            ax[0].plot(time, synthetic_gaps + 1E6, linewidth=1.0,
                       label='Synthetic Gaps\n(amplitude + 1E6)', color=gaps_color)
            yval = np.append(yval, synthetic_gaps[yi] + 1E6)
            horiz_gaps = np.where(np.isnan(akr_osc),
                                  np.nan, 1.05*np.nanmax(yval))
            ax[0].plot(time, horiz_gaps, marker='.', linewidth=0.,
                       markersize=0.1*fontsize, alpha=0.75, color=gaps_color,
                       label='Data Availability')
            yval = np.append(yval, horiz_gaps[yi])
            
        #ax[0].legend(fontsize=0.75*fontsize)
        
        # Set limits
        ax[0].set_xlim(0, xmax)
        ax[0].set_ylim(top=1.1*np.nanmax(yval))

        
        ax[0].set_xlabel('Time (hrs)', fontsize=fontsize)
        ax[0].set_ylabel('Signal Amplitude', fontsize=fontsize)
        #ax[0].tick_params(labelsize=fontsize)

        ticks_in_secs = []
        ticks_in_hrs = []
        for i in range(n_plot_osc):
            ax[0].axvline(float(i)*osc_freq*60*60, linestyle='dashed',
                       color='grey', linewidth=1.)
            ticks_in_secs.append(float(i)*osc_freq*60*60)
            ticks_in_hrs.append(str(i * osc_freq))
        ax[0].set_xticks(ticks_in_secs, ticks_in_hrs)
        
        
        leg_ln = [*ax[0].get_legend_handles_labels()[0], mod_leg[0]]
        leg_lab = [*ax[0].get_legend_handles_labels()[1], mod_leg[0].get_label()]
        ax[0].legend(leg_ln, leg_lab, fontsize=0.65*fontsize, loc='upper right')
        
        
        
        
        # Histogram
        hist_max = np.nanmax([np.nanmax(clean), np.nanmax(noisy),
                              np.nanmax(amplitude_modulated)])
        bins = np.linspace(0, hist_max, 25)
        ax[1].hist(clean, color='black', alpha=0.5, bins=bins,
                   edgecolor='black', label='Pure oscillator')
        ax[1].hist(noisy, color=noise_color, alpha=0.5, bins=bins,
                   edgecolor=noise_color,
                   label='Random\nGaussian\nNoise')
        ax[1].hist(amplitude_modulated, color=mod_color, alpha=0.5, bins=bins,
                   edgecolor=mod_color, label='Amplitude\nModulated')
        ax[1].hist(synthetic_gaps, facecolor='none', bins=bins, edgecolor='black',
                   hatch='x', label='Synthetic\nGaps')
        
        ax[1].set_xlabel('Signal Amplitude', fontsize=fontsize)
        ax[1].set_ylabel('Occurrence', fontsize=fontsize)

        
        mod_hax = ax[1].twiny()
        mod_bins = np.linspace(0, np.nanmax(modulation), 25)

        mod_hax.hist(modulation, color=modf_color, alpha=0.5, bins=mod_bins,
                    histtype='step', linewidth=2., label='Modulation\nfactor')
        ax[1].hist([], color=modf_color, alpha=0.5, bins=mod_bins,
                    histtype='step', linewidth=2., label='Modulation\nfactor')        
        mod_hax.set_xlabel('Modulation factor', color=modf_color,
                              fontsize=fontsize)
        mod_hax.tick_params(axis='x', labelcolor=(modf_color),
                               labelsize=fontsize)
        mod_hax.spines['top'].set_color(modf_color)
        
        # leg_ln = [*ax[1].get_legend_handles_labels()[0], mod_hleg[0]]
        # leg_lab = [*ax[1].get_legend_handles_labels()[1], mod_hleg[0].get_label()]
        # ax[1].legend(leg_ln, leg_lab, fontsize=0.75*fontsize, loc='upper right')
        ax[1].legend(fontsize=0.65*fontsize)        
        # add pink to legend
        # a, b labels

        for i, a in enumerate(ax):
            a.tick_params(labelsize=fontsize)       

            # a.legend(fontsize=0.75*fontsize, loc='upper right')
            
            t = a.text(0.05, 0.95, axes_labels[i], transform=a.transAxes,
                           fontsize=fontsize, va='top', ha='left')
            t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))


        fig.tight_layout()        

    return time, akr_osc
