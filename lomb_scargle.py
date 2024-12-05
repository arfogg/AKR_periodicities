# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:50:12 2024

@author: A R Fogg

Functions to run Lomb-Scargle analysis.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import scipy.signal as signal



# def test_LS():
        
#     rng = np.random.default_rng()

#     A = 2.
#     w0 = 1.  # rad/sec
#     nin = 150
#     nout = 100000

#     time = rng.uniform(0, 10*np.pi, nin)

#     y = A * np.cos(w0*time)

#     freqs = np.linspace(0.01, 10, nout)
#     periods = freq_to_period(freqs)
    
#     ls_pgram = generic_lomb_scargle(time, y, freqs)
    
    
#     plot_LS_summary(time, y, freqs, periods, ls_pgram,
#                     vertical_indicators=[])


def generic_lomb_scargle(time, y, freqs):
    
    # time in seconds
    # NaN rows removed
    


    ls_pgram = signal.lombscargle(time, y, freqs, normalize=True)

    return ls_pgram

def plot_LS_summary(time, y, freqs, periods, ls_pgram,
                    fontsize=15,
                    vertical_indicators=[],
                    pgram_fmt={'color': 'dimgrey', 'linewidth': 1.5},
                    vertical_ind_col='royalblue'):
                     # surrogate_period=None, surrogate_fft_amp=None,
                     # fft_xlims=[0, 36],
                     # signal_xlims=[np.nan, np.nan], signal_ymin=1.,
                     # fontsize=15,
                     # vertical_indicators=[],
                     # unix_to_dtime=False,
                     # resolution_lim=True,
                     # signal_y_log=False,
                     # input_fmt={'color': 'royalblue', 'linewidth': 1.},
                     # ifft_fmt={'color': 'royalblue', 'linewidth': 1.},
                     # input_ax=None, panel_label=True
    
    # fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
    # ax_t.plot(time, y, 'b+')
    # ax_t.set_xlabel('Time [s]')

    # ax_w.plot(freqs, ls_pgram)
    # ax_w.set_xlabel('Angular frequency [rad/s]')
    # ax_w.set_ylabel('Normalized amplitude')
    # plt.show()
    

    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(periods, ls_pgram, **pgram_fmt)
    ax.set_xscale('log')
    
    # Formatting
    ax.set_ylabel('Lomb-Scargle Normalised Amplitude', fontsize=fontsize)
    ax.set_xlabel('Period (hours)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    
    
    if vertical_indicators != []:
        for h in vertical_indicators:
            ax.axvline(h, color=vertical_ind_col, linestyle='dashed',
                          linewidth=1.5)
            trans = transforms.blended_transform_factory(ax.transData,
                                                         ax.transAxes)
            ax.text(h, 1.05, str(h), transform=trans,
                       fontsize=fontsize, va='top', ha='center',
                       color=vertical_ind_col)
 