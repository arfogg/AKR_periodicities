# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:15:12 2024

@author: A R Fogg
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interpolate
import scipy.signal as signal

from numpy.fft import fft, ifft

from neurodsp.sim import sim_oscillation

import fastgoertzel as G

sys.path.insert(1, r'C:\Users\admin\Documents\wind_waves_akr_code\wind_utility')
import read_integrated_power


# interesting stuff here on simulating intermittent oscillations
# https://neurodsp-tools.github.io/neurodsp/auto_tutorials/sim/plot_SimulatePeriodic.html



def paper_plots():
    
    print('hello')
    
    # ----- Read in AKR intensity data -----
    years=np.array([1995,1996,1997,1998,1999,2000,2001,2002,2003,2004])
    input_df=read_integrated_power.concat_integrated_power_years(years, 'waters')
    
    
    # ----- END -----
    
    
    # NEED TO RESAMPLE/INTERPOLATE TO MAKE THE DATA AN EVEN TEMPORAL RESOLUTION
    #   SO THE FFT WORKS NICELY
    
    # ----- Resample temporally -----
    # Interpolate / resample the AKR intensity data so it's on an even
    #   temporal resolution of 3 minutes exactly.
    
    s_time=input_df.datetime_ut.iloc[0].ceil(freq='min')
    e_time=input_df.datetime_ut.iloc[-1].floor(freq='min')
    n_periods=np.floor((e_time-s_time)/pd.Timedelta(minutes=3))

    new_time_axis=pd.date_range(s_time, periods=n_periods, freq='3T')
    unix_time_axis=(new_time_axis - pd.Timestamp('1970-01-01')) / (pd.Timedelta(seconds=1))
    intensity_df=pd.DataFrame({'datetime_ut':new_time_axis,
                               'unix':unix_time_axis})
    
    func=interpolate.interp1d(input_df.unix,input_df['P_Wsr-1_100_650_kHz'])
    
    

    intensity_df['P_Wsr-1_100_650_kHz']=func(unix_time_axis) 
    #return intensity_df

    
    # ----- END -----
    
    # ----- FFT -----
    
    fast_fourier_transform(intensity_df)
    
    # ----- END -----
    
def test_dataset():
    
    # create a noisy sinusoid
    
    n_points=1001
    
    
    time=np.linspace(0,100,n_points)
    intensity=np.sin(time)+np.random.normal(0,.1,n_points)
    
    fig,ax=plt.subplots()
    ax.plot(time, intensity)
    
    
    
    
    return pd.DataFrame({'datetime_ut':time, 'P_Wsr-1_100_650_kHz':intensity})




def lomb_scargle(intensity_df):

    freqs_in_hrs=np.linspace(0,15,16)
    
    pgram=signal.lombscargle(intensity_df['datetime_ut'], intensity_df['P_Wsr-1_100_650_kHz'], freqs_in_hrs)
    
    fig,ax=plt.subplots()
    ax.plot(freqs_in_hrs, pgram)

def fast_fourier_transform(intensity_df):
    
    sp=np.fft.fft(intensity_df['P_Wsr-1_100_650_kHz'])
    #freq=np.fft.fftfreq(np.array(intensity_df['P_Wsr-1_100_650_kHz']).shape[-1], d=(3.*60.))
    
    #   how do we calc samples per second if 3 min res?
    sr = 1./(3.*60.) 
    N = len(sp)  # number of FFT points
    n = np.arange(N)    # 0 - N array. integers
    T = N/sr    # number of FFT points / number of obs per sec
    freq = n/T  # freqs fft is evaluated at
    print('apple', len(intensity_df))
    print('banana',N)
    print('max freq evaluated in Hz', np.max(freq))
    fig,ax=plt.subplots()
    ax.plot(freq, sp.real, label='real', color='darkorchid')
    ax.plot(freq, sp.imag, label='imag', color='gold')
    
    ax.set_title('FFT of 10 years AKR')
    ax.set_xlabel('freq in Hz i think')
    ax.set_ylabel('amplitude of some sort')
    
    ax.legend()
    
    
def test_goertzel():
    
    
    def wave(amp, freq, phase, x):
        return amp * np.sin(2*np.pi * freq * x + phase)


    x = np.arange(0, 512)
    y = wave(1, 1/128, 0, x)

    amp, phase = G.goertzel(y, 1/128)
    print(f'Goertzel Amp: {amp:.4f}, phase: {phase:.4f}')



    # Compared to max amplitude FFT output 
    ft = np.fft.fft(y)
    FFT = pd.DataFrame()
    FFT['amp'] = np.sqrt(ft.real**2 + ft.imag**2) / (len(y) / 2)
    FFT['freq'] = np.fft.fftfreq(ft.size, d=1)
    FFT['phase'] = np.arctan2(ft.imag, ft.real)

    max_ = FFT.iloc[FFT['amp'].idxmax()]
    print(f'FFT amp: {max_["amp"]:.4f}, '
            f'phase: {max_["phase"]:.4f}, '
            f'freq: {max_["freq"]:.4f}')
    
    a=np.full(len(FFT), np.nan)
    p=np.full(len(FFT), np.nan)
    for i in range(len(FFT)):
        a[i], p[i] = G.goertzel(y, FFT['freq'][i])
        
    fig,ax=plt.subplots(nrows=2)

    ax[0].plot(x,y)

    ax[1].plot(FFT['phase'], FFT['amp'], marker='.', linewidth=0.0, label='fft')
    #ax[1].plot(phase,amp, marker='*', fillstyle='none', markersize=15.0, linewidth=0.0, label='goertzel')
    ax[1].plot(p,a, marker='*', fillstyle='none', markersize=15.0, linewidth=0.0, label='goertzel')
    ax[1].legend()
    
    
    
    
    # WITH DATA
    
    
    
    
def testing_fft():
    # https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
    # ----- SIGNAL -----
    # sampling rate - is this samples per second???
    #   how do we calc samples per second if 3 min res?
    sr = 2000
    # sampling interval
    ts = 1.0/sr
    
    # Constructing their fake x axis.
    #    so this is 2000 samples in 1 second
    t = np.arange(0,1,ts)
    
    # Constructing their face y axis
    freq = 1.
    x = 3*np.sin(2*np.pi*freq*t)
    
    freq = 4
    x += np.sin(2*np.pi*freq*t)
    
    freq = 7   
    x += 0.5* np.sin(2*np.pi*freq*t)
    
    
    fig,ax=plt.subplots(ncols=3, figsize = (18, 6))
    ax[0].plot(t, x, 'orange')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_xlabel('time in seconds *i think*')
    ax[0].set_title('Observations')
    
    
    # # ----- FFT IN NUMPY -----
    
    
    
    # from numpy.fft import fft, ifft
    
    
    # X = fft(x)  # y axis for FFT plot
    # N = len(X)  # number of FFT points
    # n = np.arange(N)    # 0 - N array. integers
    # T = N/sr    # number of FFT points / number of obs per sec
    # freq = n/T  # freqs fft is evaluated at
    
    # #fig,ax=plt.subplots(ncols=2, figsize = (12, 6))

    
    # ax[1].stem(freq, np.abs(X), 'c', \
    #           markerfmt=" ", basefmt="-c") 
    # ax[1].set_xlabel('Freq (Hz)')
    # ax[1].set_ylabel('FFT Amplitude |X(freq)|')
    # ax[1].set_xlim(0, 10)
    # ax[1].set_title('FFT of observations')
    
    # ax[2].plot(t, ifft(X), 'blueviolet')
    # ax[2].set_xlabel('Time (s)')
    # ax[2].set_ylabel('Amplitude')
    # ax[2].set_title('Inverse FFT')
    
    # fig.tight_layout()

    
    generic_fft_function(t, x, sr)


    # Make a nice fake oscillating AKR signal
    top=1e7
    mid=1e6
    bot=1e5
    
    
    
    # Create fake time axis
    yr_secs = 365*24*60*60    # One year in seconds
    res_secs = 3*60   # Temporal resolution of 3 minutes
    osc_freq = 12. # Oscillation frequency in hours
    
    time=np.arange(0,yr_secs, res_secs)

    akr_osc=sim_oscillation(yr_secs,1/res_secs, 1/(osc_freq*60*60), cycle='sine')
    akr_osc=(akr_osc+2.0) * 1e6     # make positive and put to the order of AKR power
    
    
    fig,ax=plt.subplots()
    ax.plot(time,akr_osc)
    ax.set_xlim(0, 4*osc_freq*60*60)


def generic_fft_function(time, y, sampling_rate, plot=True):
    
    
    X = fft(y)  # y axis for FFT plot
    N = len(X)  # number of FFT points
    n = np.arange(N)    # 0 - N array. integers
    T = N/sampling_rate    # number of FFT points / number of obs per sec
    freq = n/T  # freqs fft is evaluated at
    
    #fig,ax=plt.subplots(ncols=2, figsize = (12, 6))

    if plot :
        
        fig,ax=plt.subplots(ncols=3, figsize = (18, 6))
        
        ax[0].plot(time, y, 'orange')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_xlabel('time in UNITS?')
        ax[0].set_title('Observations')
        
        ax[1].stem(freq, np.abs(X), 'c', \
                  markerfmt=" ", basefmt="-c") 
        ax[1].set_xlabel('Freq (Hz)')
        ax[1].set_ylabel('FFT Amplitude |X(freq)|')
        ax[1].set_xlim(0, 10)
        ax[1].set_title('FFT of observations')
        
        ax[2].plot(time, ifft(X), 'blueviolet')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Amplitude')
        ax[2].set_title('Inverse FFT')
        
        fig.tight_layout()
            
    
    
    