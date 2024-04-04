# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:15:12 2024

@author: A R Fogg
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(1, r'C:\Users\admin\Documents\wind_waves_akr_code\wind_utility')
import read_integrated_power

def paper_plots():
    
    print('hello')
    
    # ----- Read in AKR intensity data -----
    years=np.array([1995,1996,1997,1998,1999,2000,2001,2002,2003,2004])
    intensity_df=read_integrated_power.concat_integrated_power_years(years, 'waters')
    
    
    # ----- END -----
    
    
    # NEED TO RESAMPLE/INTERPOLATE TO MAKE THE DATA AN EVEN TEMPORAL RESOLUTION
    #   SO THE FFT WORKS NICELY
    
    
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



def fast_fourier_transform(intensity_df):
    
    sp=np.fft.fft(intensity_df['P_Wsr-1_100_650_kHz'])
    freq=np.fft.fftfreq(np.array(intensity_df['P_Wsr-1_100_650_kHz']).shape[-1])
    fig,ax=plt.subplots()
    ax.plot(freq, sp.real, label='real')
    ax.plot(freq, sp.imag, label='imag')
    ax.legend()