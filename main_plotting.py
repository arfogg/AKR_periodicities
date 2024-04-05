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

sys.path.insert(1, r'C:\Users\admin\Documents\wind_waves_akr_code\wind_utility')
import read_integrated_power

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
    return intensity_df

    
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



def fast_fourier_transform(intensity_df):
    
    sp=np.fft.fft(intensity_df['P_Wsr-1_100_650_kHz'])
    freq=np.fft.fftfreq(np.array(intensity_df['P_Wsr-1_100_650_kHz']).shape[-1], d=(3.*60.))
    fig,ax=plt.subplots()
    ax.plot(freq, sp.real, label='real')
    ax.plot(freq, sp.imag, label='imag')
    ax.legend()