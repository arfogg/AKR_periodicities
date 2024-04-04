# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:27:36 2024

@author: admin
"""
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
import os
import pathlib
import string



import read_wind_position
import read_waters_masked_data
import run_integrate_intensity


def determine_diurnal_variation_old_using_waters_data():
    # Program to read in the intensity variation with time for all 
    #   burst-masked data (perhaps also the waters-masked data to compare)
    #   and determine the diurnal variation in intensity by:
    #
    # - first sorting the data into LT bins - the diurnal variation
    #       will vary depending on observing location
    # - then do a superposed epoch analysis of the intensity values 
    #       over a 24 hour day, and plot the resulting average intensity curve
    #
    # Some notes on plotting:
    #   one panel per LT bin (1 hour LT bins?)
    #   highlight times in each LT bin that are undersampled
    
    print('hello')
    
    # For now, read in the waters masked data as the format should be the same
    
    year=2002
    month=10
    # Shorter, test dataset
  #  akr_df=read_waters_masked_data.concat_monthly_data(year,month)
    #akr_df=read_waters_masked_data.concat_annual_data(year)
    
    # Convert to intensity variation in time
    waters_masked_intensity_fn=r'Users\admin\Documents\data\fogg_burst_data\summary\waters_intensity_summary_'+str(year)+'.csv'

    waters_masked_intensity_fn=os.path.join('C:'+os.sep,waters_masked_intensity_fn)

    if pathlib.Path(waters_masked_intensity_fn).is_file():
        print('WARNING: reading in pre-sorted data, if you have made a change to the original \nWaters-Masked data please delete the intensity summary file')
        print('Reading in intensity summary data for Waters-Masked data from: ', waters_masked_intensity_fn)
        # correct the parse dates bit
        intensity_summary_df=pd.read_csv(waters_masked_intensity_fn,delimiter=',', parse_dates=['timestamp'], float_precision='round_trip')
    else:
        print('Determining the intensity variation in time')
         
        # Shorter, test dataset
        #akr_df_fresh=read_waters_masked_data.concat_monthly_data(year,month)
        akr_df_fresh=read_waters_masked_data.concat_annual_data(year)
        # Read in the Wind position data, and interpolate it to the same time resolution
        #   as the akr_df
        wind_pos_df=read_wind_position.read_data(year)
        akr_df=read_wind_position.interpolate_wind_data(wind_pos_df,year,akr_df_fresh)
        print(akr_df.columns)

        # Determine the available frequencies    
        freqs = np.sort(akr_df['freq'].dropna().unique())

        # Initialise arrays
        intensity_arr=np.empty([np.array(akr_df['freq'].unique()).size,np.array(akr_df['datetime_ut'].unique()).size])  
        intensity_arr[:]=np.nan
        time_arr=np.empty([np.array(akr_df['freq'].unique()).size,np.array(akr_df['unix'].unique()).size],dtype=dt.datetime)
        time_arr[:]=np.nan
        un_arr=np.empty([np.array(akr_df['freq'].unique()).size,np.array(akr_df['unix'].unique()).size])
        un_arr[:]=np.nan

        # In a for loop create a matrix for each variable
        #   where each row is the intensity variation in time
        #   and subsequent rows are for different frequency bands
        print('Sorting data into intensity variation in time, for a given frequency band')
        for i in range(freqs.size):
            # Extract the dataframe rows which are at frequency i
            df=akr_df.loc[akr_df['freq']==freqs[i]]
            
            # Power and unix time arrays
            pwr_arr=np.array(df['mask_flux_si'])
            tme_arr_u=np.array(df['unix'])
        
            # convert to datetime (can't remember why this was necessary, perhaps for plotting?)
            tme_arr=np.empty(tme_arr_u.size,dtype=dt.datetime)
            for j in range(tme_arr.size):
                tme_arr[j]=dt.datetime.utcfromtimestamp(tme_arr_u[j])
                #print('size',pwr_arr.size)
            intensity_arr[i,:]=pwr_arr
            time_arr[i,:]=tme_arr
            un_arr[i,:]=tme_arr_u
    
        # Calculate the integrated/total and mean intensity at each time
        print('Estimating the total and average intensity variation in time')
        total_intensity=np.empty(time_arr[0,:].size)
        total_intensity[:]=np.nan
        av_intensity=np.empty(time_arr[0,:].size)
        av_intensity[:]=np.nan

 #       ts=np.empty(time_arr[0,:].size,dtype='datetime64[ns]')
        lt_gse=np.empty(time_arr[0,:].size)
        lat_gse=np.empty(time_arr[0,:].size)
        lon_gse=np.empty(time_arr[0,:].size)
        radius=np.empty(time_arr[0,:].size)
        
        
  #      print('banana',type(ts[0]))
  #      #ts[:]=np.nan
        lt_gse[:]=np.nan
        lat_gse[:]=np.nan
        lon_gse[:]=np.nan
        radius[:]=np.nan
        
        ts=np.sort(akr_df['datetime_ut'].unique())
        for i in range(time_arr[0,:].size):
            # If there are nans, use nansum and nanmean to ignore the nans
            # If there are only nans, use normal sum and mean to end up with a nan value
            # So to do this we find out how many values are not a nan using ~np.isnan()
            
            # So if all are nans, do 
            if np.array(np.where(~np.isnan(intensity_arr[:,i]))).size == 0:
                total_intensity[i]=np.sum(intensity_arr[:,i])
                av_intensity[i]=np.mean(intensity_arr[:,i])
            else:   # Otherwise
                total_intensity[i]=np.nansum(intensity_arr[:,i])
                av_intensity[i]=np.nanmean(intensity_arr[:,i])
            
            #print(time_arr[0,i], type(time_arr[0,i]))
            #print(un_arr[0,i],type(un_arr[0,i]),type(pd.Timestamp(un_arr[0,i],unit='s')))
            #print(type(ts[i]))
            #print(akr_df.datetime_ut)
            # Compare df with unix values converted to Timestamps
            #print('LOOP',i)
            #print(pd.Timestamp(un_arr[0,i], unit='s'))
            #print(np.where( akr_df.datetime_ut == pd.Timestamp(un_arr[0,i],unit='s') ))            
            
            #print(np.where( akr_df.datetime_ut == ts[i]))
            #ts[i]=pd.Timestamp(un_arr[0,i],unit='s')
            #t_ind,=np.where( akr_df.datetime_ut == pd.Timestamp(un_arr[0,i],unit='s') )

            t_ind,=np.where( akr_df.datetime_ut == ts[i] )

            #print('hello',t_ind)
            # Select the first t_ind because the LT will be the same when the time is the same
            lt_gse[i]=akr_df['decimal_gseLT'].iloc[t_ind[0]]
            lat_gse[i]=akr_df['lat_gse'].iloc[t_ind[0]]
            lon_gse[i]=akr_df['lon_gse'].iloc[t_ind[0]]
            radius[i]=akr_df['radius'].iloc[t_ind[0]]
            #(lt_gse[i])


        n_filled=np.empty(ts.size)
        n_filled[:]=np.nan
        # Loop through each time step, and count the amount of bins that are filled
        print('Counting the number of filled frequency bins at each time')
        loop_start=dt.datetime.now()
        loop_time=dt.datetime.now()
        for i in range(ts.size):
            # Original, slow way
            #n_filled[i]=np.array( np.where( ~np.isnan(   akr_df['mask_flux_si'].iloc[np.where(akr_df['datetime_ut'] == datetimes[i])]   ) ) ).size
            temp,=np.where( ~np.isnan(intensity_arr[:,i]) )
            n_filled[i]=temp.size
            # print(akr_df['mask_flux_si'].iloc[np.where(akr_df['datetime_ut'] == datetimes[i])])
           # print(np.where( ~np.isnan(   akr_df['mask_flux_si'].iloc[np.where(akr_df['datetime_ut'] == datetimes[i])]   ) ))
            #print(n_filled[i] - np.array( akr_df['mask_flux_si'].loc[ (akr_df['datetime_ut'] == datetimes[i]) & (np.isnan(akr_df['mask_flux_si']) == False) ] ).size)
           # print(np.array( akr_df['mask_flux_si'].loc[ (akr_df['datetime_ut'] == datetimes[i]) & (np.isnan(akr_df['mask_flux_si']) == False) ] ).size)
            
            if dt.datetime.now() > (loop_time+dt.timedelta(minutes=20)):
                loop_time=dt.datetime.now()
                print('still counting filled bins, '+str((float(i)/float(ts.size))*100.0)+'% through, time elapsed: '+str(loop_time-loop_start))
    
        print('Counting of filled bins complete, time elapsed: ', dt.datetime.now()-loop_start)

       
        intensity_summary_df=pd.DataFrame({'timestamp':ts,'unix':un_arr[0,:],'lt_gse':lt_gse,'lat_gse':lat_gse,
                                           'lon_gse':lon_gse, 'radius':radius,'total_intensity':total_intensity,
                                           'n_filled':n_filled})
        #print(intensity_summary_df)
        intensity_summary_df.to_csv(waters_masked_intensity_fn,index=False)
        
    print(intensity_summary_df)
    print(intensity_summary_df.columns)

    # Round down timestamps to nearest minute
    # --> we are averaging any observation over that 1 minute interval
    # --> necessary as Wind WAVES data has uneven ~3min3sec resolution with decimal seconds
    rounded_datetimes=np.empty(np.array(intensity_summary_df.timestamp).size,dtype=dt.datetime)
    decimal_hour=np.empty(np.array(intensity_summary_df.timestamp).size,dtype=float)
    for p in range(rounded_datetimes.size):
        rounded_datetimes[p]=dt.datetime(intensity_summary_df.timestamp[p].year, intensity_summary_df.timestamp[p].month, intensity_summary_df.timestamp[p].day, intensity_summary_df.timestamp[p].hour, intensity_summary_df.timestamp[p].minute, 0)
        decimal_hour[p]=rounded_datetimes[p].hour+( (rounded_datetimes[p].minute)/60.0)
 #       print('pineapple')
 #       print(intensity_summary_df.timestamp[p])
 #       print(rounded_datetimes[p])
 #       print(decimal_hour[p])
 #       return
    unique_dh=np.sort(pd.Series(decimal_hour).unique())
 #   print(unique_dh)
    intensity_summary_df['rounded_datetimes']=rounded_datetimes
    intensity_summary_df['decimal_hour']=decimal_hour
 #   print(intensity_summary_df)


    # Define some constants relating to the LT and UT sorting:
    #    (could parse these in if desired)
    n_intervals=24*60 # Number of integer minute intervals in a day
   # n_lt_sectors=24 # Number of local time sectors to divide data into
   # lt_sector_width=1 # Width of LT sectors in hours
    n_lt_sectors=12
    lt_sector_width=2

    dhs=np.arange(n_intervals)*(1/60)

 #   spe_intensity=np.empty( unique_dh.size )
 #   spe_intensity[:]=np.nan
#
 #   for w in range(unique_dh.size):
 #       spe_intensity[w]=np.nanmedian( total_intensity[np.where(decimal_hour==unique_dh[w])] )
    

    # Initialise an empty array to contain th
    sorted_intensity=np.empty([n_intervals,n_lt_sectors])
    sorted_intensity[:,:]=np.nan

    for i in range(n_lt_sectors):
        #print('LT sector: ',i)
        #print('lower bound: ',i-(lt_sector_width/2),i+(lt_sector_width/2))
        
        if i > 0:
            lower_bound=(i*lt_sector_width)-(lt_sector_width/2)
            upper_bound=(i*lt_sector_width)+(lt_sector_width/2)
            # Select the dataframe rows within this local time sector
            subset_df=intensity_summary_df.loc[ (intensity_summary_df.lt_gse >= lower_bound) & 
                                               (intensity_summary_df.lt_gse < upper_bound) ]
        else:   # Treat LT = 0 differently
            lower_bound=(i*lt_sector_width)-(lt_sector_width/2)+24#(n_lt_sectors*lt_sector_width)+(lt_sector_width/2)-lt_sector_width
            upper_bound=(i*lt_sector_width)+(lt_sector_width/2)
            # Select the dataframe rows within this local time sector
            subset_df=intensity_summary_df.loc[ (intensity_summary_df.lt_gse >= lower_bound) | 
                                               (intensity_summary_df.lt_gse < upper_bound) ]
            
        #print(i,lower_bound,upper_bound)
        #print(subset_df.lt_gse.unique())
        #return
        for j in range(n_intervals):
            # Check for empty df
            if np.where( ~np.isnan( subset_df['total_intensity'].loc[subset_df.decimal_hour == dhs[j]] ) )[0].size > 0:            
                sorted_intensity[j,i]=np.nanmedian( subset_df['total_intensity'].loc[subset_df.decimal_hour == dhs[j]] )
            

    fig,ax = plt.subplots(nrows=n_lt_sectors,figsize=(9,15))
    for i in range(n_lt_sectors):
        ax[i].plot(dhs,sorted_intensity[:,i],marker='o',fillstyle='none',linewidth=0,label=str(i*lt_sector_width)+' LT')
        ax[i].set_xlim([0,24])
        ax[i].legend()
        
        
        if i == n_lt_sectors-1:
            ax[i].set_xlabel('UT (hours)')
            ax[i].set_ylabel('Intensity (units?!)')

def determine_diurnal_variation_burst_data():
    # Program to read in the intensity variation with time for all 
    #   burst-masked data and determine the diurnal variation in intensity by:
    #
    # - first sorting the data into LT bins - the diurnal variation
    #       will vary depending on observing location
    # - then do a superposed epoch analysis of the intensity values 
    #       over a 24 hour day, and plot the resulting average intensity curve
    #
    # Some notes on plotting:
    #   one panel per LT bin (1 hour LT bins?)
    #   highlight times in each LT bin that are undersampled
    
    print('hello')
    
    # For now, read in the waters masked data as the format should be the same
    
    year=2002
    month=10
    # Shorter, test dataset
  #  akr_df=read_waters_masked_data.concat_monthly_data(year,month)
    #akr_df=read_waters_masked_data.concat_annual_data(year)
    
    # Convert to intensity variation in time
    intensity_fn=r'Users\admin\Documents\data\fogg_burst_data\summary\combined_burst_intensity_with_location'+str(year)+'.csv'

    intensity_fn=os.path.join('C:'+os.sep,intensity_fn)

    if pathlib.Path(intensity_fn).is_file():
        print('WARNING: reading in pre-calculated data, if you have made a change to the original \nBurst-Masked data please delete the intensity summary file')
        print('Reading in intensity summary data for Burst-Masked data from: ', intensity_fn)
        # correct the parse dates bit
        intensity_summary_df=pd.read_csv(intensity_fn,delimiter=',', parse_dates=['datetime_ut'], float_precision='round_trip')
    else:
        # Read in the integrated intensity
        clean_intensity_csv=r'Users\admin\Documents\data\fogg_burst_data\summary\fogg_combined_burst_intensity_summary_'+str(year)+'.csv'
        clean_intensity_csv=os.path.join('C:'+os.sep,clean_intensity_csv)
        print('Reading: ',clean_intensity_csv)
        
        input_int_df=pd.read_csv(clean_intensity_csv,delimiter=',', parse_dates=['datetime_ut'], float_precision='round_trip')
        #power: P_Wsr-1_100_650_kHz
        
        # Calculate unix time of each time
        input_int_df['unix']=input_int_df['datetime_ut'].astype(np.int64) /10**9

        # Combine with location info
        cyear_position_df=read_wind_position.read_data(year)
        intensity_summary_df=read_wind_position.interpolate_wind_data(cyear_position_df,year,input_int_df)
        
        intensity_summary_df.to_csv(intensity_fn,index=False)
        
    intensity_summary_df['lt_gse']=intensity_summary_df['decimal_gseLT']
    intensity_summary_df['integrated_intensity']=intensity_summary_df['P_Wsr-1_100_650_kHz']
    print(intensity_summary_df)
    print(intensity_summary_df.columns)

    # Round down timestamps to nearest minute
    # --> we are averaging any observation over that 1 minute interval
    # --> necessary as Wind WAVES data has uneven ~3min3sec resolution with decimal seconds
    rounded_datetimes=np.empty(np.array(intensity_summary_df.datetime_ut).size,dtype=dt.datetime)
    decimal_hour=np.empty(np.array(intensity_summary_df.datetime_ut).size,dtype=float)
    for p in range(rounded_datetimes.size):
        rounded_datetimes[p]=dt.datetime(intensity_summary_df.datetime_ut[p].year, intensity_summary_df.datetime_ut[p].month, intensity_summary_df.datetime_ut[p].day, intensity_summary_df.datetime_ut[p].hour, intensity_summary_df.datetime_ut[p].minute, 0)
        decimal_hour[p]=rounded_datetimes[p].hour+( (rounded_datetimes[p].minute)/60.0)
 
    unique_dh=np.sort(pd.Series(decimal_hour).unique())
    intensity_summary_df['rounded_datetimes']=rounded_datetimes
    intensity_summary_df['decimal_hour']=decimal_hour


    # Define some constants relating to the LT and UT sorting:
    #    (could parse these in if desired)
    n_intervals=24*60 # Number of integer minute intervals in a day
   # n_lt_sectors=24 # Number of local time sectors to divide data into
   # lt_sector_width=1 # Width of LT sectors in hours
    #n_lt_sectors=12
    #lt_sector_width=2
    n_lt_sectors=6
    lt_sector_width=4

    dhs=np.arange(n_intervals)*(1/60)

    # Initialise an empty array to contain th
    sorted_intensity=np.empty([n_intervals,n_lt_sectors])
    sorted_intensity[:,:]=np.nan
    n_ave=np.empty([n_intervals,n_lt_sectors])
    n_ave[:]=np.nan
    for i in range(n_lt_sectors):
        #print('LT sector: ',i)
        #print('lower bound: ',i-(lt_sector_width/2),i+(lt_sector_width/2))
        
        if i > 0:
            lower_bound=(i*lt_sector_width)-(lt_sector_width/2)
            upper_bound=(i*lt_sector_width)+(lt_sector_width/2)
            # Select the dataframe rows within this local time sector
            subset_df=intensity_summary_df.loc[ (intensity_summary_df.lt_gse >= lower_bound) & 
                                               (intensity_summary_df.lt_gse < upper_bound) ]
        else:   # Treat LT = 0 differently
            lower_bound=(i*lt_sector_width)-(lt_sector_width/2)+24#(n_lt_sectors*lt_sector_width)+(lt_sector_width/2)-lt_sector_width
            upper_bound=(i*lt_sector_width)+(lt_sector_width/2)
            # Select the dataframe rows within this local time sector
            subset_df=intensity_summary_df.loc[ (intensity_summary_df.lt_gse >= lower_bound) | 
                                               (intensity_summary_df.lt_gse < upper_bound) ]
            
        #print(i,lower_bound,upper_bound,' points ',np.array(subset_df.lt_gse.unique()).size)
        #print(subset_df.lt_gse.unique())
        #return
        for j in range(n_intervals):
            # Check for empty df
            if np.where( ~np.isnan( subset_df['integrated_intensity'].loc[subset_df.decimal_hour == dhs[j]] ) )[0].size > 0:            
                sorted_intensity[j,i]=np.nanmedian( subset_df['integrated_intensity'].loc[subset_df.decimal_hour == dhs[j]] )
                #print(subset_df['integrated_intensity'].loc[subset_df.decimal_hour == dhs[j]])
                n_ave[j,i]=np.array(subset_df['integrated_intensity'].loc[subset_df.decimal_hour == dhs[j]]).size
                #return
            else:
                n_ave[j,i]=0

    #return
#cmap = matplotlib.cm.get_cmap('Spectral')
#norm = matplotlib.colors.Normalize(vmin=10.0, vmax=20.0)

    print('Plotting...')
    cmap=matplotlib.cm.get_cmap('plasma')
    norm=matplotlib.colors.Normalize(vmin=np.nanmin(n_ave),vmax=np.nanmax(n_ave))
    

    fig,ax = plt.subplots(nrows=n_lt_sectors,figsize=(9,15))
    for i in range(n_lt_sectors):
        #ax[i].plot(dhs,sorted_intensity[:,i],marker='o',fillstyle='none',linewidth=0,label=str(i*lt_sector_width)+' LT')
        for k in range(sorted_intensity[:,i].size):
            if sorted_intensity[k,i]>0:
               ax[i].plot(dhs[k],sorted_intensity[k,i],marker='o',fillstyle='none',linewidth=0,color=cmap(norm(n_ave[k,i])))#,label=str(i*lt_sector_width)+' LT')
        ax[i].legend(handles=[ 
            matplotlib.lines.Line2D([0], [0], marker='o', label=str(i*lt_sector_width)+' LT',
                    linewidth=0, color=cmap(0), markersize=10, fillstyle='none')], loc='upper right')

        ax[i].set_xlim([0,24])
        ax[i].set_yscale('log')
        #ax[i].legend()
        
        
        if i == n_lt_sectors-1:
            ax[i].set_xlabel('UT (hours)')
            ax[i].set_ylabel('Intensity (units?!)')

    output_png_fn=r'Users\admin\Documents\figures\fogg_burst_vs_omni\diurnal_variation_'+str(year)+'_'+str(lt_sector_width)+'lt_'+'combined_burst_data.png'
    output_png_fn=os.path.join('C:'+os.sep,output_png_fn)
    fig.savefig(output_png_fn, bbox_inches='tight')

def determine_diurnal_variation_waters_data():
    # Program to read in the intensity variation with time for all 
    #   burst-masked data and determine the diurnal variation in intensity by:
    #
    # - first sorting the data into LT bins - the diurnal variation
    #       will vary depending on observing location
    # - then do a superposed epoch analysis of the intensity values 
    #       over a 24 hour day, and plot the resulting average intensity curve
    #
    # Some notes on plotting:
    #   one panel per LT bin (1 hour LT bins?)
    #   highlight times in each LT bin that are undersampled
    
    print('hello')
    
    # For now, read in the waters masked data as the format should be the same
    
    year=2002
    month=10
    # Shorter, test dataset
  #  akr_df=read_waters_masked_data.concat_monthly_data(year,month)
    #akr_df=read_waters_masked_data.concat_annual_data(year)
    
    print('Have put the following into the read in function, needs editing and checking here!')
    print('Exiting...')
    return
    
    # Read in Waters masked intensity data:
    input_int_df=run_integrate_intensity.run_waters_masked_year(year)
    
    input_int_df['unix']=input_int_df['datetime_ut'].astype(np.int64) /10**9
 
    
    # Combine with location info
    cyear_position_df=read_wind_position.read_data(year)
    intensity_summary_df=read_wind_position.interpolate_wind_data(cyear_position_df,year,input_int_df)
        
    intensity_fn=r'Users\admin\Documents\data\fogg_burst_data\summary\waters_intensity_with_location'+str(year)+'.csv'
    intensity_fn=os.path.join('C:'+os.sep,intensity_fn)
    intensity_summary_df.to_csv(intensity_fn,index=False)
        
    intensity_summary_df['lt_gse']=intensity_summary_df['decimal_gseLT']
    intensity_summary_df['integrated_intensity']=intensity_summary_df['P_Wsr-1_100_650_kHz']
    print(intensity_summary_df)
    print(intensity_summary_df.columns)

    # Round down timestamps to nearest minute
    # --> we are averaging any observation over that 1 minute interval
    # --> necessary as Wind WAVES data has uneven ~3min3sec resolution with decimal seconds
    rounded_datetimes=np.empty(np.array(intensity_summary_df.datetime_ut).size,dtype=dt.datetime)
    decimal_hour=np.empty(np.array(intensity_summary_df.datetime_ut).size,dtype=float)
    for p in range(rounded_datetimes.size):
        rounded_datetimes[p]=dt.datetime(intensity_summary_df.datetime_ut[p].year, intensity_summary_df.datetime_ut[p].month, intensity_summary_df.datetime_ut[p].day, intensity_summary_df.datetime_ut[p].hour, intensity_summary_df.datetime_ut[p].minute, 0)
        decimal_hour[p]=rounded_datetimes[p].hour+( (rounded_datetimes[p].minute)/60.0)
 
    unique_dh=np.sort(pd.Series(decimal_hour).unique())
    intensity_summary_df['rounded_datetimes']=rounded_datetimes
    intensity_summary_df['decimal_hour']=decimal_hour


    # Define some constants relating to the LT and UT sorting:
    #    (could parse these in if desired)
    n_intervals=24*60 # Number of integer minute intervals in a day
   # n_lt_sectors=24 # Number of local time sectors to divide data into
   # lt_sector_width=1 # Width of LT sectors in hours
    n_lt_sectors=12
    lt_sector_width=2

    dhs=np.arange(n_intervals)*(1/60)

    # Initialise an empty array to contain th
    sorted_intensity=np.empty([n_intervals,n_lt_sectors])
    sorted_intensity[:,:]=np.nan
    n_ave=np.empty([n_intervals,n_lt_sectors])
    n_ave[:,:]=np.nan
    for i in range(n_lt_sectors):
        #print('LT sector: ',i)
        #print('lower bound: ',i-(lt_sector_width/2),i+(lt_sector_width/2))
        
        if i > 0:
            lower_bound=(i*lt_sector_width)-(lt_sector_width/2)
            upper_bound=(i*lt_sector_width)+(lt_sector_width/2)
            # Select the dataframe rows within this local time sector
            subset_df=intensity_summary_df.loc[ (intensity_summary_df.lt_gse >= lower_bound) & 
                                               (intensity_summary_df.lt_gse < upper_bound) ]
        else:   # Treat LT = 0 differently
            lower_bound=(i*lt_sector_width)-(lt_sector_width/2)+24#(n_lt_sectors*lt_sector_width)+(lt_sector_width/2)-lt_sector_width
            upper_bound=(i*lt_sector_width)+(lt_sector_width/2)
            # Select the dataframe rows within this local time sector
            subset_df=intensity_summary_df.loc[ (intensity_summary_df.lt_gse >= lower_bound) | 
                                               (intensity_summary_df.lt_gse < upper_bound) ]
            
        #print(i,lower_bound,upper_bound,' points ',np.array(subset_df.lt_gse.unique()).size)
        #print(subset_df.lt_gse.unique())
        #return
        for j in range(n_intervals):
            # Check for empty df
            if np.where( ~np.isnan( subset_df['integrated_intensity'].loc[subset_df.decimal_hour == dhs[j]] ) )[0].size > 0:            
                sorted_intensity[j,i]=np.nanmedian( subset_df['integrated_intensity'].loc[subset_df.decimal_hour == dhs[j]] )
                #print(subset_df['integrated_intensity'].loc[subset_df.decimal_hour == dhs[j]])
                n_ave[j,i]=np.array(subset_df['integrated_intensity'].loc[subset_df.decimal_hour == dhs[j]]).size
                #return
            else:
                n_ave[j,i]=0

    #return
#cmap = matplotlib.cm.get_cmap('Spectral')
#norm = matplotlib.colors.Normalize(vmin=10.0, vmax=20.0)

    print('Plotting...')
    cmap=matplotlib.cm.get_cmap('plasma')
    norm=matplotlib.colors.Normalize(vmin=np.nanmin(n_ave),vmax=np.nanmax(n_ave))
    

    fig,ax = plt.subplots(nrows=n_lt_sectors,figsize=(9,15))
    for i in range(n_lt_sectors):
        #ax[i].plot(dhs,sorted_intensity[:,i],marker='o',fillstyle='none',linewidth=0,label=str(i*lt_sector_width)+' LT')
        for k in range(sorted_intensity[:,i].size):
            if sorted_intensity[k,i]>0:
               ax[i].plot(dhs[k],sorted_intensity[k,i],marker='o',fillstyle='none',linewidth=0,color=cmap(norm(n_ave[k,i])))#,label=str(i*lt_sector_width)+' LT')
        ax[i].legend(handles=[ 
            matplotlib.lines.Line2D([0], [0], marker='o', label=str(i*lt_sector_width)+' LT',
                    linewidth=0, color=cmap(0), markersize=10, fillstyle='none')], loc='upper right')

        ax[i].set_xlim([0,24])
        #ax[i].legend()
        
        
        if i == n_lt_sectors-1:
            ax[i].set_xlabel('UT (hours)')
            ax[i].set_ylabel('Intensity (units?!)')

    output_png_fn=r'Users\admin\Documents\figures\fogg_burst_vs_omni\diurnal_variation_'+str(year)+'_'+str(lt_sector_width)+'lt_'+'waters_mask_data.png'
    output_png_fn=os.path.join('C:'+os.sep,output_png_fn)
    fig.savefig(output_png_fn, bbox_inches='tight')

def diurnal_variation():
    
    # Look for a period of a few days spent in the same approx 2-3 hour
    #   local time sector
    
    # Then plot integrated intensity for Waters data over that period
    # Overplot onto this, the burst intensity
    
    year=2002
    
    cyear_position_df=read_wind_position.read_data(year)
    
    # Define LT sectors
    lt_sector_width=3
    n_sectors=24/lt_sector_width
    n_sectors=int(n_sectors)
    sector_labels=list(string.ascii_lowercase[:int(n_sectors)])
    
    sector_mid=np.empty(int(n_sectors))
    sector_lower=np.empty(int(n_sectors))
    sector_upper=np.empty(int(n_sectors))
    sector_mid[:]=np.nan
    sector_lower[:]=np.nan
    sector_upper[:]=np.nan
    
    for i in range(n_sectors):
        #print('LT sector: ',i*lt_sector_width)
        #print('lower bound: ',i-(lt_sector_width/2),i+(lt_sector_width/2))
        sector_mid[i]=i*lt_sector_width        
        if i > 0:
            sector_lower[i]=(i*lt_sector_width)-(lt_sector_width/2)
            sector_upper[i]=(i*lt_sector_width)+(lt_sector_width/2)
        else:
            sector_lower[i]=(i*lt_sector_width)-(lt_sector_width/2)+24
            sector_upper[i]=(i*lt_sector_width)+(lt_sector_width/2)
        #print(sector_labels[i],': ',sector_lower[i],' - ',sector_mid[i],' - ',sector_upper[i])
        
    n_iter=np.array(cyear_position_df.decimal_gseLT).size
    lt=np.empty(n_iter,dtype='str')
    lt[:]='undetermined'
    for i in range(n_iter):
 #   for i in range(10):
        
        for j in range(n_sectors):
            if j > 0:
                #print(j)
                if (sector_lower[j] <= cyear_position_df['decimal_gseLT'].iloc[i]) & (sector_upper[j] > cyear_position_df['decimal_gseLT'].iloc[i]):
                    lt[i]=sector_labels[j]
                    break
            else:
                #print(j)
                if (sector_lower[j] <= cyear_position_df['decimal_gseLT'].iloc[i]) | (sector_upper[j] > cyear_position_df['decimal_gseLT'].iloc[i]):
                    lt[i]=sector_labels[j]
                    break
        
        #print(i,cyear_position_df['decimal_gseLT'].iloc[i], lt[i])

    # Locate long periods in same sector
    s_i=np.empty(n_sectors,dtype='int')
    e_i=np.empty(n_sectors,dtype='int') 
    s_i[:]=np.nan
    e_i[:]=np.nan
    
    for i in range(n_sectors):
        pairs=np.where( np.diff( np.hstack(([False],lt==sector_labels[i],[False])) ) )[0].reshape(-1,2)
        print(sector_labels[i],': ',sector_lower[i],' - ',sector_mid[i],' - ',sector_upper[i])
        s_i[i] = pairs[np.diff(pairs,axis=1).argmax(),0]
        e_i[i] = pairs[np.diff(pairs,axis=1).argmax(),1]
        #print('longest: ',start_longest_seq,' to ',end_longest_seq)

        #print('Longest time in this sector: ', ((e_i[i]-s_i[i])*12)/60.0, ' hours')

    w_intensity_fn=r'Users\admin\Documents\data\fogg_burst_data\summary\waters_intensity_with_location'+str(year)+'.csv'
    w_intensity_fn=os.path.join('C:'+os.sep,w_intensity_fn)
    print('Reading: ',w_intensity_fn)
    w_intensity_df=pd.read_csv(w_intensity_fn,delimiter=',', parse_dates=['datetime_ut'], float_precision='round_trip')


   # b_intensity_fn=r'Users\admin\Documents\data\fogg_burst_data\summary\combined_burst_intensity_with_location'+str(year)+'.csv'
  #  b_intensity_df=pd.read_csv(b_intensity_fn,delimiter=',', parse_dates=['datetime_ut'], float_precision='round_trip')




    k=6
    
    fig,ax=plt.subplots()
    
    w_intensity_df['P_Wsr-1_100_650_kHz']=w_intensity_df['P_Wsr-1_100_650_kHz'].replace(0,np.nan)
    
    ax.plot(w_intensity_df['datetime_ut'],w_intensity_df['P_Wsr-1_100_650_kHz'])
    #ax.set_xlim(cyear_position_df['datetime'].iloc[s_i[k]],cyear_position_df['datetime'].iloc[e_i[k]])
    ax.set_xlim(dt.datetime(2002,4,10),dt.datetime(2002,4,25))
   # ax.set_ylim(top=0.2E8)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d\n%H:%M"))
    ax.set_yscale('log')

#    print(type( intensity_df['datetime_ut'].iloc[0] ), intensity_df['datetime_ut'].iloc[0].tz )
  #  print(type( cyear_position_df['datetime'].iloc[s_i[k]].tz_convert('utc') ), cyear_position_df['datetime'].iloc[s_i[k]].tz_convert('utc').tz,
   #       cyear_position_df['datetime'].iloc[s_i[k]].tz )
    w_subset_df=w_intensity_df.loc[ (w_intensity_df['unix'] >= cyear_position_df['unix'].iloc[s_i[k]]) &
                                (w_intensity_df['unix'] <= cyear_position_df['unix'].iloc[e_i[k]]) ]
 #   b_subset_df=b_intensity_df.loc[ (b_intensity_df['unix'] >= cyear_position_df['unix'].iloc[s_i[k]]) &
 #                               (b_intensity_df['unix'] <= cyear_position_df['unix'].iloc[e_i[k]]) ]
    
    # Round down timestamps to nearest minute
    # --> we are averaging any observation over that 1 minute interval
    # --> necessary as Wind WAVES data has uneven ~3min3sec resolution with decimal seconds
    w_rounded_datetimes=np.empty(np.array(w_subset_df.datetime_ut).size,dtype=dt.datetime)
    w_decimal_hour=np.empty(np.array(w_subset_df.datetime_ut).size,dtype=float)
    b_rounded_datetimes=np.empty(np.array(w_subset_df.datetime_ut).size,dtype=dt.datetime)
    b_decimal_hour=np.empty(np.array(w_subset_df.datetime_ut).size,dtype=float)

    for p in range(w_rounded_datetimes.size):
        w_rounded_datetimes[p]=dt.datetime(w_subset_df['datetime_ut'].iloc[p].year, w_subset_df['datetime_ut'].iloc[p].month, 
                                         w_subset_df['datetime_ut'].iloc[p].day, w_subset_df['datetime_ut'].iloc[p].hour, 
                                         w_subset_df['datetime_ut'].iloc[p].minute, 0)
        w_decimal_hour[p]=w_rounded_datetimes[p].hour+( (w_rounded_datetimes[p].minute)/60.0)
  #      b_rounded_datetimes[p]=dt.datetime(b_subset_df['datetime_ut'].iloc[p].year, b_subset_df['datetime_ut'].iloc[p].month, 
  #                                       b_subset_df['datetime_ut'].iloc[p].day, b_subset_df['datetime_ut'].iloc[p].hour, 
  #                                       b_subset_df['datetime_ut'].iloc[p].minute, 0)
  #      b_decimal_hour[p]=b_rounded_datetimes[p].hour+( (b_rounded_datetimes[p].minute)/60.0)
 
    #w_unique_dh=np.sort(pd.Series(w_decimal_hour).unique())
    w_subset_df['rounded_datetimes']=w_rounded_datetimes
    w_subset_df['decimal_hour']=w_decimal_hour
  #  b_subset_df['rounded_datetimes']=b_rounded_datetimes
  #  b_subset_df['decimal_hour']=b_decimal_hour

    # Calculate spe trace for intensity
    n_intervals=24*60
    dhs=np.arange(n_intervals)*(1/60)

    sorted_intensity=np.empty(n_intervals)
    sorted_intensity[:]=np.nan
    threshold_intensity=np.empty(n_intervals)
    threshold_intensity[:]=np.nan
    n_ave=np.empty(n_intervals)
    n_ave[:]=np.nan
    
    threshold=0.02E7

    for j in range(n_intervals):
        # Check for empty df
        if np.where( ~np.isnan( w_subset_df['P_Wsr-1_100_650_kHz'].loc[w_subset_df.decimal_hour == dhs[j]] ) )[0].size > 0:            
            sorted_intensity[j]=np.nanmedian( w_subset_df['P_Wsr-1_100_650_kHz'].loc[w_subset_df.decimal_hour == dhs[j]] )
            #print(subset_df['integrated_intensity'].loc[subset_df.decimal_hour == dhs[j]])
            threshold_intensity[j]=np.nanmedian( w_subset_df['P_Wsr-1_100_650_kHz'].loc[(w_subset_df.decimal_hour == dhs[j]) & (w_subset_df['P_Wsr-1_100_650_kHz'] > threshold)] )
            n_ave[j]=np.array(w_subset_df['P_Wsr-1_100_650_kHz'].loc[w_subset_df.decimal_hour == dhs[j]]).size
            #return
        else:
            n_ave[j]=0
  
    
    fig,ax=plt.subplots()
    ax.plot(w_subset_df.decimal_hour,w_subset_df['P_Wsr-1_100_650_kHz'],marker='o',linewidth=0,fillstyle='none')
  #  ax.plot(b_subset_df.decimal_hour,b_subset_df['P_Wsr-1_100_650_kHz'],marker='x',linewidth=0,fillstyle='none')
    ax.plot(dhs,sorted_intensity)
    ax.plot(dhs,threshold_intensity)
    ax.set_ylabel('Intensity')
    ax.set_xlabel('UT (hours)')
    #ax.set_yscale('log')
    ax.set_title(str(sector_lower[k])+' - '+str(sector_upper[k])
                 +' LT\n '+str(w_subset_df['datetime_ut'].iloc[0].strftime('%Y-%m-%d %H:%M'))+' '
                 +' to '+str(w_subset_df['datetime_ut'].iloc[-1].strftime('%Y-%m-%d %H:%M')))



#        for i in range(n_lt_sectors):
#        #print('LT sector: ',i)
#        #print('lower bound: ',i-(lt_sector_width/2),i+(lt_sector_width/2))
#        
#        if i > 0:
#            lower_bound=(i*lt_sector_width)-(lt_sector_width/2)
#            upper_bound=(i*lt_sector_width)+(lt_sector_width/2)
#            # Select the dataframe rows within this local time sector
#            subset_df=intensity_summary_df.loc[ (intensity_summary_df.lt_gse >= lower_bound) & 
#                                               (intensity_summary_df.lt_gse < upper_bound) ]
#        else:   # Treat LT = 0 differently
#            lower_bound=(i*lt_sector_width)-(lt_sector_width/2)+24#(n_lt_sectors*lt_sector_width)+(lt_sector_width/2)-lt_sector_width
#            upper_bound=(i*lt_sector_width)+(lt_sector_width/2)
#            # Select the dataframe rows within this local time sector
#            subset_df=intensity_summary_df.loc[ (intensity_summary_df.lt_gse >= lower_bound) | 
#                                               (intensity_summary_df.lt_gse < upper_bound) ]
#
    return   
