# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:48:12 2020

@author: Barry
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset, num2date
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import matplotlib as mpl
# mpl.rcParams['mathtext.default'] = 'regular'
import numpy as np
import sys
import math
import statistics
import hydrostats as hs
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model
from fastdtw import fastdtw
from scipy.stats import gaussian_kde
import random
from lazypredict.Supervised import LazyRegressor

# def main():
# location = ''
# year = 0
# while (location != 'Dublin' and location != 'dublin'
#         and location != 'Singapore' and location != 'singapore'):
#     location = input('Enter Geographical Area (Dublin/Singapore): ')
#     if (location != 'Dublin' and location != 'dublin'
#         and location != 'Singapore' and location != 'singapore'):
#         print('Invalid location. Please enter Dublin or Singapore.')
# while year < 1979 or year > datetime.now().year - 1:
#     try:
#         year = math.floor(float(input('Enter Year: ')))
#     except ValueError:
#         year = 0
#     if year < 1979 or year > datetime.now().year - 1:
#         print('Invalid year. Please enter a year between 1979 and last year.')
# year = str(year)

# Previous year should have data for each day so, collect data from this year
# and up to four years earlier
last_year = datetime.now().year-1
# Years to collect data from
data_years = [last_year-4, last_year-3, last_year-2, last_year-1, last_year]
regions = ['Dublin', 'Singapore'] # Climate regions
# Longitude and latitude coordinates of regions for satellite data (N/W/S/E)
area = ['53.7/-6.5/53.2/-6','1.6/103.7/1.1/104.2']
all_models = [] # Initialise list of regression model results
all_metrics = [] # Initialise list of metrics
for i in range(len(regions)): # For each region
    year_sat = [] # Initialise list of satellite temperatures for each year
    year_land = [] # Initialise list of land temperatures for each year
    year_cloud = [] # Initialise list of satellite cloud cover for each year
    for j in range(len(data_years)): # For every year
        year = str(data_years[j])
        # Read land temperatures and dates from csv file
        df_land = pd.read_csv(regions[i] + '_land_temp_' + year + '.csv',
                              index_col = 0)
        
        # area = list(input('Enter Geographical Area (N/W/S/E): ').split('/'))
        # year = list(input('Enter Year(s) (yyyy): ').split())
        # month = list(input('Enter Month(s) (mm): ').split())
        # day = list(input('Enter Day(s) (dd): ').split())
        # time = list(input('Enter Time(s) (hh:mm): ').split())
        # variable = list(input('Enter Variable(s): ').split())
        
        # if abs(float(area[0])-float(area[2]))/abs(float(area[1])-float(area[3]))
        # < 2/9 or abs(float(area[0])-float(area[2]))/abs(float(area[1])-float(area[3]))
        # > 3/4:
        #     sys.exit('Error with Aspect Ratio')
        
        # area = '/'.join(area)
        
        # Read satellite data from netcdf file
        satfile = Dataset(regions[i] + '_sat_temp_' + year + '.nc', 'r')
        # Isolate all 2m temperature values
        all_t2m = satfile.variables['t2m'][:]
        # Isolate all total cloud cover values
        all_tcc = satfile.variables['tcc'][:]
        nctime = satfile.variables['time'][:] # Isolate all time values
        t_unit = satfile.variables['time'].units # Isolate time units
        t_cal = satfile.variables['time'].calendar # Isolate time calendar
        lats = satfile.variables['latitude'][:] # Isolate all latitude values
        # Isolate all longitude values
        lons = satfile.variables['longitude'][:]
        # Remove single dimensional entries from shape
        lats = lats[:].squeeze()
        # Remove single dimensional entries from shape
        lons = lons[:].squeeze()
        satfile.close() # Close file
        satdate = [] # Initialise satellite dates list
        # Convert time and add to list
        satdate.append(num2date(nctime, units = t_unit, calendar = t_cal))
        # Initialise satellite current date
        curr_date = satdate[0][0].strftime("%Y-%m-%d")
        # Initialise satellite current month
        curr_month = satdate[0][0].strftime("%m")
        sathourtemp = [] # Initialise satellite hourly temperature values
        sathourcloud = [] # Initialise satellite hourly cloud cover values
        satdaytemp = [] # Initialise satellite daily temperature values
        satdaycloud = [] # Initialise satellite daily cloud cover values
        satmonthtemp = [] # Initialise satellite monthly temperature values
        satmonthcloud = [] # Initialise satellite monthly cloud cover values
        landdaytemp = [] # Initialise land daily temperature values
        landmonthtemp = [] # Initialise land monthly temperature values
        tempdiff = [] # Initialise temperature difference
        
        # for i in range(len(all_t2m)):
        #     if (df_land.date == pd.Timestamp(satdate[0][i].strftime("%Y-%m-%d"))).any():
        #         curr_t2m = all_t2m[i,:,:]
        #         # curr_tcc = all_tcc[i,:,:]
        #         if curr_date != satdate[0][i].strftime("%Y-%m-%d"):
        #             satdaytemp.append(statistics.mean(sathourtemp)-273.15)
        #             # satdaycloud.append(statistics.mean(sathourcloud))
        #             index = df_land[df_land['date']==curr_date].index.values[0]
        #             landdaytemp.append(df_land['meanTemp'][index])
        #             sathourtemp = []
        #             # sathourcloud = []
        #             curr_date = satdate[0][i].strftime("%Y-%m-%d")
        #         if curr_month != satdate[0][i].strftime("%m"):
        #             landmonthtemp.append(landdaytemp)
        #             landdaytemp = []
        #             satmonthtemp.append(satdaytemp)
        #             # satmonthcloud.append(satdaycloud)
        #             satdaytemp = []
        #             # satdaycloud = []
        #             curr_month = satdate[0][i].strftime("%m")
        #         sathourtemp.append(curr_t2m[2][2])
        #         # sathourcloud.append(curr_tcc[2][2])
        #         if i == len(all_t2m)-1:
        #             satdaytemp.append(statistics.mean(sathourtemp)-273.15)
        #             # satdaycloud.append(statistics.mean(sathourcloud))
        #             index = df_land[df_land['date']==curr_date].index.values[0]
        #             landdaytemp.append(df_land['meanTemp'][index])
        #             landmonthtemp.append(landdaytemp)
        #             satmonthtemp.append(satdaytemp)
        #             # satmonthcloud.append(satdaycloud)
        
        for k in range(len(all_t2m)):
            # 2m temperature for hour of current date
            curr_t2m = all_t2m[k,:,:]
            # Total cloud cover for hour of current date
            curr_tcc = all_tcc[k,:,:]
            # Moved onto next date
            if curr_date != satdate[0][k].strftime("%Y-%m-%d"):
                # Land value available for date
                if (df_land.date == curr_date).any():
                    # Get mean, covert to celsius and update satellite daily
                    # temperature
                    satdaytemp.append(statistics.mean(sathourtemp) - 273.15)
                    # Get mean and update satellite daily cloud cover
                    satdaycloud.append(statistics.mean(sathourcloud))
                    # Find value for date
                    index = df_land[df_land['date']
                                    == curr_date].index.values[0]
                    # Update land daily temperatures
                    landdaytemp.append(df_land['meanTemp'][index])
                sathourtemp = [] # Reinitialise
                sathourcloud = [] # Reinitialise
                # Update current date
                curr_date = satdate[0][k].strftime("%Y-%m-%d")
            # Moved onto next month
            if curr_month != satdate[0][k].strftime("%m"):
                # Land values available for month
                if (pd.DatetimeIndex(df_land['date']).month
                    == int(curr_month)).any():
                    # Update land monthly temperatures
                    landmonthtemp.append(landdaytemp)
                    # Update satellite monthly temperatures
                    satmonthtemp.append(satdaytemp)
                    # Update satellite monthly cloud cover
                    satmonthcloud.append(satdaycloud)
                    # Calculate temperature difference and append
                    tempdiff.append(list(np.array(satdaytemp)
                                         - np.array(landdaytemp)))
                landdaytemp = [] # Reinitialise
                satdaytemp = [] # Reinitialise
                satdaycloud = [] # Reinitialise
                # Update current month
                curr_month = satdate[0][k].strftime("%m")
            # Update satellite houly temperature
            sathourtemp.append(curr_t2m[2][2])
            # Update satellite houly cloud cover
            sathourcloud.append(curr_tcc[2][2])
            if k == len(all_t2m) - 1: # Reached last value
                # Land value available for date
                if (df_land.date == curr_date).any():
                    # Get mean, covert to celsius and update satellite daily
                    # temperature
                    satdaytemp.append(statistics.mean(sathourtemp) - 273.15)
                    # Get mean and update satellite daily cloud cover
                    satdaycloud.append(statistics.mean(sathourcloud))
                    # Find value for date
                    index = df_land[df_land['date']
                                    == curr_date].index.values[0]
                    # Update land daily temperatures
                    landdaytemp.append(df_land['meanTemp'][index])
                # land values available for month
                if (pd.DatetimeIndex(df_land['date']).month
                    == int(curr_month)).any():
                    # Update land monthly temperatures
                    landmonthtemp.append(landdaytemp)
                    # Update satellite monthly temperatures
                    satmonthtemp.append(satdaytemp)
                    # Update satellite monthly cloud cover
                    satmonthcloud.append(satdaycloud)
                    # Calculate temperature difference and append
                    tempdiff.append(list(np.array(satdaytemp)
                                         - np.array(landdaytemp)))

        plt.figure()
        plt.boxplot(tempdiff) # Create boxplots
        plt.xlabel('time (months)')
        plt.ylabel('temperature (degrees celcius)')
        plt.title(regions[i] + ' Mean Satellite and Land Temperature'
                  + ' Difference ' + year)
        
        # m_avg_cloud = [] # Initialise monthly mean cloud cover
        # m_avg_land = [] # Initialise monthly mean land temperature
        # m_avg_sat = [] # Initialise monthly mean satellite temperature
        # for i in range(len(landmonthtemp)):
        #     mean_land = statistics.mean(landmonthtemp[i]) # Land temperature mean of current month
        #     m_avg_land.append(mean_land) # Add to list
        #     mean_sat = statistics.mean(satmonthtemp[i]) # Satellite temperature mean of current month
        #     m_avg_sat.append(mean_sat) # Add to list
        #     mean_cloud = statistics.mean(satmonthcloud[i]) # Satellite cloud cover mean of current month
        #     m_avg_cloud.append(mean_cloud) # Add to list
        # year_cloud.append(m_avg_cloud)
        # year_sat.append(m_avg_sat)
        # year_land.append(m_avg_land)
        
        # Make 1D list and append
        year_sat.append([val for sublist in satmonthtemp for val in sublist])
        # Make 1D list and append
        year_land.append([val for sublist in landmonthtemp for val in sublist])
        
# total_cloud = []
# total_sat = []
# total_land = []
# for i in range(12):
#     meanc = (year_cloud[0][i]+year_cloud[1][i]+year_cloud[2][i]
#     +year_cloud[3][i]+year_cloud[4][i])/5
#     total_cloud.append(meanc)
#     means = (year_sat[0][i]+year_sat[1][i]+year_sat[2][i]
#     +year_sat[3][i]+year_sat[4][i])/5
#     total_sat.append(means)
#     meanl = (year_land[0][i]+year_land[1][i]+year_land[2][i]
#     +year_land[3][i]+year_land[4][i])/5
#     total_land.append(meanl)

    # extra = 0
    # lats = lats1[0:len(lats1)-1:math.ceil(len(lats1)*6/100)]
    # lats = np.append(lats,lats1[720])
    # lons = lons[0:len(lons1)-1:math.ceil(len(lons1)*6/100)]
    # lons = np.append(lons,180)#lons1[1439])
    
    # all_t2m1 = file.variables['u'][:]
    # all_tcc = file.variables['v'][:]
    
    curr_t2m = all_t2m[0,:,:] # 2m temperature for hour of current date
    
    # new = new + curr_t2m[2][2]
    #pp1 = all_t2m1[i,j,:,:]
    # pp_1 = np.append(pp1,pp1[:,-1:],axis=1)
    #pp1 = pp1[0:len(lats1):math.ceil(len(lats1)*6/100)
    #          ,0:len(lons1):math.ceil(len(lons1)*6/100)]
    #pp1 = np.v_stack(pp_1,pp1[720,:])
    # curr_tcc = all_tcc[i,j,:,:]
    # pp_1 = pp1[0:len(lats1)-1:math.ceil(len(lats1)*6/100)
    #           ,0:len(lons1)-1:math.ceil(len(lons1)*6/100)]
    # pp_2 = pp2[0:len(lats1)-1:math.ceil(len(lats1)*6/100)
    #           ,0:len(lons1)-1:math.ceil(len(lons1)*6/100)]
    # if lons[-1] != E:
    #     extra = 1
    #     lons = np.append(lons,E)
    #     col = pp1[0:len(lats1)-1:math.ceil(len(lats1)*6/100),len(lons1)-1]
    #     pp_1 = np.hstack((pp_1, np.atleast_2d(col).T))
    #     col = pp2[0:len(lats1)-1:math.ceil(len(lats1)*6/100),len(lons1)-1]
    #     pp_2 = np.hstack((pp_2, np.atleast_2d(col).T))
    # if lats[-1] != S:
    #     lats = np.append(lats,S)
    #     row = pp1[len(lats1)-1,0:len(lons1)-1:math.ceil(len(lons1)*6/100)]
    #     if extra == 1:
    #         row = np.append(row,pp1[len(lats1)-1,len(lons1)-1])
    #     #pp_1 = np.hstack((pp_1, np.atleast_2d(row).T))
    #     pp_1 = np.vstack((pp_1, row.reshape([1,len(row)])))
    #     row = pp2[len(lats1)-1,0:len(lons1)-1:math.ceil(len(lons1)*6/100)]
    #     if extra == 1:
    #         row = np.append(row,pp2[len(lats1)-1,len(lons1)-1])
    #     #pp_2 = np.hstack((pp_2, np.atleast_2d(row).T))
    #     pp_2 = np.vstack((pp_2, row.reshape([1,len(row)])))
    # pp1 = pp_1
    # pp2 = pp_2
    # pp_2 = np.append(pp2,pp2[:,-1:],axis=1)
    #pp1 = pp1[0:len(lats1):math.ceil((len(lats1))*6/100)
    #          ,0:len(lons1)+1+math.ceil((len(lons1)+1)*6/100):math.ceil((len(lons1)+1)*6/100)]
    #pp2 = pp2[0:len(lats1):math.ceil((len(lats1))*6/100)
    #          ,0:len(lons1)+1+math.ceil((len(lons1)+1)*6/100):math.ceil((len(lons1)+1)*6/100)]
    # pp_2 = np.column_stack(pp_2,pp2[:,1439])
    # pp2 = np.v_stack(pp_2,pp2[720,:])
    # curr_t2m = np.sqrt(np.square(pp1)+np.square(pp2))
    
    plt.figure()
    # Create axes and set projection
    ax1 = plt.axes(projection = ccrs.PlateCarree())
    
    #clevs = np.arange(min(all_t2m.flatten()),max(all_t2m.flatten())*1000,1)
    
    # Range of colour bar values
    clevs = np.arange(min(curr_t2m.flatten()), max(curr_t2m.flatten()), 1)
    
    #shear_fill = ax1.contourf(lons,lats,all_t2m*1000,clevs,
    #                          transform=ccrs.PlateCarree(),
    #                          cmap=plt.get_cmap('hsv'),linewidth=(10,),
    #                          levels=100,extend='both')
    
    # Fill spaces between values for smooth colour map
    shear_fill = ax1.contourf(lons, lats, curr_t2m - 273.15, clevs - 273.15,
                              transform = ccrs.PlateCarree(),
                              cmap = plt.get_cmap('hsv'),#linewidth=(10,),
                              levels = 100, extend = 'both')
    ax1.coastlines() # Include the world's coastlines
    ax1.gridlines() # Include gridlines
    # Isolate N/W/S/E coordinates
    coords = list(area[i].split('/'))
    # Range of values for xticks
    xrange = np.arange(float(coords[1]), float(coords[3])
                       + abs(float(coords[1])-float(coords[3]))/2,
                       abs(float(coords[1])-float(coords[3]))/2)
    # Range of values for yticks
    yrange = np.arange(float(coords[2]), float(coords[0])
                       + abs(float(coords[0])-float(coords[2]))/4,
                       abs(float(coords[0])-float(coords[2]))/4)
    ax1.set_xticks(xrange, crs = ccrs.PlateCarree())
    #ax1.set_yticks(yrange[0:7], crs=ccrs.PlateCarree())
    ax1.set_yticks(yrange, crs = ccrs.PlateCarree())
    # Include direction of longitude (W/E)
    lon_formatter = LongitudeFormatter(dateline_direction_label = True,
                                        number_format = '.1f')
    # Include direction of latitude (N/S)
    lat_formatter = LatitudeFormatter(number_format = '.1f')
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    # ax1.quiver(lons,lats,pp1,pp2,transform=ccrs.PlateCarree())
    cbar = plt.colorbar(shear_fill, orientation = 'horizontal',
                        format = '%.1f',)
    #cbar.set_ticks(np.arange(min(all_t2m.flatten()),max(all_t2m.flatten())*1000,
    #                         (max(all_t2m.flatten())*1000-min(all_t2m.flatten()))/10))
    cbar.ax.set_title('2m Temperature ($^\circ$C)')
    plt.title(regions[i] + ' ' + satdate[0][0].strftime("%H:%M") + ' '
              + satdate[0][0].strftime("%Y-%m-%d"))

# time = np.arange(1, len(landmonthtemp) + 1) # Number of months
# # fig, ax = plt.subplots()
# # ax.bar(time-0.125,total_land,width=0.25,label='land temp')
# # ax.bar(time+0.125,total_sat,width=0.25,label='satellite temp')
# # ax.set_xticks(np.arange(1, 13, 1))
# # ax.set_xlabel('time (months)')
# # # naming the y axis 
# # ax.set_ylabel('temperature (degrees celsius)')
# # ax.set_ylim([0, 50])
# # # giving a title to my graph 
# # ax.set_title('Singapore Mean Monthly Temperature & Cloud Cover 2015-2019') 
# # ax.legend(loc='upper left')
# # ax2 = ax.twinx()
# # ax2.plot(time, total_cloud, '-ok', label = "satellite cloud cover")
# # ax2.set_ylabel('cloud cover (0-1)')
# # ax2.set_ylim([0, 1.2])
# # # show a legend on the plot 
# # ax2.legend(loc='upper right') 

# fig, ax = plt.subplots()
# # Place bar plots side by side for each month
# ax.bar(time - 0.125, m_avg_land, width=0.25, label = 'land temp')
# ax.bar(time + 0.125, m_avg_sat, width=0.25, label = 'satellite temp')
# ax.set_xticks(np.arange(1, 13, 1))
# ax.set_xlabel('time (months)')
# ax.set_ylabel('temperature (degrees celsius)')
# # ax.set_ylim([0, 25])
# ax.set_ylim([0, 55])
# ax.set_title('Singapore Mean Monthly Temperature & Cloud Cover ' + year) 
# ax.legend(loc='upper left') 
# ax2 = ax.twinx() # Create secondary axis
# ax2.plot(time, m_avg_cloud, '-ok', label = "satellite cloud cover")
# ax2.set_ylabel('cloud cover (0-1)')
# # ax2.set_ylim([0, 1])
# ax2.set_ylim([0, 1.2])
# ax2.legend(loc = 'upper right') 

# time = np.arange(1, len(landmonthtemp[0]) + 1) # Number of days in first month
# plt.figure()
# plt.plot(time, landmonthtemp[0], '-ok', color='blue', label = "land")
# plt.plot(time, satmonthtemp[0], '-ok', color='orange', label = "satellite")
# plt.xlabel('time (days)')
# plt.ylabel('temperature (degrees celsius)') 
# plt.title('Singapore Mean Daily Temperature January ' + year) 
# plt.legend()

# all_indices = np.arange(0, len(all_land)) # Possible indices to select
# # Random 70 % of indices
# indices_70 = random.sample(range(0, len(all_land)),
#                            math.floor(len(all_land) * 0.7))
# indices_30 = [] # Initialise list
# for i in range(len(all_indices)):
#     # Search through indices
#     if all_indices[i] not in indices_70:
#         indices_30.append(all_indices[i]) # Remaining 30 % of indices
# indices_30 = random.sample(indices_30, len(indices_30)) # Randomise indices
# land_70 = [] # Initialise list
# land_30 = [] # Initialise list
# sat_70 = [] # Initialise list
# sat_30 = [] # Initialise list
# for i in range(len(indices_70)):
#     land_70.append(all_land[indices_70[i]]) # Append corresponding land values
#     sat_70.append(all_sat[indices_70[i]]) # Append corresponding satellite values
# for i in range(len(indices_30)):
#     land_30.append(all_land[indices_30[i]]) # Append corresponding land values
#     sat_30.append(all_sat[indices_30[i]]) # Append corresponding satellite values
# sat_30 = np.array(sat_30) # Make numpy array
# sat_70 = np.array(sat_70) # Make numpy array
# land_30 = np.array(land_30) # Make numpy array
# land_70 = np.array(land_70) # Make numpy array
# # Create Instance
# reg = LazyRegressor(verbose = 0, ignore_warnings = False, custom_metric = None)
# # Create models and predictions
# models, predictions = reg.fit(land_70, land_30, sat_70, sat_30)
    
    model_list = [] # Initialise list of model results for region
    for j in range(len(year_sat)): # For all years
        sat_train = [] # Initialise list of four years of satellite values
        sat_test = [] # Initialise list of one year of satellite values
        land_train = [] # Initialise list of four years of land values
        land_test = [] # Initialise list of one years of land values
        for k in range(len(year_sat)-1): # For four of the years
            # Join satellite values of years together
            sat_train = sat_train + year_sat[j-(len(year_sat)-1-k)]
            # Join land values of years together
            land_train = land_train + year_land[j-(len(year_land)-1-k)]
        sat_train = np.array(sat_train) # Make numpy array
        land_train = np.array(land_train) # Make numpy array
        #sat_train = np.array(year_sat[i-4] + year_sat[i-3] + year_sat[i-2] + year_sat[i-1])
        sat_test = np.array(year_sat[j]) # Choose year that was not joined
        #land_train = np.array(year_land[i-4] + year_land[i-3] + year_land[i-2] + year_land[i-1])
        land_test = np.array(year_land[j]) # Choose year that was not joined
        # Initialise regression instance
        reg = LazyRegressor(verbose = 0, ignore_warnings = False,
                            custom_metric = None)
        # Create models and predictions
        models, predictions = reg.fit(land_train, land_test, sat_train,
                                      sat_test)
        # Append results for current train and test
        model_list.append(models)
    # Append results for all trains and tests
    all_models.append(model_list)
    
    all_sat = [] # Initialise list of satellite values over 5 years
    all_land = []# Initialise list of land values over 5 years
    for j in range(len(year_sat)): # For each year
        # Join satellite values of years together
        all_sat = all_sat + year_sat[j]
        # Join land values of years together
        all_land = all_land + year_land[j]
    kge = hs.kge_2012(all_land, all_sat) # Calculate KGE
    # Store metric in dataframe
    df_metrics = pd.DataFrame({"type":["Kling-Gupta Efficiency"],
                              "value":['%.4f' % kge]})
    r2 = r2_score(all_land, all_sat) # Calculate r^2 value
    # Store metric in dataframe
    df = pd.DataFrame({"type":["R-Squared"], "value":['%.4f' % r2]})
    # Keep metrics together
    df_metrics = df_metrics.append(df)
    distance, path = fastdtw(all_land, all_sat) # Calculate DTW distance
    # Store metric in dataframe
    df = pd.DataFrame({"type":["Dynamic Time Warping Distance"],
                       "value":['%.4f' % distance]})
    # Keep metrics together
    df_metrics = df_metrics.append(df)
    regr = linear_model.LinearRegression() # Create linear regression instance
    # Arrange into dataframe
    df_line = pd.DataFrame({'x':all_land, 'y':all_sat})
    land_regr = df_line.x.values.reshape(-1, 1) # Shape array
    sat_regr = df_line.y.values.reshape(-1, 1) # Shape array
    regr.fit(land_regr, sat_regr) # Fit model
    # Store metric in dataframe
    df = pd.DataFrame({"type":["Best Fit Line Equation"],
                       "value":['y = ' + '%.4f' % regr.coef_[0][0]
                                + 'x + ' + '%.4f' % regr.intercept_[0]]})
    # Keep metrics together
    df_metrics = df_metrics.append(df)
    # Calculate MBE
    mbe = statistics.mean(np.array(all_sat) - np.array(all_land))
    # Store metric in dataframe
    df = pd.DataFrame({"type":["Mean Bias Error"], "value":['%.4f' % mbe]})
    # Keep metrics together
    df_metrics = df_metrics.append(df)
    rmse = math.sqrt(mean_squared_error(all_land, all_sat)) # Calculate RMSE
    # Store metric in dataframe
    df = pd.DataFrame({"type":["Root Mean Square Error"],
                       "value":['%.4f' % rmse]})
    df_metrics = df_metrics.append(df) # Keep metrics together
    all_metrics.append(df_metrics) # Append all metrics to list
    all_land = np.array(all_land) # Make numpy array
    all_sat = np.array(all_sat) # Make numpy array
    landsat = np.vstack([all_land, all_sat]) # Stack arrays
    z = gaussian_kde(landsat)(landsat) # Density estimate
    # Sort the points by density so that the densest points are plotted last
    idx = z.argsort()
    all_land, all_sat, z = all_land[idx], all_sat[idx], z[idx]
    fig, ax = plt.subplots()
    ax.scatter(all_land, all_sat, c = z) # Plot values
    # Plot line of best fit
    ax.plot(all_land, regr.coef_[0][0]*all_land + regr.intercept_[0],
            color='black')
    ax.set_title(regions[i] + ' Satellite and Land Data Relationship ' + year)
    ax.set_xlabel('Land Temperature (degrees celcius)')
    ax.set_ylabel('Satellite Temperature (degrees celcius)')
    
    # ax.set_xlim([min(all_land)-3, max(all_land)+3])
    # ax.set_ylim([min(all_sat)-3, max(all_sat)+3])
    # ax.text(min(all_land)-2.5, max(all_sat)+2, '$R^2$ = ' + '%.4f' % r2)
    # ax.text(min(all_land)-2.5, max(all_sat)+1, 'KGE = ' + '%.4f' % kge)
    # ax.text(22.5, 30.5,'MBE = ' + '%.4f' % mbe)
    # ax.text(22.5, 30,'RMSE = ' + '%.4f' % rmse)
    # ax.text(22.5, 29.5,'DTW Distance = ' + '%.4f' % distance)
    # ax.text(28.5, 22.5,'y = ' + '%.3f' % regr.coef_[0][0] + 'x + ' +
    #         '%.3f' % regr.intercept_[0])
    # ax.set_xlim([-5, 25])
    # ax.set_ylim([-5, 25])
    # ax.text(-4, 23,'$R^2$ = ' + '%.4f' % r2)
    # ax.text(-4, 21,'KGE = ' + '%.4f' % kge)
    # ax.text(-4, 19,'MBE = ' + '%.4f' % mbe)
    # ax.text(-4, 17,'RMSE = ' + '%.4f' % rmse)
    # ax.text(-4, 15,'DTW Distance = ' + '%.4f' % distance)
    # ax.text(15, -4,'y = ' + '%.3f' % regr.coef_[0][0] + 'x + ' +
    #         '%.3f' % regr.intercept_[0])

# if __name__ == '__main__':
#     main()