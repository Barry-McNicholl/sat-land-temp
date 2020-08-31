import math
import statistics
import numpy as np
import pandas as pd
import hydrostats as hs
import cartopy.crs as ccrs
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import gaussian_kde
from netCDF4 import Dataset, num2date
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import r2_score, mean_squared_error
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def main():
    # Initialise years to collect data from
    data_year = []
    while len(data_year) != 5: # Until the user inputs 5 strings
        # Ask user to enter years to collect data from
        data_year = input('Enter the 5 years the data is from\n(least to most'
                          + ' recent, separated by a space): ').split()
        if len(data_year) != 5:
            print("Error: 5 strings must be entered")
    regions = ['Dublin', 'Singapore'] # Climate regions
    # Longitude and latitude coordinates of regions for satellite data
    # (N/W/S/E)
    area = ['53.7/-6.5/53.2/-6', '1.6/103.7/1.1/104.2']
    all_models = [] # Initialise list of regression model results
    all_metrics = [] # Initialise list of metrics
    for i in range(len(regions)): # For each region
        # Initialise list of satellite temperatures for each year
        year_sat = []
        # Initialise list of land temperatures for each year
        year_land = []
        for j in range(len(data_year)): # For every year
            year = data_year[j]
            # Read land temperatures and dates from csv file
            df_land = pd.read_csv(regions[i] + '_land_temp_' + year + '.csv',
                                  index_col = 0)        
            # Read satellite data from netcdf file
            satfile = Dataset(regions[i] + '_sat_temp_' + year + '.nc', 'r')
            # Isolate all 2m temperature values
            all_t2m = satfile.variables['t2m'][:]
            nctime = satfile.variables['time'][:] # Isolate all time values
            t_unit = satfile.variables['time'].units # Isolate time units
            t_cal = satfile.variables['time'].calendar # Isolate time calendar
            # Isolate all latitude values
            lats = satfile.variables['latitude'][:]
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
            satdaytemp = [] # Initialise satellite daily temperature values
            # Initialise satellite monthly temperature values
            satmonthtemp = []
            landdaytemp = [] # Initialise land daily temperature values
            landmonthtemp = [] # Initialise land monthly temperature values
            tempdiff = [] # Initialise temperature difference
            # Look through all satellite values, can not assume that there are
            # values for each date
            for k in range(len(all_t2m)):
                # Check if land value available for date, only consider dates
                # where land and satellite values are available
                if (df_land.date == satdate[0][k].strftime("%Y-%m-%d")).any():
                    # Moved onto next day or reached last value
                    if (curr_date != satdate[0][k].strftime("%Y-%m-%d") or
                        k == len(all_t2m)-1):
                        # If last value is on the current date
                        if (curr_date == satdate[0][k].strftime("%Y-%m-%d")
                            and k == len(all_t2m)-1):
                            # 2m temperature for hour of current date
                            curr_t2m = all_t2m[k,:,:]
                            # Update satellite hourly temperature
                            sathourtemp.append(curr_t2m[2][2])
                        # Ensure that the list is not empty
                        if sathourtemp != []:
                            # Get mean, convert to celsius and update
                            # satellite daily temperature
                            satdaytemp.append(statistics.mean(sathourtemp)
                                              -273.15)
                            # Find land value for date
                            index = df_land[df_land['date']
                                            == curr_date].index.values[0]
                            # Update land daily temperatures
                            landdaytemp.append(df_land['meanTemp'][index])
                            sathourtemp = [] # Reinitialise
                        # If last value is not on the current date
                        if (curr_date != satdate[0][k].strftime("%Y-%m-%d")
                            and k == len(all_t2m)-1):
                            # 2m temperature for hour of current date
                            curr_t2m = all_t2m[k,:,:]
                            # Convert to celsius and update satellite daily
                            # temperature
                            satdaytemp.append(curr_t2m[2][2]-273.15)
                            # Find land value for date
                            index = df_land[df_land['date']
                                            == curr_date].index.values[0]
                            # Update land daily temperatures
                            landdaytemp.append(df_land['meanTemp'][index])
                            sathourtemp = [] # Reinitialise
                        # Update current date
                        curr_date = satdate[0][k].strftime("%Y-%m-%d")
                    # Moved onto next month or reached last value
                    if (curr_month != satdate[0][k].strftime("%m") or
                        k == len(all_t2m)-1):
                        # Ensure that the list is not empty
                        if satdaytemp != []:
                            # Update land monthly temperatures
                            landmonthtemp.append(landdaytemp)
                            # Update satellite monthly temperatures
                            satmonthtemp.append(satdaytemp)
                            # Calculate temperature difference and append
                            tempdiff.append(list(np.array(satdaytemp)
                                                  - np.array(landdaytemp)))
                            landdaytemp = [] # Reinitialise
                            satdaytemp = [] # Reinitialise
                        # Update current month
                        curr_month = satdate[0][k].strftime("%m")
                    if k < len(all_t2m)-1:
                        # 2m temperature for hour of current date
                        curr_t2m = all_t2m[k,:,:]
                        # Update satellite hourly temperature
                        sathourtemp.append(curr_t2m[2][2])
            plt.figure()
            plt.boxplot(tempdiff) # Create boxplots
            plt.xlabel('time (months)')
            plt.ylabel('temperature (degrees celcius)')
            plt.title(regions[i] + ' Mean Satellite and Land Temperature'
                      + ' Difference ' + year)
            # Make 1D list and append
            year_sat.append([val for sublist in satmonthtemp for val
                             in sublist])
            # Make 1D list and append
            year_land.append([val for sublist in landmonthtemp for val
                              in sublist])
        # 2m temperature for hour of current date, use for visualisation
        curr_t2m = all_t2m[0,:,:]
        plt.figure()
        # Create axes and set projection
        ax1 = plt.axes(projection = ccrs.PlateCarree())     
        # Range of colour bar values
        clevs = np.arange(min(curr_t2m.flatten()), max(curr_t2m.flatten()), 1)
        # Fill spaces between values for smooth colour map
        shear_fill = ax1.contourf(lons, lats, curr_t2m-273.15, clevs-273.15,
                                  transform = ccrs.PlateCarree(),
                                  cmap = plt.get_cmap('hsv'),
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
        ax1.set_yticks(yrange, crs = ccrs.PlateCarree())
        # Include direction of longitude (W/E)
        lon_formatter = LongitudeFormatter(dateline_direction_label = True,
                                            number_format = '.1f')
        # Include direction of latitude (N/S)
        lat_formatter = LatitudeFormatter(number_format = '.1f')
        ax1.xaxis.set_major_formatter(lon_formatter)
        ax1.yaxis.set_major_formatter(lat_formatter)
        cbar = plt.colorbar(shear_fill, orientation = 'horizontal',
                            format = '%.1f',)
        cbar.ax.set_title('2m Temperature ($^\circ$C)')
        plt.title(regions[i] + ' ' + satdate[0][0].strftime("%H:%M") + ' '
                  + satdate[0][0].strftime("%Y-%m-%d"))
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
            sat_test = np.array(year_sat[j]) # Choose year that was not joined
            # Choose year that was not joined
            land_test = np.array(year_land[j])
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
        all_land = [] # Initialise list of land values over 5 years
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
        # Create linear regression instance
        regr = linear_model.LinearRegression()
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
        # Calculate RMSE
        rmse = math.sqrt(mean_squared_error(all_land, all_sat))
        # Store metric in dataframe
        df = pd.DataFrame({"type":["Root Mean Square Error"],
                           "value":['%.4f' % rmse]})
        df_metrics = df_metrics.append(df) # Keep metrics together
        all_metrics.append(df_metrics) # Append all metrics to list
        all_land = np.array(all_land) # Make numpy array
        all_sat = np.array(all_sat) # Make numpy array
        landsat = np.vstack([all_land, all_sat]) # Stack arrays
        z = gaussian_kde(landsat)(landsat) # Density estimate
        # Sort the points so that the densest points are plotted last
        idx = z.argsort()
        all_land, all_sat, z = all_land[idx], all_sat[idx], z[idx]
        fig, ax = plt.subplots()
        ax.scatter(all_land, all_sat, c = z) # Plot values
        # Plot line of best fit
        ax.plot(all_land, regr.coef_[0][0]*all_land + regr.intercept_[0],
                color='black')
        ax.set_title(regions[i] + ' Satellite and Land Data Relationship '
                     + str(min(list(map(int, data_year)))) + '-'
                     + str(max(list(map(int, data_year)))))
        ax.set_xlabel('Land Temperature (degrees celcius)')
        ax.set_ylabel('Satellite Temperature (degrees celcius)')
    return all_models, all_metrics

if __name__ == '__main__':
    all_models, all_metrics = main()