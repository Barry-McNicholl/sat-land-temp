# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:41:18 2020

@author: Barry
"""

import json
import cdsapi
import requests
import pandas as pd
from datetime import datetime

# Previous year should have data for each day so, collect data from this year
# and up to four years earlier
last_year = datetime.now().year-1
# Years to collect data from
data_years = [last_year-4, last_year-3, last_year-2, last_year-1, last_year]
regions = ['Dublin', 'Singapore'] # Climate regions
# Dublin and Singapore land stations
stations = ['GHCND:EI000003969', 'GHCND:SNM00048698']
token = 'wYUaHbiJsbyWymfxUAYnkqYHqaEyjNkG' # NCDC web service key token
# Longitude and latitude coordinates of regions for satellite data (N/W/S/E)
area = ['53.7/-6.5/53.2/-6','1.6/103.7/1.1/104.2']
for i in range(len(regions)): # For each region
    for j in range(len(data_years)): # For every year
        land_dates = [] # Initialise land date fields
        land_temps = [] # Initialise land mean temperatures
        year = str(data_years[j])
        print('Getting Daily Land Temperatures in ' + regions[i]
              + ' for ' + year)
        # Make the api call for land data
        land_request = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2'
                                    + '/data?datasetid=GHCND&datatypeid=TAVG&'
                                    + 'limit=1000&stationid=' + stations[i]
                                    + '&startdate=' + year + '-01-01'
                                    + '&enddate=' + year + '-12-31',
                                    headers = {'token':token})
        # Load the api response as a json
        land_response = json.loads(land_request.text)
        # Get all items in the response which are mean temperature readings
        mean_temps = [item for item in land_response['results']
                      if item['datatype'] == 'TAVG']
        # Get the date field from all mean temperature readings
        land_dates += [item['date'] for item in mean_temps]
        # Get the actual mean temperature from all mean temperature readings
        land_temps += [item['value'] for item in mean_temps]
        df_land = pd.DataFrame() # Initialize dataframe
        # Store dates in dataframe
        df_land['date'] = [datetime.strptime(land_response,"%Y-%m-%dT%H:%M:%S")
                           for land_response in land_dates]
        # Convert mean temperatures from tenths of celsius to celsius and
        # store in dataframe
        df_land['meanTemp'] = [float(value)/10 for value in land_temps]
        # Store land data in csv file
        df_land.to_csv(regions[i] + '_land_temp_' + year + '.csv')
        print('Downloaded Daily Land Temperatures in ' + regions[i]
              + ' for ' + year)
        sat_client = cdsapi.Client() # Open a new Client instance
        sat_request = sat_client.retrieve( # Make the satellite data api call
            'reanalysis-era5-single-levels', # ERA5 data set
            {
                 # Air temperature at 2m above the surface
                'variable':'2m_temperature',
                'product_type':'reanalysis',
                'year':year,
                'month':['01','02','03','04','05','06','07','08','09','10',
                         '11','12'],
                'day':['01','02','03','04','05','06','07','08','09','10','11',
                       '12','13','14','15','16','17','18','19','20','21','22',
                       '23','24','25','26','27','28','29','30','31'],
                'time':['00:00','01:00','02:00','03:00','04:00','05:00',
                        '06:00','07:00','08:00','09:00','10:00','11:00',
                        '12:00','13:00','14:00','15:00','16:00','17:00',
                        '18:00','19:00','20:00','21:00','22:00','23:00'],
                'area':area[i],
                'format':'netcdf'
            })
        # Download request and store satellite data in netcdf file
        sat_request.download(regions[i] + '_sat_temp_' + year + '.nc')