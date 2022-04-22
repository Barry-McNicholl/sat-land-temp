# Comparison of Satellite and Land Air Temperature Data Sets for Different Climate Regions

Contained within this repoistory is the code required to reproduce the results from the paper:

> B. McNicholl, Y. H. Lee, A. Campbell, S. Dev, Analyzing Air Temperature from ERA5 Reanalysis Data, *IEEE Geoscience and Remote Sensing Letters*, 2021.

Please cite this paper if you wish to use some or all of the code. This code is for academic and research purposes only.

![Sample Regressions](https://user-images.githubusercontent.com/65912701/90794167-5c87cf00-e304-11ea-8f3b-7c1218e18c92.PNG)

*Linear Regressions of Satellite and Land Temperatures for Dublin and Singapore 2015-2019.*

## Code Organization

`python3` was used to write the code.

### Dependencies
 
The following libraries should be installed before executing the codes.

+ cdsapi 0.3.0: `pip install cdsapi`
+ requests 2.24.0: `pip install requests`
+ pandas 1.1.0: `pip install pandas`
+ matplotlib 3.3.0: `pip install matplotlib`
+ Cartopy 0.18.0: `pip install Cartopy`
+ netCDF4 1.5.4: `pip install netCDF4`
+ numpy 1.19.1: `pip install numpy`
+ hydrostats 0.78: `pip install hydrostats`
+ scikit-learn 0.23.2: `pip install scikit-learn`
+ fastdtw 0.3.4: `pip install fastdtw`
+ scipy 1.5.2: `pip install scipy`
+ lazypredict 0.2.7: `pip install lazypredict`

### Data

The source of land temperature data in this work is the Global Historical Climatology Network provided by the National Ocenanic and Atmospheric Administration ([NOAA GHCN](https://www.ncdc.noaa.gov/data-access/land-based-station-data/land-based-datasets/global-historical-climatology-network-ghcn)). The weather variable collected from the GHCN is TAVG which is defined as the average temperature for a given day, measured from 00:00 on that day to 00:00 on the following day. The satellite temperature data was sourced from the Climate Data Store provided by the European Centre for Medium-Range Weather Forecasts ([ECMWF CDS](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)). The weather variable used from the CDS is 2m temperature which is temperature measured at 2m above the surface for a given hour of a day. Data can be downloaded from these sources from scratch by following the steps outlined for the `sat_land_data.py` script in the next section. Alternatively, the processed data used in this paper is available at this [Google Drive link](https://drive.google.com/drive/folders/1N68hx--Kyj9jFi0XnZihbZUzNMBxdY0i?usp=sharing). To compare land and satellite data, the satellite data is averaged over the course of a day. Data for a given day is only utilised if both satellite and land data are available for that day. It was not possible to obtain satellite data from the exact same locations as the land data so, satellite data was instead obtained from the closest locations possible.

### Scripts

+ `sat_land_data.py`: This script is used to download the last 5 years of daily satellite and land mean temperature data for the two climate regions under investigation in this paper. These climate regions are Dublin (temperate climate) and Singapore (tropical climate). Before running the script, there are some requirements that must be satisfied. To download ERA5 satellite data, follow [these steps](https://cds.climate.copernicus.eu/api-how-to) on how to use the CDSAPI. For land data, there is an empty string in the script: `token = ''`. This string must be filled with a web services token before downloading land data from the NOAA website. Details on how to obtain a token are available [here](https://www.ncdc.noaa.gov/cdo-web/token). Once these steps have been taken, the script can be run. The satellite data is downloaded and stored in the current working directory as individual netcdf files for each year and region. The land data is downloaded also and stored in csv files in the current working directory, separated again by year and region. Note: Due to the large amount of data being requested from the CDS, this script may take a long time to execute fully.

+ `sat_land_comp.py`: This script is used to compare the satellite and land data using the same methods that are presented in the paper. This includes the heatmaps, box plots, linear regressions, statistical metrics and machine learning regression models. Please ensure that this script is placed in the same directory as the netcdf and csv files. Also, ensure that the names of the netcdf and csv files are not altered, whether they are obtained from the [Google Drive link](https://drive.google.com/drive/folders/1N68hx--Kyj9jFi0XnZihbZUzNMBxdY0i?usp=sharing) or the `sat_land_data.py` script. When this script is executed, the user will be prompted to enter the years the data has been downloaded from. Please enter the years from least recent to most recent, each separated by a single space.
