# Comparison of Satellite and Land Air Temperature Data Sets for Different Climate Regions

Contained within this repoistory is the code required to reproduce the results from the paper:

> B. McNicholl, Y. H. Lee, S. Winkler, A. Campbell, S. Dev, Analyzing Air Temperature from ERA5 Reanalysis Data, *under review*.

You are welcome to cite this paper if you wish to use some or all of the code. This code is for academic and research purposes only.

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

The source of land temperature data in this work is the Global Historical Climatology Network provided by the National Ocenanic and Atmospheric Administration ([NOAA GHCN](https://www.ncdc.noaa.gov/data-access/land-based-station-data/land-based-datasets/global-historical-climatology-network-ghcn)). The satellite temperature data was sourced from the Climate Data Store provided by the European Centre for Medium-Range Weather Forecasts ([ECMWF CDS](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)). The processed data used in this work is available to download.
