# Watershed Assessment and Visualization Engine (WAVE) Overview
The purpose of this tool is to improve access to global remote sensing and gridded climate data for analysis of global water budgets. The interactive interface allows users to define spatial periods of interest and filter to a specific basin boundary. For all datasets, the data has been minimally processed prior to aggregation for use in a monthly water budget calculation. This tab provides a brief description of the data inputs and processing workflow.

<br>

### Data Description
Below are the sources and data attributions for the datasets used in this application. Unless otherwise noted, all data sets were accessed from the Google Earth Engine Catalog through the Python API. In this tool, we prioritized datasets with global and near-real-time data updates to provide a spatially and temporally flexible framework.

<img src="images/dataset.png" alt="Diagram showing datasets used in analysis" width="1000">
<br>

### Water Budget Data and Analysis
As described in the 'Water Budget Data and Analysis' tab, the basic mass balance, water budget equation used to calculate change in storage is provided by equation 1.

***(Eq. 1) Change in Storage (&Delta;S) = Precipitation (P) - Evapotranspiration (ET) - Runoff (Q) ***

In Equation 1, &Delta;S represents total terrestrial water storage (TWS). TWS includes surface water storage, groundwater storage, soil moisture storage, snow storage, and canopy water storage among others. Given limitations in availability of global surface water storage data, at this time we are unable to fully disaggregate TWS globally to exclusively a change in groundwater storage. However, additional datasets enable the separation of snow, soil, and canopy water storage, which is done in the advanced water budget, as represented by Equation 2. 

***(Eq. 2) Change in Groundwater and Surface Water Storage (Î”GW + Î”SW) =  = Precipitation (P) - Evapotranspiration (ET) - Runoff (Q) - Change in Snow Water Storage (&Delta;SWE) Change in Soil Water Storage (&Delta;SM) - Change in Canopy Water Storage (&Delta;CWS)***


For the water budget calculations, the user is first prompted to select input datasets for each variable. As the water budget estimate may be affected by the input data selection, the 'Water Budget' data tab can be used to explore the input data available for the defined time-period and compare differences in data sources for each component of the water budget equation.

## Overview of Data Processing

### Water Budget
Below is a diagram providing an overview of the temporal and spatial aggregation and resampling workflow used to manage water budget input data of different resolutions. For each input dataset, we first aggregate the data to monthly intervals, represented as either a monthly sum (precipitation, ET, Runoff) or a monthly change (SWE, Soil Water Storage, Canopy Water Storage) for the defined analysis time-period. Once the data are temporally aggregated, the datasets are then resampled to match the resolution of the finest resolution data selected, using a mean reducer. In this analysis, the finest dataset available is the SSEBOP ET dataset, with a spatial resolution of 1 km.

<!-- ![Diagram showing temporal aggregation and processing workflow](images/processing_workflow.png) -->

<img src="images/processing_workflow.png" alt="Diagram showing temporal aggregation and processing workflow" width="700">

<br>

Following resampling, the water budget equation was applied to each pixel to produce monthly water budget estimates. For the map, the monthly water balance of each pixel was summed and divided by years in the time-period to produce an average annual change in storage estimate for each pixel. For the time-series, images are averaged to provide an estimate of the average monthly change in storage across the defined basin.

### Surface Water Elevation Data
Historic altimetric water level data was accessed via the DAHITI project, or the Database for Hydrological Time-Series of Inland Waters, which was developed by the German Geodetic Research Institute of the Technical University of Munich. This dataset hosts satellite altimetry data from multiple satellite missions and provides access to these data from a single API. For more information on the DAHITI project and API: https://dahiti.dgfi.tum.de/en/api/doc/. We did not add any additional processing to this dataset.


### GRACE DATA
Information on uncertainty estimates over the region of interest (Source: https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/): "For the JPL Mascons, measurement uncertainties are provided on a 0.5 degree grid in latitude and longitude for each monthly grid. Importantly, the uncertainties provided are the uncertainty values for each independent mascon, which are roughly 300 km in spatial extent. Thus, there are only 4,551 independent estimates of uncertainty represented for each monthly grid over the entire globe. This needs to be taken into account when computing the uncertainty of basin means. The mascon IDs for points in our regional analysis in this tutorial are determined using file containing mascon IDs. A detailed description of the uncertainty derivation is found in Wiese, Landerer and Watkins (2016). https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016WR019344".


## Visualization and Data Processing References

### Interactive Data Visualization Packages
1. Shiny for Python
3. plotly
4. ipyleaflet

### Data Processing 
* GRACE data correction and uncertaintly workflow was adapted from: https://github.com/podaac/the-coding-club/blob/main/n

* Python GEE API reference materials and documentation can be found at: https://developers.google.cn/earth-engine/apidocs

### Horizontal Datum
All datasets: WGS84

### Vertical Datums by Dataset
- DAHITI surface water elevations: EGM2008 geoid 
- SRTM DEM: EGM96 geoid 