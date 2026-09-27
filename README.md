# WAVE
## Water Assessment and Visualization Engine

### Authors  
Lea Davidson, Olivia Drukker, Sarah Newcomb, Tyler King, Roy Sando  

### Version: 0.2.0

---

## Overview

WAVE is a watershed analysis software deployed by the U.S. Geological Survey (USGS) that allows users to calculate and visualize water budgets for user-selected basins around the globe. It uses datasets from Google Earth Engine (GEE) to produce water budget estimates, basin characteristic summaries, and surface water visualizations. The software lowers technical training requirements by delivering decision ready data to the user and the pipeline flexibility allows for modification to address user-specific needs. 

Two interfaces are available depending on workflow needs:
 
- **`app.py`** — an interactive Shiny web application for point-and-click basin selection, dataset comparison, and water budget calculation
- **`WAVE_notebook.ipynb`** — a Jupyter notebook that walks through the same analyses step-by-step, suited for exploration, customization, and scripted workflows
Both interfaces share helper functions from **`app_functions.py`**.

### SWOT Water Levels

The **SWOT Water Levels** tab plots water surface elevation (WSE) time series from
the [Surface Water and Ocean Topography (SWOT)](https://swot.jpl.nasa.gov/) mission
(Version D vector products), retrieved from NASA PO.DAAC's public
[Hydrocron](https://podaac.github.io/hydrocron/) API. This feature requires **no
Earthdata Login**.

- **River reaches:** load SWOT reaches for the selected region of interest, then
  click a reach on the map to plot its full WSE record. Reach footprints come from
  NASA's Earthdata GIS river-reach feature service.
- **Prior lakes:** load lakes for the selected region of interest and click a lake
  on the map to plot its WSE record, or enter a SWOT Prior Lake Database `lake_id`
  directly. NASA does not publish a queryable lake footprint service, so the
  clickable lake map is driven by a small local Prior Lake Database (PLD) extract
  bundled under `data/` (see below). A Lesotho extract ships by default.

The time series defaults to the full available record (SWOT science observations
began in 2023).

#### Regenerating / adding a regional lake extract

Clickable lakes require a local PLD footprint file
(`data/swot_prior_lakes_<region>.geojson`). To build one for a different region,
edit `REGION_NAME` / `REGION_BBOX` in `prep_data/download_swot_prior_lakes.py`,
register the file in `SWOT_LAKE_FILES` in `app_functions.py`, then run the script
once. This step needs an [Earthdata Login](https://urs.earthdata.nasa.gov)
(the only part of the SWOT feature that does), supplied via `EARTHDATA_USERNAME` /
`EARTHDATA_PASSWORD` environment variables or a `~/.netrc` entry:

```bash
python prep_data/download_swot_prior_lakes.py
```

The running app only reads the generated GeoJSON and never needs Earthdata
credentials.

---

## Getting Started
 
### Prerequisites
 
- Python 3.13
- A Google Earth Engine service account with a downloaded JSON key file
- A DAHITI API key (for altimetric water level data; free registration at [dahiti.dgfi.tum.de](https://dahiti.dgfi.tum.de))

### Installation
 
```bash
conda env create -f wave_environment.yml
conda activate wave
```
 
### Environment Setup
 
Create a `.env` file in the project root:
 
```
EARTHENGINE_TOKEN=/path/to/your/service_account_key.json
DAHITI_API_KEY=your_dahiti_api_key
```
 
#### Note to USGS Network Users
 
If you see an SSl certificate verification error when the app tries connecting to APIs, add the following line to your `.env` file:
```
REQUESTS_CA_BUNDLE=C:\path\to\cacert_usgs.pem
```
 
### Running the Shiny App
 
```bash
shiny run app.py
```
 
### Running the Notebook
 
```bash
jupyter notebook WAVE_notebook.ipynb
```

---

### Abbreviations used throughout scripts  
| Abbreviation | Description |
| ----------- | ----------- |
| API | application programming interface |
| bbox | bounding box |
| CRS | coordinate reference system |
| CSV | comma-separated values |
| CW | canopy water |
| DAHITI | Database for Hydrological Time Series of Inland Waters |
| DEM | digital elevation model |
| EGM2008 | Earth Gravitational Model 2008 |
| EGM96 | Earth Gravitational Model 1996 |
| EPSG | European Petroleum Survey Group |
| ET | evapotranspiration |
| GEE | Google Earth Engine |
| GeoJSON | Geographic JSON |
| GLDAS | Global Land Data Assimilation System |
| GRACE | Gravity Recovery And Climate Experiment |
| GW | groundwater |
| HUC | hydrologic unit code |
| JPL | Jet Propulsion Laboratory |
| JSON | JavaScript object notation |
| LC | land cover |
| LSIB | Large Scale International Boundaries | 
| m	| meters |
| m.a.s.l | meters above sea level |
| MODIS | Moderate Resolution Imaging Spectroradiometer |
| NetCDF | network common data form |
| OLS | ordinary least squares |
| P | precipitation |
| Q | runoff (surface runoff / discharge) |
| ROI | region of interest |
| S | (total) water |
| SM | soil moisture |
| SRTM | Shuttle Radar Topography Mission |
| SW | surface water |
| SWE | snow water equivalent |
| SWOT | Surface Water and Ocean Topography |
| SWORD | SWOT River Database |
| PLD | Prior Lake Database |
| TIGER | Topologically Integrated Geographic Encoding and Referencing |
| TWS | terrestrial water storage |
| UI | user interface |
| US | United States |
| USDOS | United States Department of State |
| USGS | United States Geological Survey |
| WAVE | Water Assessment and Visualization Engine |
| WBD | Watershed Boundary Dataset |
| WGS84 | World Geodetic System 1984 |
| WSE | water surface elevation |
| WB | water budget/balance |

---

## Disclaimer
 
This software has been approved for release by the U.S. Geological Survey (USGS). Although the software has been subjected to rigorous review, the USGS reserves the right to update the software as needed pursuant to further analysis and review. No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. Furthermore, the software is released on condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from its authorized or unauthorized use.
 
