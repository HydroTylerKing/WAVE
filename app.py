import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import ee
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shinyswatch
import xarray as xr
from dotenv import load_dotenv
from google.auth.exceptions import RefreshError
from ipyleaflet import GeoJSON, Map, TileLayer
from matplotlib.colors import LinearSegmentedColormap
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

import app_functions

load_dotenv()

# Anchor all bundled file paths to this script's location so the app works
# regardless of the current working directory or which machine it runs on.
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

# Water budget variables
var_lookup = {
    "Precipitation": "P",
    "Snow Storage": "SWE",
    "Evapotranspiration": "ET",
    "Runoff": "Q",
    "Soil Moisture": "SM",
    "Canopy Storage": "CW",
    "Surface Water": "SW",
}

# Defaut dataset to be selected for basic and advanced water budget tables
default_datasets = {
    "P": "CHIRPS Precipitation",
    "ET": "ERA5 ET",
    "Q": "ERA5 Runoff",
    "SWE": "ERA5 SWE",
    "SM": "ERA5 Soil Water",
    "CW": "ERA5 Canopy Water",
}

# Color gradients for each variable
precip_col = ["#072345", "#164A87", "#457dc1", "#84ACE0", "#b9d0ee"]
et_col = ["#193c03", "#427721", "#7fc057", "#c9eab5"]
runoff_col = ["#04514a", "#4da79f", "#bcebe7"]
swe_col = ["#420459", "#a35dbc", "#e1c5eb"]
sm_col = ["#4A3005", "#9C783C", "#D0BFA4"]
cw_col = ["#df9215", "#efb545", "#ffd874"]

cols = {
    "ET": et_col,
    "P": precip_col,
    "Q": runoff_col,
    "SWE": swe_col,
    "SM": sm_col,
    "CW": cw_col,
}

# GEE authentication and initialization
# Supports three modes:
#   1. Service account via JSON contents (deployed, e.g. Posit Connect):
#      set EARTHENGINE_KEY_DATA to the full JSON key contents. Preferred on
#      hosting platforms where you set secrets as env vars, not files.
#   2. Service account via file path (local/automated): set EARTHENGINE_TOKEN
#      to the path of the JSON key file.
#   3. Personal credentials (local dev): runs ee.Authenticate() if neither of
#      the above is configured.
_ee_key_data = os.getenv("EARTHENGINE_KEY_DATA")
_ee_token_path = os.getenv("EARTHENGINE_TOKEN")

# Resolve the service-account info (a dict) from whichever mode is configured.
service_account_info = None

if _ee_key_data:
    # Mode 1: JSON contents supplied directly in an environment variable.
    try:
        service_account_info = json.loads(_ee_key_data)
    except json.JSONDecodeError as e:
        sys.exit(
            "EARTHENGINE_KEY_DATA is set but does not contain valid JSON: "
            f"{e}. Provide the full service-account key JSON as the value."
        )
elif _ee_token_path:
    # Mode 2: path to a JSON key file on the local filesystem.
    if not os.path.isfile(_ee_token_path):
        # Set but points to a missing file. Fail clearly instead of silently
        # falling back to the interactive browser flow (blocked on some
        # networks, e.g. USGS).
        sys.exit(
            f"EARTHENGINE_TOKEN is set to '{_ee_token_path}', but no file exists "
            "at that path. Point it at your service-account JSON key, set "
            "EARTHENGINE_KEY_DATA to the key contents, or unset both to use "
            "interactive authentication."
        )
    with open(_ee_token_path) as f:
        service_account_info = json.load(f)

if service_account_info is not None:
    # Service account authentication
    credentials = ee.ServiceAccountCredentials(
        email=service_account_info["client_email"],
        key_data=json.dumps(service_account_info),
    )
    try:
        ee.Initialize(credentials=credentials)
    except RefreshError as e:
        msg = str(e)
        if "invalid_grant" in msg and "Invalid JWT Signature" in msg:
            print(
                "Invalid JWT signature: likely wrong or revoked key. "
                "Request a fresh service-account JSON and retry."
            )
        else:
            print(f"Token refresh failed: {e}")
        sys.exit("Critical GEE authentication error.  Exiting program now.")
else:
    # Personal credentials authentication (local development)
    try:
        ee.Initialize(project="usgs-gee-datadrivenwaterbudget")
    except Exception:
        print("No existing GEE credentials found. Launching authentication flow...")
        ee.Authenticate()
        ee.Initialize(project="usgs-gee-datadrivenwaterbudget")

# Locate master data csv
dtst = pd.read_csv(DATA_DIR / "waterbudget_variables.csv")

# Make sure selected_basins.geojson exists as an empty feature collection on app startup
default_geojson_path = str(DATA_DIR / "selected_basins.geojson")
if not os.path.exists(default_geojson_path):
    with open(default_geojson_path, "w") as _f:
        json.dump(
            {"type": "FeatureCollection", "features": [], "id_field": "HYBAS_ID"}, _f
        )

gee_dict = pd.Series(dtst["gee_callout"].values, index=dtst["variable"]).to_dict()
var_dict = pd.Series(dtst["var_name"].values, index=dtst["variable"]).to_dict()
unit_dict = pd.Series(dtst["units"].values, index=dtst["variable"]).to_dict()
scale_dict = pd.Series(dtst["scale"].values, index=dtst["variable"]).to_dict()

lc_meta = pd.read_csv(DATA_DIR / "lc_dict.csv")
lc_dict = pd.Series(lc_meta["col"].values, index=lc_meta["class"]).to_dict()
lc_class = pd.Series(lc_meta["class"].values, index=lc_meta["val"]).to_dict()

irrType_meta = pd.read_csv(DATA_DIR / "irrType_dict.csv")
irr_dict = pd.Series(irrType_meta["col"].values, index=irrType_meta["class"]).to_dict()
irr_class = pd.Series(irrType_meta["class"].values, index=irrType_meta["val"]).to_dict()

countries = ee.FeatureCollection("USDOS/LSIB/2017")
ctry = countries.aggregate_array("COUNTRY_NA").getInfo()
_us_states = ee.FeatureCollection("TIGER/2018/States").aggregate_array("NAME").getInfo()

with open(DATA_DIR / "data_documentation.md", "r", encoding="utf-8") as file:
    data_markdown = file.read()

px.defaults.template = "simple_white"

today = pd.Timestamp(datetime.today())
end_last_year = pd.Timestamp(year=today.year - 1, month=12, day=31)

# WBD collections (US)
WBD_COLLECTIONS = {
    "HUC4": "USGS/WBD/2017/HUC04",
    "HUC6": "USGS/WBD/2017/HUC06",
    "HUC8": "USGS/WBD/2017/HUC08",
}

WBD_ID_FIELD = {
    "HUC4": "huc4",
    "HUC6": "huc6",
    "HUC8": "huc8",
}

# ─────────────────────────────────────────────────────────────────────────────
# UI
app_ui = ui.page_fluid(
    ui.busy_indicators.use(spinners=True, pulse=False),
    # styles ─────────────────────────────────────────────────────────
    ui.tags.style("""

        /* App title bar */
        .custom-header {
            color: snow;
            background-color: #00264c;
            background-image: url("images/logo.png");  /* www/images/logo.png */
            font-size: 2em;
            background-repeat: no-repeat;
            background-position: 10px center;
            background-size: 80px;
            padding-left: 0px;
            padding-bottom: 0px;
            height: 80px;
            line-height: 70px;
            text-align: center;
        }

        /* Custom dark-navy action buttons */
        .btn-custom {
            background-color: #00264c;
            color: white;
        }

        /* USGS header bar */
        .usgs-header-container {
            max-width: 100%;
            padding: 0.5rem 1rem;
            background-color: #00264c;
        }

        .usa-header.usgs-header {
            background-color: #00264c;
        }

        .usa-logo img {
            height: 50px;
        }

        /* USGS footer */
        .usa-footer--slim {
            background-color: #00264c;
            border-top: 2px solid #00264c;
            margin-top: 2rem;
            padding: 1rem 0;
        }

        .usgs-footer-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }

        .usgs-footer-nav {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            padding: 0.4rem 0;
            font-size: 0.85rem;
            align-items: center;
        }

        .usgs-footer-link {
            color: white;
            text-decoration: none;
        }

        .usgs-footer-link:hover {
            text-decoration: underline;
            color: gray;
        }
                  
        .disclaimer-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 20, 40, 0.75);
            z-index: 9999;
            align-items: center;
            justify-content: center;
        }
                  
        .disclaimer-overlay.active {
            display: flex;
        }
                  
        .disclaimer-box {
            background: #fff;
            border-top: 5px solid #c11f04;
            border-radius: 4px;
            max-width: 680px;
            width: 90%;
            padding: 2rem 2.5rem;
            box-shadow: 0 8px 40px rgba(0,0,0,0.35);
            font-family: Georgia, serif;
        }
                  
        .disclaimer-box h2 {
            color: #00264c;
            font-size: 1.3rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid #ddd;
            padding-bottom: 0.5rem;
        }
                  
        .disclaimer-body {
            font-size: 0.92rem;
            color: #333;
            line-height: 1.65;
            margin-bottom: 1.5rem;
        }
                  
        .disclaimer-body p { margin-bottom: 0.85rem; }
        .btn-agree {
            background-color: #00264c;
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            font-size: 1rem;
            border-radius: 3px;
            cursor: pointer;
            float: right;
        }
        .btn-agree:hover { background-color: #c11f04; }      

    """),
    # USGS header (top bar) ────────────────────────────────────────────────
    ui.tags.header(
        ui.tags.div(
            ui.tags.a(
                ui.tags.img(
                    src="images/logo.png", alt="USGS Home", style="height: 50px;"
                ),
                href="https://www.usgs.gov",
                target="_blank",
            ),
            class_="usgs-header-container",
            style="""
                display: flex;
                align-items: center;
                height: 60px;
                padding-left: 1rem;
            """,
        ),
        class_="usa-header usgs-header",
        role="banner",
    ),
    # WAVE app bar (under header) ────────────────────────────────────
    ui.tags.div(
        ui.tags.h1(
            "WAVE: Watershed Assessment and Visualization Engine",
            style="""
                color: white;
                font-size: 1.6rem;
                margin: 0;
                font-weight: 500;
            """,
        ),
        style="""
            background-color: #00264c;
            height: 70px;
            display: flex;
            align-items: center;
            justify-content: center;
        """,
    ),
    # Main nav ───────────────────────────────────────────────────────
    ui.navset_card_pill(
        # Tab 1: Select study area / time period
        ui.nav_panel(
            "Basin Selection",
            ui.card(
                ui.layout_columns(
                    ui.card(
                        ui.card(
                            ui.h5("1. Select Period of Interest"),
                            ui.input_date_range(
                                "var_date",
                                "Select period for mean monthly timeseries:",
                                start=pd.Timestamp(year=today.year - 4, month=1, day=1),
                                end=end_last_year,
                                format="yyyy-mm",
                                startview="year",
                                width="100%",
                            ),
                            ui.markdown("""
                                Data exists from 1981 onward. Selecting a longer period will result in longer processing times.
                                """),
                            height="300px",
                        ),
                        ui.card(
                            ui.h5("2. Define Region of Interest"),
                            ui.input_radio_buttons(
                                "roi_source",
                                "Choose ROI source:",
                                {
                                    "predefined": "Select from map",
                                    "upload": "Upload custom boundary",
                                },
                                selected="predefined",
                                inline=True,
                            ),
                            ui.panel_conditional(
                                "input.roi_source === 'predefined'",
                                ui.input_selectize(
                                    "Country",
                                    "Select a country",
                                    ctry,
                                    selected="United States",
                                    multiple=False,
                                ),
                                ui.input_selectize(
                                    "State",
                                    "For large countries, filter by smaller territory",
                                    [],
                                    selected=None,
                                    multiple=False,
                                ),
                                ui.panel_conditional(
                                    "input.Country !== 'United States'",
                                    ui.input_radio_buttons(
                                        "roi_method",
                                        "Select method to filter region",
                                        {"Basin": "Basin", "Sub-basin": "Sub-basin"},
                                        inline=True,
                                    ),
                                ),
                                ui.panel_conditional(
                                    "input.Country === 'United States'",
                                    ui.input_radio_buttons(
                                        "huc_level",
                                        "Select HUC level",
                                        {
                                            "HUC4": "HUC-4 (Subregion)",
                                            "HUC6": "HUC-6 (Basin)",
                                            "HUC8": "HUC-8 (Sub-basin)",
                                        },
                                        selected="HUC6",
                                        inline=True,
                                    ),
                                ),
                                ui.input_action_button(
                                    "update", "Update Map", class_="btn-custom"
                                ),
                                ui.markdown("""
                                    After selecting basins on map, press the button below to use these basins as the region of interest for all following analyses. A thumbnail map of the selected basins will appear when the basins have been loaded.
                                    """),
                                ui.input_action_button(
                                    "load_bas",
                                    "Use Selected Basins",
                                    style="background-color: #c11f04; color: white",
                                ),
                            ),
                            ui.panel_conditional(
                                "input.roi_source === 'upload'",
                                ui.input_file(
                                    "upload_aoi",
                                    "Upload a GeoJSON, Shapefile, or GeoPackage (.geojson, .json, .shp, .zip, .gpkg)",
                                    accept=[".geojson", ".json", ".shp", ".zip", ".gpkg"],
                                    multiple=False,
                                ),
                                ui.markdown("""
                                    Upload a GeoJSON file, a zipped Shapefile (.zip containing .shp, .shx, .dbf),
                                    or a GeoPackage (.gpkg).
                                    The uploaded boundary will be used as the region of interest for all analyses.
                                    """),
                                ui.input_action_button(
                                    "use_upload",
                                    "Use Uploaded Boundary",
                                    style="background-color: #c11f04; color: white",
                                ),
                            ),
                            ui.output_plot("thumbnail"),
                        ),
                    ),
                    ui.card(
                        ui.h2("Define Temporal and Spatial Parameters"),
                        output_widget("map"),
                    ),
                    col_widths=(3, 9),
                    height="1000px",
                    fill=True,
                )
            ),
        ),
        # Tab 2: Run basic and advanced water budget calcs
        ui.nav_panel(
            "Water Budget Analysis",
            ui.navset_pill_list(
                ui.nav_panel(
                    "Basic Water Budget",
                    ui.card(
                        ui.h2("Basic Water Budget"),
                        ui.markdown("""
                            Below are the outputs showing change in storage (ΔS) calculated using the mass budget water budget equation:
                            <br>
                            **Change in Storage (ΔS) = Precipitation (P) - Evapotranspiration (ET) - Runoff (Q)**
                            <br>
                            *Note: open water areas are masked from analysis.*
                        """),
                        ui.markdown("""
                            Please read the 'Data Overview' tab for a description of each dataset, and an overview of the data processing. The map below shows the cumulative change in storage ΔS, summed over the entirety of the selected period of record. The bar charts provide timeseries of the basin-averaged monthly change in storage.
                        """),
                        ui.h4("Select Datasets for use in Water Budget Calculations"),
                        ui.layout_columns(
                            ui.card(
                                ui.card_header("Precipitation (P)"),
                                ui.output_data_frame("ppt_table"),
                                height="250px",
                            ),
                            ui.card(
                                ui.card_header("Evapotranspiration (ET)"),
                                ui.output_data_frame("et_table"),
                                height="250px",
                            ),
                            ui.card(
                                ui.card_header("Runoff (Q)"),
                                ui.output_data_frame("q_table"),
                                height="250px",
                            ),
                            col_widths=(4, 4, 4),
                        ),
                        ui.layout_columns(
                            ui.input_action_button(
                                "run_wb1",
                                "Run Basic Water Budget",
                                style="background-color: #c11f04; color: white",
                            ),
                            ui.output_text_verbatim("scale_txt"),
                            col_widths=(2, 10),
                        ),
                        ui.accordion(
                            ui.accordion_panel(
                                "Map of Basic Water Budget Summed over Period",
                                ui.card(
                                    output_widget("wb1_map", height="500px"),
                                    ui.output_plot("wb1_colorbar", height="80px"),
                                    style="padding: 0; gap: 0; margin-bottom: 0;",
                                ),
                            ),
                            ui.accordion_panel(
                                "Timeseries of Monthly Basic Water Budget",
                                ui.navset_tab(
                                    ui.nav_panel(
                                        "Net Water Budget",
                                        ui.card(
                                            ui.h3("Net Monthly Water Budget"),
                                            output_widget("net_wb1"),
                                            height="600px",
                                            fill=True,
                                        ),
                                    ),
                                    ui.nav_panel(
                                        "Gross Water Budget",
                                        ui.card(
                                            ui.h3("Net Monthly Water Balance"),
                                            output_widget("gross_wb1"),
                                            height="600px",
                                            fill=False,
                                        ),
                                    ),
                                    ui.nav_panel(
                                        "Water Budget Ensemble",
                                        ui.card(
                                            ui.h3(
                                                "Ensemble Comparison Across All Available Datasets"
                                            ),
                                            output_widget("var_ens"),
                                            height="600px",
                                            fill=False,
                                        ),
                                    ),
                                    id="wb1_tabs",
                                ),
                            ),
                            open="Map of Basic Water Budget Summed over Period",
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Advanced Water Budget",
                    ui.card(
                        ui.h2("Advanced Water Budget"),
                        ui.markdown("""
                            Below are the outputs showing change in groundwater storage calculated using the mass budget water budget equation:
                            <br>
                            **Change in Groundwater Storage (ΔGW) = Precipitation (P) - Change in Snow Water Storage (ΔSWE) - Evapotranspiration (ET) - Runoff (Q) - Change in Soil Water Storage (ΔSM) - Change in Canopy Water Storage (ΔCWS)**
                            <br>
                            *Note: open water areas are masked from analysis.*
                        """),
                        ui.markdown("""
                            Please read the 'Data Overview' tab for a description of each dataset, and an overview of the data processing.
                        """),
                        ui.h3("Select Datasets for use in Water Budget Calculations"),
                        ui.layout_columns(
                            ui.card(
                                ui.card_header("Precipitation (P)"),
                                ui.output_data_frame("ppt_table2"),
                                height="250px",
                            ),
                            ui.card(
                                ui.card_header("Evapotranspiration (ET)"),
                                ui.output_data_frame("et_table2"),
                                height="250px",
                            ),
                            ui.card(
                                ui.card_header("Runoff (Q)"),
                                ui.output_data_frame("q_table2"),
                                height="250px",
                            ),
                            ui.card(
                                ui.card_header("Change in Snow Storage (ΔSWE)"),
                                ui.output_data_frame("swe_table2"),
                                height="250px",
                            ),
                            ui.card(
                                ui.card_header("Change in Soil Moisture (ΔSM)"),
                                ui.output_data_frame("sm_table2"),
                                height="250px",
                            ),
                            ui.card(
                                ui.card_header("Change in Canopy Water (ΔCWS)"),
                                ui.output_data_frame("cws_table2"),
                                height="250px",
                            ),
                            col_widths=(2, 2, 2, 2, 2, 2),
                        ),
                        ui.layout_columns(
                            ui.input_action_button(
                                "run_wb2",
                                "Run Advanced Water Budget",
                                style="background-color: #c11f04; color: white",
                            ),
                            ui.output_text_verbatim("scale_txt2"),
                            col_widths=(2, 10),
                        ),
                        ui.accordion(
                            ui.accordion_panel(
                                "Map of Advanced Water Budget Summed over Period",
                                ui.card(
                                    output_widget("wb2_map", height="500px"),
                                    ui.output_plot("wb2_colorbar", height="100px"),
                                    style="padding: 0; gap: 0; margin-bottom: 0;",
                                ),
                            ),
                            ui.accordion_panel(
                                "Timeseries of Monthly Advanced Water Budget",
                                ui.navset_tab(
                                    ui.nav_panel(
                                        "Net Water Budget",
                                        ui.card(
                                            ui.h3("Net Monthly Water Budget"),
                                            output_widget("net_wb2"),
                                            height="600px",
                                            fill=True,
                                        ),
                                    ),
                                    ui.nav_panel(
                                        "Gross Water Budget",
                                        ui.card(
                                            ui.h3(
                                                "Net Monthly Water Balance, by Variable"
                                            ),
                                            output_widget("gross_wb2"),
                                            height="600px",
                                            fill=False,
                                        ),
                                    ),
                                    id="wb2_tabs",
                                ),
                            ),
                            open="Map of Advanced Water Budget Summed over Period",
                        ),
                    ),
                ),
                widths=(1, 11),
            ),
        ),
        # Tab 3: Explore datasets
        ui.nav_panel(
            "Water Budget Data",
            ui.layout_columns(
                ui.card(
                    ui.h2("Compare Water Budget Input Data"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Select Variable to Visualize"),
                            ui.markdown(
                                "Select a variable from the dropdown menu, then press **Pull selected timeseries** to generate plots to the right."
                            ),
                            ui.input_select(
                                "wb_vars",
                                "Water Budget Variables:",
                                {
                                    "Precipitation": "Precipitation",
                                    "Evapotranspiration": "Evapotranspiration",
                                    "Runoff": "Runoff",
                                    "Soil Moisture": "Soil Moisture",
                                    "Snow Storage": "Snow Water Equivalent",
                                    "Canopy Storage": "Canopy Storage",
                                },
                            ),
                            ui.input_action_button(
                                "pull_var",
                                "Pull selected timeseries",
                                style="background-color: #c11f04; color: white",
                            ),
                        ),
                        ui.card(
                            ui.card_header("Select Specific Dataset to Map"),
                            ui.markdown(
                                "Select a dataset from the table, then press **Create Map** to generate a map below."
                            ),
                            ui.output_data_frame("var_map_table"),
                            height="200px",
                        ),
                        col_widths=(6, 6),
                    ),
                    ui.card(
                        ui.h3("Map of Selected Dataset"),
                        ui.layout_columns(
                            ui.input_action_button(
                                "gen_map",
                                "Create Map",
                                style="background-color: #c11f04; color: white",
                            ),
                            ui.markdown(
                                """Press button to the left to generate a map."""
                            ),
                            col_widths=(4, 8),
                        ),
                        ui.card(
                            output_widget("var_map", height="500px"),
                            ui.output_plot("colorbar", height="100px"),
                            style="padding: 0; gap: 0; margin-bottom: 0;",
                        ),
                        style="padding-bottom: 0; gap: 0;",
                        fill=False,
                    ),
                ),
                ui.card(
                    ui.h3("Comparison of Available Selected Input Variable Datasets"),
                    ui.h5("Mean Monthly Timeseries"),
                    output_widget("var_ts", height="550px"),
                    ui.h5("Annual Timeseries and Trend"),
                    output_widget("var_ts_ann", height="550px"),
                    height="1300px",
                    fill=False,
                ),
                col_widths=(6, 6),
            ),
        ),
        # Tab 4: View basin elevation and land cover chars
        ui.nav_panel(
            "Basin Characteristics",
            ui.card(ui.h3("Overview of Selected Basin Characteristics")),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Elevation",
                    ui.layout_columns(
                        ui.card(
                            ui.h4("Elevation of Selected Regions"),
                            output_widget("elev_hist"),
                        ),
                        ui.card(
                            ui.card(
                                output_widget("dem_map", height="600px"),
                                ui.output_plot("dem_colorbar", height="80px"),
                                style="padding: 0; gap: 0; margin-bottom: 0;",
                            ),
                            style="padding: 0; gap: 0;",
                        ),
                        col_widths=(6, 6),
                        height="800px",
                        fill=True,
                    ),
                ),
                ui.nav_panel(
                    "Land Cover",
                    ui.card(
                        ui.h4("Land Cover of Selected Regions (2024)"),
                        ui.layout_columns(
                            output_widget("lc_pie"),
                            output_widget("lc_map"),
                            col_widths=(6, 6),
                        ),
                        height="500px",
                        fill=True,
                    ),
                    ui.card(
                        ui.h4("Land Cover Change"),
                        output_widget("lc_ts"),
                        height="500px",
                    ),
                ),
                id="basin_chars_tabs",
            ),
        ),
        # Tab 5: Plot alt water level timeseries from DAHITI
        ui.nav_panel(
            "Surface Water Data",
            ui.card(
                ui.layout_columns(
                    ui.card(
                        ui.h2("Altimetric Water Level Data"),
                        ui.markdown(
                            """Historic altimetric water level data was accessed via the
                            [DAHITI](https://dahiti.dgfi.tum.de/) project (Database for
                            Hydrological Time Series of Inland Waters).

**To access this data, you need a free DAHITI API key:**

1. Register for a free account at [dahiti.dgfi.tum.de/en/register](https://dahiti.dgfi.tum.de/en/register/)
2. After registration, log in and navigate to your profile to find your API key
3. Paste your API key below, or set the `DAHITI_API_KEY` environment variable"""
                        ),
                        ui.input_password(
                            "dahiti_api_key",
                            "DAHITI API Key:",
                            value=os.environ.get("DAHITI_API_KEY", ""),
                            placeholder="Paste your DAHITI API key here",
                            width="100%",
                        ),
                        ui.output_plot("wl_map", height="900px"),
                    ),
                    ui.card(
                        ui.h3("Water Bodies with Water Elevation Data"),
                        ui.card(ui.output_data_frame("wl_table"), height="75px"),
                        ui.input_action_button(
                            "pull_wl",
                            "Plot selected water body elevation data",
                            style="background-color: #c11f04; color: white",
                        ),
                        output_widget("wl_ts", height="500px"),
                        height="1000px",
                    ),
                    height="1000px",
                    fill=False,
                    col_widths=(6, 6),
                )
            ),
        ),
        # Tab 6: Visualize GRACE terrestrial water storage anomalies
        ui.nav_panel(
            "GRACE",
            ui.layout_columns(
                ui.card(
                    ui.h3("GRACE Terrestrial Water Storage Data"),
                    ui.markdown(
                        """GRACE data provides information about changes in terrestrial water storage (TWS anomaly relative to 2004-2009 average)."""
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.input_action_button(
                                "load_grace",
                                "Pull GRACE data",
                                style="background-color: #c11f04; color: white",
                            ),
                            ui.output_text_verbatim("grace_text"),
                        ),
                        col_widths=(8,),
                    ),
                    ui.br(),
                    output_widget("grace_ts", height="400px"),
                    height="750px",
                ),
                ui.card(
                    output_widget("grace_map", height="600px"),
                    ui.output_plot("grace_colorbar", height="100px"),
                ),
                col_widths=(6, 6),
            ),
        ),
        # Tab 7: SWOT water surface elevation timeseries
        ui.nav_panel(
            "SWOT Water Levels",
            ui.card(
                ui.layout_columns(
                    ui.card(
                        ui.h2("SWOT Water Surface Elevation"),
                        ui.markdown(
                            """Water surface elevation (WSE) time series from the
                            [Surface Water and Ocean Topography (SWOT)](https://swot.jpl.nasa.gov/)
                            mission, retrieved from NASA PO.DAAC's public
                            [Hydrocron](https://podaac.github.io/hydrocron/) API
                            (Version D vector products). No login is required.

**Rivers:** choose *River reach*, load reaches for your region of interest,
then click a reach on the map to plot its full WSE record.

**Lakes:** choose *Prior lake*, load lakes for your region of interest, then
click a lake on the map to plot its full WSE record — or enter a SWOT Prior
Lake Database `lake_id` directly. Clickable lakes are drawn from a bundled
Prior Lake Database extract (Lesotho by default), since NASA does not publish a
queryable lake footprint service."""
                        ),
                        ui.input_radio_buttons(
                            "swot_feature_type",
                            "Feature type:",
                            {"Reach": "River reach", "PriorLake": "Prior lake"},
                            selected="Reach",
                            inline=True,
                        ),
                        ui.panel_conditional(
                            "input.swot_feature_type === 'Reach'",
                            ui.input_action_button(
                                "load_swot_reaches",
                                "Load river reaches for region of interest",
                                style="background-color: #c11f04; color: white",
                            ),
                            ui.markdown(
                                "*Click a reach on the map below to plot its "
                                "water surface elevation timeseries.*"
                            ),
                            output_widget("swot_map", height="650px"),
                        ),
                        ui.panel_conditional(
                            "input.swot_feature_type === 'PriorLake'",
                            ui.input_action_button(
                                "load_swot_lakes",
                                "Load lakes for region of interest",
                                style="background-color: #c11f04; color: white",
                            ),
                            ui.markdown(
                                "*Click a lake on the map to plot its water "
                                "surface elevation, or enter a lake_id below.*"
                            ),
                            output_widget("swot_lake_map", height="500px"),
                            ui.input_text(
                                "swot_lake_id",
                                "SWOT Prior Lake Database lake_id:",
                                placeholder="e.g. 6350036182",
                                width="100%",
                            ),
                            ui.input_action_button(
                                "pull_swot_lake",
                                "Plot lake water surface elevation",
                                style="background-color: #c11f04; color: white",
                            ),
                        ),
                    ),
                    ui.card(
                        ui.h3("Water Surface Elevation Timeseries"),
                        ui.output_text_verbatim("swot_selection_txt"),
                        ui.input_checkbox(
                            "swot_remove_outliers",
                            "Remove flagged outliers from plot "
                            "(robust MAD screening)",
                            value=False,
                        ),
                        output_widget("swot_ts", height="500px"),
                        height="1000px",
                    ),
                    height="1000px",
                    fill=False,
                    col_widths=(6, 6),
                )
            ),
        ),
        # Tab 8: Dataset descriptions and processing doc
        ui.nav_panel(
            "Data Overview",
            ui.card(ui.markdown(data_markdown)),
        ),
    ),
    # USGS Footer ───────────────────────────────────────────────────────────
    ui.tags.footer(
        ui.tags.nav(
            ui.tags.div(
                ui.tags.a(
                    "DOI Privacy Policy",
                    href="https://www.doi.gov/privacy",
                    class_="usgs-footer-link",
                ),
                " | ",
                ui.tags.a(
                    "Legal",
                    href="https://www.usgs.gov/legal",
                    class_="usgs-footer-link",
                ),
                " | ",
                ui.tags.a(
                    "Accessibility",
                    href="https://www.usgs.gov/accessibility",
                    class_="usgs-footer-link",
                ),
                " | ",
                ui.tags.a(
                    "Site Map",
                    href="https://www.usgs.gov/sitemap",
                    class_="usgs-footer-link",
                ),
                " | ",
                ui.tags.a(
                    "Contact USGS",
                    href="https://answers.usgs.gov/",
                    class_="usgs-footer-link",
                ),
                class_="usgs-footer-nav",
            ),
            ui.tags.div(
                ui.tags.a(
                    "U.S. Department of the Interior",
                    href="https://www.doi.gov/",
                    class_="usgs-footer-link",
                ),
                " | ",
                ui.tags.a(
                    "DOI Inspector General",
                    href="https://www.doioig.gov/",
                    class_="usgs-footer-link",
                ),
                " | ",
                ui.tags.a(
                    "White House",
                    href="https://www.whitehouse.gov/",
                    class_="usgs-footer-link",
                ),
                " | ",
                ui.tags.a(
                    "No Fear Act",
                    href="https://www.doi.gov/pmb/eeo/no-fear-act",
                    class_="usgs-footer-link",
                ),
                " | ",
                ui.tags.a(
                    "FOIA", href="https://www.usgs.gov/foia", class_="usgs-footer-link"
                ),
                class_="usgs-footer-nav",
            ),
            class_="usgs-footer-container",
            aria_label="Footer Navigation",
        ),
        class_="usa-footer usa-footer--slim",
        role="contentinfo",
    ),
    ui.tags.script("""
        document.addEventListener('DOMContentLoaded', function() {
            // Inject overlay into body
            var overlay = document.createElement('div');
            overlay.id = 'disclaimer-overlay';
            overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,20,40,0.78);z-index:99999;display:flex;align-items:center;justify-content:center;';
            overlay.innerHTML = `
                <div style="background:#fff;border-top:5px solid #c11f04;border-radius:4px;max-width:660px;width:90%;padding:2rem 2.5rem;box-shadow:0 8px 40px rgba(0,0,0,0.4);font-family:Georgia,serif;">
                    <h2 style="color:#00264c;font-size:1.2rem;margin:0 0 1rem 0;padding-bottom:0.75rem;border-bottom:1px solid #ddd;letter-spacing:0.01em;">
                        &#9432;&nbsp; Disclaimer Statement
                    </h2>
                    <p style="font-size:0.93rem;color:#333;line-height:1.7;margin:0 0 1.75rem 0;">
                        This software has been approved for release by the U.S. Geological Survey (USGS). 
                        Although the software has been subjected to rigorous review, the USGS reserves the 
                        right to update the software as needed pursuant to further analysis and review. No 
                        warranty, expressed or implied, is made by the USGS or the U.S. Government as to 
                        the functionality of the software and related material nor shall the fact of release 
                        constitute any such warranty. Furthermore, the software is released on condition 
                        that neither the USGS nor the U.S. Government shall be held liable for any damages 
                        resulting from its authorized or unauthorized use.
                    </p>
                    <div style="text-align:right;">
                        <button onclick="document.getElementById('disclaimer-overlay').remove();" 
                            style="background:#00264c;color:white;border:none;padding:0.6rem 2.2rem;font-size:1rem;border-radius:3px;cursor:pointer;font-family:Georgia,serif;letter-spacing:0.03em;"
                            onmouseover="this.style.background='#c11f04'" 
                            onmouseout="this.style.background='#00264c'">
                            Acknowledge
                        </button>
                    </div>
                </div>
            `;
            document.body.appendChild(overlay);
        });
    """),
    theme=shinyswatch.theme.cosmo,
)


# ─────────────────────────────────────────────────────────────────────────────
# SERVER
def server(input, output, _session):
    # Map layers to use throughout app
    sat = TileLayer(
        url="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    labels = TileLayer(
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
    )

    def active_fc_info():
        """
        Returns collection for the current selection.
        US: WBD at the chosen HUC level, filtered to chosen state.
        For all other countries: HydroSHEDS basin or sub-basin.
        """
        simplify_tol = 500  # meters

        if str(input.Country()).strip() == "United States":
            level = str(input.huc_level()) if input.huc_level() else "HUC4"
            fc = ee.FeatureCollection(WBD_COLLECTIONS[level])
            id_field = WBD_ID_FIELD[level]

            if input.State():
                state_feat = ee.FeatureCollection("TIGER/2018/States").filter(
                    ee.Filter.eq("NAME", str(input.State()))
                )
                fc = fc.filterBounds(state_feat.geometry())
            fc = fc.select([id_field]).map(lambda f: f.simplify(simplify_tol))
            return fc, id_field

        else:
            # HydroSHEDS
            method = str(input.roi_method()) if input.roi_method() else "Basin"
            fc = ee.FeatureCollection(gee_dict[method])
            country = countries.filter(ee.Filter.eq("COUNTRY_NA", str(input.Country())))
            fc = fc.filterBounds(country.geometry())
            fc = fc.select(["HYBAS_ID"]).map(lambda f: f.simplify(simplify_tol))
            return fc, "HYBAS_ID"

    @reactive.calc
    def features():
        return str(input.roi_method())

    def state():
        if str(input.Country()) == "United States":
            return _us_states
        return []

    @reactive.effect
    def default_country_state():
        default_state = "Montana" if str(input.Country()) == "United States" else None
        ui.update_selectize("State", choices=state(), selected=default_state)

    @output
    @render_widget
    def map():
        if input.use_upload() and _is_custom_upload():
            # Show the uploaded custom boundary on the map
            with reactive.isolate():
                gdf = gpd.read_file(default_geojson_path)
                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

                # Calculate zoom level from bounding box extent
                lat_range = bounds[3] - bounds[1]
                lon_range = bounds[2] - bounds[0]
                max_range = max(lat_range, lon_range)
                if max_range > 0:
                    import math as _math
                    zoom = int(_math.floor(8 - _math.log2(max_range)))
                    zoom = max(2, min(zoom, 15))
                else:
                    zoom = 10

                geojson_data = json.loads(gdf.to_json())
                geo_layer = GeoJSON(
                    data=geojson_data,
                    style={"color": "red", "fillColor": "red", "fillOpacity": 0.3, "weight": 2},
                    hover_style={"fillOpacity": 0.5},
                    name="Uploaded AOI",
                )

                leaflet_map = Map(center=center, zoom=zoom, scroll_wheel_zoom=True)
                leaflet_map.add_layer(sat)
                leaflet_map.add_layer(labels)
                leaflet_map.add_layer(geo_layer)
                leaflet_map.layout.height = "850px"
                return leaflet_map
        elif input.update():
            with reactive.isolate():
                fc, id_field = active_fc_info()
                feat_json = app_functions.feat2json(fc)

                if str(input.Country()).strip() == "United States":
                    if input.State():
                        state_feat = ee.FeatureCollection("TIGER/2018/States").filter(
                            ee.Filter.eq("NAME", str(input.State()))
                        )
                        center, bb = app_functions.calc_centroid(state_feat)
                    else:
                        country = countries.filter(
                            ee.Filter.eq("COUNTRY_NA", "United States")
                        )
                        center, bb = app_functions.calc_centroid(country)
                else:
                    country = countries.filter(
                        ee.Filter.eq("COUNTRY_NA", str(input.Country()))
                    )
                    center, bb = app_functions.calc_centroid(country)

                for feature in feat_json["features"]:
                    feature["properties"]["style"] = {
                        "color": "blue",
                        "fillColor": None,
                        "weight": 0.75,
                    }

                def style_callback(feature):
                    if feature["properties"].get("selected"):
                        return {"color": "red", "fillOpacity": 0.6, "weight": 3}
                    return {"color": "blue", "fillOpacity": 0.3, "weight": 2}

                geo_layer = GeoJSON(
                    data=feat_json,
                    style_callback=style_callback,
                    hover_style={"fillColor": "red", "fillOpacity": 0.6},
                    name="Basins",
                )

                def on_click_handler(_event=None, feature=None, **kwargs):
                    clicked_id = feature["properties"][id_field]
                    for f in feat_json["features"]:
                        if f["properties"][id_field] == clicked_id:
                            f["properties"]["selected"] = not f["properties"].get(
                                "selected", False
                            )
                    selected_features = [
                        f
                        for f in feat_json["features"]
                        if f["properties"].get("selected")
                    ]
                    selected_geojson = {
                        "type": "FeatureCollection",
                        "features": selected_features,
                        "id_field": id_field,
                    }
                    with open(default_geojson_path, "w") as f:
                        json.dump(selected_geojson, f)
                    import copy

                    geo_layer.data = copy.deepcopy(feat_json)

                geo_layer.on_click(on_click_handler)

                leaflet_map = Map(center=center, zoom=6, scroll_wheel_zoom=True)
                leaflet_map.add_layer(sat)
                leaflet_map.add_layer(labels)
                leaflet_map.add_layer(geo_layer)
                leaflet_map.layout.height = "850px"
                return leaflet_map
        else:
            leaflet_map = Map(center=(20, 0), zoom=2, scroll_wheel_zoom=True)
            leaflet_map.add_layer(sat)
            leaflet_map.add_layer(labels)
            leaflet_map.layout.height = "850px"
            return leaflet_map

    @render.plot
    @reactive.event(input.load_bas, input.use_upload)
    def thumbnail():
        sub = gpd.read_file(default_geojson_path)
        return sub.plot(color="red", edgecolor="red", alpha=0.5)

    def load_selected_bas():
        with open(default_geojson_path, "r") as f:
            sel_bas = json.load(f)
        id_field = sel_bas.get("id_field", "HYBAS_ID")
        return sel_bas, id_field

    # Reactive value to track uploaded custom AOI geometry
    _custom_aoi = reactive.value(None)

    @reactive.effect
    @reactive.event(input.use_upload)
    def _handle_upload():
        """Process uploaded GeoJSON/Shapefile and save to selected_basins.geojson."""
        file_info = input.upload_aoi()
        if not file_info:
            return
        file_path = file_info[0]["datapath"]
        file_name = file_info[0]["name"]

        # Read with geopandas (handles .geojson, .json, .shp, .zip)
        if file_name.endswith(".zip"):
            gdf = gpd.read_file(f"zip://{file_path}")
        else:
            gdf = gpd.read_file(file_path)

        # Ensure WGS84
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Dissolve all features into a single geometry for the ROI
        gdf_dissolved = gdf.dissolve()

        # Save as GeoJSON to the default path (same format the app expects)
        geojson_dict = json.loads(gdf.to_json())
        geojson_dict["id_field"] = "custom_upload"
        with open(default_geojson_path, "w") as f:
            json.dump(geojson_dict, f)

        # Convert to ee.Geometry for GEE operations
        geojson_geom = json.loads(gdf_dissolved.geometry.to_json())
        # Handle FeatureCollection vs single geometry
        if geojson_geom["type"] == "FeatureCollection":
            features = geojson_geom["features"]
            ee_geom = ee.Geometry(features[0]["geometry"])
            if len(features) > 1:
                geoms = [ee.Geometry(feat["geometry"]) for feat in features]
                ee_geom = ee.Geometry.MultiPolygon(
                    [g.coordinates() for g in geoms]
                )
        elif geojson_geom["type"] == "GeometryCollection":
            ee_geom = ee.Geometry(geojson_geom)
        else:
            ee_geom = ee.Geometry(geojson_geom)

        _custom_aoi.set(ee_geom)

    def _is_custom_upload():
        """Return True if the user loaded a custom uploaded boundary."""
        return _custom_aoi() is not None and input.use_upload() > 0

    @reactive.calc
    def roi_geom():
        # Custom upload takes priority if user clicked "Use Uploaded Boundary"
        if _is_custom_upload():
            return _custom_aoi()
        if input.load_bas():
            sel_bas, id_field = load_selected_bas()
            fc, _ = active_fc_info()
            if id_field == "HYBAS_ID":
                ids = app_functions.pull_ids(sel_bas)
                bas_feat = fc.filter(ee.Filter.inList("HYBAS_ID", ids))
            else:
                ids = app_functions.pull_huc_ids(sel_bas, id_field)
                bas_feat = fc.filter(ee.Filter.inList(id_field, ids))
            return bas_feat.geometry()

    @reactive.calc
    def basin_feat():
        # For custom upload, create a FeatureCollection from the geometry
        if _is_custom_upload():
            geom = _custom_aoi()
            return ee.FeatureCollection([ee.Feature(geom)])
        if input.load_bas():
            sel_bas, id_field = load_selected_bas()
            fc, _ = active_fc_info()
            if id_field == "HYBAS_ID":
                ids = app_functions.pull_ids(sel_bas)
                bas_feat = fc.filter(ee.Filter.inList("HYBAS_ID", ids))
            else:
                ids = app_functions.pull_huc_ids(sel_bas, id_field)
                bas_feat = fc.filter(ee.Filter.inList(id_field, ids))
            return bas_feat

    @reactive.calc
    def roi_area():
        if basin_feat():
            return basin_feat().geometry().bounds().area(maxError=1)

    @reactive.calc
    def valid_et_vars():
        """
        Return the set of ET variable names that have non-masked pixels over
        the selected ROI and date range. Computed once per basin/date change.
        """
        if not (input.load_bas() or _is_custom_upload()) or not input.var_date():
            return None  # No filter applied until a basin is loaded
        roi = roi_geom()
        if roi is None:
            return None
        i = i_date()
        f = f_date()
        et_rows = dtst[dtst["water_budget"] == "ET"]
        valid = set()
        for _, row in et_rows.iterrows():
            if app_functions.has_valid_et_pixels(
                row["gee_callout"], row["var_name"], roi, i, f
            ):
                valid.add(row["variable"])
        return valid

    @reactive.calc
    def sel_bas_layer():
        if input.load_bas():
            with open(default_geojson_path, "r") as file:
                sel_bas = json.load(file)
            for feature in sel_bas["features"]:
                feature["properties"]["style"] = {
                    "color": "red",
                    "weight": 0.5,
                    "fillOpacity": 0,
                }
            return GeoJSON(data=sel_bas, name="Basins")

    @reactive.calc
    def coords():
        if basin_feat():
            return app_functions.calc_centroid(basin_feat())

    @reactive.calc
    def dem_img():
        if not basin_feat():
            return None
        return ee.Image("USGS/SRTMGL1_003").select("elevation").clip(roi_geom())

    @render_widget
    def elev_hist():
        if not basin_feat() or input.basin_chars_tabs() != "Elevation":
            return px.histogram()
        notification_id = ui.notification_show(
            "Generating elevation histogram...", duration=None, type="message"
        )
        try:
            roi = roi_geom()
            dem = dem_img()
            dem_pixels = (
                dem.reduceRegion(reducer=ee.Reducer.toList(), geometry=roi, scale=5000)
                .get("elevation")
                .getInfo()
            )
            dem_array = np.array(dem_pixels)
            dem_fig = px.histogram(
                dem_array,
                nbins=25,
                title="Selected Basin Elevation Distribution",
                histnorm="probability",
                color_discrete_sequence=["black"],
            )
            dem_fig.update_layout(
                xaxis_title="Elevation (m)",
                yaxis_title="Portion of Watershed(s)",
                bargap=0.1,
                height=600,
                showlegend=False,
            )
            return dem_fig
        finally:
            ui.notification_remove(notification_id)

    @reactive.calc
    def dem_max():
        dem = dem_img()
        return ee.Number(
            dem.reduceRegion(ee.Reducer.max(), roi_geom(), scale=5000)
        ).getInfo()["elevation"]

    @reactive.calc
    def dem_min():
        dem = dem_img()
        return ee.Number(
            dem.reduceRegion(ee.Reducer.min(), roi_geom(), scale=5000)
        ).getInfo()["elevation"]

    @render.plot
    def dem_colorbar():
        if not basin_feat() or input.basin_chars_tabs() != "Elevation":
            return None
        fig, ax = plt.subplots(figsize=(6, 0.5))
        cmap = LinearSegmentedColormap.from_list("Elev", ["black", "white"], N=255)
        norm = mpl.colors.Normalize(vmin=dem_min(), vmax=dem_max())
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
            label="Elevation (m)",
        )
        return fig

    @render_widget
    def dem_map():
        if not basin_feat() or input.basin_chars_tabs() != "Elevation":
            empty_m = Map(center=(20, 0), zoom=2, scroll_wheel_zoom=True)
            empty_m.add_layer(sat)
            return empty_m
        notification_id = ui.notification_show(
            "Loading DEM map...", duration=None, type="message"
        )
        try:
            center, bb = coords()
            sel_layer = sel_bas_layer()
            dem = dem_img()
            dem_vis = {
                "min": dem_min(),
                "max": dem_max(),
                "palette": ["black", "white"],
            }
            m_dem = Map(center=center, zoom=8, scroll_wheel_zoom=True)
            dem_layer = app_functions.ee_image_to_ipyleaflet(dem, dem_vis, name="DEM")
            m_dem.add_layer(dem_layer)
            m_dem.add_layer(sel_layer)
            m_dem.layout.height = "600px"
            return m_dem
        finally:
            ui.notification_remove(notification_id)

    @render_widget
    def lc_ts():
        if not basin_feat() or input.basin_chars_tabs() != "Land Cover":
            return px.line()
        notification_id = ui.notification_show(
            "Generating land cover timeseries...", duration=None, type="message"
        )
        try:
            lc_ts_ic = (
                ee.ImageCollection("MODIS/061/MCD12Q1")
                .select("LC_Type1")
                .filterBounds(roi_geom())
            )
            collectionList = lc_ts_ic.toList(lc_ts_ic.size())
            collectionSize = collectionList.size().getInfo()
            area_ls = []
            years = range(2001, 2025)
            for index in range(collectionSize):
                area_df = pd.DataFrame()
                lc = ee.Image(collectionList.get(index)).clip(roi_geom())
                area_dict = app_functions.area_by_class(lc, roi_geom()).getInfo()
                labels = []
                sizes = []
                classes = []
                for land_cover_class in list(area_dict["LC_Type1"].keys()):
                    class_area = area_dict["LC_Type1"][str(land_cover_class)]
                    if class_area > 0:
                        classes.append(str(land_cover_class))
                        labels.append(
                            lc_class[int(land_cover_class.removesuffix(".0"))]
                        )
                        sizes.append(class_area)
                area_df["classes"] = classes
                area_df["labels"] = labels
                area_df["sizes"] = sizes
                area_df["year"] = years[index]
                area_ls.append(area_df)
            combined_df = pd.concat(area_ls)
            lc_fig = go.Figure()
            for land_cover in combined_df["labels"].unique():
                land_cover_data = combined_df[combined_df["labels"] == land_cover]
                lc_fig.add_trace(
                    go.Scatter(
                        x=land_cover_data["year"],
                        y=land_cover_data["sizes"],
                        mode="lines",
                        stackgroup="one",
                        line=dict(width=0.5, color=lc_dict[land_cover]),
                        fillcolor=lc_dict[land_cover],
                        name=land_cover,
                        groupnorm="percent",
                    )
                )
            lc_fig.update_layout(
                title="Stacked Area Chart of Land Cover Over Time",
                xaxis_title="Year",
                yaxis_title="Percent Coverage",
                hovermode="x unified",
                legend_title="Land Cover Type",
                template="plotly_white",
                height=400,
            )
            return lc_fig
        finally:
            ui.notification_remove(notification_id)

    @render_widget
    def lc_pie():
        if not basin_feat() or input.basin_chars_tabs() != "Land Cover":
            return px.pie()
        notification_id = ui.notification_show(
            "Loading land cover data...", duration=None, type="message"
        )
        try:
            roi = roi_geom()
            lc = (
                ee.ImageCollection("MODIS/061/MCD12Q1")
                .select("LC_Type1")
                .reduce(ee.Reducer.last())
                .clip(roi)
            )
            area_dict = app_functions.area_by_class(lc, roi).getInfo()
            labels = []
            sizes = []
            for land_cover_class in list(area_dict["LC_Type1_last"].keys()):
                class_area = area_dict["LC_Type1_last"][str(land_cover_class)]
                if class_area > 0:
                    labels.append(lc_class[int(land_cover_class.removesuffix(".0"))])
                    sizes.append(class_area)
            return px.pie(
                values=sizes,
                names=labels,
                color=labels,
                color_discrete_map=lc_dict,
                title="2024 MODIS Land Cover Distribution",
                hole=0.2,
                width=600,
                height=400,
            )
        finally:
            ui.notification_remove(notification_id)

    @render_widget
    def lc_map():
        if not basin_feat() or input.basin_chars_tabs() != "Land Cover":
            empty_m = Map(center=(20, 0), zoom=2, scroll_wheel_zoom=True)
            empty_m.add_layer(sat)
            return empty_m
        notification_id = ui.notification_show(
            "Loading land cover map...", duration=None, type="message"
        )
        try:
            center, bb = coords()
            sel_layer = sel_bas_layer()
            lc = (
                ee.ImageCollection("MODIS/061/MCD12Q1")
                .select("LC_Type1")
                .reduce(ee.Reducer.last())
                .clip(basin_feat())
            )
            LandCoverVis = {
                "min": 1.0,
                "max": 17.0,
                "palette": [
                    "05450a",
                    "086a10",
                    "54a708",
                    "78d203",
                    "009900",
                    "c6b044",
                    "dcd159",
                    "dade48",
                    "fbff13",
                    "b6ff05",
                    "27ff87",
                    "c24f44",
                    "a5a5a5",
                    "ff6d4c",
                    "69fff8",
                    "f9ffa4",
                    "1c0dff",
                ],
            }
            m_lc = Map(center=center, zoom=7, scroll_wheel_zoom=True)
            m_lc.add_layer(sat)
            m_lc.add_layer(
                app_functions.ee_image_to_ipyleaflet(lc, LandCoverVis, "Land Cover")
            )
            m_lc.add_layer(sel_layer)
            m_lc.layout.height = "500px"
            return m_lc
        finally:
            ui.notification_remove(notification_id)

    @reactive.calc
    def dahiti_dat():
        roi = roi_geom()
        # Use API key from input field, fall back to environment variable
        api_key = input.dahiti_api_key() or os.environ.get("DAHITI_API_KEY", "")
        if not api_key:
            return None
        return app_functions.pull_altim_dahiti(roi, i_date=i_date(), f_date=f_date(), api_key=api_key)

    @render.plot
    def wl_map():
        if dahiti_dat():
            with open(default_geojson_path, "r") as file:
                sel_bas = json.load(file)
            for feature in sel_bas["features"]:
                feature["properties"]["style"] = {
                    "color": "red",
                    "weight": 1,
                    "fillOpacity": 0,
                }
            alt_dict, alt_json, alt_names = dahiti_dat()
            ctry_gdf = gpd.GeoDataFrame.from_features(sel_bas["features"])
            alt_gpd = gpd.GeoDataFrame.from_features(
                alt_json["features"]
            ).drop_duplicates(subset="Name")
            ax = ctry_gdf.plot(color="red", edgecolor="red", alpha=0.3)
            alt_gpd.plot(
                ax=ax, color="blue", edgecolor="black", figsize=(10, 10), alpha=0.7
            )
            for x, y, label in zip(
                alt_gpd.geometry.centroid.x,
                alt_gpd.geometry.centroid.y,
                alt_gpd["Name"],
            ):
                ax.text(x, y, label, fontsize=12, ha="right", color="black")
            plt.title("Water Bodies with Water Elevation Data")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            return ax

    @render.data_frame
    def wl_table():
        if dahiti_dat():
            alt_dict, alt_json, alt_names = dahiti_dat()
            available_names = list(alt_dict.keys())
            alt_df = (
                alt_names[alt_names.isin(available_names)]
                .drop_duplicates()
                .reset_index(drop=True)
                .to_frame(name="Water Body")
            )
            if alt_df.empty:
                return render.DataTable(
                    pd.DataFrame(
                        {"Water Body": ["No data available in selected period"]}
                    ),
                    selection_mode="rows",
                    width="fit-content",
                )
            return render.DataTable(alt_df, selection_mode="rows", width="fit-content")
        return render.DataTable(
            pd.DataFrame(columns=["Water Body"]),
            selection_mode="rows",
            width="fit-content",
        )

    @reactive.calc
    def wl_ordered_names():
        dat = dahiti_dat()
        if dat is None:
            return []
        alt_dict, alt_json, alt_names = dat
        available_names = list(alt_dict.keys())
        return (
            alt_names[alt_names.isin(available_names)]
            .drop_duplicates()
            .reset_index(drop=True)
            .tolist()
        )

    @render_widget
    @reactive.event(input.pull_wl)
    def wl_ts():
        if not input.pull_wl():
            return px.line()
        notification_id = ui.notification_show(
            "Loading water level timeseries...", duration=None, type="message"
        )
        try:
            dat = dahiti_dat()
            if dat is None:
                return px.line()
            alt_dict, alt_json, alt_names = dat
            selected_rows = wl_table.cell_selection()["rows"]
            if not selected_rows:
                return px.line(title="Please select a water body from the table above")
            ordered = wl_ordered_names()
            if selected_rows[0] >= len(ordered):
                return px.line(title="Selection out of range")
            sel_wl = ordered[selected_rows[0]]
            wl_df = alt_dict[sel_wl]
            if wl_df.empty:
                return px.line(title=f"No data in selected period for {sel_wl}")
            fig = px.line(
                wl_df,
                x="datetime",
                y="wse",
                title=sel_wl,
                color_discrete_sequence=["blue"],
            )
            fig.update_layout(
                height=450, yaxis_title="Water Elevation (m.a.s.l)", xaxis_title=""
            )
            return fig
        finally:
            ui.notification_remove(notification_id)

    # ── SWOT water surface elevation ────────────────────────────────────────
    # Holds the identifier of the SWOT feature the user has chosen to plot,
    # plus its type ("Reach" or "PriorLake").
    _swot_selection = reactive.value(None)

    def _roi_bbox():
        """(min_lon, min_lat, max_lon, max_lat) for the current ROI, or None."""
        roi = roi_geom()
        if roi is None:
            return None
        ring = roi.bounds().getInfo()["coordinates"][0]
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        return (min(lons), min(lats), max(lons), max(lats))

    @render_widget
    def swot_map():
        # Center on the ROI if available, else a world view.
        center = (20, 0)
        zoom = 2
        if coords():
            center, _bb = coords()
            zoom = 8

        leaflet_map = Map(center=center, zoom=zoom, scroll_wheel_zoom=True)
        leaflet_map.add_layer(sat)
        leaflet_map.add_layer(labels)
        leaflet_map.layout.height = "650px"

        # Only load reaches once the user presses the button.
        if not input.load_swot_reaches():
            return leaflet_map

        notification_id = ui.notification_show(
            "Loading SWOT river reaches for region of interest...",
            duration=None,
            type="message",
        )
        try:
            bbox = _roi_bbox()
            if bbox is None:
                ui.notification_show(
                    "Define a region of interest on the Basin Selection tab first.",
                    duration=5,
                    type="warning",
                )
                return leaflet_map
            fc = app_functions.fetch_swot_reaches(bbox)
        finally:
            ui.notification_remove(notification_id)

        features = fc.get("features", [])
        if not features:
            ui.notification_show(
                "No SWOT river reaches found in this region of interest.",
                duration=5,
                type="warning",
            )
            return leaflet_map

        for feat in features:
            feat.setdefault("properties", {})["style"] = {
                "color": "#1f78b4",
                "weight": 3,
                "opacity": 0.8,
            }

        reach_layer = GeoJSON(
            data=fc,
            hover_style={"color": "#c11f04", "weight": 6},
            name="SWOT River Reaches",
        )

        def on_reach_click(_event=None, feature=None, **kwargs):
            if not feature:
                return
            rid = feature["properties"].get("reach_id")
            if rid is not None:
                _swot_selection.set(("Reach", str(rid)))

        reach_layer.on_click(on_reach_click)
        leaflet_map.add_layer(reach_layer)
        return leaflet_map

    @reactive.effect
    @reactive.event(input.pull_swot_lake)
    def _set_swot_lake_selection():
        lake_id = (input.swot_lake_id() or "").strip()
        if lake_id:
            _swot_selection.set(("PriorLake", lake_id))

    @render_widget
    def swot_lake_map():
        center = (-29.6, 28.2)  # Lesotho, the region the local PLD covers
        zoom = 8
        if coords():
            center, _bb = coords()

        leaflet_map = Map(center=center, zoom=zoom, scroll_wheel_zoom=True)
        leaflet_map.add_layer(sat)
        leaflet_map.add_layer(labels)
        leaflet_map.layout.height = "500px"

        # Only load lakes once the user presses the button.
        if not input.load_swot_lakes():
            return leaflet_map

        notification_id = ui.notification_show(
            "Loading SWOT prior lakes...", duration=None, type="message"
        )
        try:
            fc = app_functions.fetch_swot_lakes(bbox=_roi_bbox())
        finally:
            ui.notification_remove(notification_id)

        features = fc.get("features", [])
        if not features:
            ui.notification_show(
                "No SWOT prior lakes found. Ensure the region's lake file "
                "exists and your region of interest overlaps it.",
                duration=6,
                type="warning",
            )
            return leaflet_map

        for feat in features:
            feat.setdefault("properties", {})["style"] = {
                "color": "#1f78b4",
                "fillColor": "#1f78b4",
                "fillOpacity": 0.4,
                "weight": 1,
            }

        lake_layer = GeoJSON(
            data=fc,
            hover_style={"fillColor": "#c11f04", "fillOpacity": 0.7},
            name="SWOT Prior Lakes",
        )

        def on_lake_click(_event=None, feature=None, **kwargs):
            if not feature:
                return
            lid = feature["properties"].get("lake_id")
            if lid is not None:
                _swot_selection.set(("PriorLake", str(lid)))

        lake_layer.on_click(on_lake_click)
        leaflet_map.add_layer(lake_layer)
        return leaflet_map

    @reactive.effect
    @reactive.event(input.swot_feature_type)
    def _reset_swot_selection():
        # Clear any prior selection when the user switches feature type.
        _swot_selection.set(None)

    @render.text
    def swot_selection_txt():
        sel = _swot_selection()
        if sel is None:
            if input.swot_feature_type() == "Reach":
                return (
                    "Load reaches, then click one on the map to plot its "
                    "water surface elevation."
                )
            return "Enter a lake_id and press the plot button."
        feature_type, fid = sel
        label = "River reach" if feature_type == "Reach" else "Prior lake"
        return f"Selected {label}: {fid}"

    @render_widget
    def swot_ts():
        sel = _swot_selection()
        if sel is None:
            return px.line()
        feature_type, fid = sel
        notification_id = ui.notification_show(
            "Loading SWOT water surface elevation timeseries...",
            duration=None,
            type="message",
        )
        try:
            # Default to the full record (helper falls back to mission start).
            df = app_functions.pull_swot_timeseries(fid, feature_type=feature_type)
            if df.empty:
                return px.line(
                    title=f"No SWOT observations available for {fid}"
                )
            label = "River reach" if feature_type == "Reach" else "Prior lake"

            # Robustly flag outliers (MAD-based modified z-score).
            df = app_functions.flag_wse_outliers(df, value_col="wse")
            kept = df[~df["is_outlier"]]
            flagged = df[df["is_outlier"]]
            n_flagged = len(flagged)

            fig = go.Figure()
            if input.swot_remove_outliers():
                # Removed: plot only the retained points.
                fig.add_trace(
                    go.Scatter(
                        x=kept["time"],
                        y=kept["wse"],
                        mode="lines+markers",
                        name="WSE (screened)",
                        line=dict(color="#1f78b4"),
                        marker=dict(color="#1f78b4"),
                    )
                )
                title = f"SWOT WSE — {label} {fid} ({n_flagged} outliers removed)"
            else:
                # Show the full series and highlight what would be removed.
                fig.add_trace(
                    go.Scatter(
                        x=df["time"],
                        y=df["wse"],
                        mode="lines+markers",
                        name="WSE",
                        line=dict(color="#1f78b4"),
                        marker=dict(color="#1f78b4"),
                    )
                )
                if n_flagged:
                    fig.add_trace(
                        go.Scatter(
                            x=flagged["time"],
                            y=flagged["wse"],
                            mode="markers",
                            name="Flagged outlier",
                            marker=dict(color="#c11f04", size=9, symbol="x"),
                        )
                    )
                suffix = (
                    f" ({n_flagged} outliers flagged)"
                    if n_flagged
                    else " (no outliers flagged)"
                )
                title = f"SWOT WSE — {label} {fid}{suffix}"

            fig.update_layout(
                title=title,
                height=450,
                yaxis_title="Water Surface Elevation (m)",
                xaxis_title="",
            )
            return fig
        finally:
            ui.notification_remove(notification_id)

    @render.text
    @reactive.event(input.run_wb1)
    def scale_txt():
        scale_wb = scale()
        return (
            f"Scale set to {scale_wb} m"
            if scale_wb is not None
            else "No scale set — reduce temporal or spatial extent"
        )

    @render.text
    @reactive.event(input.run_wb2)
    def scale_txt2():
        scale_wb = scale2()
        return (
            f"Scale set to {scale_wb} m"
            if scale_wb is not None
            else "No scale set — reduce temporal or spatial extent"
        )

    @reactive.calc
    @reactive.event(input.run_wb1)
    def scale():
        if _is_custom_upload():
            roi_area_val = _custom_aoi().bounds().area(maxError=1)
        else:
            sel_bas, id_field = load_selected_bas()
            fc, _ = active_fc_info()
            if id_field == "HYBAS_ID":
                bas_feat = fc.filter(
                    ee.Filter.inList("HYBAS_ID", app_functions.pull_ids(sel_bas))
                )
            else:
                bas_feat = fc.filter(
                    ee.Filter.inList(
                        id_field, app_functions.pull_huc_ids(sel_bas, id_field)
                    )
                )
            roi_area_val = bas_feat.bounds().area(maxError=1)
        scale_txt = app_functions.get_res(i_date(), f_date(), roi_area_val)
        return scale_txt if not math.isnan(scale_txt) else None

    @reactive.calc
    @reactive.event(input.run_wb2)
    def scale2():
        if _is_custom_upload():
            roi_area_val = _custom_aoi().bounds().area(maxError=1)
        else:
            sel_bas, id_field = load_selected_bas()
            fc, _ = active_fc_info()
            if id_field == "HYBAS_ID":
                bas_feat = fc.filter(
                    ee.Filter.inList("HYBAS_ID", app_functions.pull_ids(sel_bas))
                )
            else:
                bas_feat = fc.filter(
                    ee.Filter.inList(
                        id_field, app_functions.pull_huc_ids(sel_bas, id_field)
                    )
                )
            roi_area_val = bas_feat.geometry().bounds().area(maxError=1)
        scale_txt = app_functions.get_res(i_date(), f_date(), roi_area_val)
        return scale_txt if not math.isnan(scale_txt) else None

    @reactive.calc
    def i_date():
        if input.var_date():
            return str(
                pd.to_datetime(input.var_date()[0])
                .to_period("M")
                .start_time.strftime("%Y-%m-%d")
            )

    @reactive.calc
    def f_date():
        if input.var_date():
            return str(
                (
                    pd.to_datetime(input.var_date()[1]).to_period("M").end_time
                    + pd.offsets.Day(1)
                ).strftime("%Y-%m-%d")
            )

    @render.data_frame
    def var_map_table():
        if coords() and input.var_date():
            var_map = var_lookup[input.wb_vars()]
            center, bb = coords()
            user_var = pd.DataFrame(
                dtst.loc[
                    (dtst["water_budget"] == var_map)
                    & (pd.to_datetime(dtst["date_onset"]) <= i_date())
                    & (pd.to_datetime(dtst["date_end"]) >= f_date())
                    & ((dtst["lat_min"] < int(bb[1])) & (dtst["lat_max"] > int(bb[3])))
                ],
                columns=["variable"],
            ).rename(columns={"variable": "Dataset"})
            return render.DataTable(
                user_var.reset_index(drop=True),
                selection_mode="rows",
                width="fit-content",
            )
        user_var = pd.DataFrame(
            {
                "WARNING": [
                    "Please define period of interest on first tab to view available datasets"
                ]
            }
        )
        return render.DataTable(
            user_var.reset_index(drop=True), selection_mode="rows", width="fit-content"
        )

    @reactive.calc
    def sel_years():
        start_year = int(
            pd.to_datetime(input.var_date()[0]).to_period("M").start_time.strftime("%Y")
        )
        end_year = int(
            (
                pd.to_datetime(input.var_date()[1]).to_period("M").end_time
                + pd.offsets.Day(1)
            ).strftime("%Y")
        )
        return len(range(start_year, end_year + 1))

    @reactive.calc
    def sel_var_map():
        rows = var_map_table.cell_selection()["rows"]
        if not rows:
            return None
        center, bb = coords()
        xmin, ymin, xmax, ymax = bb
        var_map = var_lookup[input.wb_vars()]
        user_var = pd.DataFrame(
            dtst.loc[
                (dtst["water_budget"] == var_map)
                & (pd.to_datetime(dtst["date_onset"]) <= i_date())
                & (pd.to_datetime(dtst["date_end"]) >= f_date())
                & (
                    (dtst["lat_min"] < int(bb[1])) & (dtst["lat_max"] > int(bb[3]))
                )  # ADD bbox filter to match table
            ],
            columns=["variable"],
        )
        row_idx = int(rows[0])
        if row_idx >= len(user_var):
            return None
        return str(user_var.values.tolist()[row_idx][0])

    @render_widget
    @reactive.event(input.gen_map)
    def var_map():
        if not input.gen_map():
            empty_m = Map(center=(20, 0), zoom=2, scroll_wheel_zoom=True)
            empty_m.add_layer(sat)
            return empty_m

        if not coords() or not input.var_date():
            empty_m = Map(center=(20, 0), zoom=2, scroll_wheel_zoom=True)
            empty_m.add_layer(sat)
            return empty_m

        user_var = sel_var_map()
        if user_var is None:
            ui.notification_show(
                "Please select a dataset row from the table first.",
                duration=4,
                type="warning",
            )
            empty_m = Map(center=(20, 0), zoom=2, scroll_wheel_zoom=True)
            empty_m.add_layer(sat)
            return empty_m

        notification_id = ui.notification_show(
            "Generating variable map...", duration=None, type="message"
        )
        try:
            with reactive.isolate():
                i = str(
                    pd.to_datetime(input.var_date()[0])
                    .to_period("M")
                    .start_time.strftime("%Y-%m-%d")
                )
                f = str(
                    (
                        pd.to_datetime(input.var_date()[1]).to_period("M").end_time
                        + pd.offsets.Day(1)
                    ).strftime("%Y-%m-%d")
                )
                dst = gee_dict[user_var]
                var = var_dict[user_var]
                selected_var = str(input.wb_vars())
                roi = roi_geom()
                if selected_var in ("Evapotranspiration", "Precipitation", "Runoff"):
                    var_img = (
                        ee.ImageCollection(dst)
                        .select(var)
                        .filter(ee.Filter.date(i, f))
                        .sum()
                        .divide(sel_years())
                        .clip(roi)
                    )
                else:
                    var_img = (
                        ee.ImageCollection(dst)
                        .select(var)
                        .filter(ee.Filter.date(i, f))
                        .max()
                        .divide(sel_years())
                        .clip(roi)
                    )

                unit = unit_dict[user_var]
                scale_val = scale_dict[user_var]
                var_img_converted = app_functions.unit_standardization_img(
                    var_img, unit, scale_val
                )

                stats = var_img_converted.reduceRegion(
                    reducer=ee.Reducer.percentile([2, 98]),
                    geometry=roi_geom(),
                    scale=25000,
                    maxPixels=1e8,
                    bestEffort=True,
                ).getInfo()
                var_min = stats.get(f"{var}_p2")
                var_max = stats.get(f"{var}_p98")
                title = (
                    "Mean Annual "
                    if selected_var in ("Evapotranspiration", "Precipitation", "Runoff")
                    else "Maximum "
                )
                _var_map_stats["last"] = {
                    "min": var_min,
                    "max": var_max,
                    "title": title,
                }
                center, bb = coords()
                var_col = [str(c) for c in cols[var_lookup[input.wb_vars()]][::-1]]
                var_vis = {"min": var_min, "max": var_max, "palette": var_col}
                m_var = Map(center=center, zoom=8, scroll_wheel_zoom=True)
                m_var.add_layer(sat)
                m_var.add_layer(
                    app_functions.ee_image_to_ipyleaflet(
                        var_img_converted, var_vis, str(input.wb_vars())
                    )
                )
                m_var.add_layer(sel_bas_layer())
                m_var.layout.height = "500px"
                return m_var
        finally:
            ui.notification_remove(notification_id)

    @reactive.calc
    @reactive.event(input.pull_var)
    def var_df():
        notification_id = ui.notification_show(
            "Pulling variable timeseries from GEE...", duration=None, type="message"
        )
        try:
            selected_var = [input.wb_vars()]
            roi = roi_geom()
            roi_area_loc = roi_area()
            center, bb = coords()
            i = str(
                pd.to_datetime(input.var_date()[0])
                .to_period("M")
                .start_time.strftime("%Y-%m-%d")
            )
            f = str(
                (
                    pd.to_datetime(input.var_date()[1]).to_period("M").end_time
                    + pd.offsets.Day(1)
                ).strftime("%Y-%m-%d")
            )
            set_scale = app_functions.get_res(i, f, roi_area_loc)
            for key in selected_var:
                indices = dtst[
                    (dtst["water_budget"] == var_lookup[key])
                    & (pd.to_datetime(dtst["date_onset"]) <= i)
                    & (pd.to_datetime(dtst["date_end"]) >= f)
                    & ((dtst["lat_min"] < int(bb[1])) & (dtst["lat_max"] > int(bb[3])))
                ].index.tolist()
                df_ens = pd.DataFrame()
                for index in indices:
                    if key in ("Snow Storage", "Soil Moisture", "Canopy Storage"):
                        image = (
                            ee.ImageCollection(dtst["gee_callout"][index])
                            .select(dtst["var_name"][index])
                            .filter(ee.Filter.date(i, f))
                            .filterBounds(roi)
                        )
                        months, years = app_functions.generate_month_sequence(i, f)
                        month = app_functions.rename(
                            app_functions.monthly_agg(months, years, image, "mean"),
                            [dtst["var_name"][index]],
                        )
                    else:
                        image = (
                            ee.ImageCollection(dtst["gee_callout"][index])
                            .select(dtst["var_name"][index])
                            .filter(ee.Filter.date(i, f))
                            .filterBounds(roi)
                        )
                        months, years = app_functions.generate_month_sequence(i, f)
                        month = app_functions.rename(
                            app_functions.monthly_agg(months, years, image, "sum"),
                            [dtst["var_name"][index]],
                        )
                    df_ts = (
                        app_functions.array_2_df(
                            (month.getRegion(roi, set_scale).getInfo()),
                            [dtst["var_name"][index]],
                        )
                        .groupby("datetime", as_index=False)
                        .mean()
                    )
                    unit = str(dtst["units"][index])
                    if unit == "cm":
                        unit_convert = df_ts[str(dtst["var_name"][index])] / 100
                    elif unit in ("mm", "kg/m^2 8day", "kg/m^2"):
                        unit_convert = df_ts[str(dtst["var_name"][index])] / 1000
                    else:
                        unit_convert = df_ts[str(dtst["var_name"][index])]
                    if key == "Evapotranspiration":
                        df_ens[str(dtst["variable"][index])] = (
                            abs(unit_convert) * dtst["scale"][index]
                        )
                    else:
                        df_ens[str(dtst["variable"][index])] = (
                            unit_convert * dtst["scale"][index]
                        )
                df_ens.index = df_ts["datetime"]
            return df_ens
        finally:
            ui.notification_remove(notification_id)

    @render_widget
    @reactive.event(input.pull_var)
    def var_ts():
        df_ens = var_df()
        if df_ens is None:
            return px.line()
        fig = px.line(
            df_ens,
            x=df_ens.index,
            y=df_ens.columns,
            title="Comparison of " + str(input.wb_vars()) + " Datasets",
            color_discrete_sequence=cols[var_lookup[input.wb_vars()]],
        )
        fig.update_layout(
            height=500,
            yaxis_title="Mean Monthly " + str(input.wb_vars()) + " (m)",
            xaxis_title="",
            legend_title="Data Source",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.35,
                xanchor="center",
                x=0.5,
            ),
        )
        return fig

    @render_widget
    @reactive.event(input.pull_var)
    def var_ts_ann():
        df_ens = var_df()
        if df_ens is None:
            return px.line()
        selected_var = str(input.wb_vars())

        if selected_var in ("Evapotranspiration", "Precipitation", "Runoff"):
            df_ann = df_ens.resample("YE").sum()
            title = "Annual Total "
        else:
            df_ann = df_ens.resample("YE").max()
            title = "Annual Maximum "

        year_sizes = df_ens.resample("YE").size()
        df_ann = df_ann[year_sizes == 12]

        fig = px.scatter(
            df_ann,
            x=df_ann.index,
            y=df_ann.columns,
            title=title + str(input.wb_vars()),
            color_discrete_sequence=cols[var_lookup[input.wb_vars()]],
            trendline="ols",
        )
        fig.update_layout(
            height=500,
            yaxis_title=title + str(input.wb_vars()) + " (m)",
            xaxis_title="",
            legend_title="Data Source",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.35,
                xanchor="center",
                x=0.5,
            ),
        )
        return fig

    _var_map_stats: dict = {}

    @render.plot
    def colorbar():
        if not input.gen_map() or "last" not in _var_map_stats:
            return None
        stats = _var_map_stats["last"]
        var_col = [str(c) for c in cols[var_lookup[input.wb_vars()]][::-1]]
        cmap = LinearSegmentedColormap.from_list("Var", var_col, N=255)
        norm = mpl.colors.Normalize(vmin=stats["min"], vmax=stats["max"])
        cfig, ax = plt.subplots(figsize=(6, 0.5))
        cfig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
            label=f"{stats['title']}{input.wb_vars()} (m)",
        )
        return cfig

    @render_widget
    def var_ens():
        if input.wb1_tabs() != "Water Budget Ensemble":
            return px.line()
        if not input.load_bas() or not input.run_wb1():
            return px.line()
        if roi_geom() is None or i_date() is None:
            return px.line()
        notification_id = ui.notification_show(
            "Calculating water budget ensemble...", duration=None, type="message"
        )
        try:
            roi = roi_geom()
            roi_area_loc = roi_area()
            i = i_date()
            f = f_date()
            set_scale = app_functions.get_res(i, f, roi_area_loc)
            all_vars = {}
            for key in ["Precipitation", "Evapotranspiration", "Runoff"]:
                indices = dtst[
                    (dtst["water_budget"] == var_lookup[key])
                    & (pd.to_datetime(dtst["date_onset"]) <= i)
                    & (pd.to_datetime(dtst["date_end"]) >= f)
                ].index.tolist()
                df_ens = pd.DataFrame()
                for index in indices:
                    image = (
                        ee.ImageCollection(dtst["gee_callout"][index])
                        .select(dtst["var_name"][index])
                        .filter(ee.Filter.date(i, f))
                        .filterBounds(roi)
                    )
                    months, years = app_functions.generate_month_sequence(i, f)
                    month = app_functions.rename(
                        app_functions.monthly_agg(months, years, image, "sum"),
                        [dtst["var_name"][index]],
                    )
                    df_ts = (
                        app_functions.array_2_df(
                            (month.getRegion(roi, set_scale).getInfo()),
                            [dtst["var_name"][index]],
                        )
                        .groupby("datetime", as_index=False)
                        .mean()
                    )
                    unit = str(dtst["units"][index])
                    if unit == "cm":
                        unit_convert = df_ts[str(dtst["var_name"][index])] / 100
                    elif unit in ("mm", "kg/m^2 8day", "kg/m^2"):
                        unit_convert = df_ts[str(dtst["var_name"][index])] / 1000
                    else:
                        unit_convert = df_ts[str(dtst["var_name"][index])]
                    if key == "Evapotranspiration":
                        df_ens[str(dtst["variable"][index])] = (
                            abs(unit_convert) * dtst["scale"][index]
                        )
                    else:
                        df_ens[str(dtst["variable"][index])] = (
                            unit_convert * dtst["scale"][index]
                        )
                df_ens.index = df_ts["datetime"]
                all_vars[key] = df_ens
            run_ens = [
                p - et - q
                for p in [
                    all_vars["Precipitation"].iloc[:, pc]
                    for pc in range(len(all_vars["Precipitation"].columns))
                ]
                for et in [
                    all_vars["Evapotranspiration"].iloc[:, ec]
                    for ec in range(len(all_vars["Evapotranspiration"].columns))
                ]
                for q in [
                    all_vars["Runoff"].iloc[:, qc]
                    for qc in range(len(all_vars["Runoff"].columns))
                ]
            ]
            ens = pd.DataFrame({i: run_ens[i] for i in range(len(run_ens))})
            ens_min = ens.apply(min, axis=1)
            ens_max = ens.apply(max, axis=1)
            ens_mean = ens.mean(axis=1)
            ens_mov_ave = ens_mean.rolling(
                window=12, min_periods=12, center=False
            ).mean()
            mean_fig = px.line(title="Ensemble of Calculate Storage Change Estimates")
            mean_fig.add_traces(
                [
                    px.line(
                        ens, x=ens.index, y=ens_max, color_discrete_sequence=["#E8E8E7"]
                    ).data[0],
                    px.line(
                        ens, x=ens.index, y=ens_min, color_discrete_sequence=["#E8E8E7"]
                    ).data[0],
                ]
            )
            mean_fig.add_hline(y=0, line_color="#070707", line_width=1, opacity=0.8)
            mean_fig.add_traces(
                [
                    dict(
                        type="scatter",
                        x=ens.index,
                        y=ens_max,
                        marker=dict(color="#E8E8E7"),
                        line=dict(width=0),
                        mode="lines",
                        fillcolor="#E8E8E7",
                        fill="tonexty",
                        showlegend=True,
                        name="Range",
                    )
                ]
            )
            mean_fig.add_trace(
                go.Scatter(
                    x=ens.index,
                    y=ens_mean,
                    mode="lines",
                    name="Mean",
                    line=dict(color="#2C2C2D", width=2),
                )
            )
            mean_fig.add_trace(
                go.Scatter(
                    x=ens.index,
                    y=ens_mov_ave,
                    mode="lines",
                    name="One Year Moving Average",
                    line=dict(color="#384FEA", width=2),
                )
            )
            mean_fig.update_layout(
                yaxis_title="Monthly Change in Total Water Storage (m)",
                xaxis_title="Month",
            )
            return mean_fig
        finally:
            ui.notification_remove(notification_id)

    _user_picked = {
        "ppt_table": reactive.value(False),
        "et_table": reactive.value(False),
        "q_table": reactive.value(False),
        "ppt_table2": reactive.value(False),
        "et_table2": reactive.value(False),
        "q_table2": reactive.value(False),
        "sm_table2": reactive.value(False),
        "cws_table2": reactive.value(False),
        "swe_table2": reactive.value(False),
    }

    def make_dataset_table(water_budget_code, col_label, table_id, valid_vars=None):
        if coords() and input.var_date():
            center, bb = coords()
            xmin, ymin, xmax, ymax = bb
            mask = (
                (dtst["water_budget"] == water_budget_code)
                & (pd.to_datetime(dtst["date_onset"]) <= pd.to_datetime(i_date()))
                & (pd.to_datetime(dtst["date_end"]) >= pd.to_datetime(f_date()))
                & (dtst["lat_max"] >= ymin)
                & (dtst["lat_min"] <= ymax)
                & (dtst["lon_max"] >= xmin)
                & (dtst["lon_min"] <= xmax)
            )
            df = pd.DataFrame(dtst.loc[mask], columns=["variable"]).rename(
                columns={"variable": col_label}
            )
            if water_budget_code == "ET" and valid_vars is not None:
                df = df[df[col_label].isin(valid_vars)]

            # Sort alphabetically
            df = df.sort_values(by=col_label).reset_index(drop=True)

            styles = None
            # Show the default dataset
            if not df.empty and not _user_picked[table_id].get():
                preferred = default_datasets.get(water_budget_code)
                match_idx = df.index[df[col_label] == preferred].tolist()
                idx = match_idx[0] if match_idx else 0
                styles = [
                    {
                        "rows": [idx],
                        "style": {
                            "background-color": "#d6e9ff",
                        },
                    }
                ]

            return render.DataTable(
                df,
                selection_mode="rows",
                width="fit-content",
                styles=styles,
            )

        return render.DataTable(
            pd.DataFrame({col_label: []}).reset_index(drop=True),
            selection_mode="rows",
            width="fit-content",
        )

    # Watches the table dataset selection
    @reactive.effect
    def _watch_ppt_table():
        if ppt_table.cell_selection()["rows"]:
            _user_picked["ppt_table"].set(True)

    @reactive.effect
    def _watch_et_table():
        if et_table.cell_selection()["rows"]:
            _user_picked["et_table"].set(True)

    @reactive.effect
    def _watch_q_table():
        if q_table.cell_selection()["rows"]:
            _user_picked["q_table"].set(True)

    @reactive.effect
    def _watch_ppt_table2():
        if ppt_table2.cell_selection()["rows"]:
            _user_picked["ppt_table2"].set(True)

    @reactive.effect
    def _watch_et_table2():
        if et_table2.cell_selection()["rows"]:
            _user_picked["et_table2"].set(True)

    @reactive.effect
    def _watch_q_table2():
        if q_table2.cell_selection()["rows"]:
            _user_picked["q_table2"].set(True)

    @reactive.effect
    def _watch_sm_table2():
        if sm_table2.cell_selection()["rows"]:
            _user_picked["sm_table2"].set(True)

    @reactive.effect
    def _watch_cws_table2():
        if cws_table2.cell_selection()["rows"]:
            _user_picked["cws_table2"].set(True)

    @reactive.effect
    def _watch_swe_table2():
        if swe_table2.cell_selection()["rows"]:
            _user_picked["swe_table2"].set(True)

    def sel_var(table_ref, water_budget_code, col_label, valid_vars=None):
        center, bb = coords()
        xmin, ymin, xmax, ymax = bb
        mask = (
            (dtst["water_budget"] == water_budget_code)
            & (pd.to_datetime(dtst["date_onset"]) <= pd.to_datetime(i_date()))
            & (pd.to_datetime(dtst["date_end"]) >= pd.to_datetime(f_date()))
            & (dtst["lat_max"] >= ymin)
            & (dtst["lat_min"] <= ymax)
            & (dtst["lon_max"] >= xmin)
            & (dtst["lon_min"] <= xmax)
        )
        df = pd.DataFrame(dtst.loc[mask], columns=["variable"]).rename(
            columns={"variable": col_label}
        )
        if water_budget_code == "ET" and valid_vars is not None:
            df = df[df[col_label].isin(valid_vars)]

        df = df.sort_values(by=col_label).reset_index(drop=True)

        if df.empty:
            return None

        rows = table_ref.cell_selection()["rows"]

        if rows:
            row_idx = int(rows[0])
            if row_idx < len(df):
                return str(df.iloc[row_idx][col_label])
            return None

        preferred = default_datasets.get(water_budget_code)
        match_idx = df.index[df[col_label] == preferred].tolist()
        idx = match_idx[0] if match_idx else 0
        return str(df.iloc[idx][col_label])

    @render.data_frame
    def ppt_table():
        return make_dataset_table("P", "Precipitation Dataset", "ppt_table")

    @render.data_frame
    def et_table():
        return make_dataset_table("ET", "ET Dataset", "et_table", valid_et_vars())

    @render.data_frame
    def q_table():
        return make_dataset_table("Q", "Runoff Dataset", "q_table")

    @render.data_frame
    def ppt_table2():
        return make_dataset_table("P", "Precipitation Dataset", "ppt_table2")

    @render.data_frame
    def et_table2():
        return make_dataset_table("ET", "ET Dataset", "et_table2", valid_et_vars())

    @render.data_frame
    def q_table2():
        return make_dataset_table("Q", "Runoff Dataset", "q_table2")

    @render.data_frame
    def sm_table2():
        return make_dataset_table("SM", "Soil Moisture Dataset", "sm_table2")

    @render.data_frame
    def cws_table2():
        return make_dataset_table("CW", "Canopy Water Storage Dataset", "cws_table2")

    @render.data_frame
    def swe_table2():
        return make_dataset_table("SWE", "SWE Dataset", "swe_table2")

    _table_funcs = {
        "ppt_table": ppt_table,
        "et_table": et_table,
        "q_table": q_table,
        "ppt_table2": ppt_table2,
        "et_table2": et_table2,
        "q_table2": q_table2,
        "sm_table2": sm_table2,
        "cws_table2": cws_table2,
        "swe_table2": swe_table2,
    }

    @reactive.calc
    def ppt_var_wb1():
        return sel_var(ppt_table, "P", "Precipitation Dataset")

    @reactive.calc
    def et_var_wb1():
        return sel_var(et_table, "ET", "ET Dataset", valid_et_vars())

    @reactive.calc
    def discharge_var_wb1():
        return sel_var(q_table, "Q", "Runoff Dataset")

    @reactive.calc
    def ppt_var_wb2():
        return sel_var(ppt_table2, "P", "Precipitation Dataset")

    @reactive.calc
    def et_var_wb2():
        return sel_var(et_table2, "ET", "ET Dataset", valid_et_vars())

    @reactive.calc
    def discharge_var_wb2():
        return sel_var(q_table2, "Q", "Runoff Dataset")

    @reactive.calc
    def sm_var_wb2():
        return sel_var(sm_table2, "SM", "Soil Moisture Dataset")

    @reactive.calc
    def cws_var_wb2():
        return sel_var(cws_table2, "CW", "Canopy Water Storage Dataset")

    @reactive.calc
    def swe_var_wb2():
        return sel_var(swe_table2, "SWE", "SWE Dataset")

    def build_wb1_collection(resample_scale):
        discharge = app_functions.get_var_params(
            discharge_var_wb1(), gee_dict, var_dict, unit_dict, scale_dict
        )
        precip = app_functions.get_var_params(
            ppt_var_wb1(), gee_dict, var_dict, unit_dict, scale_dict
        )
        et = app_functions.get_var_params(
            et_var_wb1(), gee_dict, var_dict, unit_dict, scale_dict
        )
        roi = roi_geom()
        i = str(
            pd.to_datetime(input.var_date()[0])
            .to_period("M")
            .start_time.strftime("%Y-%m-%d")
        )
        f = str(
            (
                pd.to_datetime(input.var_date()[1]).to_period("M").end_time
                + pd.offsets.Day(1)
            ).strftime("%Y-%m-%d")
        )
        discharge_data = (
            ee.ImageCollection(discharge["dst"])
            .select(discharge["var"])
            .filter(ee.Filter.date(i, f))
            .filterBounds(roi)
        )
        precip_data = (
            ee.ImageCollection(precip["dst"])
            .select(precip["var"])
            .filter(ee.Filter.date(i, f))
            .filterBounds(roi)
        )
        et_data = (
            ee.ImageCollection(et["dst"])
            .select(et["var"])
            .filter(ee.Filter.date(i, f))
            .filterBounds(roi)
        )
        et_data_corrected = et_data.map(lambda img: img.unmask(value=0)).map(
            lambda img: img.abs()
        )
        discharge_rename = app_functions.rename(discharge_data, "surface_runoff_sum")
        precip_rename = app_functions.rename(precip_data, "total_precipitation_sum")
        et_rename = app_functions.rename(et_data_corrected, "total_evaporation_sum")
        months, years = app_functions.generate_month_sequence(i, f)
        discharge_agg_monthly = app_functions.rename(
            app_functions.monthly_agg(months, years, discharge_rename, "sum"),
            "surface_runoff_sum",
        )
        precip_agg_monthly = app_functions.rename(
            app_functions.monthly_agg(months, years, precip_rename, "sum"),
            "total_precipitation_sum",
        )
        et_agg_monthly = app_functions.rename(
            app_functions.monthly_agg(months, years, et_rename, "sum"),
            "total_evaporation_sum",
        )
        discharge_month_convert = app_functions.unit_standardization_imgCol(
            discharge_agg_monthly, discharge["unit"], discharge["scale"]
        )
        precip_month_convert = app_functions.unit_standardization_imgCol(
            precip_agg_monthly, precip["unit"], precip["scale"]
        )
        et_month_convert = app_functions.unit_standardization_imgCol(
            et_agg_monthly, et["unit"], et["scale"]
        )
        crs = discharge_month_convert.first().projection().crs().getInfo()
        if crs is None:
            crs = "EPSG:4326"

        def resample_imgcol(img):
            return app_functions.smart_resample(img, crs, resample_scale)

        water_mask = app_functions.get_water_mask(roi)
        var_combined = (
            discharge_month_convert.map(resample_imgcol)
            .combine(precip_month_convert.map(resample_imgcol))
            .combine(et_month_convert.map(resample_imgcol))
            .map(lambda img: img.addBands(water_mask.rename("water_mask")))
        )

        return (
            var_combined.filter(
                ee.Filter.listContains("system:band_names", "surface_runoff_sum")
            )
            .map(app_functions.water_balance_basic)
            .filterBounds(roi)
            .sort("system:time_start")
        )

    @reactive.calc
    def wb1_dat_map():
        finest_res = min(
            app_functions.get_res_spat(ee.ImageCollection(gee_dict[et_var_wb1()])),
            app_functions.get_res_spat(ee.ImageCollection(gee_dict[ppt_var_wb1()])),
            app_functions.get_res_spat(
                ee.ImageCollection(gee_dict[discharge_var_wb1()])
            ),
        )
        return build_wb1_collection(resample_scale=finest_res)

    @reactive.calc
    def wb1_dat():
        return build_wb1_collection(resample_scale=scale())

    @reactive.calc
    def wb1_tot_img():
        wb_imgCol = wb1_dat_map()
        if wb_imgCol is None:
            return None
        return (
            wb_imgCol.select("water_budget_estimate")
            .sum()
            .clip(roi_geom())
            .divide(sel_years())
        )

    @reactive.calc
    def wb1_range():
        wb1_tot = wb1_tot_img()
        if wb1_tot is None:
            return None, None
        stats = wb1_tot.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=roi_geom(),
            scale=scale(),
            tileScale=4,
            bestEffort=True,
        ).getInfo()
        return stats.get("water_budget_estimate_min"), stats.get(
            "water_budget_estimate_max"
        )

    @render.plot
    @reactive.event(input.run_wb1)
    def wb1_colorbar():
        wb_min, wb_max = wb1_range()
        if wb_min is None or wb_max is None:
            return
        fig, ax = plt.subplots(figsize=(10, 0.8))
        _, vmin, vmax, palette = app_functions.diverging_vis_params(wb_min, wb_max)
        cmap = LinearSegmentedColormap.from_list("WB1", palette, N=255)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
            label="Average Annual Water Budget (m)",
        )
        fig.tight_layout(pad=0.5)
        return fig

    @render_widget
    @reactive.event(input.run_wb1)
    def wb1_map():
        notification_id = ui.notification_show(
            "Generating basic water budget map...", duration=None, type="message"
        )
        try:
            wb1_tot = wb1_tot_img()
            wb_min, wb_max = wb1_range()
            if wb1_tot is None or wb_min is None or wb_max is None:
                leaflet_map = Map(zoom=2, scroll_wheel_zoom=True)
                leaflet_map.add_layer(sat)
                return leaflet_map
            center, _ = coords()
            wb_vis, _, _, _ = app_functions.diverging_vis_params(wb_min, wb_max)
            try:
                layer = app_functions.ee_image_to_ipyleaflet(
                    wb1_tot, wb_vis, "WB1 Total"
                )
            except Exception:
                leaflet_map = Map(zoom=2, scroll_wheel_zoom=True)
                leaflet_map.add_layer(sat)
                return leaflet_map
            leaflet_map = Map(center=center, zoom=7, scroll_wheel_zoom=True)
            leaflet_map.add_layer(sat)
            leaflet_map.add_layer(labels)
            leaflet_map.add_layer(layer)
            leaflet_map.add_layer(sel_bas_layer())
            leaflet_map.layout.height = "750px"
            return leaflet_map
        finally:
            ui.notification_remove(notification_id)

    @reactive.calc
    def wb1_df():
        if not input.run_wb1():
            return None
        with reactive.isolate():
            water_budget_monthly_basic = wb1_dat()
            if water_budget_monthly_basic is None:
                return None
            set_scale = app_functions.get_res(i_date(), f_date(), roi_area())
            return app_functions.ee_collection_to_monthly_df(
                water_budget_monthly_basic,
                [
                    "water_budget_estimate",
                    "total_precipitation_sum",
                    "total_evaporation_sum",
                    "surface_runoff_sum",
                ],
                roi_geom(),
                set_scale,
            )

    @render_widget
    @reactive.event(input.run_wb1)
    def net_wb1():
        notification_id = ui.notification_show(
            "Calculating basic water budget timeseries...",
            duration=None,
            type="message",
        )
        try:
            daily_tp_wb1 = wb1_df()
            if daily_tp_wb1 is None:
                return px.line()
            values = daily_tp_wb1["water_budget_estimate"]
            wb1_mov_ave = values.rolling(window=12, min_periods=12, center=False).mean()
            bar_colors = ["#070281" if v >= 0 else "#DE7C19" for v in values]
            fig_wb1 = px.bar(
                daily_tp_wb1,
                x=daily_tp_wb1.index,
                y="water_budget_estimate",
                title="Basin-Averaged Δ Storage via Water Budget Calculation (m)",
                labels={"water_budget_estimate": "Mean Δ Storage (m)"},
            )
            fig_wb1.update_layout(
                xaxis_title="Month",
                legend_title="Δ Storage",
                title="Basic Water Budget",
            )
            fig_wb1.update_traces(marker_color=bar_colors)
            fig_wb1.add_traces(
                go.Scatter(
                    x=wb1_mov_ave.index,
                    y=wb1_mov_ave,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="12-month moving average",
                )
            )
            fig_wb1.update_xaxes(tickangle=45)
            return fig_wb1
        finally:
            ui.notification_remove(notification_id)

    @render_widget
    def gross_wb1():
        if input.wb1_tabs() != "Gross Water Budget":
            return px.bar()
        df = wb1_df()
        if df is None:
            return px.bar()
        notification_id = ui.notification_show(
            "Loading gross water budget...", duration=None, type="message"
        )
        try:
            indiv_var = df.rename(
                columns={
                    "total_precipitation_sum": "Precipitation",
                    "total_evaporation_sum": "Evapotranspiration",
                    "surface_runoff_sum": "Runoff",
                }
            )
            indiv_var["Runoff"] *= -1
            indiv_var["Evapotranspiration"] *= -1
            fig_detail = px.bar(
                indiv_var,
                x=df.index,
                y=["Precipitation", "Evapotranspiration", "Runoff"],
                title="Basin-Averaged Δ Storage, Water Budget Variables (m)",
                labels={
                    "x": "Month",
                    "value": "Mean Depth (m)",
                    "variable": "Water Budget Variables",
                },
                color_discrete_sequence=["#041467", "#31AA45", "#6EE7C3"],
            )
            fig_detail.add_hline(y=0)
            fig_detail.update_xaxes(tickangle=45)
            return fig_detail
        finally:
            ui.notification_remove(notification_id)

    def build_wb2_collection(resample_scale):
        discharge = app_functions.get_var_params(
            discharge_var_wb2(), gee_dict, var_dict, unit_dict, scale_dict
        )
        precip = app_functions.get_var_params(
            ppt_var_wb2(), gee_dict, var_dict, unit_dict, scale_dict
        )
        et = app_functions.get_var_params(
            et_var_wb2(), gee_dict, var_dict, unit_dict, scale_dict
        )
        sm = app_functions.get_var_params(
            sm_var_wb2(), gee_dict, var_dict, unit_dict, scale_dict
        )
        swe = app_functions.get_var_params(
            swe_var_wb2(), gee_dict, var_dict, unit_dict, scale_dict
        )
        cws = app_functions.get_var_params(
            cws_var_wb2(), gee_dict, var_dict, unit_dict, scale_dict
        )
        roi = roi_geom()
        i = str(
            pd.to_datetime(input.var_date()[0])
            .to_period("M")
            .start_time.strftime("%Y-%m-%d")
        )
        f = str(
            (
                pd.to_datetime(input.var_date()[1]).to_period("M").end_time
                + pd.offsets.Day(1)
            ).strftime("%Y-%m-%d")
        )
        f1m = str((pd.to_datetime(f) + pd.DateOffset(months=1)).strftime("%Y-%m-%d"))
        discharge_data = (
            ee.ImageCollection(discharge["dst"])
            .select(discharge["var"])
            .filter(ee.Filter.date(i, f))
            .filterBounds(roi)
        )
        precip_data = (
            ee.ImageCollection(precip["dst"])
            .select(precip["var"])
            .filter(ee.Filter.date(i, f))
            .filterBounds(roi)
        )
        et_data = (
            ee.ImageCollection(et["dst"])
            .select(et["var"])
            .filter(ee.Filter.date(i, f))
            .filterBounds(roi)
        )
        swe_data = (
            ee.ImageCollection(swe["dst"])
            .select(swe["var"])
            .filter(ee.Filter.date(i, f1m))
            .filterBounds(roi)
        )
        sm_data = (
            ee.ImageCollection(sm["dst"])
            .select(sm["var"])
            .filter(ee.Filter.date(i, f1m))
            .filterBounds(roi)
        )
        cws_data = (
            ee.ImageCollection(cws["dst"])
            .select(cws["var"])
            .filter(ee.Filter.date(i, f1m))
            .filterBounds(roi)
        )
        et_data_corrected = et_data.map(lambda img: img.unmask(value=0)).map(
            lambda img: img.abs()
        )
        discharge_rename = app_functions.rename(discharge_data, "surface_runoff_sum")
        precip_rename = app_functions.rename(precip_data, "total_precipitation_sum")
        et_rename = app_functions.rename(et_data_corrected, "total_evaporation_sum")
        swe_rename = app_functions.rename(swe_data, "swe_storage")
        sm_rename = app_functions.rename(sm_data, "volumetric_soil_water_layer_1")
        cws_rename = app_functions.rename(cws_data, "skin_reservoir_content")
        months, years = app_functions.generate_month_sequence(i, f)
        discharge_agg_monthly = app_functions.rename(
            app_functions.monthly_agg(months, years, discharge_rename, "sum"),
            "surface_runoff_sum",
        )
        precip_agg_monthly = app_functions.rename(
            app_functions.monthly_agg(months, years, precip_rename, "sum"),
            "total_precipitation_sum",
        )
        et_agg_monthly = app_functions.rename(
            app_functions.monthly_agg(months, years, et_rename, "sum"),
            "total_evaporation_sum",
        )
        swe_agg_monthly = app_functions.rename(
            app_functions.monthly_agg(months, years, swe_rename, "change", i, f),
            "swe_storage",
        )
        sm_agg_monthly = app_functions.rename(
            app_functions.monthly_agg(months, years, sm_rename, "change", i, f),
            "volumetric_soil_water_layer_1",
        )
        cws_agg_monthly = app_functions.rename(
            app_functions.monthly_agg(months, years, cws_rename, "change", i, f),
            "skin_reservoir_content",
        )
        discharge_month_convert = app_functions.unit_standardization_imgCol(
            discharge_agg_monthly, discharge["unit"], discharge["scale"]
        )
        precip_month_convert = app_functions.unit_standardization_imgCol(
            precip_agg_monthly, precip["unit"], precip["scale"]
        )
        et_month_convert = app_functions.unit_standardization_imgCol(
            et_agg_monthly, et["unit"], et["scale"]
        )
        swe_month_convert = app_functions.unit_standardization_imgCol(
            swe_agg_monthly, swe["unit"], swe["scale"]
        )
        sm_month_convert = app_functions.unit_standardization_imgCol(
            sm_agg_monthly, sm["unit"], sm["scale"]
        )
        cws_month_convert = app_functions.unit_standardization_imgCol(
            cws_agg_monthly, cws["unit"], cws["scale"]
        )
        crs = discharge_month_convert.first().projection().crs().getInfo()
        if crs is None:
            crs = "EPSG:4326"

        def resample_imgcol(img):
            return app_functions.smart_resample(img, crs, resample_scale)

        water_mask = app_functions.get_water_mask(roi)

        var_combined_gw = (
            sm_month_convert.map(resample_imgcol)
            .combine(cws_month_convert.map(resample_imgcol))
            .combine(swe_month_convert.map(resample_imgcol))
            .combine(discharge_month_convert.map(resample_imgcol))
            .combine(precip_month_convert.map(resample_imgcol))
            .combine(et_month_convert.map(resample_imgcol))
            .map(lambda img: img.addBands(water_mask.rename("water_mask")))
        )

        return (
            var_combined_gw.filter(
                ee.Filter.listContains("system:band_names", "surface_runoff_sum")
            )
            .map(app_functions.water_balance_advanced)
            .filterBounds(roi)
            .sort("system:time_start")
        )

    @reactive.calc
    def wb2_dat_map():
        finest_res = min(
            app_functions.get_res_spat(ee.ImageCollection(gee_dict[et_var_wb2()])),
            app_functions.get_res_spat(ee.ImageCollection(gee_dict[ppt_var_wb2()])),
            app_functions.get_res_spat(
                ee.ImageCollection(gee_dict[discharge_var_wb2()])
            ),
        )
        return build_wb2_collection(resample_scale=finest_res)

    @reactive.calc
    def wb2_dat():
        return build_wb2_collection(resample_scale=scale2())

    @reactive.calc
    def wb2_tot_img():
        wb_imgCol = wb2_dat_map()
        if wb_imgCol is None:
            return None
        return (
            wb_imgCol.select("water_budget_estimate")
            .sum()
            .clip(roi_geom())
            .divide(sel_years())
        )

    @reactive.calc
    def wb2_range():
        wb2_tot = wb2_tot_img()
        if wb2_tot is None:
            return None, None
        stats = wb2_tot.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=roi_geom(),
            scale=scale2(),
            tileScale=4,
            bestEffort=True,
        ).getInfo()
        return stats.get("water_budget_estimate_min"), stats.get(
            "water_budget_estimate_max"
        )

    @render.plot
    @reactive.event(input.run_wb2)
    def wb2_colorbar():
        wb_min, wb_max = wb2_range()
        if wb_min is None or wb_max is None:
            return
        fig, ax = plt.subplots(figsize=(10, 0.8))
        _, vmin, vmax, palette = app_functions.diverging_vis_params(wb_min, wb_max)
        cmap = LinearSegmentedColormap.from_list("WB2", palette, N=255)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
            label="Average Annual Water Budget (m)",
        )
        fig.tight_layout(pad=0.5)
        return fig

    @render_widget
    @reactive.event(input.run_wb2)
    def wb2_map():
        notification_id = ui.notification_show(
            "Generating advanced water budget map...", duration=None, type="message"
        )
        try:
            wb2_tot = wb2_tot_img()
            wb_min, wb_max = wb2_range()
            if wb2_tot is None or wb_min is None or wb_max is None:
                leaflet_map = Map(zoom=2, scroll_wheel_zoom=True)
                leaflet_map.add_layer(sat)
                return leaflet_map
            center, _ = coords()
            wb_vis, _, _, _ = app_functions.diverging_vis_params(wb_min, wb_max)
            try:
                layer = app_functions.ee_image_to_ipyleaflet(
                    wb2_tot, wb_vis, "WB2 Total"
                )
            except Exception:
                leaflet_map = Map(zoom=2, scroll_wheel_zoom=True)
                leaflet_map.add_layer(sat)
                return leaflet_map
            leaflet_map = Map(center=center, zoom=7, scroll_wheel_zoom=True)
            leaflet_map.add_layer(sat)
            leaflet_map.add_layer(labels)
            leaflet_map.add_layer(layer)
            leaflet_map.add_layer(sel_bas_layer())
            leaflet_map.layout.height = "750px"
            return leaflet_map
        finally:
            ui.notification_remove(notification_id)

    @reactive.calc
    def wb2_df():
        if not input.run_wb2():
            return None
        with reactive.isolate():
            water_budget_monthly_advanced = wb2_dat()
            if water_budget_monthly_advanced is None:
                return None
            return app_functions.ee_collection_to_monthly_df(
                water_budget_monthly_advanced,
                [
                    "water_budget_estimate",
                    "total_precipitation_sum",
                    "total_evaporation_sum",
                    "surface_runoff_sum",
                    "swe_storage",
                    "volumetric_soil_water_layer_1",
                    "skin_reservoir_content",
                ],
                roi_geom(),
                scale2(),
            )

    @render_widget
    @reactive.event(input.run_wb2)
    def net_wb2():
        notification_id = ui.notification_show(
            "Calculating advanced water budget...",
            duration=None,
            type="message",
        )
        try:
            daily_tp_wb2 = wb2_df()
            if daily_tp_wb2 is None:
                return px.line()
            values = daily_tp_wb2["water_budget_estimate"]
            wb2_mov_ave = values.rolling(window=12, min_periods=12, center=False).mean()
            bar_colors = ["#070281" if v >= 0 else "#DE7C19" for v in values]
            fig_wb2 = px.bar(
                daily_tp_wb2,
                x=daily_tp_wb2.index,
                y="water_budget_estimate",
                title="Basin-Averaged Monthly ∆Groundwater Storage via Water Budget Calculation (m)",
                labels={"water_budget_estimate": "Mean Δ Storage (m)"},
            )
            fig_wb2.update_layout(
                xaxis_title="Month",
                legend_title="Δ Storage",
                title="Advanced Water Budget",
            )
            fig_wb2.update_traces(marker_color=bar_colors)
            fig_wb2.add_traces(
                go.Scatter(
                    x=wb2_mov_ave.index,
                    y=wb2_mov_ave,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="12-month moving average",
                )
            )
            fig_wb2.update_xaxes(tickangle=45)
            return fig_wb2
        finally:
            ui.notification_remove(notification_id)

    @render_widget
    def gross_wb2():
        if input.wb2_tabs() != "Gross Water Budget":
            return px.bar()
        df = wb2_df()
        if df is None:
            return px.bar()
        notification_id = ui.notification_show(
            "Loading gross advanced water budget...", duration=None, type="message"
        )
        try:
            indiv_var = df.rename(
                columns={
                    "total_precipitation_sum": "Precipitation",
                    "swe_storage": "Snow Storage",
                    "total_evaporation_sum": "Evapotranspiration",
                    "surface_runoff_sum": "Runoff",
                    "volumetric_soil_water_layer_1": "Soil Moisture",
                    "skin_reservoir_content": "Canopy Storage",
                }
            )
            for col in [
                "Evapotranspiration",
                "Runoff",
                "Snow Storage",
                "Soil Moisture",
                "Canopy Storage",
            ]:
                indiv_var[col] *= -1
            fig_detail = px.bar(
                indiv_var,
                x=df.index,
                y=[
                    "Precipitation",
                    "Evapotranspiration",
                    "Runoff",
                    "Snow Storage",
                    "Soil Moisture",
                    "Canopy Storage",
                ],
                title="Basin-Averaged Δ Storage, Water Budget Variables (m)",
                labels={
                    "x": "Month",
                    "value": "Mean Monthly Depth (m)",
                    "variable": "Water Budget Variables",
                },
                color_discrete_sequence=[
                    "#041467",
                    "#1C8C42",
                    "#6EE7C3",
                    "#DA86D1",
                    "#CB8B5A",
                    "#E9F19E",
                ],
            )
            fig_detail.add_hline(y=0)
            fig_detail.update_xaxes(tickangle=45)
            return fig_detail
        finally:
            ui.notification_remove(notification_id)

    @render.text
    @reactive.event(input.load_grace)
    def grace_text():
        wb_start = datetime.strptime(i_date(), "%Y-%m-%d")
        wb_end = datetime.strptime(f_date(), "%Y-%m-%d")
        a_start = datetime.strptime("2004-01-01", "%Y-%m-%d")
        a_end = datetime.strptime("2009-12-31", "%Y-%m-%d")
        if wb_start > a_start and wb_end > a_end:
            return "Warning! Initial date range does not encompass the required period from 1-1-2004 through 12-31-2009. Please alter the START date."
        if wb_start < a_start and wb_end < a_end:
            return "Warning! Initial date range does not encompass the required period from 1-1-2004 through 12-31-2009. Please alter the END date."
        if wb_start > a_start and wb_end < a_end:
            return "Warning! Date range does not encompass the required period. Please alter both START and END dates."
        return "Please wait, processing data"

    @reactive.calc
    @reactive.event(input.load_grace)
    def grace_dat():
        wb_start = datetime.strptime(i_date(), "%Y-%m-%d")
        wb_end = datetime.strptime(f_date(), "%Y-%m-%d")
        a_start = datetime.strptime("2004-01-01", "%Y-%m-%d")
        a_end = datetime.strptime("2009-12-31", "%Y-%m-%d")
        if not (wb_start <= a_start and wb_end >= a_end):
            return None
        file_path = DATA_DIR / "GRCTellus.JPL.200204_202501.GLO.RL06.3M.MSCNv04CRI.nc"
        dataset = xr.open_dataset(file_path)
        scale_factor = dataset["scale_factor"].to_numpy()
        lwe_thickness_subset = (
            dataset["lwe_thickness"].sel(time=slice(i_date(), f_date())).values
        )
        tdim, ydim, xdim = lwe_thickness_subset.shape
        with open(default_geojson_path, "r") as file:
            sel_bas = json.load(file)
        sel_bas_local, id_field = load_selected_bas()
        if _is_custom_upload():
            geojson_dict = ee.FeatureCollection(
                [ee.Feature(_custom_aoi())]
            ).getInfo()
        else:
            fc, _ = active_fc_info()
            if id_field == "HYBAS_ID":
                bas_feat = fc.filter(
                    ee.Filter.inList("HYBAS_ID", app_functions.pull_ids(sel_bas))
                )
            else:
                bas_feat = fc.filter(
                    ee.Filter.inList(
                        id_field, app_functions.pull_huc_ids(sel_bas_local, id_field)
                    )
                )
            geojson_dict = bas_feat.getInfo()
        [marr, bbox] = app_functions.read_geojson_multilayers(
            xdim, ydim, str(input.Country()), geojson_dict
        )
        lat_vector = dataset.lat[:].copy()
        lon_vector = dataset.lon[:].copy()
        flip_lat = lat_vector[0] < lat_vector[-1]
        data_res = abs(lon_vector[2] - lon_vector[1])
        indexes_to_shift = int(360 / (2 * data_res))
        shift_lon = np.max(lon_vector) > 180
        area_weights = app_functions.area(lat_vector.data).reshape(1, -1)
        area_weights = np.transpose(area_weights)
        basin_area = np.multiply(area_weights, marr)
        region_mask = app_functions.shift_to_GRACE_orientation(
            shift_lon, flip_lat, marr, indexes_to_shift, 1
        )
        ma = app_functions.shift_to_GRACE_orientation(
            shift_lon, flip_lat, basin_area, indexes_to_shift, 1
        )
        lwe_thickness_scaled = np.empty(lwe_thickness_subset.shape)
        regional_timeseries = np.empty(lwe_thickness_scaled.shape[0])
        for time_index in range(len(regional_timeseries)):
            temp_lwe = lwe_thickness_subset[time_index, :, :]
            temp_lwe_scaled = np.multiply(temp_lwe, scale_factor)
            lwe_thickness_scaled[time_index, :, :] = temp_lwe_scaled
            mx = np.ma.masked_where(region_mask == 0, temp_lwe_scaled, copy=True)
            valid_weights = np.where(region_mask != 0, ma, 0)
            weight_sum = valid_weights.sum()
            if weight_sum == 0:
                regional_timeseries[time_index] = (
                    float(np.nanmean(temp_lwe_scaled[region_mask != 0]))
                    if np.any(region_mask != 0)
                    else float("nan")
                )
            else:
                regional_timeseries[time_index] = np.average(
                    mx.compressed(), weights=valid_weights[region_mask != 0]
                )
        timesteps = (
            dataset["time"]
            .sel(time=slice(i_date(), f_date()))
            .values.astype("datetime64[ms]")
        )
        timesteps = pd.to_datetime(timesteps)
        return timesteps, regional_timeseries

    @render_widget
    def grace_ts():
        if grace_dat() is not None:
            notification_id = ui.notification_show(
                "Loading GRACE timeseries...", duration=None, type="message"
            )
            try:
                timesteps, regional_timeseries = grace_dat()
                fig_grace = px.line(
                    x=timesteps,
                    y=regional_timeseries,
                    title=f"Corrected GRACE, TWS Anomaly: {input.Country()} basin",
                    labels={"water_budget_estimate": "Total (m)"},
                )
                fig_grace.update_layout(
                    xaxis_title="Month",
                    yaxis_title="TWS Thickness Anomalies (cm)",
                    legend_title="Δ GLDAS Storage Anomalies",
                    height=400,
                )
                fig_grace.update_xaxes(tickangle=45)
                fig_grace.add_hline(
                    y=0, line_width=1, line_dash="dash", line_color="black"
                )
                return fig_grace
            finally:
                ui.notification_remove(notification_id)
        return px.line()

    @reactive.calc
    def grace_map_dat():
        if input.load_grace():
            storage_dst = gee_dict["GRACE TWS"]
            storage_var = var_dict["GRACE TWS"]
            return (
                ee.ImageCollection(storage_dst)
                .select(storage_var)
                .filter(ee.Filter.date(i_date(), f_date()))
                .sum()
                .clip(roi_geom())
                .divide(100)
            )

    @reactive.calc
    def grace_max():
        if grace_map_dat() is not None:
            return ee.Number(
                grace_map_dat().reduceRegion(ee.Reducer.max(), roi_geom(), scale=5000)
            ).getInfo()["lwe_thickness"]

    @reactive.calc
    def grace_min():
        if grace_map_dat() is not None:
            return ee.Number(
                grace_map_dat().reduceRegion(ee.Reducer.min(), roi_geom(), scale=5000)
            ).getInfo()["lwe_thickness"]

    @render.plot
    def grace_colorbar():
        if input.load_grace():
            fig, ax = plt.subplots(figsize=(10, 0.5))
            cmap = LinearSegmentedColormap.from_list(
                "Grace", ["#c7522a", "#e5c185", "#fbf2c4", "#74a892", "#008585"], N=255
            )
            norm = mpl.colors.Normalize(vmin=grace_min(), vmax=grace_max())
            fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax,
                orientation="horizontal",
                label="Cumulative TWS Anomaly (m)",
            )
            return fig

    @render_widget
    def grace_map():
        if grace_map_dat() is not None:
            notification_id = ui.notification_show(
                "Loading GRACE map...", duration=None, type="message"
            )
            try:
                with reactive.isolate():
                    center, bb = coords()
                    grace_vis = {
                        "min": grace_min(),
                        "max": grace_max(),
                        "palette": [
                            "#c7522a",
                            "#e5c185",
                            "#fbf2c4",
                            "#74a892",
                            "#008585",
                        ],
                    }
                    m_grace = Map(center=center, zoom=8, scroll_wheel_zoom=True)
                    m_grace.add_layer(sat)
                    m_grace.add_layer(labels)
                    m_grace.add_layer(
                        app_functions.ee_image_to_ipyleaflet(
                            grace_map_dat(), grace_vis, "Mean GRACE TWS"
                        )
                    )
                    m_grace.add_layer(sel_bas_layer())
                    m_grace.layout.height = "650px"
                    return m_grace
            finally:
                ui.notification_remove(notification_id)
        empty_m = Map(center=(20, 0), zoom=2, scroll_wheel_zoom=True)
        empty_m.add_layer(sat)
        empty_m.layout.height = "650px"
        return empty_m


# Shiny requires an absolute path to www
app = App(app_ui, server, debug=False, static_assets=APP_DIR / "www")
