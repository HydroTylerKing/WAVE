import copy
import json
import math
import os
from datetime import datetime
from io import StringIO
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymannkendall as mk
import requests
import statsmodels.api as sm
import xarray as xr
from dateutil.relativedelta import relativedelta
from google.auth.exceptions import RefreshError, GoogleAuthError
from ipyleaflet import TileLayer
from loguru import logger
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pyproj import Geod
from shapely.geometry import Point
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf

# Anchor bundled data-file paths to this module's location so lookups do not
# depend on the current working directory or the host machine.
DATA_DIR = Path(__file__).resolve().parent / "data"

##############################################################################
#
#   Create helper functions for use in WAVE shiny application
#   The following functions will be read into the main application.
#   Copilot used to create function docstrings
#
##############################################################################


# Asset ID for GLDAS 2.1 sub daily dataset
_GLDAS_21_ID = "NASA/GLDAS/V021/NOAH/G025/T3H"


###################


def area(lats):
    """
    Compute weighted area for this region_mask.

    Parameters
    ----------
    lats : array-like of float
        Latitudes of 0.25° grid cell centres.

    Returns
    -------
    numpy.ndarray
        Cell areas in square meters.
    """
    geod = Geod(ellps="WGS84")
    dx = 0.25

    def cell_area(lat):
        lons = np.array([-dx, dx, dx, -dx])
        lats_poly = lat + np.array([-dx, -dx, dx, dx])
        return geod.polygon_area_perimeter(lons, lats_poly)[0]

    return np.array([cell_area(lat) for lat in lats])


###################


def area_by_class(image, roi):
    """
    Return a pixel frequency histogram for each land cover class within the roi.

    Parameters
    ----------
    image : ee.Image
        Classified land cover image.
    roi : ee.Geometry

    Returns
    -------
    ee.Dictionary
        Pixel counts keyed by class value.
    """
    pixel_count = image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=roi,
        scale=500,
        maxPixels=1e9,
    )

    return pixel_count


###################


def array_2_df(arr, list_of_bands):
    """
    Convert a GEE getRegion array to a pandas DataFrame with a datetime
    index and numeric band values.

    Parameters
    ----------
    arr : list of list
        Raw output from ``ee.ImageCollection.getRegion().getInfo()``.
    list_of_bands : list of str
        Band names to retain.

    Returns
    -------
    pandas.DataFrame
        Columns: ``["time", "datetime", *list_of_bands]``.
    """
    df = pd.DataFrame(arr)

    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    df = df[["longitude", "latitude", "time", *list_of_bands]].dropna()

    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors="coerce")

    df["datetime"] = pd.to_datetime(df["time"], unit="ms")

    df = df[["time", "datetime", *list_of_bands]]

    return df


###################


def autocorr_correction(df):
    """
    Test for autocorrelation within each variable using the Durbin-Watson
    method. If present, apply prewhitening to the variable prior to
    Mann-Kendall trend analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series variables to test. Each column needs at least 10
        non-NaN values.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with prewhitening applied to autocorrelated columns.
        First row is dropped due to the prewhitening lag.
    """
    columns_whiten = []  # Empty list to populate with cols with detected autocorrelation
    for col in df.columns:
        y = df[col].dropna()
        if len(y) < 10:  # Skip dw if too short
            continue
        X = np.arange(len(y))
        X = sm.add_constant(X)

        # Fit the ordinary least square method
        ols_res = OLS(y, X).fit()

        # Apply durbin-watson statistic on the ols residual
        dw_stat = durbin_watson(ols_res.resid)

        if 1.5 < dw_stat < 2.5:  # No sig autocorrelation detected for variable
            pass
        else:  # Positive or negative autocorrelation detected
            columns_whiten.append(col)

        df_pw = df.copy()

        for col in columns_whiten:
            y = df[col].dropna()
            prewhitened = prewhiten(y.values)
            df_pw.loc[y.index[1:], col] = prewhitened

        df_pw = df_pw.iloc[1:]
        return df_pw


###################


def bootstrap(array):
    """
    Find the mean and 95% CI of an array using 10000 bootstrap samples.

    Parameters
    ----------
    array : pandas.Series

    Returns
    -------
    estimate : float
    lower_ci : float
    upper_ci : float
    bootstrap_means : numpy.ndarray, shape (10000,)
    """
    response_variable = array.values

    # Number of bootstrap samples
    num_bootstrap_samples = 10000

    # Generate bootstrap samples and calculate the mean
    bootstrap_means = np.mean(
        np.random.choice(
            response_variable, (num_bootstrap_samples, len(response_variable))
        ),
        axis=1,
    )
    bootstrapped_df = pd.DataFrame({"bootstrap_mean": bootstrap_means})
    alpha = 0.05
    estimate = bootstrap_means.mean()
    lower_ci = np.quantile(bootstrapped_df, alpha / 2)
    upper_ci = np.quantile(bootstrapped_df, 1 - alpha / 2)
    return estimate, lower_ci, upper_ci, bootstrap_means


###################


def calc_centroid(fc):
    """
    Return the center coords and bbox of a GEE FeatureCollection.

    Parameters
    ----------
    fc : ee.FeatureCollection

    Returns
    -------
    cen : list of float
        ``[lat, lon]`` of the bounding box centre.
    bnd : numpy.ndarray
        ``[min_lon, min_lat, max_lon, max_lat]``.
    """
    # Convert gee feature collection to geodataframe
    feat_gdf = ee.data.computeFeatures(
        {"expression": fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"}
    )
    # Pull the coordinates of the extent of the geodataframe
    bnd = feat_gdf.total_bounds
    cen = [(bnd[1] + bnd[3]) / 2, (bnd[0] + bnd[2]) / 2]

    return cen, bnd


###################


def diverging_vis_params(
    min_val, max_val, neg_palette=None, pos_palette=None, div_palette=None
):
    """
    Return vis params dict and a (vmin, vmax, cmap) tuple for matplotlib
    colorbars depending on map colorbar values.

    Parameters
    ----------
    min_val : float
    max_val : float
    neg_palette : list of str, optional
        Defaults to dark-to-light orange.
    pos_palette : list of str, optional
        Defaults to light-to-dark blue.
    div_palette : list of str, optional
        Defaults to orange–white–blue.

    Returns
    -------
    vis_params : dict
    vmin : float
    vmax : float
    cmap : list of str
    """
    neg_palette = neg_palette or [
        "#DE7C19",
        "#F5C48A",
        "#FEF0DC",
    ]  # dark to light orange
    pos_palette = pos_palette or ["#DCE8FE", "#7BAAF7", "#070281"]  # light to dark blue
    div_palette = div_palette or ["#DE7C19", "#F5C48A", "white", "#7BAAF7", "#070281"]

    if max_val <= 0:
        # All negative- orange gradient
        return (
            {"min": min_val, "max": max_val, "palette": neg_palette},
            min_val,
            max_val,
            neg_palette,
        )
    elif min_val >= 0:
        # All positive- blue gradient
        return (
            {"min": min_val, "max": max_val, "palette": pos_palette},
            min_val,
            max_val,
            pos_palette,
        )
    else:
        # Positive and negative: orange and blue
        abs_max = max(abs(min_val), abs(max_val))
        return (
            {"min": -abs_max, "max": abs_max, "palette": div_palette},
            -abs_max,
            abs_max,
            div_palette,
        )


###################


def ee_collection_to_monthly_df(image_collection, bands, roi, scale):
    """
    Convert an ImageCollection to a monthly mean time series DataFrame for
    specified bands using a single getRegion call.

    Parameters
    ----------
    image_collection : ee.ImageCollection
    bands : list of str
    roi : ee.Geometry
    scale : int or float
        Spatial resolution in meters.

    Returns
    -------
    pandas.DataFrame
        Datetime-indexed DataFrame with one column per band. Empty if the
        collection contains no images.
    """

    def spatial_mean(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=scale, maxPixels=1e13
        )
        return ee.Feature(None, stats.set("time", img.get("system:time_start")))

    # Check for empty collection
    if image_collection.size().getInfo() == 0:
        return pd.DataFrame()
    # Pull all bands at once
    fc = image_collection.select(bands).map(spatial_mean)
    features = fc.getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in features])
    # Time index
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time").sort_index()

    return df


###################


def ee_image_to_ipyleaflet(ee_image, vis_params, name="EE Layer"):
    """
    Convert an Earth Engine Image to an ipyleaflet TileLayer.

    Parameters
    ----------
    ee_image : ee.Image
    vis_params : dict
    name : str, optional

    Returns
    -------
    ipyleaflet.TileLayer

    Raises
    ------
    TypeError
        If ``ee_image`` is not an ``ee.Image``.
    """

    if not isinstance(ee_image, ee.Image):
        raise TypeError("ee_image must be an ee.Image")

    map_id = ee.Image(ee_image).getMapId(vis_params)

    tile_url = map_id["tile_fetcher"].url_format

    layer = TileLayer(
        url=tile_url, name=name, attribution="Google Earth Engine", opacity=1.0
    )

    return layer


###################


def feat2json(features):
    """
    Convert a GEE FeatureCollection to GeoJSON.

    Parameters
    ----------
    features : ee.FeatureCollection

    Returns
    -------
    dict
        GeoJSON FeatureCollection.
    """
    feat_gdf = ee.data.computeFeatures(
        {"expression": features, "fileFormat": "GEOPANDAS_GEODATAFRAME"}
    )
    feat_json = feat_gdf.to_json()

    return json.loads(feat_json)


###################


def filter_valid_pixels(image_collection, roi, band, min_valid=1):
    """
    Remove images from a collection that contain fewer than min_valid
    unmasked pixels within the roi. Useful for dropping ET images that
    mask out hyper-arid regions entirely.

    Parameters
    ----------
    image_collection : ee.ImageCollection
    roi : ee.Geometry
    band : str
        Band name to check for valid (unmasked) pixels.
    min_valid : int, optional
        Minimum number of valid pixels required. Defaults to 1.

    Returns
    -------
    ee.ImageCollection
        Filtered collection containing only images with at least
        min_valid unmasked pixels over the roi.
    """

    def has_valid_pixels(image):
        valid = image.select(band).mask()
        pixel_count = valid.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=25000,
            maxPixels=1e9,
            bestEffort=True,
        )
        count = ee.Number(pixel_count.get(band)).int()
        return image.set("valid_pixel_count", count)

    return image_collection.map(has_valid_pixels).filter(
        ee.Filter.gte("valid_pixel_count", min_valid)
    )


###################


def generate_month_sequence(start_date, end_date):
    """
    Generate a list of integers of all months and years in a date range.

    Parameters
    ----------
    start_date : str
        ``"YYYY-MM-DD"`` format, inclusive.
    end_date : str
        ``"YYYY-MM-DD"`` format, exclusive.

    Returns
    -------
    months : list of str
        Zero-padded month strings, e.g. ``["01", "02", ...]``.
    years : list of str
        Corresponding four-character year strings.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(
        str(
            (pd.to_datetime(end_date) - pd.DateOffset(months=1))
            .to_period("M")
            .start_time.strftime("%Y-%m-%d")
        ),
        "%Y-%m-%d",
    )

    months = []
    years = []

    # Loop through each month between the start and end dates
    while start <= end:
        months.append(start.strftime("%m"))
        years.append(start.strftime("%Y"))
        start = start + relativedelta(months=1)

    return months, years


###################


def get_res(i_date, f_date, roi_area):
    """
    Calculate the coarsest spatial resolution needed to stay within GEE
    pixel limits for the time period and basin area.

    Parameters
    ----------
    i_date : str
        ``"YYYY-MM-DD"`` format.
    f_date : str
        ``"YYYY-MM-DD"`` format.
    roi_area : ee.Number
        Area of the region of interest in square meters.

    Returns
    -------
    float
        Resolution in meters rounded up to the nearest 500 m, or
        ``float("nan")`` if the required resolution exceeds 27 000 m.
    """
    months, years = generate_month_sequence(i_date, f_date)
    area = int(ee.Number(roi_area).getInfo())
    max_pix = (
        1048576 / 6.5
    )  # This is the number of max allowable pixels divided by 6 (number of bands in advanced water budget image collection)
    mo = int(len(months))

    res = math.sqrt((area * mo) / max_pix)

    logger.info(
        "For selected inputs, the finest possible resolution is "
        + str(math.ceil(res))
        + " m"
    )

    if res < 27000:
        set_scale = math.ceil(res / 500) * 500  # round up to nearest 500 meter
        logger.info("Scale set to " + str(set_scale) + " m")
    elif res > 27000:
        logger.warning(
            "Minimum resolution is above recommended, no scale was set. Please reduce number of selected basins or time period"
        )
        set_scale = float("nan")
    return set_scale


###################


def get_res_spat(collection):
    """
    Return the spatial resolution in meters of the first image in a GEE
    ImageCollection.

    Parameters
    ----------
    collection : ee.ImageCollection

    Returns
    -------
    float
        Nominal scale in meters.
    """
    # Get a sample image from the collection
    image = collection.first()
    # Get the scale of the image
    scale = image.projection().nominalScale()
    return scale.getInfo()


###################


def get_var_params(var_name, gee_dict, var_dict, unit_dict, scale_dict):
    """
    Return the GEE dataset, band name, unit, and scale for a variable.

    Parameters
    ----------
    var_name : str
    gee_dict : dict
    var_dict : dict
    unit_dict : dict
    scale_dict : dict

    Returns
    -------
    dict
        Keys: ``"dst"``, ``"var"``, ``"unit"``, ``"scale"``.
    """
    return {
        "dst": gee_dict[var_name],
        "var": var_dict[var_name],
        "unit": unit_dict[var_name],
        "scale": scale_dict[var_name],
    }


###################


def get_water_mask(roi):
    """
    Return a binary mask of permanent water bodies using JRC Global Surface
    Water. 1 = land, 0 = permanent water. Use to mask all WB images.

    Parameters
    ----------
    roi : ee.Geometry

    Returns
    -------
    ee.Image
        Binary mask clipped to ``roi``.
    """
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    # Pixels with water occurrence of more than 90% considered water
    permanent_water = jrc.select("occurrence").gte(90)
    water_mask = permanent_water.unmask(0).eq(0)  # 1=land, 0=water
    return water_mask.clip(roi)


###################


def has_valid_et_pixels(gee_dst, var_name, roi, i_date, f_date, scale=25000):
    """
    Return True if the ET ImageCollection has at least one unmasked pixel
    over the roi in the selected date range. Used to exclude datasets like
    MODIS ET that mask hyper-arid regions entirely.

    Parameters
    ----------
    gee_dst : str
        GEE asset path.
    var_name : str
        Band name to check.
    roi : ee.Geometry
    i_date : str
    f_date : str
    scale : int, optional
        Check scale in meters. Coarse is fine — we just need presence/absence.

    Returns
    -------
    bool
    """
    try:
        ic = (
            ee.ImageCollection(gee_dst)
            .select(var_name)
            .filter(ee.Filter.date(i_date, f_date))
            .filterBounds(roi)
        )
        if ic.size().getInfo() == 0:
            return False
        mosaic = ic.mosaic().clip(roi)
        count = mosaic.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=roi,
            scale=scale,
            maxPixels=1e8,
            bestEffort=True,
        ).getInfo()
        return (count.get(var_name, 0) or 0) > 0
    except ee.EEException as e:
        # gee-side failure
        logger.warning(f"GEE error checking pixel validity for {gee_dst} / {var_name}: {e}")
        return True
    except (RefreshError, GoogleAuthError):
        # Auth is broken
        raise


###################


def maskL8cloud(image):
    """
    Mask clouds and shadows from a Landsat 8 image using the QA_PIXEL band.

    Parameters
    ----------
    image : ee.Image
        Landsat 8 image with a ``QA_PIXEL`` band.

    Returns
    -------
    ee.Image
        Image with cloud and cloud-shadow pixels masked.
    """
    qa = image.select("QA_PIXEL")

    # Bits 3 and 5 are cloud shadow and cloud
    cloud_bit_mask = 1 << 3
    cloud_shadow_bit_mask = 1 << 4

    # Both flags set to zero
    mask = (
        qa.bitwiseAnd(cloud_shadow_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cloud_bit_mask).eq(0))
    )
    return image.updateMask(mask)


###################


def monthly_agg(months, years, imagecollection, func, i_date=None, f_date=None):
    """
    Aggregate an ImageCollection into monthly images, specifying aggregation
    type.

    Parameters
    ----------
    months : list of str
    years : list of str
    imagecollection : ee.ImageCollection
    func : {"sum", "mean", "change"}
    i_date : str, optional
        Required when ``func="change"``.
    f_date : str, optional
        Required when ``func="change"``.

    Returns
    -------
    ee.ImageCollection
        Monthly aggregated collection.

    Raises
    ------
    ValueError
        If ``func`` is not ``"sum"``, ``"mean"``, or ``"change"``.
    """
    if func not in {"change", "sum", "mean"}:
        raise ValueError(
            "Please specify monthly aggregating function as either mean, sum, or change"
        )

    if func == "change":
        try:
            asset_id = imagecollection.get("system:id").getInfo()
        except ee.EEException:
            asset_id = None

        if asset_id == _GLDAS_21_ID:
            return monthly_change_subDaily(imagecollection, i_date, f_date, None, None)
        else:
            return monthly_change_monthlyImgs(
                imagecollection, i_date, f_date, None, None
            )

    if func in ("sum", "mean"):
        # Get start from the months/years lists
        start_year = int(years[0])
        start_month = int(months[0])
        start = ee.Date.fromYMD(start_year, start_month, 1)
        n_months = len(months)
        month_offsets = ee.List.sequence(0, n_months - 1)

        if func == "sum":

            def make_monthly_img(offset):
                date = start.advance(offset, "month")
                return monthly_sum_ee(
                    imagecollection, date.get("year"), date.get("month")
                )
        else:

            def make_monthly_img(offset):
                date = start.advance(offset, "month")
                return monthly_mean_ee(
                    imagecollection, date.get("year"), date.get("month")
                )

        return ee.ImageCollection(month_offsets.map(make_monthly_img))


###################


def monthly_change(image_collection, year, month):
    """
    Calculate the monthly change of a variable by subtracting the first
    image of the specified month from the last. A positive value indicates
    an increase in storage. Used for storage variables such as SWE, soil
    moisture, and canopy water.

    Parameters
    ----------
    image_collection : ee.ImageCollection
    year : int
    month : int

    Returns
    -------
    ee.Image
        Single-band image named ``"monthly_chg"``.
    """
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, "month")

    prev_month_start = start_date.advance(-1, "month")

    current = image_collection.filterDate(start_date, end_date).mean()
    previous = image_collection.filterDate(prev_month_start, start_date).mean()

    chg = current.subtract(previous).rename("monthly_chg")
    return chg.set("system:time_start", start_date.millis()).setDefaultProjection(
        crs=image_collection.first().projection()
    )


###################


def monthly_change_monthlyImgs(imageCollection, i_date, f_date, bandName, roi):
    """
    Process monthly change variables (SWE, soil moisture, canopy water
    storage) when the GEE dataset presents data at a monthly timestep.

    Parameters
    ----------
    imageCollection : ee.ImageCollection
    i_date : str
    f_date : str
    bandName : str or None
    roi : ee.Geometry or None

    Returns
    -------
    ee.ImageCollection
        Monthly change images with band ``"monthly_chg"``.
    """
    # Filter by date and ROI
    end_inclusive = ee.Date(f_date).advance(2, "month")
    ic = imageCollection.filterDate(i_date, end_inclusive)
    if roi:
        ic = ic.filterBounds(roi)

    proj = imageCollection.first().projection()

    # Compute monthly change (next - current)
    lst = ic.toList(ic.size())
    n = ic.size()

    def monthly_diff(i):
        i = ee.Number(i)
        current = ee.Image(lst.get(i))
        nxt = ee.Image(lst.get(i.add(1)))
        start_date = ee.Date(current.date())
        return (
            current.subtract(nxt)  # was nxt.subtract(current)
            .rename("monthly_chg")
            .set("system:time_start", start_date.millis())
            .setDefaultProjection(proj)
        )

    mean_to_chg = ee.List.sequence(0, n.subtract(2))
    monthly_change_ic = ee.ImageCollection(mean_to_chg.map(monthly_diff))

    return monthly_change_ic


###################


def monthly_change_subDaily(imageCollection, i_date, f_date, bandName, roi):
    """
    Process monthly change variables (SWE, soil moisture, canopy water
    storage) when the GEE dataset presents data in a sub-daily timestep
    (GLDAS 2.1).

    Parameters
    ----------
    imageCollection : ee.ImageCollection
    i_date : str
    f_date : str
    bandName : str or None
    roi : ee.Geometry or None

    Returns
    -------
    ee.ImageCollection
        Monthly change images with band ``"monthly_chg"``.
    """
    # Filter by date and ROI
    start = ee.Date(i_date)
    end_inclusive = ee.Date(f_date).advance(2, "month")
    ic = imageCollection.filterDate(i_date, end_inclusive)
    if roi:
        ic = ic.filterBounds(roi)

    proj = imageCollection.first().projection()

    # Build monthly means
    numberMonths = end_inclusive.difference(start, "month").floor()
    monthStarts = ee.List.sequence(0, numberMonths.subtract(1)).map(
        lambda k: start.advance(k, "month")
    )

    def monthly_mean(month_start):
        month_start_date = ee.Date(month_start)
        month_end_date = month_start_date.advance(1, "month")
        month_ic = ic.filterDate(month_start_date, month_end_date)
        return ee.Algorithms.If(
            month_ic.size().gt(0),
            month_ic.mean()
            .set("system:time_start", month_start_date.millis())
            .setDefaultProjection(proj),
            None,
        )

    monthly_means_ic = ee.ImageCollection(monthStarts.map(monthly_mean)).filter(
        ee.Filter.notNull(["system:time_start"])
    )

    # Compute monthly change (next - current)
    lst = monthly_means_ic.toList(monthly_means_ic.size())
    n = monthly_means_ic.size()

    def monthly_diff(i):
        i = ee.Number(i)
        current = ee.Image(lst.get(i))
        nxt = ee.Image(lst.get(i.add(1)))
        start_date = ee.Date(current.date())
        return (
            current.subtract(nxt)
            .rename("monthly_chg")
            .set("system:time_start", start_date.millis())
            .copyProperties(current, current.propertyNames())
        )

    mean_to_chg = ee.List.sequence(0, n.subtract(2))
    monthly_change_ic = ee.ImageCollection(mean_to_chg.map(monthly_diff))

    return monthly_change_ic


###################


def monthly_mean(imageCollection, year, month):
    """
    Calculate the monthly mean.

    Parameters
    ----------
    imageCollection : ee.ImageCollection
    year : int
    month : int

    Returns
    -------
    ee.Image
    """
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, "month")
    monthly_collection = imageCollection.filterDate(start_date, end_date)
    return (
        monthly_collection.mean()
        .set("system:time_start", start_date.millis())
        .setDefaultProjection(crs=monthly_collection.first().projection())
    )


###################


def monthly_mean_ee(imageCollection, year, month):
    """
    Server-side monthly mean — arguments are ee.Number, not Python int.

    Parameters
    ----------
    imageCollection : ee.ImageCollection
    year : ee.Number
    month : ee.Number

    Returns
    -------
    ee.Image
    """
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, "month")
    return (
        imageCollection.filterDate(start_date, end_date)
        .mean()
        .set("system:time_start", start_date.millis())
        .setDefaultProjection(crs=imageCollection.first().projection())
    )


###################


def monthly_sum(imageCollection, year, month):
    """
    Calculate the monthly sum.

    Parameters
    ----------
    imageCollection : ee.ImageCollection
    year : int
    month : int

    Returns
    -------
    ee.Image
    """
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, "month")
    monthly_collection = imageCollection.filterDate(start_date, end_date)
    return (
        monthly_collection.sum()
        .set("system:time_start", start_date.millis())
        .setDefaultProjection(crs=monthly_collection.first().projection())
    )


###################


def monthly_sum_ee(imageCollection, year, month):
    """
    Server-side monthly sum — arguments are ee.Number, not Python int.

    Parameters
    ----------
    imageCollection : ee.ImageCollection
    year : ee.Number
    month : ee.Number

    Returns
    -------
    ee.Image
    """
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, "month")
    return (
        imageCollection.filterDate(start_date, end_date)
        .sum()
        .set("system:time_start", start_date.millis())
        .setDefaultProjection(crs=imageCollection.first().projection())
    )


###################


def prewhiten(series):
    """
    Perform prewhitening to reduce autocorrelation and the occurrence of
    Type I errors.

    Parameters
    ----------
    series : array-like of float

    Returns
    -------
    numpy.ndarray
        Prewhitened series of length ``len(series) - 1``.
    """
    series = np.array(series)
    lag1_ac = acf(series, nlags=1, fft=False)[1]
    pw_series = series[1:] - lag1_ac * series[:-1]
    return pw_series


###################


def pull_altim_dahiti(roi, i_date=None, f_date=None, api_key=None):
    """
    Search the DAHITI API for altimetric water level data in the bbox of
    the roi and return a dict of time series, GeoJSON of water body
    locations, and list of names.

    Parameters
    ----------
    roi : ee.Geometry
    i_date : str, optional
    f_date : str, optional
    api_key : str, optional
        DAHITI API key. Falls back to the DAHITI_API_KEY environment variable.

    Returns
    -------
    alt_dict : dict of {str: pandas.DataFrame}
    wat_json : dict
        GeoJSON FeatureCollection of water body locations.
    alt_names : pandas.Series of str
    """
    _api_key = api_key or os.environ.get("DAHITI_API_KEY", "")
    # Suppress SSL warnings — DAHITI uses a certificate that may not validate
    # behind corporate/DOI network proxies
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    roi_coords = roi.bounds().getInfo()["coordinates"][0]
    coords = pd.DataFrame(
        columns=["Longitude", "Latitude"], index=range(len(roi_coords))
    )
    for ind in range(len(roi_coords)):
        coords.loc[ind, "Longitude"] = roi_coords[ind][0]
        coords.loc[ind, "Latitude"] = roi_coords[ind][1]

    url = "https://dahiti.dgfi.tum.de/api/v2/list-targets/"
    args = {}
    args["api_key"] = _api_key
    args["min_lon"] = coords["Longitude"].min()
    args["max_lon"] = coords["Longitude"].max()
    args["min_lat"] = coords["Latitude"].min()
    args["max_lat"] = coords["Latitude"].max()
    response = requests.post(url, json=args, verify=False)

    if response.status_code == 200:
        data = json.loads(response.text)
    else:
        logger.error(response.text)
        logger.error(response.status_code)
        return {}, {}, pd.Series([], dtype=str)

    ids = [x["dahiti_id"] for x in data["data"]]
    names = [x["target_name"] for x in data["data"]]
    lat = [x["latitude"] for x in data["data"]]
    long = [x["longitude"] for x in data["data"]]
    data_df = {"Latitude": lat, "Longitude": long, "Name": names, "ID": ids}
    data_df = pd.DataFrame(data_df)

    geometry = [Point(xy) for xy in zip(data_df["Longitude"], data_df["Latitude"])]
    wat_gdf = gpd.GeoDataFrame(data_df, geometry=geometry, crs="EPSG:4326")
    wat_str = wat_gdf.to_json()
    wat_json = json.loads(wat_str)

    url = "https://dahiti.dgfi.tum.de/api/v2/download-water-level/"

    alt_dict = {}
    alt_names = data_df["Name"].reset_index(drop=True)

    start_dt = pd.to_datetime(i_date) if i_date else None
    end_dt = pd.to_datetime(f_date) if f_date else None

    for ind in range(len(data_df["ID"])):
        wb = str(data_df["Name"][ind])
        wb_id = data_df["ID"][ind]
        args = {}
        args["api_key"] = _api_key
        args["dahiti_id"] = wb_id
        args["format"] = "csv"
        response = requests.get(url, params=args, verify=False)

        if response.status_code == 200:
            resp_csv = StringIO(response.text)
            df = pd.read_csv(resp_csv, sep=";")
            df["datetime"] = pd.to_datetime(df["datetime"])

            if start_dt is not None:
                df = df[df["datetime"] >= start_dt]
            if end_dt is not None:
                df = df[df["datetime"] <= end_dt]

            if not df.empty:
                alt_dict[str(wb)] = df
        else:
            logger.warning(
                f"Failed to fetch water level for {wb} (ID {wb_id}): {response.status_code}"
            )

    return alt_dict, wat_json, alt_names


###################


def pull_glolakes(hl_id, hydrolakes):
    """
    Load GloLakes storage data from local NetCDF files, match lakes to
    HydroLakes IDs within the roi, and return a GEE FeatureCollection and
    monthly storage time series.

    Parameters
    ----------
    hl_id : list of int
        HydroLakes IDs within the roi.
    hydrolakes : ee.FeatureCollection

    Returns
    -------
    glo_lakes : ee.FeatureCollection
    storage : pandas.DataFrame
        Month-end indexed, lake IDs as columns. Empty if no data found.
    ids : pandas.Series of int
    """

    nc_files = [
        DATA_DIR / "Global_Lake_Absolute_Storage_LandsatPlusGREALM (1984-present).nc",
        DATA_DIR / "Global_Lake_Absolute_Storage_LandsatPlusICESat2_(1984-present).nc",
        DATA_DIR / "Global_Lake_Absolute_Storage_LandsatPlusSentinel2 (1984-present).nc",
    ]

    datasets = []
    for fp in nc_files:
        try:
            ds = xr.open_dataset(fp)
            datasets.append(ds)
        except FileNotFoundError:
            logger.warning(f"File not found, skipping — {fp}")

    if not datasets:
        logger.error("No GloLakes NetCDF files could be loaded.")
        empty_fc = hydrolakes.filter(ee.Filter.eq("Hylak_id", -1))
        return empty_fc, pd.DataFrame(), pd.Series([], dtype=int)

    merged = datasets[0]
    for ds in datasets[1:]:
        merged = xr.merge([merged, ds], compat="override")

    # Get lake_storage and normalise ids to int
    if "lake_storage" not in merged:
        logger.error("'lake_storage' variable not found in merged dataset.")
        logger.debug(f"Available variables: {list(merged.data_vars)}")
        empty_fc = hydrolakes.filter(ee.Filter.eq("Hylak_id", -1))
        return empty_fc, pd.DataFrame(), pd.Series([], dtype=int)

    data = merged["lake_storage"]

    # Identify the id dimension name
    id_dim = None
    for candidate in ["ID", "id", "lake_id", "hylak_id"]:
        if candidate in data.coords:
            id_dim = candidate
            break
    if id_dim is None:
        logger.error(
            f"Cannot find ID coordinate. Available coords: {list(data.coords)}"
        )
        empty_fc = hydrolakes.filter(ee.Filter.eq("Hylak_id", -1))
        return empty_fc, pd.DataFrame(), pd.Series([], dtype=int)

    glolake_ids_raw = data.coords[id_dim].values.tolist()
    glolake_ids_int = [int(i) for i in glolake_ids_raw]
    glolake_id_set = set(glolake_ids_int)

    hl_ids_int = [int(i) for i in hl_id]

    comm_ids = [i for i in hl_ids_int if i in glolake_id_set]

    if not comm_ids:
        logger.warning(
            "No HydroLake IDs match any GloLakes ID — no storage data for this ROI."
        )
        empty_fc = hydrolakes.filter(ee.Filter.eq("Hylak_id", -1))
        return empty_fc, pd.DataFrame(), pd.Series([], dtype=int)

    # Get timeseries for matching lakes
    int_to_raw = {int(r): r for r in glolake_ids_raw}

    records = {}
    for lake_id in comm_ids:
        raw_id = int_to_raw[lake_id]
        series = data.sel({id_dim: raw_id}).values
        records[str(lake_id)] = series
        n_valid = int(np.sum(~np.isnan(series)))
        logger.info(
            f"Lake {lake_id}: {len(series)} timesteps, {n_valid} non-NaN values"
        )

    time_index = pd.to_datetime(data.coords["time"].values)
    filt_store = pd.DataFrame(records, index=time_index)

    tmp = filt_store.dropna(how="all")
    tmp2 = tmp.dropna(axis=1, how="all")

    if tmp2.empty:
        hl_filt = ee.Filter.inList("Hylak_id", comm_ids)
        glo_lakes = hydrolakes.filter(hl_filt)
        return glo_lakes, pd.DataFrame(), pd.Series(comm_ids, dtype=int)

    storage = tmp2.resample("ME").mean()
    storage.index = storage.index.to_period("M").to_timestamp()

    dat_ids = [int(c) for c in storage.columns]

    hl_filt = ee.Filter.inList("Hylak_id", dat_ids)
    glo_lakes = hydrolakes.filter(hl_filt)
    ids = pd.Series(dat_ids, dtype=int)

    logger.info(f"Returning {len(dat_ids)} lakes/rivers with storage data: {dat_ids}")
    return glo_lakes, storage, ids


###################


def pull_huc_ids(sel_bas, id_field):
    """
    Pull HUC IDs saved in selected_basins.geojson.

    Parameters
    ----------
    sel_bas : dict
        Parsed GeoJSON FeatureCollection.
    id_field : str
        Property name holding the HUC ID.

    Returns
    -------
    list
    """
    ids = []
    for feat in sel_bas.get("features", []):
        v = feat["properties"].get(id_field)
        if v is not None:
            ids.append(v)
    return ids


###################


def pull_ids(sel_bas):
    """
    Pull the HYBAS ID saved in selected_basins.geojson.

    Parameters
    ----------
    sel_bas : dict
        Parsed GeoJSON FeatureCollection with ``"HYBAS_ID"`` properties.

    Returns
    -------
    list
    """
    bas_id = []

    if "features" in sel_bas and len(sel_bas["features"]) >= 1:
        for feature in sel_bas["features"]:
            hybas_id = feature["properties"].get("HYBAS_ID")

            if hybas_id is not None:
                bas_id.append(hybas_id)

    return bas_id


###################


def read_geojson_multilayers(xdim, ydim, sel_ctry, geojson_dict):
    """
    Read a multilayer GeoJSON file into a binary mask array aligned to the
    GRACE grid. Source: https://github.com/podaac/the-coding-club/blob/main/n

    Parameters
    ----------
    xdim : int
        Number of longitude grid columns.
    ydim : int
        Number of latitude grid rows.
    sel_ctry : str
        OGR layer name.
    geojson_dict : dict
        Parsed GeoJSON FeatureCollection to rasterise.

    Returns
    -------
    marr : numpy.ndarray of uint8, shape (ydim, xdim)
        Binary raster mask.
    bbox : list of float
        ``[min_lon, min_lat, max_lon, max_lat]`` of the first feature.
    """
    from shapely.geometry import shape

    # Collect all geometries from the GeoJSON
    geometries = []
    for feat_item in geojson_dict["features"]:
        geom = shape(feat_item["geometry"])
        geometries.append((geom, 1))

    # Define the global transform matching the GRACE grid
    # Grid spans -180 to 180 in x, -90 to 90 in y
    transform = from_bounds(-180.0, -90.0, 180.0, 90.0, xdim, ydim)

    # Rasterize all geometries onto the grid
    marr = rasterize(
        geometries,
        out_shape=(ydim, xdim),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # Get bounding box from the first feature
    first_geom = shape(geojson_dict["features"][0]["geometry"])
    bounds = first_geom.bounds  # (minx, miny, maxx, maxy)
    bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]

    return marr, bbox


###################


def rename(imageCollection, variable):
    """
    Rename all images in a GEE ImageCollection to band name.

    Parameters
    ----------
    imageCollection : ee.ImageCollection
    variable : str
        New band name.

    Returns
    -------
    ee.ImageCollection
    """
    return imageCollection.map(lambda img: img.rename(variable))


###################


def seas_mk(df):
    """
    Generate an estimate of seasonal Mann-Kendall trend analysis for each
    individual variable, as well as the water budget estimate. Modified from
    Hussain and Mahmud (2019) pyMannKendall package.

    Parameters
    ----------
    df : pandas.Series
        Monthly time series with at least 8 non-NaN observations.

    Returns
    -------
    str
        ``"increasing"``, ``"decreasing"``, ``"no trend"``, or
        ``"invalid"`` if the series is too short or has zero variance.
    """
    series = df.dropna()
    if (
        len(series)
        < 8  # Need enough data - requiring 8 points (https://www.statisticshowto.com/mann-kendall-trend-test/)
        or series.nunique() <= 1
        or series.var() == 0
    ):
        return "invalid"
    df_array = series.to_numpy()
    mk_result = mk.seasonal_test(df_array, alpha=0.05, period=12)
    if mk_result.trend != "no trend":
        fig = px.line(
            series,
            x=series.index,
            y=series,
            title=str(series.name + " (m)"),
            labels={
                "x": "Month",
                "value": "Variable Change (m)",
                "variable": "Water Budget Variables",
            },
        )
        trend_line = np.arange(len(series)) * mk_result.slope + mk_result.intercept
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=trend_line,
                mode="lines",
                name="Seas. MK Trend",
                showlegend=False,
            )
        )
        fig.add_hline(y=0)
        fig.update_xaxes(tickangle=45)
        slope_round = str(round(mk_result.slope, 3))
        fig.add_annotation(
            x=series.index[0], y=trend_line[1], text="Slope: " + slope_round
        )
        fig.show()
    else:
        pass
    return mk_result.trend


###################


def shift_to_GRACE_orientation(
    shift_lon, flip_lat, data_array, indexes_to_shift, axis_no
):
    """
    Reorient grid to match GRACE orientation.
    Source: https://github.com/podaac/the-coding-club/blob/main/n

    Parameters
    ----------
    shift_lon : bool
    flip_lat : bool
    data_array : numpy.ndarray
    indexes_to_shift : int
    axis_no : int

    Returns
    -------
    numpy.ndarray
    """
    if shift_lon:
        temp_1a = np.roll(data_array, indexes_to_shift, axis=axis_no)
    else:
        temp_1a = copy(data_array)
    if flip_lat:
        reoriented_grid = np.flipud(temp_1a)
    else:
        reoriented_grid = copy(temp_1a)
    return reoriented_grid


###################


def smart_resample(img, crs, target_scale):
    """
    Resample img to target_scale in crs, choosing the correct GEE method
    based on direction (coarser or finer).

    Parameters
    ----------
    img : ee.Image
    crs : str or None
        EPSG string. Defaults to ``"EPSG:4326"`` if ``None``.
    target_scale : int or float
        Target resolution in meters.

    Returns
    -------
    ee.Image
    """
    if crs is None:
        crs = "EPSG:4326"
    native_scale = img.projection().nominalScale()
    target_proj = ee.Projection(crs).atScale(target_scale)
    coarsened = img.reduceResolution(
        reducer=ee.Reducer.mean(), maxPixels=65536
    ).reproject(target_proj)
    reprojected = img.reproject(target_proj)
    return ee.Image(
        ee.Algorithms.If(native_scale.lt(target_scale), coarsened, reprojected)
    )


###################

# def surface_water_change(roi, i_date, f_date, gee_dst, var_name):
#     """
#     Compute monthly change in surface water extent. Returns an
#     ImageCollection of monthly surface water change (unitless binary to m
#     via 0.5m depth).

#     Parameters
#     ----------
#     roi : ee.Geometry
#     i_date : str
#     f_date : str
#     gee_dst : str
#         GEE asset path. Checked for ``"JRC"`` to select binarisation logic.
#     var_name : str

#     Returns
#     -------
#     ee.ImageCollection
#         Monthly change images with band ``"surface_water_change"`` in meters.
#     """

#     end_inclusive = ee.Date(f_date).advance(1, "month")
#     ic = (
#         ee.ImageCollection(gee_dst)
#         .select(var_name)
#         .filterDate(i_date, end_inclusive)
#         .filterBounds(roi)
#     )

#     def to_binary(img):
#         # JRC: 2=water, 1=not water, 0=no data
#         # OPERA S1 (DSWX/L3_V1/S1): 1=water, 0=not water, 250/251=masked
#         # OPERA HLS (DSWX/L3_V1/HLS): 1=water, 0=not water, 252/253/254=masked
#         is_jrc = ee.String(gee_dst).index("JRC").gte(0)
#         jrc_water = img.eq(2)
#         # Both OPERA datasets: class 1 = water, mask out fill/QA classes (>=250)
#         opera_water = img.eq(1).And(img.lt(250))
#         binary = ee.Image(ee.Algorithms.If(is_jrc, jrc_water, opera_water))
#         return binary.unmask(0).rename("surface_water_change").copyProperties(
#             img, img.propertyNames()
#         )

#     binary = ic.map(to_binary)
#     n = binary.size()

#     def make_diffs():
#         lst = binary.toList(n)
#         def monthly_diff(i):
#             i = ee.Number(i)
#             current = ee.Image(lst.get(i))
#             nxt = ee.Image(lst.get(i.add(1)))
#             start_date = ee.Date(current.date())
#             return (
#                 nxt.subtract(current)
#                 .rename("surface_water_change")
#                 .multiply(0.5)
#                 .set("system:time_start", start_date.millis())
#                 .setDefaultProjection(binary.first().projection())
#             )
#         diffs = ee.List.sequence(0, n.subtract(2))
#         return ee.ImageCollection(diffs.map(monthly_diff)).filterBounds(roi)

#     empty = ee.ImageCollection([
#         ee.Image.constant(0)
#         .rename("surface_water_change")
#         .set("system:time_start", ee.Date(i_date).millis())
#         .clip(roi)
#     ])

#     return ee.ImageCollection(ee.Algorithms.If(n.gt(1), make_diffs(), empty))

###################


def unit_standardization_img(image, variable, scale):
    """
    Standardize units for all variables used in the water balance
    calculation.

    Parameters
    ----------
    image : ee.Image
    variable : str
        Unit string. Supported: ``"cm"``, ``"mm"``, ``"kg/m^2 8day"``,
        ``"kg/m^2"``, ``"volume fraction/100"``, ``"volume fraction/7"``,
        ``"%"``. Anything else is treated as meters.
    scale : float
        Dataset-specific scaling factor.

    Returns
    -------
    ee.Image
        Pixel values converted to meters and multiplied by ``scale``.
    """
    unit = variable
    scaler = scale

    if unit == "cm":
        unit_convert = image.divide(100)
        logger.info("cm to m")

    elif unit == "mm":
        unit_convert = image.divide(1000)
        logger.info("mm to m")

    elif unit == "kg/m^2 8day":
        unit_convert = image.divide(1000)
        logger.info("kg/m^2 8day to m")

    elif unit == "kg/m^2":
        unit_convert = image.divide(1000)
        logger.info("kg/m^2 to m")

    elif unit == "volume fraction/100":
        unit_convert = image
        logger.info("Volume fraction to m (100 cm depth)")
        unit_convert = image

    elif unit == "volume fraction/7":
        unit_convert = image.multiply(0.07)
        logger.info("Volume fraction to m (7 cm depth)")
        unit_convert = image

    elif unit == "%":
        unit_convert = image
        logger.info("Unit is percentage: Not converted")
        unit_convert = image

    else:
        logger.info("Variable unit is m: No action needed")
        unit_convert = image

    return unit_convert.multiply(scaler)  # convert by data scaler


###################


def unit_standardization_imgCol(imageCollection, variable, scale):
    """
    Standardize an ImageCollection to meters.

    Parameters
    ----------
    imageCollection : ee.ImageCollection
    variable : str
        Unit string. Supported: ``"cm"``, ``"mm"``, ``"kg/m^2 8day"``,
        ``"kg/m^2"``, ``"volume fraction/100"``, ``"volume fraction/7"``,
        ``"%"``. Anything else is treated as meters.
    scale : float
        Dataset-specific scaling factor.

    Returns
    -------
    ee.ImageCollection
        Collection with pixel values converted to meters and multiplied by
        ``scale``. Image properties are preserved.
    """
    unit = variable
    scaler = scale

    def unit_convert_col(image):
        if unit == "cm":
            unit_convert = image.divide(100)
            logger.info("cm to m")

        elif unit == "mm":
            unit_convert = image.divide(1000)
            logger.info("mm to m")

        elif unit == "kg/m^2 8day":
            unit_convert = image.divide(1000)
            logger.info("kg/m^2 8day to m")

        elif unit == "kg/m^2":
            unit_convert = image.divide(1000)
            logger.info("kg/m^2 to m")

        elif unit == "volume fraction/100":
            unit_convert = image
            logger.info("Volume fraction to m (100 cm depth)")

        elif unit == "volume fraction/7":
            unit_convert = image.multiply(0.07)
            logger.info("Volume fraction to m")

        elif unit == "%":
            unit_convert = image
            logger.info("Unit is percentage: Not converted")

        else:
            logger.info("Variable unit is m: No action needed")
            unit_convert = image

        return unit_convert.multiply(scaler).copyProperties(
            image, image.propertyNames()
        )

    return imageCollection.map(unit_convert_col)


###################


def variable_min_max(image, roi, band_name, palette, scale=5000):
    """
    Compute a symmetric min/max range centered on zero for an image, for
    colorbar viz.

    Parameters
    ----------
    image : ee.Image
    roi : ee.Geometry
    band_name : str
    palette : list of str
    scale : int or float, optional
        Defaults to ``5000``.

    Returns
    -------
    dict
        GEE vis params with keys ``"min"``, ``"max"``, ``"palette"``.
        Range is symmetric about zero.
    """
    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=roi,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,  # Reduces resolution if limit hit
    )

    min_val = ee.Number(stats.get(f"{band_name}_min")).abs()
    max_val = ee.Number(stats.get(f"{band_name}_max")).abs()
    greater_value = min_val.max(max_val)
    min_range = greater_value.multiply(2).ceil().divide(2).multiply(-1)
    max_range = greater_value.multiply(2).ceil().divide(2)

    return {
        "min": min_range.getInfo(),
        "max": max_range.getInfo(),
        "palette": list(palette),
    }


###################
def water_balance_advanced(image):
    """
    Compute the advanced water budget from an image's bands and add the
    result as a new band, masking permanent water bodies.

    Expression: ``((precip - swe) - et) - runoff) - soilmoist - canopyws``

    Parameters
    ----------
    image : ee.Image
        Must contain bands: ``"total_precipitation_sum"``,
        ``"swe_storage"``, ``"total_evaporation_sum"``,
        ``"surface_runoff_sum"``, ``"volumetric_soil_water_layer_1"``,
        ``"skin_reservoir_content"``, ``"water_mask"``.

    Returns
    -------
    ee.Image
        Input image with added band ``"water_budget_estimate"``.
    """
    combined = image.expression(
        "(((precip - swe) - et) - runoff) - soilmoist - canopyws",
        {
            "precip": image.select("total_precipitation_sum"),
            "swe": image.select("swe_storage"),
            "et": image.select("total_evaporation_sum"),
            "runoff": image.select("surface_runoff_sum"),
            "soilmoist": image.select("volumetric_soil_water_layer_1"),
            "canopyws": image.select("skin_reservoir_content"),
        },
    )
    water_mask = image.select("water_mask")
    return image.addBands(
        combined.rename("water_budget_estimate").updateMask(water_mask)
    ).copyProperties(image, image.propertyNames())


###################


def water_balance_basic(image):
    """
    Compute the basic water budget from an image's bands and add the result
    as a new band, masking permanent water bodies.

    Expression: ``(precip - et) - runoff``

    Parameters
    ----------
    image : ee.Image
        Must contain bands: ``"total_precipitation_sum"``,
        ``"total_evaporation_sum"``, ``"surface_runoff_sum"``,
        ``"water_mask"``.

    Returns
    -------
    ee.Image
        Input image with added band ``"water_budget_estimate"``.
    """
    combined = image.expression(
        "(precip - et) - runoff",
        {
            "precip": image.select("total_precipitation_sum"),
            "et": image.select("total_evaporation_sum"),
            "runoff": image.select("surface_runoff_sum"),
        },
    )
    water_mask = image.select("water_mask")
    return image.addBands(
        combined.rename("water_budget_estimate").updateMask(water_mask)
    ).copyProperties(image, image.propertyNames())


##############################################################################
#
#   SWOT water surface elevation helpers
#
#   These functions support the "SWOT Water Levels" tab. River reach
#   footprints (for the clickable map) come from NASA's Earthdata GIS ArcGIS
#   REST feature service. Water-surface-elevation time series for a selected
#   reach or prior lake come from the public PO.DAAC Hydrocron API. Neither
#   endpoint requires an Earthdata Login.
#
##############################################################################


# NASA Earthdata GIS ArcGIS REST feature service for SWOT Version D river
# reaches. Layer id is 1 (not 0). Only a river-reach service is published;
# there is no equivalent lake feature service, so lakes are looked up by
# lake_id via Hydrocron.
# The "_BASIC" service exposes a reduced attribute set and is optimized for
# fast spatial querying/visualization; the full-attribute service is far too
# slow for interactive use (it times out on modest bounding boxes).
SWOT_REACH_FEATURE_SERVICE = (
    "https://gis.earthdata.nasa.gov/portal/rest/services/"
    "C3233944997_POCLOUD/SWOT_L2_HR_RiverSP_reach_vD_BASIC/FeatureServer/1/query"
)

# Public PO.DAAC Hydrocron time-series API (no authentication required).
HYDROCRON_TIMESERIES_URL = (
    "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
)

# SWOT products use large-magnitude fill values for missing measurements
# (e.g. -999999999999). Real water surface elevations on Earth fall well
# within +/- 1e7 m, so any |value| at or above this threshold is treated as
# no-data. (The SWOT fill value -9.99e11 is below 1e12, so a 1e12 cutoff would
# let it through — 1e7 is safely above all physical WSE values but far below
# any fill value.)
_SWOT_FILL_THRESHOLD = 1e7

# First SWOT science observations. Used as the default start of the "full"
# record when no explicit start date is supplied.
SWOT_RECORD_START = "2023-01-01"


###################


def fetch_swot_reaches(bbox, max_records=1000):
    """
    Query the Earthdata GIS feature service for SWOT river reaches whose
    geometry intersects a bounding box, returning one entry per unique reach.

    Parameters
    ----------
    bbox : tuple of float
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    max_records : int, optional
        Maximum number of features to request. The service also enforces its
        own transfer limit. Defaults to 1000.

    Returns
    -------
    dict
        GeoJSON FeatureCollection (EPSG:4326) of river reaches, de-duplicated
        by ``reach_id``. Empty FeatureCollection on error or no results.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "where": "1=1",
        "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "reach_id",
        "returnGeometry": "true",
        "resultRecordCount": str(max_records),
        "f": "geojson",
    }
    verify = os.environ.get("REQUESTS_CA_BUNDLE") or True
    try:
        response = requests.get(
            SWOT_REACH_FEATURE_SERVICE, params=params, timeout=60, verify=verify
        )
    except requests.RequestException as e:
        logger.error(f"SWOT reach feature service request failed: {e}")
        return {"type": "FeatureCollection", "features": []}

    if response.status_code != 200:
        logger.error(
            f"SWOT reach feature service returned {response.status_code}: "
            f"{response.text[:200]}"
        )
        return {"type": "FeatureCollection", "features": []}

    try:
        data = response.json()
    except ValueError:
        logger.error("SWOT reach feature service did not return valid JSON.")
        return {"type": "FeatureCollection", "features": []}

    # De-duplicate by reach_id (the service returns one feature per
    # observation, so the same reach can appear many times).
    seen = set()
    unique_features = []
    for feat in data.get("features", []):
        rid = feat.get("properties", {}).get("reach_id")
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        unique_features.append(feat)

    logger.info(
        f"SWOT reach query returned {len(unique_features)} unique reaches "
        f"within bbox {bbox}."
    )
    return {"type": "FeatureCollection", "features": unique_features}


###################


def pull_swot_timeseries(feature_id, feature_type="Reach", i_date=None, f_date=None):
    """
    Retrieve a SWOT water-surface-elevation time series for a single river
    reach or prior lake from the PO.DAAC Hydrocron API.

    Parameters
    ----------
    feature_id : str or int
        ``reach_id`` (for rivers) or ``lake_id`` (for prior lakes).
    feature_type : str, optional
        ``"Reach"`` or ``"PriorLake"``. Defaults to ``"Reach"``.
    i_date : str, optional
        ISO start date (``YYYY-MM-DD``). Falls back to ``SWOT_RECORD_START``
        so the default is the full record.
    f_date : str, optional
        ISO end date (``YYYY-MM-DD``). Falls back to today.

    Returns
    -------
    pandas.DataFrame
        Columns ``["time", "wse"]`` sorted by time, with fill values removed.
        Empty DataFrame if no valid observations are returned.
    """
    if feature_type not in ("Reach", "PriorLake"):
        raise ValueError("feature_type must be 'Reach' or 'PriorLake'")

    id_field = "reach_id" if feature_type == "Reach" else "lake_id"

    start = (i_date or SWOT_RECORD_START) + "T00:00:00Z"
    end_base = f_date or datetime.utcnow().strftime("%Y-%m-%d")
    end = end_base + "T00:00:00Z"

    params = {
        "feature": feature_type,
        "feature_id": str(feature_id),
        "start_time": start,
        "end_time": end,
        "output": "csv",
        "fields": f"{id_field},time_str,wse",
    }
    verify = os.environ.get("REQUESTS_CA_BUNDLE") or True
    try:
        response = requests.get(
            HYDROCRON_TIMESERIES_URL, params=params, timeout=120, verify=verify
        )
    except requests.RequestException as e:
        logger.error(f"Hydrocron request failed for {feature_type} {feature_id}: {e}")
        return pd.DataFrame(columns=["time", "wse"])

    if response.status_code != 200:
        logger.error(
            f"Hydrocron returned {response.status_code} for {feature_type} "
            f"{feature_id}: {response.text[:200]}"
        )
        return pd.DataFrame(columns=["time", "wse"])

    try:
        payload = response.json()
        csv_text = payload.get("results", {}).get("csv", "")
    except ValueError:
        logger.error("Hydrocron did not return valid JSON.")
        return pd.DataFrame(columns=["time", "wse"])

    if not csv_text:
        logger.warning(
            f"Hydrocron returned no observations for {feature_type} {feature_id}."
        )
        return pd.DataFrame(columns=["time", "wse"])

    df = pd.read_csv(StringIO(csv_text))
    if "time_str" not in df.columns or "wse" not in df.columns:
        logger.error(f"Unexpected Hydrocron columns: {list(df.columns)}")
        return pd.DataFrame(columns=["time", "wse"])

    df["time"] = pd.to_datetime(
        df["time_str"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce", utc=True
    )
    df["wse"] = pd.to_numeric(df["wse"], errors="coerce")

    # Drop rows with unparseable times or fill-valued / missing WSE.
    df = df.dropna(subset=["time", "wse"])
    df = df[df["wse"].abs() < _SWOT_FILL_THRESHOLD]

    out = (
        df[["time", "wse"]]
        .sort_values("time")
        .reset_index(drop=True)
    )
    logger.info(
        f"Hydrocron {feature_type} {feature_id}: {len(out)} valid observations."
    )
    return out


###################


def flag_wse_outliers(df, value_col="wse", threshold=3.5):
    """
    Flag outliers in a water-surface-elevation time series using a robust
    median-absolute-deviation (MAD) modified z-score.

    The modified z-score is ``0.6745 * (x - median) / MAD``. Points whose
    absolute modified z-score exceeds ``threshold`` are flagged. MAD is used
    instead of the standard deviation because it is resistant to the very
    outliers being screened, which makes it well suited to spiky SWOT WSE
    records (bad passes, layover, ice, etc.).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``value_col``. Typically the output of
        ``pull_swot_timeseries`` with columns ``["time", "wse"]``.
    value_col : str, optional
        Column to screen. Defaults to ``"wse"``.
    threshold : float, optional
        Modified z-score cutoff. Iglewicz & Hoaglin recommend 3.5. Larger
        values flag fewer points. Defaults to 3.5.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with an added boolean ``"is_outlier"`` column. When
        screening cannot be applied (fewer than 3 points, or a degenerate
        spread) no points are flagged.
    """
    out = df.copy()
    if value_col not in out.columns or len(out) < 3:
        out["is_outlier"] = False
        return out

    values = pd.to_numeric(out[value_col], errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()

    if mad == 0 or pd.isna(mad):
        # Degenerate spread (e.g. many identical values): fall back to a
        # standard-deviation z-score so a few gross outliers can still be
        # caught, but flag nothing if that is also degenerate.
        std = values.std()
        if std == 0 or pd.isna(std):
            out["is_outlier"] = False
            return out
        modified_z = (values - values.mean()).abs() / std
    else:
        modified_z = 0.6745 * (values - median).abs() / mad

    out["is_outlier"] = modified_z > threshold
    n_flagged = int(out["is_outlier"].sum())
    if n_flagged:
        logger.info(
            f"flag_wse_outliers: flagged {n_flagged} of {len(out)} points "
            f"(threshold={threshold})."
        )
    return out


###################


# Local SWOT Prior Lake Database (PLD) footprint files, keyed by region. NASA
# does not publish a spatially queryable lake feature service, so these small
# GeoJSONs are generated once via prep_data/download_swot_prior_lakes.py and
# read locally to power the clickable lake map. Each file has "lake_id" +
# polygon geometry in EPSG:4326.
SWOT_LAKE_FILES = {
    "lesotho": DATA_DIR / "swot_prior_lakes_lesotho.geojson",
}


def fetch_swot_lakes(bbox=None, region="lesotho"):
    """
    Load SWOT prior-lake footprints from a local regional PLD file, optionally
    filtered to a bounding box.

    Parameters
    ----------
    bbox : tuple of float, optional
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326. If given, only
        lakes intersecting this box are returned. If ``None``, all lakes in
        the regional file are returned.
    region : str, optional
        Key into ``SWOT_LAKE_FILES``. Defaults to ``"lesotho"``.

    Returns
    -------
    dict
        GeoJSON FeatureCollection (EPSG:4326) with a ``lake_id`` property per
        feature. Empty FeatureCollection if the file is missing or no lakes
        match.
    """
    path = SWOT_LAKE_FILES.get(region)
    if path is None or not Path(path).is_file():
        logger.warning(
            f"SWOT prior-lake file for region '{region}' not found at {path}. "
            "Run prep_data/download_swot_prior_lakes.py to generate it."
        )
        return {"type": "FeatureCollection", "features": []}

    try:
        gdf = gpd.read_file(path)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to read SWOT prior-lake file {path}: {e}")
        return {"type": "FeatureCollection", "features": []}

    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]

    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}

    # Ensure lake_id is a plain string for consistent downstream use.
    if "lake_id" in gdf.columns:
        gdf["lake_id"] = gdf["lake_id"].astype(str)

    logger.info(
        f"fetch_swot_lakes[{region}]: {len(gdf)} lakes"
        + (f" within bbox {bbox}." if bbox is not None else ".")
    )
    return json.loads(gdf.to_json())
