import os
import yaml
import logging

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio as rio
import rasterio.mask
import subprocess
import humanize

import textwrap
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

logging.basicConfig(level=logging.INFO)


def _get_text_height(fig, s, fontsize):
    """Return text height in figure coords based on fontsize."""
    renderer = fig.canvas.get_renderer()
    t = plt.text(0, 0, s, fontsize=fontsize)
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    
    return bb.height / fig.bbox.height


def _minmax_scale(data):
    """
    Performs Min-Max scaling on a NumPy array or Pandas Series.

    Args:
        data (np.ndarray or pd.Series): The input data to be scaled.

    Returns:
        np.ndarray or pd.Series: The scaled data.
    """
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    if max_val == min_val:  # Handle cases where all values are the same
        return np.zeros_like(data, dtype=float)
    
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data


def _humanize(value, number=None):
    if value < 0:
        return "0"

    # Large numbers: format with K/M using humanize.intword
    if value >= 10:
        formatter = "%.1f"
        if value > 100000:
            formatter = "%.0f"
    
        text = humanize.intword(value, formatter)
        text = text.replace(" thousand", "K").replace(" million", "M")

        # Remove trailing .0 if present (e.g., "1.0K" → "1K")
        if text.endswith(".0K") or text.endswith(".0M"):
            text = text.replace(".0", "")
        return text

    # Smaller numbers: format directly
    if value.is_integer():
        return f"{int(value)}"
    elif value < 1:
        return f"{value:.3f}"
    else:
        return f"{value:.1f}"


def _fill_holes(geometry):
    if isinstance(geometry, Polygon):
        # Create a new Polygon from its exterior ring, effectively removing holes
        return Polygon(geometry.exterior)
    elif isinstance(geometry, MultiPolygon):
        # Apply to each polygon within the MultiPolygon
        return MultiPolygon([Polygon(p.exterior) for p in geometry.geoms])
    return geometry # Return other geometry types as is    


def _merge_data(
    full_data: gpd.GeoDataFrame, columns: list = [], how: str = "inner"
) -> gpd.GeoDataFrame:
    merged = full_data[0]

    for data in full_data[1:]:
        #if not set(data.columns) <= set(merged.columns):
        merged = pd.merge(merged, data, on=columns, how=how)

    if "geometry" in columns:
        merged = gpd.GeoDataFrame(merged, geometry="geometry")

    return merged


def _clip_raster(
    global_tif: str, local_tif: str, admin: gpd.GeoDataFrame, nodata: list = []
) -> rio.io.DatasetReader:

    # Return existing raster if the clipped file already exists
    if not os.path.exists(local_tif):
        with rio.open(global_tif) as src:
            if src.nodata is not None:
                nodata = [src.nodata] + nodata

            # Reproject the admin boundaries if CRS differs
            if src.crs != admin.crs:
                admin = admin.to_crs(src.crs)

            # Extract the country boundary geometry for clipping
            shape = [admin.iloc[0]["geometry"]]

            # Perform raster clipping using rasterio.mask
            out_image, out_transform = rio.mask.mask(
                src, shape, crop=True, all_touched=True
            )
            for val in nodata:
                out_image[out_image == val] = -1

            # Update raster metadata to reflect changes
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": -1,
                }
            )

        # Save the clipped raster to the specified output path
        with rasterio.open(local_tif, "w", **out_meta) as dest:
            dest.write(out_image)

    # Return the clipped raster
    return rio.open(local_tif)


def read_config(config_file: str):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config
