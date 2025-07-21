import os
import yaml
import logging

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio as rio
import rasterio.mask
import subprocess

logging.basicConfig(level=logging.INFO)


def _merge_data(
    full_data: gpd.GeoDataFrame, columns: list = [], how: str = "inner"
) -> gpd.GeoDataFrame:
    merged = full_data[0]

    for data in full_data[1:]:
        if not set(data.columns) <= set(merged.columns):
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
    if config_file is None:
        config_file = "configs/config.yaml"

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config
